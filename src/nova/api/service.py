"""
Incident Management Service for the NOVA FastAPI Gateway.

Coordinates:
- Non-blocking asynchronous incident execution via a background worker pool.
- Durable checkpoint persistence via SqliteSaver (with fallback/test injection).
- Persistent incident metadata, approval audit trails, and telemetry in SQLite.
- Safe Human-in-the-Loop approval resumption using LangGraph Command(resume=...).
- Duplicate approval & replay prevention under thread synchronization.
- State translation between raw graph state and typed API schemas.
"""

from __future__ import annotations

import concurrent.futures
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import Command

from nova.api.schemas import (
    ApprovalSubmitRequest,
    ApprovalSubmitResponse,
    IncidentListItem,
    IncidentStateResponse,
    IncidentTriggerRequest,
    IncidentTriggerResponse,
    PendingApprovalDTO,
)
from nova.config import settings
from nova.environments.kestra_runner import KestraEnvironment
from nova.environments.mock_workflow_runner import MockWorkflowRunner, ScenarioType, WorkflowState
from nova.graph import build_nova_graph
from nova.llm.client import BaseLLMClient, GeminiLLMClient, MockLLMClient
from nova.observability import (
    configure_logging,
    incident_context,
    log_event,
    redact_secrets,
    telemetry,
)
from nova.state import (
    ALLOWED_APPROVER_ROLES,
    ApprovalDecision,
    IncidentStatus,
    NovaGraphState,
)
from nova.storage.incident_store import IncidentMetadataStore
from nova.storage.sqlite_saver import SqliteSaver

logger = logging.getLogger("nova.api.service")

TERMINAL_STATUSES = {
    IncidentStatus.RESOLVED,
    IncidentStatus.ESCALATED,
    IncidentStatus.FAILED,
}


class ConflictError(Exception):
    """Raised when an operation conflicts with in-flight or already-consumed state."""


class IncidentService:
    """
    Core backend orchestrator managing NOVA incident lifecycles, background execution,
    durable metadata/audit storage, and checkpoint queries.
    """

    def __init__(
        self,
        env: Optional[MockWorkflowRunner] = None,
        checkpointer: Optional[BaseCheckpointSaver] = None,
        llm: Optional[BaseLLMClient] = None,
        max_workers: int = 4,
        kestra_env: Optional[KestraEnvironment] = None,
    ):
        configure_logging(settings.log_level)
        self.env = env or MockWorkflowRunner(base_dir=settings.mock_env_dir)
        self._injected_kestra_env = kestra_env

        # Durable SQLite checkpointer by default
        if checkpointer is not None:
            self.checkpointer = checkpointer
        else:
            db_path = settings.sqlite_db_path
            self.checkpointer = SqliteSaver(db_path=db_path)

        # Persistent metadata & approval audit store when backed by SqliteSaver
        self._store: Optional[IncidentMetadataStore] = (
            IncidentMetadataStore(db_path=self.checkpointer.db_path)
            if isinstance(self.checkpointer, SqliteSaver)
            else None
        )

        # LLM Client selection
        if llm is not None:
            self.llm = llm
        elif settings.gemini_api_key:
            self.llm = GeminiLLMClient(
                api_key=settings.gemini_api_key,
                model_name=settings.default_model,
            )
        else:
            self.llm = MockLLMClient()

        # Build compiled LangGraph state machine with persistence and HITL
        self.graph = build_nova_graph(
            env=self.env,
            checkpointer=self.checkpointer,
            require_approval=True,
            llm=self.llm,
        )

        # Background task execution pool
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="nova-incident-worker",
        )
        self._running_tasks: Dict[str, concurrent.futures.Future[Any]] = {}
        self._incident_metadata: Dict[str, Dict[str, Any]] = {}
        self._incident_envs: Dict[str, Any] = {}
        self._incident_graphs: Dict[str, Any] = {}
        self._in_memory_audits: Dict[str, List[Dict[str, Any]]] = {}
        self._consumed_approvals: Set[Tuple[str, str]] = set()
        self._lock = threading.RLock()
        # M11: Track the single active (non-terminal) incident ID.
        # NOVA is a prototype and processes one incident at a time.
        self._active_incident_id: Optional[str] = None

        # Hydrate metadata and persisted telemetry from SQLite on startup if available
        if self._store is not None:
            for item in self._store.list_incidents():
                inc_id = item["incident_id"]
                self._incident_metadata[inc_id] = item
                saved_telem = self._store.get_telemetry(inc_id)
                if saved_telem:
                    telemetry.hydrate_incident(inc_id, saved_telem)

    def _has_pending_interrupt(self, incident_id: str) -> bool:
        """Returns True if the checkpointed graph state has an unresolved HITL interrupt."""
        try:
            inc_graph = self._incident_graphs.get(incident_id, self.graph)
            st = inc_graph.get_state({"configurable": {"thread_id": incident_id}})
            return bool(st and st.tasks and st.tasks[0].interrupts)
        except Exception:
            return False

    def _find_active_incident_id(self) -> Optional[str]:
        """
        Returns the incident_id of any incident with an active background worker
        or a pending HITL approval interrupt (M11 single active incident guard).
        Must be called while holding self._lock.
        """
        for inc_id, future in self._running_tasks.items():
            if future is not None and not future.done():
                return inc_id
        if self._active_incident_id and self._has_pending_interrupt(self._active_incident_id):
            return self._active_incident_id
        for inc_id in self._incident_metadata.keys():
            if self._has_pending_interrupt(inc_id):
                self._active_incident_id = inc_id
                return inc_id
        return None

    def _refresh_active_incident_after_cycle(self, incident_id: str) -> None:
        """Clears _active_incident_id once an incident leaves active remediation / pending approval."""
        still_paused = self._has_pending_interrupt(incident_id)
        with self._lock:
            if self._active_incident_id == incident_id and not still_paused:
                self._active_incident_id = None

    def inject_scenario(self, scenario_type: ScenarioType) -> WorkflowState:
        """
        Injects a synthetic scenario into the shared MockWorkflowRunner and runs
        the initial batch execution.

        M11: Rejected with ConflictError if any incident is currently running
        or awaiting human approval, preventing scenario resets from invalidating
        in-flight operations.
        """
        with self._lock:
            active_id = self._find_active_incident_id()
            if active_id is not None:
                raise ConflictError(
                    f"Cannot reset or inject scenario while incident '{active_id}' is active "
                    "or awaiting human approval."
                )
            self.env.inject_scenario(scenario_type)
            return self.env.run_workflow()

    def _ensure_telemetry_loaded(self, incident_id: str) -> None:
        """Restores prior persisted telemetry before recording new resume/approval events."""
        if self._store is not None and telemetry.get_incident_metrics(incident_id) is None:
            saved_telem = self._store.get_telemetry(incident_id)
            if saved_telem:
                telemetry.hydrate_incident(incident_id, saved_telem)

    def _record_audit(self, record: Dict[str, Any]) -> None:
        incident_id = record["incident_id"]
        if self._store is not None:
            self._store.record_approval_audit(record)
        with self._lock:
            self._in_memory_audits.setdefault(incident_id, []).append(record)
            if record.get("outcome") in ("APPROVED_RESUMED", "REJECTED_ESCALATED"):
                self._consumed_approvals.add((incident_id, record["action_id"]))
                if record.get("approval_request_id"):
                    self._consumed_approvals.add(
                        (incident_id, record["approval_request_id"])
                    )
        telemetry.record_approval_event(incident_id, record["outcome"])

    def _is_approval_consumed(
        self,
        incident_id: str,
        action_id: str,
        approval_request_id: Optional[str] = None,
    ) -> bool:
        with self._lock:
            if (incident_id, action_id) in self._consumed_approvals:
                return True
            if (
                approval_request_id
                and (incident_id, approval_request_id) in self._consumed_approvals
            ):
                return True
        if self._store is not None:
            return self._store.has_consumed_approval(
                incident_id, action_id, approval_request_id
            )
        return False

    def _persist_telemetry(self, incident_id: str) -> None:
        if self._store is None:
            return
        metrics = telemetry.get_incident_metrics(incident_id)
        if metrics is not None:
            now_iso = datetime.now(timezone.utc).isoformat()
            self._store.save_telemetry(incident_id, metrics, now_iso)

    def _mark_failed_safely(
        self, thread_config: Dict[str, Any], incident_id: str, reason: str
    ) -> None:
        clean_reason = redact_secrets(reason)
        telemetry.record_terminal_reason(incident_id, "FAILED", clean_reason)
        try:
            inc_graph = self._incident_graphs.get(incident_id, self.graph)
            inc_graph.update_state(
                thread_config,
                {
                    "status": IncidentStatus.FAILED,
                    "final_summary": clean_reason,
                },
            )
        except Exception as update_exc:
            log_event(
                logger,
                logging.ERROR,
                "state_failure_update_error",
                f"Failed to persist FAILED status after worker error: {update_exc}",
                incident_id=incident_id,
            )

    def trigger_incident(
        self, request: IncidentTriggerRequest
    ) -> IncidentTriggerResponse:
        """
        Submits a new operational incident to NOVA for non-blocking background remediation.
        """
        incident_id = request.incident_id or f"INC_{uuid.uuid4().hex[:8].upper()}"
        now_iso = datetime.now(timezone.utc).isoformat()

        with self._lock:
            existing_future = self._running_tasks.get(incident_id)
            if existing_future is not None and not existing_future.done():
                raise ConflictError(
                    f"Incident '{incident_id}' is already running."
                )
            if incident_id in self._incident_metadata and request.incident_id is not None:
                raise ConflictError(
                    f"Incident '{incident_id}' already exists."
                )
            # M11: Enforce single active incident against the shared workflow environment.
            active_id = self._find_active_incident_id()
            if active_id is not None and active_id != incident_id:
                raise ConflictError(
                    f"Cannot trigger incident '{incident_id}': incident '{active_id}' is already "
                    "active or awaiting approval. Resolve or escalate it before starting another incident."
                )

            use_kestra = request.environment == "kestra"
            if use_kestra:
                if not request.execution_id or not request.execution_id.strip():
                    raise ValueError("execution_id is required when environment='kestra'.")
                eid = request.execution_id.strip()
                if self._injected_kestra_env is not None:
                    inc_env = self._injected_kestra_env
                    inc_env.bind_execution(eid)
                else:
                    inc_env = KestraEnvironment(
                        kestra_url=request.kestra_url or "http://127.0.0.1:8080",
                        default_execution_id=eid,
                    )
                try:
                    wf_state = inc_env.get_state(execution_id=eid)
                except Exception as exc:
                    raise ValueError(f"Failed to query Kestra execution '{eid}': {exc}") from exc
                if wf_state is None:
                    raise ValueError(f"Kestra execution '{eid}' not found.")
                inc_graph = build_nova_graph(
                    env=inc_env,
                    checkpointer=self.checkpointer,
                    require_approval=True,
                    llm=self.llm,
                )
            else:
                inc_env = self.env
                inc_graph = self.graph
                wf_state = inc_env.get_state()

            bound_wf_id = wf_state.workflow_id if wf_state else None
            bound_run_id = wf_state.run_id if wf_state else None
            baseline_attempts = getattr(wf_state, "total_attempts", None) if wf_state else None
            baseline_task_attempts = None
            baseline_task_states = None
            if wf_state and getattr(wf_state, "jobs", None):
                t_atts = {
                    tid: getattr(t, "attempt_count", 0)
                    for tid, t in wf_state.jobs.items()
                    if hasattr(t, "attempt_count")
                }
                if t_atts:
                    baseline_task_attempts = t_atts
                baseline_task_states = {
                    tid: (t.status.value if hasattr(t.status, "value") else str(t.status))
                    for tid, t in wf_state.jobs.items()
                }
            self._active_incident_id = incident_id
            self._incident_envs[incident_id] = inc_env
            self._incident_graphs[incident_id] = inc_graph

            self._incident_metadata[incident_id] = {
                "incident_id": incident_id,
                "created_at": now_iso,
                "updated_at": now_iso,
                "target_service": request.target_service,
                "raw_symptoms": request.raw_symptoms,
                "environment": request.environment,
                "execution_id": bound_run_id,
            }

        if self._store is not None:
            self._store.upsert_incident(
                incident_id=incident_id,
                target_service=request.target_service,
                raw_symptoms=request.raw_symptoms,
                created_at=now_iso,
                updated_at=now_iso,
            )

        telemetry.reset_incident(incident_id)

        initial_state: NovaGraphState = {
            "incident_id": incident_id,
            "target_service": request.target_service or "UNKNOWN",
            "raw_symptoms": request.raw_symptoms,
            "status": IncidentStatus.TRIGGERED,
            "workflow_id": bound_wf_id,
            "workflow_run_id": bound_run_id,
            "baseline_attempts": baseline_attempts,
            "baseline_task_attempts": baseline_task_attempts,
            "baseline_task_states": baseline_task_states,
            "triage_summary": None,
            "root_cause_hypothesis": None,
            "plan": [],
            "current_action_index": 0,
            "execution_history": [],
            "verification_history": [],
            "pending_approval": None,
            "retry_count": 0,
            "max_retries": request.max_retries,
            "final_summary": None,
            "diagnostic_history": [],
            "investigation_evidence": [],
            "investigation_steps": 0,
            "max_investigation_steps": request.max_investigation_steps,
            "reflection_history": [],
            "hypothesis_revisions": [],
            "plan_revisions": [],
        }

        thread_config = {"configurable": {"thread_id": incident_id}}

        def _execute() -> None:
            t0 = time.perf_counter()
            with incident_context(incident_id):
                try:
                    log_event(
                        logger,
                        logging.INFO,
                        "incident_execution_started",
                        f"Starting graph execution for incident {incident_id}",
                        target_service=request.target_service or "UNKNOWN",
                        environment=request.environment,
                        execution_id=bound_run_id,
                    )
                    inc_graph.invoke(initial_state, thread_config)
                except (RuntimeError, concurrent.futures.CancelledError) as re_err:
                    if "shutdown" in str(re_err).lower():
                        log_event(
                            logger,
                            logging.INFO,
                            "incident_worker_shutdown",
                            f"Incident {incident_id} worker halted due to service shutdown.",
                        )
                    else:
                        err_msg = redact_secrets(str(re_err))
                        log_event(
                            logger,
                            logging.ERROR,
                            "incident_worker_runtime_error",
                            f"Runtime error in incident {incident_id}: {err_msg}",
                        )
                        self._mark_failed_safely(
                            thread_config,
                            incident_id,
                            f"Incident execution failed: {err_msg}",
                        )
                except Exception as exc:
                    err_msg = redact_secrets(str(exc))
                    log_event(
                        logger,
                        logging.ERROR,
                        "incident_worker_unhandled_error",
                        f"Unhandled error in incident {incident_id}: {err_msg}",
                    )
                    self._mark_failed_safely(
                        thread_config,
                        incident_id,
                        f"Incident execution failed: {err_msg}",
                    )
                finally:
                    dt = time.perf_counter() - t0
                    telemetry.record_execution_duration(incident_id, dt)
                    updated_iso = datetime.now(timezone.utc).isoformat()
                    with self._lock:
                        if incident_id in self._incident_metadata:
                            self._incident_metadata[incident_id]["updated_at"] = updated_iso
                    if self._store is not None:
                        self._store.touch_incident(incident_id, updated_iso)
                    self._persist_telemetry(incident_id)
                    self._refresh_active_incident_after_cycle(incident_id)
                    log_event(
                        logger,
                        logging.INFO,
                        "incident_execution_cycle_finished",
                        f"Background execution cycle finished in {dt:.3f}s",
                        duration_seconds=round(dt, 4),
                    )

        future = self._executor.submit(_execute)
        with self._lock:
            self._running_tasks[incident_id] = future

        return IncidentTriggerResponse(
            incident_id=incident_id,
            status=IncidentStatus.TRIGGERED,
            target_service=request.target_service,
            raw_symptoms=request.raw_symptoms,
            created_at=now_iso,
        )

    def get_incident_state(self, incident_id: str) -> Optional[IncidentStateResponse]:
        """
        Polls the full, persistent state of an incident from checkpointer storage.
        """
        thread_config = {"configurable": {"thread_id": incident_id}}
        try:
            checkpoint_state = self.graph.get_state(thread_config)
        except Exception as e:
            log_event(
                logger,
                logging.WARNING,
                "checkpoint_read_error",
                f"Error reading checkpoint for {incident_id}: {e}",
                incident_id=incident_id,
            )
            return None

        with self._lock:
            future = self._running_tasks.get(incident_id)
            worker_active = bool(future is not None and not future.done())
            meta = self._incident_metadata.get(incident_id)
            in_mem_audits = list(self._in_memory_audits.get(incident_id, []))

        if meta is None and self._store is not None:
            meta = self._store.get_incident(incident_id)

        if not checkpoint_state or not checkpoint_state.values:
            if meta is None:
                return None
            return IncidentStateResponse(
                incident_id=incident_id,
                status=IncidentStatus.TRIGGERED,
                target_service=meta.get("target_service"),
                raw_symptoms=meta.get("raw_symptoms", ""),
                worker_active=worker_active,
                interrupted_by_restart=not worker_active,
                telemetry=telemetry.get_incident_metrics(incident_id)
                or (self._store.get_telemetry(incident_id) if self._store else None),
                approval_audit_history=(
                    self._store.get_approval_audits(incident_id)
                    if self._store is not None
                    else in_mem_audits
                ),
            )

        vals = checkpoint_state.values

        # Check for pending Human-in-the-Loop interrupts
        pending_dto: Optional[PendingApprovalDTO] = None
        current_status = vals.get("status", IncidentStatus.TRIGGERED)
        if isinstance(current_status, str):
            try:
                current_status = IncidentStatus(current_status)
            except ValueError:
                pass

        if checkpoint_state.tasks and checkpoint_state.tasks[0].interrupts:
            interrupt_val = checkpoint_state.tasks[0].interrupts[0].value
            pending_dto = PendingApprovalDTO(
                action_id=interrupt_val.get("action_id", "UNKNOWN"),
                tool_name=interrupt_val.get("tool_name", "UNKNOWN"),
                risk_level=interrupt_val.get("risk_level", "DESTRUCTIVE"),
                parameters=interrupt_val.get("parameters", {}),
                rationale=interrupt_val.get("rationale", ""),
                expected_outcome=interrupt_val.get("expected_outcome", ""),
                requested_at=interrupt_val.get("requested_at"),
                approval_request_id=interrupt_val.get("approval_request_id"),
                action_hash=interrupt_val.get("action_hash"),
            )
            current_status = IncidentStatus.AWAITING_APPROVAL

        interrupted_by_restart = (
            current_status not in TERMINAL_STATUSES
            and current_status != IncidentStatus.AWAITING_APPROVAL
            and not worker_active
        )

        def _to_dicts(items: List[Any]) -> List[Dict[str, Any]]:
            result = []
            for item in items:
                if hasattr(item, "model_dump"):
                    result.append(item.model_dump())
                elif isinstance(item, dict):
                    result.append(item)
                else:
                    result.append({"data": str(item)})
            return result

        incident_telemetry = telemetry.get_incident_metrics(incident_id)
        if incident_telemetry is None and self._store is not None:
            incident_telemetry = self._store.get_telemetry(incident_id)

        audits = (
            self._store.get_approval_audits(incident_id)
            if self._store is not None
            else in_mem_audits
        )

        return IncidentStateResponse(
            incident_id=incident_id,
            status=current_status,
            target_service=vals.get("target_service"),
            raw_symptoms=vals.get("raw_symptoms", ""),
            workflow_id=vals.get("workflow_id"),
            workflow_run_id=vals.get("workflow_run_id"),
            investigation_steps=vals.get("investigation_steps", 0),
            max_investigation_steps=vals.get("max_investigation_steps", 5),
            investigation_evidence=list(vals.get("investigation_evidence") or []),
            diagnostic_history=_to_dicts(list(vals.get("diagnostic_history") or [])),
            reflection_history=list(vals.get("reflection_history") or []),
            hypothesis_revisions=list(vals.get("hypothesis_revisions") or []),
            plan_revisions=list(vals.get("plan_revisions") or []),
            failure_type=vals.get("failure_type"),
            implicated_resource=vals.get("implicated_resource"),
            is_recoverable=vals.get("is_recoverable"),
            triage_summary=vals.get("triage_summary"),
            root_cause_hypothesis=vals.get("root_cause_hypothesis"),
            plan=_to_dicts(list(vals.get("plan") or [])),
            current_action_index=vals.get("current_action_index", 0),
            execution_history=_to_dicts(list(vals.get("execution_history") or [])),
            verification_history=_to_dicts(list(vals.get("verification_history") or [])),
            pending_approval=pending_dto,
            retry_count=vals.get("retry_count", 0),
            max_retries=vals.get("max_retries", 3),
            final_summary=vals.get("final_summary"),
            telemetry=incident_telemetry,
            approval_audit_history=audits,
            worker_active=worker_active,
            interrupted_by_restart=interrupted_by_restart,
        )

    def submit_approval(
        self,
        incident_id: str,
        request: ApprovalSubmitRequest,
    ) -> ApprovalSubmitResponse:
        """
        Processes human authorization decision, validating RBAC, action binding,
        duplicate submission / replay guards, and resumes LangGraph execution asynchronously.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        self._ensure_telemetry_loaded(incident_id)

        # 1. RBAC Check: Ensure approver role is authorized
        if request.approver_role not in ALLOWED_APPROVER_ROLES:
            self._record_audit(
                {
                    "audit_id": f"AUD_{uuid.uuid4().hex[:10].upper()}",
                    "incident_id": incident_id,
                    "action_id": request.action_id,
                    "approval_request_id": None,
                    "action_hash": request.action_hash,
                    "tool_name": None,
                    "approved": request.approved,
                    "approver_id": request.approver_id,
                    "approver_role": request.approver_role,
                    "decision_timestamp": now_iso,
                    "notes": request.notes,
                    "outcome": "REJECTED_RBAC",
                }
            )
            log_event(
                logger,
                logging.WARNING,
                "approval_rbac_denied",
                f"Unauthorized approver role '{request.approver_role}' for incident {incident_id}",
                incident_id=incident_id,
                approver_id=request.approver_id,
                approver_role=request.approver_role,
            )
            raise PermissionError(
                f"Unauthorized approver role '{request.approver_role}'. "
                f"Must be one of: {sorted(ALLOWED_APPROVER_ROLES)}"
            )

        thread_config = {"configurable": {"thread_id": incident_id}}

        with self._lock:
            existing_future = self._running_tasks.get(incident_id)
            if existing_future is not None and not existing_future.done():
                self._record_audit(
                    {
                        "audit_id": f"AUD_{uuid.uuid4().hex[:10].upper()}",
                        "incident_id": incident_id,
                        "action_id": request.action_id,
                        "approval_request_id": None,
                        "action_hash": request.action_hash,
                        "tool_name": None,
                        "approved": request.approved,
                        "approver_id": request.approver_id,
                        "approver_role": request.approver_role,
                        "decision_timestamp": now_iso,
                        "notes": request.notes,
                        "outcome": "REJECTED_IN_FLIGHT",
                    }
                )
                raise ConflictError(
                    f"Approval or execution is already in progress for incident '{incident_id}'."
                )

            checkpoint_state = self.graph.get_state(thread_config)

            if (
                not checkpoint_state
                or not checkpoint_state.tasks
                or not checkpoint_state.tasks[0].interrupts
            ):
                if self._store is not None and self._store.has_consumed_approval(
                    incident_id, request.action_id
                ):
                    raise ConflictError(
                        f"Approval for action '{request.action_id}' on incident '{incident_id}' was already processed."
                    )
                if (incident_id, request.action_id) in self._consumed_approvals:
                    raise ConflictError(
                        f"Approval for action '{request.action_id}' on incident '{incident_id}' was already processed."
                    )
                raise ValueError(
                    f"Incident '{incident_id}' is not currently awaiting human approval."
                )

            interrupt_val = checkpoint_state.tasks[0].interrupts[0].value
            expected_action_id = interrupt_val.get("action_id")
            expected_request_id = interrupt_val.get("approval_request_id")
            expected_action_hash = interrupt_val.get("action_hash")
            tool_name = interrupt_val.get("tool_name")

            # M11/M13: Verify workflow run binding before resuming a destructive operation
            bound_run_id = interrupt_val.get("workflow_run_id") or checkpoint_state.values.get("workflow_run_id")
            bound_wf_id = interrupt_val.get("workflow_id") or checkpoint_state.values.get("workflow_id")
            inc_env = self._incident_envs.get(incident_id)
            inc_graph = self._incident_graphs.get(incident_id)
            if inc_env is None and bound_wf_id and "/" in str(bound_wf_id) and bound_run_id:
                inc_env = (
                    self._injected_kestra_env
                    if self._injected_kestra_env is not None
                    else KestraEnvironment(default_execution_id=bound_run_id)
                )
                inc_env.bind_execution(bound_run_id)
                inc_graph = build_nova_graph(
                    env=inc_env,
                    checkpointer=self.checkpointer,
                    require_approval=True,
                    llm=self.llm,
                )
                self._incident_envs[incident_id] = inc_env
                self._incident_graphs[incident_id] = inc_graph
            inc_env = inc_env or self.env
            inc_graph = inc_graph or self.graph

            if bound_run_id:
                current_wf = (
                    inc_env.get_state(execution_id=bound_run_id)
                    if hasattr(inc_env, "bind_execution")
                    else inc_env.get_state()
                )
                if current_wf is None or current_wf.run_id != bound_run_id:
                    raise ConflictError(
                        f"Workflow run binding mismatch: approval was requested for run '{bound_run_id}', "
                        f"but active run is '{current_wf.run_id if current_wf else 'NONE'}'."
                    )

            if request.action_id != expected_action_id:
                raise ValueError(
                    f"Action ID mismatch: pending action is '{expected_action_id}', "
                    f"but received approval for '{request.action_id}'."
                )

            if (
                request.action_hash is not None
                and expected_action_hash is not None
                and request.action_hash != expected_action_hash
            ):
                raise ValueError(
                    f"Action hash mismatch: pending action hash is '{expected_action_hash}', "
                    f"but received '{request.action_hash}'."
                )

            if (
                (incident_id, request.action_id) in self._consumed_approvals
                or (
                    expected_request_id
                    and (incident_id, expected_request_id) in self._consumed_approvals
                )
                or (
                    self._store is not None
                    and self._store.has_consumed_approval(
                        incident_id, request.action_id, expected_request_id
                    )
                )
            ):
                raise ConflictError(
                    f"Duplicate/replayed approval rejected for action '{request.action_id}' on incident '{incident_id}'."
                )

            outcome_label = "APPROVED_RESUMED" if request.approved else "REJECTED_ESCALATED"
            audit_record = {
                "audit_id": f"AUD_{uuid.uuid4().hex[:10].upper()}",
                "incident_id": incident_id,
                "action_id": request.action_id,
                "approval_request_id": expected_request_id,
                "action_hash": expected_action_hash,
                "tool_name": tool_name,
                "approved": request.approved,
                "approver_id": request.approver_id,
                "approver_role": request.approver_role,
                "decision_timestamp": now_iso,
                "notes": request.notes
                or ("Authorized by operator" if request.approved else "Rejected by operator"),
                "outcome": outcome_label,
            }

            # Mark consumed under lock BEFORE spawning worker to close duplicate-submit race window
            self._consumed_approvals.add((incident_id, request.action_id))
            if expected_request_id:
                self._consumed_approvals.add((incident_id, expected_request_id))
            self._in_memory_audits.setdefault(incident_id, []).append(audit_record)
            if self._store is not None:
                self._store.record_approval_audit(audit_record)
            telemetry.record_approval_event(incident_id, outcome_label)

            decision = ApprovalDecision(
                approved=request.approved,
                approver_id=request.approver_id,
                approver_role=request.approver_role,
                decision_timestamp=now_iso,
                notes=audit_record["notes"],
                action_id=request.action_id,
                incident_id=incident_id,
                approval_request_id=expected_request_id,
                action_hash=expected_action_hash,
            )

            log_event(
                logger,
                logging.INFO,
                "approval_recorded",
                f"Recorded {outcome_label} for action '{request.action_id}' by '{request.approver_id}' ({request.approver_role})",
                incident_id=incident_id,
                action_id=request.action_id,
                approval_request_id=expected_request_id,
                approver_id=request.approver_id,
                approver_role=request.approver_role,
                approved=request.approved,
            )

            def _resume() -> None:
                t0 = time.perf_counter()
                with incident_context(incident_id):
                    try:
                        log_event(
                            logger,
                            logging.INFO,
                            "incident_resume_started",
                            f"Resuming incident {incident_id} with approved={request.approved}",
                            action_id=request.action_id,
                        )
                        inc_graph.invoke(Command(resume=decision), thread_config)
                    except (RuntimeError, concurrent.futures.CancelledError) as re_err:
                        if "shutdown" in str(re_err).lower():
                            log_event(
                                logger,
                                logging.INFO,
                                "incident_resume_shutdown",
                                f"Incident {incident_id} resume worker halted due to service shutdown.",
                            )
                        else:
                            err_msg = redact_secrets(str(re_err))
                            log_event(
                                logger,
                                logging.ERROR,
                                "incident_resume_runtime_error",
                                f"Runtime error resuming incident {incident_id}: {err_msg}",
                            )
                            self._mark_failed_safely(
                                thread_config,
                                incident_id,
                                f"Incident resumption failed: {err_msg}",
                            )
                    except Exception as exc:
                        err_msg = redact_secrets(str(exc))
                        log_event(
                            logger,
                            logging.ERROR,
                            "incident_resume_unhandled_error",
                            f"Error resuming incident {incident_id}: {err_msg}",
                        )
                        self._mark_failed_safely(
                            thread_config,
                            incident_id,
                            f"Incident resumption failed: {err_msg}",
                        )
                    finally:
                        dt = time.perf_counter() - t0
                        telemetry.record_execution_duration(incident_id, dt)
                        updated_iso = datetime.now(timezone.utc).isoformat()
                        with self._lock:
                            if incident_id in self._incident_metadata:
                                self._incident_metadata[incident_id]["updated_at"] = updated_iso
                        if self._store is not None:
                            self._store.touch_incident(incident_id, updated_iso)
                        self._persist_telemetry(incident_id)
                        self._refresh_active_incident_after_cycle(incident_id)

            future = self._executor.submit(_resume)
            self._running_tasks[incident_id] = future

        action_msg = "approved" if request.approved else "rejected"
        return ApprovalSubmitResponse(
            incident_id=incident_id,
            action_id=request.action_id,
            approved=request.approved,
            resumed=True,
            status=IncidentStatus.EXECUTING if request.approved else IncidentStatus.ESCALATED,
            message=f"Action '{request.action_id}' {action_msg}. Graph execution resumed.",
        )

    def list_incidents(self) -> List[IncidentListItem]:
        """
        Lists all tracked and persisted incidents and their current status summaries.
        Supports full incident history retrieval across backend restarts.
        """
        ordered_ids: List[str] = []
        seen: Set[str] = set()

        with self._lock:
            for inc_id in self._incident_metadata.keys():
                if inc_id not in seen:
                    ordered_ids.append(inc_id)
                    seen.add(inc_id)

        if self._store is not None:
            for item in self._store.list_incidents():
                inc_id = item["incident_id"]
                if inc_id not in seen:
                    ordered_ids.append(inc_id)
                    seen.add(inc_id)

        if isinstance(self.checkpointer, SqliteSaver):
            for tid in self.checkpointer.list_thread_ids():
                if tid not in seen:
                    ordered_ids.append(tid)
                    seen.add(tid)

        items: List[IncidentListItem] = []
        for inc_id in ordered_ids:
            state = self.get_incident_state(inc_id)
            if state:
                with self._lock:
                    meta = self._incident_metadata.get(inc_id, {})
                if not meta and self._store is not None:
                    meta = self._store.get_incident(inc_id) or {}
                items.append(
                    IncidentListItem(
                        incident_id=inc_id,
                        status=state.status,
                        target_service=state.target_service,
                        failure_type=state.failure_type,
                        implicated_resource=state.implicated_resource,
                        created_at=meta.get("created_at", ""),
                        updated_at=meta.get("updated_at", ""),
                    )
                )
        return items

    def wait_for_incident(
        self, incident_id: str, timeout: float = 15.0
    ) -> Optional[IncidentStateResponse]:
        """
        Helper method allowing tests or synchronous callers to await execution completion
        or human approval checkpoint.
        """
        with self._lock:
            future = self._running_tasks.get(incident_id)
        if future:
            try:
                future.result(timeout=timeout)
            except Exception as e:
                log_event(
                    logger,
                    logging.ERROR,
                    "wait_for_incident_error",
                    f"Error waiting for incident {incident_id}: {e}",
                    incident_id=incident_id,
                )
        return self.get_incident_state(incident_id)

    def shutdown(self) -> None:
        """Cleanly shuts down the background thread pool."""
        self._executor.shutdown(wait=False, cancel_futures=True)
