import hashlib
import json
import logging
from typing import Any, Dict, List, Optional
from nova.llm.client import BaseLLMClient
from nova.llm.schemas import PlannedActionSchema
from nova.observability import incident_context, log_event, redact_secrets, telemetry
from nova.state import IncidentStatus, NovaGraphState, PlannedAction, RiskLevel
from nova.tools.runner import ToolRunner

logger = logging.getLogger("nova.nodes.planner")


# The tool catalogue exposed to the LLM planner.
# This is what a real tool registry lookup would return.
# The LLM must only use tools from this list.
AVAILABLE_TOOLS_FOR_LLM: List[Dict[str, Any]] = [
    {
        "name": "clear_lockfile",
        "description": "Removes an orphaned lockfile from the locks directory.",
        "parameters": {"lock_name": "str — exact filename of the lockfile to remove"},
        "risk_level": "DESTRUCTIVE",
    },
    {
        "name": "restart_service",
        "description": "Brings an offline or crashed service back online.",
        "parameters": {"service_name": "str — exact name of the service to restart"},
        "risk_level": "DESTRUCTIVE",
    },
    {
        "name": "check_service_status",
        "description": "Returns current health status of a service (READ-ONLY, safe).",
        "parameters": {"service_name": "str — name of the service to check"},
        "risk_level": "READ_ONLY",
    },
    {
        "name": "read_job_logs",
        "description": "Returns the last N lines from a job's log file (READ-ONLY, safe).",
        "parameters": {
            "job_id": "str — the job identifier",
            "tail_lines": "int — number of log lines to return (default 50)",
        },
        "risk_level": "READ_ONLY",
    },
    {
        "name": "resume_workflow",
        "description": "Signals the batch orchestrator to retry the pipeline from last checkpoint.",
        "parameters": {"workflow_id": "str — exact workflow_id of the active workflow run"},
        "risk_level": "LOW_RISK",
    },
    {
        "name": "get_workflow_status",
        "description": "Returns status of all jobs in the workflow (READ-ONLY, safe).",
        "parameters": {"workflow_id": "str (optional) — workflow identifier"},
        "risk_level": "READ_ONLY",
    },
    {
        "name": "inspect_job_dependencies",
        "description": "Inspects batch job DAG dependencies (READ-ONLY, safe).",
        "parameters": {"job_id": "str (optional) — specific job identifier"},
        "risk_level": "READ_ONLY",
    },
    {
        "name": "inspect_lockfiles",
        "description": "Discovers active lockfiles in the locks directory (READ-ONLY, safe).",
        "parameters": {"lock_name": "str (optional) — specific lock filename"},
        "risk_level": "READ_ONLY",
    },
]


def _schema_to_domain(
    schema: PlannedActionSchema, action_id: str, incident_id: str = ""
) -> PlannedAction:
    """
    Converts a PlannedActionSchema (LLM output) into a PlannedAction (domain model).
    
    Why separate schemas?
    PlannedActionSchema is what we tell the LLM to produce — loose enough for the model.
    PlannedAction is the validated domain object used throughout the graph — strict.
    This conversion is the validation boundary.
    Also computes a deterministic SHA256 action_hash for cross-process idempotency.
    """
    canonical_params = json.dumps(schema.parameters, sort_keys=True)
    raw_sig = f"{incident_id}:{action_id}:{schema.tool_name}:{canonical_params}"
    action_hash = hashlib.sha256(raw_sig.encode("utf-8")).hexdigest()[:16]

    return PlannedAction(
        action_id=action_id,
        tool_name=schema.tool_name,
        parameters=schema.parameters,
        risk_level=schema.risk_level,
        rationale=schema.rationale,
        expected_outcome=schema.expected_outcome,
        action_hash=action_hash,
    )


def create_planner_node(
    llm: BaseLLMClient,
    env: Optional[Any] = None,
    runner: Optional[ToolRunner] = None,
):
    """Factory returning a planner node bound to an LLM client, environment, and tool registry."""

    def planner_node(state: NovaGraphState) -> Dict[str, Any]:
        incident_id = state.get("incident_id", "")
        with incident_context(incident_id):
            triage_summary = state.get("triage_summary", "")
            hypothesis = state.get("root_cause_hypothesis", "")
            retry_count = state.get("retry_count", 0)
            action_seq = retry_count + 1

            from nova.llm.schemas import TriageResult

            failure_type = state.get("failure_type") or _infer_failure_type(triage_summary, hypothesis)
            implicated_resource = state.get("implicated_resource") or _extract_implicated_resource(triage_summary)
            is_rec = state.get("is_recoverable")
            if is_rec is None:
                is_rec = _infer_recoverability(triage_summary, hypothesis)

            bound_run_id = state.get("workflow_run_id")
            active_workflow_id = state.get("workflow_id")
            if not active_workflow_id and env is not None:
                try:
                    wf_state = (
                        env.get_state(execution_id=bound_run_id)
                        if bound_run_id and hasattr(env, "bind_execution")
                        else env.get_state()
                    )
                except Exception:
                    wf_state = None
                if wf_state is not None:
                    active_workflow_id = getattr(wf_state, "workflow_id", None)
                    bound_run_id = bound_run_id or getattr(wf_state, "run_id", None)

            # Build structured prior-attempt history so planner sees failed tools/postconditions
            prior_attempts: List[Dict[str, Any]] = []
            for rec in state.get("execution_history", []):
                rec_dict = rec.model_dump() if hasattr(rec, "model_dump") else dict(rec)
                prior_attempts.append(
                    {
                        "kind": "execution",
                        "action_id": rec_dict.get("action_id"),
                        "tool_name": rec_dict.get("tool_name"),
                        "success": rec_dict.get("success"),
                        "exit_code": rec_dict.get("exit_code"),
                        "stderr": (rec_dict.get("stderr") or "")[:400],
                    }
                )
            for vrec in state.get("verification_history", []):
                v_dict = vrec.model_dump() if hasattr(vrec, "model_dump") else dict(vrec)
                prior_attempts.append(
                    {
                        "kind": "verification",
                        "action_id": v_dict.get("action_id"),
                        "passed": v_dict.get("passed"),
                        "evidence": (v_dict.get("evidence") or "")[:400],
                        "details": v_dict.get("details") or {},
                    }
                )

            strategy_notes = None
            for refl in reversed(state.get("reflection_history", [])):
                if isinstance(refl, dict) and refl.get("phase") == "outcome":
                    strategy_notes = refl.get("alternative_strategy_notes") or refl.get("outcome_reflection")
                    break

            proxy_triage = TriageResult(
                failure_type=failure_type,
                implicated_resource=implicated_resource,
                triage_summary=triage_summary,
                root_cause_hypothesis=hypothesis,
                is_recoverable=is_rec,
                confidence="HIGH",
                workflow_id=active_workflow_id,
                workflow_run_id=bound_run_id,
                prior_attempts=prior_attempts,
                investigation_evidence=list(state.get("investigation_evidence") or []),
                strategy_notes=strategy_notes,
            )

            available_tools = runner.describe_tools() if runner is not None else AVAILABLE_TOOLS_FOR_LLM

            try:
                planner_result = llm.plan(
                    triage=proxy_triage,
                    available_tools=available_tools,
                    retry_count=retry_count,
                )
            except Exception as exc:
                reason = f"Planning failed due to LLM error: {redact_secrets(str(exc))}"
                telemetry.record_terminal_reason(incident_id, "FAILED", reason)
                log_event(logger, logging.ERROR, "planner_llm_failed", reason)
                return {
                    "plan": [],
                    "current_action_index": 0,
                    "status": IncidentStatus.FAILED,
                    "final_summary": reason,
                }

            # Contradictory plan check (defense-in-depth beyond Pydantic)
            if planner_result.is_escalation and planner_result.actions:
                reason = (
                    "Planning failed: Contradictory plan returned is_escalation=True alongside non-empty actions."
                )
                telemetry.record_terminal_reason(incident_id, "FAILED", reason)
                log_event(logger, logging.ERROR, "planner_contradictory_plan", reason)
                return {
                    "plan": [],
                    "current_action_index": 0,
                    "status": IncidentStatus.FAILED,
                    "final_summary": reason,
                }

            # Escalation path: LLM determined no automated fix is possible
            if planner_result.is_escalation:
                reason = f"Incident Escalated: {planner_result.escalation_reason}"
                telemetry.record_terminal_reason(incident_id, "ESCALATED", reason)
                log_event(logger, logging.WARNING, "planner_escalated", reason, failure_type=failure_type)
                esc_out: Dict[str, Any] = {
                    "plan": [],
                    "current_action_index": 0,
                    "status": IncidentStatus.ESCALATED,
                    "final_summary": reason,
                }
                if planner_result.reflection is not None:
                    esc_out["reflection_history"] = [
                        {"phase": "planning", "retry_count": retry_count, **planner_result.reflection.model_dump()}
                    ]
                return esc_out

            # Empty non-escalating plan guard: never transition to EXECUTING with zero actions
            if not planner_result.actions:
                reason = (
                    "Planning failed: Planner returned an empty non-escalating remediation plan. "
                    "Escalating safely instead of entering EXECUTING with no actions."
                )
                telemetry.record_terminal_reason(incident_id, "ESCALATED", reason)
                log_event(logger, logging.WARNING, "planner_empty_plan_rejected", reason)
                return {
                    "plan": [],
                    "current_action_index": 0,
                    "status": IncidentStatus.ESCALATED,
                    "final_summary": reason,
                }

            plan = [
                _schema_to_domain(
                    action_schema,
                    action_id=f"ACT_{action_seq:02d}_{i+1:02d}",
                    incident_id=incident_id,
                )
                for i, action_schema in enumerate(planner_result.actions)
            ]

            log_event(
                logger,
                logging.INFO,
                "plan_created",
                f"Formulated {len(plan)}-step remediation plan",
                action_count=len(plan),
                tools=[a.tool_name for a in plan],
                retry_count=retry_count,
            )

            out_plan: Dict[str, Any] = {
                "plan": plan,
                "current_action_index": 0,
                "status": IncidentStatus.EXECUTING,
            }
            if planner_result.reflection is not None:
                out_plan["reflection_history"] = [
                    {"phase": "planning", "retry_count": retry_count, **planner_result.reflection.model_dump()}
                ]
            return out_plan

    return planner_node


# ---------------------------------------------------------------------------
# Deterministic helpers — classify triage_summary strings into typed signals
# so we can build a valid TriageResult proxy even though state only stores strings
# ---------------------------------------------------------------------------

def _infer_failure_type(triage_summary: str, hypothesis: str) -> str:
    """
    Classify failure type from triage strings — used to build proxy TriageResult.

    Priority matters: SERVICE_OFFLINE must be checked before STALE_LOCKFILE because
    the replanner's updated_triage_summary may mention 'clear_lockfile' (the previous
    fix action), which contains the word 'lock'. Without this ordering, that mention
    would incorrectly classify a service-offline failure as a stale lockfile.
    """
    text = (triage_summary + " " + hypothesis).lower()

    # SERVICE_OFFLINE first — most specific signal (connection refused on a port)
    if "connection" in text and "refused" in text:
        return "SERVICE_OFFLINE"

    # DATA_CORRUPTION — unrecoverable, must escalate immediately
    if "checksum" in text or "corrupt" in text or "zero-byte" in text:
        return "DATA_CORRUPTION"

    # STALE_LOCKFILE — requires .lock extension or explicit "orphaned lockfile" phrase
    # The word "lock" alone is too broad (e.g. "clear_lockfile" action name contains it)
    if ".lock" in text or "orphaned lockfile" in text or "stale lock" in text:
        return "STALE_LOCKFILE"

    # DEPENDENCY_FAILURE — downstream component failure after upstream fix
    if "dependency" in text or "downstream" in text:
        return "DEPENDENCY_FAILURE"

    return "UNKNOWN"


def _extract_implicated_resource(triage_summary: str) -> str:
    """
    Extract the specific resource name (lock filename, service name) from the
    triage summary string.  Runs before the LLM so the proxy TriageResult has
    the right implicated_resource for the LLM planner to use in parameters.
    """
    import re
    # Try lockfile pattern first
    lock_match = re.search(r"'([\w./]+\.lock)'", triage_summary)
    if lock_match:
        return lock_match.group(1)
    # Try service name pattern
    svc_match = re.search(r"'([\w_]+(?:service|db|warehouse|svc)[\w_]*)'", triage_summary, re.IGNORECASE)
    if svc_match:
        return svc_match.group(1)
    # Try quoted name fallback
    quoted = re.search(r"'([^']{3,50})'", triage_summary)
    if quoted:
        return quoted.group(1)
    return "unknown_resource"


def _infer_recoverability(triage_summary: str, hypothesis: str) -> bool:
    """Determine recoverability from triage strings."""
    text = (triage_summary + " " + hypothesis).lower()
    return "checksum" not in text and "corrupt" not in text and "zero-byte" not in text


# Keep the old function name as an alias so existing graph.py import still works
# temporarily — will be updated in graph.py
def planner_node(state: NovaGraphState) -> Dict[str, Any]:
    raise RuntimeError(
        "planner_node() called without LLM client. Use create_planner_node(llm) instead."
    )
