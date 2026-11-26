import hashlib
import inspect
import json
import logging
from typing import Any, Dict, List, Optional
from nova.llm.client import BaseLLMClient
from nova.llm.schemas import PlanCritiqueResult, PlannedActionSchema, PlannerResult, TriageResult
from nova.observability import incident_context, log_event, redact_secrets, telemetry
from nova.state import IncidentStatus, NovaGraphState, PlannedAction, RiskLevel, build_prior_attempts
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


def _validate_structural_actions(
    actions: List[PlannedActionSchema],
    runner: Optional[ToolRunner],
    triage: TriageResult,
) -> List[str]:
    """Deterministic recoverability boundary check (safe across both ablation arms)."""
    errors: List[str] = []
    if triage.is_recoverable is False and actions:
        errors.append(
            f"Plan proposed {len(actions)} action(s) despite diagnosis marking is_recoverable=False."
        )
    return errors


def _evaluate_plan_semantics(
    actions: List[PlannedActionSchema],
    env: Optional[Any],
    triage: TriageResult,
    state: NovaGraphState,
    runner: Optional[ToolRunner] = None,
) -> List[str]:
    """
    Planning critique checks validating that proposed tools and arguments are
    available in the live registry, schema-valid, compatible with environment
    capabilities, and consistent with prior attempts and observed evidence.
    """
    issues: List[str] = []
    exec_history = state.get("execution_history") or []
    evidence_text = " ".join(triage.investigation_evidence or []).lower()

    for idx, act in enumerate(actions, start=1):
        if runner is not None:
            if not runner.is_tool_allowed(act.tool_name):
                issues.append(f"Step {idx}: tool '{act.tool_name}' is not in the active ToolRunner registry.")
                continue
            try:
                tool = runner.get_tool(act.tool_name)
                tool.args_schema.model_validate(act.parameters)
            except Exception as exc:
                issues.append(f"Step {idx}: invalid parameters for '{act.tool_name}': {exc}")

        if act.tool_name == "restart_service" and env is not None:
            svc = act.parameters.get("service_name", "")
            if hasattr(env, "is_registered_service") and not env.is_registered_service(svc):
                issues.append(f"Step {idx}: service '{svc}' is not a registered dependency.")
            elif hasattr(env, "is_restartable_service") and not env.is_restartable_service(svc):
                issues.append(f"Step {idx}: service '{svc}' cannot be restarted by NOVA.")

        if act.tool_name == "clear_lockfile" and env is not None and hasattr(env, "is_lock_active"):
            lock_name = act.parameters.get("lock_name", "")
            if lock_name and env.is_lock_active(lock_name):
                issues.append(f"Step {idx}: lockfile '{lock_name}' belongs to an active worker.")

        if act.tool_name == "resume_workflow" and triage.workflow_run_id and env is not None and hasattr(env, "bind_execution"):
            req_eid = act.parameters.get("execution_id")
            if req_eid and req_eid != triage.workflow_run_id:
                issues.append(
                    f"Step {idx}: resume_workflow execution_id '{req_eid}' does not match bound '{triage.workflow_run_id}'."
                )

    # Check if the plan repeats an action that failed in the most recent cycle with no intervening env change
    if exec_history:
        last_rec = exec_history[-1]
        last_dict = last_rec.model_dump() if hasattr(last_rec, "model_dump") else dict(last_rec)
        if not last_dict.get("success", True):
            last_tool = last_dict.get("tool_name")
            last_params = json.dumps(last_dict.get("parameters") or {}, sort_keys=True)
            for act in actions:
                act_params = json.dumps(act.parameters or {}, sort_keys=True)
                if act.tool_name == last_tool and act_params == last_params and len(actions) == 1:
                    issues.append(
                        f"Plan repeats failed action '{act.tool_name}' with identical parameters without addressing prior failure."
                    )

    # Check if a bare resume_workflow is proposed when a restartable dependency is confirmed offline
    if len(actions) == 1 and actions[0].tool_name == "resume_workflow":
        implicated = triage.implicated_resource or ""
        is_restartable_target = bool(
            env is not None
            and implicated
            and (
                (hasattr(env, "is_restartable_service") and env.is_restartable_service(implicated))
                or (
                    not hasattr(env, "is_restartable_service")
                    and hasattr(env, "is_registered_service")
                    and env.is_registered_service(implicated)
                )
            )
        )
        if (triage.failure_type == "SERVICE_OFFLINE" and is_restartable_target) or (
            "healthy: false" in evidence_text and ("pgdatabase" in evidence_text or "reporting_warehouse_service" in evidence_text)
        ):
            issues.append(
                "Plan proposes bare workflow restart without first restarting confirmed offline dependency."
            )

    return issues


def _call_critique_plan(
    llm: BaseLLMClient,
    triage: TriageResult,
    proposed_actions: List[Dict[str, Any]],
    available_tools: List[Dict[str, Any]],
    retry_count: int,
) -> Optional[PlanCritiqueResult]:
    critique_fn = getattr(llm, "critique_plan", None)
    if critique_fn is None:
        return None
    sig = inspect.signature(critique_fn)
    kwargs: Dict[str, Any] = {
        "triage": triage,
        "proposed_actions": proposed_actions,
        "available_tools": available_tools,
        "retry_count": retry_count,
    }
    filtered = {
        k: v for k, v in kwargs.items()
        if k in sig.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    }
    return critique_fn(**filtered)


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

            # Build complete, non-lossy prior-attempt history so planner sees parameters, outputs,
            # command vs. postcondition status, environment changes, and execution/attempt IDs.
            prior_attempts = build_prior_attempts(state)

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

            initial_plan_dump = [a.model_dump() for a in planner_result.actions]
            reflection_entries: List[Dict[str, Any]] = []
            plan_revision_entries: List[Dict[str, Any]] = []
            critique_enabled = bool(getattr(llm, "enable_reflection", False))
            if planner_result.reflection is not None:
                reflection_entries.append(
                    {"phase": "planning", "retry_count": retry_count, **planner_result.reflection.model_dump()}
                )

            structural_errors = _validate_structural_actions(planner_result.actions, runner, proxy_triage)
            semantic_errors = (
                _evaluate_plan_semantics(planner_result.actions, env, proxy_triage, state, runner=runner)
                if critique_enabled
                else []
            )

            critique_res: Optional[PlanCritiqueResult] = None
            if critique_enabled:
                try:
                    critique_res = _call_critique_plan(
                        llm=llm,
                        triage=proxy_triage,
                        proposed_actions=initial_plan_dump,
                        available_tools=available_tools,
                        retry_count=retry_count,
                    )
                except Exception as critique_exc:
                    log_event(
                        logger,
                        logging.WARNING,
                        "planner_critique_error",
                        f"Planning critique failed softly: {redact_secrets(str(critique_exc))}",
                    )

            refl_obj = planner_result.reflection
            refl_rejected = bool(
                critique_enabled
                and refl_obj is not None
                and (
                    not refl_obj.addresses_root_cause
                    or refl_obj.contradicts_prior_attempts
                    or refl_obj.requires_revision
                    or refl_obj.requires_further_investigation
                    or refl_obj.should_escalate
                )
            )
            critique_rejected = bool(
                critique_res is not None
                and (
                    not critique_res.approved
                    or not critique_res.addresses_root_cause
                    or critique_res.contradicts_prior_attempts
                    or not critique_res.tools_valid_and_appropriate
                    or critique_res.requires_further_investigation
                    or critique_res.should_escalate
                )
            )

            needs_critique_action = bool(structural_errors or semantic_errors or refl_rejected or critique_rejected)

            if needs_critique_action:
                critique_reasons: List[str] = list(structural_errors) + list(semantic_errors)
                if refl_rejected and refl_obj is not None:
                    critique_reasons.append(
                        refl_obj.critique_notes
                        or refl_obj.strategy_assessment
                        or "Planning reflection flagged misalignment with root cause or prior attempts."
                    )
                if critique_res is not None and critique_rejected:
                    critique_reasons.append(critique_res.critique_summary)
                    if critique_res.recommended_adjustment:
                        critique_reasons.append(f"Recommended adjustment: {critique_res.recommended_adjustment}")

                critique_summary = " | ".join(r for r in critique_reasons if r)
                critique_record: Dict[str, Any] = {
                    "phase": "planning_critique",
                    "retry_count": retry_count,
                    "approved": False,
                    "structural_errors": structural_errors,
                    "semantic_errors": semantic_errors,
                    "critique_summary": critique_summary,
                }
                if critique_res is not None:
                    critique_record["model_critique"] = critique_res.model_dump()
                reflection_entries.append(critique_record)

                want_reinvestigate = bool(
                    (critique_res and critique_res.requires_further_investigation)
                    or (refl_obj and refl_obj.requires_further_investigation)
                )
                want_escalate = bool(
                    (critique_res and critique_res.should_escalate)
                    or (refl_obj and refl_obj.should_escalate)
                    or (proxy_triage.is_recoverable is False)
                )

                # Route back to triage if critique determines more diagnostic observation is needed
                if want_reinvestigate and state.get("investigation_steps", 0) < state.get("max_investigation_steps", 5):
                    plan_revision_entries.append(
                        {
                            "retry_count": retry_count,
                            "initial_plan": initial_plan_dump,
                            "critique": critique_summary,
                            "final_plan": [],
                            "outcome": "REINVESTIGATE",
                        }
                    )
                    log_event(logger, logging.INFO, "planner_critique_reinvestigate", critique_summary)
                    return {
                        "plan": [],
                        "current_action_index": 0,
                        "status": IncidentStatus.TRIAGING,
                        "investigation_evidence": [f"[Planning Critique] {critique_summary}"],
                        "reflection_history": reflection_entries,
                        "plan_revisions": plan_revision_entries,
                    }

                if want_escalate or not critique_enabled:
                    reason = f"Incident Escalated after planning validation/critique: {critique_summary}"
                    telemetry.record_terminal_reason(incident_id, "ESCALATED", reason)
                    plan_revision_entries.append(
                        {
                            "retry_count": retry_count,
                            "initial_plan": initial_plan_dump,
                            "critique": critique_summary,
                            "final_plan": [],
                            "outcome": "ESCALATED",
                        }
                    )
                    log_event(logger, logging.WARNING, "planner_critique_escalated", reason)
                    return {
                        "plan": [],
                        "current_action_index": 0,
                        "status": IncidentStatus.ESCALATED,
                        "final_summary": reason,
                        "reflection_history": reflection_entries,
                        "plan_revisions": plan_revision_entries,
                    }

                # Attempt one bounded revision using explicit critique feedback
                revised_triage = proxy_triage.model_copy(update={"critique_feedback": critique_summary})
                try:
                    revised_result = llm.plan(
                        triage=revised_triage,
                        available_tools=available_tools,
                        retry_count=retry_count,
                    )
                except Exception as rev_exc:
                    reason = f"Incident Escalated: Plan critique rejected initial plan ({critique_summary}), and revision failed: {redact_secrets(str(rev_exc))}"
                    telemetry.record_terminal_reason(incident_id, "ESCALATED", reason)
                    plan_revision_entries.append(
                        {
                            "retry_count": retry_count,
                            "initial_plan": initial_plan_dump,
                            "critique": critique_summary,
                            "final_plan": [],
                            "outcome": "ESCALATED",
                        }
                    )
                    return {
                        "plan": [],
                        "current_action_index": 0,
                        "status": IncidentStatus.ESCALATED,
                        "final_summary": reason,
                        "reflection_history": reflection_entries,
                        "plan_revisions": plan_revision_entries,
                    }

                if revised_result.reflection is not None:
                    reflection_entries.append(
                        {
                            "phase": "planning_revised",
                            "retry_count": retry_count,
                            **revised_result.reflection.model_dump(),
                        }
                    )

                if revised_result.is_escalation:
                    reason = (
                        f"Incident Escalated after planning critique: "
                        f"{revised_result.escalation_reason or critique_summary}"
                    )
                    telemetry.record_terminal_reason(incident_id, "ESCALATED", reason)
                    plan_revision_entries.append(
                        {
                            "retry_count": retry_count,
                            "initial_plan": initial_plan_dump,
                            "critique": critique_summary,
                            "final_plan": [],
                            "outcome": "ESCALATED",
                        }
                    )
                    log_event(logger, logging.WARNING, "planner_revised_to_escalation", reason)
                    return {
                        "plan": [],
                        "current_action_index": 0,
                        "status": IncidentStatus.ESCALATED,
                        "final_summary": reason,
                        "reflection_history": reflection_entries,
                        "plan_revisions": plan_revision_entries,
                    }

                rev_struct_err = _validate_structural_actions(revised_result.actions, runner, proxy_triage)
                rev_sem_err = _evaluate_plan_semantics(revised_result.actions, env, proxy_triage, state, runner=runner)
                if not revised_result.actions or rev_struct_err or rev_sem_err:
                    combined = " | ".join(rev_struct_err + rev_sem_err) or "Revised plan was empty."
                    reason = (
                        f"Incident Escalated: Revised plan failed validation after critique "
                        f"({combined}). Initial critique: {critique_summary}"
                    )
                    telemetry.record_terminal_reason(incident_id, "ESCALATED", reason)
                    plan_revision_entries.append(
                        {
                            "retry_count": retry_count,
                            "initial_plan": initial_plan_dump,
                            "critique": critique_summary,
                            "final_plan": [a.model_dump() for a in revised_result.actions],
                            "outcome": "ESCALATED",
                        }
                    )
                    log_event(logger, logging.WARNING, "planner_revised_plan_invalid", reason)
                    return {
                        "plan": [],
                        "current_action_index": 0,
                        "status": IncidentStatus.ESCALATED,
                        "final_summary": reason,
                        "reflection_history": reflection_entries,
                        "plan_revisions": plan_revision_entries,
                    }

                planner_result = revised_result
                plan_revision_entries.append(
                    {
                        "retry_count": retry_count,
                        "initial_plan": initial_plan_dump,
                        "critique": critique_summary,
                        "final_plan": [a.model_dump() for a in planner_result.actions],
                        "outcome": "REVISED",
                    }
                )
            elif critique_res is not None:
                reflection_entries.append(
                    {
                        "phase": "planning_critique",
                        "retry_count": retry_count,
                        "approved": True,
                        "model_critique": critique_res.model_dump(),
                    }
                )

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
            if reflection_entries:
                out_plan["reflection_history"] = reflection_entries
            if plan_revision_entries:
                out_plan["plan_revisions"] = plan_revision_entries
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
