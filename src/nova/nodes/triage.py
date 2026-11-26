import inspect
import json
import logging
import re
import time
from typing import Any, Dict, Optional

from nova.llm.client import BaseLLMClient
from nova.llm.schemas import (
    DiagnosticToolCall,
    InvestigationActionType,
    InvestigationDecision,
    TriageResult,
)
from nova.observability import incident_context, log_event, redact_secrets, telemetry
from nova.state import (
    DiagnosticRecord,
    IncidentStatus,
    NovaGraphState,
    RiskLevel,
    build_prior_attempts,
)
from nova.tools.runner import ToolRunner, create_default_tool_runner

logger = logging.getLogger("nova.nodes.triage")


def _compute_env_fingerprint(
    env: Any,
    bound_run_id: Optional[str],
    retry_count: int,
) -> Optional[str]:
    """Computes a lightweight deterministic fingerprint of observable environment state."""
    try:
        wf_state = (
            env.get_state(execution_id=bound_run_id)
            if bound_run_id and hasattr(env, "bind_execution")
            else env.get_state()
        )
    except Exception:
        wf_state = None
    if wf_state is None:
        return f"retry={retry_count}"
    jobs_sig = sorted(
        (
            jid,
            j.status.value if hasattr(j.status, "value") else str(j.status),
            getattr(j, "attempt_count", 0),
        )
        for jid, j in (getattr(wf_state, "jobs", None) or {}).items()
    )
    svcs_sig = sorted((getattr(wf_state, "services", None) or {}).items())
    locks_sig = sorted(getattr(wf_state, "active_locks", None) or [])
    fp = {
        "retry": retry_count,
        "run_id": getattr(wf_state, "run_id", None),
        "exec_state": getattr(wf_state, "execution_state", None),
        "attempts": getattr(wf_state, "total_attempts", None),
        "jobs": jobs_sig,
        "services": svcs_sig,
        "locks": locks_sig,
    }
    return json.dumps(fp, sort_keys=True)


def _call_investigate(
    llm: BaseLLMClient,
    target_service: Optional[str],
    raw_symptoms: str,
    investigation_evidence: list[str],
    diagnostic_history: list[dict[str, Any]],
    available_tools: list[dict[str, Any]],
    step_number: int,
    prior_attempts: list[dict[str, Any]],
    current_diagnosis: Optional[dict[str, Any]],
    remediation_capabilities: Optional[list[dict[str, Any]]] = None,
) -> InvestigationDecision:
    """Invokes llm.investigate while preserving backward compatibility with 6-arg test stubs."""
    sig = inspect.signature(llm.investigate)
    kwargs: Dict[str, Any] = {
        "target_service": target_service,
        "raw_symptoms": raw_symptoms,
        "investigation_evidence": investigation_evidence,
        "diagnostic_history": diagnostic_history,
        "available_tools": available_tools,
        "step_number": step_number,
    }
    has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if has_varkw or "prior_attempts" in sig.parameters:
        kwargs["prior_attempts"] = prior_attempts
    if has_varkw or "current_diagnosis" in sig.parameters:
        kwargs["current_diagnosis"] = current_diagnosis
    if has_varkw or "remediation_capabilities" in sig.parameters:
        kwargs["remediation_capabilities"] = remediation_capabilities
    return llm.investigate(**kwargs)


def create_triage_node(
    env: Any,
    llm: BaseLLMClient,
    tool_runner: Optional[ToolRunner] = None,
):
    """
    Factory returning an autonomous investigation node bound to the environment,
    LLM client, and read-only diagnostic tool registry.
    """
    runner = tool_runner or create_default_tool_runner(env)

    def triage_node(state: NovaGraphState) -> Dict[str, Any]:
        with incident_context(state.get("incident_id")):
            return _run_triage(state)

    def _run_triage(state: NovaGraphState) -> Dict[str, Any]:
        """
        Autonomous Investigation Node (Milestone 7.2 Phase C):

        Transforms Triage from a one-shot log reader into a bounded, iterative
        investigation mechanism.

        1. Inspects accumulated evidence, target_service (can be UNKNOWN), and diagnostic history.
        2. Enforces bounded investigation budget (investigation_steps < max_investigation_steps).
        3. Invokes LLM.investigate() to decide:
           - DIAGNOSTIC_ACTION: Selects a read-only tool (e.g. get_workflow_status, read_job_logs,
             inspect_lockfiles, check_service_status, inspect_job_dependencies).
           - DIAGNOSIS_READY: Concludes sufficient evidence is gathered; maps structured diagnosis
             into graph state and advances to PLANNING.
           - ESCALATE: Safely halts and escalates unrecoverable or out-of-scope incidents.
        4. Deterministic Safeguards:
           - Strict READ_ONLY restriction: Destructive tools cannot be invoked in triage.
           - Loop detection: Repeated identical diagnostic calls halt unproductive cycles.
           - Structured telemetry: All diagnostic calls recorded into state.diagnostic_history.
        """
        incident_id = state.get("incident_id")
        target_service = state.get("target_service")
        raw_symptoms = state.get("raw_symptoms", "")
        diagnostic_history = list(state.get("diagnostic_history") or [])
        investigation_evidence = list(state.get("investigation_evidence") or [])
        step_number = state.get("investigation_steps", 0) + 1
        max_steps = state.get("max_investigation_steps", 5)

        # M11/M12/M13: Capture active workflow_id / workflow_run_id & baseline attempt counts if unbound
        bound_wf_id = state.get("workflow_id")
        bound_run_id = state.get("workflow_run_id")
        baseline_attempts = state.get("baseline_attempts")
        baseline_task_attempts = state.get("baseline_task_attempts")
        baseline_task_states = state.get("baseline_task_states")
        wf_state = None
        if (
            not bound_run_id
            or baseline_attempts is None
            or baseline_task_attempts is None
            or baseline_task_states is None
        ):
            try:
                wf_state = env.get_state(execution_id=bound_run_id) if bound_run_id and hasattr(env, "bind_execution") else env.get_state()
            except Exception:
                wf_state = None
            if wf_state is not None:
                bound_wf_id = bound_wf_id or getattr(wf_state, "workflow_id", None)
                bound_run_id = bound_run_id or getattr(wf_state, "run_id", None)
                if baseline_attempts is None and hasattr(wf_state, "total_attempts"):
                    baseline_attempts = getattr(wf_state, "total_attempts")
                if getattr(wf_state, "jobs", None):
                    if baseline_task_attempts is None:
                        task_atts = {
                            tid: getattr(t, "attempt_count", 0)
                            for tid, t in wf_state.jobs.items()
                            if hasattr(t, "attempt_count")
                        }
                        if task_atts:
                            baseline_task_attempts = task_atts
                    if baseline_task_states is None:
                        baseline_task_states = {
                            tid: (t.status.value if hasattr(t.status, "value") else str(t.status))
                            for tid, t in wf_state.jobs.items()
                        }

        # Budget Check: Bounded investigation loop
        if step_number > max_steps:
            reason = (
                f"Incident Escalated: Autonomous investigation exceeded maximum budget "
                f"({max_steps} steps) without forming a conclusive diagnosis."
            )
            telemetry.record_terminal_reason(incident_id, "ESCALATED", reason)
            log_event(logger, logging.WARNING, "investigation_budget_exhausted", reason, max_steps=max_steps)
            return {
                "status": IncidentStatus.ESCALATED,
                "final_summary": reason,
            }

        # Dynamic read-only tool discovery directly from active ToolRunner registry
        diagnostic_tools = runner.describe_tools(risk_filter=RiskLevel.READ_ONLY)
        remediation_capabilities = [
            t for t in runner.describe_tools() if t.get("risk_level") != RiskLevel.READ_ONLY.value
        ]
        prior_attempts = build_prior_attempts(state)
        current_diagnosis = None
        if state.get("failure_type") or state.get("root_cause_hypothesis"):
            current_diagnosis = {
                "failure_type": state.get("failure_type"),
                "implicated_resource": state.get("implicated_resource"),
                "triage_summary": state.get("triage_summary"),
                "root_cause_hypothesis": state.get("root_cause_hypothesis"),
                "is_recoverable": state.get("is_recoverable"),
            }
        execution_epoch = len(state.get("execution_history") or [])
        current_fp = _compute_env_fingerprint(env, bound_run_id, state.get("retry_count", 0))

        # Call LLM to investigate
        try:
            decision: InvestigationDecision = _call_investigate(
                llm=llm,
                target_service=target_service,
                raw_symptoms=raw_symptoms,
                investigation_evidence=investigation_evidence,
                diagnostic_history=[d.model_dump() if hasattr(d, "model_dump") else d for d in diagnostic_history],
                available_tools=diagnostic_tools,
                step_number=step_number,
                prior_attempts=prior_attempts,
                current_diagnosis=current_diagnosis,
                remediation_capabilities=remediation_capabilities,
            )
        except Exception as exc:
            safe_exc = redact_secrets(str(exc))
            # Fallback only when single-shot triage is viable for a known service
            if target_service and target_service != "UNKNOWN":
                try:
                    log_content = env.read_job_log(target_service, tail_lines=50)
                    triage_res = llm.triage(target_service=target_service, log_content=log_content)
                    decision = InvestigationDecision(
                        decision_type=InvestigationActionType.DIAGNOSIS_READY,
                        diagnosis=triage_res,
                    )
                except Exception as fallback_exc:
                    safe_fb = redact_secrets(str(fallback_exc))
                    reason = f"Investigation failed due to LLM error: {safe_fb}"
                    telemetry.record_terminal_reason(incident_id, "FAILED", reason)
                    log_event(logger, logging.ERROR, "investigation_llm_failed", reason)
                    return {
                        "status": IncidentStatus.FAILED,
                        "final_summary": reason,
                    }
            else:
                reason = f"Investigation failed due to LLM error: {safe_exc}"
                telemetry.record_terminal_reason(incident_id, "FAILED", reason)
                log_event(logger, logging.ERROR, "investigation_llm_failed", reason)
                return {
                    "status": IncidentStatus.FAILED,
                    "final_summary": reason,
                }

        def _execute_diagnostic_call(
            diag_call: DiagnosticToolCall,
            extra_reflection: Optional[Dict[str, Any]] = None,
            extra_evidence: Optional[list[str]] = None,
        ) -> Dict[str, Any]:
            tool_name = diag_call.tool_name

            # Security Principle: Only READ_ONLY tools allowed
            try:
                tool = runner.get_tool(tool_name)
            except Exception as e:
                reason = f"Security Alert: Requested diagnostic tool '{tool_name}' is not registered: {e}"
                telemetry.record_terminal_reason(incident_id, "ESCALATED", reason)
                log_event(logger, logging.WARNING, "investigation_unknown_tool", reason, tool_name=tool_name)
                return {
                    "status": IncidentStatus.ESCALATED,
                    "final_summary": reason,
                }

            if tool.risk_level != RiskLevel.READ_ONLY:
                reason = (
                    f"Security Alert: Non-read-only tool '{tool_name}' ({tool.risk_level.value}) "
                    "requested during investigation. Disallowed by safety policy."
                )
                telemetry.record_terminal_reason(incident_id, "ESCALATED", reason)
                log_event(
                    logger,
                    logging.WARNING,
                    "investigation_policy_violation",
                    reason,
                    tool_name=tool_name,
                    risk_level=tool.risk_level.value,
                )
                return {
                    "status": IncidentStatus.ESCALATED,
                    "final_summary": reason,
                }

            # Context-aware Unproductive Loop Detection:
            # Allow repeating a diagnostic call if the environment changed since the prior call
            # (e.g., remediation executed or workflow state changed); block pointless duplicates
            # against unchanged conditions.
            for prev in reversed(diagnostic_history):
                prev_tool = prev.tool_name if hasattr(prev, "tool_name") else prev.get("tool_name")
                prev_params = prev.parameters if hasattr(prev, "parameters") else prev.get("parameters", {})
                if prev_tool == tool_name and prev_params == diag_call.parameters:
                    prev_epoch = prev.execution_epoch if hasattr(prev, "execution_epoch") else prev.get("execution_epoch", 0)
                    prev_fp = prev.env_fingerprint if hasattr(prev, "env_fingerprint") else prev.get("env_fingerprint")
                    env_changed = (execution_epoch > prev_epoch) or (
                        prev_fp is not None and current_fp is not None and prev_fp != current_fp
                    )
                    if not env_changed:
                        reason = (
                            f"Incident Escalated: Detected repeated unproductive investigation call to "
                            f"'{tool_name}' with parameters {diag_call.parameters}. Investigation halted."
                        )
                        telemetry.record_terminal_reason(incident_id, "ESCALATED", reason)
                        log_event(
                            logger,
                            logging.WARNING,
                            "investigation_loop_detected",
                            reason,
                            tool_name=tool_name,
                        )
                        return {
                            "status": IncidentStatus.ESCALATED,
                            "final_summary": reason,
                        }
                    break

            # Execute read-only diagnostic tool
            executed_at = time.strftime("%Y-%m-%d %H:%M:%S")
            t0 = time.perf_counter()
            try:
                result = tool.run(**diag_call.parameters)
                tool_output = result.stdout if result.success else f"Error: {result.stderr}"
                success = result.success
            except Exception as e:
                tool_output = f"Execution error: {str(e)}"
                success = False
            dt = time.perf_counter() - t0
            telemetry.record_tool_call(
                incident_id=incident_id,
                tool_name=tool_name,
                risk_level=tool.risk_level.value,
                is_diagnostic=True,
                duration_seconds=dt,
                success=success,
            )
            log_event(
                logger,
                logging.INFO,
                "diagnostic_tool_invoked",
                f"Step {step_number}: diagnostic tool '{tool_name}' (success={success})",
                step=step_number,
                tool_name=tool_name,
                success=success,
                duration_seconds=round(dt, 4),
            )

            rec = DiagnosticRecord(
                step=step_number,
                tool_name=tool_name,
                parameters=diag_call.parameters,
                output=tool_output,
                success=success,
                timestamp=executed_at,
                rationale=diag_call.rationale,
                execution_epoch=execution_epoch,
                env_fingerprint=current_fp,
            )
            evidence_line = f"Step {step_number} [{tool_name}({diag_call.parameters})]: {tool_output}"
            new_evidence = list(extra_evidence or []) + [evidence_line]
            out_dict: Dict[str, Any] = {
                "diagnostic_history": [rec],
                "investigation_evidence": new_evidence,
                "investigation_steps": step_number,
                "workflow_id": bound_wf_id,
                "workflow_run_id": bound_run_id,
                "status": IncidentStatus.TRIAGING,
            }
            if baseline_attempts is not None:
                out_dict["baseline_attempts"] = baseline_attempts
            if baseline_task_attempts is not None:
                out_dict["baseline_task_attempts"] = baseline_task_attempts
            if baseline_task_states is not None:
                out_dict["baseline_task_states"] = baseline_task_states
            if extra_reflection is not None:
                out_dict["reflection_history"] = [extra_reflection]
            elif decision.reflection is not None:
                out_dict["reflection_history"] = [
                    {
                        "phase": "diagnostic",
                        "step": step_number,
                        **decision.reflection.model_dump(),
                    }
                ]
            return out_dict

        # Case 1: DIAGNOSTIC_ACTION
        if decision.decision_type == InvestigationActionType.DIAGNOSTIC_ACTION:
            return _execute_diagnostic_call(decision.diagnostic_tool)

        # Case 2: DIAGNOSIS_READY
        elif decision.decision_type == InvestigationActionType.DIAGNOSIS_READY:
            diag = decision.diagnosis

            # Consequential Diagnostic Reflection Gate:
            # Never transition to PLANNING if reflection flags insufficient evidence!
            if decision.reflection is not None and decision.reflection.evidence_sufficient is False:
                refl_entry = {
                    "phase": "diagnostic",
                    "step": step_number,
                    "blocked_premature_diagnosis": True,
                    **decision.reflection.model_dump(),
                }
                if step_number >= max_steps:
                    reason = (
                        f"Incident Escalated: Investigation budget ({max_steps} steps) exhausted while "
                        f"diagnostic reflection reports insufficient evidence: "
                        f"{decision.reflection.remaining_uncertainty or 'missing distinguishing observation'}."
                    )
                    telemetry.record_terminal_reason(incident_id, "ESCALATED", reason)
                    log_event(logger, logging.WARNING, "investigation_insufficient_evidence_exhausted", reason)
                    return {
                        "status": IncidentStatus.ESCALATED,
                        "final_summary": reason,
                        "reflection_history": [refl_entry],
                        "investigation_evidence": [
                            f"[Diagnostic Reflection Step {step_number}] Escalated: {reason}"
                        ],
                    }

                critique_note = (
                    f"[Diagnostic Reflection Step {step_number}] Rejected premature DIAGNOSIS_READY "
                    f"because evidence_sufficient=False ({decision.reflection.remaining_uncertainty or 'uncertainty remains'})."
                )
                if decision.reflection.recommended_diagnostic_tool is not None:
                    return _execute_diagnostic_call(
                        decision.reflection.recommended_diagnostic_tool,
                        extra_reflection=refl_entry,
                        extra_evidence=[critique_note],
                    )
                return {
                    "investigation_steps": step_number,
                    "investigation_evidence": [critique_note],
                    "reflection_history": [refl_entry],
                    "workflow_id": bound_wf_id,
                    "workflow_run_id": bound_run_id,
                    "status": IncidentStatus.TRIAGING,
                }

            # Groundedness guard: UNKNOWN target with zero diagnostic observations cannot jump straight to PLANNING
            if (not target_service or target_service == "UNKNOWN") and len(diagnostic_history) == 0:
                if step_number >= max_steps:
                    reason = (
                        "Incident Escalated: Investigation budget exhausted with zero diagnostic "
                        "observations for an UNKNOWN target."
                    )
                    telemetry.record_terminal_reason(incident_id, "ESCALATED", reason)
                    return {
                        "status": IncidentStatus.ESCALATED,
                        "final_summary": reason,
                    }
                return {
                    "investigation_steps": step_number,
                    "investigation_evidence": [
                        f"[Diagnostic Guard Step {step_number}] Rejected ungrounded DIAGNOSIS_READY with zero observations for UNKNOWN target; gather workflow status/logs first."
                    ],
                    "workflow_id": bound_wf_id,
                    "workflow_run_id": bound_run_id,
                    "status": IncidentStatus.TRIAGING,
                }

            new_target = target_service
            if not new_target or new_target == "UNKNOWN":
                # Check diagnostic history (e.g. job_id from read_job_logs)
                for rec in reversed(diagnostic_history):
                    rec_tool = rec.tool_name if hasattr(rec, "tool_name") else rec.get("tool_name")
                    rec_params = rec.parameters if hasattr(rec, "parameters") else rec.get("parameters", {})
                    if rec_tool == "read_job_logs" and rec_params.get("job_id"):
                        new_target = rec_params.get("job_id")
                        break
                if (not new_target or new_target == "UNKNOWN") and diag.implicated_resource:
                    new_target = diag.implicated_resource
                if not new_target or new_target == "UNKNOWN":
                    job_m = re.search(r"(JOB_0[1-4]_[A-Z]+)", f"{diag.triage_summary} {diag.root_cause_hypothesis}")
                    if job_m:
                        new_target = job_m.group(1)

            log_event(
                logger,
                logging.INFO,
                "diagnosis_ready",
                f"Diagnosis ready: {diag.failure_type} on {diag.implicated_resource}",
                failure_type=diag.failure_type,
                implicated_resource=diag.implicated_resource,
                is_recoverable=diag.is_recoverable,
            )
            out_ready: Dict[str, Any] = {
                "failure_type": diag.failure_type,
                "implicated_resource": diag.implicated_resource,
                "is_recoverable": diag.is_recoverable,
                "triage_summary": diag.triage_summary,
                "root_cause_hypothesis": diag.root_cause_hypothesis,
                "target_service": new_target or diag.implicated_resource,
                "workflow_id": bound_wf_id,
                "workflow_run_id": bound_run_id,
                "status": IncidentStatus.PLANNING,
            }
            if baseline_attempts is not None:
                out_ready["baseline_attempts"] = baseline_attempts
            if baseline_task_attempts is not None:
                out_ready["baseline_task_attempts"] = baseline_task_attempts
            if baseline_task_states is not None:
                out_ready["baseline_task_states"] = baseline_task_states
            if decision.reflection is not None:
                out_ready["reflection_history"] = [
                    {
                        "phase": "diagnostic",
                        "step": step_number,
                        **decision.reflection.model_dump(),
                    }
                ]
            if current_diagnosis is not None and (
                diag.failure_type != current_diagnosis.get("failure_type")
                or diag.implicated_resource != current_diagnosis.get("implicated_resource")
                or diag.root_cause_hypothesis != current_diagnosis.get("root_cause_hypothesis")
            ):
                out_ready["hypothesis_revisions"] = [
                    {
                        "phase": "triage_reinvestigation",
                        "step": step_number,
                        "previous": current_diagnosis,
                        "revised": {
                            "failure_type": diag.failure_type,
                            "implicated_resource": diag.implicated_resource,
                            "triage_summary": diag.triage_summary,
                            "root_cause_hypothesis": diag.root_cause_hypothesis,
                            "is_recoverable": diag.is_recoverable,
                        },
                    }
                ]
            return out_ready

        # Case 3: ESCALATE
        elif decision.decision_type == InvestigationActionType.ESCALATE:
            reason = f"Incident Escalated during investigation: {decision.escalation_reason}"
            telemetry.record_terminal_reason(incident_id, "ESCALATED", reason)
            log_event(logger, logging.WARNING, "investigation_escalated", reason)
            esc_dict: Dict[str, Any] = {
                "status": IncidentStatus.ESCALATED,
                "final_summary": reason,
                "workflow_id": bound_wf_id,
                "workflow_run_id": bound_run_id,
            }
            if decision.reflection is not None:
                esc_dict["reflection_history"] = [
                    {
                        "phase": "diagnostic",
                        "step": step_number,
                        **decision.reflection.model_dump(),
                    }
                ]
            return esc_dict

        reason = f"Unknown investigation decision received: {decision.decision_type}"
        telemetry.record_terminal_reason(incident_id, "ESCALATED", reason)
        return {
            "status": IncidentStatus.ESCALATED,
            "final_summary": reason,
        }

    return triage_node
