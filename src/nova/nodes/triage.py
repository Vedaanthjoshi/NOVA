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
)
from nova.tools.runner import ToolRunner, create_default_tool_runner

logger = logging.getLogger("nova.nodes.triage")


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

        # M11/M12: Capture active workflow_id / workflow_run_id & baseline attempt count if unbound
        bound_wf_id = state.get("workflow_id")
        bound_run_id = state.get("workflow_run_id")
        baseline_attempts = state.get("baseline_attempts")
        wf_state = None
        if not bound_run_id or baseline_attempts is None:
            try:
                wf_state = env.get_state(execution_id=bound_run_id) if bound_run_id and hasattr(env, "bind_execution") else env.get_state()
            except Exception:
                wf_state = None
            if wf_state is not None:
                bound_wf_id = bound_wf_id or getattr(wf_state, "workflow_id", None)
                bound_run_id = bound_run_id or getattr(wf_state, "run_id", None)
                if baseline_attempts is None and hasattr(wf_state, "total_attempts"):
                    baseline_attempts = getattr(wf_state, "total_attempts")

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

        # Call LLM to investigate
        try:
            decision: InvestigationDecision = llm.investigate(
                target_service=target_service,
                raw_symptoms=raw_symptoms,
                investigation_evidence=investigation_evidence,
                diagnostic_history=[d.model_dump() if hasattr(d, "model_dump") else d for d in diagnostic_history],
                available_tools=diagnostic_tools,
                step_number=step_number,
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

        # Case 1: DIAGNOSTIC_ACTION
        if decision.decision_type == InvestigationActionType.DIAGNOSTIC_ACTION:
            diag_call = decision.diagnostic_tool
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

            # Unproductive Loop Detection
            for prev in diagnostic_history:
                prev_tool = prev.tool_name if hasattr(prev, "tool_name") else prev.get("tool_name")
                prev_params = prev.parameters if hasattr(prev, "parameters") else prev.get("parameters", {})
                if prev_tool == tool_name and prev_params == diag_call.parameters:
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
            )
            evidence_line = f"Step {step_number} [{tool_name}]: {tool_output}"
            out_dict: Dict[str, Any] = {
                "diagnostic_history": [rec],
                "investigation_evidence": [evidence_line],
                "investigation_steps": step_number,
                "workflow_id": bound_wf_id,
                "workflow_run_id": bound_run_id,
                "status": IncidentStatus.TRIAGING,
            }
            if baseline_attempts is not None:
                out_dict["baseline_attempts"] = baseline_attempts
            if decision.reflection is not None:
                out_dict["reflection_history"] = [
                    {
                        "phase": "diagnostic",
                        "step": step_number,
                        **decision.reflection.model_dump(),
                    }
                ]
            return out_dict

        # Case 2: DIAGNOSIS_READY
        elif decision.decision_type == InvestigationActionType.DIAGNOSIS_READY:
            diag = decision.diagnosis
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
            if decision.reflection is not None:
                out_ready["reflection_history"] = [
                    {
                        "phase": "diagnostic",
                        "step": step_number,
                        **decision.reflection.model_dump(),
                    }
                ]
            return out_ready

        # Case 3: ESCALATE
        elif decision.decision_type == InvestigationActionType.ESCALATE:
            reason = f"Incident Escalated during investigation: {decision.escalation_reason}"
            telemetry.record_terminal_reason(incident_id, "ESCALATED", reason)
            log_event(logger, logging.WARNING, "investigation_escalated", reason)
            return {
                "status": IncidentStatus.ESCALATED,
                "final_summary": reason,
            }

        reason = f"Unknown investigation decision received: {decision.decision_type}"
        telemetry.record_terminal_reason(incident_id, "ESCALATED", reason)
        return {
            "status": IncidentStatus.ESCALATED,
            "final_summary": reason,
        }

    return triage_node
