import inspect
import logging
import re
from typing import Any, Dict, List
from nova.environments.mock_workflow_runner import MockWorkflowRunner
from nova.llm.client import BaseLLMClient
from nova.observability import incident_context, log_event, redact_secrets, telemetry
from nova.state import IncidentStatus, NovaGraphState, build_prior_attempts

logger = logging.getLogger("nova.nodes.replanner")


def _call_replan(llm: BaseLLMClient, **kwargs: Any):
    """Calls llm.replan while preserving backward compatibility with custom test overrides."""
    sig = inspect.signature(llm.replan)
    has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if has_varkw:
        return llm.replan(**kwargs)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return llm.replan(**filtered)


def create_replanner_node(env: Any, llm: BaseLLMClient):
    """Factory returning a replanning node bound to the environment and LLM client."""

    def replanner_node(state: NovaGraphState) -> Dict[str, Any]:
        with incident_context(state.get("incident_id")):
            return _run_replanner(state)

    def _run_replanner(state: NovaGraphState) -> Dict[str, Any]:
        """
        Replanner Node — Hybrid (Deterministic extraction + LLM re-diagnosis):

        What was hardcoded before:
          - Template string hypothesis: "Downstream dependency failure: {new_target}... Log indicates: {last_line}"
          - Just glued the last log line into a string — zero LLM analysis

        What it does now:
        
        Step 1 (Deterministic):
          - Read `verification_history` to find the failed jobs.
          - Use regex to extract the new target job ID.
          - Read the new target's log from disk.
          These are I/O operations — Python is better than AI here.

        Step 2 (LLM):
          - Send the new log content, original context, and first action taken to the LLM.
          - LLM re-diagnoses the NEW failure (e.g. offline database that only became
            visible after the lockfile was cleared and the pipeline advanced).
          - Returns ReplannerResult with updated triage_summary and hypothesis.

        Step 3 (Deterministic):
          - Map validated LLM output back into graph state.
          - Increment retry_count — this is arithmetic, not AI.
        """
        current_retries = state.get("retry_count", 0)
        verification_history = state.get("verification_history", [])
        execution_history = state.get("execution_history", [])
        new_retry_count = current_retries + 1

        if not verification_history:
            return {
                "retry_count": new_retry_count,
                "status": IncidentStatus.FAILED,
                "final_summary": "Replanner invoked without preceding verification history.",
            }

        last_verification = verification_history[-1]
        failed_jobs = last_verification.details.get("failed_jobs", [])

        # --- Step 1: Deterministic data extraction ---

        original_target = state.get("target_service", "")

        # Extract which remediation tool was used in the last execution cycle.
        # If resume_workflow ran after a remediation tool and surfaced a downstream failure,
        # identify the preceding remediation tool so replanning context stays clear.
        first_action_taken = "unknown_action"
        if execution_history:
            last_exec = execution_history[-1]
            if (
                last_exec.tool_name == "resume_workflow"
                and not last_exec.success
                and len(execution_history) >= 2
                and execution_history[-2].success
            ):
                first_action_taken = execution_history[-2].tool_name
            else:
                first_action_taken = last_exec.tool_name

        # Extract the new failing job/task ID from verification evidence
        # Supports both uppercase JOB_03_LOAD and lowercase Kestra task IDs (e.g. green_copy_in_to_staging_table)
        new_target = original_target
        if failed_jobs:
            first_failed = failed_jobs[0]
            match = re.match(r"^([A-Za-z0-9_.-]+)", first_failed)
            if match:
                new_target = match.group(1)

        # Read diagnostic log for the new target
        bound_run_id = state.get("workflow_run_id")
        if bound_run_id and hasattr(env, "bind_execution"):
            new_diagnostic_log = env.read_job_log(new_target, execution_id=bound_run_id)
        else:
            new_diagnostic_log = env.read_job_log(new_target)
        if not new_diagnostic_log.strip():
            new_diagnostic_log = f"No log content found for {new_target}"

        # --- Step 2: LLM re-diagnosis with full execution & verification evidence ---
        original_diagnosis = {
            "failure_type": state.get("failure_type"),
            "implicated_resource": state.get("implicated_resource"),
            "is_recoverable": state.get("is_recoverable"),
            "triage_summary": state.get("triage_summary"),
            "root_cause_hypothesis": state.get("root_cause_hypothesis"),
        }
        prior_attempts = build_prior_attempts(state)
        exec_records_payload = [p for p in prior_attempts if p.get("kind") == "execution"]
        verif_records_payload = [p for p in prior_attempts if p.get("kind") == "verification"]

        try:
            replan_result = _call_replan(
                llm,
                original_target=original_target,
                first_action_taken=first_action_taken,
                new_target=new_target,
                new_log_content=new_diagnostic_log,
                verification_evidence=last_verification.evidence,
                retry_count=new_retry_count,
                execution_records=exec_records_payload,
                verification_records=verif_records_payload,
                investigation_evidence=state.get("investigation_evidence", []),
                original_diagnosis=original_diagnosis,
                attempted_plan=[
                    a.model_dump() if hasattr(a, "model_dump") else dict(a)
                    for a in (state.get("plan") or [])
                ],
            )
        except Exception as exc:
            reason = f"Replanning failed due to LLM error: {redact_secrets(str(exc))}"
            telemetry.record_terminal_reason(state.get("incident_id"), "FAILED", reason)
            log_event(logger, logging.ERROR, "replanner_llm_failed", reason)
            return {
                "retry_count": new_retry_count,
                "status": IncidentStatus.FAILED,
                "final_summary": reason,
                "plan": [],
                "current_action_index": 0,
            }

        reflection_entries: List[Dict[str, Any]] = []
        evidence_entries: List[str] = []
        revision_entries: List[Dict[str, Any]] = []

        has_budget_for_reinvestigate = (
            state.get("investigation_steps", 0) < state.get("max_investigation_steps", 5)
        )
        can_reinvestigate = bool(replan_result.needs_further_investigation and has_budget_for_reinvestigate)

        if (
            replan_result.outcome_reflection
            or replan_result.outcome_classification
            or replan_result.expected_vs_observed
        ):
            reflection_entries.append(
                {
                    "phase": "outcome",
                    "retry_count": new_retry_count,
                    "needs_further_investigation": can_reinvestigate,
                    "requested_further_investigation": replan_result.needs_further_investigation,
                    "outcome_classification": replan_result.outcome_classification,
                    "expected_vs_observed": replan_result.expected_vs_observed,
                    "outcome_reflection": replan_result.outcome_reflection,
                    "alternative_strategy_notes": replan_result.alternative_strategy_notes,
                    "diagnosis_revision_needed": replan_result.diagnosis_revision_needed,
                }
            )
            if replan_result.outcome_reflection:
                evidence_entries.append(
                    f"[Outcome Reflection cycle {new_retry_count}] {replan_result.outcome_reflection}"
                )
        elif replan_result.alternative_strategy_notes:
            evidence_entries.append(
                f"[Replanner Strategy cycle {new_retry_count}] {replan_result.alternative_strategy_notes}"
            )

        if (
            replan_result.diagnosis_revision_needed
            or replan_result.new_failure_type != original_diagnosis.get("failure_type")
            or replan_result.implicated_resource != original_diagnosis.get("implicated_resource")
            or replan_result.updated_hypothesis != original_diagnosis.get("root_cause_hypothesis")
        ):
            revision_entries.append(
                {
                    "phase": "replanner",
                    "retry_count": new_retry_count,
                    "previous": original_diagnosis,
                    "revised": {
                        "target_service": replan_result.new_target_service,
                        "failure_type": replan_result.new_failure_type,
                        "implicated_resource": replan_result.implicated_resource,
                        "is_recoverable": replan_result.is_recoverable,
                        "triage_summary": replan_result.updated_triage_summary,
                        "root_cause_hypothesis": replan_result.updated_hypothesis,
                    },
                    "classification": replan_result.outcome_classification,
                }
            )

        # Preserve revised diagnosis and outcome reflection even on direct escalation
        revised_state: Dict[str, Any] = {
            "retry_count": new_retry_count,
            "target_service": replan_result.new_target_service,
            "failure_type": replan_result.new_failure_type,
            "implicated_resource": replan_result.implicated_resource,
            "is_recoverable": replan_result.is_recoverable,
            "triage_summary": replan_result.updated_triage_summary,
            "root_cause_hypothesis": replan_result.updated_hypothesis,
            "plan": [],
            "current_action_index": 0,
        }
        if reflection_entries:
            revised_state["reflection_history"] = reflection_entries
        if evidence_entries:
            revised_state["investigation_evidence"] = evidence_entries
        if revision_entries:
            revised_state["hypothesis_revisions"] = revision_entries

        # Check for immediate escalation during replanning
        if replan_result.is_escalation:
            reason = f"Incident Escalated during replanning: {replan_result.escalation_reason}"
            telemetry.record_terminal_reason(state.get("incident_id"), "ESCALATED", reason)
            log_event(logger, logging.WARNING, "replanner_escalated", reason)
            revised_state["status"] = IncidentStatus.ESCALATED
            revised_state["final_summary"] = reason
            return revised_state

        # If further investigation is necessary but budget is exhausted, escalate cleanly
        # instead of blindly proceeding with an unsupported remediation plan.
        if replan_result.needs_further_investigation and not has_budget_for_reinvestigate:
            max_steps = state.get("max_investigation_steps", 5)
            reason = (
                f"Incident Escalated: Replanner determined further diagnostic investigation is necessary, "
                f"but investigation budget ({max_steps} steps) is exhausted."
            )
            telemetry.record_terminal_reason(state.get("incident_id"), "ESCALATED", reason)
            log_event(logger, logging.WARNING, "replanner_investigation_budget_exhausted", reason, max_steps=max_steps)
            revised_state["status"] = IncidentStatus.ESCALATED
            revised_state["final_summary"] = reason
            return revised_state

        next_status = IncidentStatus.TRIAGING if can_reinvestigate else IncidentStatus.REPLANNING

        log_event(
            logger,
            logging.INFO,
            "replan_formulated",
            f"Replanning cycle {new_retry_count}: target={replan_result.new_target_service}, "
            f"failure={replan_result.new_failure_type}, reinvestigate={can_reinvestigate}",
            retry_count=new_retry_count,
            new_target=replan_result.new_target_service,
            new_failure_type=replan_result.new_failure_type,
            reinvestigate=can_reinvestigate,
        )

        revised_state["status"] = next_status
        return revised_state

    return replanner_node
