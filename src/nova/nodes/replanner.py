import logging
import re
from typing import Any, Dict
from nova.environments.mock_workflow_runner import MockWorkflowRunner
from nova.llm.client import BaseLLMClient
from nova.observability import incident_context, log_event, redact_secrets, telemetry
from nova.state import IncidentStatus, NovaGraphState

logger = logging.getLogger("nova.nodes.replanner")


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
            "triage_summary": state.get("triage_summary"),
            "root_cause_hypothesis": state.get("root_cause_hypothesis"),
        }

        try:
            replan_result = llm.replan(
                original_target=original_target,
                first_action_taken=first_action_taken,
                new_target=new_target,
                new_log_content=new_diagnostic_log,
                verification_evidence=last_verification.evidence,
                retry_count=new_retry_count,
                execution_records=[r.model_dump() if hasattr(r, "model_dump") else r for r in execution_history],
                verification_records=[v.model_dump() if hasattr(v, "model_dump") else v for v in verification_history],
                investigation_evidence=state.get("investigation_evidence", []),
                original_diagnosis=original_diagnosis,
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

        # Check for immediate escalation during replanning
        if replan_result.is_escalation:
            reason = f"Incident Escalated during replanning: {replan_result.escalation_reason}"
            telemetry.record_terminal_reason(state.get("incident_id"), "ESCALATED", reason)
            log_event(logger, logging.WARNING, "replanner_escalated", reason)
            return {
                "retry_count": new_retry_count,
                "status": IncidentStatus.ESCALATED,
                "final_summary": reason,
                "plan": [],
                "current_action_index": 0,
            }

        # Check for further investigation routing (Phase 3D + Outcome Reflection)
        can_reinvestigate = (
            replan_result.needs_further_investigation
            and state.get("investigation_steps", 0) < state.get("max_investigation_steps", 5)
        )
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

        # --- Step 3: Map validated output into graph state ---
        out_state: Dict[str, Any] = {
            "retry_count": new_retry_count,
            "status": next_status,
            "target_service": replan_result.new_target_service,
            "failure_type": replan_result.new_failure_type,
            "implicated_resource": replan_result.implicated_resource,
            "is_recoverable": replan_result.is_recoverable,
            "triage_summary": replan_result.updated_triage_summary,
            "root_cause_hypothesis": replan_result.updated_hypothesis,
            "plan": [],
            "current_action_index": 0,
        }
        if replan_result.outcome_reflection or replan_result.alternative_strategy_notes:
            out_state["reflection_history"] = [
                {
                    "phase": "outcome",
                    "retry_count": new_retry_count,
                    "needs_further_investigation": can_reinvestigate,
                    "outcome_reflection": replan_result.outcome_reflection,
                    "alternative_strategy_notes": replan_result.alternative_strategy_notes,
                }
            ]
            if replan_result.outcome_reflection:
                out_state["investigation_evidence"] = [
                    f"[Outcome Reflection cycle {new_retry_count}] {replan_result.outcome_reflection}"
                ]
        return out_state

    return replanner_node
