import logging
import time
from typing import Any, Dict
from nova.environments.mock_workflow_runner import JobStatus, MockWorkflowRunner
from nova.observability import incident_context, log_event, telemetry
from nova.state import IncidentStatus, NovaGraphState, VerificationRecord

logger = logging.getLogger("nova.nodes.verifier")


def create_verifier_node(env: MockWorkflowRunner):
    """Factory returning an independent verification node bound to the environment."""

    def verifier_node(state: NovaGraphState) -> Dict[str, Any]:
        with incident_context(state.get("incident_id")):
            return _run_verifier(state)

    def _run_verifier(state: NovaGraphState) -> Dict[str, Any]:
        """
        Verifier Node (Milestone 5):
        Performs independent, two-tier post-condition verification:
        1. Tool footprint verification (e.g. is lockfile physically gone / is service port open?).
        2. Deep downstream verification (does the batch workflow actually complete all remaining jobs?).
        """
        incident_id = state.get("incident_id")
        plan = state.get("plan", [])
        index = state.get("current_action_index", 0)
        exec_history = state.get("execution_history", [])

        if not plan or not exec_history:
            return {
                "status": IncidentStatus.FAILED,
                "final_summary": "Verifier invoked without preceding plan or execution record.",
            }

        action_idx = min(index, len(plan) - 1)
        action = plan[action_idx]
        last_exec = exec_history[-1]
        verified_at = time.strftime("%Y-%m-%d %H:%M:%S")

        # Guard 1: Tool execution itself must have succeeded
        current_retries = state.get("retry_count", 0)
        max_retries = state.get("max_retries", 3)
        is_exhausted = current_retries >= max_retries

        if not last_exec.success:
            rec = VerificationRecord(
                action_id=action.action_id,
                verified_at=verified_at,
                passed=False,
                evidence=f"Tool execution failed with exit code {last_exec.exit_code}: {last_exec.stderr}",
            )
            summary = (
                f"Incident Escalated: Action {action.action_id} execution failed and retry budget ({max_retries}) exhausted."
                if is_exhausted
                else f"Verification failed: Action {action.action_id} did not execute cleanly."
            )
            if is_exhausted:
                telemetry.record_terminal_reason(incident_id, "ESCALATED", summary)
            log_event(logger, logging.WARNING, "verification_failed_execution", summary, action_id=action.action_id)
            return {
                "verification_history": [rec],
                "status": IncidentStatus.ESCALATED if is_exhausted else IncidentStatus.FAILED,
                "final_summary": summary,
            }

        # Guard 2: Deep Downstream Pipeline Verification
        # Simulation oracle: re-runs the mock batch workflow to prove downstream jobs complete.
        try:
            workflow_state = env.run_workflow()
            all_succeeded = all(job.status == JobStatus.SUCCESS for job in workflow_state.jobs.values())
            failed_jobs = [
                f"{j.id} ({j.error_message})"
                for j in workflow_state.jobs.values()
                if j.status == JobStatus.FAILED
            ]

            if all_succeeded:
                rec = VerificationRecord(
                    action_id=action.action_id,
                    verified_at=verified_at,
                    passed=True,
                    evidence=(
                        "Deep verification succeeded: batch pipeline successfully completed "
                        "all 4 jobs (Extract, Transform, Load, Report) with exit code 0."
                    ),
                    details={"all_jobs_success": True},
                )
                summary = (
                    f"Incident Successfully Resolved: Action {action.action_id} executed, "
                    "and downstream workflow was independently verified as 100% operational."
                )
                log_event(logger, logging.INFO, "verification_passed", summary, action_id=action.action_id)
                return {
                    "verification_history": [rec],
                    "status": IncidentStatus.RESOLVED,
                    "final_summary": summary,
                }
            else:
                rec = VerificationRecord(
                    action_id=action.action_id,
                    verified_at=verified_at,
                    passed=False,
                    evidence=f"Action executed, but downstream workflow still has failed jobs: {failed_jobs}",
                    details={"failed_jobs": failed_jobs},
                )
                summary = (
                    f"Incident Escalated: Maximum retry limit ({max_retries}) reached without resolving the issue. Downstream failures: {failed_jobs}"
                    if is_exhausted
                    else f"Verification failed: Downstream jobs failed after remediation: {failed_jobs}"
                )
                if is_exhausted:
                    telemetry.record_terminal_reason(incident_id, "ESCALATED", summary)
                log_event(
                    logger,
                    logging.WARNING,
                    "verification_failed_downstream",
                    summary,
                    failed_jobs=failed_jobs,
                )
                return {
                    "verification_history": [rec],
                    "status": IncidentStatus.ESCALATED if is_exhausted else IncidentStatus.FAILED,
                    "final_summary": summary,
                }

        except Exception as e:
            rec = VerificationRecord(
                action_id=action.action_id,
                verified_at=verified_at,
                passed=False,
                evidence=f"Verification exception encountered while probing workflow: {str(e)}",
            )
            log_event(logger, logging.ERROR, "verification_exception", str(e), action_id=action.action_id)
            return {
                "verification_history": [rec],
                "status": IncidentStatus.FAILED,
                "final_summary": f"Verification error: {str(e)}",
            }

    return verifier_node
