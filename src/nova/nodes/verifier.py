import logging
import time
from typing import Any, Dict, Optional
from nova.environments.mock_workflow_runner import JobStatus
from nova.observability import incident_context, log_event, telemetry
from nova.state import IncidentStatus, NovaGraphState, RiskLevel, VerificationRecord
from nova.tools.runner import ToolRunner

logger = logging.getLogger("nova.nodes.verifier")

_MUTATING_FALLBACK_TOOLS = frozenset({"clear_lockfile", "restart_service", "resume_workflow"})


def create_verifier_node(env: Any, tool_runner: Optional[ToolRunner] = None):
    """Factory returning an independent verification node bound to the environment.

    M11/M12 correctness: the verifier is strictly READ-ONLY.
    - Reads env.get_state(...) — never mutates workflow state or calls env.run_workflow().
    - Enforces at least one executed non-read-only remediation or resumption action.
    - Validates workflow_run_id binding and fresh task attempts when baseline_attempts is set.
    - Runs independent read-only output verification (`env.verify_output(...)`) when supported.
    """

    def _is_mutating_tool(tool_name: str) -> bool:
        if tool_runner is not None:
            try:
                return tool_runner.get_tool(tool_name).risk_level != RiskLevel.READ_ONLY
            except Exception:
                return False
        return tool_name in _MUTATING_FALLBACK_TOOLS

    def verifier_node(state: NovaGraphState) -> Dict[str, Any]:
        with incident_context(state.get("incident_id")):
            return _run_verifier(state)

    def _run_verifier(state: NovaGraphState) -> Dict[str, Any]:
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

        current_retries = state.get("retry_count", 0)
        max_retries = state.get("max_retries", 3)
        is_exhausted = current_retries >= max_retries
        bound_run_id = state.get("workflow_run_id")

        # Read-only state observation (used for both failed-exec telemetry and pass checks)
        try:
            if bound_run_id and hasattr(env, "bind_execution"):
                workflow_state = env.get_state(execution_id=bound_run_id)
            else:
                workflow_state = env.get_state()
        except Exception as e:
            rec = VerificationRecord(
                action_id=action.action_id,
                verified_at=verified_at,
                passed=False,
                evidence=f"Verification exception while reading workflow state: {str(e)}",
            )
            log_event(logger, logging.ERROR, "verification_exception", str(e), action_id=action.action_id)
            return {
                "verification_history": [rec],
                "status": IncidentStatus.ESCALATED if is_exhausted else IncidentStatus.FAILED,
                "final_summary": f"Verification error: {str(e)}",
            }

        failed_jobs = (
            [
                f"{j.id} ({j.error_message})"
                for j in workflow_state.jobs.values()
                if j.status == JobStatus.FAILED
            ]
            if workflow_state
            else []
        )

        # Guard 1: Tool execution itself (and its executor-level postcondition) must have succeeded.
        if not last_exec.success or state.get("status") == IncidentStatus.FAILED:
            evidence_msg = f"Tool execution failed with exit code {last_exec.exit_code}: {last_exec.stderr}"
            if failed_jobs:
                evidence_msg += f" | Current failed jobs: {failed_jobs}"
            rec = VerificationRecord(
                action_id=action.action_id,
                verified_at=verified_at,
                passed=False,
                evidence=evidence_msg,
                details={
                    "failed_jobs": failed_jobs,
                    "workflow_id": workflow_state.workflow_id if workflow_state else None,
                },
            )
            summary = (
                f"Incident Escalated: Maximum retry limit ({max_retries}) reached — "
                f"action {action.action_id} execution failed. Failed jobs: {failed_jobs}"
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

        # Guard 2: Read-only actions alone cannot declare an incident resolved.
        has_mutating_exec = any(_is_mutating_tool(rec.tool_name) for rec in exec_history if rec.success)
        if not has_mutating_exec:
            rec = VerificationRecord(
                action_id=action.action_id,
                verified_at=verified_at,
                passed=False,
                evidence=(
                    "Verification failed: executed plan contained only read-only actions "
                    "and did not perform remediation or workflow resumption."
                ),
                details={"failed_jobs": failed_jobs},
            )
            summary = (
                "Incident Escalated: Read-only inspection cannot resolve an incident without remediation or workflow resumption."
                if is_exhausted
                else "Verification failed: no mutating remediation or resume_workflow action was executed."
            )
            if is_exhausted:
                telemetry.record_terminal_reason(incident_id, "ESCALATED", summary)
            log_event(logger, logging.WARNING, "verification_read_only_plan_rejected", summary, action_id=action.action_id)
            return {
                "verification_history": [rec],
                "status": IncidentStatus.ESCALATED if is_exhausted else IncidentStatus.FAILED,
                "final_summary": summary,
            }

        # Guard 3: Workflow state presence & run binding validation
        if not workflow_state:
            rec = VerificationRecord(
                action_id=action.action_id,
                verified_at=verified_at,
                passed=False,
                evidence="Verification failed: no workflow state found after action execution.",
            )
            summary = "Verification failed: workflow state is absent. Run resume_workflow before verifying."
            log_event(logger, logging.WARNING, "verification_no_state", summary, action_id=action.action_id)
            return {
                "verification_history": [rec],
                "status": IncidentStatus.ESCALATED if is_exhausted else IncidentStatus.FAILED,
                "final_summary": summary,
            }

        bound_wf_id = state.get("workflow_id")
        if bound_run_id and (
            workflow_state.run_id != bound_run_id
            or (bound_wf_id and workflow_state.workflow_id != bound_wf_id)
        ):
            reason = (
                f"Incident Escalated: Verification run binding mismatch. Expected run '{bound_run_id}' "
                f"({bound_wf_id}), observed '{workflow_state.run_id}' ({workflow_state.workflow_id})."
            )
            rec = VerificationRecord(
                action_id=action.action_id,
                verified_at=verified_at,
                passed=False,
                evidence=reason,
                details={
                    "expected_run_id": bound_run_id,
                    "observed_run_id": workflow_state.run_id,
                },
            )
            telemetry.record_terminal_reason(incident_id, "ESCALATED", reason)
            log_event(logger, logging.WARNING, "verification_run_binding_mismatch", reason)
            return {
                "verification_history": [rec],
                "status": IncidentStatus.ESCALATED,
                "final_summary": reason,
            }

        # Guard 4: Fresh attempt binding (do not credit stale prior successes when tracking task attempts)
        baseline_attempts = state.get("baseline_attempts")
        observed_attempts = getattr(workflow_state, "total_attempts", None)
        if baseline_attempts is not None and observed_attempts is not None:
            if observed_attempts <= baseline_attempts:
                rec = VerificationRecord(
                    action_id=action.action_id,
                    verified_at=verified_at,
                    passed=False,
                    evidence=(
                        f"Verification failed: no fresh workflow task attempt observed "
                        f"(baseline_attempts={baseline_attempts}, observed_attempts={observed_attempts})."
                    ),
                    details={"failed_jobs": failed_jobs, "observed_attempts": observed_attempts},
                )
                summary = (
                    f"Incident Escalated: No fresh workflow execution attempt occurred within retry limit ({max_retries})."
                    if is_exhausted
                    else "Verification failed: workflow execution did not produce fresh task attempts."
                )
                if is_exhausted:
                    telemetry.record_terminal_reason(incident_id, "ESCALATED", summary)
                return {
                    "verification_history": [rec],
                    "status": IncidentStatus.ESCALATED if is_exhausted else IncidentStatus.FAILED,
                    "final_summary": summary,
                }

        # Guard 5: All jobs/tasks in the expected workflow run must be SUCCESS
        all_succeeded = bool(workflow_state.jobs) and all(
            job.status == JobStatus.SUCCESS for job in workflow_state.jobs.values()
        )
        non_success_jobs = [
            f"{j.id} (status={j.status.value}{f', {j.error_message}' if j.error_message else ''})"
            for j in workflow_state.jobs.values()
            if j.status != JobStatus.SUCCESS
        ]

        if not all_succeeded:
            rec = VerificationRecord(
                action_id=action.action_id,
                verified_at=verified_at,
                passed=False,
                evidence=f"Action executed, but workflow state is not complete: {non_success_jobs}",
                details={"failed_jobs": failed_jobs or non_success_jobs},
            )
            summary = (
                f"Incident Escalated: Maximum retry limit ({max_retries}) reached without resolving the issue. "
                f"Incomplete/failed jobs: {non_success_jobs}"
                if is_exhausted
                else f"Verification failed: Workflow jobs not all SUCCESS after remediation: {non_success_jobs}"
            )
            if is_exhausted:
                telemetry.record_terminal_reason(incident_id, "ESCALATED", summary)
            log_event(
                logger,
                logging.WARNING,
                "verification_failed_downstream",
                summary,
                failed_jobs=non_success_jobs,
            )
            return {
                "verification_history": [rec],
                "status": IncidentStatus.ESCALATED if is_exhausted else IncidentStatus.FAILED,
                "final_summary": summary,
            }

        # Guard 6: Independent read-only database output verification (when environment supports it)
        output_details: Dict[str, Any] = {}
        output_evidence = ""
        if hasattr(env, "verify_output"):
            expected_cfg = dict(state.get("expected_output") or {})
            if hasattr(workflow_state, "inputs") and workflow_state.inputs:
                for k in ("taxi", "year", "month"):
                    if k in workflow_state.inputs and k not in expected_cfg:
                        expected_cfg[k] = workflow_state.inputs[k]
            pg_ok, output_details, output_evidence = env.verify_output(expected_cfg)
            if not pg_ok:
                rec = VerificationRecord(
                    action_id=action.action_id,
                    verified_at=verified_at,
                    passed=False,
                    evidence=f"Workflow tasks succeeded, but independent output verification failed: {output_evidence}",
                    details={"output_verification": output_details, "failed_jobs": failed_jobs},
                )
                summary = (
                    f"Incident Escalated: Output verification failed after max retries ({max_retries}): {output_evidence}"
                    if is_exhausted
                    else f"Verification failed: {output_evidence}"
                )
                if is_exhausted:
                    telemetry.record_terminal_reason(incident_id, "ESCALATED", summary)
                log_event(logger, logging.WARNING, "verification_output_failed", summary, details=output_details)
                return {
                    "verification_history": [rec],
                    "status": IncidentStatus.ESCALATED if is_exhausted else IncidentStatus.FAILED,
                    "final_summary": summary,
                }

        task_count = len(workflow_state.jobs)
        evidence = (
            f"Read-only state verification passed: workflow state confirms all {task_count} "
            f"tasks/jobs completed with status SUCCESS."
        )
        if output_evidence:
            evidence = f"{evidence} {output_evidence}"

        rec = VerificationRecord(
            action_id=action.action_id,
            verified_at=verified_at,
            passed=True,
            evidence=evidence,
            details={
                "all_jobs_success": True,
                "workflow_id": workflow_state.workflow_id,
                "run_id": workflow_state.run_id,
                "output_verification": output_details,
            },
        )
        summary = (
            f"Incident Successfully Resolved: Action {action.action_id} executed, "
            "and workflow state independently confirms 100% job completion."
        )
        if output_evidence:
            summary = f"{summary} {output_evidence}"
        log_event(logger, logging.INFO, "verification_passed", summary, action_id=action.action_id)
        return {
            "verification_history": [rec],
            "status": IncidentStatus.RESOLVED,
            "final_summary": summary,
        }

    return verifier_node
