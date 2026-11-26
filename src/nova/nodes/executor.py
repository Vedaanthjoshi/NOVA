import hashlib
import json
import logging
import time
from typing import Any, Dict, Optional
from langgraph.types import interrupt
from nova.environments.mock_workflow_runner import (
    JobStatus,
    resolve_contained_path,
)
from nova.observability import incident_context, log_event, telemetry
from nova.state import (
    ALLOWED_APPROVER_ROLES,
    ApprovalDecision,
    IncidentStatus,
    NovaGraphState,
    RiskLevel,
)
from nova.tools.runner import ToolRunner

logger = logging.getLogger("nova.nodes.executor")


def _resolve_env(runner: ToolRunner, env: Optional[Any]) -> Optional[Any]:
    """Extracts environment instance from injected argument or registered tools."""
    if env is not None:
        return env
    for tool in runner._registry.values():
        tool_env = getattr(tool, "env", None)
        if tool_env is not None and hasattr(tool_env, "get_state"):
            return tool_env
    return None


def _get_env_state(env: Any, run_id: Optional[str]) -> Any:
    if env is None:
        return None
    if run_id and hasattr(env, "bind_execution"):
        return env.get_state(execution_id=run_id)
    return env.get_state()


def _check_run_binding(state: NovaGraphState, env: Optional[Any]) -> Optional[str]:
    """
    Validates that the current workflow state still matches the workflow_id
    and workflow_run_id captured when this incident started.
    Returns None if valid or unbound; returns an error message if mismatched.
    """
    bound_run_id = state.get("workflow_run_id")
    bound_wf_id = state.get("workflow_id")
    if not bound_run_id or env is None:
        return None
    current_wf = _get_env_state(env, bound_run_id)
    if current_wf is None:
        return (
            f"Workflow run binding violation: expected run '{bound_run_id}' "
            f"({bound_wf_id}), but active workflow state is absent."
        )
    if current_wf.run_id != bound_run_id or (bound_wf_id and current_wf.workflow_id != bound_wf_id):
        return (
            f"Workflow run binding violation: incident is bound to run '{bound_run_id}' "
            f"({bound_wf_id}), but environment currently holds run '{current_wf.run_id}' "
            f"({current_wf.workflow_id})."
        )
    return None


def _check_postcondition(
    tool_name: str,
    parameters: Dict[str, Any],
    env: Optional[Any],
    bound_run_id: Optional[str] = None,
) -> Optional[str]:
    """
    Deterministic post-condition check for mutating actions.

    Returns None if the postcondition is satisfied (action had its intended effect).
    Returns an error string if the postcondition failed (action did not produce expected state).

    READ_ONLY tools have no postcondition — they observe state, they do not change it.
    """
    if env is None:
        return None

    if tool_name == "clear_lockfile" and hasattr(env, "locks_dir"):
        lock_name = parameters.get("lock_name", "")
        if not lock_name:
            return None
        state = env.get_state()
        if state and lock_name in getattr(state, "active_locks", []):
            return f"Postcondition failed: '{lock_name}' is still listed in active_locks after clear_lockfile."
        safe_lock = resolve_contained_path(env.locks_dir, lock_name)
        if safe_lock is None:
            return f"Postcondition check: unsafe lock_name '{lock_name}' rejected."
        if safe_lock.exists():
            return f"Postcondition failed: lockfile '{lock_name}' still present on disk after removal."
        return None

    if tool_name == "restart_service":
        service_name = parameters.get("service_name", "")
        if not service_name:
            return None
        try:
            status = env.check_service_status(service_name)
            if not status.get("healthy", False):
                return (
                    f"Postcondition failed: '{service_name}' is still unhealthy "
                    f"(status={status.get('status', 'UNKNOWN')}) after restart."
                )
        except Exception as exc:
            return f"Postcondition check error for restart_service: {exc}"
        return None

    if tool_name == "resume_workflow":
        target_eid = parameters.get("execution_id") or bound_run_id
        state = _get_env_state(env, target_eid)
        if state is None:
            return "Postcondition failed: workflow state absent after resume_workflow."
        failed = [j.id for j in state.jobs.values() if j.status != JobStatus.SUCCESS]
        if failed:
            return f"Postcondition failed: tasks/jobs not in SUCCESS state after resume_workflow: {failed}"
        return None

    return None


def create_executor_node(
    runner: ToolRunner,
    env: Optional[Any] = None,
    require_approval: bool = True,
):
    """Factory returning an executor node bound to a ToolRunner, environment, and HITL safety policy."""
    resolved_env = _resolve_env(runner, env)

    def executor_node(state: NovaGraphState) -> Dict[str, Any]:
        with incident_context(state.get("incident_id")):
            return _run_executor(state)

    def _run_executor(state: NovaGraphState) -> Dict[str, Any]:
        """
        Executor Node (Hardened Safety & Authorization):
        1. Retrieves the next planned action.
        2. Ground-Truth Risk Lookup from registry — LLM risk rating is ignored.
        3. Validates workflow run binding (M11) and tool parameter schema before HITL interrupt.
        4. Policy Gate: DESTRUCTIVE actions require LangGraph interrupt() with structured approval.
           Permissive raw-bool and incomplete dict approval shortcuts are REJECTED.
        5. Re-validates workflow run binding after HITL resumption.
        6. Executes the authorized tool via ToolRunner.
        7. Deterministic postcondition check for mutating actions (M11).
        8. Advances current_action_index on clean execution + passing postcondition.
        """
        incident_id = state.get("incident_id", "INC_DEFAULT")
        plan = state.get("plan", [])
        index = state.get("current_action_index", 0)

        if not plan:
            reason = "Executor invoked with no active planned action."
            telemetry.record_terminal_reason(incident_id, "FAILED", reason)
            return {
                "status": IncidentStatus.FAILED,
                "final_summary": reason,
            }

        if index >= len(plan):
            return {
                "status": IncidentStatus.VERIFYING,
            }

        action = plan[index]

        # Compute deterministic action_hash fallback if action came directly from a unit test
        action_hash = action.action_hash
        if not action_hash:
            canonical_params = json.dumps(action.parameters, sort_keys=True)
            raw_sig = f"{incident_id}:{action.action_id}:{action.tool_name}:{canonical_params}"
            action_hash = hashlib.sha256(raw_sig.encode("utf-8")).hexdigest()[:16]

        # Security Principle: Never trust LLM risk rating. Look up ground-truth from registry.
        try:
            tool = runner.get_tool(action.tool_name)
            effective_risk = tool.risk_level
            if not isinstance(effective_risk, RiskLevel):
                raise ValueError(f"Invalid ground-truth risk level '{effective_risk}'")
        except Exception as exc:
            reason = (
                f"Incident Escalated: Security Alert - Ground-truth risk level for tool '{action.tool_name}' "
                f"({action.action_id}) cannot be determined: {str(exc)}. Fail-closed safety policy enforced."
            )
            telemetry.record_terminal_reason(incident_id, "ESCALATED", reason)
            log_event(logger, logging.WARNING, "executor_unknown_tool", reason, tool_name=action.tool_name)
            return {
                "status": IncidentStatus.ESCALATED,
                "final_summary": reason,
            }

        # Check workflow run binding before interrupting or executing mutating operations
        if effective_risk != RiskLevel.READ_ONLY:
            binding_err = _check_run_binding(state, resolved_env)
            if binding_err:
                reason = f"Incident Escalated: {binding_err}"
                telemetry.record_terminal_reason(incident_id, "ESCALATED", reason)
                log_event(logger, logging.WARNING, "executor_run_binding_mismatch", reason)
                return {
                    "status": IncidentStatus.ESCALATED,
                    "final_summary": reason,
                }

        decision: Optional[ApprovalDecision] = None

        # Safety Gate: Human-in-the-Loop Checkpoint for DESTRUCTIVE actions
        if require_approval and effective_risk == RiskLevel.DESTRUCTIVE:
            # Validate parameters against tool schema BEFORE presenting an approval interrupt
            try:
                tool.args_schema.model_validate(action.parameters)
            except Exception as param_exc:
                reason = (
                    f"Incident Escalated: Destructive action '{action.tool_name}' ({action.action_id}) "
                    f"failed parameter validation before approval: {param_exc}"
                )
                telemetry.record_terminal_reason(incident_id, "ESCALATED", reason)
                log_event(logger, logging.WARNING, "executor_invalid_destructive_params", reason)
                return {
                    "status": IncidentStatus.ESCALATED,
                    "final_summary": reason,
                }

            # Deterministic approval_request_id so LangGraph node replay on resume produces the exact same ID
            approval_request_id = f"APPR_{incident_id}_{action.action_id}_{action_hash[:8]}"
            approval_payload = {
                "approval_request_id": approval_request_id,
                "action_id": action.action_id,
                "action_hash": action_hash,
                "incident_id": state.get("incident_id"),
                "workflow_id": state.get("workflow_id"),
                "workflow_run_id": state.get("workflow_run_id"),
                "tool_name": action.tool_name,
                "parameters": action.parameters,
                "risk_level": effective_risk.value,
                "rationale": action.rationale,
                "expected_outcome": action.expected_outcome,
                "target_service": state.get("target_service"),
                "requested_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "reason": (
                    f"Safety Policy: Tool '{action.tool_name}' has ground-truth risk level '{effective_risk.value}'. "
                    "Human authorization is strictly required before execution."
                ),
            }
            log_event(
                logger,
                logging.INFO,
                "hitl_interrupt_requested",
                f"Awaiting human authorization for '{action.tool_name}' ({action.action_id})",
                action_id=action.action_id,
                action_hash=action_hash,
                approval_request_id=approval_request_id,
                tool_name=action.tool_name,
            )

            # Pauses graph execution and yields control to operator
            response = interrupt(approval_payload)

            is_approved = False
            rejection_reason = "No approval decision provided."

            # Case 1: Pydantic ApprovalDecision model
            if isinstance(response, ApprovalDecision):
                if response.approved is not True:
                    is_approved = False
                    rejection_reason = (
                        f"Action was REJECTED by human operator (approved={response.approved}). "
                        f"Operator notes: {response.notes}"
                    )
                elif response.action_id != action.action_id:
                    is_approved = False
                    rejection_reason = (
                        f"Security Alert: Binding mismatch on action_id. Expected '{action.action_id}', "
                        f"got '{response.action_id}'."
                    )
                elif state.get("incident_id") and response.incident_id != state.get("incident_id"):
                    is_approved = False
                    rejection_reason = (
                        f"Security Alert: Binding mismatch on incident_id. Expected '{state.get('incident_id')}', "
                        f"got '{response.incident_id}'."
                    )
                elif response.action_hash and response.action_hash != action_hash:
                    is_approved = False
                    rejection_reason = (
                        f"Security Alert: Binding mismatch on action_hash. Expected '{action_hash}', "
                        f"got '{response.action_hash}'."
                    )
                elif response.approver_role not in ALLOWED_APPROVER_ROLES:
                    is_approved = False
                    rejection_reason = (
                        f"Security Alert: Unauthorized approver role '{response.approver_role}'. "
                        f"Allowed roles: {sorted(ALLOWED_APPROVER_ROLES)}."
                    )
                else:
                    is_approved = True
                    decision = response

            # Case 2: Dictionary payload — requires explicit approver_id and approver_role (no defaults injected)
            elif isinstance(response, dict):
                raw_approved = response.get("approved")
                if type(raw_approved) is not bool or raw_approved is not True:
                    is_approved = False
                    notes = response.get("notes", "No operator notes provided")
                    rejection_reason = f"Action was REJECTED by human operator. Operator notes: {notes}"
                else:
                    res_action_id = response.get("action_id", action.action_id)
                    res_incident_id = response.get("incident_id", state.get("incident_id"))
                    res_action_hash = response.get("action_hash")

                    if res_action_id != action.action_id:
                        is_approved = False
                        rejection_reason = (
                            f"Security Alert: Binding mismatch on action_id. Expected '{action.action_id}', "
                            f"got '{res_action_id}'."
                        )
                    elif state.get("incident_id") and res_incident_id != state.get("incident_id"):
                        is_approved = False
                        rejection_reason = (
                            f"Security Alert: Binding mismatch on incident_id. Expected '{state.get('incident_id')}', "
                            f"got '{res_incident_id}'."
                        )
                    elif res_action_hash and res_action_hash != action_hash:
                        is_approved = False
                        rejection_reason = (
                            f"Security Alert: Binding mismatch on action_hash. Expected '{action_hash}', "
                            f"got '{res_action_hash}'."
                        )
                    else:
                        approver_id = response.get("approver_id")
                        approver_role = response.get("approver_role")

                        if not isinstance(approver_id, str) or not approver_id.strip():
                            is_approved = False
                            rejection_reason = "Security Alert: approver_id is required in approval payload."
                        elif not isinstance(approver_role, str) or not approver_role.strip():
                            is_approved = False
                            rejection_reason = "Security Alert: approver_role is required in approval payload."
                        elif approver_role not in ALLOWED_APPROVER_ROLES:
                            is_approved = False
                            rejection_reason = (
                                f"Security Alert: Unauthorized approver role '{approver_role}'. "
                                f"Allowed roles: {sorted(ALLOWED_APPROVER_ROLES)}."
                            )
                        else:
                            try:
                                decision = ApprovalDecision(
                                    approved=True,
                                    approver_id=approver_id.strip(),
                                    approver_role=approver_role,
                                    decision_timestamp=(
                                        response.get("decision_timestamp")
                                        or time.strftime("%Y-%m-%d %H:%M:%S")
                                    ),
                                    incident_id=res_incident_id or state.get("incident_id", "INC_DEFAULT"),
                                    action_id=action.action_id,
                                    approval_request_id=response.get("approval_request_id") or approval_request_id,
                                    action_hash=action_hash,
                                    notes=response.get("notes"),
                                )
                                is_approved = True
                            except Exception as exc:
                                is_approved = False
                                rejection_reason = f"Security Alert: Malformed approval payload: {str(exc)}"

            # Case 3: Raw boolean — REJECTED in operational mode (no identity, no role, no binding)
            elif type(response) is bool:
                is_approved = False
                rejection_reason = (
                    "Security Alert: Raw boolean approval rejected in operational mode. "
                    "Submit an ApprovalDecision or dict with approver_id and approver_role."
                )

            # Case 4: Any other malformed input
            else:
                is_approved = False
                rejection_reason = (
                    f"Malformed approval payload received ({type(response).__name__}: {response}). "
                    "Strictly rejected."
                )

            if not is_approved:
                reason = (
                    f"Incident Escalated: Destructive action '{action.tool_name}' ({action.action_id}) "
                    f"was REJECTED by human operator. Reason: {rejection_reason}. Execution aborted safely."
                )
                telemetry.record_terminal_reason(incident_id, "ESCALATED", reason)
                log_event(
                    logger,
                    logging.WARNING,
                    "hitl_rejected_or_invalid",
                    reason,
                    action_id=action.action_id,
                    tool_name=action.tool_name,
                )
                return {
                    "status": IncidentStatus.ESCALATED,
                    "final_summary": reason,
                }

            # Re-verify workflow run binding after HITL resume in case environment changed while paused
            post_resume_binding_err = _check_run_binding(state, resolved_env)
            if post_resume_binding_err:
                reason = f"Incident Escalated: {post_resume_binding_err} Execution aborted after approval."
                telemetry.record_terminal_reason(incident_id, "ESCALATED", reason)
                log_event(logger, logging.WARNING, "executor_post_resume_run_mismatch", reason)
                return {
                    "status": IncidentStatus.ESCALATED,
                    "final_summary": reason,
                }

        # Safe autonomous execution (or approved destructive execution)
        pre_wf = None
        try:
            pre_wf = _get_env_state(resolved_env, state.get("workflow_run_id"))
        except Exception:
            pre_wf = None

        record = runner.execute(action)

        post_wf = None
        try:
            post_wf = _get_env_state(resolved_env, state.get("workflow_run_id"))
        except Exception:
            post_wf = None

        exec_id = (
            record.execution_id
            or state.get("workflow_run_id")
            or (getattr(post_wf, "run_id", None) if post_wf is not None else None)
        )
        task_attempts = (
            record.task_attempts
            if record.task_attempts is not None
            else (getattr(post_wf, "total_attempts", None) if post_wf is not None else None)
        )
        attempts_changed = (
            pre_wf is not None
            and post_wf is not None
            and getattr(pre_wf, "total_attempts", None) is not None
            and getattr(post_wf, "total_attempts", None) is not None
            and getattr(post_wf, "total_attempts", None) != getattr(pre_wf, "total_attempts", None)
        )

        update_fields: Dict[str, Any] = {
            "parameters": record.parameters or dict(action.parameters),
            "command_success": record.success if record.command_success is None else record.command_success,
            "execution_id": exec_id,
            "task_attempts": task_attempts,
        }
        if decision is not None:
            update_fields.update(
                {
                    "approved_by": decision.approver_id,
                    "approver_role": decision.approver_role,
                    "approval_id": decision.approval_request_id,
                }
            )
        record = record.model_copy(update=update_fields)

        # Multi-Step Execution & Failure Handling
        retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", 3)
        is_exhausted = retry_count >= max_retries

        # If action execution failed, stop remaining plan and route to verifier (which records
        # VerificationRecord and escalates if retry budget is exhausted).
        if not record.success:
            if attempts_changed and record.tool_name == "resume_workflow":
                record = record.model_copy(update={"environment_changed": True})
            reason = (
                f"Execution failed for action '{action.action_id}' ({action.tool_name}): "
                f"{record.stderr}. Aborting remaining plan."
            )
            if is_exhausted:
                telemetry.record_terminal_reason(incident_id, "ESCALATED", reason)
            return {
                "execution_history": [record],
                "status": IncidentStatus.FAILED,
                "final_summary": reason,
            }

        # M11: Deterministic postcondition check for mutating actions.
        postcondition_error = _check_postcondition(
            action.tool_name,
            action.parameters,
            resolved_env,
            bound_run_id=state.get("workflow_run_id"),
        )
        if postcondition_error:
            combined_stderr = (
                f"{record.stderr}\n{postcondition_error}".strip()
                if record.stderr
                else postcondition_error
            )
            record = record.model_copy(
                update={
                    "command_success": True,
                    "success": False,
                    "postcondition_passed": False,
                    "postcondition_error": postcondition_error,
                    "stderr": combined_stderr,
                    "environment_changed": False,
                }
            )
            reason = (
                f"Execution postcondition failed for '{action.action_id}' ({action.tool_name}): "
                f"{postcondition_error}. Aborting remaining plan."
            )
            if is_exhausted:
                telemetry.record_terminal_reason(incident_id, "ESCALATED", reason)
            log_event(
                logger,
                logging.WARNING,
                "executor_postcondition_failed",
                reason,
                action_id=action.action_id,
                tool_name=action.tool_name,
            )
            return {
                "execution_history": [record],
                "status": IncidentStatus.FAILED,
                "final_summary": reason,
            }

        if effective_risk != RiskLevel.READ_ONLY:
            record = record.model_copy(update={"postcondition_passed": True})

        # Action succeeded and postcondition passed: advance to the next planned action
        next_index = index + 1
        if next_index < len(plan):
            return {
                "execution_history": [record],
                "current_action_index": next_index,
                "status": IncidentStatus.EXECUTING,
            }
        else:
            return {
                "execution_history": [record],
                "current_action_index": next_index,
                "status": IncidentStatus.VERIFYING,
            }

    return executor_node
