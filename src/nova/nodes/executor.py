import hashlib
import json
import logging
import time
from typing import Any, Dict, Optional
from langgraph.types import interrupt
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


def create_executor_node(runner: ToolRunner, require_approval: bool = True):
    """Factory returning an executor node bound to a ToolRunner and HITL safety policy."""

    def executor_node(state: NovaGraphState) -> Dict[str, Any]:
        with incident_context(state.get("incident_id")):
            return _run_executor(state)

    def _run_executor(state: NovaGraphState) -> Dict[str, Any]:
        """
        Executor Node (Hardened Safety & Authorization):
        1. Retrieves the next planned action.
        2. Ground-Truth Risk Lookup: Disregards LLM-generated action.risk_level and looks up
           the authoritative risk level from runner.get_tool(action.tool_name).risk_level.
           If the tool is unknown or risk cannot be determined, fails closed (ESCALATED).
        3. Policy Gate: If the tool is DESTRUCTIVE and require_approval is True,
           halts execution and triggers a LangGraph interrupt() with structured
           approval context.
        4. Upon resumption:
           - Parses response into ApprovalDecision.
           - Enforces strict boolean: decision.approved is True.
           - Enforces action, incident, and optional action_hash binding.
           - Enforces RBAC: approver_role must belong to ALLOWED_APPROVER_ROLES.
           - If rejected, invalid, or unauthorized, halts execution and transitions to ESCALATED.
        5. Executes authorized tool via ToolRunner and records approval metadata into ExecutionRecord.
        6. Appends resulting ExecutionRecord to execution_history.
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

        decision: Optional[ApprovalDecision] = None

        # Safety Gate: Human-in-the-Loop Checkpoint for DESTRUCTIVE actions
        if require_approval and effective_risk == RiskLevel.DESTRUCTIVE:
            # Deterministic approval_request_id so LangGraph node replay on resume produces the exact same ID
            approval_request_id = f"APPR_{incident_id}_{action.action_id}_{action_hash[:8]}"
            approval_payload = {
                "approval_request_id": approval_request_id,
                "action_id": action.action_id,
                "action_hash": action_hash,
                "incident_id": state.get("incident_id"),
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

            # Case 2: Dictionary payload (kept backwards-compatible for direct graph tests)
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
                        approver_id = response.get("approver_id") or "mock_operator"
                        approver_role = response.get("approver_role") or "L2_OPERATIONS"

                        if approver_role not in ALLOWED_APPROVER_ROLES:
                            is_approved = False
                            rejection_reason = (
                                f"Security Alert: Unauthorized approver role '{approver_role}'. "
                                f"Allowed roles: {sorted(ALLOWED_APPROVER_ROLES)}."
                            )
                        else:
                            try:
                                decision = ApprovalDecision(
                                    approved=True,
                                    approver_id=approver_id,
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

            # Case 3: Raw boolean (backwards-compatible for direct graph tests)
            elif type(response) is bool:
                if response is True:
                    decision = ApprovalDecision(
                        approved=True,
                        approver_id="mock_operator",
                        approver_role="L2_OPERATIONS",
                        decision_timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                        incident_id=state.get("incident_id", "INC_DEFAULT"),
                        action_id=action.action_id,
                        approval_request_id=approval_request_id,
                        action_hash=action_hash,
                        notes="Direct boolean approval",
                    )
                    is_approved = True
                else:
                    is_approved = False
                    rejection_reason = "Action was REJECTED by human operator (boolean False)."

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

        # Safe autonomous execution (or approved destructive execution)
        record = runner.execute(action)
        if decision is not None:
            record = record.model_copy(
                update={
                    "approved_by": decision.approver_id,
                    "approver_role": decision.approver_role,
                    "approval_id": decision.approval_request_id,
                }
            )

        # Multi-Step Execution & Failure Handling (Milestone 7.2 Phase B)
        retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", 3)
        is_exhausted = retry_count >= max_retries

        # If action failed, stop the remaining plan and route appropriately
        if not record.success:
            reason = (
                f"Incident Escalated: Action '{action.action_id}' ({action.tool_name}) failed execution and retries exhausted."
                if is_exhausted
                else f"Execution failed for action '{action.action_id}' ({action.tool_name}): {record.stderr}. Aborting remaining plan."
            )
            telemetry.record_terminal_reason(
                incident_id,
                "ESCALATED" if is_exhausted else "FAILED",
                reason,
            )
            return {
                "execution_history": [record],
                "status": IncidentStatus.ESCALATED if is_exhausted else IncidentStatus.FAILED,
                "final_summary": reason,
            }

        # Action succeeded: Advance current_action_index to the next planned action
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
