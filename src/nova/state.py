import operator
from enum import Enum
from typing import Annotated, Any, Dict, List, Optional, TypedDict
from pydantic import BaseModel, Field


class IncidentStatus(str, Enum):
    """Lifecycle stages of an incident being handled by NOVA."""
    TRIGGERED = "TRIGGERED"
    TRIAGING = "TRIAGING"
    PLANNING = "PLANNING"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    EXECUTING = "EXECUTING"
    VERIFYING = "VERIFYING"
    REPLANNING = "REPLANNING"
    RESOLVED = "RESOLVED"
    ESCALATED = "ESCALATED"
    FAILED = "FAILED"


class RiskLevel(str, Enum):
    """
    Safety classification for operations.
    - READ_ONLY: Diagnostic/inspection commands (always safe).
    - LOW_RISK: Idempotent changes with negligible side effects.
    - DESTRUCTIVE: Modifies state, deletes files, stops processes (requires Human-in-the-Loop approval).
    """
    READ_ONLY = "READ_ONLY"
    LOW_RISK = "LOW_RISK"
    DESTRUCTIVE = "DESTRUCTIVE"


class PlannedAction(BaseModel):
    """Represents a discrete step formulated by the planning LLM."""
    action_id: str = Field(description="Unique identifier for this action, e.g. 'ACT_01'")
    tool_name: str = Field(description="Exact name of tool to invoke, e.g. 'clear_lockfile'")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Typed arguments passed to the tool")
    risk_level: RiskLevel = Field(default=RiskLevel.LOW_RISK, description="Safety rating of the action")
    rationale: str = Field(description="Why this action is needed to remediate the issue")
    expected_outcome: str = Field(description="What condition should be true after this executes")
    action_hash: Optional[str] = Field(default=None, description="Deterministic SHA256 signature for idempotency")
    idempotency_token: Optional[str] = Field(default=None, description="Unique execution token to prevent duplicate runs")


ALLOWED_APPROVER_ROLES = {
    "SRE_LEAD",
    "PLATFORM_ADMIN",
    "L2_OPERATIONS",
    "INCIDENT_COMMANDER",
}


class ApprovalDecision(BaseModel):
    """Structured decision recorded when human operator reviews an action."""
    approved: bool
    approver_id: str
    approver_role: str
    decision_timestamp: str
    incident_id: str
    action_id: str
    approval_request_id: Optional[str] = None
    action_hash: Optional[str] = None
    notes: Optional[str] = None


class ExecutionRecord(BaseModel):
    """Records the exact outcome of invoking a tool."""
    action_id: str
    tool_name: str
    executed_at: str
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    success: bool
    parameters: Dict[str, Any] = Field(default_factory=dict)
    command_success: Optional[bool] = None
    postcondition_passed: Optional[bool] = None
    postcondition_error: Optional[str] = None
    environment_changed: Optional[bool] = None
    execution_id: Optional[str] = None
    task_attempts: Optional[int] = None
    details: Dict[str, Any] = Field(default_factory=dict)
    action_hash: Optional[str] = None
    idempotency_token: Optional[str] = None
    approved_by: Optional[str] = None
    approver_role: Optional[str] = None
    approval_id: Optional[str] = None


class VerificationRecord(BaseModel):
    """Records the result of the independent post-condition check."""
    action_id: str
    verified_at: str
    passed: bool
    evidence: str
    details: Dict[str, Any] = Field(default_factory=dict)


class DiagnosticRecord(BaseModel):
    """Records a read-only diagnostic tool execution during autonomous investigation."""
    step: int
    tool_name: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    output: str
    success: bool
    timestamp: str
    rationale: Optional[str] = None
    execution_epoch: int = 0
    env_fingerprint: Optional[str] = None


class ApprovalRequest(BaseModel):
    """Contains context when execution is halted for human review."""
    action: PlannedAction
    requested_at: str
    reason: str
    status: str = "PENDING"  # PENDING, APPROVED, REJECTED
    operator_notes: Optional[str] = None


class NovaState(BaseModel):
    """
    Pydantic schema representation of NOVA's state.
    Used for external API serialization, input validation, and database storage.
    """
    incident_id: str
    target_service: str
    raw_symptoms: str
    status: IncidentStatus = IncidentStatus.TRIGGERED

    # Workflow Run Binding (M11 / M12 / M13)
    workflow_id: Optional[str] = None
    workflow_run_id: Optional[str] = None
    baseline_attempts: Optional[int] = None
    baseline_task_attempts: Optional[Dict[str, int]] = None
    baseline_task_states: Optional[Dict[str, str]] = None
    expected_output: Optional[Dict[str, Any]] = None

    # Investigation & Triage
    investigation_steps: int = 0
    max_investigation_steps: int = 5
    investigation_evidence: List[str] = Field(default_factory=list)
    diagnostic_history: List[DiagnosticRecord] = Field(default_factory=list)
    reflection_history: List[Dict[str, Any]] = Field(default_factory=list)
    hypothesis_revisions: List[Dict[str, Any]] = Field(default_factory=list)
    plan_revisions: List[Dict[str, Any]] = Field(default_factory=list)

    # Diagnosis & Planning
    failure_type: Optional[str] = None
    implicated_resource: Optional[str] = None
    is_recoverable: Optional[bool] = None
    triage_summary: Optional[str] = None
    root_cause_hypothesis: Optional[str] = None
    plan: List[PlannedAction] = Field(default_factory=list)
    current_action_index: int = 0

    # History & Audit Trails
    execution_history: List[ExecutionRecord] = Field(default_factory=list)
    verification_history: List[VerificationRecord] = Field(default_factory=list)

    # Safety & Governance
    pending_approval: Optional[ApprovalRequest] = None

    # Retry Budgets & Resilience
    retry_count: int = 0
    max_retries: int = 3

    # Final Output
    final_summary: Optional[str] = None


class NovaGraphState(TypedDict, total=False):
    """
    LangGraph-native state structure with explicit Reducers.
    Using Annotated[List[...], operator.add] ensures that returning
    a single new record appends to the history rather than overwriting it.
    """
    incident_id: str
    target_service: str
    raw_symptoms: str
    status: IncidentStatus

    # Workflow Run Binding (M11 / M12 / M13)
    workflow_id: Optional[str]
    workflow_run_id: Optional[str]
    baseline_attempts: Optional[int]
    baseline_task_attempts: Optional[Dict[str, int]]
    baseline_task_states: Optional[Dict[str, str]]
    expected_output: Optional[Dict[str, Any]]

    # Investigation & Triage
    investigation_steps: int
    max_investigation_steps: int
    investigation_evidence: Annotated[List[str], operator.add]
    diagnostic_history: Annotated[List[DiagnosticRecord], operator.add]
    reflection_history: Annotated[List[Dict[str, Any]], operator.add]
    hypothesis_revisions: Annotated[List[Dict[str, Any]], operator.add]
    plan_revisions: Annotated[List[Dict[str, Any]], operator.add]

    # Diagnosis & Planning
    failure_type: Optional[str]
    implicated_resource: Optional[str]
    is_recoverable: Optional[bool]
    triage_summary: Optional[str]
    root_cause_hypothesis: Optional[str]
    plan: List[PlannedAction]
    current_action_index: int

    # Reducer fields: Appends new records across graph execution cycles
    execution_history: Annotated[List[ExecutionRecord], operator.add]
    verification_history: Annotated[List[VerificationRecord], operator.add]

    # Safety & Governance
    pending_approval: Optional[ApprovalRequest]

    # Retry Budgets & Resilience
    retry_count: int
    max_retries: int

    # Final Output
    final_summary: Optional[str]


def build_prior_attempts(state: NovaGraphState | Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Constructs a complete, non-lossy history of attempted actions, parameters,
    tool outputs, command vs. postcondition outcomes, environment changes,
    execution/attempt IDs, expected outcomes, and verification records.
    """
    plan_by_id: Dict[str, Dict[str, Any]] = {}
    for act in state.get("plan") or []:
        act_dict = act.model_dump() if hasattr(act, "model_dump") else dict(act)
        if act_dict.get("action_id"):
            plan_by_id[act_dict["action_id"]] = act_dict

    prior_attempts: List[Dict[str, Any]] = []
    for rec in state.get("execution_history") or []:
        rec_dict = rec.model_dump() if hasattr(rec, "model_dump") else dict(rec)
        act_id = rec_dict.get("action_id")
        planned = plan_by_id.get(act_id or "", {})
        cmd_ok = rec_dict.get("command_success")
        if cmd_ok is None:
            cmd_ok = rec_dict.get("success")
        prior_attempts.append(
            {
                "kind": "execution",
                "action_id": act_id,
                "tool_name": rec_dict.get("tool_name"),
                "parameters": rec_dict.get("parameters") or planned.get("parameters") or {},
                "expected_outcome": planned.get("expected_outcome"),
                "stdout": (rec_dict.get("stdout") or "")[:600],
                "stderr": (rec_dict.get("stderr") or "")[:600],
                "command_success": cmd_ok,
                "success": rec_dict.get("success"),
                "exit_code": rec_dict.get("exit_code"),
                "postcondition_passed": rec_dict.get("postcondition_passed"),
                "postcondition_error": rec_dict.get("postcondition_error"),
                "environment_changed": rec_dict.get("environment_changed"),
                "execution_id": rec_dict.get("execution_id") or state.get("workflow_run_id"),
                "task_attempts": rec_dict.get("task_attempts"),
                "details": rec_dict.get("details") or {},
            }
        )
    for vrec in state.get("verification_history") or []:
        v_dict = vrec.model_dump() if hasattr(vrec, "model_dump") else dict(vrec)
        prior_attempts.append(
            {
                "kind": "verification",
                "action_id": v_dict.get("action_id"),
                "passed": v_dict.get("passed"),
                "evidence": (v_dict.get("evidence") or "")[:600],
                "details": v_dict.get("details") or {},
            }
        )
    return prior_attempts
