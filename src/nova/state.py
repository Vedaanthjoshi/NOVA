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

    # Investigation & Triage
    investigation_steps: int = 0
    max_investigation_steps: int = 5
    investigation_evidence: List[str] = Field(default_factory=list)
    diagnostic_history: List[DiagnosticRecord] = Field(default_factory=list)

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

    # Investigation & Triage
    investigation_steps: int
    max_investigation_steps: int
    investigation_evidence: Annotated[List[str], operator.add]
    diagnostic_history: Annotated[List[DiagnosticRecord], operator.add]

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
