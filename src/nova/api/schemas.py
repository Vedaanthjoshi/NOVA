"""
Pydantic API contracts for the NOVA FastAPI Gateway.

Enforces strict input validation, type safety, and standardized responses
for incident submission, status polling, and Human-in-the-Loop approvals.
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator

from nova.state import IncidentStatus


class IncidentTriggerRequest(BaseModel):
    """Payload for submitting a new operational incident to NOVA."""
    incident_id: Optional[str] = Field(
        default=None,
        description="Optional custom identifier. Auto-generated if omitted (e.g. 'INC_A1B2C3D4').",
    )
    target_service: Optional[str] = Field(
        default="UNKNOWN",
        description="Implicated service or job. Set to 'UNKNOWN' if unknown to trigger autonomous investigation.",
    )
    raw_symptoms: str = Field(
        ...,
        description="Unstructured symptom text, alert payload, or log excerpt.",
        examples=["JOB_02_TRANSFORM exited with code 1. Lockfile detected."],
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum bounded replanning recovery cycles before escalating.",
    )
    max_investigation_steps: int = Field(
        default=5,
        ge=1,
        le=15,
        description="Maximum autonomous investigation diagnostic steps.",
    )
    environment: Literal["mock", "kestra"] = Field(
        default="mock",
        description="Execution backend to bind: 'mock' (MockWorkflowRunner) or 'kestra' (KestraEnvironment).",
    )
    execution_id: Optional[str] = Field(
        default=None,
        description="Kestra execution ID to bind when environment='kestra'.",
    )
    kestra_url: Optional[str] = Field(
        default=None,
        description="Optional Kestra server base URL override when environment='kestra'.",
    )


class IncidentTriggerResponse(BaseModel):
    """Immediate acknowledgment returned upon incident submission (non-blocking)."""
    incident_id: str
    status: IncidentStatus
    target_service: Optional[str]
    raw_symptoms: str
    created_at: str
    message: str = "Incident submitted and queued for autonomous remediation."


class ApprovalSubmitRequest(BaseModel):
    """Payload for operator submitting a Human-in-the-Loop authorization decision."""
    approved: bool = Field(
        ...,
        description="True to authorize the pending action; False to reject.",
    )
    approver_id: str = Field(
        ...,
        min_length=1,
        description="Identifier of the operator reviewing the action (e.g. 'OP_JSMITH_01').",
    )
    approver_role: str = Field(
        ...,
        description="Role of the operator. Must be in ALLOWED_APPROVER_ROLES (e.g. 'SRE_LEAD', 'PLATFORM_ADMIN').",
    )
    action_id: str = Field(
        ...,
        description="ID of the specific action being authorized (must match pending interrupt).",
    )
    action_hash: Optional[str] = Field(
        default=None,
        description="Optional deterministic action SHA256 signature for strict binding.",
    )
    notes: Optional[str] = Field(
        default=None,
        description="Audit notes explaining the reason for approval or rejection.",
    )

    @field_validator("approver_id")
    @classmethod
    def validate_approver_id_non_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("approver_id must be a non-empty identifier.")
        return cleaned


class ApprovalSubmitResponse(BaseModel):
    """Response returned upon processing an operator's approval decision."""
    incident_id: str
    action_id: str
    approved: bool
    resumed: bool
    status: IncidentStatus
    message: str


class PendingApprovalDTO(BaseModel):
    """Details of an action currently interrupted and awaiting human review."""
    action_id: str
    tool_name: str
    risk_level: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    rationale: str
    expected_outcome: str
    requested_at: Optional[str] = None
    approval_request_id: Optional[str] = None
    action_hash: Optional[str] = None


class IncidentStateResponse(BaseModel):
    """Comprehensive state representation returned during polling."""
    incident_id: str
    status: IncidentStatus
    target_service: Optional[str]
    raw_symptoms: str

    # Workflow Run Binding (M11)
    workflow_id: Optional[str] = None
    workflow_run_id: Optional[str] = None

    # Autonomous Investigation Telemetry
    investigation_steps: int = 0
    max_investigation_steps: int = 5
    investigation_evidence: List[str] = Field(default_factory=list)
    diagnostic_history: List[Dict[str, Any]] = Field(default_factory=list)
    reflection_history: List[Dict[str, Any]] = Field(default_factory=list)
    hypothesis_revisions: List[Dict[str, Any]] = Field(default_factory=list)
    plan_revisions: List[Dict[str, Any]] = Field(default_factory=list)

    # Diagnosis & Planning
    failure_type: Optional[str] = None
    implicated_resource: Optional[str] = None
    is_recoverable: Optional[bool] = None
    triage_summary: Optional[str] = None
    root_cause_hypothesis: Optional[str] = None
    plan: List[Dict[str, Any]] = Field(default_factory=list)
    current_action_index: int = 0

    # Execution & Verification Telemetry
    execution_history: List[Dict[str, Any]] = Field(default_factory=list)
    verification_history: List[Dict[str, Any]] = Field(default_factory=list)

    # Safety & Governance Checkpoints
    pending_approval: Optional[PendingApprovalDTO] = None

    # Retry Budgets & Final Resolution
    retry_count: int = 0
    max_retries: int = 3
    final_summary: Optional[str] = None

    # Lightweight Observability & Durability Status
    telemetry: Optional[Dict[str, Any]] = None
    approval_audit_history: List[Dict[str, Any]] = Field(default_factory=list)
    worker_active: bool = False
    interrupted_by_restart: bool = False


class IncidentListItem(BaseModel):
    """Summary item for the incident list endpoint."""
    incident_id: str
    status: IncidentStatus
    target_service: Optional[str]
    failure_type: Optional[str]
    implicated_resource: Optional[str]
    created_at: str
    updated_at: str


class ScenarioInjectRequest(BaseModel):
    """Payload to trigger environment scenario injection for testing/demos."""
    scenario: str = Field(
        ...,
        description="Scenario to inject: STALE_LOCKFILE, SERVICE_OFFLINE, DATA_CORRUPTION, CASCADING_FAILURE",
    )


class ScenarioInjectResponse(BaseModel):
    """Response returned upon injecting a mock operational scenario."""
    workflow_id: str
    run_id: Optional[str] = None
    scenario: str
    failed_job: Optional[str]
    error_message: Optional[str]
    active_locks: List[str]


class HealthResponse(BaseModel):
    """Service health check response."""
    status: str
    version: str
    environment: str
    checkpointer: str
    active_incidents_count: int
    telemetry_summary: Optional[Dict[str, Any]] = None
