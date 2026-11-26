"""
Pydantic schemas for structured LLM outputs.

These are the CONTRACTS between the LLM and the rest of the graph.
Every field the LLM produces is validated here before it touches any node state.
Pydantic will raise ValidationError — not let bad LLM output silently corrupt state.
"""
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, model_validator

from nova.state import RiskLevel


class InvestigationActionType(str, Enum):
    """Actions the LLM can take during autonomous incident investigation."""
    DIAGNOSTIC_ACTION = "DIAGNOSTIC_ACTION"  # Request another read-only diagnostic tool
    DIAGNOSIS_READY = "DIAGNOSIS_READY"      # Sufficient evidence gathered; ready for planning
    ESCALATE = "ESCALATE"                    # Unrecoverable, out of scope, or inconclusive; escalate to human


class DiagnosticToolCall(BaseModel):
    """Specification of a read-only diagnostic tool requested by the LLM."""
    tool_name: str = Field(description="Exact name of the read-only diagnostic tool to invoke")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters for the diagnostic tool")
    rationale: str = Field(description="Why this diagnostic tool is needed to gather evidence")


class DiagnosticReflection(BaseModel):
    """Diagnostic self-critique evaluating evidence sufficiency, observations, and alternatives."""
    evidence_sufficient: bool = Field(
        default=True,
        description="True only if gathered observations conclusively support the diagnosis without guessing."
    )
    supporting_observations: List[str] = Field(
        default_factory=list,
        description="Specific collected observations that directly support the leading hypothesis."
    )
    contradicting_observations: List[str] = Field(
        default_factory=list,
        description="Observations that weaken or contradict the leading or prior hypothesis."
    )
    alternative_hypotheses: List[str] = Field(
        default_factory=list,
        description="Plausible alternative causes considered and ruled out (or still open)."
    )
    remaining_uncertainty: Optional[str] = Field(
        default=None,
        description="Any remaining ambiguity or unverified assumption in the evidence."
    )
    recommended_diagnostic_tool: Optional[DiagnosticToolCall] = Field(
        default=None,
        description="Specific distinguishing diagnostic observation to run when evidence_sufficient is False."
    )


class InvestigationDecision(BaseModel):
    """
    Structured output from an investigation step.
    The LLM inspects current evidence and either requests more diagnostic tools,
    concludes that diagnosis is ready, or escalates.
    """
    decision_type: InvestigationActionType
    diagnostic_tool: Optional[DiagnosticToolCall] = Field(
        default=None,
        description="Required if decision_type is DIAGNOSTIC_ACTION"
    )
    diagnosis: Optional["TriageResult"] = Field(
        default=None,
        description="Required if decision_type is DIAGNOSIS_READY"
    )
    escalation_reason: Optional[str] = Field(
        default=None,
        description="Required if decision_type is ESCALATE"
    )
    reflection: Optional[DiagnosticReflection] = Field(
        default=None,
        description="Optional diagnostic reflection assessing evidence sufficiency and alternative explanations."
    )

    @model_validator(mode="after")
    def validate_consistency(self) -> "InvestigationDecision":
        if self.decision_type == InvestigationActionType.DIAGNOSTIC_ACTION:
            if not self.diagnostic_tool:
                raise ValueError("diagnostic_tool must be provided when decision_type is DIAGNOSTIC_ACTION")
        elif self.decision_type == InvestigationActionType.DIAGNOSIS_READY:
            if not self.diagnosis:
                raise ValueError("diagnosis must be provided when decision_type is DIAGNOSIS_READY")
        elif self.decision_type == InvestigationActionType.ESCALATE:
            if not self.escalation_reason:
                raise ValueError("escalation_reason must be provided when decision_type is ESCALATE")
        return self


class TriageResult(BaseModel):
    """
    Structured output from the Triage LLM call.
    
    The LLM reads raw log text and produces this structured summary.
    The planner consumes this — it never sees raw log text directly.
    """
    failure_type: str = Field(
        description=(
            "Short category of the failure. Examples: 'STALE_LOCKFILE', 'SERVICE_OFFLINE', "
            "'DATA_CORRUPTION', 'DEPENDENCY_FAILURE', 'UNKNOWN'."
        )
    )
    implicated_resource: str = Field(
        description=(
            "The exact name of the resource involved in the failure. "
            "Examples: 'transform_job.lock', 'reporting_warehouse_service', 'JOB_01_EXTRACT'. "
            "Must be the precise value extracted from the log — do NOT invent or generalize."
        )
    )
    triage_summary: str = Field(
        description=(
            "A concise, factual one-to-two sentence summary of what failed and why, "
            "written as an SRE would write an incident summary."
        )
    )
    root_cause_hypothesis: str = Field(
        description=(
            "A technical hypothesis explaining why this failure occurred. "
            "Must be specific and actionable, not generic."
        )
    )
    is_recoverable: bool = Field(
        description=(
            "True if automated tooling can fix this. False if the data or state is fundamentally "
            "corrupt and requires human intervention (e.g. checksum mismatch, missing source file)."
        )
    )
    confidence: str = Field(
        default="HIGH",
        description="Confidence level in the diagnosis: HIGH, MEDIUM, or LOW."
    )
    workflow_id: Optional[str] = Field(
        default=None,
        description="Active workflow ID associated with the diagnosed incident, if known."
    )
    workflow_run_id: Optional[str] = Field(
        default=None,
        description="Active workflow run or execution ID bound to this incident, if known."
    )
    prior_attempts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Structured history of prior attempted actions, execution outcomes, and verification failures."
    )
    investigation_evidence: List[str] = Field(
        default_factory=list,
        description="Diagnostic observations gathered during investigation."
    )
    strategy_notes: Optional[str] = Field(
        default=None,
        description="Notes from replanning explaining why previous remediation failed or what changed."
    )
    critique_feedback: Optional[str] = Field(
        default=None,
        description="Critique feedback when revising a rejected initial remediation plan."
    )


class PlannedActionSchema(BaseModel):
    """
    A single remediation action as produced by the Planner LLM.
    
    Mirrors nova.state.PlannedAction but is a separate schema so the LLM
    output can be validated before being converted to domain objects.
    The key difference: the LLM must derive parameters from the triage context,
    not from hardcoded knowledge.
    """
    tool_name: str = Field(
        description=(
            "Exact tool name from the registered tool list. "
            "Must be one of the tools provided in the system prompt."
        )
    )
    parameters: Dict[str, Any] = Field(
        description=(
            "Arguments to pass to the tool. Keys and value types must exactly match "
            "the tool's documented parameter schema. Extract values from the triage context — "
            "never invent values not present in the logs or triage summary."
        )
    )
    risk_level: RiskLevel = Field(
        description="Risk classification: READ_ONLY, LOW_RISK, or DESTRUCTIVE."
    )
    rationale: str = Field(
        description="One sentence explaining why this specific action is the correct remediation."
    )
    expected_outcome: str = Field(
        description="Specific, verifiable condition that will be true after this action succeeds."
    )


class PlanningReflection(BaseModel):
    """Planning self-critique verifying alignment with root cause, tool appropriateness, and prior outcomes."""
    addresses_root_cause: bool = Field(
        default=True,
        description="Whether the proposed actions directly resolve the verified root cause."
    )
    contradicts_prior_attempts: bool = Field(
        default=False,
        description="True if this plan repeats a strategy that already failed without new justification."
    )
    requires_revision: bool = Field(
        default=False,
        description="True if the plan has flaws and must be revised before execution."
    )
    requires_further_investigation: bool = Field(
        default=False,
        description="True if evidence is insufficient/contradictory and triage must inspect further."
    )
    should_escalate: bool = Field(
        default=False,
        description="True if no valid automated action can resolve the diagnosed cause."
    )
    critique_notes: Optional[str] = Field(
        default=None,
        description="Specific critique identifying gaps, inappropriate parameters, or missing prerequisites."
    )
    strategy_assessment: str = Field(
        default="",
        description="Brief check confirming parameters match observed evidence and preconditions."
    )


class PlanCritiqueResult(BaseModel):
    """Structured critique evaluating a proposed remediation plan before execution."""
    approved: bool = Field(
        description="True ONLY if every proposed action is valid, grounded in evidence, addresses the diagnosed cause, and avoids repeating failed strategies."
    )
    addresses_root_cause: bool = Field(
        default=True,
        description="Whether proposed actions directly fix the diagnosed root cause."
    )
    contradicts_prior_attempts: bool = Field(
        default=False,
        description="True if the plan repeats an action/strategy that already failed without new evidence."
    )
    tools_valid_and_appropriate: bool = Field(
        default=True,
        description="True if all tools and parameters exist in the registry and are permitted/appropriate for the resource."
    )
    requires_further_investigation: bool = Field(
        default=False,
        description="True if planning cannot proceed without additional diagnostic observations."
    )
    should_escalate: bool = Field(
        default=False,
        description="True if the incident should escalate rather than execute remediation."
    )
    critique_summary: str = Field(
        description="Concrete, evidence-cited critique of the proposed plan."
    )
    recommended_adjustment: Optional[str] = Field(
        default=None,
        description="Concrete revision instruction if approved=False."
    )


class PlannerResult(BaseModel):
    """
    Structured output from the Planner LLM call.
    
    Either contains an ordered list of actions OR an escalation decision.
    The graph uses is_escalation to route — not keyword matching on strings.

    M11 correctness fix: a non-escalating plan with zero actions is invalid.
    The LLM must either provide at least one action or declare is_escalation=True.
    This prevents the graph from entering EXECUTING status with nothing to execute.
    """
    is_escalation: bool = Field(
        description=(
            "True if the failure is unrecoverable and must be escalated to a human operator "
            "immediately. No tools should be run. False means automated remediation is possible."
        )
    )
    escalation_reason: Optional[str] = Field(
        default=None,
        description="Required if is_escalation=True. Explain why no automated fix is possible."
    )
    actions: List[PlannedActionSchema] = Field(
        default_factory=list,
        description=(
            "Ordered list of actions to execute. Must be empty if is_escalation=True. "
            "Must be non-empty if is_escalation=False. "
            "Actions are executed in order; each depends on the previous succeeding."
        )
    )
    reflection: Optional[PlanningReflection] = Field(
        default=None,
        description="Optional planning reflection checking strategy alignment against prior attempts."
    )

    @model_validator(mode="after")
    def validate_escalation_consistency(self) -> "PlannerResult":
        """
        Enforce the escalation contract:
          - is_escalation=True  MUST have escalation_reason set and actions empty.
          - is_escalation=False MUST NOT have escalation_reason set.
          - is_escalation=False MUST have at least one action (M11 fix).

        This catches malformed LLM output at the schema boundary before it
        reaches any node or routing logic.
        """
        if self.is_escalation and (not self.escalation_reason or not self.escalation_reason.strip()):
            raise ValueError(
                "escalation_reason must be set when is_escalation=True. "
                "The LLM must explain why no automated fix is possible."
            )
        if self.is_escalation and len(self.actions) > 0:
            raise ValueError(
                "actions must be empty when is_escalation=True. "
                "An escalating plan cannot also schedule remediation actions."
            )
        if not self.is_escalation and self.escalation_reason:
            raise ValueError(
                "escalation_reason must be None when is_escalation=False. "
                "Only set it when actually escalating."
            )
        # M11: Reject empty non-escalating plans — they produce EXECUTING with no execution.
        if not self.is_escalation and len(self.actions) == 0:
            raise ValueError(
                "actions must be non-empty when is_escalation=False. "
                "Provide at least one remediation action, or set is_escalation=True with an escalation_reason."
            )
        return self


class ReplannerResult(BaseModel):
    """
    Structured output from the Replanner LLM call.
    
    After a verification failure, the LLM re-diagnoses the NEW failure
    (a downstream component that only became visible after the first fix).
    """
    new_target_service: str = Field(
        description=(
            "The job or service that is NOW the blocking failure. "
            "This should be a different component than the original target — "
            "it's the downstream dependency that surfaced after the first fix."
        )
    )
    updated_triage_summary: str = Field(
        description=(
            "Updated factual summary incorporating what the first remediation accomplished "
            "and what new failure was discovered in the downstream component."
        )
    )
    updated_hypothesis: str = Field(
        description="New technical hypothesis for the downstream failure, derived from the new log content."
    )
    new_failure_type: str = Field(
        description="Category of the newly discovered failure, e.g. 'SERVICE_OFFLINE', 'STALE_LOCKFILE'."
    )
    implicated_resource: Optional[str] = Field(
        default=None,
        description="Exact name of resource causing new failure (e.g. database service name, lockfile name)."
    )
    is_recoverable: bool = Field(
        default=True,
        description="Whether automated tooling can address the newly identified failure."
    )
    diagnosis_revision_needed: bool = Field(
        default=False,
        description="True if previous diagnosis was incorrect or must be revised based on execution/verification evidence."
    )
    is_escalation: bool = Field(
        default=False,
        description="True if replanning determined that the incident is unrecoverable and must escalate immediately."
    )
    escalation_reason: Optional[str] = Field(
        default=None,
        description="Required if is_escalation is True."
    )
    alternative_strategy_notes: Optional[str] = Field(
        default=None,
        description="Detailed notes explaining why previous plan failed and what alternative strategy is chosen."
    )
    needs_further_investigation: bool = Field(
        default=False,
        description="True if existing evidence is insufficient or contradicted and triage must gather more diagnostics."
    )
    outcome_classification: Optional[str] = Field(
        default=None,
        description=(
            "Distinguishes outcome cause: 'INCORRECT_DIAGNOSIS', 'INEFFECTIVE_REMEDIATION', "
            "'NEW_DOWNSTREAM_FAILURE', 'UNRECOVERABLE_DEFECT', or 'UNKNOWN'."
        ),
    )
    expected_vs_observed: Optional[str] = Field(
        default=None,
        description="Direct comparison of planned expected_outcome vs observed execution and verification telemetry."
    )
    outcome_reflection: Optional[str] = Field(
        default=None,
        description="Reflection on why the prior attempt failed or surfaced a new failure and whether diagnosis must change."
    )

    @model_validator(mode="after")
    def validate_replanner_escalation(self) -> "ReplannerResult":
        if self.is_escalation and (not self.escalation_reason or not self.escalation_reason.strip()):
            raise ValueError("escalation_reason must be provided when is_escalation is True.")
        if not self.is_escalation and self.escalation_reason:
            raise ValueError("escalation_reason must be None when is_escalation is False.")
        return self
