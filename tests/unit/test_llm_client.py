"""
Unit tests for LLM client layer — MockLLMClient and schema validation.

Design principle (AGENTS.md):
  "LLM reasoning only where it adds genuine value."

These tests prove:
  1. MockLLMClient produces the same Pydantic types GeminiLLMClient would.
  2. The classifier helpers (_infer_failure_type, _extract_implicated_resource)
     correctly parse LLM-generated strings — including the cross-contamination
     edge case where a replanner summary mentions the previous fix action.
  3. Schema validation rejects structurally invalid LLM output at the boundary.
  4. The escalation path triggers correctly for unrecoverable failure types.
  5. The replanner → planner handoff preserves correct failure type across cycles.

None of these tests make a network call. All use MockLLMClient.
"""
import pytest
from nova.llm.client import MockLLMClient
from nova.llm.schemas import (
    PlannedActionSchema,
    PlannerResult,
    ReplannerResult,
    TriageResult,
)
from nova.nodes.planner import (
    AVAILABLE_TOOLS_FOR_LLM,
    _extract_implicated_resource,
    _infer_failure_type,
    _infer_recoverability,
)
from nova.state import RiskLevel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def llm() -> MockLLMClient:
    return MockLLMClient()


# ---------------------------------------------------------------------------
# MockLLMClient.triage() — log parsing and TriageResult output
# ---------------------------------------------------------------------------

class TestMockLLMClientTriage:

    def test_stale_lockfile_scenario(self, llm: MockLLMClient):
        """Lock filename is extracted dynamically from the log, not hardcoded."""
        log = (
            "[ERROR] Cannot acquire lock: file '/locks/transform_job.lock' exists\n"
            "[FATAL] Process 48291 exited with code 1 — job aborted\n"
        )
        result = llm.triage("JOB_02_TRANSFORM", log)

        assert isinstance(result, TriageResult)
        assert result.failure_type == "STALE_LOCKFILE"
        assert "transform_job.lock" in result.implicated_resource  # extracted, not hardcoded
        assert result.is_recoverable is True
        assert result.confidence == "HIGH"

    def test_service_offline_scenario(self, llm: MockLLMClient):
        """Service name is extracted dynamically from the log."""
        log = (
            "[ERROR] Connection to 'reporting_warehouse_service' failed\n"
            "[FATAL] Connection refused on port 5432 — aborting\n"
        )
        result = llm.triage("JOB_03_LOAD", log)

        assert isinstance(result, TriageResult)
        assert result.failure_type == "SERVICE_OFFLINE"
        assert result.implicated_resource == "reporting_warehouse_service"
        assert result.is_recoverable is True
        assert result.confidence == "HIGH"

    def test_data_corruption_scenario(self, llm: MockLLMClient):
        """Data corruption is flagged as non-recoverable — requires human."""
        log = (
            "[ERROR] Checksum mismatch: expected 0xDEAD, got 0x0000\n"
            "[FATAL] Zero-byte records detected in source feed — data unreadable\n"
        )
        result = llm.triage("JOB_02_TRANSFORM", log)

        assert isinstance(result, TriageResult)
        assert result.failure_type == "DATA_CORRUPTION"
        assert result.is_recoverable is False  # critical: cannot auto-fix
        assert result.confidence == "HIGH"

    def test_unknown_failure_returns_low_confidence(self, llm: MockLLMClient):
        """When no recognizable pattern exists, triage returns LOW confidence."""
        log = "[ERROR] Unexpected NullPointerException in transformation step\n"
        result = llm.triage("JOB_02_TRANSFORM", log)

        assert isinstance(result, TriageResult)
        assert result.failure_type == "UNKNOWN"
        assert result.is_recoverable is False
        assert result.confidence == "LOW"

    def test_implicated_resource_comes_from_log_not_target_service(self, llm: MockLLMClient):
        """
        The lock filename is extracted from the log content, not the target_service argument.
        This proves the triage is dynamic — different logs produce different resource names.
        """
        log_a = "[FATAL] Cannot acquire lock: file 'etl_job.lock' exists — aborting"
        log_b = "[FATAL] Cannot acquire lock: file 'archive_pipeline.lock' exists — aborting"

        result_a = llm.triage("JOB_02_TRANSFORM", log_a)
        result_b = llm.triage("JOB_02_TRANSFORM", log_b)

        # Both are lockfile failures — but different lock names extracted from log
        assert result_a.failure_type == "STALE_LOCKFILE"
        assert result_b.failure_type == "STALE_LOCKFILE"
        assert result_a.implicated_resource != result_b.implicated_resource
        assert "etl_job.lock" in result_a.implicated_resource
        assert "archive_pipeline.lock" in result_b.implicated_resource


# ---------------------------------------------------------------------------
# MockLLMClient.plan() — TriageResult → PlannerResult
# ---------------------------------------------------------------------------

class TestMockLLMClientPlan:

    def test_stale_lockfile_plans_clear_lockfile(self, llm: MockLLMClient):
        """Planner produces clear_lockfile action with dynamic lock_name from triage."""
        triage = TriageResult(
            failure_type="STALE_LOCKFILE",
            implicated_resource="transform_job.lock",
            triage_summary="Orphaned lockfile blocking transform job.",
            root_cause_hypothesis="Previous run crashed without releasing lock.",
            is_recoverable=True,
            confidence="HIGH",
        )
        result = llm.plan(triage, AVAILABLE_TOOLS_FOR_LLM, retry_count=0)

        assert isinstance(result, PlannerResult)
        assert result.is_escalation is False
        assert len(result.actions) == 2
        action = result.actions[0]
        assert action.tool_name == "clear_lockfile"
        assert action.parameters["lock_name"] == "transform_job.lock"  # from triage, not hardcoded
        assert action.risk_level == RiskLevel.DESTRUCTIVE
        assert result.actions[1].tool_name == "resume_workflow"
        assert result.actions[1].parameters["workflow_id"] == "WF_NIGHTLY_BATCH_1001"

    def test_service_offline_plans_restart_service(self, llm: MockLLMClient):
        """Planner produces restart_service with the exact service name from triage."""
        triage = TriageResult(
            failure_type="SERVICE_OFFLINE",
            implicated_resource="reporting_warehouse_service",
            triage_summary="Service is offline, connection refused.",
            root_cause_hypothesis="Service crashed before batch run.",
            is_recoverable=True,
            confidence="HIGH",
        )
        result = llm.plan(triage, AVAILABLE_TOOLS_FOR_LLM, retry_count=0)

        assert result.is_escalation is False
        assert len(result.actions) == 2
        assert result.actions[0].tool_name == "restart_service"
        assert result.actions[0].parameters["service_name"] == "reporting_warehouse_service"
        assert result.actions[1].tool_name == "resume_workflow"

    def test_data_corruption_triggers_escalation(self, llm: MockLLMClient):
        """Non-recoverable failures MUST escalate — never produce tool actions."""
        triage = TriageResult(
            failure_type="DATA_CORRUPTION",
            implicated_resource="JOB_02_TRANSFORM",
            triage_summary="Checksum mismatch — corrupt payload.",
            root_cause_hypothesis="Upstream feed delivered truncated data.",
            is_recoverable=False,
            confidence="HIGH",
        )
        result = llm.plan(triage, AVAILABLE_TOOLS_FOR_LLM, retry_count=0)

        assert result.is_escalation is True
        assert result.escalation_reason is not None
        assert len(result.actions) == 0  # no automated actions for unrecoverable failures

    def test_unknown_failure_escalates(self, llm: MockLLMClient):
        """Planner escalates when failure type has no automated strategy."""
        triage = TriageResult(
            failure_type="UNKNOWN",
            implicated_resource="JOB_02_TRANSFORM",
            triage_summary="Unrecognized failure signature.",
            root_cause_hypothesis="Insufficient evidence.",
            is_recoverable=False,
            confidence="LOW",
        )
        result = llm.plan(triage, AVAILABLE_TOOLS_FOR_LLM, retry_count=0)

        assert result.is_escalation is True
        assert len(result.actions) == 0


# ---------------------------------------------------------------------------
# MockLLMClient.replan() — downstream failure re-diagnosis
# ---------------------------------------------------------------------------

class TestMockLLMClientReplan:

    def test_replan_service_offline_downstream(self, llm: MockLLMClient):
        """
        After clear_lockfile succeeded, verifier found JOB_03_LOAD is now failing.
        replan() must correctly diagnose the new SERVICE_OFFLINE failure.
        """
        new_log = (
            "[ERROR] Connection to 'reporting_warehouse_service' failed\n"
            "[FATAL] Connection refused on port 5432\n"
        )
        result = llm.replan(
            original_target="JOB_02_TRANSFORM",
            first_action_taken="clear_lockfile",
            new_target="JOB_03_LOAD",
            new_log_content=new_log,
            verification_evidence="Pipeline advanced past JOB_02 but JOB_03 failed.",
            retry_count=1,
        )

        assert isinstance(result, ReplannerResult)
        assert result.new_target_service == "JOB_03_LOAD"
        assert result.new_failure_type == "SERVICE_OFFLINE"
        assert "reporting_warehouse_service" in result.updated_hypothesis
        # Acknowledge the first fix worked
        assert "clear_lockfile" in result.updated_triage_summary or \
               "JOB_02_TRANSFORM" in result.updated_triage_summary

    def test_replan_stale_lockfile_downstream(self, llm: MockLLMClient):
        """Downstream stale lockfile is correctly identified in replan."""
        new_log = "[FATAL] Cannot acquire lock 'archive.lock' — file exists\n"
        result = llm.replan(
            original_target="JOB_01_EXTRACT",
            first_action_taken="restart_service",
            new_target="JOB_02_TRANSFORM",
            new_log_content=new_log,
            verification_evidence="JOB_01 passed but JOB_02 failed.",
            retry_count=1,
        )

        assert result.new_failure_type == "STALE_LOCKFILE"
        assert result.new_target_service == "JOB_02_TRANSFORM"

    def test_replan_unknown_fallback(self, llm: MockLLMClient):
        """When new log has no recognizable pattern, replan uses DEPENDENCY_FAILURE."""
        result = llm.replan(
            original_target="JOB_01_EXTRACT",
            first_action_taken="clear_lockfile",
            new_target="JOB_02_TRANSFORM",
            new_log_content="[ERROR] Unexpected internal error in transformation logic\n",
            verification_evidence="JOB_01 resolved, JOB_02 now failing.",
            retry_count=1,
        )

        assert result.new_failure_type == "DEPENDENCY_FAILURE"
        assert result.new_target_service == "JOB_02_TRANSFORM"


# ---------------------------------------------------------------------------
# _infer_failure_type() — classifier priority edge cases
# ---------------------------------------------------------------------------

class TestInferFailureType:

    def test_service_offline_takes_priority_over_lock_mention(self):
        """
        CRITICAL edge case: replanner summary mentions 'clear_lockfile' (previous action)
        AND 'connection refused' (new failure). SERVICE_OFFLINE must win.

        This was the root cause of the cascading failure test regression.
        Before fix: 'lock' in 'clear_lockfile' caused STALE_LOCKFILE misclassification.
        """
        summary = (
            "Replanning cycle 1: Primary fix 'clear_lockfile' resolved JOB_02_TRANSFORM. "
            "Deep verification revealed JOB_03_LOAD now fails: connection to "
            "'reporting_warehouse_service' refused on port 5432."
        )
        hyp = (
            "'reporting_warehouse_service' is offline. After JOB_02_TRANSFORM was fixed "
            "by 'clear_lockfile', the pipeline advanced to JOB_03_LOAD which cannot "
            "connect to 'reporting_warehouse_service'."
        )
        result = _infer_failure_type(summary, hyp)
        assert result == "SERVICE_OFFLINE", (
            f"Expected SERVICE_OFFLINE but got {result}. "
            "The word 'lock' inside 'clear_lockfile' must NOT trigger STALE_LOCKFILE "
            "when 'connection refused' is also present."
        )

    def test_stale_lockfile_detected_by_extension(self):
        """STALE_LOCKFILE is correctly detected when .lock extension is present."""
        summary = "Orphaned 'transform_job.lock' is blocking the job."
        hyp = "Previous run crashed without releasing transform_job.lock."
        assert _infer_failure_type(summary, hyp) == "STALE_LOCKFILE"

    def test_data_corruption_detected(self):
        summary = "Checksum mismatch — payload is corrupt."
        hyp = "Upstream feed delivered a corrupted file."
        assert _infer_failure_type(summary, hyp) == "DATA_CORRUPTION"

    def test_dependency_failure_detected(self):
        summary = "Downstream dependency failure in JOB_03 after JOB_02 was resolved."
        hyp = "Component failed due to upstream dependency resolution."
        assert _infer_failure_type(summary, hyp) == "DEPENDENCY_FAILURE"

    def test_unknown_when_no_pattern_matches(self):
        summary = "Job exited with an internal error."
        hyp = "Root cause unclear."
        assert _infer_failure_type(summary, hyp) == "UNKNOWN"


# ---------------------------------------------------------------------------
# _extract_implicated_resource() — resource name extraction
# ---------------------------------------------------------------------------

class TestExtractImplicatedResource:

    def test_extracts_lock_filename(self):
        summary = "Orphaned lockfile 'transform_job.lock' is blocking the transform job."
        assert _extract_implicated_resource(summary) == "transform_job.lock"

    def test_extracts_service_name(self):
        summary = "Cannot connect to 'reporting_warehouse_service'. Connection refused."
        assert _extract_implicated_resource(summary) == "reporting_warehouse_service"

    def test_lock_takes_precedence_over_service(self):
        """When both appear, lockfile wins (it's more specific)."""
        summary = "Lockfile 'etl.lock' blocking 'some_service' from running."
        result = _extract_implicated_resource(summary)
        assert result == "etl.lock"

    def test_fallback_quoted_string(self):
        """Falls back to any quoted name if no specific pattern matches."""
        summary = "Failed to process 'batch_config_file' in the transform stage."
        result = _extract_implicated_resource(summary)
        assert result == "batch_config_file"

    def test_unknown_resource_fallback(self):
        """Returns 'unknown_resource' when nothing can be extracted."""
        summary = "Job failed without mentioning any specific resource."
        assert _extract_implicated_resource(summary) == "unknown_resource"


# ---------------------------------------------------------------------------
# _infer_recoverability() — escalation gate
# ---------------------------------------------------------------------------

class TestInferRecoverability:

    def test_lockfile_is_recoverable(self):
        summary = "Orphaned 'transform_job.lock' blocking the job."
        hyp = "Lock was not released by previous run."
        assert _infer_recoverability(summary, hyp) is True

    def test_service_offline_is_recoverable(self):
        summary = "Connection refused to 'reporting_warehouse_service'."
        hyp = "Service crashed and needs restart."
        assert _infer_recoverability(summary, hyp) is True

    def test_data_corruption_is_not_recoverable(self):
        summary = "Checksum mismatch detected in source payload."
        hyp = "Upstream delivered a corrupt file — zero-byte records present."
        assert _infer_recoverability(summary, hyp) is False

    def test_corrupt_keyword_alone_triggers_non_recoverable(self):
        summary = "Corrupt data blocks found in input file."
        hyp = "Data integrity failure."
        assert _infer_recoverability(summary, hyp) is False


# ---------------------------------------------------------------------------
# Schema validation — Pydantic boundary enforcement
# ---------------------------------------------------------------------------

class TestSchemaValidation:

    def test_triage_result_requires_all_fields(self):
        """TriageResult raises ValidationError if required fields are missing."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            TriageResult(
                failure_type="STALE_LOCKFILE",
                # implicated_resource missing
                triage_summary="Some summary",
                root_cause_hypothesis="Some hypothesis",
                is_recoverable=True,
            )  # type: ignore

    def test_planned_action_schema_valid_risk_levels(self):
        """PlannedActionSchema accepts valid RiskLevel values."""
        for level in [RiskLevel.READ_ONLY, RiskLevel.LOW_RISK, RiskLevel.DESTRUCTIVE]:
            schema = PlannedActionSchema(
                tool_name="check_service_status",
                parameters={"service_name": "some_service"},
                risk_level=level,
                rationale="Testing risk level acceptance.",
                expected_outcome="Tool executes without error.",
            )
            assert schema.risk_level == level

    def test_planner_result_escalation_requires_reason(self):
        """
        When is_escalation=True, escalation_reason must be set.
        (Enforced by validator in PlannerResult.)
        """
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            PlannerResult(
                is_escalation=True,
                escalation_reason=None,  # must be set when escalating
                actions=[],
            )

    def test_planner_result_no_escalation_allows_null_reason(self):
        """When is_escalation=False, escalation_reason can be None."""
        result = PlannerResult(
            is_escalation=False,
            escalation_reason=None,
            actions=[
                PlannedActionSchema(
                    tool_name="resume_workflow",
                    parameters={},
                    risk_level=RiskLevel.LOW_RISK,
                    rationale="Retry from checkpoint.",
                    expected_outcome="Pipeline resumes.",
                )
            ],
        )
        assert result.is_escalation is False
        assert result.escalation_reason is None
