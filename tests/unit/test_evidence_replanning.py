"""
Unit Tests for Evidence-Based Replanning (Milestone 7.2 Phase D).

Verifies that replanning:
1. Does NOT assume previous remediation succeeded.
2. Detects action execution failure from ExecutionRecord telemetry and revises diagnosis.
3. Detects downstream component failure when previous action succeeded.
4. Escalates safely when verification reveals unrecoverable data corruption.
"""
import pytest

from nova.environments.mock_workflow_runner import (
    MockWorkflowRunner,
    ScenarioType,
)
from nova.llm.client import MockLLMClient
from nova.llm.schemas import ReplannerResult
from nova.nodes.replanner import create_replanner_node
from nova.state import (
    ExecutionRecord,
    IncidentStatus,
    NovaGraphState,
    PlannedAction,
    RiskLevel,
    VerificationRecord,
)


@pytest.fixture
def env(tmp_path):
    runner = MockWorkflowRunner(base_dir=tmp_path / "mock_state")
    runner.inject_scenario(ScenarioType.CASCADING_FAILURE)
    return runner


def test_replan_when_previous_action_failed(env):
    """
    When an action fails execution (e.g. permission error, timeout),
    replanner must NOT assume downstream failure. It must inspect execution_history,
    recognize the failure, and flag diagnosis_revision_needed=True.
    """
    llm = MockLLMClient()
    replanner_node = create_replanner_node(env, llm)

    failed_exec = ExecutionRecord(
        action_id="ACT_01_01",
        tool_name="clear_lockfile",
        executed_at="2026-11-24 10:00:00",
        exit_code=1,
        stdout="",
        stderr="EACCES: Permission denied while attempting to unlink lockfile",
        duration_seconds=0.05,
        success=False,
    )

    verif_record = VerificationRecord(
        action_id="ACT_01_01",
        verified_at="2026-11-24 10:00:01",
        passed=False,
        evidence="Action ACT_01_01 failed execution: EACCES Permission denied",
        details={"failed_jobs": ["JOB_02_TRANSFORM"]},
    )

    state: NovaGraphState = {
        "incident_id": "INC_REPLAN_01",
        "target_service": "JOB_02_TRANSFORM",
        "raw_symptoms": "Transform job stalled",
        "status": IncidentStatus.FAILED,
        "failure_type": "STALE_LOCKFILE",
        "implicated_resource": "transform_job.lock",
        "triage_summary": "Orphaned lock blocking transform",
        "root_cause_hypothesis": "Previous run crashed without releasing lock",
        "retry_count": 0,
        "max_retries": 3,
        "current_action_index": 0,
        "execution_history": [failed_exec],
        "verification_history": [verif_record],
        "plan": [
            PlannedAction(
                action_id="ACT_01_01",
                tool_name="clear_lockfile",
                parameters={"lock_name": "transform_job.lock"},
                risk_level=RiskLevel.DESTRUCTIVE,
                rationale="Remove lock",
                expected_outcome="Lock gone",
            )
        ],
    }

    result = replanner_node(state)

    # 1. Stays on the same target service to seek alternative remediation
    assert result["target_service"] == "JOB_02_TRANSFORM"
    assert result["status"] == IncidentStatus.REPLANNING
    assert result["retry_count"] == 1

    # 2. Hypothesis explicitly acknowledges that the action FAILED
    assert "FAILED" in result["triage_summary"]
    assert "Revising diagnosis" in result["root_cause_hypothesis"]


def test_replan_when_previous_action_succeeded_downstream_discovered(env):
    """
    When the previous action succeeded and deep verification exposes a downstream failure,
    replanner identifies the new broken component and formulates a downstream hypothesis.
    """
    llm = MockLLMClient()
    replanner_node = create_replanner_node(env, llm)

    # Simulate: lockfile was cleared successfully
    env.clear_lock("transform_job.lock")
    # Now run workflow: JOB_02 passes, but JOB_03 fails due to warehouse offline
    env.run_workflow()

    successful_exec = ExecutionRecord(
        action_id="ACT_01_01",
        tool_name="clear_lockfile",
        executed_at="2026-11-24 10:00:00",
        exit_code=0,
        stdout="Removed lockfile",
        stderr="",
        duration_seconds=0.01,
        success=True,
    )

    verif_record = VerificationRecord(
        action_id="ACT_01_01",
        verified_at="2026-11-24 10:00:01",
        passed=False,
        evidence="Action executed, but downstream jobs failed: ['JOB_03_LOAD (Connection refused: reporting_warehouse_service is OFFLINE.)']",
        details={"failed_jobs": ["JOB_03_LOAD (Connection refused: reporting_warehouse_service is OFFLINE.)"]},
    )

    state: NovaGraphState = {
        "incident_id": "INC_REPLAN_02",
        "target_service": "JOB_02_TRANSFORM",
        "raw_symptoms": "Transform job stalled",
        "status": IncidentStatus.FAILED,
        "failure_type": "STALE_LOCKFILE",
        "implicated_resource": "transform_job.lock",
        "triage_summary": "Orphaned lock cleared",
        "root_cause_hypothesis": "Lock cleared",
        "retry_count": 0,
        "max_retries": 3,
        "current_action_index": 1,
        "execution_history": [successful_exec],
        "verification_history": [verif_record],
        "plan": [],
    }

    result = replanner_node(state)

    # Advances to downstream component
    assert result["target_service"] == "JOB_03_LOAD"
    assert result["failure_type"] == "SERVICE_OFFLINE"
    assert "reporting_warehouse_service" in result["implicated_resource"]
    assert result["is_recoverable"] is True
    assert result["status"] == IncidentStatus.REPLANNING


def test_replan_unrecoverable_corruption_escalates(env):
    """
    When verification reveals unrecoverable data corruption during replanning,
    replanner must escalate immediately.
    """
    class CorruptionReplannerLLM(MockLLMClient):
        def replan(self, *args, **kwargs) -> ReplannerResult:
            return ReplannerResult(
                new_target_service="JOB_01_EXTRACT",
                updated_triage_summary="Payload checksum corrupt",
                updated_hypothesis="Zero-byte source records",
                new_failure_type="DATA_CORRUPTION",
                implicated_resource="JOB_01_EXTRACT",
                is_recoverable=False,
                is_escalation=True,
                escalation_reason="Data payload checksum mismatch is unrecoverable by automation.",
            )

    replanner_node = create_replanner_node(env, CorruptionReplannerLLM())

    verif_record = VerificationRecord(
        action_id="ACT_01_01",
        verified_at="2026-11-24 10:00:01",
        passed=False,
        evidence="Checksum mismatch in source feed",
        details={"failed_jobs": ["JOB_01_EXTRACT"]},
    )

    state: NovaGraphState = {
        "incident_id": "INC_REPLAN_CORRUPT",
        "target_service": "JOB_01_EXTRACT",
        "raw_symptoms": "Extract failed",
        "status": IncidentStatus.FAILED,
        "retry_count": 0,
        "max_retries": 3,
        "current_action_index": 1,
        "execution_history": [
            ExecutionRecord(
                action_id="ACT_01_01",
                tool_name="resume_workflow",
                executed_at="2026-11-24 10:00:00",
                exit_code=0,
                stdout="Resumed",
                stderr="",
                duration_seconds=0.01,
                success=True,
            )
        ],
        "verification_history": [verif_record],
    }

    result = replanner_node(state)

    # Escalates cleanly without planning tools
    assert result["status"] == IncidentStatus.ESCALATED
    assert "Data payload checksum mismatch is unrecoverable" in result["final_summary"]
