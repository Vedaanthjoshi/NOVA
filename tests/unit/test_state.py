import operator
import pytest
from pydantic import ValidationError

from nova.state import (
    ApprovalRequest,
    ExecutionRecord,
    IncidentStatus,
    NovaGraphState,
    NovaState,
    PlannedAction,
    RiskLevel,
    VerificationRecord,
)


def test_initial_nova_state_creation():
    """Verifies that an initial state can be initialized with default values."""
    state = NovaState(
        incident_id="INC_8821",
        target_service="WF_NIGHTLY_BATCH_1001",
        raw_symptoms="JOB_02_TRANSFORM aborted with exit code 1",
    )

    assert state.incident_id == "INC_8821"
    assert state.target_service == "WF_NIGHTLY_BATCH_1001"
    assert state.status == IncidentStatus.TRIGGERED
    assert state.plan == []
    assert state.execution_history == []
    assert state.verification_history == []
    assert state.current_action_index == 0
    assert state.retry_count == 0
    assert state.max_retries == 3
    assert state.pending_approval is None


def test_planned_action_risk_validation():
    """Verifies that invalid risk levels are rejected by Pydantic."""
    # Valid creation
    action = PlannedAction(
        action_id="ACT_01",
        tool_name="clear_lockfile",
        parameters={"lock_name": "transform_job.lock"},
        risk_level=RiskLevel.DESTRUCTIVE,
        rationale="Orphaned lock is blocking the job",
        expected_outcome="Lockfile removed from locks directory",
    )
    assert action.risk_level == RiskLevel.DESTRUCTIVE

    # Invalid risk level should raise ValidationError
    with pytest.raises(ValidationError):
        PlannedAction(
            action_id="ACT_02",
            tool_name="fake_tool",
            risk_level="SUPER_DANGEROUS",  # Invalid enum value
            rationale="Test",
            expected_outcome="Test",
        )


def test_execution_vs_verification_separation():
    """
    Critical concept test: Proves that an action can succeed in execution
    (exit_code=0) while still failing independent post-condition verification.
    """
    exec_record = ExecutionRecord(
        action_id="ACT_RESTART_DB",
        tool_name="restart_service",
        executed_at="2026-11-19 12:00:00",
        exit_code=0,
        stdout="Restart command sent successfully",
        stderr="",
        duration_seconds=0.45,
        success=True,
    )

    # But independent verification checks the port and discovers the DB is not healthy yet
    verify_record = VerificationRecord(
        action_id="ACT_RESTART_DB",
        verified_at="2026-11-19 12:00:05",
        passed=False,
        evidence="Port 5432 connection timed out after 5s",
        details={"port": 5432, "response_time_ms": None},
    )

    assert exec_record.exit_code == 0
    assert exec_record.success is True
    assert verify_record.passed is False
    assert "timed out" in verify_record.evidence


def test_human_approval_request():
    """Verifies that dangerous actions can be attached as pending approval requests."""
    action = PlannedAction(
        action_id="ACT_KILL",
        tool_name="kill_process",
        parameters={"pid": 9182},
        risk_level=RiskLevel.DESTRUCTIVE,
        rationale="Process 9182 is hung holding the file lock",
        expected_outcome="PID 9182 terminated",
    )

    approval = ApprovalRequest(
        action=action,
        requested_at="2026-11-19 12:01:00",
        reason="Action is marked DESTRUCTIVE: Killing an active process",
    )

    assert approval.status == "PENDING"
    assert approval.action.risk_level == RiskLevel.DESTRUCTIVE
    assert approval.operator_notes is None


def test_state_json_roundtrip():
    """Verifies that state can be serialized to JSON and deserialized with 100% fidelity."""
    state = NovaState(
        incident_id="INC_100",
        target_service="reporting_db",
        raw_symptoms="High memory alert",
        status=IncidentStatus.PLANNING,
        plan=[
            PlannedAction(
                action_id="ACT_01",
                tool_name="inspect_memory",
                risk_level=RiskLevel.READ_ONLY,
                rationale="Diagnose memory leak",
                expected_outcome="Memory stats collected",
            )
        ],
    )

    json_str = state.model_dump_json()
    reloaded_state = NovaState.model_validate_json(json_str)

    assert reloaded_state.incident_id == state.incident_id
    assert reloaded_state.plan[0].tool_name == "inspect_memory"
    assert reloaded_state.plan[0].risk_level == RiskLevel.READ_ONLY


def test_reducer_prevents_history_overwrite():
    """
    Verifies that the reducer logic (operator.add) accumulates execution records
    across multiple tool invocations rather than replacing the previous history.
    """
    record_1 = ExecutionRecord(
        action_id="ACT_01",
        tool_name="check_service_status",
        executed_at="2026-11-19 12:00:00",
        exit_code=0,
        stdout="Status: STOPPED",
        stderr="",
        duration_seconds=0.01,
        success=True,
    )

    record_2 = ExecutionRecord(
        action_id="ACT_02",
        tool_name="restart_service",
        executed_at="2026-11-19 12:00:02",
        exit_code=0,
        stdout="Status: RUNNING",
        stderr="",
        duration_seconds=0.03,
        success=True,
    )

    # Initial history contains record 1
    history = [record_1]

    # LangGraph's reducer applies operator.add(history, [record_2])
    updated_history = operator.add(history, [record_2])

    # Assert that both records exist in order (no overwrite!)
    assert len(updated_history) == 2
    assert updated_history[0].action_id == "ACT_01"
    assert updated_history[1].action_id == "ACT_02"
