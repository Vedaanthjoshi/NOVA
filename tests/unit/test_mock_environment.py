import tempfile
from pathlib import Path
import pytest

from nova.environments.mock_workflow_runner import (
    JobStatus,
    MockWorkflowRunner,
    ScenarioType,
)


@pytest.fixture
def mock_env():
    """Provides an isolated temporary mock environment for each test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = MockWorkflowRunner(base_dir=tmpdir)
        yield runner


def test_happy_path_execution(mock_env: MockWorkflowRunner):
    """Verifies that all 4 jobs execute successfully when no faults exist."""
    state = mock_env.inject_scenario(ScenarioType.HAPPY_PATH)
    assert state.workflow_id == "WF_NIGHTLY_BATCH_1001"

    final_state = mock_env.run_workflow()
    assert all(job.status == JobStatus.SUCCESS for job in final_state.jobs.values())
    assert final_state.jobs["JOB_04_REPORT"].exit_code == 0


def test_stale_lockfile_failure_and_remediation(mock_env: MockWorkflowRunner):
    """
    Verifies that STALE_LOCKFILE stops at JOB_02_TRANSFORM,
    and after clearing the lock, re-running completes the remaining jobs.
    """
    mock_env.inject_scenario(ScenarioType.STALE_LOCKFILE)

    # Initial run should fail at Job 2
    state = mock_env.run_workflow()
    assert state.jobs["JOB_01_EXTRACT"].status == JobStatus.SUCCESS
    assert state.jobs["JOB_02_TRANSFORM"].status == JobStatus.FAILED
    assert "transform_job.lock" in state.jobs["JOB_02_TRANSFORM"].error_message
    assert state.jobs["JOB_03_LOAD"].status == JobStatus.PENDING

    # Verify log output was written to disk
    log_content = mock_env.read_job_log("JOB_02_TRANSFORM")
    assert "FATAL: Lockfile 'transform_job.lock' is present" in log_content

    # Simulate remediation: clear lock
    cleared = mock_env.clear_lock("transform_job.lock")
    assert cleared is True

    # Re-run workflow from where it stopped
    state_after_remediation = mock_env.run_workflow()
    assert state_after_remediation.jobs["JOB_02_TRANSFORM"].status == JobStatus.SUCCESS
    assert state_after_remediation.jobs["JOB_03_LOAD"].status == JobStatus.SUCCESS
    assert state_after_remediation.jobs["JOB_04_REPORT"].status == JobStatus.SUCCESS


def test_service_offline_failure_and_remediation(mock_env: MockWorkflowRunner):
    """
    Verifies that SERVICE_OFFLINE stops at JOB_03_LOAD,
    and after restarting the service, re-running completes the workflow.
    """
    mock_env.inject_scenario(ScenarioType.SERVICE_OFFLINE)

    # Initial run should fail at Job 3
    state = mock_env.run_workflow()
    assert state.jobs["JOB_01_EXTRACT"].status == JobStatus.SUCCESS
    assert state.jobs["JOB_02_TRANSFORM"].status == JobStatus.SUCCESS
    assert state.jobs["JOB_03_LOAD"].status == JobStatus.FAILED
    assert "reporting_warehouse_service is OFFLINE" in state.jobs["JOB_03_LOAD"].error_message

    # Check service health API
    health = mock_env.check_service_status("reporting_warehouse_service")
    assert health["status"] == "STOPPED"
    assert health["healthy"] is False

    # Simulate remediation: restart service
    restarted = mock_env.restart_service("reporting_warehouse_service")
    assert restarted is True

    health_after = mock_env.check_service_status("reporting_warehouse_service")
    assert health_after["status"] == "RUNNING"
    assert health_after["healthy"] is True

    # Re-run workflow
    state_after_remediation = mock_env.run_workflow()
    assert state_after_remediation.jobs["JOB_03_LOAD"].status == JobStatus.SUCCESS
    assert state_after_remediation.jobs["JOB_04_REPORT"].status == JobStatus.SUCCESS


def test_data_corruption_unrecoverable(mock_env: MockWorkflowRunner):
    """
    Verifies that DATA_CORRUPTION immediately fails at JOB_01_EXTRACT,
    producing an unrecoverable error signature.
    """
    mock_env.inject_scenario(ScenarioType.DATA_CORRUPTION)

    state = mock_env.run_workflow()
    assert state.jobs["JOB_01_EXTRACT"].status == JobStatus.FAILED
    assert state.jobs["JOB_01_EXTRACT"].exit_code == 13
    assert "checksum mismatch" in state.jobs["JOB_01_EXTRACT"].error_message
    assert state.jobs["JOB_02_TRANSFORM"].status == JobStatus.PENDING

