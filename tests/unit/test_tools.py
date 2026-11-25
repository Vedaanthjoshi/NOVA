import tempfile
import pytest

from nova.environments.mock_workflow_runner import (
    MockWorkflowRunner,
    ScenarioType,
)
from nova.state import PlannedAction, RiskLevel
from nova.tools.base import UnknownToolError
from nova.tools.runner import ToolRunner, create_default_tool_runner


@pytest.fixture
def mock_env():
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = MockWorkflowRunner(base_dir=tmpdir)
        yield runner


@pytest.fixture
def tool_runner(mock_env: MockWorkflowRunner):
    return create_default_tool_runner(mock_env)


def test_tool_runner_registration(tool_runner: ToolRunner):
    """Verifies all expected enterprise tools are registered in the whitelist."""
    expected_tools = {
        "read_job_logs",
        "check_service_status",
        "clear_lockfile",
        "restart_service",
        "resume_workflow",
    }
    for tool_name in expected_tools:
        assert tool_runner.is_tool_allowed(tool_name) is True
        assert tool_runner.get_tool(tool_name).name == tool_name


def test_read_job_logs_execution(mock_env: MockWorkflowRunner, tool_runner: ToolRunner):
    """Verifies read_job_logs successfully retrieves log text from disk."""
    mock_env.inject_scenario(ScenarioType.STALE_LOCKFILE)
    mock_env.run_workflow()  # Generates log entries

    action = PlannedAction(
        action_id="ACT_01",
        tool_name="read_job_logs",
        parameters={"job_id": "JOB_02_TRANSFORM", "tail_lines": 10},
        risk_level=RiskLevel.READ_ONLY,
        rationale="Read failure logs",
        expected_outcome="Log lines returned",
    )

    record = tool_runner.execute(action)
    assert record.exit_code == 0
    assert record.success is True
    assert "transform_job.lock" in record.stdout
    assert record.duration_seconds >= 0.0


def test_check_service_status_execution(mock_env: MockWorkflowRunner, tool_runner: ToolRunner):
    """
    Verifies service status probe succeeds (exit_code=0) while accurately reporting
    STOPPED/unhealthy vs. RUNNING/healthy service state in stdout.
    """
    mock_env.inject_scenario(ScenarioType.SERVICE_OFFLINE)

    action = PlannedAction(
        action_id="ACT_02",
        tool_name="check_service_status",
        parameters={"service_name": "reporting_warehouse_service"},
        risk_level=RiskLevel.READ_ONLY,
        rationale="Check DB status",
        expected_outcome="Service status observed",
    )

    record = tool_runner.execute(action)
    assert record.exit_code == 0
    assert record.success is True
    assert "STOPPED" in record.stdout
    assert "Healthy: False" in record.stdout

    mock_env.restart_service("reporting_warehouse_service")
    record_healthy = tool_runner.execute(action)
    assert record_healthy.exit_code == 0
    assert record_healthy.success is True
    assert "RUNNING" in record_healthy.stdout
    assert "Healthy: True" in record_healthy.stdout


def test_clear_lockfile_execution(mock_env: MockWorkflowRunner, tool_runner: ToolRunner):
    """Verifies destructive clear_lockfile tool physically deletes the lockfile."""
    mock_env.inject_scenario(ScenarioType.STALE_LOCKFILE)
    lock_file = mock_env.locks_dir / "transform_job.lock"
    assert lock_file.exists() is True

    action = PlannedAction(
        action_id="ACT_03",
        tool_name="clear_lockfile",
        parameters={"lock_name": "transform_job.lock"},
        risk_level=RiskLevel.DESTRUCTIVE,
        rationale="Delete orphaned lock",
        expected_outcome="Lockfile removed",
    )

    record = tool_runner.execute(action)
    assert record.exit_code == 0
    assert record.success is True
    assert lock_file.exists() is False


def test_unknown_tool_rejection(tool_runner: ToolRunner):
    """
    CRITICAL SECURITY TEST:
    Verifies that any tool not in the whitelist is rejected immediately.
    """
    action = PlannedAction(
        action_id="ACT_HACK",
        tool_name="drop_all_tables",
        parameters={"db": "production"},
        risk_level=RiskLevel.DESTRUCTIVE,
        rationale="Malicious or hallucinated tool",
        expected_outcome="Exploit",
    )

    with pytest.raises(UnknownToolError) as exc_info:
        tool_runner.execute(action)

    assert "Security Alert: Tool 'drop_all_tables' is not in the approved whitelist registry" in str(exc_info.value)


def test_parameter_validation_failure(tool_runner: ToolRunner):
    """
    Verifies that malformed arguments (e.g. invalid type or negative bounds)
    are caught cleanly by Pydantic before tool logic executes.
    """
    action = PlannedAction(
        action_id="ACT_BAD_PARAM",
        tool_name="read_job_logs",
        parameters={"job_id": "JOB_01_EXTRACT", "tail_lines": -5},  # Invalid: ge=1 constraint
        risk_level=RiskLevel.READ_ONLY,
        rationale="Invalid argument test",
        expected_outcome="Should fail validation",
    )

    record = tool_runner.execute(action)
    assert record.exit_code == 126
    assert record.success is False
    assert "Parameter Validation Error" in record.stderr

