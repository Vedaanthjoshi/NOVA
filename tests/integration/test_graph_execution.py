import tempfile
import pytest

from nova.environments.mock_workflow_runner import (
    JobStatus,
    MockWorkflowRunner,
    ScenarioType,
)
from nova.graph import build_nova_graph
from nova.state import IncidentStatus, NovaGraphState


@pytest.fixture
def mock_env():
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = MockWorkflowRunner(base_dir=tmpdir)
        yield runner


def test_end_to_end_stale_lockfile_resolution(mock_env: MockWorkflowRunner):
    """
    Milestone 4 & 5 Integration Test:
    Proves that the LangGraph state machine:
    1. Triages the broken job and finds the lockfile in logs.
    2. Plans the clear_lockfile action.
    3. Executes the action safely via ToolRunner.
    4. Independently verifies that the downstream pipeline completes 100%.
    """
    mock_env.inject_scenario(ScenarioType.STALE_LOCKFILE)
    mock_env.run_workflow()  # Halts on JOB_02_TRANSFORM

    # Verify initial failure state
    initial_wf = mock_env.get_state()
    assert initial_wf.jobs["JOB_02_TRANSFORM"].status == JobStatus.FAILED
    assert initial_wf.jobs["JOB_03_LOAD"].status == JobStatus.PENDING

    # Assemble and compile the NOVA LangGraph
    graph = build_nova_graph(mock_env)

    initial_state: NovaGraphState = {
        "incident_id": "INC_TEST_001",
        "target_service": "JOB_02_TRANSFORM",
        "raw_symptoms": "JOB_02_TRANSFORM exited with code 1",
        "status": IncidentStatus.TRIGGERED,
        "triage_summary": None,
        "root_cause_hypothesis": None,
        "plan": [],
        "current_action_index": 0,
        "execution_history": [],
        "verification_history": [],
        "pending_approval": None,
        "retry_count": 0,
        "max_retries": 3,
        "final_summary": None,
    }

    # Execute the complete graph
    final_state = graph.invoke(initial_state)

    # Assertions on final graph state
    assert final_state["status"] == IncidentStatus.RESOLVED
    assert "transform_job.lock" in final_state["triage_summary"]
    assert len(final_state["plan"]) == 2
    assert final_state["plan"][0].tool_name == "clear_lockfile"
    assert final_state["plan"][1].tool_name == "resume_workflow"

    # Verify execution history
    assert len(final_state["execution_history"]) == 2
    assert final_state["execution_history"][0].tool_name == "clear_lockfile"
    assert final_state["execution_history"][0].exit_code == 0
    assert final_state["execution_history"][0].success is True
    assert final_state["execution_history"][1].tool_name == "resume_workflow"
    assert final_state["execution_history"][1].success is True

    # Verify post-condition verification record
    assert len(final_state["verification_history"]) == 1
    assert final_state["verification_history"][0].passed is True
    assert "Read-only state verification passed" in final_state["verification_history"][0].evidence

    # Verify the physical target environment
    final_wf = mock_env.get_state()
    assert all(job.status == JobStatus.SUCCESS for job in final_wf.jobs.values())


def test_end_to_end_service_offline_resolution(mock_env: MockWorkflowRunner):
    """
    Proves that LangGraph diagnoses an offline database service,
    restarts it, and verifies downstream pipeline resolution.
    """
    mock_env.inject_scenario(ScenarioType.SERVICE_OFFLINE)
    mock_env.run_workflow()  # Halts on JOB_03_LOAD

    graph = build_nova_graph(mock_env)

    initial_state: NovaGraphState = {
        "incident_id": "INC_TEST_002",
        "target_service": "JOB_03_LOAD",
        "raw_symptoms": "Connection refused to reporting_warehouse_service",
        "status": IncidentStatus.TRIGGERED,
        "triage_summary": None,
        "root_cause_hypothesis": None,
        "plan": [],
        "current_action_index": 0,
        "execution_history": [],
        "verification_history": [],
        "pending_approval": None,
        "retry_count": 0,
        "max_retries": 3,
        "final_summary": None,
    }

    final_state = graph.invoke(initial_state)

    assert final_state["status"] == IncidentStatus.RESOLVED
    assert final_state["plan"][0].tool_name == "restart_service"
    assert final_state["execution_history"][0].success is True
    assert final_state["verification_history"][0].passed is True

    # Verify warehouse is running
    status = mock_env.check_service_status("reporting_warehouse_service")
    assert status["healthy"] is True


def test_end_to_end_data_corruption_escalation(mock_env: MockWorkflowRunner):
    """
    Proves that when source data is corrupt, NOVA:
    1. Recognizes that automated retries cannot fix corrupt data.
    2. Bypasses tool execution entirely.
    3. Escalates directly to human operators without taking destructive actions.
    """
    mock_env.inject_scenario(ScenarioType.DATA_CORRUPTION)
    mock_env.run_workflow()  # Halts on JOB_01_EXTRACT

    graph = build_nova_graph(mock_env)

    initial_state: NovaGraphState = {
        "incident_id": "INC_TEST_003",
        "target_service": "JOB_01_EXTRACT",
        "raw_symptoms": "JOB_01_EXTRACT exited with code 13",
        "status": IncidentStatus.TRIGGERED,
        "triage_summary": None,
        "root_cause_hypothesis": None,
        "plan": [],
        "current_action_index": 0,
        "execution_history": [],
        "verification_history": [],
        "pending_approval": None,
        "retry_count": 0,
        "max_retries": 3,
        "final_summary": None,
    }

    final_state = graph.invoke(initial_state)

    # Graph must halt in ESCALATED status
    assert final_state["status"] == IncidentStatus.ESCALATED
    # No tools should have been executed!
    assert len(final_state["execution_history"]) == 0
    assert "Incident Escalated" in final_state["final_summary"]


def test_end_to_end_cascading_failure_replan_recovery(mock_env: MockWorkflowRunner):
    """
    Milestone 6 Integration Test:
    Proves that when a cascading failure occurs (stale lock on JOB_02 AND offline DB on JOB_03):
    1. NOVA first plans and executes clear_lockfile.
    2. Deep downstream verification detects that JOB_02 passed, but JOB_03 failed.
    3. Verification failure triggers the Replanner node.
    4. Replanner extracts JOB_03 failure diagnostics, increments retry_count to 1.
    5. Planner synthesizes restart_service for the database.
    6. Executor restarts the database.
    7. Second deep verification succeeds 100%.
    8. Complete execution and verification histories are preserved with reducers.
    """
    mock_env.inject_scenario(ScenarioType.CASCADING_FAILURE)
    mock_env.run_workflow()  # Halts on JOB_02_TRANSFORM

    # Initial state assertions
    initial_wf = mock_env.get_state()
    assert initial_wf.jobs["JOB_02_TRANSFORM"].status == JobStatus.FAILED
    assert initial_wf.jobs["JOB_03_LOAD"].status == JobStatus.PENDING

    graph = build_nova_graph(mock_env)

    initial_state: NovaGraphState = {
        "incident_id": "INC_TEST_004",
        "target_service": "JOB_02_TRANSFORM",
        "raw_symptoms": "JOB_02_TRANSFORM failed due to lock contention",
        "status": IncidentStatus.TRIGGERED,
        "triage_summary": None,
        "root_cause_hypothesis": None,
        "plan": [],
        "current_action_index": 0,
        "execution_history": [],
        "verification_history": [],
        "pending_approval": None,
        "retry_count": 0,
        "max_retries": 3,
        "final_summary": None,
    }

    final_state = graph.invoke(initial_state)

    # Verify overall resolution
    assert final_state["status"] == IncidentStatus.RESOLVED
    assert final_state["retry_count"] == 1

    # Verify execution history contains actions across both cycles (including explicit resume_workflow)
    assert len(final_state["execution_history"]) == 4
    assert final_state["execution_history"][0].tool_name == "clear_lockfile"
    assert final_state["execution_history"][0].success is True
    assert final_state["execution_history"][1].tool_name == "resume_workflow"
    assert final_state["execution_history"][1].success is False
    assert final_state["execution_history"][2].tool_name == "restart_service"
    assert final_state["execution_history"][2].success is True
    assert final_state["execution_history"][3].tool_name == "resume_workflow"
    assert final_state["execution_history"][3].success is True

    # Verify verification history: Cycle 1 failed, Cycle 2 succeeded
    assert len(final_state["verification_history"]) == 2
    assert final_state["verification_history"][0].passed is False
    assert "JOB_03_LOAD" in final_state["verification_history"][0].evidence
    assert final_state["verification_history"][1].passed is True
    assert "Read-only state verification passed" in final_state["verification_history"][1].evidence

    # Verify physical environment reached 100% operational success
    final_wf = mock_env.get_state()
    assert all(job.status == JobStatus.SUCCESS for job in final_wf.jobs.values())


def test_end_to_end_retry_budget_exhaustion(mock_env: MockWorkflowRunner):
    """
    Milestone 6 Integration Test:
    Proves that when a failure occurs and max_retries is reached,
    NOVA deterministically halts, sets status to ESCALATED, and prevents infinite loops.
    """
    mock_env.inject_scenario(ScenarioType.CASCADING_FAILURE)
    mock_env.run_workflow()

    graph = build_nova_graph(mock_env)

    # Configure state with max_retries = 0 so no replan cycles are permitted
    initial_state: NovaGraphState = {
        "incident_id": "INC_TEST_005",
        "target_service": "JOB_02_TRANSFORM",
        "raw_symptoms": "JOB_02_TRANSFORM failed due to lock contention",
        "status": IncidentStatus.TRIGGERED,
        "triage_summary": None,
        "root_cause_hypothesis": None,
        "plan": [],
        "current_action_index": 0,
        "execution_history": [],
        "verification_history": [],
        "pending_approval": None,
        "retry_count": 0,
        "max_retries": 0,  # Zero retry budget!
        "final_summary": None,
    }

    final_state = graph.invoke(initial_state)

    # Must escalate because first action (clear lock) still left downstream broken and retries=0
    assert final_state["status"] == IncidentStatus.ESCALATED
    assert final_state["retry_count"] == 0
    assert len(final_state["execution_history"]) == 2
    assert len(final_state["verification_history"]) == 1
    assert final_state["verification_history"][0].passed is False
    assert "Maximum retry limit" in final_state["final_summary"]
