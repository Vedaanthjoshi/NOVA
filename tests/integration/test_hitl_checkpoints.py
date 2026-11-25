import gc
import os
import tempfile
import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from nova.environments.mock_workflow_runner import (
    JobStatus,
    MockWorkflowRunner,
    ScenarioType,
)
from nova.graph import build_nova_graph
from nova.state import IncidentStatus, NovaGraphState
from nova.storage.sqlite_saver import SqliteSaver


@pytest.fixture
def mock_env():
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = MockWorkflowRunner(base_dir=tmpdir)
        yield runner


def test_hitl_destructive_action_interrupts_and_awaits_approval(mock_env: MockWorkflowRunner):
    """
    Milestone 7 Test 1:
    Proves that for DESTRUCTIVE actions:
    1. The graph automatically halts execution BEFORE the tool runs.
    2. The target lockfile is untouched on disk while paused.
    3. The interrupt payload contains the full approval context.
    4. Execution history remains empty while awaiting human authorization.
    """
    mock_env.inject_scenario(ScenarioType.STALE_LOCKFILE)
    mock_env.run_workflow()

    checkpointer = MemorySaver()
    graph = build_nova_graph(mock_env, checkpointer=checkpointer, require_approval=True)

    thread_config = {"configurable": {"thread_id": "INC_HITL_001"}}

    initial_state: NovaGraphState = {
        "incident_id": "INC_HITL_001",
        "target_service": "JOB_02_TRANSFORM",
        "raw_symptoms": "Stale lockfile detected",
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

    # Execute graph: must halt on interrupt at executor node
    graph.invoke(initial_state, thread_config)

    # Inspect paused state from checkpointer
    checkpoint_state = graph.get_state(thread_config)
    assert checkpoint_state.next == ("executor",)
    assert len(checkpoint_state.tasks) == 1
    assert len(checkpoint_state.tasks[0].interrupts) == 1

    interrupt_payload = checkpoint_state.tasks[0].interrupts[0].value
    assert interrupt_payload["tool_name"] == "clear_lockfile"
    assert interrupt_payload["risk_level"] == "DESTRUCTIVE"
    assert "transform_job.lock" in interrupt_payload["parameters"]["lock_name"]

    # Verify physical environment: lockfile MUST still exist on disk!
    lock_file = mock_env.locks_dir / "transform_job.lock"
    assert lock_file.exists()

    # Verify state history: zero tools executed yet
    assert len(checkpoint_state.values.get("execution_history", [])) == 0


def test_hitl_destructive_action_approved_and_resumed(mock_env: MockWorkflowRunner):
    """
    Milestone 7 Test 2:
    Proves that upon human approval:
    1. The graph resumes cleanly from the exact paused node.
    2. The destructive tool executes and clears the lockfile.
    3. The graph proceeds through independent verification to RESOLVED.
    """
    mock_env.inject_scenario(ScenarioType.STALE_LOCKFILE)
    mock_env.run_workflow()

    checkpointer = MemorySaver()
    graph = build_nova_graph(mock_env, checkpointer=checkpointer, require_approval=True)
    thread_config = {"configurable": {"thread_id": "INC_HITL_002"}}

    initial_state: NovaGraphState = {
        "incident_id": "INC_HITL_002",
        "target_service": "JOB_02_TRANSFORM",
        "raw_symptoms": "Stale lockfile detected",
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

    # Step 1: Run until interrupt
    graph.invoke(initial_state, thread_config)
    assert (mock_env.locks_dir / "transform_job.lock").exists()

    # Step 2: Operator approves via Command(resume=...)
    final_state = graph.invoke(
        Command(resume={"approved": True, "notes": "Approved by SRE on call"}),
        thread_config,
    )

    # Step 3: Assert full successful resolution
    assert final_state["status"] == IncidentStatus.RESOLVED
    assert len(final_state["execution_history"]) == 1
    assert final_state["execution_history"][0].tool_name == "clear_lockfile"
    assert final_state["execution_history"][0].success is True
    assert final_state["verification_history"][0].passed is True

    # Verify physical environment
    assert not (mock_env.locks_dir / "transform_job.lock").exists()
    final_wf = mock_env.get_state()
    assert all(job.status == JobStatus.SUCCESS for job in final_wf.jobs.values())


def test_hitl_destructive_action_rejection_halts_safely(mock_env: MockWorkflowRunner):
    """
    Milestone 7 Test 3:
    Proves that upon operator rejection:
    1. The destructive tool is NEVER executed.
    2. The lockfile remains intact on disk.
    3. The incident transitions immediately to ESCALATED.
    """
    mock_env.inject_scenario(ScenarioType.STALE_LOCKFILE)
    mock_env.run_workflow()

    checkpointer = MemorySaver()
    graph = build_nova_graph(mock_env, checkpointer=checkpointer, require_approval=True)
    thread_config = {"configurable": {"thread_id": "INC_HITL_003"}}

    initial_state: NovaGraphState = {
        "incident_id": "INC_HITL_003",
        "target_service": "JOB_02_TRANSFORM",
        "raw_symptoms": "Stale lockfile detected",
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

    # Step 1: Run until interrupt
    graph.invoke(initial_state, thread_config)

    # Step 2: Operator rejects the action
    final_state = graph.invoke(
        Command(resume={"approved": False, "notes": "Hold for maintenance window"}),
        thread_config,
    )

    # Step 3: Assert fail-closed safety
    assert final_state["status"] == IncidentStatus.ESCALATED
    assert len(final_state["execution_history"]) == 0  # Tool was NEVER executed!
    assert "REJECTED by human operator" in final_state["final_summary"]

    # Verify physical environment is untouched
    assert (mock_env.locks_dir / "transform_job.lock").exists()


def test_hitl_sqlite_persistence_across_process_restart(mock_env: MockWorkflowRunner):
    """
    Milestone 7 Test 4:
    Proves that:
    1. An incident paused on a HITL gate persists all checkpoints to SQLite on disk.
    2. Simulating a process crash / restart by instantiating a brand-new SqliteSaver
       and graph instance successfully reloads the paused state from SQLite.
    3. Resuming on the fresh instance executes cleanly to resolution.
    """
    mock_env.inject_scenario(ScenarioType.STALE_LOCKFILE)
    mock_env.run_workflow()

    with tempfile.TemporaryDirectory() as db_dir:
        db_path = os.path.join(db_dir, "checkpoints.db")
        thread_config = {"configurable": {"thread_id": "INC_HITL_004"}}

        # Phase 1: Process 1 runs and interrupts on destructive action
        saver1 = SqliteSaver(db_path)
        graph1 = build_nova_graph(mock_env, checkpointer=saver1, require_approval=True)

        initial_state: NovaGraphState = {
            "incident_id": "INC_HITL_004",
            "target_service": "JOB_02_TRANSFORM",
            "raw_symptoms": "Stale lockfile detected",
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

        graph1.invoke(initial_state, thread_config)

        # Assert database file exists on disk with checkpoint content
        assert os.path.exists(db_path)
        assert os.path.getsize(db_path) > 0

        # Simulate complete process termination
        del graph1
        saver1.close()
        del saver1
        gc.collect()

        # Phase 2: Fresh Process 2 initializes from SQLite database on disk
        saver2 = SqliteSaver(db_path)
        graph2 = build_nova_graph(mock_env, checkpointer=saver2, require_approval=True)

        # Query state from disk on the new instance
        reloaded_state = graph2.get_state(thread_config)
        assert reloaded_state.next == ("executor",)
        assert len(reloaded_state.tasks) == 1
        assert len(reloaded_state.tasks[0].interrupts) == 1
        assert reloaded_state.tasks[0].interrupts[0].value["tool_name"] == "clear_lockfile"

        # Resume execution on the fresh process
        final_state = graph2.invoke(
            Command(resume={"approved": True, "notes": "Authorized post-restart"}),
            thread_config,
        )

        assert final_state["status"] == IncidentStatus.RESOLVED
        assert len(final_state["execution_history"]) == 1
        assert final_state["verification_history"][0].passed is True

        saver2.close()
        del saver2
        gc.collect()
