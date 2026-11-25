"""
Integration Tests for Multi-Step Plan Execution (Milestone 7.2 Phase B).

Verifies that NOVA can execute arbitrary ordered multi-step plans:
1. Ordered sequential execution of multiple planned actions.
2. Accurate tracking and advancement of current_action_index.
3. Halting remaining plan actions immediately when an action fails.
4. Independent validation, ground-truth risk lookup, and authorization for every action.
5. Intermediate Human-in-the-Loop approval checkpoints and state persistence across interruptions.
6. Immediate safe escalation and halt when an intermediate action is rejected by an operator.
7. Final workflow verification executing only after all plan actions complete.
"""
import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from nova.environments.mock_workflow_runner import (
    MockWorkflowRunner,
    ScenarioType,
)
from nova.graph import build_nova_graph
from nova.llm.client import MockLLMClient
from nova.llm.schemas import PlannedActionSchema, PlannerResult, TriageResult
from nova.state import (
    ApprovalDecision,
    IncidentStatus,
    NovaGraphState,
    PlannedAction,
    RiskLevel,
)
from nova.tools.runner import create_default_tool_runner


class MultiStepMockLLM(MockLLMClient):
    """Custom Mock LLM producing arbitrary ordered multi-step plans."""

    def __init__(self, plan_actions: list[PlannedActionSchema]):
        super().__init__()
        self._custom_actions = plan_actions

    def plan(self, triage, available_tools, retry_count: int) -> PlannerResult:
        return PlannerResult(
            is_escalation=False,
            actions=self._custom_actions,
        )


@pytest.fixture
def env(tmp_path):
    runner = MockWorkflowRunner(base_dir=tmp_path / "mock_state")
    runner.inject_scenario(ScenarioType.CASCADING_FAILURE)
    runner.run_workflow()  # Populate initial failure state & logs on disk
    return runner


def test_multistep_plan_executes_in_correct_order(env):
    """
    Given a 3-step plan:
      1. inspect_lockfiles (READ_ONLY)
      2. clear_lockfile (DESTRUCTIVE - require_approval=False)
      3. restart_service (DESTRUCTIVE - require_approval=False)
    NOVA must execute all 3 in exact sequential order, advance current_action_index
    from 0 -> 1 -> 2 -> 3, record all 3 in execution_history, and finally verify
    the workflow resolves cleanly.
    """
    actions = [
        PlannedActionSchema(
            tool_name="inspect_lockfiles",
            parameters={"lock_name": "transform_job.lock"},
            risk_level=RiskLevel.READ_ONLY,
            rationale="Verify lockfile presence on disk before removal",
            expected_outcome="Lockfile verified",
        ),
        PlannedActionSchema(
            tool_name="clear_lockfile",
            parameters={"lock_name": "transform_job.lock"},
            risk_level=RiskLevel.DESTRUCTIVE,
            rationale="Clear orphaned transform lockfile",
            expected_outcome="Lockfile removed",
        ),
        PlannedActionSchema(
            tool_name="restart_service",
            parameters={"service_name": "reporting_warehouse_service"},
            risk_level=RiskLevel.DESTRUCTIVE,
            rationale="Restart reporting warehouse database",
            expected_outcome="Service is operational",
        ),
    ]

    llm = MultiStepMockLLM(plan_actions=actions)
    graph = build_nova_graph(env, llm=llm, require_approval=False)

    initial_state: NovaGraphState = {
        "incident_id": "INC_MULTI_01",
        "target_service": "JOB_02_TRANSFORM",
        "raw_symptoms": "Cascading pipeline blockage",
        "status": IncidentStatus.TRIGGERED,
        "retry_count": 0,
        "max_retries": 3,
        "current_action_index": 0,
        "execution_history": [],
        "verification_history": [],
    }

    result = graph.invoke(initial_state)

    # 1. State must be fully resolved
    assert result["status"] == IncidentStatus.RESOLVED

    # 2. Exactly 3 actions executed
    exec_history = result["execution_history"]
    assert len(exec_history) == 3

    # 3. Actions executed in exact declared order
    assert exec_history[0].tool_name == "inspect_lockfiles"
    assert exec_history[0].action_id == "ACT_01_01"
    assert exec_history[0].success is True

    assert exec_history[1].tool_name == "clear_lockfile"
    assert exec_history[1].action_id == "ACT_01_02"
    assert exec_history[1].success is True

    assert exec_history[2].tool_name == "restart_service"
    assert exec_history[2].action_id == "ACT_01_03"
    assert exec_history[2].success is True

    # 4. current_action_index reached 3
    assert result["current_action_index"] == 3

    # 5. Verification history confirms resolution
    assert len(result["verification_history"]) >= 1
    assert result["verification_history"][-1].passed is True


def test_multistep_plan_halts_remaining_actions_on_action_failure(env):
    """
    If an action in a multi-step plan fails execution (e.g. invalid parameter),
    the executor MUST NOT advance current_action_index and MUST NOT execute
    any remaining plan actions.
    """
    actions = [
        PlannedActionSchema(
            tool_name="inspect_lockfiles",
            parameters={"lock_name": "transform_job.lock"},
            risk_level=RiskLevel.READ_ONLY,
            rationale="Initial status probe",
            expected_outcome="Status reported",
        ),
        # Action 2 fails with parameter validation error
        PlannedActionSchema(
            tool_name="read_job_logs",
            parameters={"job_id": "JOB_02_TRANSFORM", "tail_lines": -999},  # ge=1 violation
            risk_level=RiskLevel.READ_ONLY,
            rationale="Read logs with invalid parameter",
            expected_outcome="Fails validation",
        ),
        # Action 3 should NEVER execute
        PlannedActionSchema(
            tool_name="clear_lockfile",
            parameters={"lock_name": "transform_job.lock"},
            risk_level=RiskLevel.DESTRUCTIVE,
            rationale="Should not be reached",
            expected_outcome="Not executed",
        ),
    ]

    llm = MultiStepMockLLM(plan_actions=actions)
    # Set max_retries=0 to force immediate escalation on failure
    graph = build_nova_graph(env, llm=llm, require_approval=False)

    initial_state: NovaGraphState = {
        "incident_id": "INC_MULTI_FAIL_01",
        "target_service": "JOB_02_TRANSFORM",
        "raw_symptoms": "Test parameter failure",
        "status": IncidentStatus.TRIGGERED,
        "retry_count": 0,
        "max_retries": 0,  # Exhausted immediately
        "current_action_index": 0,
        "execution_history": [],
        "verification_history": [],
    }

    result = graph.invoke(initial_state)

    # Execution history has exactly 2 records (Action 1 succeeded, Action 2 failed)
    exec_history = result["execution_history"]
    assert len(exec_history) == 2
    assert exec_history[0].tool_name == "inspect_lockfiles"
    assert exec_history[0].success is True

    assert exec_history[1].tool_name == "read_job_logs"
    assert exec_history[1].success is False

    # Action 3 was NEVER executed
    tool_names = [e.tool_name for e in exec_history]
    assert "clear_lockfile" not in tool_names

    # Graph escalated safely
    assert result["status"] == IncidentStatus.ESCALATED


def test_multistep_plan_hitl_checkpoints_and_intermediate_resumption(env):
    """
    Test multi-step execution with intermediate HITL checkpoints:
      Action 1: inspect_lockfiles (READ_ONLY -> executes without interrupt)
      Action 2: clear_lockfile (DESTRUCTIVE -> interrupts, awaits approval)
      Action 3: restart_service (DESTRUCTIVE -> interrupts on next step, awaits approval)

    Proves:
      - Graph pauses at Action 2.
      - Checkpoint persists state with current_action_index = 1.
      - Resuming with ApprovalDecision executes Action 2.
      - Graph then pauses at Action 3.
      - Resuming with ApprovalDecision executes Action 3.
      - Full resolution verified.
    """
    actions = [
        PlannedActionSchema(
            tool_name="inspect_lockfiles",
            parameters={"lock_name": "transform_job.lock"},
            risk_level=RiskLevel.READ_ONLY,
            rationale="Initial check",
            expected_outcome="Status reported",
        ),
        PlannedActionSchema(
            tool_name="clear_lockfile",
            parameters={"lock_name": "transform_job.lock"},
            risk_level=RiskLevel.DESTRUCTIVE,
            rationale="Clear lock",
            expected_outcome="Lock removed",
        ),
        PlannedActionSchema(
            tool_name="restart_service",
            parameters={"service_name": "reporting_warehouse_service"},
            risk_level=RiskLevel.DESTRUCTIVE,
            rationale="Restart database",
            expected_outcome="Database online",
        ),
    ]

    llm = MultiStepMockLLM(plan_actions=actions)
    checkpointer = MemorySaver()
    graph = build_nova_graph(env, llm=llm, checkpointer=checkpointer, require_approval=True)

    thread_config = {"configurable": {"thread_id": "thread_multistep_hitl_01"}}

    initial_state: NovaGraphState = {
        "incident_id": "INC_HITL_MULTI_01",
        "target_service": "JOB_02_TRANSFORM",
        "raw_symptoms": "Multiple component failure requiring approval",
        "status": IncidentStatus.TRIGGERED,
        "retry_count": 0,
        "max_retries": 3,
        "current_action_index": 0,
        "execution_history": [],
        "verification_history": [],
    }

    # Step 1: Run until first interrupt (at Action 2)
    graph.invoke(initial_state, config=thread_config)

    snapshot_1 = graph.get_state(thread_config)
    assert len(snapshot_1.tasks) > 0
    assert len(snapshot_1.tasks[0].interrupts) > 0

    interrupt_payload_1 = snapshot_1.tasks[0].interrupts[0].value
    assert interrupt_payload_1["action_id"] == "ACT_01_02"
    assert interrupt_payload_1["tool_name"] == "clear_lockfile"

    # Action 1 already executed
    assert len(snapshot_1.values["execution_history"]) == 1
    assert snapshot_1.values["execution_history"][0].tool_name == "inspect_lockfiles"
    assert snapshot_1.values["current_action_index"] == 1

    # Step 2: Approve Action 2 -> Resume graph
    approval_1 = ApprovalDecision(
        approved=True,
        approver_id="sre_alice",
        approver_role="SRE_LEAD",
        decision_timestamp="2026-11-24 10:00:00",
        incident_id="INC_HITL_MULTI_01",
        action_id="ACT_01_02",
        approval_request_id=interrupt_payload_1["approval_request_id"],
    )

    graph.invoke(Command(resume=approval_1), config=thread_config)

    # Step 3: Graph should now pause at Action 3!
    snapshot_2 = graph.get_state(thread_config)
    assert len(snapshot_2.tasks) > 0
    assert len(snapshot_2.tasks[0].interrupts) > 0

    interrupt_payload_2 = snapshot_2.tasks[0].interrupts[0].value
    assert interrupt_payload_2["action_id"] == "ACT_01_03"
    assert interrupt_payload_2["tool_name"] == "restart_service"

    # Action 1 and Action 2 have executed
    assert len(snapshot_2.values["execution_history"]) == 2
    assert snapshot_2.values["execution_history"][1].tool_name == "clear_lockfile"
    assert snapshot_2.values["current_action_index"] == 2

    # Step 4: Approve Action 3 -> Resume graph
    approval_2 = ApprovalDecision(
        approved=True,
        approver_id="admin_bob",
        approver_role="PLATFORM_ADMIN",
        decision_timestamp="2026-11-24 10:01:00",
        incident_id="INC_HITL_MULTI_01",
        action_id="ACT_01_03",
        approval_request_id=interrupt_payload_2["approval_request_id"],
    )

    final_result = graph.invoke(Command(resume=approval_2), config=thread_config)

    # All 3 actions completed and incident resolved
    assert final_result["status"] == IncidentStatus.RESOLVED
    assert len(final_result["execution_history"]) == 3
    assert final_result["current_action_index"] == 3


def test_multistep_plan_intermediate_rejection_halts_without_subsequent_actions(env):
    """
    If human operator rejects Action 2 in a 3-step plan:
      - Execution terminates immediately.
      - Action 3 is NEVER executed.
      - Graph transitions to ESCALATED.
    """
    actions = [
        PlannedActionSchema(
            tool_name="inspect_lockfiles",
            parameters={"lock_name": "transform_job.lock"},
            risk_level=RiskLevel.READ_ONLY,
            rationale="Initial status check",
            expected_outcome="Status reported",
        ),
        PlannedActionSchema(
            tool_name="clear_lockfile",
            parameters={"lock_name": "transform_job.lock"},
            risk_level=RiskLevel.DESTRUCTIVE,
            rationale="Clear lock",
            expected_outcome="Lock removed",
        ),
        PlannedActionSchema(
            tool_name="restart_service",
            parameters={"service_name": "reporting_warehouse_service"},
            risk_level=RiskLevel.DESTRUCTIVE,
            rationale="Restart database",
            expected_outcome="Database online",
        ),
    ]

    llm = MultiStepMockLLM(plan_actions=actions)
    checkpointer = MemorySaver()
    graph = build_nova_graph(env, llm=llm, checkpointer=checkpointer, require_approval=True)

    thread_config = {"configurable": {"thread_id": "thread_multistep_rejection_01"}}

    initial_state: NovaGraphState = {
        "incident_id": "INC_REJECT_MULTI_01",
        "target_service": "JOB_02_TRANSFORM",
        "raw_symptoms": "Cascading failure",
        "status": IncidentStatus.TRIGGERED,
        "retry_count": 0,
        "max_retries": 3,
        "current_action_index": 0,
        "execution_history": [],
        "verification_history": [],
    }

    # Run until Action 2 interrupts
    graph.invoke(initial_state, config=thread_config)

    snapshot = graph.get_state(thread_config)
    interrupt_payload = snapshot.tasks[0].interrupts[0].value

    # Reject Action 2
    rejection = ApprovalDecision(
        approved=False,
        approver_id="sre_charlie",
        approver_role="SRE_LEAD",
        decision_timestamp="2026-11-24 10:05:00",
        incident_id="INC_REJECT_MULTI_01",
        action_id="ACT_01_02",
        approval_request_id=interrupt_payload["approval_request_id"],
        notes="Lockfile removal rejected pending database dump",
    )

    result = graph.invoke(Command(resume=rejection), config=thread_config)

    # 1. Status is ESCALATED
    assert result["status"] == IncidentStatus.ESCALATED
    assert "REJECTED by human operator" in result["final_summary"]

    # 2. Only Action 1 was executed; Action 2 was rejected; Action 3 was NEVER run
    exec_history = result["execution_history"]
    assert len(exec_history) == 1
    assert exec_history[0].tool_name == "inspect_lockfiles"
    assert "restart_service" not in [e.tool_name for e in exec_history]

