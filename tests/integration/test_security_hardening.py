import tempfile
import time
import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from nova.environments.mock_workflow_runner import (
    MockWorkflowRunner,
    ScenarioType,
)
from nova.graph import build_nova_graph
from nova.llm.client import MockLLMClient
from nova.llm.schemas import PlannedActionSchema, PlannerResult
from nova.nodes.executor import create_executor_node
from nova.state import (
    ApprovalDecision,
    IncidentStatus,
    NovaGraphState,
    PlannedAction,
    RiskLevel,
)
from nova.tools.runner import create_default_tool_runner


@pytest.fixture
def mock_env():
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = MockWorkflowRunner(base_dir=tmpdir)
        yield runner


class DeceptiveRiskLLMClient(MockLLMClient):
    """
    Simulates a compromised, hallucinating, or misconfigured LLM that
    attempts to classify a DESTRUCTIVE action as READ_ONLY in order
    to bypass Human-in-the-Loop gating.
    """

    def plan(self, triage, available_tools, retry_count):
        return PlannerResult(
            is_escalation=False,
            actions=[
                PlannedActionSchema(
                    tool_name="clear_lockfile",
                    parameters={"lock_name": "transform_job.lock"},
                    risk_level=RiskLevel.READ_ONLY,  # Deceptive label!
                    rationale="Malicious or hallucinated downgrade of tool risk rating",
                    expected_outcome="Lockfile cleared without authorization",
                )
            ],
        )


def test_registry_overrides_llm_risk_label_and_interrupts(mock_env: MockWorkflowRunner):
    """
    SECURITY TEST 1:
    Proves that when the LLM labels a destructive tool as READ_ONLY,
    NOVA's executor node disregards the LLM label, looks up the ground-truth
    risk from the ToolRunner registry (RiskLevel.DESTRUCTIVE), and triggers
    the LangGraph interrupt() HITL checkpoint before executing the tool.
    """
    mock_env.inject_scenario(ScenarioType.STALE_LOCKFILE)
    mock_env.run_workflow()

    checkpointer = MemorySaver()
    deceptive_llm = DeceptiveRiskLLMClient()
    graph = build_nova_graph(
        mock_env,
        checkpointer=checkpointer,
        require_approval=True,
        llm=deceptive_llm,
    )

    thread_config = {"configurable": {"thread_id": "INC_SEC_DECEPTIVE_01"}}

    initial_state: NovaGraphState = {
        "incident_id": "INC_SEC_DECEPTIVE_01",
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

    # Execute graph: must halt on interrupt despite LLM labeling as READ_ONLY
    graph.invoke(initial_state, thread_config)

    checkpoint_state = graph.get_state(thread_config)
    assert checkpoint_state.next == ("executor",)
    assert len(checkpoint_state.tasks) == 1
    assert len(checkpoint_state.tasks[0].interrupts) == 1

    payload = checkpoint_state.tasks[0].interrupts[0].value
    assert payload["tool_name"] == "clear_lockfile"
    assert payload["risk_level"] == "DESTRUCTIVE"

    # Physical environment must be untouched
    assert (mock_env.locks_dir / "transform_job.lock").exists()
    assert len(checkpoint_state.values.get("execution_history", [])) == 0


def test_approval_binding_mismatch_action_id_rejected_and_escalated(mock_env: MockWorkflowRunner):
    """
    SECURITY TEST 2:
    Proves that an approval decision targeting an action_id that does not
    match the current pending action is strictly rejected and escalated.
    """
    mock_env.inject_scenario(ScenarioType.STALE_LOCKFILE)
    mock_env.run_workflow()

    checkpointer = MemorySaver()
    graph = build_nova_graph(mock_env, checkpointer=checkpointer, require_approval=True)
    thread_config = {"configurable": {"thread_id": "INC_SEC_BINDING_01"}}

    initial_state: NovaGraphState = {
        "incident_id": "INC_SEC_BINDING_01",
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

    # Step 1: Halt at interrupt
    graph.invoke(initial_state, thread_config)

    # Step 2: Attempt to resume with wrong action_id
    final_state = graph.invoke(
        Command(
            resume={
                "approved": True,
                "action_id": "ACT_ATTACKER_INJECTED_999",
                "approver_id": "operator_sre",
                "approver_role": "SRE_LEAD",
            }
        ),
        thread_config,
    )

    # Step 3: Verify strict escalation and zero execution
    assert final_state["status"] == IncidentStatus.ESCALATED
    assert "Binding mismatch on action_id" in final_state["final_summary"]
    assert len(final_state["execution_history"]) == 0
    assert (mock_env.locks_dir / "transform_job.lock").exists()


def test_approval_binding_mismatch_incident_id_rejected(mock_env: MockWorkflowRunner):
    """
    SECURITY TEST 2b:
    Proves that an approval decision targeting a different incident_id is strictly rejected.
    """
    mock_env.inject_scenario(ScenarioType.STALE_LOCKFILE)
    mock_env.run_workflow()

    checkpointer = MemorySaver()
    graph = build_nova_graph(mock_env, checkpointer=checkpointer, require_approval=True)
    thread_config = {"configurable": {"thread_id": "INC_SEC_BINDING_02"}}

    initial_state: NovaGraphState = {
        "incident_id": "INC_SEC_BINDING_02",
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

    graph.invoke(initial_state, thread_config)

    final_state = graph.invoke(
        Command(
            resume={
                "approved": True,
                "incident_id": "INC_WRONG_INCIDENT_888",
                "approver_id": "operator_sre",
                "approver_role": "SRE_LEAD",
            }
        ),
        thread_config,
    )

    assert final_state["status"] == IncidentStatus.ESCALATED
    assert "Binding mismatch on incident_id" in final_state["final_summary"]
    assert len(final_state["execution_history"]) == 0
    assert (mock_env.locks_dir / "transform_job.lock").exists()


def test_approval_unauthorized_role_rejected_and_escalated(mock_env: MockWorkflowRunner):
    """
    SECURITY TEST 3:
    Proves that an approval signed with an unauthorized role (e.g. 'INTERN', 'AUDITOR')
    fails RBAC validation and is escalated without running the destructive tool.
    """
    mock_env.inject_scenario(ScenarioType.STALE_LOCKFILE)
    mock_env.run_workflow()

    checkpointer = MemorySaver()
    graph = build_nova_graph(mock_env, checkpointer=checkpointer, require_approval=True)
    thread_config = {"configurable": {"thread_id": "INC_SEC_RBAC_01"}}

    initial_state: NovaGraphState = {
        "incident_id": "INC_SEC_RBAC_01",
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

    graph.invoke(initial_state, thread_config)

    # Operator claims approval with unauthorized role 'INTERN'
    final_state = graph.invoke(
        Command(
            resume={
                "approved": True,
                "approver_id": "intern_bob",
                "approver_role": "INTERN",
            }
        ),
        thread_config,
    )

    assert final_state["status"] == IncidentStatus.ESCALATED
    assert "Unauthorized approver role 'INTERN'" in final_state["final_summary"]
    assert len(final_state["execution_history"]) == 0
    assert (mock_env.locks_dir / "transform_job.lock").exists()


def test_approval_rejection_boolean_false_and_malformed_payload(mock_env: MockWorkflowRunner):
    """
    SECURITY TEST 4:
    Proves that boolean False, malformed types (strings, invalid dicts, non-boolean approved)
    are strictly rejected and escalated.
    """
    mock_env.inject_scenario(ScenarioType.STALE_LOCKFILE)
    mock_env.run_workflow()

    checkpointer = MemorySaver()
    graph = build_nova_graph(mock_env, checkpointer=checkpointer, require_approval=True)

    # Subtest 4a: Direct boolean False
    thread_config_a = {"configurable": {"thread_id": "INC_SEC_REJECT_01"}}
    initial_state_a: NovaGraphState = {
        "incident_id": "INC_SEC_REJECT_01",
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
    graph.invoke(initial_state_a, thread_config_a)
    final_a = graph.invoke(Command(resume=False), thread_config_a)
    assert final_a["status"] == IncidentStatus.ESCALATED
    assert len(final_a["execution_history"]) == 0

    # Subtest 4b: Malformed string "true" (strict boolean enforcement!)
    thread_config_b = {"configurable": {"thread_id": "INC_SEC_REJECT_02"}}
    initial_state_b: NovaGraphState = {
        "incident_id": "INC_SEC_REJECT_02",
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
    graph.invoke(initial_state_b, thread_config_b)
    final_b = graph.invoke(Command(resume={"approved": "true"}), thread_config_b)
    assert final_b["status"] == IncidentStatus.ESCALATED
    assert len(final_b["execution_history"]) == 0

    # Subtest 4c: Arbitrary garbage string
    thread_config_c = {"configurable": {"thread_id": "INC_SEC_REJECT_03"}}
    initial_state_c: NovaGraphState = {
        "incident_id": "INC_SEC_REJECT_03",
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
    graph.invoke(initial_state_c, thread_config_c)
    final_c = graph.invoke(Command(resume="some_arbitrary_string"), thread_config_c)
    assert final_c["status"] == IncidentStatus.ESCALATED
    assert len(final_c["execution_history"]) == 0


def test_authorized_rbac_approval_records_audit_trail_in_execution_record(mock_env: MockWorkflowRunner):
    """
    SECURITY TEST 5:
    Proves that a fully authorized approval by an authorized role ('PLATFORM_ADMIN')
    executes cleanly and records approved_by, approver_role, and approval_id
    into the ExecutionRecord audit trail.
    """
    mock_env.inject_scenario(ScenarioType.STALE_LOCKFILE)
    mock_env.run_workflow()

    checkpointer = MemorySaver()
    graph = build_nova_graph(mock_env, checkpointer=checkpointer, require_approval=True)
    thread_config = {"configurable": {"thread_id": "INC_SEC_AUTH_01"}}

    initial_state: NovaGraphState = {
        "incident_id": "INC_SEC_AUTH_01",
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

    graph.invoke(initial_state, thread_config)

    # Resume with explicit PLATFORM_ADMIN credentials and approval_request_id
    final_state = graph.invoke(
        Command(
            resume={
                "approved": True,
                "approver_id": "admin_dave",
                "approver_role": "PLATFORM_ADMIN",
                "approval_request_id": "APPR_TICKET_99182",
                "notes": "Emergency remediation approved during P1 incident bridge",
            }
        ),
        thread_config,
    )

    assert final_state["status"] == IncidentStatus.RESOLVED
    assert len(final_state["execution_history"]) == 2

    exec_record = final_state["execution_history"][0]
    assert exec_record.tool_name == "clear_lockfile"
    assert exec_record.success is True
    assert exec_record.approved_by == "admin_dave"
    assert exec_record.approver_role == "PLATFORM_ADMIN"
    assert exec_record.approval_id == "APPR_TICKET_99182"
    assert final_state["execution_history"][1].tool_name == "resume_workflow"
    assert not (mock_env.locks_dir / "transform_job.lock").exists()


def test_approval_decision_pydantic_model_resume(mock_env: MockWorkflowRunner):
    """
    SECURITY TEST 6:
    Proves that a typed ApprovalDecision Pydantic instance can be passed directly
    via Command(resume=decision) and enforces strict RBAC and binding checks.
    """
    mock_env.inject_scenario(ScenarioType.STALE_LOCKFILE)
    mock_env.run_workflow()

    checkpointer = MemorySaver()
    graph = build_nova_graph(mock_env, checkpointer=checkpointer, require_approval=True)
    thread_config = {"configurable": {"thread_id": "INC_SEC_MODEL_01"}}

    initial_state: NovaGraphState = {
        "incident_id": "INC_SEC_MODEL_01",
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

    graph.invoke(initial_state, thread_config)
    checkpoint_state = graph.get_state(thread_config)
    pending_action_id = checkpoint_state.tasks[0].interrupts[0].value["action_id"]

    decision = ApprovalDecision(
        approved=True,
        approver_id="commander_carol",
        approver_role="INCIDENT_COMMANDER",
        decision_timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        incident_id="INC_SEC_MODEL_01",
        action_id=pending_action_id,
        approval_request_id="APPR_IC_001",
        notes="Authorized by IC on duty",
    )

    final_state = graph.invoke(Command(resume=decision), thread_config)

    assert final_state["status"] == IncidentStatus.RESOLVED
    exec_record = final_state["execution_history"][0]
    assert exec_record.approved_by == "commander_carol"
    assert exec_record.approver_role == "INCIDENT_COMMANDER"
    assert exec_record.approval_id == "APPR_IC_001"


def test_unknown_tool_fails_closed_without_execution(mock_env: MockWorkflowRunner):
    """
    SECURITY TEST 7:
    Proves that when an action specifies an unknown or unauthorized tool name,
    the executor node fails closed, transitions immediately to ESCALATED,
    and NEVER executes any tool.
    """
    runner = create_default_tool_runner(mock_env)
    executor_node = create_executor_node(runner, require_approval=True)

    bad_action = PlannedAction(
        action_id="ACT_EXPLOIT_01",
        tool_name="unregistered_dangerous_tool",
        parameters={"drop": "all"},
        risk_level=RiskLevel.READ_ONLY,
        rationale="Simulated attacker injecting unknown tool",
        expected_outcome="Fail closed",
    )

    state: NovaGraphState = {
        "incident_id": "INC_SEC_UNKNOWN_01",
        "target_service": "JOB_01_EXTRACT",
        "raw_symptoms": "Exploit attempt",
        "status": IncidentStatus.PLANNING,
        "triage_summary": None,
        "root_cause_hypothesis": None,
        "plan": [bad_action],
        "current_action_index": 0,
        "execution_history": [],
        "verification_history": [],
        "pending_approval": None,
        "retry_count": 0,
        "max_retries": 3,
        "final_summary": None,
    }

    result = executor_node(state)
    assert result["status"] == IncidentStatus.ESCALATED
    assert "Security Alert" in result["final_summary"]
    assert "cannot be determined" in result["final_summary"]
    assert "execution_history" not in result or len(result["execution_history"]) == 0
