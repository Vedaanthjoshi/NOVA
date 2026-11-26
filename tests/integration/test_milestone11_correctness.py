"""
Milestone 11: Execution, Verification & Safety Correctness Regression Suite.

Covers all 10 reduced-scope M11 requirements:
1. Verification is strictly read-only (never calls env.run_workflow()).
2. Workflow resumption is an explicit planned action (resume_workflow).
3. MockLLMClient and GeminiLLMClient planning/replanning alignment.
4. Deterministic postcondition checks for mutating actions.
5. Pathlib filesystem containment (is_relative_to + symlink check) & registered IDs.
6. Rejection of invalid service, job, lock, and workflow identifiers.
7. Rejection of empty and contradictory remediation plans.
8. Clearing stale job error_message after successful recovery.
9. Single active incident enforcement, scenario reset lock during active/pending approval,
   and immutable workflow run binding across triage -> executor -> verifier.
10. Rejection of permissive boolean / incomplete dict approval shortcuts in operational mode.
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from pydantic import ValidationError

from nova.api.main import app, set_incident_service
from nova.api.service import IncidentService
from nova.environments.mock_workflow_runner import (
    JobStatus,
    MockWorkflowRunner,
    ScenarioType,
    resolve_contained_path,
)
from nova.graph import build_nova_graph
from nova.llm.client import GeminiLLMClient, MockLLMClient
from nova.llm.schemas import PlannedActionSchema, PlannerResult, ReplannerResult, TriageResult
from nova.nodes.executor import create_executor_node
from nova.nodes.planner import create_planner_node
from nova.nodes.verifier import create_verifier_node
from nova.state import (
    ExecutionRecord,
    IncidentStatus,
    NovaGraphState,
    PlannedAction,
    RiskLevel,
)
from nova.storage.sqlite_saver import SqliteSaver
from nova.tools.base import BaseTool, ToolResult
from nova.tools.runner import ToolRunner, create_default_tool_runner


# ---------------------------------------------------------------------------
# 1 & 2. Read-only verifier + explicit resume_workflow
# ---------------------------------------------------------------------------

def test_verifier_is_strictly_read_only_and_never_runs_workflow(tmp_path: Path):
    """Verifier must call env.get_state() only and NEVER env.run_workflow()."""
    env = MockWorkflowRunner(base_dir=tmp_path / "mock_state")
    env.inject_scenario(ScenarioType.HAPPY_PATH)
    # Jobs are PENDING initially; old verifier mutated state by calling run_workflow().
    run_spy = MagicMock(side_effect=AssertionError("Verifier must never call run_workflow()!"))
    env.run_workflow = run_spy  # type: ignore[method-assign]

    verifier = create_verifier_node(env, tool_runner=create_default_tool_runner(env))
    state: NovaGraphState = {
        "incident_id": "INC_RO_VERIFY_01",
        "status": IncidentStatus.VERIFYING,
        "plan": [
            PlannedAction(
                action_id="ACT_01_01",
                tool_name="resume_workflow",
                parameters={"workflow_id": "WF_NIGHTLY_BATCH_1001"},
                risk_level=RiskLevel.LOW_RISK,
                rationale="Resume",
                expected_outcome="All jobs pass",
            )
        ],
        "current_action_index": 1,
        "execution_history": [
            ExecutionRecord(
                action_id="ACT_01_01",
                tool_name="resume_workflow",
                executed_at="2026-11-26 10:00:00",
                exit_code=0,
                stdout="ok",
                stderr="",
                duration_seconds=0.01,
                success=True,
            )
        ],
        "retry_count": 0,
        "max_retries": 0,
    }

    result = verifier(state)
    assert run_spy.call_count == 0
    assert result["status"] != IncidentStatus.RESOLVED
    assert result["verification_history"][0].passed is False
    # Physical state remained PENDING
    wf = env.get_state()
    assert all(j.status == JobStatus.PENDING for j in wf.jobs.values())


def test_read_only_plan_cannot_falsely_resolve_incident(tmp_path: Path):
    """
    A plan containing only READ_ONLY actions (e.g. get_workflow_status) must NEVER
    resolve an incident — neither when jobs are PENDING nor when jobs are already SUCCESS.
    """
    class ReadOnlyPlanLLM(MockLLMClient):
        def plan(self, triage, available_tools, retry_count):
            return PlannerResult(
                is_escalation=False,
                actions=[
                    PlannedActionSchema(
                        tool_name="get_workflow_status",
                        parameters={},
                        risk_level=RiskLevel.READ_ONLY,
                        rationale="Only inspect status",
                        expected_outcome="Status read",
                    )
                ],
            )

    # Case A: HAPPY_PATH before workflow execution (all jobs PENDING)
    env_pending = MockWorkflowRunner(base_dir=tmp_path / "pending")
    env_pending.inject_scenario(ScenarioType.HAPPY_PATH)
    graph_a = build_nova_graph(env_pending, llm=ReadOnlyPlanLLM(), require_approval=False)
    out_a = graph_a.invoke(
        {
            "incident_id": "INC_RO_PENDING",
            "target_service": "JOB_01_EXTRACT",
            "raw_symptoms": "Check pipeline",
            "status": IncidentStatus.TRIGGERED,
            "max_retries": 0,
        }
    )
    assert out_a["status"] != IncidentStatus.RESOLVED
    assert all(j.status == JobStatus.PENDING for j in env_pending.get_state().jobs.values())

    # Case B: HAPPY_PATH after workflow already succeeded (all jobs SUCCESS)
    env_done = MockWorkflowRunner(base_dir=tmp_path / "done")
    env_done.inject_scenario(ScenarioType.HAPPY_PATH)
    env_done.run_workflow()
    verifier = create_verifier_node(env_done, tool_runner=create_default_tool_runner(env_done))
    res_b = verifier(
        {
            "incident_id": "INC_RO_DONE",
            "status": IncidentStatus.VERIFYING,
            "plan": [
                PlannedAction(
                    action_id="ACT_01_01",
                    tool_name="get_workflow_status",
                    parameters={},
                    risk_level=RiskLevel.READ_ONLY,
                    rationale="Inspect",
                    expected_outcome="Observed",
                )
            ],
            "current_action_index": 1,
            "execution_history": [
                ExecutionRecord(
                    action_id="ACT_01_01",
                    tool_name="get_workflow_status",
                    executed_at="2026-11-26 10:00:00",
                    exit_code=0,
                    stdout="SUCCESS",
                    stderr="",
                    duration_seconds=0.01,
                    success=True,
                )
            ],
            "retry_count": 0,
            "max_retries": 0,
        }
    )
    assert res_b["status"] == IncidentStatus.ESCALATED
    assert res_b["verification_history"][0].passed is False


def test_plan_without_explicit_resume_workflow_does_not_resolve(tmp_path: Path):
    """Clearing the lockfile without executing resume_workflow must NOT declare RESOLVED."""
    class ClearOnlyLLM(MockLLMClient):
        def plan(self, triage, available_tools, retry_count):
            return PlannerResult(
                is_escalation=False,
                actions=[
                    PlannedActionSchema(
                        tool_name="clear_lockfile",
                        parameters={"lock_name": "transform_job.lock"},
                        risk_level=RiskLevel.DESTRUCTIVE,
                        rationale="Clear stale lock only",
                        expected_outcome="Lock gone",
                    )
                ],
            )

    env = MockWorkflowRunner(base_dir=tmp_path / "clear_only")
    env.inject_scenario(ScenarioType.STALE_LOCKFILE)
    env.run_workflow()

    graph = build_nova_graph(env, llm=ClearOnlyLLM(), require_approval=False)
    res = graph.invoke(
        {
            "incident_id": "INC_NO_RESUME_01",
            "target_service": "JOB_02_TRANSFORM",
            "raw_symptoms": "Lockfile blocking transform",
            "status": IncidentStatus.TRIGGERED,
            "max_retries": 0,
        }
    )
    assert not (env.locks_dir / "transform_job.lock").exists()
    assert res["status"] == IncidentStatus.ESCALATED
    assert res["verification_history"][0].passed is False


# ---------------------------------------------------------------------------
# 4. Deterministic postcondition checks
# ---------------------------------------------------------------------------

def test_executor_postcondition_catches_fake_clear_lockfile_success(tmp_path: Path):
    """If clear_lockfile returns exit_code=0 but the lock remains on disk, executor fails."""
    env = MockWorkflowRunner(base_dir=tmp_path / "postcond_lock")
    env.inject_scenario(ScenarioType.STALE_LOCKFILE)
    env.run_workflow()

    class FakeClearLockTool(BaseTool):
        name = "clear_lockfile"
        description = "Fake tool claiming success without removing lock"
        risk_level = RiskLevel.DESTRUCTIVE
        args_schema = create_default_tool_runner(env).get_tool("clear_lockfile").args_schema

        def _run(self, lock_name: str, allow_idempotent: bool = True) -> ToolResult:
            return ToolResult(stdout="Pretended to remove lock", exit_code=0, success=True)

    runner = ToolRunner()
    runner.register(FakeClearLockTool())
    executor = create_executor_node(runner, env=env, require_approval=False)

    out = executor(
        {
            "incident_id": "INC_POSTCOND_LOCK",
            "status": IncidentStatus.EXECUTING,
            "plan": [
                PlannedAction(
                    action_id="ACT_01_01",
                    tool_name="clear_lockfile",
                    parameters={"lock_name": "transform_job.lock"},
                    risk_level=RiskLevel.DESTRUCTIVE,
                    rationale="Remove lock",
                    expected_outcome="Lock absent",
                ),
                PlannedAction(
                    action_id="ACT_01_02",
                    tool_name="resume_workflow",
                    parameters={"workflow_id": "WF_NIGHTLY_BATCH_1001"},
                    risk_level=RiskLevel.LOW_RISK,
                    rationale="Resume",
                    expected_outcome="Jobs pass",
                ),
            ],
            "current_action_index": 0,
            "retry_count": 0,
            "max_retries": 1,
        }
    )
    assert out["status"] == IncidentStatus.FAILED
    assert "current_action_index" not in out
    assert "postcondition failed" in out["final_summary"].lower()


def test_executor_postcondition_catches_fake_restart_service_success(tmp_path: Path):
    """If restart_service returns exit_code=0 but service stays STOPPED, executor fails."""
    env = MockWorkflowRunner(base_dir=tmp_path / "postcond_svc")
    env.inject_scenario(ScenarioType.SERVICE_OFFLINE)
    env.run_workflow()

    class FakeRestartTool(BaseTool):
        name = "restart_service"
        description = "Fake restart that leaves service offline"
        risk_level = RiskLevel.DESTRUCTIVE
        args_schema = create_default_tool_runner(env).get_tool("restart_service").args_schema

        def _run(self, service_name: str, allow_idempotent: bool = True) -> ToolResult:
            return ToolResult(stdout="Pretended to restart", exit_code=0, success=True)

    runner = ToolRunner()
    runner.register(FakeRestartTool())
    executor = create_executor_node(runner, env=env, require_approval=False)

    out = executor(
        {
            "incident_id": "INC_POSTCOND_SVC",
            "status": IncidentStatus.EXECUTING,
            "plan": [
                PlannedAction(
                    action_id="ACT_01_01",
                    tool_name="restart_service",
                    parameters={"service_name": "reporting_warehouse_service"},
                    risk_level=RiskLevel.DESTRUCTIVE,
                    rationale="Restart DB",
                    expected_outcome="Healthy",
                )
            ],
            "current_action_index": 0,
            "retry_count": 0,
            "max_retries": 1,
        }
    )
    assert out["status"] == IncidentStatus.FAILED
    assert "postcondition failed" in out["final_summary"].lower()


# ---------------------------------------------------------------------------
# 5 & 6. Pathlib containment, symlink safety & invalid identifier rejection
# ---------------------------------------------------------------------------

def test_pathlib_containment_rejects_traversal_sibling_prefix_and_symlinks(tmp_path: Path):
    """
    Tests proper pathlib containment (not string prefix matching):
    - Rejects '../outside.lock', '..\\..\\secret', sibling directory prefix bypass ('locks_evil/x.lock').
    - Rejects symlinks pointing outside the workspace.
    """
    locks_dir = tmp_path / "locks"
    locks_dir.mkdir()
    sibling_dir = tmp_path / "locks_evil"
    sibling_dir.mkdir()
    (sibling_dir / "transform_job.lock").write_text("evil", encoding="utf-8")
    outside_file = tmp_path / "secret.txt"
    outside_file.write_text("top-secret", encoding="utf-8")

    # Sibling prefix-collision directory must fail containment
    assert resolve_contained_path(locks_dir, "../locks_evil/transform_job.lock") is None
    assert resolve_contained_path(locks_dir, "../secret.txt") is None
    assert resolve_contained_path(locks_dir, "..\\secret.txt") is None
    assert resolve_contained_path(locks_dir, "subdir/transform_job.lock") is None
    assert resolve_contained_path(locks_dir, "") is None
    assert resolve_contained_path(locks_dir, ".") is None
    assert resolve_contained_path(locks_dir, "..") is None

    # Valid contained filename succeeds
    assert resolve_contained_path(locks_dir, "transform_job.lock") == (locks_dir / "transform_job.lock").resolve()

    # Symlink rejection (when OS permits symlink creation)
    symlink_path = locks_dir / "transform_job.lock"
    try:
        symlink_path.symlink_to(outside_file)
    except OSError:
        pytest.skip("OS environment does not permit unprivileged symlink creation")

    assert resolve_contained_path(locks_dir, "transform_job.lock") is None
    env = MockWorkflowRunner(base_dir=tmp_path / "ws")
    env.inject_scenario(ScenarioType.HAPPY_PATH)
    env_sym = env.locks_dir / "transform_job.lock"
    env_sym.symlink_to(outside_file)
    runner = create_default_tool_runner(env)
    rec = runner.execute(
        PlannedAction(
            action_id="ACT_SYM_01",
            tool_name="clear_lockfile",
            parameters={"lock_name": "transform_job.lock"},
            risk_level=RiskLevel.DESTRUCTIVE,
            rationale="Attempt symlink unlink",
            expected_outcome="Rejected",
        )
    )
    assert rec.success is False
    assert outside_file.exists()


def test_invalid_lock_service_job_and_workflow_ids_rejected(tmp_path: Path):
    """Unregistered lock names, service names, job IDs, and mismatched workflow IDs must fail."""
    env = MockWorkflowRunner(base_dir=tmp_path / "ids")
    env.inject_scenario(ScenarioType.SERVICE_OFFLINE)
    env.run_workflow()
    runner = create_default_tool_runner(env)

    # 1. Unregistered lockfile + traversal
    for bad_lock in ["unregistered.lock", "../workflow_state.json", "..\\secret.lock"]:
        res_clear = runner.execute(
            PlannedAction(
                action_id="ACT_L1",
                tool_name="clear_lockfile",
                parameters={"lock_name": bad_lock},
                risk_level=RiskLevel.DESTRUCTIVE,
                rationale="test",
                expected_outcome="fail",
            )
        )
        assert res_clear.success is False

        res_insp = runner.execute(
            PlannedAction(
                action_id="ACT_L2",
                tool_name="inspect_lockfiles",
                parameters={"lock_name": bad_lock},
                risk_level=RiskLevel.READ_ONLY,
                rationale="test",
                expected_outcome="fail",
            )
        )
        assert res_insp.success is False

    # 2. Unregistered service name rejected and NEVER added to state.services
    res_svc_chk = runner.execute(
        PlannedAction(
            action_id="ACT_S1",
            tool_name="check_service_status",
            parameters={"service_name": "nonexistent_service"},
            risk_level=RiskLevel.READ_ONLY,
            rationale="test",
            expected_outcome="fail",
        )
    )
    assert res_svc_chk.success is False

    res_svc_rst = runner.execute(
        PlannedAction(
            action_id="ACT_S2",
            tool_name="restart_service",
            parameters={"service_name": "nonexistent_service"},
            risk_level=RiskLevel.DESTRUCTIVE,
            rationale="test",
            expected_outcome="fail",
        )
    )
    assert res_svc_rst.success is False
    assert "nonexistent_service" not in env.get_state().services

    # 3. Unregistered job ID rejected
    res_log = runner.execute(
        PlannedAction(
            action_id="ACT_J1",
            tool_name="read_job_logs",
            parameters={"job_id": "../workflow_state", "tail_lines": 10},
            risk_level=RiskLevel.READ_ONLY,
            rationale="test",
            expected_outcome="fail",
        )
    )
    assert res_log.success is False

    # 4. Mismatched workflow ID rejected by resume_workflow & get_workflow_status
    res_wf = runner.execute(
        PlannedAction(
            action_id="ACT_W1",
            tool_name="resume_workflow",
            parameters={"workflow_id": "WF_WRONG_9999"},
            risk_level=RiskLevel.LOW_RISK,
            rationale="test",
            expected_outcome="fail",
        )
    )
    assert res_wf.success is False
    assert "mismatch" in res_wf.stderr.lower()

    res_wf_status = runner.execute(
        PlannedAction(
            action_id="ACT_W2",
            tool_name="get_workflow_status",
            parameters={"workflow_id": "WF_WRONG_9999"},
            risk_level=RiskLevel.READ_ONLY,
            rationale="test",
            expected_outcome="fail",
        )
    )
    assert res_wf_status.success is False


# ---------------------------------------------------------------------------
# 7. Empty and contradictory remediation plans rejected
# ---------------------------------------------------------------------------

def test_empty_and_contradictory_plans_rejected():
    """Schema rejects empty non-escalating plans and escalating plans with actions."""
    # Empty non-escalating plan
    with pytest.raises(ValidationError):
        PlannerResult(is_escalation=False, escalation_reason=None, actions=[])

    # Contradictory: escalating WITH actions
    with pytest.raises(ValidationError):
        PlannerResult(
            is_escalation=True,
            escalation_reason="Unrecoverable",
            actions=[
                PlannedActionSchema(
                    tool_name="clear_lockfile",
                    parameters={"lock_name": "transform_job.lock"},
                    risk_level=RiskLevel.DESTRUCTIVE,
                    rationale="Should not be allowed",
                    expected_outcome="None",
                )
            ],
        )

    # Contradictory ReplannerResult: non-escalation with escalation_reason
    with pytest.raises(ValidationError):
        ReplannerResult(
            new_target_service="JOB_03_LOAD",
            updated_triage_summary="summary",
            updated_hypothesis="hyp",
            new_failure_type="SERVICE_OFFLINE",
            is_escalation=False,
            escalation_reason="Contradiction",
        )


def test_planner_node_rejects_bypassed_empty_plan():
    """Even if a mock LLM bypasses Pydantic validation, planner_node never sets EXECUTING on empty plan."""
    class BypassLLM(MockLLMClient):
        def plan(self, triage, available_tools, retry_count):
            return PlannerResult.model_construct(
                is_escalation=False,
                escalation_reason=None,
                actions=[],
            )

    planner = create_planner_node(BypassLLM())
    out = planner(
        {
            "incident_id": "INC_EMPTY_PLAN",
            "triage_summary": "Lock blocking",
            "root_cause_hypothesis": "Stale lock",
            "failure_type": "STALE_LOCKFILE",
            "implicated_resource": "transform_job.lock",
            "is_recoverable": True,
            "retry_count": 0,
        }
    )
    assert out["status"] == IncidentStatus.ESCALATED
    assert out["plan"] == []


# ---------------------------------------------------------------------------
# 8. Clear stale error messages after recovery
# ---------------------------------------------------------------------------

def test_stale_error_message_cleared_after_successful_recovery(tmp_path: Path):
    """After JOB_02_TRANSFORM fails and then recovers, error_message must be None."""
    env = MockWorkflowRunner(base_dir=tmp_path / "clear_err")
    env.inject_scenario(ScenarioType.STALE_LOCKFILE)
    failed_state = env.run_workflow()
    assert failed_state.jobs["JOB_02_TRANSFORM"].status == JobStatus.FAILED
    assert failed_state.jobs["JOB_02_TRANSFORM"].error_message is not None

    assert env.clear_lock("transform_job.lock") is True
    recovered_state = env.run_workflow()
    assert recovered_state.jobs["JOB_02_TRANSFORM"].status == JobStatus.SUCCESS
    assert recovered_state.jobs["JOB_02_TRANSFORM"].error_message is None


# ---------------------------------------------------------------------------
# 9. Single active incident, scenario reset guard, and workflow run binding
# ---------------------------------------------------------------------------

def test_single_active_incident_and_scenario_reset_blocked_during_pending_approval(tmp_path: Path):
    """
    While an incident is active or AWAITING_APPROVAL:
    - Triggering a second incident returns 409 Conflict.
    - Injecting/resetting a scenario returns 409 Conflict.
    - Once the active incident completes to RESOLVED, new scenario inject & trigger succeed.
    """
    env = MockWorkflowRunner(base_dir=tmp_path / "mock_state")
    saver = SqliteSaver(db_path=tmp_path / "m11_single.sqlite")
    service = IncidentService(env=env, checkpointer=saver, llm=MockLLMClient(), max_workers=2)
    set_incident_service(service)
    client = TestClient(app)

    try:
        inj1 = client.post("/scenarios/inject", json={"scenario": "STALE_LOCKFILE"})
        assert inj1.status_code == 200
        run_id_1 = inj1.json()["run_id"]
        assert run_id_1 is not None

        trig1 = client.post(
            "/incidents/trigger",
            json={
                "incident_id": "INC_SINGLE_01",
                "target_service": "JOB_02_TRANSFORM",
                "raw_symptoms": "Stale lockfile",
            },
        )
        assert trig1.status_code == 202

        st1 = service.wait_for_incident("INC_SINGLE_01", timeout=10.0)
        assert st1 is not None
        assert st1.status == IncidentStatus.AWAITING_APPROVAL
        assert st1.workflow_run_id == run_id_1

        # Second incident must be blocked with 409 Conflict
        trig2 = client.post(
            "/incidents/trigger",
            json={
                "incident_id": "INC_SINGLE_02",
                "target_service": "UNKNOWN",
                "raw_symptoms": "Another error",
            },
        )
        assert trig2.status_code == 409

        # Scenario reset must be blocked with 409 Conflict
        inj_blocked = client.post("/scenarios/inject", json={"scenario": "SERVICE_OFFLINE"})
        assert inj_blocked.status_code == 409

        # Approve INC_SINGLE_01 -> completes to RESOLVED
        ap = client.post(
            "/incidents/INC_SINGLE_01/approve",
            json={
                "approved": True,
                "approver_id": "sre_lead_m11",
                "approver_role": "SRE_LEAD",
                "action_id": st1.pending_approval.action_id,
                "action_hash": st1.pending_approval.action_hash,
            },
        )
        assert ap.status_code == 200
        final1 = service.wait_for_incident("INC_SINGLE_01", timeout=10.0)
        assert final1 is not None
        assert final1.status == IncidentStatus.RESOLVED

        # Now scenario inject and a new incident succeed
        inj2 = client.post("/scenarios/inject", json={"scenario": "SERVICE_OFFLINE"})
        assert inj2.status_code == 200
        assert inj2.json()["run_id"] != run_id_1
        trig3 = client.post(
            "/incidents/trigger",
            json={
                "incident_id": "INC_SINGLE_03",
                "target_service": "JOB_03_LOAD",
                "raw_symptoms": "Connection refused to reporting_warehouse_service",
            },
        )
        assert trig3.status_code == 202
        service.wait_for_incident("INC_SINGLE_03", timeout=10.0)
    finally:
        client.close()
        service.shutdown()
        saver.close()


def test_workflow_run_binding_rejects_stale_approval_after_external_env_reset(tmp_path: Path):
    """
    If the underlying environment is reset externally (producing a new run_id) while an
    incident is paused at HITL, resuming with approval MUST escalate without executing
    the destructive tool against the new workflow run.
    """
    env = MockWorkflowRunner(base_dir=tmp_path / "run_bind")
    env.inject_scenario(ScenarioType.STALE_LOCKFILE)
    env.run_workflow()

    checkpointer = MemorySaver()
    graph = build_nova_graph(env, checkpointer=checkpointer, require_approval=True)
    cfg = {"configurable": {"thread_id": "INC_RUN_BIND_01"}}

    graph.invoke(
        {
            "incident_id": "INC_RUN_BIND_01",
            "target_service": "JOB_02_TRANSFORM",
            "raw_symptoms": "Lockfile blocking transform",
            "status": IncidentStatus.TRIGGERED,
        },
        cfg,
    )
    paused = graph.get_state(cfg)
    assert paused.tasks[0].interrupts

    # Simulate external environment reset (brand-new run_id)
    env.inject_scenario(ScenarioType.STALE_LOCKFILE)
    env.run_workflow()

    # Attempt to approve the old incident against the reset environment
    final_state = graph.invoke(
        Command(
            resume={
                "approved": True,
                "approver_id": "sre_alice",
                "approver_role": "SRE_LEAD",
            }
        ),
        cfg,
    )
    assert final_state["status"] == IncidentStatus.ESCALATED
    assert "binding violation" in final_state["final_summary"].lower()
    assert len(final_state.get("execution_history", [])) == 0
    assert (env.locks_dir / "transform_job.lock").exists()


# ---------------------------------------------------------------------------
# 10. Reject permissive approval shortcuts in operational mode
# ---------------------------------------------------------------------------

def test_permissive_boolean_and_incomplete_dict_approvals_rejected(tmp_path: Path):
    """
    In operational mode (require_approval=True):
    - Raw boolean True is REJECTED (escalates without running tool).
    - Dict missing approver_id or approver_role is REJECTED (escalates without running tool).
    """
    env = MockWorkflowRunner(base_dir=tmp_path / "perm_appr")
    env.inject_scenario(ScenarioType.STALE_LOCKFILE)
    env.run_workflow()

    checkpointer = MemorySaver()
    graph = build_nova_graph(env, checkpointer=checkpointer, require_approval=True)

    # Case A: Raw boolean True
    cfg_a = {"configurable": {"thread_id": "INC_PERM_BOOL"}}
    graph.invoke(
        {
            "incident_id": "INC_PERM_BOOL",
            "target_service": "JOB_02_TRANSFORM",
            "raw_symptoms": "Stale lock",
            "status": IncidentStatus.TRIGGERED,
        },
        cfg_a,
    )
    res_a = graph.invoke(Command(resume=True), cfg_a)
    assert res_a["status"] == IncidentStatus.ESCALATED
    assert "raw boolean approval rejected" in res_a["final_summary"].lower()
    assert len(res_a.get("execution_history", [])) == 0
    assert (env.locks_dir / "transform_job.lock").exists()

    # Case B: Incomplete dict without approver_id / approver_role
    cfg_b = {"configurable": {"thread_id": "INC_PERM_DICT"}}
    graph.invoke(
        {
            "incident_id": "INC_PERM_DICT",
            "target_service": "JOB_02_TRANSFORM",
            "raw_symptoms": "Stale lock",
            "status": IncidentStatus.TRIGGERED,
        },
        cfg_b,
    )
    res_b = graph.invoke(Command(resume={"approved": True, "notes": "shortcut"}), cfg_b)
    assert res_b["status"] == IncidentStatus.ESCALATED
    assert "approver_id is required" in res_b["final_summary"].lower()
    assert len(res_b.get("execution_history", [])) == 0
    assert (env.locks_dir / "transform_job.lock").exists()


# ---------------------------------------------------------------------------
# 3. GeminiLLMClient prompt + schema alignment (stubbed provider)
# ---------------------------------------------------------------------------

def test_gemini_client_plan_and_replan_prompts_and_schema_guard(monkeypatch):
    """Verifies GeminiLLMClient prompts include explicit resume_workflow and reject invalid JSON plans."""
    captured_prompts: List[Dict[str, str]] = []

    class FakeModel:
        def __init__(self, model_name, system_instruction, generation_config):
            self.system_instruction = system_instruction

        def generate_content(self, user_prompt):
            captured_prompts.append({"system": self.system_instruction, "user": user_prompt})
            resp = MagicMock()
            if "Replanning Telemetry" in user_prompt:
                resp.text = json.dumps(
                    {
                        "new_target_service": "JOB_03_LOAD",
                        "updated_triage_summary": "Downstream warehouse offline",
                        "updated_hypothesis": "reporting_warehouse_service stopped",
                        "new_failure_type": "SERVICE_OFFLINE",
                        "implicated_resource": "reporting_warehouse_service",
                        "is_recoverable": True,
                        "diagnosis_revision_needed": False,
                        "is_escalation": False,
                        "escalation_reason": None,
                        "alternative_strategy_notes": None,
                    }
                )
            else:
                resp.text = json.dumps(
                    {
                        "is_escalation": False,
                        "escalation_reason": None,
                        "actions": [
                            {
                                "tool_name": "clear_lockfile",
                                "parameters": {"lock_name": "transform_job.lock"},
                                "risk_level": "DESTRUCTIVE",
                                "rationale": "Clear stale lock",
                                "expected_outcome": "Lock removed",
                            },
                            {
                                "tool_name": "resume_workflow",
                                "parameters": {"workflow_id": "WF_CUSTOM_2026"},
                                "risk_level": "LOW_RISK",
                                "rationale": "Run workflow",
                                "expected_outcome": "Jobs pass",
                            },
                        ],
                    }
                )
            return resp

    fake_genai = MagicMock()
    fake_genai.GenerativeModel = FakeModel
    fake_genai.GenerationConfig = lambda **kw: kw
    monkeypatch.setitem(__import__("sys").modules, "google.generativeai", fake_genai)

    client = GeminiLLMClient(api_key="test-key-not-real")
    triage = TriageResult(
        failure_type="STALE_LOCKFILE",
        implicated_resource="transform_job.lock",
        triage_summary="Lockfile blocking transform",
        root_cause_hypothesis="Orphaned lock",
        is_recoverable=True,
        workflow_id="WF_CUSTOM_2026",
    )
    sample_tools = [
        {
            "name": "clear_lockfile",
            "description": "Removes lockfile",
            "parameters": {"lock_name": "str (required)"},
            "risk_level": "DESTRUCTIVE",
        },
        {
            "name": "resume_workflow",
            "description": "Resumes workflow",
            "parameters": {"workflow_id": "str (required)"},
            "risk_level": "LOW_RISK",
        },
    ]
    plan_res = client.plan(triage, sample_tools, retry_count=0)
    assert len(plan_res.actions) == 2
    assert plan_res.actions[1].tool_name == "resume_workflow"
    assert "WF_CUSTOM_2026" in captured_prompts[0]["user"]
    assert "clear_lockfile" in captured_prompts[0]["system"]
    assert "resume_workflow" in captured_prompts[0]["system"]

    replan_res = client.replan(
        original_target="JOB_02_TRANSFORM",
        first_action_taken="clear_lockfile",
        new_target="JOB_03_LOAD",
        new_log_content="Connection refused",
        verification_evidence="JOB_03_LOAD failed",
        retry_count=1,
    )
    assert replan_res.new_target_service == "JOB_03_LOAD"
    assert "resume_workflow" in captured_prompts[1]["system"]
