"""
Milestone 13 Regression & Verification Suite:
1. Phase 1: Complete execution evidence preservation, postcondition failure fix,
   escalation diagnosis/reflection preservation, context-aware reinvestigation,
   and budget-exhaustion escalation.
2. Phase 2: Consequential diagnostic reflection (evidence_sufficient=False blocks
   premature remediation), planning critique preventing inappropriate action execution
   and preserving initial/critique/revised plans, and outcome reflection.
3. Phase 3: Recovery evaluation probes (Codex's stale-row probe, upstream-only-attempt
   probe, and idempotent retry semantics).
4. Phase 4: Second Kestra workflow (DuckDB output verification) and FastAPI Kestra
   execution binding.
"""

from pathlib import Path
from typing import Any, Dict, List
import httpx
import pytest
from fastapi.testclient import TestClient

from nova.api.main import app, set_incident_service
from nova.api.service import IncidentService
from nova.environments.kestra_runner import KestraEnvironment
from nova.environments.mock_workflow_runner import JobStatus, MockWorkflowRunner, ScenarioType
from nova.graph import build_nova_graph
from nova.llm.client import MockLLMClient
from nova.llm.schemas import (
    DiagnosticReflection,
    DiagnosticToolCall,
    InvestigationActionType,
    InvestigationDecision,
    PlanCritiqueResult,
    PlannedActionSchema,
    PlannerResult,
    PlanningReflection,
    ReplannerResult,
    TriageResult,
)
from nova.nodes.executor import create_executor_node
from nova.nodes.planner import create_planner_node
from nova.nodes.replanner import create_replanner_node
from nova.nodes.triage import create_triage_node
from nova.nodes.verifier import create_verifier_node
from nova.state import (
    DiagnosticRecord,
    ExecutionRecord,
    IncidentStatus,
    NovaGraphState,
    PlannedAction,
    RiskLevel,
    VerificationRecord,
)
from nova.storage.sqlite_saver import SqliteSaver
from nova.tools.base import BaseTool, ToolResult
from nova.tools.kestra_tools import create_kestra_tool_runner
from nova.tools.runner import ToolRunner, create_default_tool_runner


# ---------------------------------------------------------------------------
# Phase 1: Evidence preservation & postcondition defect fix
# ---------------------------------------------------------------------------

def test_postcondition_failure_updates_execution_record_and_verifier_evidence(tmp_path: Path):
    """
    Reproduces and verifies fix for confirmed defect:
    A tool returning exit_code=0 whose postcondition fails must record:
      - command_success=True
      - postcondition_passed=False
      - postcondition_error=<reason>
      - success=False
      - environment_changed=False
    And verifier must NOT report 'Tool execution failed with exit code 0: '.
    """
    env = MockWorkflowRunner(base_dir=tmp_path / "postcond")
    env.inject_scenario(ScenarioType.SERVICE_OFFLINE)
    env.run_workflow()

    class NoOpRestartTool(BaseTool):
        name = "restart_service"
        description = "Fake restart returning 0 without starting service"
        risk_level = RiskLevel.DESTRUCTIVE
        args_schema = create_default_tool_runner(env).get_tool("restart_service").args_schema

        def _run(self, service_name: str, allow_idempotent: bool = True) -> ToolResult:
            return ToolResult(stdout="Issued restart signal", exit_code=0, success=True)

    runner = ToolRunner()
    runner.register(NoOpRestartTool())
    executor = create_executor_node(runner, env=env, require_approval=False)
    verifier = create_verifier_node(env, tool_runner=runner)

    state: NovaGraphState = {
        "incident_id": "INC_M13_POSTCOND",
        "status": IncidentStatus.EXECUTING,
        "plan": [
            PlannedAction(
                action_id="ACT_01_01",
                tool_name="restart_service",
                parameters={"service_name": "reporting_warehouse_service"},
                risk_level=RiskLevel.DESTRUCTIVE,
                rationale="Restart DB",
                expected_outcome="reporting_warehouse_service is RUNNING",
            )
        ],
        "current_action_index": 0,
        "retry_count": 0,
        "max_retries": 1,
    }

    exec_out = executor(state)
    assert exec_out["status"] == IncidentStatus.FAILED
    rec: ExecutionRecord = exec_out["execution_history"][0]
    assert rec.command_success is True
    assert rec.postcondition_passed is False
    assert rec.postcondition_error is not None
    assert "still unhealthy" in rec.postcondition_error.lower()
    assert rec.success is False
    assert rec.environment_changed is False
    assert rec.parameters == {"service_name": "reporting_warehouse_service"}
    assert rec.stdout == "Issued restart signal"

    verif_state: NovaGraphState = {
        **state,
        "status": exec_out["status"],
        "final_summary": exec_out["final_summary"],
        "execution_history": exec_out["execution_history"],
    }
    verif_out = verifier(verif_state)
    vrec: VerificationRecord = verif_out["verification_history"][0]
    assert vrec.passed is False
    assert "Tool execution failed with exit code 0:" not in vrec.evidence
    assert "postcondition failed" in vrec.evidence.lower()
    assert vrec.details["command_success"] is True
    assert vrec.details["postcondition_passed"] is False
    assert vrec.details["parameters"] == {"service_name": "reporting_warehouse_service"}


def test_replanner_preserves_revised_diagnosis_and_reflection_on_direct_escalation(tmp_path: Path):
    """
    When replanner chooses direct is_escalation=True, revised diagnosis and outcome
    reflection must still be persisted into incident state (not dropped).
    """
    env = MockWorkflowRunner(base_dir=tmp_path / "replan_esc")
    env.inject_scenario(ScenarioType.STALE_LOCKFILE)
    env.run_workflow()

    class EscalatingOutcomeLLM(MockLLMClient):
        def __init__(self):
            super().__init__(enable_reflection=True)

        def replan(self, *args, **kwargs) -> ReplannerResult:
            # Verify replanner received attempted_plan and full execution evidence
            assert kwargs.get("attempted_plan")
            exec_recs = kwargs.get("execution_records") or []
            assert exec_recs[0]["parameters"] == {"lock_name": "transform_job.lock"}
            return ReplannerResult(
                new_target_service="JOB_01_EXTRACT",
                updated_triage_summary="Upstream payload checksum corruption discovered after lock cleared.",
                updated_hypothesis="Corrupt source batch cannot be repaired automatically.",
                new_failure_type="DATA_CORRUPTION",
                implicated_resource="JOB_01_EXTRACT",
                is_recoverable=False,
                diagnosis_revision_needed=True,
                is_escalation=True,
                escalation_reason="Unrecoverable checksum mismatch in source feed.",
                outcome_classification="NEW_DOWNSTREAM_FAILURE",
                expected_vs_observed="Expected jobs to pass; observed checksum mismatch.",
                outcome_reflection="Lock removal succeeded, exposing underlying corrupt source payload.",
            )

    replanner = create_replanner_node(env, EscalatingOutcomeLLM())
    out = replanner(
        {
            "incident_id": "INC_ESC_PRESERVE",
            "target_service": "JOB_02_TRANSFORM",
            "failure_type": "STALE_LOCKFILE",
            "implicated_resource": "transform_job.lock",
            "triage_summary": "Old summary",
            "root_cause_hypothesis": "Old hypothesis",
            "retry_count": 0,
            "max_retries": 2,
            "plan": [
                PlannedAction(
                    action_id="ACT_01_01",
                    tool_name="clear_lockfile",
                    parameters={"lock_name": "transform_job.lock"},
                    risk_level=RiskLevel.DESTRUCTIVE,
                    rationale="Clear lock",
                    expected_outcome="All jobs complete",
                )
            ],
            "execution_history": [
                ExecutionRecord(
                    action_id="ACT_01_01",
                    tool_name="clear_lockfile",
                    executed_at="2026-11-26 12:00:00",
                    exit_code=0,
                    stdout="Removed lock",
                    stderr="",
                    duration_seconds=0.01,
                    success=True,
                    parameters={"lock_name": "transform_job.lock"},
                    command_success=True,
                    postcondition_passed=True,
                    environment_changed=True,
                )
            ],
            "verification_history": [
                VerificationRecord(
                    action_id="ACT_01_01",
                    verified_at="2026-11-26 12:00:01",
                    passed=False,
                    evidence="JOB_01_EXTRACT failed with checksum mismatch",
                    details={"failed_jobs": ["JOB_01_EXTRACT"]},
                )
            ],
        }
    )

    assert out["status"] == IncidentStatus.ESCALATED
    assert out["failure_type"] == "DATA_CORRUPTION"
    assert out["implicated_resource"] == "JOB_01_EXTRACT"
    assert out["is_recoverable"] is False
    assert "Corrupt source batch" in out["root_cause_hypothesis"]
    assert len(out["reflection_history"]) == 1
    assert out["reflection_history"][0]["outcome_classification"] == "NEW_DOWNSTREAM_FAILURE"
    assert len(out["hypothesis_revisions"]) == 1


def test_context_aware_reinvestigation_allows_repeat_after_env_change_and_escalates_on_exhausted_budget(tmp_path: Path):
    """
    - Repeated diagnostic call is ALLOWED when execution_epoch/environment changed.
    - Repeated diagnostic call is BLOCKED when environment has NOT changed.
    - When replanner sets needs_further_investigation=True and investigation budget is exhausted,
      NOVA escalates cleanly instead of blindly proceeding to REPLANNING.
    """
    env = MockWorkflowRunner(base_dir=tmp_path / "reinv")
    env.inject_scenario(ScenarioType.SERVICE_OFFLINE)
    env.run_workflow()
    runner = create_default_tool_runner(env)

    class RepeatStatusLLM(MockLLMClient):
        def investigate(self, *args, **kwargs) -> InvestigationDecision:
            return InvestigationDecision(
                decision_type=InvestigationActionType.DIAGNOSTIC_ACTION,
                diagnostic_tool=DiagnosticToolCall(
                    tool_name="check_service_status",
                    parameters={"service_name": "reporting_warehouse_service"},
                    rationale="Re-checking service after environment change",
                ),
            )

    triage = create_triage_node(env, RepeatStatusLLM(), tool_runner=runner)

    # Prior diagnostic call occurred at execution_epoch=0 while service was stopped.
    # Now execution_history has 1 action (execution_epoch=1) -> repeat MUST be allowed!
    prev_rec = DiagnosticRecord(
        step=1,
        tool_name="check_service_status",
        parameters={"service_name": "reporting_warehouse_service"},
        output="STOPPED",
        success=True,
        timestamp="2026-11-26 12:00:00",
        execution_epoch=0,
        env_fingerprint="epoch0",
    )
    env.restart_service("reporting_warehouse_service")
    out_allowed = triage(
        {
            "incident_id": "INC_CTX_REINV",
            "target_service": "JOB_03_LOAD",
            "raw_symptoms": "Check service",
            "investigation_steps": 1,
            "max_investigation_steps": 5,
            "diagnostic_history": [prev_rec],
            "investigation_evidence": ["Step 1: STOPPED"],
            "execution_history": [
                ExecutionRecord(
                    action_id="ACT_01_01",
                    tool_name="restart_service",
                    executed_at="2026-11-26 12:00:05",
                    exit_code=0,
                    stdout="Started",
                    stderr="",
                    duration_seconds=0.01,
                    success=True,
                    environment_changed=True,
                )
            ],
        }
    )
    assert out_allowed["status"] == IncidentStatus.TRIAGING
    assert out_allowed["diagnostic_history"][0].tool_name == "check_service_status"

    # Calling the EXACT same tool again in the same epoch with unchanged env -> blocked!
    rec_epoch1 = out_allowed["diagnostic_history"][0]
    out_blocked = triage(
        {
            "incident_id": "INC_CTX_REINV",
            "target_service": "JOB_03_LOAD",
            "raw_symptoms": "Check service",
            "investigation_steps": 2,
            "max_investigation_steps": 5,
            "diagnostic_history": [prev_rec, rec_epoch1],
            "investigation_evidence": ["Step 1", "Step 2"],
            "execution_history": [
                ExecutionRecord(
                    action_id="ACT_01_01",
                    tool_name="restart_service",
                    executed_at="2026-11-26 12:00:05",
                    exit_code=0,
                    stdout="Started",
                    stderr="",
                    duration_seconds=0.01,
                    success=True,
                    environment_changed=True,
                )
            ],
        }
    )
    assert out_blocked["status"] == IncidentStatus.ESCALATED
    assert "repeated unproductive investigation call" in out_blocked["final_summary"]

    # Budget exhaustion in replanner when further investigation is needed -> MUST ESCALATE
    class NeedMoreInvLLM(MockLLMClient):
        def replan(self, *args, **kwargs) -> ReplannerResult:
            return ReplannerResult(
                new_target_service="JOB_03_LOAD",
                updated_triage_summary="Evidence contradictory",
                updated_hypothesis="Unknown cause",
                new_failure_type="UNKNOWN",
                implicated_resource="JOB_03_LOAD",
                is_recoverable=True,
                needs_further_investigation=True,
                outcome_reflection="Need more diagnostics before planning.",
            )

    replanner = create_replanner_node(env, NeedMoreInvLLM())
    esc_out = replanner(
        {
            "incident_id": "INC_BUDGET_REPLAN",
            "target_service": "JOB_03_LOAD",
            "investigation_steps": 5,
            "max_investigation_steps": 5,
            "retry_count": 0,
            "max_retries": 3,
            "verification_history": [
                VerificationRecord(
                    action_id="ACT_01_01",
                    verified_at="2026-11-26 12:00:01",
                    passed=False,
                    evidence="Still failing",
                    details={"failed_jobs": ["JOB_03_LOAD"]},
                )
            ],
        }
    )
    assert esc_out["status"] == IncidentStatus.ESCALATED
    assert "budget" in esc_out["final_summary"].lower()


# ---------------------------------------------------------------------------
# Phase 2: Consequential diagnostic & planning reflection
# ---------------------------------------------------------------------------

def test_diagnostic_reflection_blocks_premature_diagnosis_ready(tmp_path: Path):
    """
    If LLM returns DIAGNOSIS_READY while reflection.evidence_sufficient=False,
    triage must NOT transition to PLANNING; it must run the distinguishing
    observation or escalate when budget is exhausted.
    """
    env = MockWorkflowRunner(base_dir=tmp_path / "diag_refl")
    env.inject_scenario(ScenarioType.SERVICE_OFFLINE)
    env.run_workflow()
    runner = create_default_tool_runner(env)

    class ContradictoryDiagLLM(MockLLMClient):
        def investigate(self, *args, **kwargs) -> InvestigationDecision:
            return InvestigationDecision(
                decision_type=InvestigationActionType.DIAGNOSIS_READY,
                diagnosis=TriageResult(
                    failure_type="SERVICE_OFFLINE",
                    implicated_resource="reporting_warehouse_service",
                    triage_summary="Guessing DB offline",
                    root_cause_hypothesis="May be offline",
                    is_recoverable=True,
                ),
                reflection=DiagnosticReflection(
                    evidence_sufficient=False,
                    supporting_observations=[],
                    alternative_hypotheses=["Network ACL block", "Service stopped"],
                    remaining_uncertainty="Have not checked live service status yet.",
                    recommended_diagnostic_tool=DiagnosticToolCall(
                        tool_name="check_service_status",
                        parameters={"service_name": "reporting_warehouse_service"},
                        rationale="Confirm service health before planning",
                    ),
                ),
            )

    triage = create_triage_node(env, ContradictoryDiagLLM(), tool_runner=runner)

    # With budget remaining -> runs recommended diagnostic tool and stays TRIAGING
    out_1 = triage(
        {
            "incident_id": "INC_DIAG_GATE",
            "target_service": "JOB_03_LOAD",
            "raw_symptoms": "JOB_03_LOAD failed",
            "investigation_steps": 0,
            "max_investigation_steps": 3,
            "diagnostic_history": [],
            "investigation_evidence": [],
        }
    )
    assert out_1["status"] == IncidentStatus.TRIAGING
    assert out_1["diagnostic_history"][0].tool_name == "check_service_status"
    assert out_1["reflection_history"][0]["blocked_premature_diagnosis"] is True

    # At budget limit -> MUST ESCALATE, never proceed to PLANNING
    out_exhausted = triage(
        {
            "incident_id": "INC_DIAG_GATE_EXH",
            "target_service": "JOB_03_LOAD",
            "raw_symptoms": "JOB_03_LOAD failed",
            "investigation_steps": 2,
            "max_investigation_steps": 3,
            "diagnostic_history": [],
            "investigation_evidence": [],
        }
    )
    assert out_exhausted["status"] == IncidentStatus.ESCALATED
    assert "insufficient evidence" in out_exhausted["final_summary"].lower()


def test_planning_critique_prevents_inappropriate_action_and_revises_plan(tmp_path: Path):
    """
    Acceptance criterion unit proof:
    Planner initially proposes an inappropriate bare `resume_workflow` when
    `reporting_warehouse_service` is offline; planning critique rejects it,
    preserves initial_plan + critique + final_plan, and revises to restart + resume
    so the inappropriate single-step plan NEVER executes.
    """
    env = MockWorkflowRunner(base_dir=tmp_path / "plan_crit")
    env.inject_scenario(ScenarioType.SERVICE_OFFLINE)
    env.run_workflow()
    runner = create_default_tool_runner(env)

    class CritiqueRevisingLLM(MockLLMClient):
        def __init__(self):
            super().__init__(enable_reflection=True)
            self.plan_calls = 0

        def plan(self, triage, available_tools, retry_count):
            self.plan_calls += 1
            if not triage.critique_feedback:
                # Initial inappropriate plan: bare retry without restarting offline service
                return PlannerResult(
                    is_escalation=False,
                    actions=[
                        PlannedActionSchema(
                            tool_name="resume_workflow",
                            parameters={"workflow_id": "WF_NIGHTLY_BATCH_1001"},
                            risk_level=RiskLevel.LOW_RISK,
                            rationale="Retry workflow immediately",
                            expected_outcome="Workflow passes",
                        )
                    ],
                )
            # Revised plan after critique
            return PlannerResult(
                is_escalation=False,
                actions=[
                    PlannedActionSchema(
                        tool_name="restart_service",
                        parameters={"service_name": "reporting_warehouse_service"},
                        risk_level=RiskLevel.DESTRUCTIVE,
                        rationale="Restart confirmed offline DB before resuming workflow",
                        expected_outcome="Service RUNNING",
                    ),
                    PlannedActionSchema(
                        tool_name="resume_workflow",
                        parameters={"workflow_id": "WF_NIGHTLY_BATCH_1001"},
                        risk_level=RiskLevel.LOW_RISK,
                        rationale="Retry workflow after DB recovery",
                        expected_outcome="Workflow passes",
                    ),
                ],
            )

    llm = CritiqueRevisingLLM()
    graph = build_nova_graph(env, tool_runner=runner, llm=llm, require_approval=False)
    res = graph.invoke(
        {
            "incident_id": "INC_PLAN_CRIT_01",
            "target_service": "UNKNOWN",
            "raw_symptoms": "Nightly batch failed",
            "status": IncidentStatus.TRIGGERED,
            "max_investigation_steps": 6,
            "max_retries": 1,
        }
    )

    assert res["status"] == IncidentStatus.RESOLVED
    assert llm.plan_calls == 2
    assert len(res["plan_revisions"]) == 1
    rev = res["plan_revisions"][0]
    assert rev["outcome"] == "REVISED"
    assert [a["tool_name"] for a in rev["initial_plan"]] == ["resume_workflow"]
    assert [a["tool_name"] for a in rev["final_plan"]] == ["restart_service", "resume_workflow"]
    assert "without restarting" in rev["critique"].lower()
    # The inappropriate initial 1-step plan never executed before restart_service!
    assert [e.tool_name for e in res["execution_history"]] == ["restart_service", "resume_workflow"]


# ---------------------------------------------------------------------------
# Phase 3: Codex's stale-row and upstream-only-attempt probes
# ---------------------------------------------------------------------------

def _make_taxi_probe_env(holder: Dict[str, Any]) -> KestraEnvironment:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        eid = holder["execution"]["id"]
        if request.method == "GET" and path == f"/api/v1/main/executions/{eid}":
            return httpx.Response(200, json=holder["execution"])
        if request.method == "GET" and path == f"/api/v1/main/executions/{eid}/graph":
            return httpx.Response(200, json=holder["graph"])
        if request.method == "GET" and path == f"/api/v1/main/logs/{eid}":
            return httpx.Response(200, json=holder.get("logs", []))
        if request.method == "POST" and path == f"/api/v1/main/executions/{eid}/restart":
            if holder.get("restart_behavior") == "upstream_only_bump":
                # Probe 1: ONLY upstream `extract` gets an incremented attempt;
                # load/merge tasks remain at their stale/unrun state, while execution claims SUCCESS.
                holder["execution"]["state"]["current"] = "SUCCESS"
                for tr in holder["execution"]["taskRunList"]:
                    if tr["taskId"] == "extract":
                        tr["attempts"].append({"state": {"current": "SUCCESS"}})
                    tr["state"]["current"] = "SUCCESS"
            elif holder.get("restart_behavior") == "idempotent_full_retry":
                holder["execution"]["state"]["current"] = "SUCCESS"
                for tr in holder["execution"]["taskRunList"]:
                    tr["state"]["current"] = "SUCCESS"
                    tr["attempts"].append({"state": {"current": "SUCCESS"}})
            return httpx.Response(200, json=holder["execution"])
        return httpx.Response(404, json={})

    def svc_ctrl(action: str, service_name: str):
        if action == "status":
            return (holder.get("pg_healthy", True), "ok")
        if action == "restart":
            holder["pg_healthy"] = True
            return (True, "restarted")
        return (False, "no")

    def pg_verifier(expected: Dict[str, Any]):
        # Simulates pre-existing rows ALREADY present in public.green_tripdata
        return (
            True,
            {"row_count": 35612, "distinct_ids": 35612, "sum_total_amount": 701002.26},
            "PostgreSQL read-only verification passed: row_count=35612.",
        )

    return KestraEnvironment(
        default_execution_id="exec_probe_01",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        service_controller=svc_ctrl,
        pg_verifier=pg_verifier,
    )


def test_codex_probe_upstream_only_attempt_and_stale_pg_rows_rejected():
    """
    Reproduces Codex's two probes:
    1. Upstream `extract` attempt count increases while load/merge tasks have no fresh attempt.
    2. PostgreSQL table ALREADY contains 35,612 stale rows from an earlier run.
    Verifier MUST reject recovery (must NOT return RESOLVED).
    """
    holder: Dict[str, Any] = {
        "pg_healthy": False,
        "restart_behavior": "upstream_only_bump",
        "execution": {
            "id": "exec_probe_01",
            "namespace": "zoomcamp",
            "flowId": "04_postgres_taxi",
            "inputs": {"taxi": "green", "year": "2020", "month": "04"},
            "state": {"current": "FAILED"},
            "taskRunList": [
                {
                    "id": "tr_1",
                    "taskId": "extract",
                    "state": {"current": "SUCCESS"},
                    "attempts": [{"state": {"current": "SUCCESS"}}],
                },
                {
                    "id": "tr_2",
                    "taskId": "green_create_table",
                    "state": {"current": "FAILED"},
                    "attempts": [{"state": {"current": "FAILED"}}],
                },
                # Suppose prior stale attempt existed on merge_data, or it wasn't freshly retried
                {
                    "id": "tr_3",
                    "taskId": "green_copy_in_to_staging_table",
                    "state": {"current": "SUCCESS"},
                    "attempts": [{"state": {"current": "SUCCESS"}}],
                },
                {
                    "id": "tr_4",
                    "taskId": "green_add_unique_id_and_filename",
                    "state": {"current": "SUCCESS"},
                    "attempts": [{"state": {"current": "SUCCESS"}}],
                },
                {
                    "id": "tr_5",
                    "taskId": "green_merge_data",
                    "state": {"current": "SUCCESS"},
                    "attempts": [{"state": {"current": "SUCCESS"}}],
                },
            ],
        },
        "graph": {"nodes": [], "edges": []},
    }
    env = _make_taxi_probe_env(holder)
    verifier = create_verifier_node(env, tool_runner=create_kestra_tool_runner(env))

    # Baseline had total_attempts=5, extract=1, green_create_table=1, merge=1
    baseline_task_attempts = {
        "extract": 1,
        "green_create_table": 1,
        "green_copy_in_to_staging_table": 1,
        "green_add_unique_id_and_filename": 1,
        "green_merge_data": 1,
    }
    baseline_task_states = {
        "extract": "SUCCESS",
        "green_create_table": "FAILED",
        "green_copy_in_to_staging_table": "SUCCESS",
        "green_add_unique_id_and_filename": "SUCCESS",
        "green_merge_data": "SUCCESS",
    }

    # Execute resume_workflow which bumps ONLY `extract` (total_attempts 5 -> 6)
    runner = create_kestra_tool_runner(env)
    holder["pg_healthy"] = True
    res_rec = runner.execute(
        PlannedAction(
            action_id="ACT_01_02",
            tool_name="resume_workflow",
            parameters={"execution_id": "exec_probe_01"},
            risk_level=RiskLevel.LOW_RISK,
            rationale="Resume",
            expected_outcome="Load/merge succeed",
        )
    )
    assert res_rec.success is True

    verif_out = verifier(
        {
            "incident_id": "INC_PROBE_01",
            "status": IncidentStatus.VERIFYING,
            "workflow_id": "zoomcamp/04_postgres_taxi",
            "workflow_run_id": "exec_probe_01",
            "baseline_attempts": 5,
            "baseline_task_attempts": baseline_task_attempts,
            "baseline_task_states": baseline_task_states,
            "plan": [
                PlannedAction(
                    action_id="ACT_01_02",
                    tool_name="resume_workflow",
                    parameters={"execution_id": "exec_probe_01"},
                    risk_level=RiskLevel.LOW_RISK,
                    rationale="Resume",
                    expected_outcome="Load/merge succeed",
                )
            ],
            "current_action_index": 1,
            "execution_history": [res_rec],
            "retry_count": 0,
            "max_retries": 0,
        }
    )
    assert verif_out["status"] == IncidentStatus.ESCALATED
    assert verif_out["verification_history"][0].passed is False
    assert "fresh" in verif_out["verification_history"][0].evidence.lower()


def test_idempotent_taxi_retry_with_fresh_load_merge_attempts_succeeds():
    """
    When PostgreSQL already holds idempotent rows AND the retry produces fresh
    SUCCESS attempts on green_copy_in_to_staging_table, green_add_unique_id_and_filename,
    green_merge_data, and green_create_table, verification PASSES.
    Also verifies Kestra parent flow-control container tasks (`if_green_taxi`) do not
    block recovery when Kestra transitions them to SUCCESS without incrementing attempts.
    """
    holder: Dict[str, Any] = {
        "pg_healthy": True,
        "restart_behavior": "idempotent_full_retry",
        "execution": {
            "id": "exec_probe_01",
            "namespace": "zoomcamp",
            "flowId": "04_postgres_taxi",
            "inputs": {"taxi": "green", "year": "2020", "month": "04"},
            "state": {"current": "FAILED"},
            "taskRunList": [
                {"id": "tr_1", "taskId": "extract", "state": {"current": "SUCCESS"}, "attempts": [{"state": {"current": "SUCCESS"}}]},
                {"id": "tr_if", "taskId": "if_green_taxi", "state": {"current": "FAILED"}, "attempts": [{"state": {"current": "FAILED"}}]},
                {"id": "tr_2", "taskId": "green_create_table", "parentTaskRunId": "tr_if", "state": {"current": "FAILED"}, "attempts": [{"state": {"current": "FAILED"}}]},
            ],
        },
        "graph": {"nodes": [], "edges": []},
    }

    # Override restart to also add fresh load/merge task runs (while leaving container `if_green_taxi` at attempt=1)
    env = _make_taxi_probe_env(holder)
    orig_transport = env._http._transport

    def full_handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path.endswith("/restart"):
            holder["execution"]["state"]["current"] = "SUCCESS"
            for tr in holder["execution"]["taskRunList"]:
                tr["state"]["current"] = "SUCCESS"
                if tr["taskId"] != "if_green_taxi":
                    tr["attempts"].append({"state": {"current": "SUCCESS"}})
            for tid in ("green_copy_in_to_staging_table", "green_add_unique_id_and_filename", "green_merge_data"):
                holder["execution"]["taskRunList"].append(
                    {
                        "id": f"tr_{tid}",
                        "taskId": tid,
                        "parentTaskRunId": "tr_if",
                        "state": {"current": "SUCCESS"},
                        "attempts": [{"state": {"current": "SUCCESS"}}],
                    }
                )
            return httpx.Response(200, json=holder["execution"])
        return orig_transport.handle_request(request)

    env._http = httpx.Client(transport=httpx.MockTransport(full_handler))
    runner = create_kestra_tool_runner(env)
    verifier = create_verifier_node(env, tool_runner=runner)

    res_rec = runner.execute(
        PlannedAction(
            action_id="ACT_01_01",
            tool_name="resume_workflow",
            parameters={"execution_id": "exec_probe_01"},
            risk_level=RiskLevel.LOW_RISK,
            rationale="Idempotent retry",
            expected_outcome="All tasks succeed",
        )
    )
    verif_out = verifier(
        {
            "incident_id": "INC_IDEMP_01",
            "status": IncidentStatus.VERIFYING,
            "workflow_id": "zoomcamp/04_postgres_taxi",
            "workflow_run_id": "exec_probe_01",
            "baseline_attempts": 3,
            "baseline_task_attempts": {"extract": 1, "if_green_taxi": 1, "green_create_table": 1},
            "baseline_task_states": {"extract": "SUCCESS", "if_green_taxi": "FAILED", "green_create_table": "FAILED"},
            "plan": [
                PlannedAction(
                    action_id="ACT_01_01",
                    tool_name="resume_workflow",
                    parameters={"execution_id": "exec_probe_01"},
                    risk_level=RiskLevel.LOW_RISK,
                    rationale="Idempotent retry",
                    expected_outcome="All tasks succeed",
                )
            ],
            "current_action_index": 1,
            "execution_history": [res_rec],
            "retry_count": 0,
            "max_retries": 1,
        }
    )
    assert verif_out["status"] == IncidentStatus.RESOLVED
    assert verif_out["verification_history"][0].passed is True


# ---------------------------------------------------------------------------
# Phase 4: Second workflow DuckDB verification + API execution binding
# ---------------------------------------------------------------------------

def test_duckdb_second_workflow_verification_and_api_binding(tmp_path: Path):
    """
    Tests:
    1. KestraEnvironment.verify_output handles 03_getting_started_data_pipeline
       (DuckDB stored query output URI & record inspection).
    2. FastAPI /incidents/trigger accepts environment='kestra' & execution_id
       and binds the real graph to KestraEnvironment.
    """
    holder: Dict[str, Any] = {
        "execution": {
            "id": "exec_duckdb_01",
            "namespace": "zoomcamp",
            "flowId": "03_getting_started_data_pipeline",
            "inputs": {"columns_to_keep": ["brand", "price"]},
            "state": {"current": "FAILED"},
            "taskRunList": [
                {"id": "tr_1", "taskId": "extract", "state": {"current": "SUCCESS"}, "attempts": [{"state": {"current": "SUCCESS"}}]},
                {"id": "tr_2", "taskId": "transform", "state": {"current": "FAILED"}, "attempts": [{"state": {"current": "FAILED"}}]},
            ],
        },
        "graph": {
            "nodes": [
                {"uid": "root.extract", "task": {"id": "extract"}},
                {"uid": "root.transform", "task": {"id": "transform"}},
                {"uid": "root.query", "task": {"id": "query"}},
            ],
            "edges": [
                {"source": "root.extract", "target": "root.transform"},
                {"source": "root.transform", "target": "root.query"},
            ],
        },
        "logs": [
            {"taskId": "transform", "timestamp": "2026-11-26T12:00:00Z", "level": "ERROR", "attemptNumber": 0, "message": "KeyError: 'nonexistent_col'"},
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        p = request.url.path
        if p == "/api/v1/main/executions/exec_duckdb_01":
            return httpx.Response(200, json=holder["execution"])
        if p == "/api/v1/main/executions/exec_duckdb_01/graph":
            return httpx.Response(200, json=holder["graph"])
        if p == "/api/v1/main/logs/exec_duckdb_01":
            return httpx.Response(200, json=holder["logs"])
        if p == "/api/v1/main/executions/exec_duckdb_01/file":
            return httpx.Response(200, text='{brand:"Apple",avg_price:1299.99}\n{brand:"Samsung",avg_price:899.50}\n')
        return httpx.Response(404, json={})

    kestra_env = KestraEnvironment(
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        service_controller=lambda a, s: (True, "ok"),
    )

    # Simulate completed DuckDB query output and verify read-only output check
    holder["execution"]["state"]["current"] = "SUCCESS"
    holder["execution"]["taskRunList"].append(
        {
            "id": "tr_3",
            "taskId": "query",
            "state": {"current": "SUCCESS"},
            "attempts": [{"state": {"current": "SUCCESS"}}],
            "outputs": {"uri": "kestra:///zoomcamp/03/query/out.ion", "size": 2},
        }
    )
    ok, details, msg = kestra_env.verify_output(execution_id="exec_duckdb_01")
    assert ok is True
    assert details["stored_record_count"] == 2
    assert "DuckDB" in msg

    # Reset back to FAILED and test API binding via IncidentService + TestClient
    holder["execution"]["state"]["current"] = "FAILED"
    holder["execution"]["taskRunList"].pop()
    mock_env = MockWorkflowRunner(base_dir=tmp_path / "mock")
    saver = SqliteSaver(db_path=tmp_path / "api_kestra.sqlite")
    service = IncidentService(
        env=mock_env,
        checkpointer=saver,
        llm=MockLLMClient(),
        kestra_env=kestra_env,
    )
    set_incident_service(service)
    client = TestClient(app)
    try:
        r = client.post(
            "/incidents/trigger",
            json={
                "incident_id": "INC_API_KESTRA_01",
                "environment": "kestra",
                "execution_id": "exec_duckdb_01",
                "target_service": "UNKNOWN",
                "raw_symptoms": "Kestra execution exec_duckdb_01 failed.",
            },
        )
        assert r.status_code == 202
        st = service.wait_for_incident("INC_API_KESTRA_01", timeout=10.0)
        assert st is not None
        assert st.workflow_id == "zoomcamp/03_getting_started_data_pipeline"
        assert st.workflow_run_id == "exec_duckdb_01"
        assert st.status == IncidentStatus.ESCALATED
    finally:
        client.close()
        service.shutdown()
        saver.close()
