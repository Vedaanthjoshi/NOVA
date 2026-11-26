"""
Deterministic Unit & Integration Tests for Milestone 12:
- KestraEnvironment adapter & Kestra tool contracts (via httpx.MockTransport).
- Exclusion of mock lockfile tools in Kestra runner.
- Dynamic tool discovery in triage & planner nodes.
- Preservation of failed-action evidence across replanner -> planner.
- Replanner -> Triage reinvestigation routing (bounded).
- Fresh task attempt verification & independent PostgreSQL output verification checks.
"""

from typing import Any, Dict, List
import httpx
import pytest

from nova.environments.kestra_runner import KestraEnvironment
from nova.environments.mock_workflow_runner import JobStatus
from nova.graph import build_nova_graph
from nova.llm.client import MockLLMClient
from nova.llm.schemas import (
    DiagnosticReflection,
    DiagnosticToolCall,
    InvestigationActionType,
    InvestigationDecision,
    PlannedActionSchema,
    PlannerResult,
    PlanningReflection,
    ReplannerResult,
    TriageResult,
)
from nova.state import IncidentStatus, NovaGraphState, RiskLevel
from nova.tools.kestra_tools import create_kestra_tool_runner


def _make_mock_kestra_transport(state_holder: Dict[str, Any]) -> httpx.MockTransport:
    """In-memory Kestra REST API simulator for deterministic testing."""

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        eid = state_holder["execution"]["id"]

        if request.method == "GET" and path == f"/api/v1/main/executions/{eid}":
            return httpx.Response(200, json=state_holder["execution"])

        if request.method == "GET" and path == f"/api/v1/main/executions/{eid}/graph":
            return httpx.Response(200, json=state_holder["graph"])

        if request.method == "GET" and path == f"/api/v1/main/logs/{eid}":
            task_id = request.url.params.get("taskId")
            entries = [
                e for e in state_holder["logs"]
                if not task_id or e.get("taskId") == task_id
            ]
            return httpx.Response(200, json=entries)

        if request.method == "POST" and path == f"/api/v1/main/executions/{eid}/restart":
            if state_holder["execution"]["state"]["current"] == "SUCCESS":
                return httpx.Response(422, json={"message": "No task found to restart execution from!"})
            if state_holder.get("pg_healthy", False):
                state_holder["execution"]["state"]["current"] = "SUCCESS"
                for tr in state_holder["execution"]["taskRunList"]:
                    tr["state"]["current"] = "SUCCESS"
                    tr["attempts"].append({"state": {"current": "SUCCESS"}})
            else:
                # Still failing on DB task with incremented attempt
                for tr in state_holder["execution"]["taskRunList"]:
                    if tr["taskId"] == "green_create_table":
                        tr["attempts"].append({"state": {"current": "FAILED"}})
            return httpx.Response(200, json=state_holder["execution"])

        return httpx.Response(404, json={"message": f"Not found: {path}"})

    return httpx.MockTransport(handler)


@pytest.fixture
def kestra_fixture():
    holder: Dict[str, Any] = {
        "pg_healthy": False,
        "execution": {
            "id": "exec_kestra_001",
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
            ],
        },
        "graph": {
            "nodes": [
                {"uid": "root.extract", "task": {"id": "extract"}},
                {"uid": "root.green_create_table", "task": {"id": "green_create_table"}},
                {"uid": "root.green_copy_in", "task": {"id": "green_copy_in"}},
            ],
            "edges": [
                {"source": "root.extract", "target": "root.green_create_table"},
                {"source": "root.green_create_table", "target": "root.green_copy_in"},
            ],
        },
        "logs": [
            {
                "taskId": "extract",
                "timestamp": "2026-11-26T12:00:00Z",
                "level": "INFO",
                "attemptNumber": 0,
                "message": "Downloaded green_tripdata_2020-04.csv",
            },
            {
                "taskId": "green_create_table",
                "timestamp": "2026-11-26T12:00:05Z",
                "level": "ERROR",
                "attemptNumber": 0,
                "message": "Connection to pgdatabase:5432 refused. Check that hostname and port are correct.",
            },
        ],
    }

    def svc_ctrl(action: str, service_name: str):
        if action == "status":
            if service_name == "pgdatabase":
                ok = holder["pg_healthy"]
                return (ok, "PostgreSQL ready" if ok else "Connection refused on 5432")
            return (True, "HTTP 200")
        if action == "restart" and service_name == "pgdatabase":
            holder["pg_healthy"] = True
            return (True, "Started pgdatabase")
        return (False, "Unsupported")

    def pg_verifier(expected: Dict[str, Any]):
        if not holder.get("pg_valid", True):
            return (False, {"row_count": 0}, "PostgreSQL table empty")
        return (
            True,
            {"row_count": 35612, "distinct_ids": 35612, "sum_total_amount": 701002.26},
            "PostgreSQL read-only verification passed: row_count=35612, distinct_ids=35612.",
        )

    client = httpx.Client(transport=_make_mock_kestra_transport(holder))
    env = KestraEnvironment(
        default_execution_id="exec_kestra_001",
        http_client=client,
        service_controller=svc_ctrl,
        pg_verifier=pg_verifier,
    )
    return env, holder


def test_kestra_tool_runner_excludes_lockfile_tools(kestra_fixture):
    env, _ = kestra_fixture
    runner = create_kestra_tool_runner(env)

    assert runner.is_tool_allowed("get_workflow_status")
    assert runner.is_tool_allowed("read_job_logs")
    assert runner.is_tool_allowed("inspect_job_dependencies")
    assert runner.is_tool_allowed("check_service_status")
    assert runner.is_tool_allowed("restart_service")
    assert runner.is_tool_allowed("resume_workflow")

    # Mock lockfile tools must NOT be registered in Kestra environment
    assert not runner.is_tool_allowed("clear_lockfile")
    assert not runner.is_tool_allowed("inspect_lockfiles")

    # Dynamic schema discovery includes parameter specs
    catalog = {t["name"]: t for t in runner.describe_tools()}
    assert "execution_id" in catalog["resume_workflow"]["parameters"]
    assert "job_id" in catalog["read_job_logs"]["parameters"]


def test_kestra_tools_preserve_actual_task_ids_and_service_policy(kestra_fixture):
    env, _ = kestra_fixture
    runner = create_kestra_tool_runner(env)

    status_res = runner.get_tool("get_workflow_status").run()
    assert status_res.success is True
    assert "green_create_table" in status_res.data["failed_jobs"]
    assert "Scenario:" not in status_res.stdout  # No root cause leakage

    log_res = runner.get_tool("read_job_logs").run(job_id="green_create_table")
    assert log_res.success is True
    assert "pgdatabase:5432 refused" in log_res.stdout

    dep_res = runner.get_tool("inspect_job_dependencies").run(job_id="green_create_table")
    assert dep_res.success is True
    assert dep_res.data["dependencies"] == ["extract"]
    assert dep_res.data["downstream"] == ["green_copy_in"]

    # upstream_source cannot be restarted by NOVA
    ext_restart = runner.get_tool("restart_service").run(service_name="upstream_source")
    assert ext_restart.success is False
    assert "cannot be restarted" in ext_restart.stderr.lower()


def test_kestra_end_to_end_recovery_with_reflection_and_pg_verification(kestra_fixture):
    """
    Tests full LangGraph Kestra flow:
    triage (with diagnostic reflection) -> planner (with planning reflection & dynamic catalog)
    -> executor (restart pgdatabase + resume_workflow) -> verifier (fresh attempts + PG check).
    """
    env, holder = kestra_fixture
    seen_tools_in_planner: List[str] = []
    seen_tools_in_triage: List[str] = []

    class KestraMockLLM(MockLLMClient):
        def investigate(self, target_service, raw_symptoms, investigation_evidence, diagnostic_history, available_tools, step_number):
            seen_tools_in_triage[:] = [t["name"] for t in available_tools]
            tools_called = [d["tool_name"] for d in diagnostic_history]
            if "get_workflow_status" not in tools_called:
                return InvestigationDecision(
                    decision_type=InvestigationActionType.DIAGNOSTIC_ACTION,
                    diagnostic_tool=DiagnosticToolCall(
                        tool_name="get_workflow_status",
                        parameters={},
                        rationale="Discover failed Kestra task",
                    ),
                )
            if "read_job_logs" not in tools_called:
                return InvestigationDecision(
                    decision_type=InvestigationActionType.DIAGNOSTIC_ACTION,
                    diagnostic_tool=DiagnosticToolCall(
                        tool_name="read_job_logs",
                        parameters={"job_id": "green_create_table"},
                        rationale="Inspect failed task log",
                    ),
                )
            if "check_service_status" not in tools_called:
                return InvestigationDecision(
                    decision_type=InvestigationActionType.DIAGNOSTIC_ACTION,
                    diagnostic_tool=DiagnosticToolCall(
                        tool_name="check_service_status",
                        parameters={"service_name": "pgdatabase"},
                        rationale="Confirm database connectivity",
                    ),
                )
            return InvestigationDecision(
                decision_type=InvestigationActionType.DIAGNOSIS_READY,
                diagnosis=TriageResult(
                    failure_type="SERVICE_OFFLINE",
                    implicated_resource="pgdatabase",
                    triage_summary="Task green_create_table failed connecting to pgdatabase:5432.",
                    root_cause_hypothesis="pgdatabase is offline.",
                    is_recoverable=True,
                ),
                reflection=DiagnosticReflection(
                    evidence_sufficient=True,
                    alternative_hypotheses=["SQL syntax error ruled out by connection refused log"],
                    remaining_uncertainty=None,
                ),
            )

        def plan(self, triage, available_tools, retry_count):
            seen_tools_in_planner[:] = [t["name"] for t in available_tools]
            return PlannerResult(
                is_escalation=False,
                actions=[
                    PlannedActionSchema(
                        tool_name="restart_service",
                        parameters={"service_name": "pgdatabase"},
                        risk_level=RiskLevel.DESTRUCTIVE,
                        rationale="Bring pgdatabase back online",
                        expected_outcome="pgdatabase healthy",
                    ),
                    PlannedActionSchema(
                        tool_name="resume_workflow",
                        parameters={"execution_id": triage.workflow_run_id},
                        risk_level=RiskLevel.LOW_RISK,
                        rationale="Restart failed Kestra execution",
                        expected_outcome="Kestra execution completes SUCCESS",
                    ),
                ],
                reflection=PlanningReflection(
                    addresses_root_cause=True,
                    contradicts_prior_attempts=False,
                    strategy_assessment="Restarts confirmed offline DB then restarts bound Kestra execution.",
                ),
            )

    graph = build_nova_graph(env, llm=KestraMockLLM(), require_approval=False)
    res = graph.invoke(
        {
            "incident_id": "INC_KESTRA_01",
            "target_service": "UNKNOWN",
            "raw_symptoms": "Kestra execution exec_kestra_001 failed.",
            "status": IncidentStatus.TRIGGERED,
            "workflow_id": "zoomcamp/04_postgres_taxi",
            "workflow_run_id": "exec_kestra_001",
            "expected_output": {
                "taxi": "green",
                "year": "2020",
                "month": "04",
                "expected_row_count": 35612,
                "expected_distinct_ids": 35612,
            },
        }
    )

    assert res["status"] == IncidentStatus.RESOLVED
    assert "clear_lockfile" not in seen_tools_in_planner
    assert "inspect_lockfiles" not in seen_tools_in_triage
    assert res["verification_history"][-1].passed is True
    assert res["verification_history"][-1].details["output_verification"]["row_count"] == 35612
    assert len(res["reflection_history"]) >= 2

    # Ensure PostgreSQL verification failure blocks RESOLVED even when Kestra tasks succeed
    holder["pg_valid"] = False
    holder["execution"]["state"]["current"] = "FAILED"
    holder["execution"]["taskRunList"][1]["state"]["current"] = "FAILED"
    res_bad_pg = graph.invoke(
        {
            "incident_id": "INC_KESTRA_BAD_PG",
            "target_service": "UNKNOWN",
            "raw_symptoms": "Kestra execution exec_kestra_001 failed.",
            "status": IncidentStatus.TRIGGERED,
            "workflow_id": "zoomcamp/04_postgres_taxi",
            "workflow_run_id": "exec_kestra_001",
            "max_retries": 0,
        }
    )
    assert res_bad_pg["status"] == IncidentStatus.ESCALATED
    assert res_bad_pg["verification_history"][-1].passed is False


def test_replanner_routes_back_to_triage_for_further_investigation_preserving_prior_evidence(kestra_fixture):
    """
    Phase 3C & 3D: When remediation fails, replanner preserves failed-action evidence
    and routes back to triage (`status=TRIAGING`) to gather additional evidence before replanning.
    """
    env, holder = kestra_fixture
    planner_saw_prior_attempts: List[Any] = []

    class ReinvestigateLLM(MockLLMClient):
        def investigate(self, target_service, raw_symptoms, investigation_evidence, diagnostic_history, available_tools, step_number):
            tools_called = [d["tool_name"] for d in diagnostic_history]
            has_outcome_reflection = any("[Outcome Reflection" in ev for ev in investigation_evidence)
            if "get_workflow_status" not in tools_called:
                return InvestigationDecision(
                    decision_type=InvestigationActionType.DIAGNOSTIC_ACTION,
                    diagnostic_tool=DiagnosticToolCall(
                        tool_name="get_workflow_status",
                        parameters={},
                        rationale="Initial check",
                    ),
                )
            if has_outcome_reflection and "check_service_status" not in tools_called:
                return InvestigationDecision(
                    decision_type=InvestigationActionType.DIAGNOSTIC_ACTION,
                    diagnostic_tool=DiagnosticToolCall(
                        tool_name="check_service_status",
                        parameters={"service_name": "pgdatabase"},
                        rationale="Re-investigating DB dependency after initial resume failed",
                    ),
                )
            if "check_service_status" in tools_called:
                return InvestigationDecision(
                    decision_type=InvestigationActionType.DIAGNOSIS_READY,
                    diagnosis=TriageResult(
                        failure_type="SERVICE_OFFLINE",
                        implicated_resource="pgdatabase",
                        triage_summary="Confirmed pgdatabase is offline.",
                        root_cause_hypothesis="pgdatabase stopped.",
                        is_recoverable=True,
                    ),
                )
            # Initial premature diagnosis before checking DB
            return InvestigationDecision(
                decision_type=InvestigationActionType.DIAGNOSIS_READY,
                diagnosis=TriageResult(
                    failure_type="TRANSIENT",
                    implicated_resource="green_create_table",
                    triage_summary="Task green_create_table failed.",
                    root_cause_hypothesis="Transient failure.",
                    is_recoverable=True,
                ),
            )

        def plan(self, triage, available_tools, retry_count):
            if triage.prior_attempts:
                planner_saw_prior_attempts.extend(triage.prior_attempts)
            if triage.implicated_resource == "pgdatabase":
                return PlannerResult(
                    is_escalation=False,
                    actions=[
                        PlannedActionSchema(
                            tool_name="restart_service",
                            parameters={"service_name": "pgdatabase"},
                            risk_level=RiskLevel.DESTRUCTIVE,
                            rationale="Restart DB",
                            expected_outcome="DB online",
                        ),
                        PlannedActionSchema(
                            tool_name="resume_workflow",
                            parameters={"execution_id": triage.workflow_run_id},
                            risk_level=RiskLevel.LOW_RISK,
                            rationale="Resume workflow",
                            expected_outcome="Workflow succeeds",
                        ),
                    ],
                )
            return PlannerResult(
                is_escalation=False,
                actions=[
                    PlannedActionSchema(
                        tool_name="resume_workflow",
                        parameters={"execution_id": triage.workflow_run_id},
                        risk_level=RiskLevel.LOW_RISK,
                        rationale="Retry workflow",
                        expected_outcome="Workflow succeeds",
                    )
                ],
            )

        def replan(self, *args, **kwargs):
            return ReplannerResult(
                new_target_service="green_create_table",
                updated_triage_summary="Retry failed; need to check underlying warehouse service health.",
                updated_hypothesis="Warehouse DB may be offline.",
                new_failure_type="UNKNOWN",
                implicated_resource="green_create_table",
                is_recoverable=True,
                diagnosis_revision_needed=True,
                needs_further_investigation=True,
                outcome_reflection="resume_workflow failed again without fixing a dependency; re-investigate.",
            )

    graph = build_nova_graph(env, llm=ReinvestigateLLM(), require_approval=False)
    res = graph.invoke(
        {
            "incident_id": "INC_REINVEST_01",
            "target_service": "UNKNOWN",
            "raw_symptoms": "Execution exec_kestra_001 failed.",
            "status": IncidentStatus.TRIGGERED,
            "workflow_id": "zoomcamp/04_postgres_taxi",
            "workflow_run_id": "exec_kestra_001",
            "max_investigation_steps": 6,
            "max_retries": 2,
        }
    )

    assert res["status"] == IncidentStatus.RESOLVED
    assert any(d.tool_name == "check_service_status" for d in res["diagnostic_history"])
    assert len(planner_saw_prior_attempts) >= 1
    assert any(p.get("tool_name") == "resume_workflow" and p.get("success") is False for p in planner_saw_prior_attempts)
