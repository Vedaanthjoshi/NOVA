"""
Milestone 10 Hardening & Full-Stack Integration Tests.

Covers:
1. Active lockfile protection (tool-level and investigation-level).
2. False-positive alert non-intervention.
3. Cascading multi-fault replanning through IncidentService + FastAPI + SQLite.
4. Approval replay, duplicate submission, blank operator ID, and action_hash binding.
5. Backend restart while awaiting human approval + incident history retrieval after restart.
6. Interrupted background task detection after restart.
7. Gemini API failure handling without leaking secrets or leaving incidents stuck.
8. Concurrent simulator thread safety on MockWorkflowRunner.
"""

import concurrent.futures
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from fastapi.testclient import TestClient

from nova.api.main import app, set_incident_service
from nova.api.service import IncidentService
from nova.environments.mock_workflow_runner import (
    JobStatus,
    MockWorkflowRunner,
    ScenarioType,
)
from nova.llm.client import BaseLLMClient, MockLLMClient
from nova.llm.schemas import (
    InvestigationDecision,
    PlannedActionSchema,
    PlannerResult,
    ReplannerResult,
    TriageResult,
)
from nova.observability import telemetry
from nova.state import IncidentStatus, PlannedAction, RiskLevel
from nova.storage.sqlite_saver import SqliteSaver
from nova.tools.runner import create_default_tool_runner


def _make_service(base_path: Path, llm: Optional[BaseLLMClient] = None):
    env = MockWorkflowRunner(base_dir=base_path / "mock_state")
    db_path = base_path / "m10_checkpoints.sqlite"
    checkpointer = SqliteSaver(db_path=db_path)
    service = IncidentService(
        env=env,
        checkpointer=checkpointer,
        llm=llm or MockLLMClient(),
        max_workers=2,
    )
    set_incident_service(service)
    return service, env, checkpointer, db_path


def test_active_lockfile_protection_tool_and_graph(tmp_path: Path):
    """
    Active worker lockfiles (STATUS=ACTIVE_WORKER) must NOT be deleted:
    - ClearLockfileTool refuses deletion and returns success=False.
    - Autonomous investigation recognizes active worker lock and escalates safely.
    """
    env = MockWorkflowRunner(base_dir=tmp_path / "mock_state")
    env.inject_scenario(ScenarioType.STALE_LOCKFILE)
    lock_path = env.locks_dir / "transform_job.lock"
    lock_path.write_text(
        "PID=10492 STATUS=ACTIVE_WORKER HEARTBEAT=RECENT OWNER=worker_03\n",
        encoding="utf-8",
    )
    env.run_workflow()

    # 1. Tool-level protection
    runner = create_default_tool_runner(env)
    rec = runner.execute(
        PlannedAction(
            action_id="ACT_ACTLOCK_01",
            tool_name="clear_lockfile",
            parameters={"lock_name": "transform_job.lock"},
            risk_level=RiskLevel.DESTRUCTIVE,
            rationale="Attempt lock removal",
            expected_outcome="Refused if active",
        )
    )
    assert rec.success is False
    assert "Safety Guard: Refusing to delete active lockfile" in rec.stderr
    assert lock_path.exists()

    # 2. Full API / Service flow escalates without deleting active lock
    service, _, _, _ = _make_service(tmp_path / "svc")
    service.env.inject_scenario(ScenarioType.STALE_LOCKFILE)
    svc_lock = service.env.locks_dir / "transform_job.lock"
    svc_lock.write_text(
        "PID=10492 STATUS=ACTIVE_WORKER HEARTBEAT=RECENT OWNER=worker_03\n",
        encoding="utf-8",
    )
    service.env.run_workflow()

    client = TestClient(app)
    try:
        r = client.post(
            "/incidents/trigger",
            json={
                "incident_id": "INC_ACTIVE_LOCK_01",
                "target_service": "JOB_02_TRANSFORM",
                "raw_symptoms": "JOB_02_TRANSFORM blocked on lock 'transform_job.lock'",
            },
        )
        assert r.status_code == 202
        final_state = service.wait_for_incident("INC_ACTIVE_LOCK_01", timeout=10.0)
        assert final_state is not None
        assert final_state.status == IncidentStatus.ESCALATED
        assert len(final_state.execution_history) == 0
        assert svc_lock.exists()
    finally:
        client.close()
        service.shutdown()


def test_false_positive_alert_non_intervention(tmp_path: Path):
    """
    When all batch jobs are SUCCESS and target service is healthy,
    NOVA investigates using read-only tools and escalates without mutating state.
    """
    service, env, _, _ = _make_service(tmp_path)
    env.inject_scenario(ScenarioType.HAPPY_PATH)
    wf = env.run_workflow()
    assert all(j.status == JobStatus.SUCCESS for j in wf.jobs.values())

    client = TestClient(app)
    try:
        r = client.post(
            "/incidents/trigger",
            json={
                "incident_id": "INC_FALSE_POS_01",
                "target_service": "reporting_warehouse_service",
                "raw_symptoms": "Alert: reporting_warehouse_service may be unreachable.",
            },
        )
        assert r.status_code == 202
        final_state = service.wait_for_incident("INC_FALSE_POS_01", timeout=10.0)
        assert final_state is not None
        assert final_state.status == IncidentStatus.ESCALATED
        assert "False-positive alert" in (final_state.final_summary or "")
        assert len(final_state.execution_history) == 0
        assert len(final_state.diagnostic_history) >= 1
        assert env.check_service_status("reporting_warehouse_service")["healthy"] is True
    finally:
        client.close()
        service.shutdown()


def test_cascading_failure_replanning_and_telemetry_via_api(tmp_path: Path):
    """
    Verifies full cascading multi-fault cycle through FastAPI + IncidentService + SQLite:
    clear_lockfile (HITL #1) -> verifier fails on JOB_03_LOAD -> replanner -> restart_service (HITL #2) -> RESOLVED.
    """
    service, env, _, _ = _make_service(tmp_path)
    env.inject_scenario(ScenarioType.CASCADING_FAILURE)
    env.run_workflow()

    client = TestClient(app)
    try:
        r = client.post(
            "/incidents/trigger",
            json={
                "incident_id": "INC_CASCADE_API_01",
                "target_service": "UNKNOWN",
                "raw_symptoms": "Nightly batch workflow halted.",
            },
        )
        assert r.status_code == 202

        st1 = service.wait_for_incident("INC_CASCADE_API_01", timeout=10.0)
        assert st1 is not None
        assert st1.status == IncidentStatus.AWAITING_APPROVAL
        assert st1.pending_approval is not None
        assert st1.pending_approval.tool_name == "clear_lockfile"

        # Approve step 1
        ap1 = client.post(
            "/incidents/INC_CASCADE_API_01/approve",
            json={
                "approved": True,
                "approver_id": "sre_lead_1",
                "approver_role": "SRE_LEAD",
                "action_id": st1.pending_approval.action_id,
                "action_hash": st1.pending_approval.action_hash,
            },
        )
        assert ap1.status_code == 200

        # Wait for cycle 2 HITL pause (restart_service)
        st2 = service.wait_for_incident("INC_CASCADE_API_01", timeout=10.0)
        assert st2 is not None
        assert st2.status == IncidentStatus.AWAITING_APPROVAL
        assert st2.pending_approval is not None
        assert st2.pending_approval.tool_name == "restart_service"
        assert st2.retry_count == 1

        # Approve step 2
        ap2 = client.post(
            "/incidents/INC_CASCADE_API_01/approve",
            json={
                "approved": True,
                "approver_id": "sre_lead_1",
                "approver_role": "SRE_LEAD",
                "action_id": st2.pending_approval.action_id,
                "action_hash": st2.pending_approval.action_hash,
            },
        )
        assert ap2.status_code == 200

        final_st = service.wait_for_incident("INC_CASCADE_API_01", timeout=10.0)
        assert final_st is not None
        assert final_st.status == IncidentStatus.RESOLVED
        assert len(final_st.execution_history) == 2
        assert len(final_st.verification_history) == 2
        assert len(final_st.approval_audit_history) == 2
        assert final_st.telemetry is not None
        assert final_st.telemetry["llm_request_count"] >= 4
        assert final_st.telemetry["diagnostic_tool_count"] >= 2
        assert final_st.telemetry["remediation_tool_count"] == 2

        # Verify physical simulator consistency
        wf = env.get_state()
        assert wf is not None
        assert all(j.status == JobStatus.SUCCESS for j in wf.jobs.values())
    finally:
        client.close()
        service.shutdown()


def test_approval_replay_duplicate_blank_operator_and_hash_binding(tmp_path: Path):
    """
    Tests approval hardening:
    - Blank approver_id rejected (422).
    - Mismatched action_hash rejected (400).
    - Replayed approval for already-consumed action rejected (409).
    """
    service, env, _, _ = _make_service(tmp_path)
    env.inject_scenario(ScenarioType.STALE_LOCKFILE)
    env.run_workflow()

    client = TestClient(app)
    try:
        client.post(
            "/incidents/trigger",
            json={
                "incident_id": "INC_REPLAY_01",
                "target_service": "JOB_02_TRANSFORM",
                "raw_symptoms": "Lockfile 'transform_job.lock' detected.",
            },
        )
        st = service.wait_for_incident("INC_REPLAY_01", timeout=10.0)
        assert st is not None
        assert st.pending_approval is not None
        act_id = st.pending_approval.action_id

        # 1. Blank approver_id -> 422
        r_blank = client.post(
            "/incidents/INC_REPLAY_01/approve",
            json={
                "approved": True,
                "approver_id": "   ",
                "approver_role": "SRE_LEAD",
                "action_id": act_id,
            },
        )
        assert r_blank.status_code == 422

        # 2. Wrong action_hash -> 400
        r_hash = client.post(
            "/incidents/INC_REPLAY_01/approve",
            json={
                "approved": True,
                "approver_id": "op_alice",
                "approver_role": "SRE_LEAD",
                "action_id": act_id,
                "action_hash": "deadbeef00000000",
            },
        )
        assert r_hash.status_code == 400
        assert "action hash mismatch" in r_hash.json()["detail"].lower()

        # 3. Valid approval -> 200
        r_ok = client.post(
            "/incidents/INC_REPLAY_01/approve",
            json={
                "approved": True,
                "approver_id": "op_alice",
                "approver_role": "SRE_LEAD",
                "action_id": act_id,
                "action_hash": st.pending_approval.action_hash,
            },
        )
        assert r_ok.status_code == 200

        # 4. Immediate replay (either in-flight or already consumed) -> 409 Conflict
        r_dup = client.post(
            "/incidents/INC_REPLAY_01/approve",
            json={
                "approved": True,
                "approver_id": "op_alice",
                "approver_role": "SRE_LEAD",
                "action_id": act_id,
            },
        )
        assert r_dup.status_code == 409

        final_st = service.wait_for_incident("INC_REPLAY_01", timeout=10.0)
        assert final_st is not None
        assert final_st.status == IncidentStatus.RESOLVED

        # 5. Post-resolution replay -> 409 Conflict
        r_replay = client.post(
            "/incidents/INC_REPLAY_01/approve",
            json={
                "approved": True,
                "approver_id": "op_alice",
                "approver_role": "SRE_LEAD",
                "action_id": act_id,
            },
        )
        assert r_replay.status_code == 409
    finally:
        client.close()
        service.shutdown()


def test_backend_restart_while_awaiting_approval_and_history_retrieval(tmp_path: Path):
    """
    Simulates a full backend restart while an incident is paused at AWAITING_APPROVAL:
    1. Service 1 runs incident to HITL interrupt and shuts down.
    2. Service 2 starts cleanly against the same SQLite database and mock_state directory.
    3. GET /incidents retrieves the historical/paused incident after restart.
    4. POST /incidents/{id}/approve resumes and resolves the incident.
    5. Replay of approval on Service 2 returns 409 Conflict.
    """
    service1, env1, saver1, db_path = _make_service(tmp_path)
    env1.inject_scenario(ScenarioType.STALE_LOCKFILE)
    env1.run_workflow()

    client1 = TestClient(app)
    client1.post(
        "/incidents/trigger",
        json={
            "incident_id": "INC_RESTART_HITL_01",
            "target_service": "JOB_02_TRANSFORM",
            "raw_symptoms": "Lockfile 'transform_job.lock' detected.",
        },
    )
    st1 = service1.wait_for_incident("INC_RESTART_HITL_01", timeout=10.0)
    assert st1 is not None
    assert st1.status == IncidentStatus.AWAITING_APPROVAL
    action_id = st1.pending_approval.action_id
    action_hash = st1.pending_approval.action_hash

    client1.close()
    service1.shutdown()
    saver1.close()

    # Start fresh Service 2 pointing at the exact same SQLite DB & mock_state
    service2, env2, saver2, _ = _make_service(tmp_path)
    client2 = TestClient(app)
    try:
        # History retrieval after restart
        list_resp = client2.get("/incidents")
        assert list_resp.status_code == 200
        ids = [item["incident_id"] for item in list_resp.json()]
        assert "INC_RESTART_HITL_01" in ids

        state_resp = client2.get("/incidents/INC_RESTART_HITL_01/state")
        assert state_resp.status_code == 200
        state_data = state_resp.json()
        assert state_data["status"] == "AWAITING_APPROVAL"
        assert state_data["pending_approval"]["action_id"] == action_id

        # Approve on fresh backend instance
        approve_resp = client2.post(
            "/incidents/INC_RESTART_HITL_01/approve",
            json={
                "approved": True,
                "approver_id": "sre_after_restart",
                "approver_role": "PLATFORM_ADMIN",
                "action_id": action_id,
                "action_hash": action_hash,
                "notes": "Approved after backend restart",
            },
        )
        assert approve_resp.status_code == 200

        final_st = service2.wait_for_incident("INC_RESTART_HITL_01", timeout=10.0)
        assert final_st is not None
        assert final_st.status == IncidentStatus.RESOLVED
        assert not (env2.locks_dir / "transform_job.lock").exists()

        # Replay after resolution must still be rejected across restart
        replay_resp = client2.post(
            "/incidents/INC_RESTART_HITL_01/approve",
            json={
                "approved": True,
                "approver_id": "sre_after_restart",
                "approver_role": "PLATFORM_ADMIN",
                "action_id": action_id,
            },
        )
        assert replay_resp.status_code == 409
    finally:
        client2.close()
        service2.shutdown()
        saver2.close()


def test_telemetry_preserved_across_restart_and_approval_resumption(tmp_path: Path):
    """
    Regression test for telemetry reset across backend restart + approval resumption:
    Even when in-memory telemetry is wiped between HITL pause and approval resumption
    (simulating a fresh backend process), prior LLM and diagnostic tool counts must be
    hydrated from SQLite and preserved in the final RESOLVED telemetry.
    """
    service1, env1, saver1, _ = _make_service(tmp_path)
    env1.inject_scenario(ScenarioType.STALE_LOCKFILE)
    env1.run_workflow()

    client1 = TestClient(app)
    client1.post(
        "/incidents/trigger",
        json={
            "incident_id": "INC_TELEM_PERSIST_01",
            "target_service": "UNKNOWN",
            "raw_symptoms": "Batch workflow stalled.",
        },
    )
    st1 = service1.wait_for_incident("INC_TELEM_PERSIST_01", timeout=10.0)
    assert st1 is not None
    assert st1.status == IncidentStatus.AWAITING_APPROVAL
    assert st1.telemetry is not None
    pre_llm_count = st1.telemetry["llm_request_count"]
    pre_diag_count = st1.telemetry["diagnostic_tool_count"]
    assert pre_llm_count >= 3
    assert pre_diag_count >= 2
    action_id = st1.pending_approval.action_id

    client1.close()
    service1.shutdown()
    saver1.close()

    # Wipe in-memory telemetry registry to simulate a brand new OS process
    telemetry._incidents.clear()

    service2, _, saver2, _ = _make_service(tmp_path)
    client2 = TestClient(app)
    try:
        ap = client2.post(
            "/incidents/INC_TELEM_PERSIST_01/approve",
            json={
                "approved": True,
                "approver_id": "sre_telem",
                "approver_role": "SRE_LEAD",
                "action_id": action_id,
            },
        )
        assert ap.status_code == 200

        final_st = service2.wait_for_incident("INC_TELEM_PERSIST_01", timeout=10.0)
        assert final_st is not None
        assert final_st.status == IncidentStatus.RESOLVED
        assert final_st.telemetry is not None
        assert final_st.telemetry["llm_request_count"] >= pre_llm_count
        assert final_st.telemetry["diagnostic_tool_count"] >= pre_diag_count
        assert final_st.telemetry["remediation_tool_count"] >= 1
        assert final_st.telemetry["approval_audit_count"] == 1
    finally:
        client2.close()
        service2.shutdown()
        saver2.close()


def test_diagnostic_check_service_status_in_plan_does_not_fail_verification(tmp_path: Path):
    """
    Regression test for diagnostic exit-code bug:
    When a remediation plan starts with check_service_status on an OFFLINE service
    followed by restart_service, check_service_status must succeed (exit_code=0) in
    observing Healthy: False so the plan proceeds to restart_service and resolves
    cleanly without repeated verification failures.
    """

    class DiagnosticThenRestartLLM(MockLLMClient):
        def plan(self, triage, available_tools, retry_count: int) -> PlannerResult:
            return PlannerResult(
                is_escalation=False,
                actions=[
                    PlannedActionSchema(
                        tool_name="check_service_status",
                        parameters={"service_name": "reporting_warehouse_service"},
                        risk_level=RiskLevel.READ_ONLY,
                        rationale="Confirm service health before restarting",
                        expected_outcome="Service status observed",
                    ),
                    PlannedActionSchema(
                        tool_name="restart_service",
                        parameters={"service_name": "reporting_warehouse_service"},
                        risk_level=RiskLevel.DESTRUCTIVE,
                        rationale="Restart offline warehouse service",
                        expected_outcome="Service is RUNNING",
                    ),
                ],
            )

    service, env, saver, _ = _make_service(tmp_path, llm=DiagnosticThenRestartLLM())
    env.inject_scenario(ScenarioType.SERVICE_OFFLINE)
    env.run_workflow()

    client = TestClient(app)
    try:
        r = client.post(
            "/incidents/trigger",
            json={
                "incident_id": "INC_DIAG_PLAN_01",
                "target_service": "JOB_03_LOAD",
                "raw_symptoms": "Connection to 'reporting_warehouse_service' refused on port 5432.",
            },
        )
        assert r.status_code == 202

        st = service.wait_for_incident("INC_DIAG_PLAN_01", timeout=10.0)
        assert st is not None
        assert st.status == IncidentStatus.AWAITING_APPROVAL
        assert st.pending_approval is not None
        assert st.pending_approval.tool_name == "restart_service"
        assert len(st.execution_history) == 1
        assert st.execution_history[0]["tool_name"] == "check_service_status"
        assert st.execution_history[0]["success"] is True
        assert st.execution_history[0]["exit_code"] == 0

        ap = client.post(
            "/incidents/INC_DIAG_PLAN_01/approve",
            json={
                "approved": True,
                "approver_id": "sre_diag",
                "approver_role": "SRE_LEAD",
                "action_id": st.pending_approval.action_id,
            },
        )
        assert ap.status_code == 200

        final_st = service.wait_for_incident("INC_DIAG_PLAN_01", timeout=10.0)
        assert final_st is not None
        assert final_st.status == IncidentStatus.RESOLVED
        assert len(final_st.verification_history) == 1
        assert final_st.verification_history[0]["passed"] is True
    finally:
        client.close()
        service.shutdown()
        saver.close()


def test_llm_api_failure_transitions_to_failed_without_leaking_secrets(tmp_path: Path):
    """
    When the LLM client raises an API/rate-limit failure, IncidentService transitions
    the incident cleanly to FAILED and redacts any embedded key material.
    """

    class FailingLLMClient(BaseLLMClient):
        def investigate(self, *args, **kwargs) -> InvestigationDecision:
            raise RuntimeError(
                "429 Quota exceeded using key AIzaSySecretKeyShouldNeverLeak1234567890"
            )

        def triage(self, *args, **kwargs) -> TriageResult:
            raise RuntimeError(
                "429 Quota exceeded using key AIzaSySecretKeyShouldNeverLeak1234567890"
            )

        def plan(self, *args, **kwargs) -> PlannerResult:
            raise RuntimeError("Unavailable")

        def replan(self, *args, **kwargs) -> ReplannerResult:
            raise RuntimeError("Unavailable")

    service, env, saver, _ = _make_service(tmp_path, llm=FailingLLMClient())
    env.inject_scenario(ScenarioType.STALE_LOCKFILE)
    env.run_workflow()

    client = TestClient(app)
    try:
        r = client.post(
            "/incidents/trigger",
            json={
                "incident_id": "INC_LLM_FAIL_01",
                "target_service": "JOB_02_TRANSFORM",
                "raw_symptoms": "Transform failed",
            },
        )
        assert r.status_code == 202
        final_st = service.wait_for_incident("INC_LLM_FAIL_01", timeout=10.0)
        assert final_st is not None
        assert final_st.status == IncidentStatus.FAILED
        assert final_st.final_summary is not None
        assert "AIzaSySecretKeyShouldNeverLeak" not in final_st.final_summary
        assert "[REDACTED]" in final_st.final_summary
    finally:
        client.close()
        service.shutdown()
        saver.close()


def test_mock_workflow_runner_concurrent_thread_safety(tmp_path: Path):
    """
    Exercises concurrent reads, workflow runs, and status checks on shared MockWorkflowRunner
    to verify atomic state writes and thread locking prevent JSON corruption.
    """
    env = MockWorkflowRunner(base_dir=tmp_path / "concurrent_env")
    env.inject_scenario(ScenarioType.STALE_LOCKFILE)
    errors: List[Exception] = []

    def worker(idx: int) -> None:
        try:
            for _ in range(5):
                env.run_workflow()
                st = env.get_state()
                assert st is not None
                env.check_service_status("reporting_warehouse_service")
                env.read_job_log("JOB_02_TRANSFORM", tail_lines=10)
        except Exception as exc:
            errors.append(exc)

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futs = [pool.submit(worker, i) for i in range(6)]
        for f in futs:
            f.result()

    assert errors == []


def test_interrupted_background_task_detection_after_restart(tmp_path: Path):
    """
    If a backend restarts while an incident is mid-execution (non-terminal status
    with no active interrupt and no running worker thread), IncidentService surfaces
    worker_active=False and interrupted_by_restart=True.
    """
    service1, env1, saver1, _ = _make_service(tmp_path)
    env1.inject_scenario(ScenarioType.STALE_LOCKFILE)
    env1.run_workflow()

    # Persist an in-progress checkpoint (e.g. TRIAGING) simulating crash before HITL/terminal
    thread_config = {"configurable": {"thread_id": "INC_INTERRUPTED_01"}}
    service1.graph.update_state(
        thread_config,
        {
            "incident_id": "INC_INTERRUPTED_01",
            "target_service": "JOB_02_TRANSFORM",
            "raw_symptoms": "Interrupted mid-triage",
            "status": IncidentStatus.TRIAGING,
        },
    )
    service1.shutdown()
    saver1.close()

    service2, _, saver2, _ = _make_service(tmp_path)
    client2 = TestClient(app)
    try:
        resp = client2.get("/incidents/INC_INTERRUPTED_01/state")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "TRIAGING"
        assert data["worker_active"] is False
        assert data["interrupted_by_restart"] is True
    finally:
        client2.close()
        service2.shutdown()
        saver2.close()
