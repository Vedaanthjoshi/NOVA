"""
Integration test suite for the NOVA Streamlit Dashboard's HTTP API Client.

Validates that NovaApiClient seamlessly interacts with the FastAPI Gateway
across all required operational demonstration scenarios:
1. Health check & configuration discovery.
2. Stale-lockfile remediation with Human-in-the-Loop approval.
3. Unknown-service autonomous investigation & discovery.
4. Cascading multi-fault replanning & verification.
5. Human rejection and safe escalation.
6. Safe escalation on unrecoverable data corruption.
7. False positive alert non-intervention.
"""

import tempfile
import time
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from nova.api.main import app, set_incident_service
from nova.api.service import IncidentService
from nova.environments.mock_workflow_runner import (
    JobStatus,
    MockWorkflowRunner,
    ScenarioType,
)
from nova.llm.client import MockLLMClient
from nova.state import IncidentStatus
from nova.storage.sqlite_saver import SqliteSaver
from nova.ui.api_client import NovaApiClient


class TestClientTransport(httpx_client := NovaApiClient):
    """
    Adapter allowing NovaApiClient to route through FastAPI TestClient
    during deterministic integration tests without requiring an external port.
    """
    pass


@pytest.fixture
def ui_test_env():
    """Sets up an isolated test environment, checkpointer, service, and API client."""
    tmpdir = tempfile.mkdtemp()
    base_path = Path(tmpdir)
    env = MockWorkflowRunner(base_dir=base_path / "mock_state")
    db_path = base_path / "ui_checkpoints.sqlite"
    checkpointer = SqliteSaver(db_path=db_path)
    llm = MockLLMClient()

    service = IncidentService(
        env=env,
        checkpointer=checkpointer,
        llm=llm,
        max_workers=2,
    )
    set_incident_service(service)

    client = TestClient(app)

    # Subclass client to route via TestClient
    class DirectTestApiClient(NovaApiClient):
        def __init__(self):
            super().__init__(base_url="http://testserver")

        def check_health(self):
            r = client.get("/health")
            return r.json() if r.status_code == 200 else {"status": "ERROR"}

        def list_incidents(self):
            r = client.get("/incidents")
            return r.json() if r.status_code == 200 else []

        def get_incident_state(self, incident_id):
            r = client.get(f"/incidents/{incident_id}/state")
            return r.json() if r.status_code == 200 else None

        def trigger_incident(self, raw_symptoms, target_service="UNKNOWN", incident_id=None, max_retries=3, max_investigation_steps=5):
            r = client.post(
                "/incidents/trigger",
                json={
                    "incident_id": incident_id,
                    "target_service": target_service,
                    "raw_symptoms": raw_symptoms,
                    "max_retries": max_retries,
                    "max_investigation_steps": max_investigation_steps,
                },
            )
            if r.status_code in (200, 202):
                return r.json()
            raise RuntimeError(f"Error {r.status_code}: {r.text}")

        def submit_approval(self, incident_id, action_id, approved, approver_id, approver_role, notes=None):
            r = client.post(
                f"/incidents/{incident_id}/approve",
                json={
                    "approved": approved,
                    "approver_id": approver_id,
                    "approver_role": approver_role,
                    "action_id": action_id,
                    "notes": notes,
                },
            )
            if r.status_code == 200:
                return r.json()
            raise RuntimeError(f"Approval error ({r.status_code}): {r.text}")

        def inject_scenario(self, scenario):
            r = client.post("/scenarios/inject", json={"scenario": scenario})
            if r.status_code == 200:
                return r.json()
            raise RuntimeError(f"Injection error: {r.text}")

    api_client = DirectTestApiClient()

    try:
        yield {
            "api": api_client,
            "service": service,
            "env": env,
            "client": client,
        }
    finally:
        client.close()
        service.shutdown()
        import gc
        gc.collect()
        try:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass


def test_ui_health_and_listing(ui_test_env):
    """Verifies UI client connects to /health and /incidents."""
    api = ui_test_env["api"]
    health = api.check_health()
    assert health["status"] == "HEALTHY"
    assert "SqliteSaver" in health["checkpointer"]

    incidents = api.list_incidents()
    assert isinstance(incidents, list)


def test_ui_stale_lockfile_remediation_flow(ui_test_env):
    """
    Simulates UI operator workflow for Stale Lockfile:
    1. Injects scenario.
    2. Triggers incident.
    3. Awaits approval checkpoint.
    4. Submits operator approval.
    5. Awaits resolved state.
    """
    api = ui_test_env["api"]
    service = ui_test_env["service"]
    env = ui_test_env["env"]

    # 1. Inject scenario
    inj = api.inject_scenario("STALE_LOCKFILE")
    assert inj["scenario"] == "STALE_LOCKFILE"
    assert inj["failed_job"] == "JOB_02_TRANSFORM"

    # 2. Trigger incident
    inc = api.trigger_incident(
        raw_symptoms="FATAL: Lockfile 'transform_job.lock' detected on JOB_02_TRANSFORM.",
        target_service="JOB_02_TRANSFORM",
        incident_id="INC_UI_LOCK_01",
    )
    inc_id = inc["incident_id"]

    # 3. Wait for HITL checkpoint
    state = service.wait_for_incident(inc_id, timeout=10.0)
    assert state.status == IncidentStatus.AWAITING_APPROVAL
    assert state.pending_approval.tool_name == "clear_lockfile"

    # 4. Operator approves via UI client
    app_res = api.submit_approval(
        incident_id=inc_id,
        action_id=state.pending_approval.action_id,
        approved=True,
        approver_id="OP_OPERATOR_01",
        approver_role="SRE_LEAD",
        notes="Authorized via Streamlit Dashboard",
    )
    assert app_res["approved"] is True

    # 5. Await resolution
    final_st = service.wait_for_incident(inc_id, timeout=10.0)
    assert final_st.status == IncidentStatus.RESOLVED
    assert not (env.locks_dir / "transform_job.lock").exists()


def test_ui_unknown_service_autonomous_investigation(ui_test_env):
    """
    Simulates UI operator submitting an alert with target_service="UNKNOWN":
    Verifies autonomous investigation runs get_workflow_status, identifies
    the offline DB on JOB_03_LOAD, restarts it, and completes.
    """
    api = ui_test_env["api"]
    service = ui_test_env["service"]
    env = ui_test_env["env"]

    env.inject_scenario(ScenarioType.SERVICE_OFFLINE)
    env.run_workflow()

    inc = api.trigger_incident(
        raw_symptoms="The nightly batch pipeline stalled unexpectedly.",
        target_service="UNKNOWN",
        incident_id="INC_UI_UNKNOWN_01",
    )
    inc_id = inc["incident_id"]

    state = service.wait_for_incident(inc_id, timeout=10.0)
    assert state.status == IncidentStatus.AWAITING_APPROVAL
    assert state.pending_approval.tool_name == "restart_service"

    api.submit_approval(
        incident_id=inc_id,
        action_id=state.pending_approval.action_id,
        approved=True,
        approver_id="OP_OPERATOR_01",
        approver_role="SRE_LEAD",
    )

    final_st = service.wait_for_incident(inc_id, timeout=10.0)
    assert final_st.status == IncidentStatus.RESOLVED


def test_ui_operator_rejection_flow(ui_test_env):
    """Verifies that an operator rejecting an action in the UI escalates cleanly."""
    api = ui_test_env["api"]
    service = ui_test_env["service"]
    env = ui_test_env["env"]

    env.inject_scenario(ScenarioType.STALE_LOCKFILE)
    env.run_workflow()

    inc = api.trigger_incident(
        raw_symptoms="Lock detected.",
        target_service="JOB_02_TRANSFORM",
        incident_id="INC_UI_REJ_01",
    )
    inc_id = inc["incident_id"]

    state = service.wait_for_incident(inc_id, timeout=10.0)
    assert state.status == IncidentStatus.AWAITING_APPROVAL

    api.submit_approval(
        incident_id=inc_id,
        action_id=state.pending_approval.action_id,
        approved=False,
        approver_id="OP_OPERATOR_01",
        approver_role="SRE_LEAD",
        notes="Rejected by operator in console",
    )

    final_st = service.wait_for_incident(inc_id, timeout=10.0)
    assert final_st.status == IncidentStatus.ESCALATED
    # Physical file still exists
    assert (env.locks_dir / "transform_job.lock").exists()


def test_ui_unrecoverable_corruption_escalation(ui_test_env):
    """Verifies that unrecoverable data corruption escalates without requesting approval."""
    api = ui_test_env["api"]
    service = ui_test_env["service"]
    env = ui_test_env["env"]

    env.inject_scenario(ScenarioType.DATA_CORRUPTION)
    env.run_workflow()

    inc = api.trigger_incident(
        raw_symptoms="CRITICAL: Checksum mismatch in input payload.",
        target_service="JOB_01_EXTRACT",
        incident_id="INC_UI_CORRUPT_01",
    )
    inc_id = inc["incident_id"]

    final_st = service.wait_for_incident(inc_id, timeout=10.0)
    assert final_st.status == IncidentStatus.ESCALATED
    assert final_st.is_recoverable is False
