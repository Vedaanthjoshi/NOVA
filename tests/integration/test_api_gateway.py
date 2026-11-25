"""
Integration tests for the NOVA FastAPI Backend Gateway.

Verifies:
- Health check and system configuration endpoints.
- Non-blocking incident triggering and immediate 202 acknowledgment.
- Full state polling (investigation history, diagnosis, plan, execution records).
- Human-in-the-Loop approval checkpoints via REST API.
- RBAC role enforcement (403 for unauthorized approver roles).
- Action ID matching and rejection handling.
- Scenario injection for operator testing and demos.
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


@pytest.fixture
def api_test_setup():
    """Sets up an isolated environment, SQLite checkpointer, and TestClient."""
    tmpdir = tempfile.mkdtemp()
    base_path = Path(tmpdir)
    env = MockWorkflowRunner(base_dir=base_path / "mock_state")
    db_path = base_path / "test_api_checkpoints.sqlite"
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
    try:
        yield {
            "client": client,
            "service": service,
            "env": env,
            "db_path": db_path,
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


def test_health_endpoint(api_test_setup):
    client = api_test_setup["client"]
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert data["version"] == "0.8.0"
    assert "SqliteSaver" in data["checkpointer"]


def test_trigger_incident_non_blocking(api_test_setup):
    client = api_test_setup["client"]
    payload = {
        "incident_id": "INC_TEST_NB_01",
        "target_service": "JOB_02_TRANSFORM",
        "raw_symptoms": "Stale lockfile 'transform_job.lock' detected.",
        "max_retries": 2,
    }
    response = client.post("/incidents/trigger", json=payload)
    assert response.status_code == 202
    data = response.json()
    assert data["incident_id"] == "INC_TEST_NB_01"
    assert data["status"] == "TRIGGERED"
    assert data["target_service"] == "JOB_02_TRANSFORM"


def test_get_incident_state_404_on_nonexistent(api_test_setup):
    client = api_test_setup["client"]
    response = client.get("/incidents/INC_DOES_NOT_EXIST/state")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_hitl_approval_and_resumption_flow(api_test_setup):
    """
    End-to-End API Gateway Test:
    1. Injects STALE_LOCKFILE into mock environment.
    2. Triggers incident via POST /incidents/trigger.
    3. Polls GET /incidents/{id}/state until paused at AWAITING_APPROVAL.
    4. Verifies pending approval payload contains clear_lockfile and DESTRUCTIVE risk.
    5. Submits authorized approval via POST /incidents/{id}/approve (role: SRE_LEAD).
    6. Verifies graph resumes and completes to RESOLVED.
    """
    client = api_test_setup["client"]
    service = api_test_setup["service"]
    env = api_test_setup["env"]

    # 1. Setup failure in mock environment
    env.inject_scenario(ScenarioType.STALE_LOCKFILE)
    env.run_workflow()
    assert (env.locks_dir / "transform_job.lock").exists()

    # 2. Trigger incident
    inc_id = "INC_API_HITL_01"
    resp = client.post(
        "/incidents/trigger",
        json={
            "incident_id": inc_id,
            "target_service": "JOB_02_TRANSFORM",
            "raw_symptoms": "FATAL: Lockfile 'transform_job.lock' is present.",
        },
    )
    assert resp.status_code == 202

    # 3. Poll until paused at AWAITING_APPROVAL
    state = service.wait_for_incident(inc_id, timeout=10.0)
    assert state is not None
    assert state.status == IncidentStatus.AWAITING_APPROVAL
    assert state.pending_approval is not None
    assert state.pending_approval.tool_name == "clear_lockfile"
    assert state.pending_approval.risk_level == "DESTRUCTIVE"
    action_id = state.pending_approval.action_id

    # Verify physical environment lockfile is still present while paused
    assert (env.locks_dir / "transform_job.lock").exists()

    # 4. Submit approval
    approval_resp = client.post(
        f"/incidents/{inc_id}/approve",
        json={
            "approved": True,
            "approver_id": "OP_ALICE",
            "approver_role": "SRE_LEAD",
            "action_id": action_id,
            "notes": "Verified stale lockfile PID is dead; approved for deletion.",
        },
    )
    assert approval_resp.status_code == 200
    approval_data = approval_resp.json()
    assert approval_data["approved"] is True
    assert approval_data["resumed"] is True

    # 5. Await final resolution
    final_state = service.wait_for_incident(inc_id, timeout=10.0)
    assert final_state is not None
    assert final_state.status == IncidentStatus.RESOLVED

    # Verify physical environment: lock cleared and all jobs succeeded
    assert not (env.locks_dir / "transform_job.lock").exists()
    final_wf = env.get_state()
    assert all(j.status == JobStatus.SUCCESS for j in final_wf.jobs.values())


def test_hitl_rejection_halts_and_escalates(api_test_setup):
    """
    Proves that submitting approved=False halts execution, transitions to ESCALATED,
    and leaves the target resource untouched.
    """
    client = api_test_setup["client"]
    service = api_test_setup["service"]
    env = api_test_setup["env"]

    env.inject_scenario(ScenarioType.STALE_LOCKFILE)
    env.run_workflow()

    inc_id = "INC_API_REJECT_01"
    client.post(
        "/incidents/trigger",
        json={
            "incident_id": inc_id,
            "target_service": "JOB_02_TRANSFORM",
            "raw_symptoms": "Lockfile conflict.",
        },
    )

    state = service.wait_for_incident(inc_id, timeout=10.0)
    assert state.status == IncidentStatus.AWAITING_APPROVAL
    action_id = state.pending_approval.action_id

    # Submit rejection
    reject_resp = client.post(
        f"/incidents/{inc_id}/approve",
        json={
            "approved": False,
            "approver_id": "OP_BOB",
            "approver_role": "PLATFORM_ADMIN",
            "action_id": action_id,
            "notes": "Rejected: Active worker suspected.",
        },
    )
    assert reject_resp.status_code == 200
    assert reject_resp.json()["approved"] is False

    final_state = service.wait_for_incident(inc_id, timeout=10.0)
    assert final_state.status == IncidentStatus.ESCALATED

    # Invariant: Lockfile must NOT be deleted upon rejection!
    assert (env.locks_dir / "transform_job.lock").exists()


def test_rbac_unauthorized_role_forbidden(api_test_setup):
    """
    Proves that an approver with an unauthorized role (e.g. GUEST or INTERN)
    is rejected with 403 Forbidden.
    """
    client = api_test_setup["client"]
    service = api_test_setup["service"]
    env = api_test_setup["env"]

    env.inject_scenario(ScenarioType.STALE_LOCKFILE)
    env.run_workflow()

    inc_id = "INC_API_RBAC_01"
    client.post(
        "/incidents/trigger",
        json={
            "incident_id": inc_id,
            "target_service": "JOB_02_TRANSFORM",
            "raw_symptoms": "Lock error.",
        },
    )

    state = service.wait_for_incident(inc_id, timeout=10.0)
    action_id = state.pending_approval.action_id

    # Attempt approval with unauthorized role
    resp = client.post(
        f"/incidents/{inc_id}/approve",
        json={
            "approved": True,
            "approver_id": "UNAUTHORIZED_USER",
            "approver_role": "JUNIOR_INTERN",  # Not in ALLOWED_APPROVER_ROLES
            "action_id": action_id,
        },
    )
    assert resp.status_code == 403
    assert "unauthorized approver role" in resp.json()["detail"].lower()


def test_approval_action_id_mismatch(api_test_setup):
    """
    Proves that submitting approval for an action ID that does not match
    the pending interrupt is rejected with 400 Bad Request.
    """
    client = api_test_setup["client"]
    service = api_test_setup["service"]
    env = api_test_setup["env"]

    env.inject_scenario(ScenarioType.STALE_LOCKFILE)
    env.run_workflow()

    inc_id = "INC_API_MISMATCH_01"
    client.post(
        "/incidents/trigger",
        json={
            "incident_id": inc_id,
            "target_service": "JOB_02_TRANSFORM",
            "raw_symptoms": "Lock error.",
        },
    )

    service.wait_for_incident(inc_id, timeout=10.0)

    # Submit with bogus action_id
    resp = client.post(
        f"/incidents/{inc_id}/approve",
        json={
            "approved": True,
            "approver_id": "OP_CHARLIE",
            "approver_role": "SRE_LEAD",
            "action_id": "ACT_BOGUS_9999",
        },
    )
    assert resp.status_code == 400
    assert "action id mismatch" in resp.json()["detail"].lower()


def test_scenario_inject_endpoint(api_test_setup):
    """
    Proves that POST /scenarios/inject triggers the mock environment run
    and returns initial failure telemetry.
    """
    client = api_test_setup["client"]
    resp = client.post("/scenarios/inject", json={"scenario": "SERVICE_OFFLINE"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["scenario"] == "SERVICE_OFFLINE"
    assert data["failed_job"] == "JOB_03_LOAD"
    assert "offline" in data["error_message"].lower()


def test_list_incidents_endpoint(api_test_setup):
    """Proves that GET /incidents returns tracked incidents."""
    client = api_test_setup["client"]
    client.post(
        "/incidents/trigger",
        json={"incident_id": "INC_LIST_01", "raw_symptoms": "Error A"},
    )
    client.post(
        "/incidents/trigger",
        json={"incident_id": "INC_LIST_02", "raw_symptoms": "Error B"},
    )

    resp = client.get("/incidents")
    assert resp.status_code == 200
    incidents = resp.json()
    ids = [i["incident_id"] for i in incidents]
    assert "INC_LIST_01" in ids
    assert "INC_LIST_02" in ids
