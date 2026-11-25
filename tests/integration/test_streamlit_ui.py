"""
Headless Streamlit UI Integration Tests using `streamlit.testing.v1.AppTest`.

Distinction:
- `test_api_gateway.py` & `test_ui_gateway.py` test HTTP/API contracts via NovaApiClient.
- This suite executes the actual Streamlit frontend (`src/nova/ui/app.py`) widget tree,
  session state transitions, navigation, dashboard refresh, and HITL approval buttons,
  routed through FastAPI TestClient -> IncidentService -> LangGraph -> SqliteSaver.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from nova.api.main import app, set_incident_service
from nova.api.service import IncidentService
from nova.environments.mock_workflow_runner import MockWorkflowRunner
from nova.llm.client import MockLLMClient
from nova.storage.sqlite_saver import SqliteSaver
from nova.ui.api_client import NovaApiClient


APP_PATH = str(Path(__file__).resolve().parents[2] / "src" / "nova" / "ui" / "app.py")


@pytest.fixture
def streamlit_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Wires a real IncidentService + SqliteSaver behind TestClient for AppTest."""
    env = MockWorkflowRunner(base_dir=tmp_path / "mock_state")
    db_path = tmp_path / "ui_apptest.sqlite"
    saver = SqliteSaver(db_path=db_path)
    service = IncidentService(
        env=env,
        checkpointer=saver,
        llm=MockLLMClient(),
        max_workers=2,
    )
    set_incident_service(service)
    test_client = TestClient(app)

    class AppTestApiClient(NovaApiClient):
        def __init__(self, base_url: Optional[str] = None, timeout: float = 10.0):
            super().__init__(base_url=base_url or "http://testserver", timeout=timeout)

        def check_health(self) -> Dict[str, Any]:
            r = test_client.get("/health")
            return r.json() if r.status_code == 200 else {"status": "ERROR"}

        def list_incidents(self) -> List[Dict[str, Any]]:
            r = test_client.get("/incidents")
            return r.json() if r.status_code == 200 else []

        def get_incident_state(self, incident_id: str) -> Optional[Dict[str, Any]]:
            r = test_client.get(f"/incidents/{incident_id}/state")
            return r.json() if r.status_code == 200 else None

        def trigger_incident(
            self,
            raw_symptoms: str,
            target_service: Optional[str] = "UNKNOWN",
            incident_id: Optional[str] = None,
            max_retries: int = 3,
            max_investigation_steps: int = 5,
        ) -> Dict[str, Any]:
            r = test_client.post(
                "/incidents/trigger",
                json={
                    "incident_id": incident_id,
                    "target_service": target_service or "UNKNOWN",
                    "raw_symptoms": raw_symptoms,
                    "max_retries": max_retries,
                    "max_investigation_steps": max_investigation_steps,
                },
            )
            if r.status_code in (200, 202):
                return r.json()
            raise RuntimeError(f"Gateway error {r.status_code}: {r.text}")

        def submit_approval(
            self,
            incident_id: str,
            action_id: str,
            approved: bool,
            approver_id: str,
            approver_role: str,
            notes: Optional[str] = None,
            action_hash: Optional[str] = None,
        ) -> Dict[str, Any]:
            payload: Dict[str, Any] = {
                "approved": approved,
                "approver_id": approver_id,
                "approver_role": approver_role,
                "action_id": action_id,
                "notes": notes,
            }
            if action_hash:
                payload["action_hash"] = action_hash
            r = test_client.post(f"/incidents/{incident_id}/approve", json=payload)
            if r.status_code == 200:
                return r.json()
            raise RuntimeError(f"Approval error ({r.status_code}): {r.text}")

        def inject_scenario(self, scenario: str) -> Dict[str, Any]:
            r = test_client.post("/scenarios/inject", json={"scenario": scenario})
            if r.status_code == 200:
                return r.json()
            raise RuntimeError(f"Injection error {r.status_code}: {r.text}")

    monkeypatch.setattr("nova.ui.api_client.NovaApiClient", AppTestApiClient)
    try:
        yield {"service": service, "env": env}
    finally:
        test_client.close()
        service.shutdown()
        saver.close()


def test_streamlit_full_incident_hitl_and_refresh_flow(streamlit_backend):
    """
    Exercises the Streamlit app widget flow end-to-end:
    1. Overview renders with Connected (SqliteSaver).
    2. Navigate to Trigger & Fault Injection -> click Inject Stale Lockfile.
    3. Navigate to Investigation & Remediation Console -> refresh/rerun -> see HITL box.
    4. Click Authorize Action -> refresh -> verify RESOLVED and telemetry/audit rendering.
    5. Navigate to Incident Catalog & Audit Trails -> verify incident row & export button.
    """
    service: IncidentService = streamlit_backend["service"]

    at = AppTest.from_file(APP_PATH, default_timeout=15)
    at.run()
    assert not at.exception
    assert any("Connected (SqliteSaver)" in s.value for s in at.sidebar.success)

    # 1. Switch to Fault Injection view and inject STALE_LOCKFILE
    at.sidebar.radio(key="nav_view").set_value("🚨 Incident Trigger & Fault Injection").run()
    assert not at.exception

    inject_btn = next(b for b in at.button if b.label == "Inject Stale Lockfile")
    inject_btn.click().run()
    assert not at.exception

    inc_id = at.session_state["selected_incident_id"]
    assert inc_id is not None

    # Wait for background worker to reach HITL checkpoint
    paused = service.wait_for_incident(inc_id, timeout=10.0)
    assert paused is not None
    assert paused.status.value == "AWAITING_APPROVAL"

    # 2. Switch to Console & simulate dashboard refresh
    at.sidebar.radio(key="nav_view").set_value("🔍 Investigation & Remediation Console").run()
    assert not at.exception
    assert any(b.label == "✅ Authorize Action" for b in at.button)

    # Simulate user clicking Refresh Data Now
    refresh_btn = next(b for b in at.sidebar.button if b.label == "🔄 Refresh Data Now")
    refresh_btn.click().run()
    assert not at.exception
    assert any(b.label == "✅ Authorize Action" for b in at.button)

    # 3. Click Authorize Action in Streamlit UI
    approve_btn = next(b for b in at.button if b.label == "✅ Authorize Action")
    approve_btn.click().run()
    assert not at.exception

    final_st = service.wait_for_incident(inc_id, timeout=10.0)
    assert final_st is not None
    assert final_st.status.value == "RESOLVED"

    # Simulate a fresh browser page refresh on the Console after resolution
    at_refreshed = AppTest.from_file(APP_PATH, default_timeout=15)
    at_refreshed.session_state["selected_incident_id"] = inc_id
    at_refreshed.session_state["nav_view"] = "🔍 Investigation & Remediation Console"
    at_refreshed.run()
    assert not at_refreshed.exception
    assert any("Final Resolution Summary" in s.value for s in at_refreshed.success)
    metric_labels = {m.label for m in at_refreshed.metric}
    assert {"LLM Requests", "Diagnostic Tools", "Remediation Tools"}.issubset(metric_labels)

    # 4. Navigate to Incident Catalog & Audit Trails
    at_refreshed.sidebar.radio(key="nav_view").set_value("📜 Incident Catalog & Audit Trails").run()
    assert not at_refreshed.exception
    assert len(at_refreshed.dataframe) == 1


def test_streamlit_custom_unknown_trigger_and_operator_rejection(streamlit_backend):
    """
    Exercises custom incident submission with target_service='UNKNOWN' and
    Human-in-the-Loop rejection via Streamlit UI buttons.
    """
    service: IncidentService = streamlit_backend["service"]
    env: MockWorkflowRunner = streamlit_backend["env"]

    from nova.environments.mock_workflow_runner import ScenarioType

    env.inject_scenario(ScenarioType.SERVICE_OFFLINE)
    env.run_workflow()

    at = AppTest.from_file(APP_PATH, default_timeout=15)
    at.run()
    assert not at.exception

    # Navigate to Trigger & Fault Injection -> submit custom incident form
    at.sidebar.radio(key="nav_view").set_value("🚨 Incident Trigger & Fault Injection").run()
    submit_btn = next(b for b in at.button if b.label == "Submit Incident to NOVA")
    submit_btn.click().run()
    assert not at.exception

    inc_id = at.session_state["selected_incident_id"]
    assert inc_id is not None

    paused = service.wait_for_incident(inc_id, timeout=10.0)
    assert paused is not None
    assert paused.status.value == "AWAITING_APPROVAL"
    assert paused.pending_approval is not None
    assert paused.pending_approval.tool_name == "restart_service"

    # Navigate to Console -> reject the action
    at.sidebar.radio(key="nav_view").set_value("🔍 Investigation & Remediation Console").run()
    assert not at.exception
    reject_btn = next(b for b in at.button if b.label == "❌ Reject & Halt Action")
    reject_btn.click().run()
    assert not at.exception

    final_st = service.wait_for_incident(inc_id, timeout=10.0)
    assert final_st is not None
    assert final_st.status.value == "ESCALATED"
    assert env.check_service_status("reporting_warehouse_service")["healthy"] is False
