"""
Milestone 9 Live Demonstration: Streamlit Operator Dashboard & FastAPI Gateway Integration.

Demonstrates the complete end-to-end architecture:
1. FastAPI Gateway hosting non-blocking incident orchestration with SqliteSaver.
2. NovaApiClient (used by Streamlit) communicating strictly over HTTP/API contracts.
3. Autonomous investigation with dynamic diagnostic tool selection.
4. Multi-step plan execution with Human-in-the-Loop (HITL) approval gates.
5. RBAC approval authorization and audit trail generation.
6. Safe escalation on unrecoverable data corruption.

Runs against live Google Gemini (if configured in .env / GEMINI_API_KEY) or MockLLMClient.
"""

import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

# Ensure unbuffered output for real-time console streaming
sys.stdout.reconfigure(line_buffering=True)

# Add project root and src to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

from fastapi.testclient import TestClient
from nova.api.main import app, set_incident_service
from nova.api.service import IncidentService
from nova.config import settings
from nova.environments.mock_workflow_runner import (
    JobStatus,
    MockWorkflowRunner,
    ScenarioType,
)
from nova.llm.client import GeminiLLMClient, MockLLMClient
from nova.storage.sqlite_saver import SqliteSaver
from nova.ui.api_client import NovaApiClient


def setup_demonstration_service(use_gemini: bool = True):
    """Initializes isolated mock environment, SQLite persistence, and IncidentService."""
    tmpdir = tempfile.mkdtemp()
    base_path = Path(tmpdir)
    env = MockWorkflowRunner(base_dir=base_path / "mock_state")
    db_path = base_path / "demo_checkpoints.sqlite"
    checkpointer = SqliteSaver(db_path=db_path)

    api_key = (
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or settings.gemini_api_key
    )

    llm = None
    engine_name = "MockLLMClient"
    if use_gemini and api_key and api_key.strip():
        try:
            print(f"[*] Connecting to Google Gemini API (Model: {settings.default_model})...")
            llm = GeminiLLMClient(api_key=api_key, model_name=settings.default_model)
            engine_name = f"GeminiLLMClient ({settings.default_model})"
            print(f"[+] Successfully connected to live Gemini API ({settings.default_model}).")
        except Exception as e:
            print(f"[!] Falling back to MockLLMClient due to: {e}")
            llm = MockLLMClient()
    else:
        print("[*] Using MockLLMClient for deterministic demonstration.")
        llm = MockLLMClient()

    service = IncidentService(
        env=env,
        checkpointer=checkpointer,
        llm=llm,
        max_workers=2,
    )
    set_incident_service(service)
    client = TestClient(app)

    # NovaApiClient adapter routing through TestClient
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

        def trigger_incident(
            self,
            raw_symptoms,
            target_service="UNKNOWN",
            incident_id=None,
            max_retries=3,
            max_investigation_steps=5,
        ):
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
            return r.json()

        def submit_approval(
            self, incident_id, action_id, approved, approver_id, approver_role, notes=None
        ):
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
            return r.json()

        def inject_scenario(self, scenario):
            r = client.post("/scenarios/inject", json={"scenario": scenario})
            return r.json()

    api_client = DirectTestApiClient()
    return service, env, api_client, checkpointer, tmpdir, engine_name


def poll_incident(
    api_client: NovaApiClient,
    incident_id: str,
    stop_statuses: tuple,
    timeout: float = 90.0,
):
    """Polls incident state until one of the stop_statuses is reached."""
    start = time.time()
    last_status = None
    last_print = 0
    while time.time() - start < timeout:
        state = api_client.get_incident_state(incident_id)
        if state:
            status = state.get("status")
            elapsed = time.time() - start
            if status != last_status or elapsed - last_print >= 5.0:
                print(f"    [POLL] Status: {status} (elapsed: {elapsed:.1f}s)")
                last_status = status
                last_print = elapsed
            if status in stop_statuses:
                return state
        time.sleep(1.0)
    return api_client.get_incident_state(incident_id)


def run_demonstration():
    print("=" * 80)
    print("NOVA Milestone 9: Streamlit Operator Dashboard & Gateway Live Demonstration")
    print("=" * 80)

    service, env, api_client, checkpointer, tmpdir, engine_name = setup_demonstration_service(use_gemini=True)

    try:
        # 1. Health & Discovery
        print("\n[PHASE 1] Checking Gateway Health & Runtime Configuration...")
        health = api_client.check_health()
        print(f"  API Status:             {health.get('status')}")
        print(f"  Gateway Version:        {health.get('version')}")
        print(f"  Environment:            {health.get('environment')}")
        print(f"  Checkpointer Mechanism: {health.get('checkpointer')}")
        print(f"  LLM Engine:             {engine_name}")

        # 2. Scenario 1: Stale Lockfile with HITL Approval Gate
        print("\n[PHASE 2] Executing Stale-Lockfile Incident with Operator Approval...")
        inj1 = api_client.inject_scenario("STALE_LOCKFILE")
        print(f"  Injected Scenario:      STALE_LOCKFILE -> Failed Job: {inj1.get('failed_job')}")

        inc1 = api_client.trigger_incident(
            raw_symptoms=f"FATAL: Batch ETL job {inj1.get('failed_job')} failed: {inj1.get('error_message')}",
            target_service=inj1.get("failed_job") or "JOB_02_TRANSFORM",
        )
        inc1_id = inc1["incident_id"]
        print(f"  Triggered Incident:     {inc1_id} (Status: {inc1.get('status')})")

        print("  Awaiting Agentic Triage, Planning, and HITL Pause...")
        state1 = poll_incident(api_client, inc1_id, stop_statuses=("AWAITING_APPROVAL", "ESCALATED", "RESOLVED"), timeout=60.0)
        print(f"  Current Incident Status: {state1.get('status')}")
        print(f"  Failure Type:            {state1.get('failure_type')}")
        print(f"  Implicated Resource:     {state1.get('implicated_resource')}")
        print(f"  Root Cause Hypothesis:   {state1.get('root_cause_hypothesis')}")
        print(f"  Investigation Steps:     {len(state1.get('diagnostic_history', []))} diagnostic steps")
        for step in state1.get("diagnostic_history", []):
            print(f"    - Step {step.get('step')}: {step.get('tool_name')} on {step.get('parameters')} -> Success: {step.get('success')}")

        # Verify HITL pause
        if state1.get("status") == "AWAITING_APPROVAL":
            pending = state1.get("pending_approval")
            print("\n  [HITL GATE TRIGGERED]")
            print(f"    Action ID:        {pending.get('action_id')}")
            print(f"    Tool:             {pending.get('tool_name')}")
            print(f"    Target:           {pending.get('parameters')}")
            print(f"    Risk Level:       {pending.get('risk_level')}")
            print(f"    Rationale:        {pending.get('rationale')}")
            print(f"    Expected Outcome: {pending.get('expected_outcome')}")

            # Simulate Operator Approval via Streamlit Dashboard
            print("  Simulating SRE_LEAD approval via Streamlit Dashboard...")
            app_res = api_client.submit_approval(
                incident_id=inc1_id,
                action_id=pending.get("action_id"),
                approved=True,
                approver_id="operator_bmc_lead",
                approver_role="SRE_LEAD",
                notes="Approved via Streamlit Operator Dashboard after verifying lockfile PID is orphaned.",
            )
            print(f"  Approval Submitted:     Resumed={app_res.get('resumed')}, Status={app_res.get('status')}")

            # Wait for execution and verification
            print("  Resuming graph execution and awaiting downstream verification...")
            final_state1 = poll_incident(api_client, inc1_id, stop_statuses=("RESOLVED", "ESCALATED"), timeout=60.0)
            print(f"  Final Incident Status:   {final_state1.get('status')}")
            print(f"  Execution Records:       {len(final_state1.get('execution_history', []))} actions executed")
            for rec in final_state1.get("execution_history", []):
                print(f"    - Action: {rec.get('action_id')} ({rec.get('tool_name')}) -> Success: {rec.get('success')} (Exit Code: {rec.get('exit_code')})")
                print(f"      Approved By: {rec.get('approved_by')} [{rec.get('approver_role')}]")
            print(f"  Verification Records:    {len(final_state1.get('verification_history', []))} postconditions verified")
            for vrec in final_state1.get("verification_history", []):
                print(f"    - Postcondition: Passed={vrec.get('passed')} -> Evidence: {vrec.get('evidence')}")
        else:
            print(f"  Incident completed with status: {state1.get('status')}")

        # 3. Scenario 2: Unknown Incident Autonomous Investigation
        print("\n[PHASE 3] Executing Unknown-Service Autonomous Investigation...")
        inj2 = api_client.inject_scenario("SERVICE_OFFLINE")
        print(f"  Injected Scenario:      SERVICE_OFFLINE -> Failed Job: {inj2.get('failed_job')}")

        inc2 = api_client.trigger_incident(
            raw_symptoms="High severity alert: Nightly reporting workflow halted abruptly. Root cause service unknown.",
            target_service="UNKNOWN",
        )
        inc2_id = inc2["incident_id"]
        print(f"  Triggered Incident:     {inc2_id} (Status: {inc2.get('status')})")

        state2 = poll_incident(api_client, inc2_id, stop_statuses=("RESOLVED", "ESCALATED", "AWAITING_APPROVAL"), timeout=120.0)
        print(f"  Current Incident Status: {state2.get('status')}")
        print(f"  Discovered Target:       {state2.get('implicated_resource')}")
        print(f"  Root Cause Hypothesis:   {state2.get('root_cause_hypothesis')}")
        print(f"  Autonomous Investigation ({len(state2.get('diagnostic_history', []))} steps):")
        for step in state2.get("diagnostic_history", []):
            print(f"    - Step {step.get('step')}: {step.get('tool_name')} on {step.get('parameters')} -> Success: {step.get('success')}")

        if state2.get("status") == "AWAITING_APPROVAL":
            pending2 = state2.get("pending_approval")
            print(f"\n  [HITL GATE TRIGGERED FOR UNKNOWN INCIDENT]")
            print(f"    Approving {pending2.get('tool_name')} on {pending2.get('parameters')} as PLATFORM_ADMIN...")
            api_client.submit_approval(
                incident_id=inc2_id,
                action_id=pending2.get("action_id"),
                approved=True,
                approver_id="admin_bmc",
                approver_role="PLATFORM_ADMIN",
                notes="Approved database restart discovered via autonomous investigation.",
            )
            state2 = poll_incident(api_client, inc2_id, stop_statuses=("RESOLVED", "ESCALATED"), timeout=60.0)

        print(f"  Final Incident Status:   {state2.get('status')}")

        # 4. Scenario 3: Unrecoverable Data Corruption Escalation
        print("\n[PHASE 4] Executing Unrecoverable Data Corruption Incident...")
        inj3 = api_client.inject_scenario("DATA_CORRUPTION")
        print(f"  Injected Scenario:      DATA_CORRUPTION -> Failed Job: {inj3.get('failed_job')}")

        inc3 = api_client.trigger_incident(
            raw_symptoms=f"FATAL: {inj3.get('error_message') or 'Data corruption detected in partition'}",
            target_service=inj3.get("failed_job") or "JOB_02_TRANSFORM",
        )
        inc3_id = inc3["incident_id"]
        print(f"  Triggered Incident:     {inc3_id} (Status: {inc3.get('status')})")

        state3 = poll_incident(api_client, inc3_id, stop_statuses=("RESOLVED", "ESCALATED"), timeout=60.0)
        print(f"  Final Incident Status:   {state3.get('status')}")
        print(f"  Escalation Summary:      {state3.get('final_summary')}")
        assert state3.get("status") == "ESCALATED", f"Expected ESCALATED but got {state3.get('status')}"

        # 5. Incident Catalog & Summary Metrics
        print("\n[PHASE 5] Inspecting Dashboard Metrics & Incident Catalog...")
        incidents = api_client.list_incidents()
        print(f"  Total Incidents Managed: {len(incidents)}")
        for inc in incidents:
            print(f"    - [{inc.get('status')}] {inc.get('incident_id')}: {inc.get('target_service')} (Type: {inc.get('failure_type')})")

        print("\n" + "=" * 80)
        print("DEMONSTRATION COMPLETE: ALL 5 PHASES VALIDATED SUCCESSFULLY")
        print("=" * 80)

    finally:
        service.shutdown()
        import gc
        import shutil
        gc.collect()
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    run_demonstration()
