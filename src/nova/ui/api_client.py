"""
HTTP API Client for the NOVA Streamlit Operator Dashboard.

Communicates exclusively through the FastAPI Backend Gateway:
- Health & checkpointer telemetry
- Incident triggering (custom & demo scenarios)
- Incident state polling (investigation, planning, execution, verification)
- Human-in-the-Loop approval/rejection submissions
- Incident catalog listing
"""

import os
from typing import Any, Dict, List, Optional
import httpx


class NovaApiClient:
    """
    Client for interacting with the NOVA FastAPI Gateway.
    Adheres strictly to the architectural boundary: Streamlit never invokes
    the LangGraph engine directly.
    """

    def __init__(self, base_url: Optional[str] = None, timeout: float = 10.0):
        # Default to localhost:8000 or NOVA_API_URL environment variable
        self.base_url = (
            base_url
            or os.environ.get("NOVA_API_URL")
            or "http://127.0.0.1:8000"
        ).rstrip("/")
        self.timeout = timeout

    def check_health(self) -> Dict[str, Any]:
        """Queries /health endpoint to verify gateway connectivity and status."""
        url = f"{self.base_url}/health"
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(url)
                if response.status_code == 200:
                    return response.json()
                return {
                    "status": "ERROR",
                    "detail": f"HTTP {response.status_code}: {response.text}",
                }
        except Exception as e:
            return {"status": "UNREACHABLE", "detail": str(e)}

    def list_incidents(self) -> List[Dict[str, Any]]:
        """Queries /incidents to retrieve all active and historical incidents."""
        url = f"{self.base_url}/incidents"
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(url)
                if response.status_code == 200:
                    return response.json()
                return []
        except Exception:
            return []

    def get_incident_state(self, incident_id: str) -> Optional[Dict[str, Any]]:
        """Polls full state snapshot for a given incident."""
        url = f"{self.base_url}/incidents/{incident_id}/state"
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(url)
                if response.status_code == 200:
                    return response.json()
                return None
        except Exception:
            return None

    def trigger_incident(
        self,
        raw_symptoms: str,
        target_service: Optional[str] = "UNKNOWN",
        incident_id: Optional[str] = None,
        max_retries: int = 3,
        max_investigation_steps: int = 5,
        environment: Optional[str] = None,
        execution_id: Optional[str] = None,
        kestra_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Submits an operational incident for non-blocking autonomous remediation."""
        url = f"{self.base_url}/incidents/trigger"
        payload: Dict[str, Any] = {
            "incident_id": incident_id,
            "target_service": target_service or "UNKNOWN",
            "raw_symptoms": raw_symptoms,
            "max_retries": max_retries,
            "max_investigation_steps": max_investigation_steps,
        }
        if environment:
            payload["environment"] = environment
        if execution_id:
            payload["execution_id"] = execution_id
        if kestra_url:
            payload["kestra_url"] = kestra_url
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(url, json=payload)
            if response.status_code in (200, 202):
                return response.json()
            raise RuntimeError(
                f"Gateway error {response.status_code}: {response.text}"
            )

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
        """Submits human operator decision (approve/reject) for an interrupted action."""
        url = f"{self.base_url}/incidents/{incident_id}/approve"
        payload: Dict[str, Any] = {
            "approved": approved,
            "approver_id": approver_id,
            "approver_role": approver_role,
            "action_id": action_id,
            "notes": notes,
        }
        if action_hash:
            payload["action_hash"] = action_hash
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(url, json=payload)
            if response.status_code == 200:
                return response.json()
            error_detail = response.json().get("detail", response.text) if response.headers.get("content-type") == "application/json" else response.text
            raise RuntimeError(
                f"Approval error ({response.status_code}): {error_detail}"
            )

    def inject_scenario(self, scenario: str) -> Dict[str, Any]:
        """Triggers synthetic failure injection into mock enterprise environment."""
        url = f"{self.base_url}/scenarios/inject"
        payload = {"scenario": scenario}
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(url, json=payload)
            if response.status_code == 200:
                return response.json()
            raise RuntimeError(
                f"Scenario injection error {response.status_code}: {response.text}"
            )
