"""
LLM Client Abstraction Layer for NOVA.

Design principle (from AGENTS.md):
  "LLM reasoning only where it adds genuine value."

This module provides:
  - BaseLLMClient: abstract interface every backend must satisfy.
  - MockLLMClient: fully deterministic fake for unit/integration tests.
                   Zero API calls, zero latency, zero cost. 
                   Test assertions are written against mock output — not against LLM responses.
  - GeminiLLMClient: real Google Gemini backend for production / manual runs.

Inject the right client via dependency injection in build_nova_graph().
Tests always inject MockLLMClient — they never reach the network.
"""
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel

from nova.llm.schemas import (
    DiagnosticToolCall,
    InvestigationActionType,
    InvestigationDecision,
    PlannedActionSchema,
    PlannerResult,
    ReplannerResult,
    TriageResult,
)
from nova.observability import log_event, redact_secrets, telemetry
from nova.state import RiskLevel

logger = logging.getLogger("nova.llm.client")

T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------

class BaseLLMClient(ABC):
    """
    Abstract contract every LLM backend must satisfy.

    Each method receives ONLY structured Python data (strings, dicts, lists).
    Prompt construction lives here — nodes stay clean.
    Each method returns a VALIDATED Pydantic model.
    """

    @abstractmethod
    def investigate(
        self,
        target_service: Optional[str],
        raw_symptoms: str,
        investigation_evidence: List[str],
        diagnostic_history: List[Dict[str, Any]],
        available_tools: List[Dict[str, Any]],
        step_number: int,
    ) -> InvestigationDecision:
        """
        Inspect gathered diagnostic evidence and decide next action:
        - Request another read-only diagnostic tool
        - Return finalized diagnosis ready for planning
        - Escalate safely to human
        """

    @abstractmethod
    def triage(self, target_service: str, log_content: str) -> TriageResult:
        """
        Read raw log content for a failed service.
        Return a structured triage analysis.
        """

    @abstractmethod
    def plan(
        self,
        triage: TriageResult,
        available_tools: List[Dict[str, Any]],
        retry_count: int,
    ) -> PlannerResult:
        """
        Given a triage result and the tool catalogue, produce a remediation plan.
        Return structured actions or an escalation decision.
        """

    @abstractmethod
    def replan(
        self,
        original_target: str,
        first_action_taken: str,
        new_target: str,
        new_log_content: str,
        verification_evidence: str,
        retry_count: int,
        execution_records: Optional[List[Dict[str, Any]]] = None,
        verification_records: Optional[List[Dict[str, Any]]] = None,
        investigation_evidence: Optional[List[str]] = None,
        original_diagnosis: Optional[Dict[str, Any]] = None,
    ) -> ReplannerResult:
        """
        Evidence-based replanning after execution / verification feedback.
        Does NOT assume previous remediation succeeded. Evaluates ground-truth records.
        """


# ---------------------------------------------------------------------------
# Mock client  (used in ALL tests)
# ---------------------------------------------------------------------------

class MockLLMClient(BaseLLMClient):
    """
    Deterministic LLM stub for testing.

    Mimics realistic LLM behaviour by pattern-matching on the same signals
    a real LLM would use, but via simple Python string checks.
    This keeps tests fast, free, and 100% reproducible.

    Key guarantee: MockLLMClient returns the SAME Pydantic types as the real client.
    If a test passes with MockLLMClient, the graph wiring is proven correct.
    The only thing that changes in production is the client implementation.
    """

    def investigate(
        self,
        target_service: Optional[str],
        raw_symptoms: str,
        investigation_evidence: List[str],
        diagnostic_history: List[Dict[str, Any]],
        available_tools: List[Dict[str, Any]],
        step_number: int,
    ) -> InvestigationDecision:
        """
        Autonomous investigation agent simulation:
        1. If target_service is unknown or empty, requests 'get_workflow_status' to identify failed jobs.
        2. Detects healthy / false-positive alerts without unnecessary intervention.
        3. Once failed job is known, requests 'read_job_logs' to inspect error output.
        4. If logs indicate connection failure, requests 'check_service_status' to verify health.
        5. If logs indicate lockfile, requests 'inspect_lockfiles' to confirm disk presence & ownership.
        6. Escalates if an active worker owns the lockfile (concurrency safety).
        7. When evidence is complete, returns DIAGNOSIS_READY with structured TriageResult.
        """
        t0 = time.perf_counter()
        decision = self._investigate_impl(
            target_service=target_service,
            raw_symptoms=raw_symptoms,
            investigation_evidence=investigation_evidence,
            diagnostic_history=diagnostic_history,
            available_tools=available_tools,
            step_number=step_number,
        )
        telemetry.record_llm_call(
            incident_id=None,
            operation="investigate",
            model_name="MockLLMClient",
            duration_seconds=time.perf_counter() - t0,
        )
        return decision

    def _investigate_impl(
        self,
        target_service: Optional[str],
        raw_symptoms: str,
        investigation_evidence: List[str],
        diagnostic_history: List[Dict[str, Any]],
        available_tools: List[Dict[str, Any]],
        step_number: int,
    ) -> InvestigationDecision:
        # Collect tool names already executed in this investigation
        tools_called = []
        for d in diagnostic_history:
            if isinstance(d, dict):
                tools_called.append(d.get("tool_name", ""))
            else:
                tools_called.append(getattr(d, "tool_name", ""))

        all_evidence = " ".join(investigation_evidence) + " " + raw_symptoms
        evidence_lower = all_evidence.lower()

        # Check if workflow status was already inspected and all jobs are healthy (false positive)
        workflow_checked = "get_workflow_status" in tools_called
        workflow_has_jobs = "Job States:" in all_evidence
        workflow_has_failed = "status=FAILED" in all_evidence
        if workflow_checked and workflow_has_jobs and not workflow_has_failed:
            return InvestigationDecision(
                decision_type=InvestigationActionType.ESCALATE,
                escalation_reason=(
                    "False-positive alert verified: all workflow jobs are in SUCCESS state and no failures exist. "
                    "No remediation action required."
                ),
            )

        # Phase 1: Determine target job if unknown (or if target_service is a service name rather than JOB_)
        active_target = target_service
        if not active_target or active_target == "UNKNOWN":
            if "get_workflow_status" not in tools_called:
                return InvestigationDecision(
                    decision_type=InvestigationActionType.DIAGNOSTIC_ACTION,
                    diagnostic_tool=DiagnosticToolCall(
                        tool_name="get_workflow_status",
                        parameters={},
                        rationale="Target service unknown; querying workflow status to identify failed jobs.",
                    ),
                )
            # Find failed job from evidence
            job_match = re.search(r"(JOB_0[1-4]_[A-Z]+): status=FAILED", all_evidence)
            if job_match:
                active_target = job_match.group(1)
            else:
                active_target = "JOB_02_TRANSFORM"

        # If target_service looks like a service name (e.g. reporting_warehouse_service), verify health first
        if (
            active_target
            and not active_target.startswith("JOB_")
            and "check_service_status" not in tools_called
        ):
            return InvestigationDecision(
                decision_type=InvestigationActionType.DIAGNOSTIC_ACTION,
                diagnostic_tool=DiagnosticToolCall(
                    tool_name="check_service_status",
                    parameters={"service_name": active_target},
                    rationale=f"Checking live health status of '{active_target}' before any intervention.",
                ),
            )

        if (
            active_target
            and not active_target.startswith("JOB_")
            and "check_service_status" in tools_called
            and f"Service '{active_target}' status: RUNNING (Healthy: True)" in all_evidence
        ):
            if "get_workflow_status" not in tools_called:
                return InvestigationDecision(
                    decision_type=InvestigationActionType.DIAGNOSTIC_ACTION,
                    diagnostic_tool=DiagnosticToolCall(
                        tool_name="get_workflow_status",
                        parameters={},
                        rationale="Service reported healthy; verifying batch workflow state for false-positive check.",
                    ),
                )
            return InvestigationDecision(
                decision_type=InvestigationActionType.ESCALATE,
                escalation_reason=(
                    f"False-positive alert: service '{active_target}' is RUNNING and healthy. "
                    "No remediation required."
                ),
            )

        # Phase 2: Read job logs if not yet read
        if "read_job_logs" not in tools_called:
            return InvestigationDecision(
                decision_type=InvestigationActionType.DIAGNOSTIC_ACTION,
                diagnostic_tool=DiagnosticToolCall(
                    tool_name="read_job_logs",
                    parameters={"job_id": active_target, "tail_lines": 50},
                    rationale=f"Reading recent log output for {active_target} to identify failure signature.",
                ),
            )

        # Active worker lockfile protection (must check before classifying as stale)
        if (
            "status=active_worker" in evidence_lower
            or "heartbeat=recent" in evidence_lower
            or "active worker holding lock" in evidence_lower
            or "another active instance is running" in evidence_lower
        ):
            return InvestigationDecision(
                decision_type=InvestigationActionType.ESCALATE,
                escalation_reason=(
                    "Active worker lockfile detected (live process heartbeat present). "
                    "Refusing automated deletion to prevent split-brain data corruption."
                ),
            )

        # Phase 3: Gather corroborating diagnostic evidence
        if "connection" in evidence_lower and "refused" in evidence_lower:
            svc_match = re.search(r"'([\w_]+service[\w_]*)'", all_evidence)
            svc_name = svc_match.group(1) if svc_match else "reporting_warehouse_service"
            if "check_service_status" not in tools_called:
                return InvestigationDecision(
                    decision_type=InvestigationActionType.DIAGNOSTIC_ACTION,
                    diagnostic_tool=DiagnosticToolCall(
                        tool_name="check_service_status",
                        parameters={"service_name": svc_name},
                        rationale=f"Verifying health and listener status of {svc_name}.",
                    ),
                )
            triage_res = self.triage(active_target, all_evidence)
            return InvestigationDecision(
                decision_type=InvestigationActionType.DIAGNOSIS_READY,
                diagnosis=triage_res,
            )

        if ".lock" in evidence_lower:
            lock_match = re.search(r"'?([\w./]+\.lock)'?", all_evidence)
            lock_name = lock_match.group(1) if lock_match else "transform_job.lock"
            if "inspect_lockfiles" not in tools_called:
                return InvestigationDecision(
                    decision_type=InvestigationActionType.DIAGNOSTIC_ACTION,
                    diagnostic_tool=DiagnosticToolCall(
                        tool_name="inspect_lockfiles",
                        parameters={"lock_name": lock_name},
                        rationale=f"Checking filesystem presence of lockfile '{lock_name}'.",
                    ),
                )
            # Re-check lockfile inspection output for active worker ownership
            if "status=active_worker" in evidence_lower or "heartbeat=recent" in evidence_lower:
                return InvestigationDecision(
                    decision_type=InvestigationActionType.ESCALATE,
                    escalation_reason=(
                        f"Lockfile '{lock_name}' is held by an active worker process. "
                        "Escalating without deletion to preserve concurrency safety."
                    ),
                )
            triage_res = self.triage(active_target, all_evidence)
            return InvestigationDecision(
                decision_type=InvestigationActionType.DIAGNOSIS_READY,
                diagnosis=triage_res,
            )

        if "checksum mismatch" in evidence_lower or "zero-byte" in evidence_lower or "corrupt" in evidence_lower:
            triage_res = self.triage(active_target, all_evidence)
            return InvestigationDecision(
                decision_type=InvestigationActionType.DIAGNOSIS_READY,
                diagnosis=triage_res,
            )

        if "hardware" in evidence_lower or "kernel panic" in evidence_lower or "disk controller failure" in evidence_lower:
            return InvestigationDecision(
                decision_type=InvestigationActionType.ESCALATE,
                escalation_reason="Severe hardware or kernel failure cannot be diagnosed or resolved via software runbooks.",
            )

        # Default: Produce triage result from gathered evidence
        triage_res = self.triage(active_target, all_evidence)
        return InvestigationDecision(
            decision_type=InvestigationActionType.DIAGNOSIS_READY,
            diagnosis=triage_res,
        )

    def triage(self, target_service: str, log_content: str) -> TriageResult:
        """
        Parse log signals deterministically and return a structured TriageResult.
        This is the same analysis a real LLM would do — just via string matching
        since we know the exact mock log format.
        """
        t0 = time.perf_counter()
        result = self._triage_impl(target_service, log_content)
        telemetry.record_llm_call(
            incident_id=None,
            operation="triage",
            model_name="MockLLMClient",
            duration_seconds=time.perf_counter() - t0,
        )
        return result

    def _triage_impl(self, target_service: str, log_content: str) -> TriageResult:
        log_lower = log_content.lower()

        # SERVICE_OFFLINE — check first (most specific: port + refused together)
        if "connection" in log_lower and "refused" in log_lower:
            svc_match = re.search(r"'([\w_]+service[\w_]*)'", log_content)
            svc_name = svc_match.group(1) if svc_match else "reporting_warehouse_service"
            return TriageResult(
                failure_type="SERVICE_OFFLINE",
                implicated_resource=svc_name,
                triage_summary=(
                    f"FATAL: {target_service} cannot connect to '{svc_name}'. "
                    "Connection refused on port 5432 — service is OFFLINE."
                ),
                root_cause_hypothesis=(
                    f"'{svc_name}' crashed or was stopped before the batch load job ran. "
                    "No listener is active on the expected port."
                ),
                is_recoverable=True,
                confidence="HIGH",
            )

        # STALE_LOCKFILE — any .lock file found in log (dynamic extraction, not hardcoded)
        if ".lock" in log_lower:
            lock_match = re.search(r"'?([\w./]+\.lock)'?", log_content)
            lock_name = lock_match.group(1) if lock_match else "unknown.lock"
            return TriageResult(
                failure_type="STALE_LOCKFILE",
                implicated_resource=lock_name,
                triage_summary=(
                    f"FATAL: Orphaned lockfile '{lock_name}' is blocking {target_service}. "
                    "Job cannot acquire execution lock and aborted with exit code 1."
                ),
                root_cause_hypothesis=(
                    f"A previous batch run crashed without releasing '{lock_name}'. "
                    "The lock is orphaned — the PID that created it no longer exists."
                ),
                is_recoverable=True,
                confidence="HIGH",
            )

        elif "checksum mismatch" in log_lower or "zero-byte" in log_lower or "corrupt" in log_lower:
            return TriageResult(
                failure_type="DATA_CORRUPTION",
                implicated_resource=target_service,
                triage_summary=(
                    f"CRITICAL: {target_service} detected payload checksum mismatch. "
                    "Zero-byte records found in the source data feed — data is unreadable."
                ),
                root_cause_hypothesis=(
                    "The upstream data feed delivered a corrupt or truncated payload. "
                    "Automated retry cannot reconstitute missing or corrupt source records."
                ),
                is_recoverable=False,
                confidence="HIGH",
            )

        else:
            return TriageResult(
                failure_type="UNKNOWN",
                implicated_resource=target_service,
                triage_summary=f"Unrecognized failure signature in {target_service} logs.",
                root_cause_hypothesis="Insufficient log evidence to determine root cause.",
                is_recoverable=False,
                confidence="LOW",
            )

    def plan(
        self,
        triage: TriageResult,
        available_tools: List[Dict[str, Any]],
        retry_count: int,
    ) -> PlannerResult:
        """
        Convert a TriageResult into a structured remediation plan.
        Uses the implicated_resource from triage to populate tool parameters
        dynamically — no hardcoded lock names or service names here.
        """
        t0 = time.perf_counter()
        result = self._plan_impl(triage, available_tools, retry_count)
        telemetry.record_llm_call(
            incident_id=None,
            operation="plan",
            model_name="MockLLMClient",
            duration_seconds=time.perf_counter() - t0,
        )
        return result

    def _plan_impl(
        self,
        triage: TriageResult,
        available_tools: List[Dict[str, Any]],
        retry_count: int,
    ) -> PlannerResult:
        seq = retry_count + 1

        if not triage.is_recoverable:
            return PlannerResult(
                is_escalation=True,
                escalation_reason=(
                    f"Failure type '{triage.failure_type}' is not recoverable by automated tooling. "
                    f"Root cause: {triage.root_cause_hypothesis}"
                ),
                actions=[],
            )

        active_workflow_id = triage.workflow_id or "WF_NIGHTLY_BATCH_1001"

        if triage.failure_type == "STALE_LOCKFILE":
            return PlannerResult(
                is_escalation=False,
                actions=[
                    PlannedActionSchema(
                        tool_name="clear_lockfile",
                        # Key: resource name comes FROM triage, not hardcoded
                        parameters={"lock_name": triage.implicated_resource},
                        risk_level=RiskLevel.DESTRUCTIVE,
                        rationale=(
                            f"Remove orphaned '{triage.implicated_resource}' so the transform job "
                            "can re-acquire the execution lock and proceed."
                        ),
                        expected_outcome=f"Lockfile '{triage.implicated_resource}' no longer exists on the filesystem.",
                    ),
                    # M11: resume_workflow is an explicit planned action — the verifier is now
                    # read-only and does NOT call run_workflow() internally.
                    PlannedActionSchema(
                        tool_name="resume_workflow",
                        parameters={"workflow_id": active_workflow_id},
                        risk_level=RiskLevel.LOW_RISK,
                        rationale=(
                            "Re-execute the batch pipeline now that the stale lock has been cleared. "
                            "This is required so the verifier can observe a completed workflow state."
                        ),
                        expected_outcome="All batch jobs complete successfully with exit_code=0.",
                    ),
                ],
            )

        elif triage.failure_type == "SERVICE_OFFLINE":
            return PlannerResult(
                is_escalation=False,
                actions=[
                    PlannedActionSchema(
                        tool_name="restart_service",
                        # Key: service name comes FROM triage, not hardcoded
                        parameters={"service_name": triage.implicated_resource},
                        risk_level=RiskLevel.DESTRUCTIVE,
                        rationale=(
                            f"Restart '{triage.implicated_resource}' to restore its TCP listener "
                            "and unblock the downstream load job."
                        ),
                        expected_outcome=f"'{triage.implicated_resource}' status is RUNNING and port is accepting connections.",
                    ),
                    # M11: resume_workflow is an explicit planned action — verifier is read-only.
                    PlannedActionSchema(
                        tool_name="resume_workflow",
                        parameters={"workflow_id": active_workflow_id},
                        risk_level=RiskLevel.LOW_RISK,
                        rationale=(
                            "Re-execute the batch pipeline now that the offline service has been restarted. "
                            "This is required so the verifier can observe a completed workflow state."
                        ),
                        expected_outcome="All batch jobs complete successfully with exit_code=0.",
                    ),
                ],
            )

        elif triage.failure_type == "DEPENDENCY_FAILURE":
            # Downstream dependency — service likely offline
            return PlannerResult(
                is_escalation=False,
                actions=[
                    PlannedActionSchema(
                        tool_name="restart_service",
                        parameters={"service_name": triage.implicated_resource},
                        risk_level=RiskLevel.DESTRUCTIVE,
                        rationale=(
                            f"Restart '{triage.implicated_resource}' which failed as a downstream "
                            "dependency after upstream remediation completed."
                        ),
                        expected_outcome=f"'{triage.implicated_resource}' is healthy and batch pipeline can complete.",
                    ),
                    # M11: resume_workflow is an explicit planned action — verifier is read-only.
                    PlannedActionSchema(
                        tool_name="resume_workflow",
                        parameters={"workflow_id": active_workflow_id},
                        risk_level=RiskLevel.LOW_RISK,
                        rationale=(
                            "Re-execute the batch pipeline now that the dependency service has been restarted."
                        ),
                        expected_outcome="All batch jobs complete successfully with exit_code=0.",
                    ),
                ],
            )

        # Unknown or unhandled — escalate safely
        return PlannerResult(
            is_escalation=True,
            escalation_reason=(
                f"No automated remediation strategy available for failure type '{triage.failure_type}'. "
                f"Manual investigation required. Hypothesis: {triage.root_cause_hypothesis}"
            ),
            actions=[],
        )

    def replan(
        self,
        original_target: str,
        first_action_taken: str,
        new_target: str,
        new_log_content: str,
        verification_evidence: str,
        retry_count: int,
        execution_records: Optional[List[Dict[str, Any]]] = None,
        verification_records: Optional[List[Dict[str, Any]]] = None,
        investigation_evidence: Optional[List[str]] = None,
        original_diagnosis: Optional[Dict[str, Any]] = None,
    ) -> ReplannerResult:
        """
        Evidence-based replanning. Does NOT assume previous remediation succeeded.
        Inspects execution records: if previous action failed execution or postcondition,
        revises hypothesis rather than assuming downstream component failure.
        """
        t0 = time.perf_counter()
        result = self._replan_impl(
            original_target=original_target,
            first_action_taken=first_action_taken,
            new_target=new_target,
            new_log_content=new_log_content,
            verification_evidence=verification_evidence,
            retry_count=retry_count,
            execution_records=execution_records,
            verification_records=verification_records,
            investigation_evidence=investigation_evidence,
            original_diagnosis=original_diagnosis,
        )
        telemetry.record_llm_call(
            incident_id=None,
            operation="replan",
            model_name="MockLLMClient",
            duration_seconds=time.perf_counter() - t0,
        )
        return result

    def _replan_impl(
        self,
        original_target: str,
        first_action_taken: str,
        new_target: str,
        new_log_content: str,
        verification_evidence: str,
        retry_count: int,
        execution_records: Optional[List[Dict[str, Any]]] = None,
        verification_records: Optional[List[Dict[str, Any]]] = None,
        investigation_evidence: Optional[List[str]] = None,
        original_diagnosis: Optional[Dict[str, Any]] = None,
    ) -> ReplannerResult:
        # Case 1: Check if previous remediation action failed.
        # Note (M11): If the last record was `resume_workflow` failing because a DOWNSTREAM job
        # failed (`new_target != original_target`), the preceding remediation tool succeeded and
        # exposed the downstream failure — fall through to Case 2 instead of blaming the upstream fix.
        if execution_records:
            last_record = execution_records[-1]
            last_tool = last_record.get("tool_name") if isinstance(last_record, dict) else getattr(last_record, "tool_name", "")
            success = last_record.get("success") if isinstance(last_record, dict) else getattr(last_record, "success", True)
            is_downstream_resume_halt = (
                last_tool == "resume_workflow"
                and not success
                and new_target
                and new_target != original_target
            )
            if not success and not is_downstream_resume_halt:
                stderr = last_record.get("stderr") if isinstance(last_record, dict) else getattr(last_record, "stderr", "")
                exit_code = last_record.get("exit_code") if isinstance(last_record, dict) else getattr(last_record, "exit_code", 1)
                orig_diag = original_diagnosis or {}
                return ReplannerResult(
                    new_target_service=original_target,
                    updated_triage_summary=(
                        f"Replanning cycle {retry_count}: Previous action '{first_action_taken}' FAILED "
                        f"with exit code {exit_code}: {stderr}. Action did not resolve {original_target}."
                    ),
                    updated_hypothesis=(
                        f"Remediation action '{first_action_taken}' on {original_target} failed to execute or satisfy postconditions. "
                        "Revising diagnosis to formulate an alternative remediation strategy."
                    ),
                    new_failure_type=orig_diag.get("failure_type", "UNKNOWN"),
                    implicated_resource=orig_diag.get("implicated_resource", original_target),
                    is_recoverable=True,
                    diagnosis_revision_needed=True,
                    alternative_strategy_notes=f"Previous execution of '{first_action_taken}' failed: {stderr}",
                )

        # Case 2: Previous action succeeded; deep verification exposed failure
        log_lower = new_log_content.lower()

        if "connection" in log_lower and "refused" in log_lower:
            svc_match = re.search(r"'([\w_]+service[\w_]*)'", new_log_content)
            svc_name = svc_match.group(1) if svc_match else new_target
            new_failure_type = "SERVICE_OFFLINE"
            new_hypothesis = (
                f"'{svc_name}' is offline. After {original_target} remediation by '{first_action_taken}', "
                f"the pipeline advanced to {new_target} which cannot connect to '{svc_name}'."
            )
            updated_summary = (
                f"Replanning cycle {retry_count}: Action '{first_action_taken}' executed cleanly on {original_target}. "
                f"Deep verification revealed {new_target} now fails: connection to '{svc_name}' refused on port 5432."
            )
            return ReplannerResult(
                new_target_service=new_target,
                updated_triage_summary=updated_summary,
                updated_hypothesis=new_hypothesis,
                new_failure_type=new_failure_type,
                implicated_resource=svc_name,
                is_recoverable=True,
                diagnosis_revision_needed=False,
            )

        elif "transform_job.lock" in new_log_content or ".lock" in new_log_content:
            lock_match = re.search(r"'?([\w./]+\.lock)'?", new_log_content)
            lock_name = lock_match.group(1) if lock_match else "unknown.lock"
            new_failure_type = "STALE_LOCKFILE"
            new_hypothesis = (
                f"A stale lockfile '{lock_name}' is blocking {new_target} "
                f"after {original_target} remediation by '{first_action_taken}'."
            )
            updated_summary = (
                f"Replanning cycle {retry_count}: {original_target} remediation completed. "
                f"Downstream {new_target} now blocked by orphaned '{lock_name}'."
            )
            return ReplannerResult(
                new_target_service=new_target,
                updated_triage_summary=updated_summary,
                updated_hypothesis=new_hypothesis,
                new_failure_type=new_failure_type,
                implicated_resource=lock_name,
                is_recoverable=True,
                diagnosis_revision_needed=False,
            )

        elif "checksum mismatch" in log_lower or "zero-byte" in log_lower or "corrupt" in log_lower:
            return ReplannerResult(
                new_target_service=new_target,
                updated_triage_summary=(
                    f"Replanning cycle {retry_count}: Pipeline verification detected unrecoverable data corruption in {new_target}."
                ),
                updated_hypothesis="Unrecoverable corruption detected in downstream data payload.",
                new_failure_type="DATA_CORRUPTION",
                implicated_resource=new_target,
                is_recoverable=False,
                is_escalation=True,
                escalation_reason="Data corruption in payload cannot be reconstituted by automated tooling.",
                diagnosis_revision_needed=True,
            )

        # Generic downstream failure
        last_line = new_log_content.strip().splitlines()[-1] if new_log_content.strip() else "No log content"
        return ReplannerResult(
            new_target_service=new_target,
            updated_triage_summary=(
                f"Replanning cycle {retry_count}: Verification revealed remaining failure in {new_target}: {last_line}"
            ),
            updated_hypothesis=(
                f"Verification failure in {new_target}. Log tail: {last_line}"
            ),
            new_failure_type="DEPENDENCY_FAILURE",
            implicated_resource=new_target,
            is_recoverable=False,
            is_escalation=True,
            escalation_reason=f"No automated runbook found for failure signature in {new_target}: {last_line}",
            diagnosis_revision_needed=True,
        )


# ---------------------------------------------------------------------------
# Gemini production client
# ---------------------------------------------------------------------------

class GeminiLLMClient(BaseLLMClient):
    """
    Real Gemini LLM backend using google-generativeai.

    Uses JSON mode (response_mime_type='application/json') so the model
    is constrained to produce valid JSON that we parse into Pydantic models.

    M12 enhancements:
      - Dynamic tool discovery directly from `available_tools` passed by ToolRunner.
      - Token usage tracking (`input_tokens`, `output_tokens`, `total_tokens`).
      - Zero hardcoded workflow IDs or static tool catalogs in operational prompts.
      - Optional structured reflection (`enable_reflection=True`) across investigation,
        planning, and replanning (toggleable for ablation studies).
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-3.5-flash-lite",
        enable_reflection: bool = True,
    ):
        if not api_key or not api_key.strip():
            raise ValueError(
                "Gemini API key is required but not provided. "
                "Set GEMINI_API_KEY environment variable."
            )
        try:
            import google.generativeai as genai
        except ImportError as e:
            raise ImportError(
                "google-generativeai is required for GeminiLLMClient. "
                "Install it with: pip install google-generativeai"
            ) from e

        genai.configure(api_key=api_key)
        self._model_name = model_name
        self._genai = genai
        self.enable_reflection = enable_reflection

    @staticmethod
    def _format_tools(available_tools: List[Dict[str, Any]]) -> str:
        """Formats runtime ToolRunner tool descriptors (names, params, risk, descriptions)."""
        if not available_tools:
            return "No tools are currently registered."
        lines: List[str] = []
        for tool in available_tools:
            name = tool.get("name", "unknown")
            risk = tool.get("risk_level", "UNKNOWN")
            desc = tool.get("description", "")
            params = tool.get("parameters") or {}
            if isinstance(params, dict) and params:
                param_str = ", ".join(f"{k}: {v}" for k, v in params.items())
            else:
                param_str = "(no parameters)"
            lines.append(f"- {name} [{risk}] — {desc}\n  Parameters: {param_str}")
        return "\n".join(lines)

    @staticmethod
    def _extract_token_usage(response: Any) -> tuple[int, int, int]:
        meta = getattr(response, "usage_metadata", None)
        if meta is None:
            return (0, 0, 0)
        in_tok = int(getattr(meta, "prompt_token_count", 0) or 0)
        out_tok = int(getattr(meta, "candidates_token_count", 0) or 0)
        tot_tok = int(getattr(meta, "total_token_count", 0) or (in_tok + out_tok))
        return (in_tok, out_tok, tot_tok)

    def _call(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Type[T],
        operation: str = "llm_call",
    ) -> T:
        """
        Core LLM call with JSON mode, Pydantic validation, structured telemetry,
        token tracking, and rate-limit backoff. Never logs API keys.
        """
        max_retries = 4
        t_total = time.perf_counter()
        retries_used = 0
        rate_limits_hit = 0
        raw_text = ""
        in_tokens = 0
        out_tokens = 0
        total_tokens = 0

        for attempt in range(max_retries):
            try:
                model = self._genai.GenerativeModel(
                    model_name=self._model_name,
                    system_instruction=system_prompt,
                    generation_config=self._genai.GenerationConfig(
                        response_mime_type="application/json",
                        temperature=0.1,
                    ),
                )

                response = model.generate_content(user_prompt)
                in_tokens, out_tokens, total_tokens = self._extract_token_usage(response)
                raw_text = (getattr(response, "text", None) or "").strip()
                if not raw_text:
                    raise ValueError("Gemini API returned an empty response body.")
                break
            except Exception as e:
                err_str = redact_secrets(str(e))
                is_rate_limit = (
                    "429" in err_str
                    or "Quota exceeded" in err_str
                    or "ResourceExhausted" in err_str
                    or "rate limit" in err_str.lower()
                )
                if is_rate_limit:
                    rate_limits_hit += 1
                if is_rate_limit and attempt < max_retries - 1:
                    retries_used += 1
                    retry_match = re.search(
                        r"retry(?:_delay)?\s*\{?\s*seconds:\s*(\d+)",
                        err_str,
                        re.IGNORECASE,
                    )
                    wait_sec = (
                        int(retry_match.group(1)) + 2
                        if retry_match
                        else (15 * (attempt + 1))
                    )
                    log_event(
                        logger,
                        logging.WARNING,
                        "llm_rate_limit_backoff",
                        f"Gemini rate limit encountered during '{operation}'; backing off {wait_sec}s (attempt {attempt + 1}/{max_retries})",
                        operation=operation,
                        model=self._model_name,
                        attempt=attempt + 1,
                        wait_seconds=wait_sec,
                    )
                    time.sleep(wait_sec)
                else:
                    dt = time.perf_counter() - t_total
                    telemetry.record_llm_call(
                        incident_id=None,
                        operation=operation,
                        model_name=self._model_name,
                        duration_seconds=dt,
                        retries=retries_used,
                        rate_limits=rate_limits_hit,
                        success=False,
                        error=err_str,
                        input_tokens=in_tokens,
                        output_tokens=out_tokens,
                        total_tokens=total_tokens,
                    )
                    log_event(
                        logger,
                        logging.ERROR,
                        "llm_api_error",
                        f"Gemini call '{operation}' failed: {err_str}",
                        operation=operation,
                        model=self._model_name,
                        retries=retries_used,
                        rate_limits=rate_limits_hit,
                        duration_seconds=round(dt, 4),
                    )
                    raise

        # Strip markdown code fences if the model adds them
        if raw_text.startswith("```"):
            raw_text = re.sub(r"^```(?:json)?\n?", "", raw_text)
            raw_text = re.sub(r"\n?```$", "", raw_text)

        try:
            parsed = json.loads(raw_text)
            validated = schema.model_validate(parsed)
        except (json.JSONDecodeError, Exception) as e:
            dt = time.perf_counter() - t_total
            safe_err = redact_secrets(str(e))
            telemetry.record_llm_call(
                incident_id=None,
                operation=operation,
                model_name=self._model_name,
                duration_seconds=dt,
                retries=retries_used,
                rate_limits=rate_limits_hit,
                success=False,
                error=f"Invalid JSON/schema for {schema.__name__}: {safe_err}",
                input_tokens=in_tokens,
                output_tokens=out_tokens,
                total_tokens=total_tokens,
            )
            raise ValueError(
                f"LLM returned invalid JSON for schema {schema.__name__}. "
                f"Raw response: {raw_text!r}. Error: {e}"
            ) from e

        dt = time.perf_counter() - t_total
        telemetry.record_llm_call(
            incident_id=None,
            operation=operation,
            model_name=self._model_name,
            duration_seconds=dt,
            retries=retries_used,
            rate_limits=rate_limits_hit,
            success=True,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            total_tokens=total_tokens,
        )
        log_event(
            logger,
            logging.INFO,
            "llm_call_completed",
            f"Gemini '{operation}' completed in {dt:.2f}s",
            operation=operation,
            model=self._model_name,
            retries=retries_used,
            rate_limits=rate_limits_hit,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            total_tokens=total_tokens,
            duration_seconds=round(dt, 4),
        )
        return validated

    def investigate(
        self,
        target_service: Optional[str],
        raw_symptoms: str,
        investigation_evidence: List[str],
        diagnostic_history: List[Dict[str, Any]],
        available_tools: List[Dict[str, Any]],
        step_number: int,
    ) -> InvestigationDecision:
        tools_text = self._format_tools(available_tools)
        reflection_schema = (
            """,
  "reflection": {
    "evidence_sufficient": boolean,
    "alternative_hypotheses": [string],
    "remaining_uncertainty": string | null
  }"""
            if self.enable_reflection
            else ""
        )
        reflection_rules = (
            """
- Before choosing DIAGNOSIS_READY, perform diagnostic reflection: verify evidence_sufficient=true against actual logs/status outputs, list plausible alternative hypotheses considered, and if genuine uncertainty remains, choose DIAGNOSTIC_ACTION to gather missing evidence."""
            if self.enable_reflection
            else ""
        )

        system_prompt = f"""You are an expert SRE investigator diagnosing an enterprise batch workflow incident.
Your goal is to gather empirical evidence using ONLY the registered read-only diagnostic tools below before concluding a diagnosis or escalating.

Available READ-ONLY Diagnostic Tools (discovered dynamically from registry):
{tools_text}

Output MUST be valid JSON matching EXACTLY one of these decisions:

1. Request another diagnostic action (if evidence is incomplete):
{{
  "decision_type": "DIAGNOSTIC_ACTION",
  "diagnostic_tool": {{
    "tool_name": string,
    "parameters": {{}},
    "rationale": string
  }}{reflection_schema}
}}

2. Diagnosis ready (when root cause is proven by empirical evidence):
{{
  "decision_type": "DIAGNOSIS_READY",
  "diagnosis": {{
    "failure_type": "SERVICE_OFFLINE" | "DATA_CORRUPTION" | "SOURCE_UNAVAILABLE" | "MALFORMED_INPUT" | "STALE_LOCKFILE" | "DEPENDENCY_FAILURE" | "UNKNOWN",
    "implicated_resource": string,
    "triage_summary": string,
    "root_cause_hypothesis": string,
    "is_recoverable": boolean,
    "confidence": "HIGH" | "MEDIUM" | "LOW"
  }}{reflection_schema}
}}

3. Escalation required (if workflow is healthy false-positive, unrecoverable data/input/external defect, or out of scope):
{{
  "decision_type": "ESCALATE",
  "escalation_reason": string{reflection_schema}
}}

Rules:
- NEVER invent tool names or parameters not listed in Available READ-ONLY Diagnostic Tools above.
- Start with `get_workflow_status` if execution/task status has not yet been observed.
- Inspect the failed task's logs (`read_job_logs`) and relevant service health (`check_service_status`) before concluding root cause.
- If workflow status shows all tasks SUCCESS (healthy workflow, no failed tasks), return ESCALATE indicating a healthy workflow / false-positive alert with no intervention needed.
- If lockfiles show STATUS=ACTIVE_WORKER or recent live heartbeat, return ESCALATE (never delete active locks).
- Do NOT mark external source outages (e.g. HTTP 404/upstream release unreachable), malformed CSV/input schema errors, or checksum data corruption as recoverable unless a registered remediation tool can actually fix them.
- Do NOT repeat identical diagnostic tool calls with identical parameters.{reflection_rules}
"""
        evidence_text = "\n".join(f"- {ev}" for ev in investigation_evidence) if investigation_evidence else "None yet."
        history_text = json.dumps(diagnostic_history, indent=2) if diagnostic_history else "No diagnostic tools run yet."

        user_prompt = f"""Current Investigation State (Step {step_number}):
Target Service / Task: {target_service or 'UNKNOWN'}
Raw Symptoms: {raw_symptoms}

Accumulated Evidence:
{evidence_text}

Diagnostic History:
{history_text}

Determine the next investigation decision as JSON."""

        return self._call(system_prompt, user_prompt, InvestigationDecision, operation="investigate")

    def triage(self, target_service: str, log_content: str) -> TriageResult:
        system_prompt = f"""You are an expert SRE (Site Reliability Engineer) specializing in batch job incident diagnosis.

Your task: analyze a failed job's log and produce a structured diagnosis.

Output MUST be valid JSON matching this exact schema:
{{
  "failure_type": string,
  "implicated_resource": string,
  "triage_summary": string,
  "root_cause_hypothesis": string,
  "is_recoverable": boolean,
  "confidence": "HIGH" | "MEDIUM" | "LOW"
}}

Rules:
- implicated_resource MUST come from the log or observed telemetry — never invent names.
- is_recoverable = false for data corruption, malformed input, missing upstream external files, or unfixable defects.
- Do not add fields beyond the schema.
"""
        user_prompt = f"""Failed service/task: {target_service}

Log content (last 50 lines):
{log_content}

Analyze and return JSON diagnosis."""

        return self._call(system_prompt, user_prompt, TriageResult, operation="triage")

    def plan(
        self,
        triage: TriageResult,
        available_tools: List[Dict[str, Any]],
        retry_count: int,
    ) -> PlannerResult:
        tools_text = self._format_tools(available_tools)
        reflection_schema = (
            """,
  "reflection": {
    "addresses_root_cause": boolean,
    "contradicts_prior_attempts": boolean,
    "strategy_assessment": string
  }"""
            if self.enable_reflection
            else ""
        )
        reflection_rules = (
            """
- Perform planning reflection: verify your plan directly addresses the root cause, does NOT repeat a strategy that already failed in Prior Attempts without new justification, and uses exact parameter names/values from observed evidence."""
            if self.enable_reflection
            else ""
        )

        system_prompt = f"""You are a remediation planner for an enterprise workflow orchestrator.

Given a structured incident triage, prior attempt outcomes, and the LIVE tool registry below, produce an ordered remediation plan or escalate.

Available Tools (discovered dynamically from active ToolRunner registry):
{tools_text}

Output MUST be valid JSON:
{{
  "is_escalation": boolean,
  "escalation_reason": string | null,
  "actions": [
    {{
      "tool_name": string,
      "parameters": {{}},
      "risk_level": "READ_ONLY" | "LOW_RISK" | "DESTRUCTIVE",
      "rationale": string,
      "expected_outcome": string
    }}
  ]{reflection_schema}
}}

Rules:
- You MUST ONLY choose tools listed in Available Tools above. Never reference unregistered tools.
- If no registered tool can resolve the diagnosed failure (e.g. external source unavailable, malformed CSV/input, data corruption, non-restartable dependency), set is_escalation=true, actions=[], and explain why in escalation_reason.
- If is_escalation=true, actions MUST be empty and escalation_reason MUST be non-empty.
- If is_escalation=false, escalation_reason MUST be null and actions MUST be non-empty.
- parameters MUST use the exact parameter keys documented for each tool and values derived strictly from context (never invent IDs or names).
- After any mutating remediation action that fixes an underlying blocker, include `resume_workflow` as the final action using the exact active execution/workflow parameter required by `resume_workflow` in Available Tools above (the verifier is strictly read-only and will not run the workflow itself).
- Inspect Prior Attempts carefully: this is retry cycle {retry_count + 1}. Do not repeat actions/strategies that already failed.{reflection_rules}
"""
        prior_json = json.dumps(triage.prior_attempts or [], indent=2)
        evidence_text = "\n".join(f"- {ev}" for ev in (triage.investigation_evidence or [])) or "None."
        user_prompt = f"""Triage Result:
- workflow_id: {triage.workflow_id or 'UNKNOWN'}
- workflow_run_id / execution_id: {triage.workflow_run_id or 'UNKNOWN'}
- failure_type: {triage.failure_type}
- implicated_resource: {triage.implicated_resource}
- triage_summary: {triage.triage_summary}
- root_cause_hypothesis: {triage.root_cause_hypothesis}
- is_recoverable: {triage.is_recoverable}
- replanner_strategy_notes: {triage.strategy_notes or 'None'}

Investigation Evidence:
{evidence_text}

Prior Attempted Actions & Verification Outcomes:
{prior_json}

Produce the remediation plan as JSON."""

        return self._call(system_prompt, user_prompt, PlannerResult, operation="plan")

    def replan(
        self,
        original_target: str,
        first_action_taken: str,
        new_target: str,
        new_log_content: str,
        verification_evidence: str,
        retry_count: int,
        execution_records: Optional[List[Dict[str, Any]]] = None,
        verification_records: Optional[List[Dict[str, Any]]] = None,
        investigation_evidence: Optional[List[str]] = None,
        original_diagnosis: Optional[Dict[str, Any]] = None,
    ) -> ReplannerResult:
        reflection_fields = (
            """,
  "needs_further_investigation": boolean,
  "outcome_reflection": string | null"""
            if self.enable_reflection
            else ""
        )
        reflection_instructions = (
            """
- Perform outcome reflection: explicitly explain in `outcome_reflection` why the previous action or workflow restart failed. If the current logs/evidence are insufficient or contradict the original diagnosis, set `needs_further_investigation=true` so NOVA returns to diagnostic investigation with full context."""
            if self.enable_reflection
            else ""
        )

        system_prompt = f"""You are an expert SRE replanning agent.
An initial remediation plan was executed, and independent post-condition verification has provided new telemetry.

Your task: analyze the actual execution and verification evidence to determine the current operational reality.

Output MUST be valid JSON:
{{
  "new_target_service": string,
  "updated_triage_summary": string,
  "updated_hypothesis": string,
  "new_failure_type": "SERVICE_OFFLINE" | "DATA_CORRUPTION" | "SOURCE_UNAVAILABLE" | "MALFORMED_INPUT" | "STALE_LOCKFILE" | "DEPENDENCY_FAILURE" | "UNKNOWN",
  "implicated_resource": string | null,
  "is_recoverable": boolean,
  "diagnosis_revision_needed": boolean,
  "is_escalation": boolean,
  "escalation_reason": string | null,
  "alternative_strategy_notes": string | null{reflection_fields}
}}

Critical Reasoning Instructions:
- Do NOT assume the previous action succeeded. Check Execution Records and Verification Records carefully.
- If a remediation tool (e.g. restart_service, clear_lockfile) failed or failed its postcondition, revise the hypothesis (`diagnosis_revision_needed=true`) and determine an alternative strategy, additional investigation, or escalation.
- If an upstream remediation tool succeeded and `resume_workflow` halted on a DIFFERENT downstream failed task, diagnose the newly surfaced failure in `new_target` from its log.
- If an external source is unavailable, input data is malformed/corrupt, or no registered tool can fix the newly surfaced issue, set `is_escalation=true`, `escalation_reason` non-empty, and `is_recoverable=false`.
- If `is_escalation=false`, `escalation_reason` MUST be null.
- `implicated_resource` must come strictly from logs or telemetry — never invent names.{reflection_instructions}
"""
        exec_json = json.dumps(execution_records or [], indent=2)
        verif_json = json.dumps(verification_records or [], indent=2)
        orig_diag_json = json.dumps(original_diagnosis or {}, indent=2)
        inv_text = "\n".join(f"- {ev}" for ev in (investigation_evidence or [])) or "None."

        user_prompt = f"""Replanning Telemetry (Cycle {retry_count}):
Original Target: {original_target}
Action Attempted: {first_action_taken}
Original Diagnosis:
{orig_diag_json}

Prior Investigation Evidence:
{inv_text}

Execution Records:
{exec_json}

Verification Records & Evidence:
{verif_json}
Verification Detail: {verification_evidence}

New Target Candidate: {new_target}
Diagnostic Log for {new_target}:
{new_log_content}

Replan as JSON."""

        return self._call(system_prompt, user_prompt, ReplannerResult, operation="replan")

