"""
Unit Tests for Autonomous Investigation & Diagnostic Tool Registry (Milestone 7.2 Phase C).

Verifies:
1. Multi-step diagnostic investigation starting with UNKNOWN target service.
2. Ground-truth security enforcement: Destructive tools strictly disallowed in investigation.
3. Unproductive loop detection when an LLM requests identical diagnostic calls.
4. Bounded investigation budget enforcement (max_investigation_steps).
5. Deterministic read-only diagnostic tools (get_workflow_status, inspect_job_dependencies, inspect_lockfiles).
"""
import pytest

from nova.environments.mock_workflow_runner import (
    MockWorkflowRunner,
    ScenarioType,
)
from nova.llm.client import MockLLMClient
from nova.llm.schemas import (
    DiagnosticToolCall,
    InvestigationActionType,
    InvestigationDecision,
    TriageResult,
)
from nova.nodes.triage import create_triage_node
from nova.state import IncidentStatus, NovaGraphState, RiskLevel
from nova.tools.runner import create_default_tool_runner
from nova.tools.workflow_tools import (
    GetWorkflowStatusTool,
    InspectJobDependenciesTool,
    InspectLockfilesTool,
)


@pytest.fixture
def env(tmp_path):
    runner = MockWorkflowRunner(base_dir=tmp_path / "mock_state")
    runner.inject_scenario(ScenarioType.STALE_LOCKFILE)
    return runner


@pytest.fixture
def runner(env):
    return create_default_tool_runner(env)


def test_diagnostic_tools_read_only_contracts(env, runner):
    """Verify that all diagnostic tools are registered and strictly READ_ONLY."""
    diag_tool_names = [
        "get_workflow_status",
        "inspect_job_dependencies",
        "inspect_lockfiles",
        "check_service_status",
        "read_job_logs",
    ]
    for name in diag_tool_names:
        assert runner.is_tool_allowed(name)
        tool = runner.get_tool(name)
        assert tool.risk_level == RiskLevel.READ_ONLY


def test_get_workflow_status_tool_discovers_failure(env):
    """Verify get_workflow_status returns high-level job execution telemetry."""
    tool = GetWorkflowStatusTool(env)
    # Trigger a workflow failure first
    env.run_workflow()

    result = tool.run()
    assert result.success is True
    assert result.exit_code == 0
    assert "JOB_02_TRANSFORM: status=FAILED" in result.stdout
    assert "JOB_02_TRANSFORM" in result.data["failed_jobs"]
    assert "transform_job.lock" in result.data["active_locks"]


def test_inspect_job_dependencies_tool(env):
    """Verify inspect_job_dependencies returns DAG topology."""
    tool = InspectJobDependenciesTool(env)
    res_all = tool.run()
    assert res_all.success is True
    assert "JOB_02_TRANSFORM depends on: ['JOB_01_EXTRACT']" in res_all.stdout

    res_single = tool.run(job_id="JOB_02_TRANSFORM")
    assert res_single.success is True
    assert res_single.data["dependencies"] == ["JOB_01_EXTRACT"]
    assert "JOB_03_LOAD" in res_single.data["downstream"]


def test_inspect_lockfiles_tool(env):
    """Verify inspect_lockfiles discovers active lockfiles on disk."""
    tool = InspectLockfilesTool(env)
    res = tool.run()
    assert res.success is True
    assert "transform_job.lock" in res.data["locks"]

    res_single = tool.run(lock_name="transform_job.lock")
    assert res_single.success is True
    assert res_single.data["present"] is True
    assert "PID=9182" in res_single.stdout


def test_autonomous_investigation_with_unknown_target_service(env, runner):
    """
    Test autonomous multi-step investigation starting from target_service='UNKNOWN'.
    The LLM dynamically:
      Step 1: Queries workflow status -> discovers JOB_02_TRANSFORM failed.
      Step 2: Reads logs for JOB_02_TRANSFORM -> discovers transform_job.lock.
      Step 3: Inspects lock directory -> confirms lock on disk.
      Step 4: Concludes diagnosis ready -> transitions to PLANNING.
    """
    env.run_workflow()  # Populate failed job state
    llm = MockLLMClient()
    triage_node = create_triage_node(env, llm, tool_runner=runner)

    state: NovaGraphState = {
        "incident_id": "INC_INV_01",
        "target_service": "UNKNOWN",
        "raw_symptoms": "Pipeline execution stalled",
        "status": IncidentStatus.TRIGGERED,
        "investigation_steps": 0,
        "max_investigation_steps": 5,
        "investigation_evidence": [],
        "diagnostic_history": [],
    }

    # Step 1: Discover failing job
    out_1 = triage_node(state)
    assert out_1["status"] == IncidentStatus.TRIAGING
    assert out_1["investigation_steps"] == 1
    assert out_1["diagnostic_history"][0].tool_name == "get_workflow_status"

    state["diagnostic_history"].extend(out_1["diagnostic_history"])
    state["investigation_evidence"].extend(out_1["investigation_evidence"])
    state["investigation_steps"] = out_1["investigation_steps"]

    # Step 2: Read job logs
    out_2 = triage_node(state)
    assert out_2["status"] == IncidentStatus.TRIAGING
    assert out_2["investigation_steps"] == 2
    assert out_2["diagnostic_history"][0].tool_name == "read_job_logs"
    assert out_2["diagnostic_history"][0].parameters["job_id"] == "JOB_02_TRANSFORM"

    state["diagnostic_history"].extend(out_2["diagnostic_history"])
    state["investigation_evidence"].extend(out_2["investigation_evidence"])
    state["investigation_steps"] = out_2["investigation_steps"]

    # Step 3: Inspect lockfiles
    out_3 = triage_node(state)
    assert out_3["status"] == IncidentStatus.TRIAGING
    assert out_3["investigation_steps"] == 3
    assert out_3["diagnostic_history"][0].tool_name == "inspect_lockfiles"

    state["diagnostic_history"].extend(out_3["diagnostic_history"])
    state["investigation_evidence"].extend(out_3["investigation_evidence"])
    state["investigation_steps"] = out_3["investigation_steps"]

    # Step 4: Diagnosis ready!
    out_4 = triage_node(state)
    assert out_4["status"] == IncidentStatus.PLANNING
    assert out_4["failure_type"] == "STALE_LOCKFILE"
    assert "transform_job.lock" in out_4["implicated_resource"]
    assert out_4["is_recoverable"] is True
    assert out_4["target_service"] == "JOB_02_TRANSFORM"


def test_investigation_security_rejects_destructive_tools(env, runner):
    """
    If an LLM or adversarial input attempts to invoke a destructive tool
    (e.g. clear_lockfile) during investigation, triage_node MUST fail-closed
    and escalate immediately without executing the tool.
    """
    class MaliciousLLM(MockLLMClient):
        def investigate(self, *args, **kwargs) -> InvestigationDecision:
            return InvestigationDecision(
                decision_type=InvestigationActionType.DIAGNOSTIC_ACTION,
                diagnostic_tool=DiagnosticToolCall(
                    tool_name="clear_lockfile",  # DESTRUCTIVE!
                    parameters={"lock_name": "transform_job.lock"},
                    rationale="Attempting premature destructive execution",
                ),
            )

    triage_node = create_triage_node(env, MaliciousLLM(), tool_runner=runner)

    state: NovaGraphState = {
        "incident_id": "INC_MAL_01",
        "target_service": "JOB_02_TRANSFORM",
        "raw_symptoms": "Stalled job",
        "status": IncidentStatus.TRIGGERED,
        "investigation_steps": 0,
        "max_investigation_steps": 5,
        "investigation_evidence": [],
        "diagnostic_history": [],
    }

    result = triage_node(state)
    assert result["status"] == IncidentStatus.ESCALATED
    assert "Security Alert: Non-read-only tool 'clear_lockfile'" in result["final_summary"]


def test_investigation_unproductive_loop_detection(env, runner):
    """
    If the LLM gets stuck requesting the exact same diagnostic tool with identical
    parameters, triage_node MUST detect the loop and halt safely.
    """
    class LoopingLLM(MockLLMClient):
        def investigate(self, *args, **kwargs) -> InvestigationDecision:
            return InvestigationDecision(
                decision_type=InvestigationActionType.DIAGNOSTIC_ACTION,
                diagnostic_tool=DiagnosticToolCall(
                    tool_name="get_workflow_status",
                    parameters={},
                    rationale="Querying status repeatedly",
                ),
            )

    triage_node = create_triage_node(env, LoopingLLM(), tool_runner=runner)

    state: NovaGraphState = {
        "incident_id": "INC_LOOP_01",
        "target_service": "UNKNOWN",
        "raw_symptoms": "Pipeline stuck",
        "status": IncidentStatus.TRIGGERED,
        "investigation_steps": 0,
        "max_investigation_steps": 5,
        "investigation_evidence": [],
        "diagnostic_history": [],
    }

    # First call succeeds
    out_1 = triage_node(state)
    assert out_1["status"] == IncidentStatus.TRIAGING

    state["diagnostic_history"].extend(out_1["diagnostic_history"])
    state["investigation_evidence"].extend(out_1["investigation_evidence"])
    state["investigation_steps"] = out_1["investigation_steps"]

    # Second identical call detected as unproductive loop!
    out_2 = triage_node(state)
    assert out_2["status"] == IncidentStatus.ESCALATED
    assert "repeated unproductive investigation call" in out_2["final_summary"]


def test_investigation_budget_exhaustion(env, runner):
    """
    Verify that an investigation halts cleanly when reaching max_investigation_steps.
    """
    class AlwaysMoreEvidenceLLM(MockLLMClient):
        def investigate(self, *args, **kwargs) -> InvestigationDecision:
            step = kwargs.get("step_number", 1)
            return InvestigationDecision(
                decision_type=InvestigationActionType.DIAGNOSTIC_ACTION,
                diagnostic_tool=DiagnosticToolCall(
                    tool_name="read_job_logs",
                    parameters={"job_id": f"JOB_0{step}_EXTRACT", "tail_lines": 10},
                    rationale=f"Reading step {step}",
                ),
            )

    triage_node = create_triage_node(env, AlwaysMoreEvidenceLLM(), tool_runner=runner)

    state: NovaGraphState = {
        "incident_id": "INC_BUDGET_01",
        "target_service": "JOB_01_EXTRACT",
        "raw_symptoms": "Stalled job",
        "status": IncidentStatus.TRIGGERED,
        "investigation_steps": 2,  # Already at budget limit
        "max_investigation_steps": 2,
        "investigation_evidence": [],
        "diagnostic_history": [],
    }

    result = triage_node(state)
    assert result["status"] == IncidentStatus.ESCALATED
    assert "exceeded maximum budget (2 steps)" in result["final_summary"]
