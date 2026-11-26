from typing import Literal, Optional

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from nova.environments.mock_workflow_runner import MockWorkflowRunner
from nova.llm.client import BaseLLMClient, MockLLMClient
from nova.nodes.executor import create_executor_node
from nova.nodes.planner import create_planner_node
from nova.nodes.replanner import create_replanner_node
from nova.nodes.triage import create_triage_node
from nova.nodes.verifier import create_verifier_node
from nova.state import IncidentStatus, NovaGraphState
from nova.tools.runner import ToolRunner, create_default_tool_runner


def route_after_triage(state: NovaGraphState) -> Literal["triage", "planner", "__end__"]:
    """
    Conditional routing after investigation / triage:
    - TRIAGING: LLM requested another diagnostic tool → loop back to triage.
    - ESCALATED / FAILED: Unrecoverable failure or budget exhausted → go to END.
    - PLANNING: Diagnosis ready with complete evidence → proceed to planner.
    """
    status = state.get("status")
    if status == IncidentStatus.TRIAGING:
        return "triage"
    if status in (IncidentStatus.ESCALATED, IncidentStatus.FAILED):
        return "__end__"
    return "planner"


def route_after_planner(state: NovaGraphState) -> Literal["executor", "__end__"]:
    """
    Conditional routing after planning:
    - ESCALATED: LLM determined failure is unrecoverable → go to END.
    - FAILED: Planner has no strategy → go to END.
    - Valid plan exists → go to executor.
    """
    status = state.get("status")
    plan = state.get("plan", [])

    if status in (IncidentStatus.ESCALATED, IncidentStatus.FAILED) or not plan:
        return "__end__"
    return "executor"


def route_after_executor(state: NovaGraphState) -> Literal["executor", "verifier", "__end__"]:
    """
    Conditional routing after execution (Multi-Step Plan Execution):
    - ESCALATED: Rejected by human or fatal security policy violation → go to END.
    - EXECUTING: Intermediate action succeeded; more actions remain in plan → loop to executor.
    - VERIFYING / FAILED: All actions complete, or action failed → go to verifier.
    """
    status = state.get("status")
    if status == IncidentStatus.ESCALATED:
        return "__end__"
    if status == IncidentStatus.EXECUTING:
        return "executor"
    return "verifier"


def route_after_verifier(state: NovaGraphState) -> Literal["replanner", "__end__"]:
    """
    Conditional routing after verification:
    - RESOLVED → END (success).
    - ESCALATED or retry budget exhausted → END (escalate).
    - Otherwise → replanner (bounded retry cycle).
    """
    status = state.get("status")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)

    if status == IncidentStatus.RESOLVED:
        return "__end__"
    if status == IncidentStatus.ESCALATED or retry_count >= max_retries:
        return "__end__"
    return "replanner"


def route_after_replanner(state: NovaGraphState) -> Literal["planner", "__end__"]:
    """
    Conditional routing after replanning:
    - ESCALATED: Replanner determined incident is unrecoverable → go to END.
    - Otherwise → go to planner to formulate the updated remediation plan.
    """
    status = state.get("status")
    if status in (IncidentStatus.ESCALATED, IncidentStatus.FAILED):
        return "__end__"
    return "planner"



def build_nova_graph(
    env: MockWorkflowRunner,
    tool_runner: Optional[ToolRunner] = None,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    require_approval: Optional[bool] = None,
    llm: Optional[BaseLLMClient] = None,
):
    """
    Constructs and compiles the core LangGraph state machine.

    Parameters:
        env: The MockWorkflowRunner providing the simulated enterprise environment.
        tool_runner: Optional custom ToolRunner. Defaults to the standard registry.
        checkpointer: Optional BaseCheckpointSaver for HITL state persistence.
                      When provided, HITL approval defaults to True.
        require_approval: Override for HITL policy. None = auto-detect from checkpointer.
        llm: LLM client for triage, planning, and replanning.
             Defaults to MockLLMClient for safety (never accidentally calls real API).

    Graph topology:
        START → triage → planner → [executor → verifier → replanner ↩] → END
                                 ↘ END (escalation / no plan)
    """
    runner = tool_runner or create_default_tool_runner(env)
    approval_policy = (checkpointer is not None) if require_approval is None else require_approval

    # Default to mock LLM if none provided — prevents accidental API calls in tests
    llm_client = llm or MockLLMClient()

    graph_builder = StateGraph(NovaGraphState)

    # Add all nodes — each factory injects its dependencies
    graph_builder.add_node("triage", create_triage_node(env, llm_client, tool_runner=runner))
    graph_builder.add_node("planner", create_planner_node(llm_client, env=env))
    graph_builder.add_node("executor", create_executor_node(runner, env=env, require_approval=approval_policy))
    graph_builder.add_node("verifier", create_verifier_node(env, tool_runner=runner))
    graph_builder.add_node("replanner", create_replanner_node(env, llm_client))

    # Wire transitions
    graph_builder.add_edge(START, "triage")

    graph_builder.add_conditional_edges(
        "triage",
        route_after_triage,
        {"triage": "triage", "planner": "planner", "__end__": END},
    )

    graph_builder.add_conditional_edges(
        "planner",
        route_after_planner,
        {"executor": "executor", "__end__": END},
    )

    graph_builder.add_conditional_edges(
        "executor",
        route_after_executor,
        {"executor": "executor", "verifier": "verifier", "__end__": END},
    )

    graph_builder.add_conditional_edges(
        "verifier",
        route_after_verifier,
        {"replanner": "replanner", "__end__": END},
    )

    graph_builder.add_conditional_edges(
        "replanner",
        route_after_replanner,
        {"planner": "planner", "__end__": END},
    )

    return graph_builder.compile(checkpointer=checkpointer)
