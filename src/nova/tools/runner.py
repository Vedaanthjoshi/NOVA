import logging
import time
from typing import Dict, Optional

from nova.environments.mock_workflow_runner import MockWorkflowRunner
from nova.observability import log_event, telemetry
from nova.state import ExecutionRecord, PlannedAction, RiskLevel
from nova.tools.base import BaseTool, ParameterValidationError, ToolResult, UnknownToolError
from nova.tools.workflow_tools import (
    CheckServiceStatusTool,
    ClearLockfileTool,
    GetWorkflowStatusTool,
    InspectJobDependenciesTool,
    InspectLockfilesTool,
    ReadJobLogsTool,
    RestartServiceTool,
    ResumeWorkflowTool,
)

logger = logging.getLogger("nova.tools.runner")


class ToolRunner:
    """
    Central, deterministic execution engine for all agent tools.
    Enforces the approved tool registry whitelist, argument schema validation,
    execution latency measurement, and standardized ExecutionRecord creation.
    """

    def __init__(self, registry: Optional[Dict[str, BaseTool]] = None):
        self._registry: Dict[str, BaseTool] = registry or {}

    def register(self, tool: BaseTool) -> None:
        """Adds a tool to the whitelist registry."""
        self._registry[tool.name] = tool

    def get_tool(self, tool_name: str) -> BaseTool:
        """Retrieves a tool from registry or raises UnknownToolError."""
        if tool_name not in self._registry:
            raise UnknownToolError(
                f"Security Alert: Tool '{tool_name}' is not in the approved whitelist registry. "
                f"Available tools: {list(self._registry.keys())}"
            )
        return self._registry[tool_name]

    def is_tool_allowed(self, tool_name: str) -> bool:
        """Returns True if the tool exists in the approved registry."""
        return tool_name in self._registry

    def execute(self, action: PlannedAction) -> ExecutionRecord:
        """
        Safely executes a PlannedAction:
        1. Whitelist validation.
        2. Parameter validation.
        3. Execution duration tracking.
        4. Standardized ExecutionRecord generation.
        """
        tool = self.get_tool(action.tool_name)
        executed_at = time.strftime("%Y-%m-%d %H:%M:%S")

        start_time = time.perf_counter()
        is_diag = tool.risk_level == RiskLevel.READ_ONLY
        try:
            result: ToolResult = tool.run(**action.parameters)
            duration = time.perf_counter() - start_time
            telemetry.record_tool_call(
                incident_id=None,
                tool_name=action.tool_name,
                risk_level=tool.risk_level.value,
                is_diagnostic=is_diag,
                duration_seconds=duration,
                success=result.success,
            )
            log_event(
                logger,
                logging.INFO if result.success else logging.WARNING,
                "tool_executed",
                f"Executed tool '{action.tool_name}' (success={result.success}, exit_code={result.exit_code})",
                tool_name=action.tool_name,
                action_id=action.action_id,
                risk_level=tool.risk_level.value,
                success=result.success,
                exit_code=result.exit_code,
                duration_seconds=round(duration, 4),
            )

            return ExecutionRecord(
                action_id=action.action_id,
                tool_name=action.tool_name,
                executed_at=executed_at,
                exit_code=result.exit_code,
                stdout=result.stdout,
                stderr=result.stderr,
                duration_seconds=round(duration, 4),
                success=result.success,
                action_hash=action.action_hash,
            )

        except ParameterValidationError as pve:
            duration = time.perf_counter() - start_time
            telemetry.record_tool_call(
                incident_id=None,
                tool_name=action.tool_name,
                risk_level=tool.risk_level.value,
                is_diagnostic=is_diag,
                duration_seconds=duration,
                success=False,
            )
            log_event(
                logger,
                logging.WARNING,
                "tool_parameter_validation_failed",
                f"Parameter validation failed for '{action.tool_name}': {pve}",
                tool_name=action.tool_name,
                action_id=action.action_id,
                duration_seconds=round(duration, 4),
            )
            return ExecutionRecord(
                action_id=action.action_id,
                tool_name=action.tool_name,
                executed_at=executed_at,
                exit_code=126,  # Standard shell exit code for invalid parameters / command invoked cannot execute
                stdout="",
                stderr=f"Parameter Validation Error: {str(pve)}",
                duration_seconds=round(duration, 4),
                success=False,
                action_hash=action.action_hash,
            )

        except Exception as e:
            duration = time.perf_counter() - start_time
            telemetry.record_tool_call(
                incident_id=None,
                tool_name=action.tool_name,
                risk_level=tool.risk_level.value,
                is_diagnostic=is_diag,
                duration_seconds=duration,
                success=False,
            )
            log_event(
                logger,
                logging.ERROR,
                "tool_execution_exception",
                f"Unhandled exception executing '{action.tool_name}': {e}",
                tool_name=action.tool_name,
                action_id=action.action_id,
                duration_seconds=round(duration, 4),
            )
            return ExecutionRecord(
                action_id=action.action_id,
                tool_name=action.tool_name,
                executed_at=executed_at,
                exit_code=1,
                stdout="",
                stderr=f"Tool Execution Exception: {str(e)}",
                duration_seconds=round(duration, 4),
                success=False,
                action_hash=action.action_hash,
            )


def create_default_tool_runner(env: MockWorkflowRunner) -> ToolRunner:
    """Factory creating a ToolRunner loaded with all standard enterprise tools."""
    runner = ToolRunner()
    # Read-only diagnostic tools
    runner.register(ReadJobLogsTool(env))
    runner.register(CheckServiceStatusTool(env))
    runner.register(GetWorkflowStatusTool(env))
    runner.register(InspectJobDependenciesTool(env))
    runner.register(InspectLockfilesTool(env))
    # Destructive / Low-risk remediation tools
    runner.register(ClearLockfileTool(env))
    runner.register(RestartServiceTool(env))
    runner.register(ResumeWorkflowTool(env))
    return runner

