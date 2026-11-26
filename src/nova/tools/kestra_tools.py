"""
Kestra Workflow Tool Contracts for NOVA (Milestone 12).

Exposes ONLY the 6 operational tools valid for an executing Kestra/PostgreSQL workflow:
  - get_workflow_status (READ_ONLY)
  - read_job_logs (READ_ONLY)
  - inspect_job_dependencies (READ_ONLY)
  - check_service_status (READ_ONLY)
  - restart_service (DESTRUCTIVE — restricted to designated test dependency `pgdatabase`)
  - resume_workflow (LOW_RISK — restarts failed Kestra execution with bounded polling)

Mock lockfile tools (`clear_lockfile`, `inspect_lockfiles`) are intentionally NOT
registered in the Kestra tool runner.
"""

from __future__ import annotations

from typing import Optional, Type
from pydantic import BaseModel, Field

from nova.environments.kestra_runner import KestraEnvironment
from nova.environments.mock_workflow_runner import JobStatus
from nova.state import RiskLevel
from nova.tools.base import BaseTool, ToolResult
from nova.tools.runner import ToolRunner


# --- Input Schemas ---

class KestraGetWorkflowStatusInput(BaseModel):
    execution_id: Optional[str] = Field(
        default=None,
        description="Kestra execution ID to inspect (defaults to active incident execution_id).",
    )


class KestraReadJobLogsInput(BaseModel):
    job_id: str = Field(
        description="Exact Kestra task ID to read logs for (e.g. 'extract', 'green_copy_in_to_staging_table')."
    )
    tail_lines: int = Field(default=50, ge=1, le=500, description="Number of log lines to read from the end.")
    execution_id: Optional[str] = Field(default=None, description="Optional Kestra execution ID.")


class KestraInspectDependenciesInput(BaseModel):
    job_id: Optional[str] = Field(
        default=None,
        description="Optional Kestra task ID to inspect dependencies for, or omit for full execution DAG.",
    )
    execution_id: Optional[str] = Field(default=None, description="Optional Kestra execution ID.")


class KestraCheckServiceStatusInput(BaseModel):
    service_name: str = Field(
        description="Registered dependency name to probe: 'pgdatabase' or 'upstream_source'."
    )


class KestraRestartServiceInput(BaseModel):
    service_name: str = Field(
        description="Designated restartable dependency name (currently only 'pgdatabase')."
    )
    allow_idempotent: bool = Field(
        default=True,
        description="If True, returns success when the dependency is already healthy.",
    )


class KestraResumeWorkflowInput(BaseModel):
    execution_id: Optional[str] = Field(
        default=None,
        description="Kestra execution ID to restart and poll (defaults to bound execution_id).",
    )
    workflow_id: Optional[str] = Field(
        default=None,
        description="Optional workflow identifier (e.g. 'zoomcamp/04_postgres_taxi') if execution_id is inferred.",
    )


# --- Concrete Kestra Tools ---

class KestraGetWorkflowStatusTool(BaseTool):
    name = "get_workflow_status"
    description = (
        "Queries Kestra execution status, listing all actual task IDs, task states, "
        "attempt counts, and error messages for the active execution."
    )
    risk_level = RiskLevel.READ_ONLY
    args_schema: Type[BaseModel] = KestraGetWorkflowStatusInput

    def __init__(self, env: KestraEnvironment):
        self.env = env

    def _run(self, execution_id: Optional[str] = None) -> ToolResult:
        try:
            eid = execution_id or self.env.default_execution_id
            state = self.env.get_state(execution_id=eid)
            if state is None:
                return ToolResult(
                    stdout="No active Kestra execution ID bound.",
                    stderr="",
                    exit_code=0,
                    success=True,
                    data={"status": "NO_STATE"},
                )

            summary_lines = [
                f"Workflow ID: {state.workflow_id}",
                f"Execution ID: {state.run_id}",
                f"Execution State: {state.execution_state}",
                f"Inputs: {state.inputs}",
                f"Total Task Attempts: {state.total_attempts}",
                "Task States:",
            ]
            failed_jobs = []
            for tid, task in state.jobs.items():
                err = f" | Error: {task.error_message}" if task.error_message else ""
                summary_lines.append(
                    f"  - {tid}: status={task.status.value} (kestra_state={task.raw_state}, attempts={task.attempt_count}){err}"
                )
                if task.status == JobStatus.FAILED:
                    failed_jobs.append(tid)

            return ToolResult(
                stdout="\n".join(summary_lines),
                stderr="",
                exit_code=0,
                success=True,
                data={
                    "workflow_id": state.workflow_id,
                    "execution_id": state.run_id,
                    "execution_state": state.execution_state,
                    "failed_jobs": failed_jobs,
                    "total_attempts": state.total_attempts,
                },
            )
        except Exception as exc:
            return ToolResult(
                stdout="",
                stderr=f"Failed to query Kestra execution status: {exc}",
                exit_code=1,
                success=False,
            )


class KestraReadJobLogsTool(BaseTool):
    name = "read_job_logs"
    description = "Reads recent log entries for a specific Kestra task ID in the active execution."
    risk_level = RiskLevel.READ_ONLY
    args_schema: Type[BaseModel] = KestraReadJobLogsInput

    def __init__(self, env: KestraEnvironment):
        self.env = env

    def _run(
        self,
        job_id: str,
        tail_lines: int = 50,
        execution_id: Optional[str] = None,
    ) -> ToolResult:
        try:
            content = self.env.read_job_log(
                job_id=job_id,
                tail_lines=tail_lines,
                execution_id=execution_id,
            )
            if content.startswith("Error:"):
                return ToolResult(stdout="", stderr=content, exit_code=1, success=False)
            return ToolResult(
                stdout=content,
                stderr="",
                exit_code=0,
                success=True,
                data={"job_id": job_id, "lines_read": len(content.splitlines())},
            )
        except Exception as exc:
            return ToolResult(
                stdout="",
                stderr=f"Failed to read Kestra logs for '{job_id}': {exc}",
                exit_code=1,
                success=False,
            )


class KestraInspectJobDependenciesTool(BaseTool):
    name = "inspect_job_dependencies"
    description = "Inspects the Kestra execution DAG to discover upstream and downstream task dependencies."
    risk_level = RiskLevel.READ_ONLY
    args_schema: Type[BaseModel] = KestraInspectDependenciesInput

    def __init__(self, env: KestraEnvironment):
        self.env = env

    def _run(
        self,
        job_id: Optional[str] = None,
        execution_id: Optional[str] = None,
    ) -> ToolResult:
        try:
            info = self.env.get_dependencies(job_id=job_id, execution_id=execution_id)
            if job_id:
                return ToolResult(
                    stdout=(
                        f"Task '{job_id}' upstream dependencies: {info['dependencies']}. "
                        f"Downstream dependents: {info['downstream']}."
                    ),
                    stderr="",
                    exit_code=0,
                    success=True,
                    data=info,
                )
            lines = [f"Kestra Execution DAG ({info['execution_id']}):"]
            for tid, deps in info["dag"].items():
                lines.append(f"  - {tid} depends on: {deps}")
            return ToolResult(
                stdout="\n".join(lines),
                stderr="",
                exit_code=0,
                success=True,
                data=info,
            )
        except Exception as exc:
            return ToolResult(
                stdout="",
                stderr=f"Failed to inspect Kestra dependencies: {exc}",
                exit_code=1,
                success=False,
            )


class KestraCheckServiceStatusTool(BaseTool):
    name = "check_service_status"
    description = "Probes health and reachability of a registered external dependency ('pgdatabase' or 'upstream_source')."
    risk_level = RiskLevel.READ_ONLY
    args_schema: Type[BaseModel] = KestraCheckServiceStatusInput

    def __init__(self, env: KestraEnvironment):
        self.env = env

    def _run(self, service_name: str) -> ToolResult:
        if not self.env.is_registered_service(service_name):
            return ToolResult(
                stdout="",
                stderr=f"Error: Service '{service_name}' is not a registered Kestra dependency.",
                exit_code=1,
                success=False,
                data={"service": service_name, "registered": False},
            )
        info = self.env.check_service_status(service_name)
        is_healthy = info.get("healthy", False)
        status_label = info.get("status", "UNKNOWN")
        detail = info.get("detail", "")
        return ToolResult(
            stdout=f"Service '{service_name}' status: {status_label} (Healthy: {is_healthy}). Detail: {detail}",
            stderr="",
            exit_code=0,
            success=True,
            data=info,
        )


class KestraRestartServiceTool(BaseTool):
    name = "restart_service"
    description = "Restarts the designated warehouse database dependency ('pgdatabase')."
    risk_level = RiskLevel.DESTRUCTIVE
    args_schema: Type[BaseModel] = KestraRestartServiceInput

    def __init__(self, env: KestraEnvironment):
        self.env = env

    def _run(self, service_name: str, allow_idempotent: bool = True) -> ToolResult:
        if not self.env.is_registered_service(service_name):
            return ToolResult(
                stdout="",
                stderr=f"Error: Service '{service_name}' is not a registered Kestra dependency.",
                exit_code=1,
                success=False,
                data={"service_name": service_name, "restarted": False, "unknown_service": True},
            )
        if not self.env.is_restartable_service(service_name):
            return ToolResult(
                stdout="",
                stderr=(
                    f"Policy Guard: Service '{service_name}' is an external dependency and cannot be "
                    "restarted by NOVA. Only designated dependency 'pgdatabase' is restartable."
                ),
                exit_code=1,
                success=False,
                data={"service_name": service_name, "restarted": False, "non_restartable": True},
            )

        status_info = self.env.check_service_status(service_name)
        if status_info.get("healthy") and allow_idempotent:
            return ToolResult(
                stdout=f"Service '{service_name}' is already healthy. Idempotent check satisfied.",
                stderr="",
                exit_code=0,
                success=True,
                data={"service_name": service_name, "restarted": False, "idempotent_noop": True},
            )

        ok, detail = self.env.restart_service(service_name)
        if ok:
            return ToolResult(
                stdout=detail,
                stderr="",
                exit_code=0,
                success=True,
                data={"service_name": service_name, "restarted": True, "idempotent_noop": False},
            )
        return ToolResult(
            stdout="",
            stderr=detail,
            exit_code=1,
            success=False,
            data={"service_name": service_name, "restarted": False},
        )


class KestraResumeWorkflowTool(BaseTool):
    name = "resume_workflow"
    description = (
        "Restarts a failed Kestra execution by its execution_id using Kestra's execution restart API "
        "and polls boundedly until completion."
    )
    risk_level = RiskLevel.LOW_RISK
    args_schema: Type[BaseModel] = KestraResumeWorkflowInput

    def __init__(self, env: KestraEnvironment):
        self.env = env

    def _run(
        self,
        execution_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
    ) -> ToolResult:
        try:
            eid = execution_id or self.env.default_execution_id
            if not eid:
                return ToolResult(
                    stdout="",
                    stderr="execution_id is required to restart a Kestra execution.",
                    exit_code=1,
                    success=False,
                )
            if self.env.default_execution_id and eid != self.env.default_execution_id:
                return ToolResult(
                    stdout="",
                    stderr=(
                        f"Execution ID mismatch: requested '{eid}' but incident is bound to "
                        f"'{self.env.default_execution_id}'."
                    ),
                    exit_code=1,
                    success=False,
                )

            current = self.env.get_state(execution_id=eid)
            if current is None:
                return ToolResult(
                    stdout="",
                    stderr=f"Kestra execution '{eid}' not found.",
                    exit_code=1,
                    success=False,
                )
            if workflow_id and workflow_id not in (current.workflow_id, current.flow_id):
                return ToolResult(
                    stdout="",
                    stderr=(
                        f"Workflow ID mismatch: requested '{workflow_id}', active execution belongs to "
                        f"'{current.workflow_id}'."
                    ),
                    exit_code=1,
                    success=False,
                )
            if current.execution_state == "SUCCESS":
                return ToolResult(
                    stdout="",
                    stderr=(
                        f"Kestra execution '{eid}' is already in SUCCESS state and cannot be restarted as a failed run."
                    ),
                    exit_code=1,
                    success=False,
                )

            before_task_attempts = {tid: t.attempt_count for tid, t in current.jobs.items()}
            final_state = self.env.restart_execution(eid)
            failed_jobs = [tid for tid, t in final_state.jobs.items() if t.status == JobStatus.FAILED]
            freshly_succeeded_tasks = [
                tid
                for tid, t in final_state.jobs.items()
                if t.status == JobStatus.SUCCESS
                and (t.last_attempt_state in (None, "SUCCESS"))
                and t.attempt_count > before_task_attempts.get(tid, 0)
            ]
            task_attempts = {tid: t.attempt_count for tid, t in final_state.jobs.items()}

            if final_state.execution_state == "SUCCESS" and not failed_jobs:
                return ToolResult(
                    stdout=(
                        f"Kestra execution '{eid}' ({final_state.workflow_id}) restarted and completed "
                        f"with state SUCCESS (total_attempts={final_state.total_attempts}, "
                        f"freshly_succeeded={freshly_succeeded_tasks})."
                    ),
                    stderr="",
                    exit_code=0,
                    success=True,
                    data={
                        "execution_id": eid,
                        "workflow_id": final_state.workflow_id,
                        "execution_state": final_state.execution_state,
                        "total_attempts": final_state.total_attempts,
                        "task_attempts": task_attempts,
                        "freshly_succeeded_tasks": freshly_succeeded_tasks,
                        "failed_jobs": [],
                    },
                )

            return ToolResult(
                stdout=f"Kestra execution '{eid}' restarted, but ended in state {final_state.execution_state} on tasks {failed_jobs}",
                stderr=f"Kestra execution '{eid}' failed after restart on task(s): {failed_jobs}",
                exit_code=1,
                success=False,
                data={
                    "execution_id": eid,
                    "workflow_id": final_state.workflow_id,
                    "execution_state": final_state.execution_state,
                    "total_attempts": final_state.total_attempts,
                    "task_attempts": task_attempts,
                    "freshly_succeeded_tasks": freshly_succeeded_tasks,
                    "failed_jobs": failed_jobs,
                },
            )
        except Exception as exc:
            return ToolResult(
                stdout="",
                stderr=f"Failed to restart Kestra execution: {exc}",
                exit_code=2,
                success=False,
            )


def create_kestra_tool_runner(env: KestraEnvironment) -> ToolRunner:
    """
    Factory creating a ToolRunner configured exclusively for Kestra/PostgreSQL workflows.
    Omits MockWorkflowRunner lockfile tools (`clear_lockfile`, `inspect_lockfiles`).
    """
    runner = ToolRunner()
    runner.register(KestraGetWorkflowStatusTool(env))
    runner.register(KestraReadJobLogsTool(env))
    runner.register(KestraInspectJobDependenciesTool(env))
    runner.register(KestraCheckServiceStatusTool(env))
    runner.register(KestraRestartServiceTool(env))
    runner.register(KestraResumeWorkflowTool(env))
    return runner
