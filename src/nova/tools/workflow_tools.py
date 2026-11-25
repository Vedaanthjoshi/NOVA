from typing import Optional, Type
from pydantic import BaseModel, Field

from nova.environments.mock_workflow_runner import JobStatus, MockWorkflowRunner
from nova.state import RiskLevel
from nova.tools.base import BaseTool, ToolResult


# --- Input Schemas ---

class ReadJobLogsInput(BaseModel):
    job_id: str = Field(description="The unique identifier of the batch job, e.g. 'JOB_02_TRANSFORM'")
    tail_lines: int = Field(default=50, ge=1, le=500, description="Number of log lines to read from the end")


class CheckServiceStatusInput(BaseModel):
    service_name: str = Field(description="Name of the service to inspect, e.g. 'reporting_warehouse_service'")


class GetWorkflowStatusInput(BaseModel):
    workflow_id: Optional[str] = Field(default=None, description="Optional workflow ID to query status for")


class InspectJobDependenciesInput(BaseModel):
    job_id: Optional[str] = Field(default=None, description="Optional job ID to inspect dependencies for, or omit for all")


class InspectLockfilesInput(BaseModel):
    lock_name: Optional[str] = Field(default=None, description="Optional lockfile name to inspect, or omit to list all")


class ClearLockfileInput(BaseModel):
    lock_name: str = Field(description="Filename of the lock to remove, e.g. 'transform_job.lock'")
    allow_idempotent: bool = Field(
        default=True,
        description="If True, returns success if lockfile is already absent (idempotency contract)"
    )


class RestartServiceInput(BaseModel):
    service_name: str = Field(description="Name of the service to restart, e.g. 'reporting_warehouse_service'")
    allow_idempotent: bool = Field(
        default=True,
        description="If True, returns success if service is already healthy and operational"
    )


class ResumeWorkflowInput(BaseModel):
    workflow_id: str = Field(description="Workflow ID to resume execution, e.g. 'WF_NIGHTLY_BATCH_1001'")


# --- Concrete Tool Implementations ---

class ReadJobLogsTool(BaseTool):
    name = "read_job_logs"
    description = "Reads recent log entries for a specified batch job to identify errors and stack traces."
    risk_level = RiskLevel.READ_ONLY
    args_schema: Type[BaseModel] = ReadJobLogsInput

    def __init__(self, env: MockWorkflowRunner):
        self.env = env

    def _run(self, job_id: str, tail_lines: int = 50) -> ToolResult:
        log_content = self.env.read_job_log(job_id=job_id, tail_lines=tail_lines)
        if log_content.startswith("Error:"):
            return ToolResult(
                stdout="",
                stderr=log_content,
                exit_code=1,
                success=False,
            )
        return ToolResult(
            stdout=log_content,
            stderr="",
            exit_code=0,
            success=True,
            data={"job_id": job_id, "lines_read": len(log_content.splitlines())},
        )


class CheckServiceStatusTool(BaseTool):
    name = "check_service_status"
    description = "Checks the operational status and port connectivity of a target service."
    risk_level = RiskLevel.READ_ONLY
    args_schema: Type[BaseModel] = CheckServiceStatusInput

    def __init__(self, env: MockWorkflowRunner):
        self.env = env

    def _run(self, service_name: str) -> ToolResult:
        status_info = self.env.check_service_status(service_name)
        is_healthy = status_info.get("healthy", False)
        status_label = status_info.get("status", "UNKNOWN")
        # Diagnostic probes succeed (exit_code=0, success=True) when they accurately
        # observe resource health, even if the inspected service is STOPPED/unhealthy.
        return ToolResult(
            stdout=f"Service '{service_name}' status: {status_label} (Healthy: {is_healthy})",
            stderr="",
            exit_code=0,
            success=True,
            data=status_info,
        )


class ClearLockfileTool(BaseTool):
    name = "clear_lockfile"
    description = "Removes a stale or orphaned lockfile preventing a batch job from starting."
    risk_level = RiskLevel.DESTRUCTIVE
    args_schema: Type[BaseModel] = ClearLockfileInput

    def __init__(self, env: MockWorkflowRunner):
        self.env = env

    def _run(self, lock_name: str, allow_idempotent: bool = True) -> ToolResult:
        lock_path = self.env.locks_dir / lock_name
        already_absent = not lock_path.exists()
        if not already_absent and self.env.is_lock_active(lock_name):
            return ToolResult(
                stdout="",
                stderr=(
                    f"Safety Guard: Refusing to delete active lockfile '{lock_name}' "
                    "owned by an active worker process."
                ),
                exit_code=1,
                success=False,
                data={
                    "lock_name": lock_name,
                    "removed": False,
                    "active_lock_protected": True,
                },
            )
        removed = self.env.clear_lock(lock_name)
        if removed:
            return ToolResult(
                stdout=f"Successfully removed lockfile: '{lock_name}'",
                stderr="",
                exit_code=0,
                success=True,
                data={"lock_name": lock_name, "removed": True, "idempotent_noop": False},
            )
        if already_absent and allow_idempotent:
            return ToolResult(
                stdout=f"Lockfile '{lock_name}' is already absent. Idempotent pre-condition satisfied.",
                stderr="",
                exit_code=0,
                success=True,
                data={"lock_name": lock_name, "removed": False, "idempotent_noop": True},
            )
        return ToolResult(
            stdout="",
            stderr=f"Lockfile '{lock_name}' was not found or could not be removed",
            exit_code=1,
            success=False,
            data={"lock_name": lock_name, "removed": False, "idempotent_noop": False},
        )


class RestartServiceTool(BaseTool):
    name = "restart_service"
    description = "Restarts a stopped or failing enterprise service."
    risk_level = RiskLevel.DESTRUCTIVE
    args_schema: Type[BaseModel] = RestartServiceInput

    def __init__(self, env: MockWorkflowRunner):
        self.env = env

    def _run(self, service_name: str, allow_idempotent: bool = True) -> ToolResult:
        status_info = self.env.check_service_status(service_name)
        is_already_healthy = status_info.get("healthy", False)

        if is_already_healthy and allow_idempotent:
            return ToolResult(
                stdout=f"Service '{service_name}' is already healthy and accepting traffic. Idempotent check satisfied.",
                stderr="",
                exit_code=0,
                success=True,
                data={"service_name": service_name, "restarted": False, "idempotent_noop": True},
            )

        restarted = self.env.restart_service(service_name)
        if restarted:
            return ToolResult(
                stdout=f"Service '{service_name}' restarted successfully.",
                stderr="",
                exit_code=0,
                success=True,
                data={"service_name": service_name, "restarted": True, "idempotent_noop": False},
            )
        return ToolResult(
            stdout="",
            stderr=f"Failed to restart service '{service_name}'. Service not found in environment.",
            exit_code=1,
            success=False,
            data={"service_name": service_name, "restarted": False, "idempotent_noop": False},
        )


class ResumeWorkflowTool(BaseTool):
    name = "resume_workflow"
    description = "Resumes batch workflow execution from the last failed job."
    risk_level = RiskLevel.LOW_RISK
    args_schema: Type[BaseModel] = ResumeWorkflowInput

    def __init__(self, env: MockWorkflowRunner):
        self.env = env

    def _run(self, workflow_id: str) -> ToolResult:
        try:
            state = self.env.run_workflow()
            all_successful = all(job.status == "SUCCESS" for job in state.jobs.values())
            failed_jobs = [job.id for job in state.jobs.values() if job.status == "FAILED"]

            if all_successful:
                return ToolResult(
                    stdout=f"Workflow '{workflow_id}' completed successfully with all jobs passing.",
                    stderr="",
                    exit_code=0,
                    success=True,
                    data={"workflow_id": workflow_id, "failed_jobs": []},
                )
            else:
                return ToolResult(
                    stdout=f"Workflow resumed, but halted on job(s): {failed_jobs}",
                    stderr=f"Unresolved failure in jobs: {failed_jobs}",
                    exit_code=1,
                    success=False,
                    data={"workflow_id": workflow_id, "failed_jobs": failed_jobs},
                )
        except Exception as e:
            return ToolResult(
                stdout="",
                stderr=f"Failed to resume workflow: {str(e)}",
                exit_code=2,
                success=False,
            )


class GetWorkflowStatusTool(BaseTool):
    name = "get_workflow_status"
    description = "Discovers high-level status of the batch pipeline, listing all jobs, execution states, and failed steps."
    risk_level = RiskLevel.READ_ONLY
    args_schema: Type[BaseModel] = GetWorkflowStatusInput

    def __init__(self, env: MockWorkflowRunner):
        self.env = env

    def _run(self, workflow_id: Optional[str] = None) -> ToolResult:
        state = self.env.get_state()
        if not state:
            return ToolResult(
                stdout="No active workflow state found on disk.",
                stderr="",
                exit_code=0,
                success=True,
                data={"status": "NO_STATE"},
            )

        summary_lines = [
            f"Workflow ID: {state.workflow_id} ({state.name})",
            f"Scenario: {state.scenario.value}",
            "Job States:",
        ]
        failed_jobs = []
        for jid, job in state.jobs.items():
            err = f" | Error: {job.error_message}" if job.error_message else ""
            summary_lines.append(f"  - {jid}: status={job.status.value}, exit_code={job.exit_code}{err}")
            if job.status == JobStatus.FAILED:
                failed_jobs.append(jid)

        active_locks = state.active_locks
        if active_locks:
            summary_lines.append(f"Active Locks: {active_locks}")
        services = state.services
        if services:
            summary_lines.append(f"Services: {services}")

        return ToolResult(
            stdout="\n".join(summary_lines),
            stderr="",
            exit_code=0,
            success=True,
            data={
                "workflow_id": state.workflow_id,
                "scenario": state.scenario.value,
                "failed_jobs": failed_jobs,
                "active_locks": active_locks,
                "services": services,
            },
        )


class InspectJobDependenciesTool(BaseTool):
    name = "inspect_job_dependencies"
    description = "Inspects batch job dependency graph to determine upstream/downstream relationships."
    risk_level = RiskLevel.READ_ONLY
    args_schema: Type[BaseModel] = InspectJobDependenciesInput

    def __init__(self, env: MockWorkflowRunner):
        self.env = env

    def _run(self, job_id: Optional[str] = None) -> ToolResult:
        state = self.env.get_state()
        jobs = state.jobs if state else {j.id: j for j in self.env.DEFAULT_JOBS}

        if job_id:
            if job_id not in jobs:
                return ToolResult(
                    stdout="",
                    stderr=f"Job '{job_id}' not found in workflow definition.",
                    exit_code=1,
                    success=False,
                )
            job = jobs[job_id]
            deps = job.dependencies
            downstream = [jid for jid, j in jobs.items() if job_id in j.dependencies]
            return ToolResult(
                stdout=f"Job '{job_id}' dependencies: {deps}. Downstream dependents: {downstream}.",
                stderr="",
                exit_code=0,
                success=True,
                data={"job_id": job_id, "dependencies": deps, "downstream": downstream},
            )

        dag = {jid: j.dependencies for jid, j in jobs.items()}
        lines = ["Workflow Dependency DAG:"]
        for jid, deps in dag.items():
            lines.append(f"  - {jid} depends on: {deps}")
        return ToolResult(
            stdout="\n".join(lines),
            stderr="",
            exit_code=0,
            success=True,
            data={"dag": dag},
        )


class InspectLockfilesTool(BaseTool):
    name = "inspect_lockfiles"
    description = "Inspects the locks directory on disk to discover active or orphaned lockfiles and their contents."
    risk_level = RiskLevel.READ_ONLY
    args_schema: Type[BaseModel] = InspectLockfilesInput

    def __init__(self, env: MockWorkflowRunner):
        self.env = env

    def _run(self, lock_name: Optional[str] = None) -> ToolResult:
        locks_dir = self.env.locks_dir
        if not locks_dir.exists():
            return ToolResult(
                stdout="Locks directory does not exist or is empty.",
                stderr="",
                exit_code=0,
                success=True,
                data={"locks": []},
            )

        if lock_name:
            target = locks_dir / lock_name
            if target.exists():
                content = target.read_text(encoding="utf-8").strip()
                return ToolResult(
                    stdout=f"Lockfile '{lock_name}' is PRESENT. Content: '{content}'",
                    stderr="",
                    exit_code=0,
                    success=True,
                    data={"lock_name": lock_name, "present": True, "content": content},
                )
            return ToolResult(
                stdout=f"Lockfile '{lock_name}' is NOT PRESENT.",
                stderr="",
                exit_code=0,
                success=True,
                data={"lock_name": lock_name, "present": False},
            )

        lock_files = list(locks_dir.glob("*.lock"))
        if not lock_files:
            return ToolResult(
                stdout="No active lockfiles found in locks directory.",
                stderr="",
                exit_code=0,
                success=True,
                data={"locks": []},
            )

        details = []
        for lf in lock_files:
            content = lf.read_text(encoding="utf-8").strip()
            details.append(f"  - {lf.name}: {content}")

        return ToolResult(
            stdout="Active lockfiles found:\n" + "\n".join(details),
            stderr="",
            exit_code=0,
            success=True,
            data={"locks": [lf.name for lf in lock_files]},
        )


