import json
import os
import shutil
import tempfile
import threading
import time
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class ScenarioType(str, Enum):
    HAPPY_PATH = "HAPPY_PATH"
    STALE_LOCKFILE = "STALE_LOCKFILE"
    SERVICE_OFFLINE = "SERVICE_OFFLINE"
    DATA_CORRUPTION = "DATA_CORRUPTION"
    CASCADING_FAILURE = "CASCADING_FAILURE"


class JobDefinition(BaseModel):
    id: str
    name: str
    description: str
    dependencies: List[str] = Field(default_factory=list)
    status: JobStatus = JobStatus.PENDING
    exit_code: Optional[int] = None
    error_message: Optional[str] = None


class WorkflowState(BaseModel):
    workflow_id: str
    name: str
    scenario: ScenarioType
    jobs: Dict[str, JobDefinition]
    services: Dict[str, bool] = Field(default_factory=dict)  # service_name -> is_running
    active_locks: List[str] = Field(default_factory=list)


class MockWorkflowRunner:
    """
    Simulates an enterprise batch workload execution engine (analogous to Control-M jobs).
    Creates actual log files and state artifacts on disk so tools can inspect real files.
    """

    DEFAULT_JOBS = [
        JobDefinition(
            id="JOB_01_EXTRACT",
            name="Extract Source Data",
            description="Extracts nightly transactional batches from upstream feeds",
            dependencies=[],
        ),
        JobDefinition(
            id="JOB_02_TRANSFORM",
            name="Transform & Aggregate Metrics",
            description="Processes raw feeds into aggregated reporting tables",
            dependencies=["JOB_01_EXTRACT"],
        ),
        JobDefinition(
            id="JOB_03_LOAD",
            name="Load into Data Warehouse",
            description="Streams transformed records into the reporting database",
            dependencies=["JOB_02_TRANSFORM"],
        ),
        JobDefinition(
            id="JOB_04_REPORT",
            name="Generate Executive Summary",
            description="Computes executive KPIs and publishes metrics",
            dependencies=["JOB_03_LOAD"],
        ),
    ]

    def __init__(self, base_dir: Path | str = "./mock_state"):
        self.base_dir = Path(base_dir)
        self.logs_dir = self.base_dir / "logs"
        self.locks_dir = self.base_dir / "locks"
        self.state_file = self.base_dir / "workflow_state.json"
        self._lock = threading.RLock()
        self._initialize_environment()

    def _initialize_environment(self) -> None:
        """Create clean directory layout for mock operational state."""
        with self._lock:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            self.locks_dir.mkdir(parents=True, exist_ok=True)

    def reset(self) -> None:
        """Completely resets all files, logs, and state."""
        with self._lock:
            if self.base_dir.exists():
                shutil.rmtree(self.base_dir)
            self._initialize_environment()

    def get_state(self) -> Optional[WorkflowState]:
        """Loads current workflow state from disk."""
        with self._lock:
            if not self.state_file.exists():
                return None
            with open(self.state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return WorkflowState.model_validate(data)

    def _save_state(self, state: WorkflowState) -> None:
        """Atomically persists workflow state to disk."""
        with self._lock:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            payload = state.model_dump_json(indent=2)
            fd, tmp_path = tempfile.mkstemp(
                dir=str(self.base_dir), prefix=".workflow_state_", suffix=".tmp"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(payload)
                os.replace(tmp_path, self.state_file)
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise

    def _write_log(self, job_id: str, message: str) -> None:
        """Appends log entry to the job's log file."""
        with self._lock:
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            log_file = self.logs_dir / f"{job_id}.log"
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] {message}\n")

    def read_job_log(self, job_id: str, tail_lines: int = 50) -> str:
        """Reads the latest lines from a job's log file."""
        with self._lock:
            log_file = self.logs_dir / f"{job_id}.log"
            if not log_file.exists():
                return f"Error: No log file found for job '{job_id}'."
            with open(log_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                return "".join(lines[-tail_lines:])

    def inject_scenario(self, scenario: ScenarioType) -> WorkflowState:
        """Prepares a workflow run configured with a specific operational scenario."""
        with self._lock:
            self.reset()
            jobs_dict = {job.id: job.model_copy() for job in self.DEFAULT_JOBS}
            services = {"reporting_warehouse_service": True}
            active_locks: List[str] = []

            if scenario == ScenarioType.STALE_LOCKFILE:
                # Create a stale lock file that prevents transform job from starting
                lock_path = self.locks_dir / "transform_job.lock"
                with open(lock_path, "w", encoding="utf-8") as f:
                    f.write("PID=9182 STATUS=ORPHANED_PROCESS CREATED_AT=PREVIOUS_RUN\n")
                active_locks.append("transform_job.lock")

            elif scenario == ScenarioType.SERVICE_OFFLINE:
                # Reporting database service is offline
                services["reporting_warehouse_service"] = False

            elif scenario == ScenarioType.CASCADING_FAILURE:
                # Compound failure: Stale lock on JOB_02 AND offline database on JOB_03
                lock_path = self.locks_dir / "transform_job.lock"
                with open(lock_path, "w", encoding="utf-8") as f:
                    f.write("PID=9182 STATUS=ORPHANED_PROCESS CREATED_AT=PREVIOUS_RUN\n")
                active_locks.append("transform_job.lock")
                services["reporting_warehouse_service"] = False

            state = WorkflowState(
                workflow_id="WF_NIGHTLY_BATCH_1001",
                name="Nightly Financial Ingestion Pipeline",
                scenario=scenario,
                jobs=jobs_dict,
                services=services,
                active_locks=active_locks,
            )
            self._save_state(state)
            return state

    def run_workflow(self) -> WorkflowState:
        """
        Executes jobs in dependency sequence. Halts if any job fails.
        """
        with self._lock:
            return self._run_workflow_locked()

    def _run_workflow_locked(self) -> WorkflowState:
        state = self.get_state()
        if not state:
            raise ValueError("No active workflow state. Call inject_scenario first.")

        for job_id, job in state.jobs.items():
            if job.status == JobStatus.SUCCESS:
                continue

            # Verify dependencies
            deps_met = all(
                state.jobs[dep].status == JobStatus.SUCCESS for dep in job.dependencies
            )
            if not deps_met:
                self._write_log(
                    job_id,
                    f"BLOCKED: Dependencies {[dep for dep in job.dependencies if state.jobs[dep].status != JobStatus.SUCCESS]} not completed.",
                )
                job.status = JobStatus.FAILED
                job.exit_code = 2
                job.error_message = "Dependency check failed."
                break

            # Execute Job 01: Extract
            if job_id == "JOB_01_EXTRACT":
                self._write_log(job_id, "INFO: Starting extraction from upstream feeds...")
                if state.scenario == ScenarioType.DATA_CORRUPTION:
                    self._write_log(
                        job_id,
                        "CRITICAL_ERROR: Input payload checksum mismatch. Zero-byte records found. Cannot proceed.",
                    )
                    job.status = JobStatus.FAILED
                    job.exit_code = 13
                    job.error_message = "Payload checksum mismatch: corrupt source data."
                    break
                else:
                    self._write_log(job_id, "INFO: Extracted 45,210 transactional records.")
                    self._write_log(job_id, "SUCCESS: Extraction completed cleanly.")
                    job.status = JobStatus.SUCCESS
                    job.exit_code = 0

            # Execute Job 02: Transform
            elif job_id == "JOB_02_TRANSFORM":
                self._write_log(job_id, "INFO: Checking concurrency locks before transform...")
                lock_file = self.locks_dir / "transform_job.lock"
                if lock_file.exists():
                    self._write_log(
                        job_id,
                        "FATAL: Lockfile 'transform_job.lock' is present. Another process is active or crashed. Aborting.",
                    )
                    job.status = JobStatus.FAILED
                    job.exit_code = 1
                    job.error_message = "Stale lockfile 'transform_job.lock' detected."
                    break
                else:
                    self._write_log(job_id, "INFO: Lock check clear. Aggregating transaction metrics...")
                    self._write_log(job_id, "SUCCESS: Transformed 45,210 records into 12 summary partitions.")
                    job.status = JobStatus.SUCCESS
                    job.exit_code = 0

            # Execute Job 03: Load
            elif job_id == "JOB_03_LOAD":
                self._write_log(job_id, "INFO: Connecting to target service 'reporting_warehouse_service'...")
                if not state.services.get("reporting_warehouse_service", False):
                    self._write_log(
                        job_id,
                        "FATAL: Connection to 'reporting_warehouse_service' refused on port 5432. Service is OFFLINE.",
                    )
                    job.status = JobStatus.FAILED
                    job.exit_code = 111
                    job.error_message = "Connection refused: reporting_warehouse_service is OFFLINE."
                    break
                else:
                    self._write_log(job_id, "INFO: Connected to warehouse. Ingesting summary partitions...")
                    self._write_log(job_id, "SUCCESS: 12 partitions successfully committed to warehouse tables.")
                    job.status = JobStatus.SUCCESS
                    job.exit_code = 0

            # Execute Job 04: Report
            elif job_id == "JOB_04_REPORT":
                self._write_log(job_id, "INFO: Computing executive KPIs from data warehouse...")
                self._write_log(job_id, "SUCCESS: Executive scorecard published to notification stream.")
                job.status = JobStatus.SUCCESS
                job.exit_code = 0

        self._save_state(state)
        return state

    # --- Remediation Actions Callable by NOVA's Tools ---

    def is_lock_active(self, lock_name: str) -> bool:
        """Checks whether a lockfile on disk belongs to an active live worker."""
        with self._lock:
            lock_path = self.locks_dir / lock_name
            if not lock_path.exists():
                return False
            content = lock_path.read_text(encoding="utf-8", errors="ignore").upper()
            if "ORPHANED" in content or "STALE" in content:
                return False
            return "STATUS=ACTIVE_WORKER" in content or "HEARTBEAT=RECENT" in content

    def clear_lock(self, lock_name: str) -> bool:
        """Removes a specified lockfile from the locks directory."""
        with self._lock:
            lock_path = self.locks_dir / lock_name
            if lock_path.exists():
                lock_path.unlink()
                state = self.get_state()
                if state and lock_name in state.active_locks:
                    state.active_locks.remove(lock_name)
                    self._save_state(state)
                return True
            return False

    def restart_service(self, service_name: str) -> bool:
        """Simulates restarting or bringing an offline service back online."""
        with self._lock:
            state = self.get_state()
            if not state:
                return False
            state.services[service_name] = True
            self._save_state(state)
            return True

    def check_service_status(self, service_name: str) -> dict:
        """Returns the operational health status of a service."""
        with self._lock:
            state = self.get_state()
            if not state:
                return {"service": service_name, "status": "UNKNOWN"}
            is_running = state.services.get(service_name, False)
            return {
                "service": service_name,
                "status": "RUNNING" if is_running else "STOPPED",
                "port": 5432 if "warehouse" in service_name else 8080,
                "healthy": is_running,
            }

