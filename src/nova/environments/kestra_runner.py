"""
Kestra & PostgreSQL Execution Environment Adapter for NOVA (Milestone 12).

Connects NOVA to a real executing Kestra + PostgreSQL instance:
- Queries Kestra executions, task runs, logs, and execution DAGs via Kestra's REST API.
- Restarts failed executions explicitly (`POST /api/v1/{tenant}/executions/{id}/restart`)
  with bounded polling and task-attempt tracking.
- Checks designated service dependencies (`pgdatabase` via PostgreSQL `SELECT 1`,
  `upstream_source` via HTTP probe) and permits `restart_service` ONLY for
  the designated test dependency (`pgdatabase`).
- Performs independent read-only PostgreSQL output verification (row count, distinct
  unique_row_id count, and total_amount aggregate) on a dedicated read-only transaction.
- Exposes zero scenario labels, zero fault-injection metadata, and zero lockfile tools.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx
from pydantic import BaseModel, Field

from nova.environments.mock_workflow_runner import JobStatus

# Only `pgdatabase` is an agent-owned restartable dependency.
# `upstream_source` is inspectable via check_service_status but NOT restartable by NOVA.
REGISTERED_KESTRA_SERVICES = frozenset({"pgdatabase", "upstream_source"})
RESTARTABLE_KESTRA_SERVICES = frozenset({"pgdatabase"})

DEFAULT_DOCKER_EXE_CANDIDATES = (
    shutil.which("docker"),
    r"C:\Users\vedaa\AppData\Local\Programs\DockerDesktop\resources\bin\docker.exe",
    r"C:\Program Files\Docker\Docker\resources\bin\docker.exe",
)


def _find_docker_bin() -> Optional[str]:
    for candidate in DEFAULT_DOCKER_EXE_CANDIDATES:
        if candidate and os.path.exists(candidate):
            return candidate
    return None


class KestraTaskState(BaseModel):
    id: str
    name: str
    status: JobStatus
    raw_state: str
    attempt_count: int = 0
    last_attempt_state: Optional[str] = None
    is_container: bool = False
    task_run_id: Optional[str] = None
    exit_code: Optional[int] = None
    error_message: Optional[str] = None
    dependencies: List[str] = Field(default_factory=list)
    outputs: Dict[str, Any] = Field(default_factory=dict)


class KestraWorkflowState(BaseModel):
    workflow_id: str
    run_id: str
    namespace: str
    flow_id: str
    execution_state: str
    jobs: Dict[str, KestraTaskState] = Field(default_factory=dict)
    services: Dict[str, bool] = Field(default_factory=dict)
    inputs: Dict[str, Any] = Field(default_factory=dict)
    total_attempts: int = 0


def _map_kestra_state(raw_state: str) -> JobStatus:
    upper = (raw_state or "").upper()
    if upper == "SUCCESS":
        return JobStatus.SUCCESS
    if upper in ("FAILED", "KILLED", "WARNING", "CANCELLED"):
        return JobStatus.FAILED
    if upper in ("RUNNING", "RESTARTED", "PAUSED"):
        return JobStatus.RUNNING
    return JobStatus.PENDING


class KestraEnvironment:
    """
    Production-like adapter connecting NOVA to Kestra's REST API and PostgreSQL.
    All diagnostics and verifications read genuine runtime state and logs.
    """

    def __init__(
        self,
        kestra_url: str = "http://127.0.0.1:8080",
        kestra_user: Optional[str] = None,
        kestra_password: Optional[str] = None,
        tenant: str = "main",
        pg_host: str = "127.0.0.1",
        pg_port: int = 5432,
        pg_dbname: str = "ny_taxi",
        pg_user: str = "root",
        pg_password: str = "root",
        pg_container_name: str = "nova_taxi_pgdatabase",
        source_probe_url: str = "https://github.com/DataTalksClub/nyc-tlc-data/releases",
        default_execution_id: Optional[str] = None,
        http_client: Optional[httpx.Client] = None,
        service_controller: Optional[Callable[[str, str], Tuple[bool, str]]] = None,
        pg_verifier: Optional[Callable[[Dict[str, Any]], Tuple[bool, Dict[str, Any], str]]] = None,
    ) -> None:
        self.kestra_url = kestra_url.rstrip("/")
        self.kestra_user = (
            kestra_user if kestra_user is not None else os.environ.get("KESTRA_USER", "admin@kestra.io")
        )
        self.kestra_password = (
            kestra_password if kestra_password is not None else os.environ.get("KESTRA_PASSWORD", "Admin1234!")
        )
        self.tenant = tenant
        self.pg_host = pg_host
        self.pg_port = pg_port
        self.pg_dbname = pg_dbname
        self.pg_user = pg_user
        self.pg_password = pg_password
        self.pg_container_name = pg_container_name
        self.source_probe_url = source_probe_url
        self.default_execution_id = default_execution_id
        self._http = http_client
        self._service_controller = service_controller
        self._pg_verifier = pg_verifier

    @property
    def api_base(self) -> str:
        return f"{self.kestra_url}/api/v1/{self.tenant}"

    def _auth(self) -> Optional[Tuple[str, str]]:
        if self.kestra_user and self.kestra_password:
            return (self.kestra_user, self.kestra_password)
        return None

    def _request(self, method: str, path: str, timeout: float = 15.0, **kwargs: Any) -> httpx.Response:
        url = f"{self.api_base}{path}"
        if self._http is not None:
            return self._http.request(method, url, auth=self._auth(), timeout=timeout, **kwargs)
        with httpx.Client(timeout=timeout) as client:
            return client.request(method, url, auth=self._auth(), **kwargs)

    # --- Registered Resource Helpers ---

    def is_registered_service(self, service_name: str) -> bool:
        return service_name in REGISTERED_KESTRA_SERVICES

    def is_restartable_service(self, service_name: str) -> bool:
        return service_name in RESTARTABLE_KESTRA_SERVICES

    def bind_execution(self, execution_id: str) -> None:
        self.default_execution_id = execution_id

    # --- Kestra Execution & Log Queries ---

    def get_execution_raw(self, execution_id: str) -> Dict[str, Any]:
        resp = self._request("GET", f"/executions/{execution_id}")
        resp.raise_for_status()
        return resp.json()

    def get_execution_logs(
        self,
        execution_id: str,
        task_id: Optional[str] = None,
        tail_lines: int = 50,
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {}
        if task_id:
            params["taskId"] = task_id
        resp = self._request("GET", f"/logs/{execution_id}", params=params)
        resp.raise_for_status()
        entries = resp.json()
        if not isinstance(entries, list):
            return []
        return entries[-tail_lines:] if tail_lines > 0 else entries

    def get_execution_graph(self, execution_id: str) -> Dict[str, Any]:
        resp = self._request("GET", f"/executions/{execution_id}/graph")
        resp.raise_for_status()
        return resp.json()

    def is_registered_job(self, job_id: str, execution_id: Optional[str] = None) -> bool:
        eid = execution_id or self.default_execution_id
        if not eid:
            return False
        raw = self.get_execution_raw(eid)
        task_ids = {tr.get("taskId") for tr in raw.get("taskRunList", []) if tr.get("taskId")}
        if job_id in task_ids:
            return True
        # Also check flow tasks in case task didn't run yet
        ns = raw.get("namespace")
        fid = raw.get("flowId")
        if ns and fid:
            try:
                flow = self._request("GET", f"/flows/{ns}/{fid}").json()
                return job_id in self._collect_flow_task_ids(flow.get("tasks", []))
            except Exception:
                pass
        return False

    def _collect_flow_task_ids(self, tasks: List[Dict[str, Any]]) -> set[str]:
        ids: set[str] = set()
        for t in tasks or []:
            if isinstance(t, dict):
                if t.get("id"):
                    ids.add(t["id"])
                for key in ("tasks", "then", "else", "errors", "finally"):
                    sub = t.get(key)
                    if isinstance(sub, list):
                        ids.update(self._collect_flow_task_ids(sub))
        return ids

    def get_dependencies(
        self,
        job_id: Optional[str] = None,
        execution_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        eid = execution_id or self.default_execution_id
        if not eid:
            raise ValueError("No execution_id bound to inspect dependencies.")
        graph = self.get_execution_graph(eid)
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])

        uid_to_task: Dict[str, str] = {}
        for node in nodes:
            uid = node.get("uid")
            task = node.get("task") or {}
            tid = task.get("id")
            if uid and tid:
                uid_to_task[uid] = tid

        dag: Dict[str, List[str]] = {tid: [] for tid in uid_to_task.values()}
        downstream_map: Dict[str, List[str]] = {tid: [] for tid in uid_to_task.values()}
        for edge in edges:
            src_tid = uid_to_task.get(edge.get("source"))
            tgt_tid = uid_to_task.get(edge.get("target"))
            if src_tid and tgt_tid and src_tid != tgt_tid:
                if src_tid not in dag[tgt_tid]:
                    dag[tgt_tid].append(src_tid)
                if tgt_tid not in downstream_map[src_tid]:
                    downstream_map[src_tid].append(tgt_tid)

        if job_id:
            if job_id not in dag:
                raise ValueError(f"Task '{job_id}' not found in execution '{eid}' dependency graph.")
            return {
                "execution_id": eid,
                "job_id": job_id,
                "dependencies": dag[job_id],
                "downstream": downstream_map[job_id],
            }
        return {"execution_id": eid, "dag": dag, "downstream": downstream_map}

    def read_job_log(
        self,
        job_id: str,
        tail_lines: int = 50,
        execution_id: Optional[str] = None,
    ) -> str:
        eid = execution_id or self.default_execution_id
        if not eid:
            return "Error: No execution_id bound to read logs."
        if not self.is_registered_job(job_id, execution_id=eid):
            return f"Error: Task '{job_id}' does not exist in Kestra execution '{eid}'."
        entries = self.get_execution_logs(eid, task_id=job_id, tail_lines=tail_lines)
        if not entries:
            return f"No log entries recorded for task '{job_id}' in execution '{eid}'."
        lines: List[str] = []
        for entry in entries:
            ts = entry.get("timestamp", "")
            lvl = entry.get("level", "INFO")
            att = entry.get("attemptNumber", 0)
            msg = entry.get("message", "")
            lines.append(f"[{ts}] {lvl} (attempt {att}): {msg}")
        return "\n".join(lines)

    def get_state(self, execution_id: Optional[str] = None) -> Optional[KestraWorkflowState]:
        eid = execution_id or self.default_execution_id
        if not eid:
            return None
        raw = self.get_execution_raw(eid)
        ns = raw.get("namespace", "unknown")
        fid = raw.get("flowId", "unknown")
        exec_state = (raw.get("state") or {}).get("current", "UNKNOWN")

        deps_info = {}
        try:
            deps_info = self.get_dependencies(execution_id=eid).get("dag", {})
        except Exception:
            pass

        jobs: Dict[str, KestraTaskState] = {}
        total_attempts = 0
        task_runs = raw.get("taskRunList", [])
        parent_run_ids = {
            tr.get("parentTaskRunId")
            for tr in task_runs
            if isinstance(tr, dict) and tr.get("parentTaskRunId")
        }
        for tr in task_runs:
            tid = tr.get("taskId")
            if not tid:
                continue
            tr_state = (tr.get("state") or {}).get("current", "UNKNOWN")
            attempts = tr.get("attempts") or []
            att_count = len(attempts)
            last_att_state = (
                (attempts[-1].get("state") or {}).get("current")
                if attempts and isinstance(attempts[-1], dict)
                else None
            )
            total_attempts += att_count
            err_msg: Optional[str] = None
            if _map_kestra_state(tr_state) == JobStatus.FAILED:
                try:
                    logs = self.get_execution_logs(eid, task_id=tid, tail_lines=20)
                    err_lines = [
                        e.get("message", "")
                        for e in logs
                        if str(e.get("level", "")).upper() in ("ERROR", "WARN")
                    ]
                    if err_lines:
                        err_msg = " | ".join(err_lines[-3:])[:500]
                    elif logs:
                        err_msg = str(logs[-1].get("message", ""))[:500]
                except Exception:
                    err_msg = f"Task '{tid}' ended in state {tr_state}"

            jobs[tid] = KestraTaskState(
                id=tid,
                name=tid,
                status=_map_kestra_state(tr_state),
                raw_state=tr_state,
                attempt_count=att_count,
                last_attempt_state=last_att_state,
                is_container=(tr.get("id") in parent_run_ids) or tid.startswith("if_"),
                task_run_id=tr.get("id"),
                exit_code=0 if tr_state == "SUCCESS" else (1 if _map_kestra_state(tr_state) == JobStatus.FAILED else None),
                error_message=err_msg,
                dependencies=deps_info.get(tid, []),
                outputs=tr.get("outputs") or {},
            )

        pg_ok = self.check_service_status("pgdatabase").get("healthy", False)
        return KestraWorkflowState(
            workflow_id=f"{ns}/{fid}",
            run_id=eid,
            namespace=ns,
            flow_id=fid,
            execution_state=exec_state,
            jobs=jobs,
            services={"pgdatabase": pg_ok},
            inputs=raw.get("inputs") or {},
            total_attempts=total_attempts,
        )

    # --- Execution Restart & Bounded Polling ---

    def restart_execution(
        self,
        execution_id: str,
        poll_timeout: float = 120.0,
        poll_interval: float = 1.5,
    ) -> KestraWorkflowState:
        """
        Restarts a failed Kestra execution via `POST /executions/{id}/restart` and polls
        boundedly until a terminal state is reached AND fresh task attempt(s) occurred.
        """
        before = self.get_state(execution_id=execution_id)
        if before is None:
            raise ValueError(f"Execution '{execution_id}' not found.")
        baseline_attempts = before.total_attempts

        resp = self._request("POST", f"/executions/{execution_id}/restart")
        if resp.status_code not in (200, 201, 202):
            raise RuntimeError(
                f"Kestra restart failed for '{execution_id}' ({resp.status_code}): {resp.text[:300]}"
            )

        deadline = time.monotonic() + poll_timeout
        saw_fresh_attempt = False
        last_state: Optional[KestraWorkflowState] = None

        while time.monotonic() < deadline:
            cur = self.get_state(execution_id=execution_id)
            if cur is not None:
                last_state = cur
                if cur.total_attempts > baseline_attempts:
                    saw_fresh_attempt = True
                # Only treat terminal state as complete once a fresh attempt or non-pending state settled
                if saw_fresh_attempt and cur.execution_state in ("SUCCESS", "FAILED", "KILLED", "WARNING"):
                    return cur
            time.sleep(poll_interval)

        raise TimeoutError(
            f"Timed out after {poll_timeout}s waiting for restarted execution '{execution_id}' "
            f"(last_state={last_state.execution_state if last_state else 'UNKNOWN'}, "
            f"saw_fresh_attempt={saw_fresh_attempt})."
        )

    # --- Service Health & Designated Dependency Control ---

    def check_service_status(self, service_name: str) -> Dict[str, Any]:
        if not self.is_registered_service(service_name):
            return {
                "service": service_name,
                "status": "UNREGISTERED",
                "healthy": False,
                "registered": False,
            }

        if self._service_controller is not None:
            ok, detail = self._service_controller("status", service_name)
            return {
                "service": service_name,
                "status": "RUNNING" if ok else "STOPPED",
                "healthy": ok,
                "registered": True,
                "detail": detail,
            }

        if service_name == "pgdatabase":
            try:
                import psycopg2

                conn = psycopg2.connect(
                    host=self.pg_host,
                    port=self.pg_port,
                    dbname=self.pg_dbname,
                    user=self.pg_user,
                    password=self.pg_password,
                    connect_timeout=3,
                )
                try:
                    with conn.cursor() as cur:
                        cur.execute("SELECT 1;")
                        cur.fetchone()
                finally:
                    conn.close()
                return {
                    "service": "pgdatabase",
                    "status": "RUNNING",
                    "port": self.pg_port,
                    "healthy": True,
                    "registered": True,
                    "detail": f"PostgreSQL accepting connections on {self.pg_host}:{self.pg_port}/{self.pg_dbname}",
                }
            except Exception as exc:
                return {
                    "service": "pgdatabase",
                    "status": "STOPPED",
                    "port": self.pg_port,
                    "healthy": False,
                    "registered": True,
                    "detail": str(exc).strip(),
                }

        # upstream_source (read-only reachability check; dynamically detects HTTP Download URI from bound flow if present)
        probe_url = self.source_probe_url
        if self.default_execution_id:
            try:
                raw = self.get_execution_raw(self.default_execution_id)
                ns = raw.get("namespace")
                fid = raw.get("flowId")
                if ns and fid:
                    flow = self._request("GET", f"/flows/{ns}/{fid}").json()
                    for t in flow.get("tasks") or []:
                        if isinstance(t, dict) and t.get("type") == "io.kestra.plugin.core.http.Download":
                            uri = t.get("uri")
                            if isinstance(uri, str) and uri.startswith("http") and "{{" not in uri:
                                probe_url = uri
                                break
            except Exception:
                pass
        try:
            r = httpx.head(probe_url, timeout=5.0, follow_redirects=True)
            if r.status_code == 405:
                r = httpx.get(probe_url, timeout=5.0, follow_redirects=True)
            ok = r.status_code < 400
            return {
                "service": "upstream_source",
                "status": "REACHABLE" if ok else f"HTTP_{r.status_code}",
                "healthy": ok,
                "registered": True,
                "detail": f"HTTP {r.status_code} from {probe_url}",
            }
        except Exception as exc:
            return {
                "service": "upstream_source",
                "status": "UNREACHABLE",
                "healthy": False,
                "registered": True,
                "detail": str(exc).strip(),
            }

    def restart_service(self, service_name: str) -> Tuple[bool, str]:
        """
        Restarts ONLY the designated test dependency (`pgdatabase`).
        Rejects unregistered services and external services NOVA does not own.
        """
        if not self.is_registered_service(service_name):
            return (False, f"Service '{service_name}' is not a registered dependency.")
        if not self.is_restartable_service(service_name):
            return (
                False,
                f"Service '{service_name}' is an external read-only dependency and cannot be restarted by NOVA.",
            )

        if self._service_controller is not None:
            return self._service_controller("restart", service_name)

        docker_bin = _find_docker_bin()
        if not docker_bin:
            return (False, "Docker CLI binary not found to restart container dependency.")

        env = os.environ.copy()
        docker_dir = os.path.dirname(docker_bin)
        if docker_dir not in env.get("PATH", ""):
            env["PATH"] = f"{docker_dir}{os.pathsep}{env.get('PATH', '')}"

        proc = subprocess.run(
            [docker_bin, "start", self.pg_container_name],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
        if proc.returncode != 0:
            return (False, f"Failed to start '{self.pg_container_name}': {proc.stderr.strip()}")

        # Wait up to 20s for PostgreSQL to accept queries
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            status = self.check_service_status("pgdatabase")
            if status.get("healthy"):
                return (True, f"Container '{self.pg_container_name}' started and PostgreSQL is healthy.")
            time.sleep(1.0)
        return (False, f"Container '{self.pg_container_name}' started, but PostgreSQL did not become ready in 20s.")

    # --- Fresh Task-Attempt Attribution & Output Verification (M13) ---

    def verify_task_attempt_recovery(
        self,
        workflow_state: KestraWorkflowState,
        baseline_task_attempts: Optional[Dict[str, int]] = None,
        baseline_task_states: Optional[Dict[str, str]] = None,
        execution_history: Optional[List[Any]] = None,
        expected: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Dict[str, Any], str]:
        """
        Verifies that recovery is backed by fresh SUCCESS task attempts attributable
        to remediation and retry — NOT merely an upstream attempt bump or stale DB rows.
        """
        expected = expected or {}
        baseline_task_attempts = baseline_task_attempts or {}
        baseline_task_states = baseline_task_states or {}

        # Collect freshly succeeded tasks reported by successful resume_workflow execution(s)
        resume_records: List[Dict[str, Any]] = []
        fresh_from_resume: set[str] = set()
        for rec in execution_history or []:
            rec_dict = rec.model_dump() if hasattr(rec, "model_dump") else dict(rec)
            if rec_dict.get("tool_name") == "resume_workflow" and rec_dict.get("success"):
                resume_records.append(rec_dict)
                details = rec_dict.get("details") or {}
                for tid in details.get("freshly_succeeded_tasks") or []:
                    fresh_from_resume.add(tid)

        if not resume_records:
            return (
                False,
                {"reason": "missing_successful_resume_workflow"},
                "Verification failed: no successful resume_workflow retry execution recorded.",
            )

        # Determine required downstream / load-merge tasks for this workflow
        verifier_kind = expected.get("verifier")
        if not verifier_kind:
            if workflow_state.flow_id == "04_postgres_taxi" or expected.get("taxi") or expected.get("table"):
                verifier_kind = "postgres_taxi"
            elif workflow_state.flow_id == "03_getting_started_data_pipeline":
                verifier_kind = "duckdb"

        required_tasks: List[str] = list(expected.get("required_tasks") or [])
        if not required_tasks and verifier_kind == "postgres_taxi":
            taxi = expected.get("taxi") or (workflow_state.inputs or {}).get("taxi") or "green"
            required_tasks = [
                f"{taxi}_copy_in_to_staging_table",
                f"{taxi}_add_unique_id_and_filename",
                f"{taxi}_merge_data",
            ]
        elif not required_tasks and verifier_kind == "duckdb":
            required_tasks = ["transform", "query"]

        # Any leaf worker task that was FAILED/non-SUCCESS at baseline must also have freshly succeeded.
        # Flow-control container tasks (e.g. Kestra `If` branching nodes) transition to SUCCESS when their
        # child tasks recover without incrementing their own attempt array, so only leaf tasks require fresh attempts.
        if baseline_task_states:
            for tid, prev_status in baseline_task_states.items():
                if prev_status != JobStatus.SUCCESS.value and tid not in required_tasks:
                    t_obj = workflow_state.jobs.get(tid)
                    if t_obj is not None and t_obj.is_container:
                        continue
                    required_tasks.append(tid)

        missing_tasks: List[str] = []
        stale_tasks: List[str] = []
        unsuccessful_tasks: List[str] = []
        verified_tasks: Dict[str, Any] = {}

        for tid in required_tasks:
            task = workflow_state.jobs.get(tid)
            if task is None:
                missing_tasks.append(tid)
                continue
            if task.status != JobStatus.SUCCESS or (
                task.last_attempt_state is not None and task.last_attempt_state != "SUCCESS"
            ):
                unsuccessful_tasks.append(
                    f"{tid} (status={task.status.value}, last_attempt={task.last_attempt_state})"
                )
                continue

            base_att = baseline_task_attempts.get(tid)
            has_fresh_attempt = False
            if base_att is not None:
                has_fresh_attempt = task.attempt_count > base_att
            elif baseline_task_attempts:
                # Task wasn't in baseline_task_attempts because it hadn't run yet -> attempt_count > 0 is fresh
                has_fresh_attempt = task.attempt_count > 0
            elif tid in fresh_from_resume:
                has_fresh_attempt = True

            if not has_fresh_attempt:
                stale_tasks.append(
                    f"{tid} (baseline_attempts={base_att}, observed_attempts={task.attempt_count})"
                )
            else:
                verified_tasks[tid] = {
                    "attempt_count": task.attempt_count,
                    "baseline_attempts": base_att,
                    "last_attempt_state": task.last_attempt_state or "SUCCESS",
                }

        if missing_tasks or unsuccessful_tasks or stale_tasks:
            parts = []
            if missing_tasks:
                parts.append(f"missing required tasks: {missing_tasks}")
            if unsuccessful_tasks:
                parts.append(f"tasks not ending in SUCCESS: {unsuccessful_tasks}")
            if stale_tasks:
                parts.append(f"tasks lacking fresh post-remediation attempts: {stale_tasks}")
            msg = (
                "Verification failed: recovery requires fresh successful downstream/remediation task attempts; "
                + "; ".join(parts)
            )
            return (
                False,
                {
                    "missing_tasks": missing_tasks,
                    "unsuccessful_tasks": unsuccessful_tasks,
                    "stale_tasks": stale_tasks,
                    "verified_tasks": verified_tasks,
                },
                msg,
            )

        # If no required_tasks list was available (generic flow), ensure at least one non-baseline-SUCCESS task freshly succeeded
        if not required_tasks:
            fresh_any = [
                tid
                for tid, task in workflow_state.jobs.items()
                if task.status == JobStatus.SUCCESS
                and (task.last_attempt_state in (None, "SUCCESS"))
                and (
                    task.attempt_count > baseline_task_attempts.get(tid, 0)
                    or tid in fresh_from_resume
                )
            ]
            if not fresh_any:
                return (
                    False,
                    {"fresh_tasks": []},
                    "Verification failed: no fresh successful task attempt attributable to remediation/retry.",
                )
            verified_tasks = {
                tid: {"attempt_count": workflow_state.jobs[tid].attempt_count}
                for tid in fresh_any
            }

        return (
            True,
            {"verified_tasks": verified_tasks, "verifier_kind": verifier_kind or "generic"},
            f"Fresh task-attempt attribution passed for tasks: {sorted(verified_tasks.keys())}.",
        )

    # --- Independent Read-Only Output Verification ---

    def _verify_duckdb_output(
        self,
        expected: Dict[str, Any],
        execution_id: Optional[str] = None,
    ) -> Tuple[bool, Dict[str, Any], str]:
        """Verifies stored DuckDB aggregation output from a Kestra execution."""
        eid = execution_id or expected.get("execution_id") or self.default_execution_id
        if not eid:
            return (False, {}, "DuckDB verification failed: no execution_id available.")
        wf_state = self.get_state(execution_id=eid)
        if wf_state is None:
            return (False, {}, f"DuckDB verification failed: execution '{eid}' not found.")

        task_id = expected.get("duckdb_task_id", "query")
        query_task = wf_state.jobs.get(task_id)
        if query_task is None or query_task.status != JobStatus.SUCCESS:
            return (
                False,
                {"task_id": task_id, "status": query_task.status.value if query_task else "MISSING"},
                f"DuckDB verification failed: task '{task_id}' did not complete with SUCCESS.",
            )

        outputs = query_task.outputs or {}
        uri = outputs.get("uri")
        size = outputs.get("size")
        if not uri and isinstance(outputs.get("outputs"), list):
            for entry in reversed(outputs["outputs"]):
                if isinstance(entry, dict) and entry.get("uri"):
                    uri = entry.get("uri")
                    size = entry.get("size")
                    break
        details: Dict[str, Any] = {
            "execution_id": eid,
            "task_id": task_id,
            "output_uri": uri,
            "output_size": size,
        }

        if not uri:
            return (
                False,
                details,
                f"DuckDB verification failed: task '{task_id}' produced no stored output URI.",
            )
        if size is not None and int(size) <= 0:
            return (
                False,
                details,
                f"DuckDB verification failed: task '{task_id}' stored output size is {size} (expected > 0).",
            )

        # Fetch the stored Ion/output file from Kestra storage to confirm non-empty aggregation rows
        try:
            resp = self._request("GET", f"/executions/{eid}/file", params={"path": uri})
            if resp.status_code != 200:
                return (
                    False,
                    details,
                    f"DuckDB verification failed: unable to read stored output '{uri}' (HTTP {resp.status_code}).",
                )
            body = resp.text.strip()
            lines = [ln for ln in body.splitlines() if ln.strip()]
            details["stored_record_count"] = len(lines)
            details["sample_record"] = lines[0][:200] if lines else ""
            if not lines:
                return (
                    False,
                    details,
                    f"DuckDB verification failed: stored output file '{uri}' is empty.",
                )
            expected_fields = expected.get("expected_fields") or []
            for fld in expected_fields:
                if fld not in body:
                    return (
                        False,
                        details,
                        f"DuckDB verification failed: expected field '{fld}' missing from stored output.",
                    )
        except Exception as exc:
            return (
                False,
                {**details, "error": str(exc)},
                f"DuckDB stored output read error: {exc}",
            )

        return (
            True,
            details,
            f"DuckDB read-only output verification passed for '{task_id}' ({details.get('stored_record_count', size)} records).",
        )

    def verify_output(
        self,
        expected: Optional[Dict[str, Any]] = None,
        execution_id: Optional[str] = None,
    ) -> Tuple[bool, Dict[str, Any], str]:
        """
        Opens an independent read-only verification check for the workflow output:
        - PostgreSQL taxi workflow (`04_postgres_taxi`): row count, distinct unique_row_id, sum(total_amount).
        - DuckDB pipeline (`03_getting_started_data_pipeline`): stored DuckDB aggregation URI & non-empty records.
        """
        expected = expected or {}
        if self._pg_verifier is not None:
            return self._pg_verifier(expected)

        verifier_kind = expected.get("verifier")
        eid = execution_id or expected.get("execution_id") or self.default_execution_id
        if not verifier_kind and eid:
            try:
                wf_state = self.get_state(execution_id=eid)
                if wf_state is not None:
                    if wf_state.flow_id == "03_getting_started_data_pipeline":
                        verifier_kind = "duckdb"
                    elif wf_state.flow_id == "04_postgres_taxi":
                        verifier_kind = "postgres_taxi"
                    elif "query" in wf_state.jobs:
                        q_out = wf_state.jobs["query"].outputs or {}
                        if q_out.get("uri") or isinstance(q_out.get("outputs"), list):
                            verifier_kind = "duckdb"
            except Exception:
                pass

        if verifier_kind == "execution_only":
            return (True, {"verifier": "execution_only"}, "Execution-only output verification passed.")

        if verifier_kind == "duckdb":
            return self._verify_duckdb_output(expected, execution_id=eid)

        if not verifier_kind and not any(k in expected for k in ("taxi", "table", "expected_row_count", "year", "month")):
            return (True, {"verifier": "execution_only"}, "No external output table configured; task-attempt verification passed.")

        taxi = expected.get("taxi", "green")
        year = expected.get("year", "2020")
        month = expected.get("month", "04")
        table = expected.get("table", f"public.{taxi}_tripdata")
        if table not in ("public.green_tripdata", "public.yellow_tripdata"):
            return (False, {}, f"Refused unapproved verification table '{table}'.")
        filename = expected.get("filename", f"{taxi}_tripdata_{year}-{month}.csv")

        try:
            import psycopg2

            conn = psycopg2.connect(
                host=self.pg_host,
                port=self.pg_port,
                dbname=self.pg_dbname,
                user=self.pg_user,
                password=self.pg_password,
                connect_timeout=5,
            )
            conn.set_session(readonly=True, autocommit=True)
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        f"""
                        SELECT
                            COUNT(*)::bigint,
                            COUNT(DISTINCT unique_row_id)::bigint,
                            ROUND(COALESCE(SUM(total_amount), 0)::numeric, 2)::text
                        FROM {table}
                        WHERE filename = %s
                        """,
                        (filename,),
                    )
                    row = cur.fetchone()
            finally:
                conn.close()

            row_count = int(row[0]) if row else 0
            distinct_ids = int(row[1]) if row else 0
            sum_total_amount = float(row[2]) if row and row[2] is not None else 0.0

            details: Dict[str, Any] = {
                "table": table,
                "filename": filename,
                "row_count": row_count,
                "distinct_ids": distinct_ids,
                "sum_total_amount": sum_total_amount,
            }

            if row_count <= 0 or distinct_ids != row_count:
                return (
                    False,
                    details,
                    f"PostgreSQL output check failed for {table} ({filename}): "
                    f"row_count={row_count}, distinct_ids={distinct_ids}.",
                )

            exp_rows = expected.get("expected_row_count")
            exp_distinct = expected.get("expected_distinct_ids")
            exp_sum = expected.get("expected_sum_total_amount")

            if exp_rows is not None and row_count != int(exp_rows):
                return (
                    False,
                    details,
                    f"PostgreSQL row_count mismatch for {filename}: expected {exp_rows}, observed {row_count}.",
                )
            if exp_distinct is not None and distinct_ids != int(exp_distinct):
                return (
                    False,
                    details,
                    f"PostgreSQL distinct_ids mismatch for {filename}: expected {exp_distinct}, observed {distinct_ids}.",
                )
            if exp_sum is not None and abs(sum_total_amount - float(exp_sum)) > 0.01:
                return (
                    False,
                    details,
                    f"PostgreSQL sum_total_amount mismatch for {filename}: expected {exp_sum}, observed {sum_total_amount}.",
                )

            return (
                True,
                details,
                f"PostgreSQL read-only verification passed for {table} ({filename}): "
                f"row_count={row_count}, distinct_ids={distinct_ids}, sum_total_amount={sum_total_amount:.2f}.",
            )
        except Exception as exc:
            return (
                False,
                {"table": table, "filename": filename, "error": str(exc)},
                f"PostgreSQL verification error: {exc}",
            )
