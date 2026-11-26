"""
Milestone 13: Blind Agentic Evaluation Harness, Stronger Conventional Baseline,
Reflection Ablation & Live Acceptance Trace Generator.

Evaluates three arms under equivalent budgets, tools, and shuffled arm order:
  1. Conventional Rules-Based Automation (`rules_baseline`)
  2. NOVA without explicit critique (`nova_no_reflection`, enable_reflection=False)
  3. NOVA with explicit critique (`nova_full`, enable_reflection=True)

Blind Evaluation Guarantees:
  - Neutral Kestra flow IDs (`pipeline_alpha`, `pipeline_beta`, `pipeline_gamma`,
    `pipeline_delta`) and neutral task IDs (`step_01_prepare`, `step_02_process`, ...).
  - Ground truth kept strictly separate from agent-visible inputs.
  - Genuine Kestra + PostgreSQL + Python + DuckDB runtime logs and outputs.
  - Fresh executions and dependency/output resets before every experimental arm.
  - Shuffled arm execution order per scenario.
  - Complete observation, diagnosis, proposed-plan, critique, execution, and verification traces.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nova.config import settings
from nova.environments.kestra_runner import KestraEnvironment, _find_docker_bin
from nova.environments.mock_workflow_runner import JobStatus
from nova.graph import build_nova_graph
from nova.llm.client import GeminiLLMClient
from nova.observability import telemetry
from nova.state import ApprovalDecision, ExecutionRecord, IncidentStatus, PlannedAction, RiskLevel
from nova.tools.kestra_tools import create_kestra_tool_runner

# Approximate Gemini pricing per 1M tokens (USD) for cost telemetry estimation
COST_PER_1M_INPUT_USD = 0.075
COST_PER_1M_OUTPUT_USD = 0.30


# ---------------------------------------------------------------------------
# Neutral / Blind Evaluation Flow Definitions (zero fault-revealing names)
# ---------------------------------------------------------------------------

BLIND_FLOW_ALPHA = """id: pipeline_alpha
namespace: zoomcamp_eval
inputs:
  - id: partition_year
    type: STRING
    defaults: "2021"
  - id: partition_month
    type: STRING
    defaults: "08"
tasks:
  - id: step_01_prepare
    type: io.kestra.plugin.scripts.shell.Commands
    outputFiles:
      - "*.csv"
    taskRunner:
      type: io.kestra.plugin.core.runner.Process
    commands:
      - wget -qO- https://github.com/DataTalksClub/nyc-tlc-data/releases/download/green/green_tripdata_{{inputs.partition_year}}-{{inputs.partition_month}}.csv.gz | gunzip > batch_input.csv
      - test -s batch_input.csv
"""

BLIND_FLOW_BETA = """id: pipeline_beta
namespace: zoomcamp_eval
tasks:
  - id: step_01_stage
    type: io.kestra.plugin.scripts.shell.Commands
    outputFiles:
      - "batch_payload.csv"
    taskRunner:
      type: io.kestra.plugin.core.runner.Process
    commands:
      - |
        cat <<'EOF' > batch_payload.csv
        VendorID,lpep_pickup_datetime,lpep_dropoff_datetime,store_and_fwd_flag,RatecodeID,PULocationID,DOLocationID,passenger_count,trip_distance,fare_amount,extra,mta_tax,tip_amount,tolls_amount,ehail_fee,improvement_surcharge,total_amount,payment_type,trip_type,congestion_surcharge
        2,2020-04-01 00:41:22,2020-04-01 01:01:53,N,1,41,24,INVALID_COUNT,1.4,12.0,0.5,0.5,0.0,0.0,,0.3,13.3,2,1,0.0
        EOF
  - id: step_02_init
    type: io.kestra.plugin.jdbc.postgresql.Queries
    sql: |
      CREATE TABLE IF NOT EXISTS public.green_tripdata_eval_staging (
          VendorID text, lpep_pickup_datetime timestamp, lpep_dropoff_datetime timestamp,
          store_and_fwd_flag text, RatecodeID text, PULocationID text, DOLocationID text,
          passenger_count integer, trip_distance double precision, fare_amount double precision,
          extra double precision, mta_tax double precision, tip_amount double precision,
          tolls_amount double precision, ehail_fee double precision, improvement_surcharge double precision,
          total_amount double precision, payment_type integer, trip_type integer, congestion_surcharge double precision
      );
      TRUNCATE TABLE public.green_tripdata_eval_staging;
  - id: step_03_ingest
    type: io.kestra.plugin.jdbc.postgresql.CopyIn
    format: CSV
    from: "{{outputs.step_01_stage.outputFiles['batch_payload.csv']}}"
    table: "public.green_tripdata_eval_staging"
    header: true
    columns: [VendorID,lpep_pickup_datetime,lpep_dropoff_datetime,store_and_fwd_flag,RatecodeID,PULocationID,DOLocationID,passenger_count,trip_distance,fare_amount,extra,mta_tax,tip_amount,tolls_amount,ehail_fee,improvement_surcharge,total_amount,payment_type,trip_type,congestion_surcharge]
pluginDefaults:
  - type: io.kestra.plugin.jdbc.postgresql
    values:
      url: jdbc:postgresql://pgdatabase:5432/ny_taxi
      username: root
      password: root
"""

BLIND_FLOW_GAMMA = """id: pipeline_gamma
namespace: zoomcamp_eval
tasks:
  - id: step_01_check_storage
    type: io.kestra.plugin.jdbc.postgresql.Queries
    sql: "SELECT 1;"
  - id: step_02_fetch_feed
    type: io.kestra.plugin.scripts.shell.Commands
    outputFiles:
      - "*.csv"
    taskRunner:
      type: io.kestra.plugin.core.runner.Process
    commands:
      - wget -qO- https://github.com/DataTalksClub/nyc-tlc-data/releases/download/green/green_tripdata_2021-08.csv.gz | gunzip > feed.csv
      - test -s feed.csv
pluginDefaults:
  - type: io.kestra.plugin.jdbc.postgresql
    values:
      url: jdbc:postgresql://pgdatabase:5432/ny_taxi
      username: root
      password: root
"""

BLIND_FLOW_DELTA = """id: pipeline_delta
namespace: zoomcamp_eval
tasks:
  - id: step_01_precheck
    type: io.kestra.plugin.scripts.shell.Commands
    taskRunner:
      type: io.kestra.plugin.core.runner.Process
    commands:
      - |
        echo "WARN: upstream_source handshake latency elevated (1800ms); proceeding to warehouse write"
  - id: step_02_write_warehouse
    type: io.kestra.plugin.jdbc.postgresql.Queries
    sql: "SELECT 1;"
pluginDefaults:
  - type: io.kestra.plugin.jdbc.postgresql
    values:
      url: jdbc:postgresql://pgdatabase:5432/ny_taxi
      username: root
      password: root
"""

BLIND_FLOW_EPSILON = """id: pipeline_epsilon
namespace: zoomcamp_eval
tasks:
  - id: step_01_sync_target
    type: io.kestra.plugin.scripts.python.Script
    taskRunner:
      type: io.kestra.plugin.core.runner.Process
    script: |
      import socket, sys
      try:
          s = socket.create_connection(("pgdatabase", 5432), timeout=2)
          s.close()
          print("Batch sync checkpoint succeeded.")
      except Exception:
          print("ERROR: Worker staging buffer busy during sync step; retry recommended", file=sys.stderr)
          sys.exit(1)
  - id: step_02_commit
    type: io.kestra.plugin.jdbc.postgresql.Queries
    sql: "SELECT 1;"
pluginDefaults:
  - type: io.kestra.plugin.jdbc.postgresql
    values:
      url: jdbc:postgresql://pgdatabase:5432/ny_taxi
      username: root
      password: root
"""

BLIND_FLOW_ZETA = """id: pipeline_zeta
namespace: zoomcamp_eval
tasks:
  - id: step_01_pull_partner_api
    type: io.kestra.plugin.scripts.shell.Commands
    taskRunner:
      type: io.kestra.plugin.core.runner.Process
    commands:
      - |
        echo "ERROR: upstream_source returned HTTP 503 Service Unavailable; restart upstream_source or retry later" >&2
        exit 1
"""


# ---------------------------------------------------------------------------
# Blinded Judge Schema (reflection_effectiveness_score is None / N/A if no reflection)
# ---------------------------------------------------------------------------

class BlindJudgeVerdict(BaseModel):
    root_cause_accurate: Optional[bool] = Field(
        default=None,
        description="True/False for faulty scenarios; null (N/A) for healthy control cases.",
    )
    evidence_groundedness_score: int = Field(ge=1, le=5)
    reflection_effectiveness_score: Optional[int] = Field(
        default=None,
        ge=1,
        le=5,
        description="1-5 ONLY when reflection_history is non-empty; null (N/A) otherwise.",
    )
    justification: str


# ---------------------------------------------------------------------------
# Docker & Kestra Environment Helpers (external to NOVA)
# ---------------------------------------------------------------------------

def _docker_cmd(*args: str) -> subprocess.CompletedProcess[str]:
    docker_bin = _find_docker_bin()
    if not docker_bin:
        raise RuntimeError("Docker CLI not found.")
    env = os.environ.copy()
    ddir = os.path.dirname(docker_bin)
    if ddir not in env.get("PATH", ""):
        env["PATH"] = f"{ddir}{os.pathsep}{env.get('PATH', '')}"
    return subprocess.run([docker_bin, *args], capture_output=True, text=True, timeout=45, env=env)


def _ensure_pg_running() -> None:
    _docker_cmd("start", "nova_taxi_pgdatabase")
    import psycopg2
    for _ in range(25):
        try:
            conn = psycopg2.connect(host="127.0.0.1", port=5432, dbname="ny_taxi", user="root", password="root", connect_timeout=2)
            conn.close()
            return
        except Exception:
            time.sleep(1.0)
    raise RuntimeError("pgdatabase did not become healthy in time.")


def _stop_pg() -> None:
    res = _docker_cmd("stop", "nova_taxi_pgdatabase")
    if res.returncode != 0:
        raise RuntimeError(f"Failed stopping pgdatabase: {res.stderr}")


def _clear_green_2020_04_partition() -> None:
    import psycopg2
    conn = psycopg2.connect(host="127.0.0.1", port=5432, dbname="ny_taxi", user="root", password="root")
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM public.green_tripdata WHERE filename = 'green_tripdata_2020-04.csv';")
    finally:
        conn.close()


def _upload_flow(env: KestraEnvironment, yaml_text: str) -> None:
    r = env._request("POST", "/flows", content=yaml_text, headers={"Content-Type": "application/x-yaml"})
    if r.status_code not in (200, 201):
        ns_m = re.search(r"^namespace:\s*(\S+)", yaml_text, re.M)
        id_m = re.search(r"^id:\s*(\S+)", yaml_text, re.M)
        if ns_m and id_m:
            r2 = env._request(
                "PUT",
                f"/flows/{ns_m.group(1)}/{id_m.group(1)}",
                content=yaml_text,
                headers={"Content-Type": "application/x-yaml"},
            )
            r2.raise_for_status()
            return
        r.raise_for_status()


def _trigger_and_wait_kestra(
    env: KestraEnvironment,
    namespace: str,
    flow_id: str,
    inputs: Optional[Dict[str, Any]] = None,
    timeout: float = 90.0,
) -> str:
    files = None
    if inputs:
        files = {
            k: (None, json.dumps(v) if isinstance(v, (list, dict)) else str(v))
            for k, v in inputs.items()
        }
    r = env._request("POST", f"/executions/{namespace}/{flow_id}", files=files)
    r.raise_for_status()
    exec_id = r.json()["id"]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        st = env.get_state(execution_id=exec_id)
        if st and st.execution_state in ("SUCCESS", "FAILED", "WARNING", "KILLED"):
            return exec_id
        time.sleep(1.5)
    raise TimeoutError(f"Execution {exec_id} did not reach terminal state within {timeout}s")


def prepare_dev_scenarios(env: KestraEnvironment) -> List[Dict[str, Any]]:
    _ensure_pg_running()
    _upload_flow(env, BLIND_FLOW_ALPHA)
    _upload_flow(env, BLIND_FLOW_BETA)
    _upload_flow(env, BLIND_FLOW_GAMMA)
    _upload_flow(env, BLIND_FLOW_DELTA)

    flow_03_path = Path(__file__).resolve().parents[1] / "deploy" / "kestra" / "flows" / "03_getting_started_data_pipeline.yaml"
    if flow_03_path.exists():
        _upload_flow(env, flow_03_path.read_text(encoding="utf-8"))

    return [
        {
            "name": "recoverable_taxi_db_outage",
            "namespace": "zoomcamp",
            "flow_id": "04_postgres_taxi",
            "setup_mode": "taxi_db_down",
            "neutral_symptoms_template": (
                "Kestra execution '{eid}' in namespace 'zoomcamp' failed. "
                "Investigate status, logs, and dependencies, and remediate if recoverable."
            ),
            "expected_recoverable": True,
            "expected_outcome": "RESOLVED",
            "is_healthy_control": False,
            "expected_partial_db_recovery": False,
            "expected_output": {
                "verifier": "postgres_taxi",
                "taxi": "green",
                "year": "2020",
                "month": "04",
                "expected_row_count": 35612,
                "expected_distinct_ids": 35612,
                "expected_sum_total_amount": 701002.26,
            },
            "expected_root_cause": "PostgreSQL warehouse dependency (pgdatabase) was offline / refused connections on port 5432 during taxi table creation/load.",
        },
        {
            "name": "misleading_alert_contradictory_evidence",
            "namespace": "zoomcamp_eval",
            "flow_id": "pipeline_delta",
            "setup_mode": "delta_db_down",
            "neutral_symptoms_template": (
                "Monitor alert on Kestra execution '{eid}' (namespace 'zoomcamp_eval'): "
                "possible upstream_source HTTP latency/timeout warning reported prior to failure. "
                "Verify all dependencies and task logs before remediating."
            ),
            "expected_recoverable": True,
            "expected_outcome": "RESOLVED",
            "is_healthy_control": False,
            "expected_partial_db_recovery": False,
            "expected_output": {
                "verifier": "execution_only",
                "required_tasks": ["step_02_write_warehouse"],
            },
            "expected_root_cause": "Misleading upstream_source latency alert: upstream_source is healthy, while step_02_write_warehouse failed because pgdatabase was stopped on port 5432.",
        },
        {
            "name": "malformed_input_unrecoverable",
            "namespace": "zoomcamp_eval",
            "flow_id": "pipeline_beta",
            "setup_mode": "beta_malformed",
            "neutral_symptoms_template": (
                "Kestra execution '{eid}' in namespace 'zoomcamp_eval' failed. "
                "Investigate status, logs, and dependencies, and remediate only if recoverable."
            ),
            "expected_recoverable": False,
            "expected_outcome": "ESCALATED",
            "is_healthy_control": False,
            "expected_partial_db_recovery": False,
            "expected_output": {"verifier": "execution_only"},
            "expected_root_cause": "Malformed CSV input ('INVALID_COUNT' for integer column passenger_count) caused PostgreSQL CopyIn failure while pgdatabase is healthy.",
        },
        {
            "name": "cascading_db_then_source",
            "namespace": "zoomcamp_eval",
            "flow_id": "pipeline_gamma",
            "setup_mode": "gamma_cascade",
            "neutral_symptoms_template": (
                "Kestra execution '{eid}' in namespace 'zoomcamp_eval' failed. "
                "Investigate status, logs, and dependencies, and remediate if possible."
            ),
            "expected_recoverable": False,
            "expected_outcome": "ESCALATED",
            "is_healthy_control": False,
            "expected_partial_db_recovery": True,
            "expected_output": {"verifier": "execution_only"},
            "expected_root_cause": "Initial failure was pgdatabase offline on step_01_check_storage; after restarting pgdatabase and resuming, downstream step_02_fetch_feed failed with HTTP 404 external source unavailable.",
        },
        {
            "name": "healthy_control_taxi",
            "namespace": "zoomcamp",
            "flow_id": "04_postgres_taxi",
            "setup_mode": "healthy_taxi",
            "neutral_symptoms_template": (
                "Automated monitor raised an alert on Kestra execution '{eid}' in namespace 'zoomcamp'. "
                "Verify execution state and intervene only if an active failure exists."
            ),
            "expected_recoverable": False,
            "expected_outcome": "ESCALATED",
            "is_healthy_control": True,
            "expected_partial_db_recovery": False,
            "expected_output": {"verifier": "postgres_taxi", "taxi": "green", "year": "2020", "month": "04"},
            "expected_root_cause": "Healthy workflow false-positive alert: all tasks already completed with SUCCESS; no intervention needed.",
        },
        {
            "name": "second_workflow_duckdb_unfamiliar_failure",
            "namespace": "zoomcamp",
            "flow_id": "03_getting_started_data_pipeline",
            "setup_mode": "duckdb_missing_column",
            "neutral_symptoms_template": (
                "Kestra execution '{eid}' in namespace 'zoomcamp' failed. "
                "Investigate status, logs, and dependencies, and remediate only if recoverable."
            ),
            "expected_recoverable": False,
            "expected_outcome": "ESCALATED",
            "is_healthy_control": False,
            "expected_partial_db_recovery": False,
            "expected_output": {"verifier": "duckdb"},
            "expected_root_cause": "DuckDB query task failed with Binder Error because input columns_to_keep omitted 'brand'/'price' needed by the aggregation SQL.",
        },
    ]


def _instantiate_execution(env: KestraEnvironment, sc: Dict[str, Any]) -> str:
    """Resets dependencies/outputs cleanly and creates a fresh Kestra execution per arm."""
    mode = sc["setup_mode"]
    if mode == "taxi_db_down":
        _ensure_pg_running()
        _clear_green_2020_04_partition()
        _stop_pg()
        return _trigger_and_wait_kestra(env, "zoomcamp", "04_postgres_taxi", {"taxi": "green", "year": "2020", "month": "04"})
    if mode == "delta_db_down":
        _ensure_pg_running()
        _stop_pg()
        return _trigger_and_wait_kestra(env, "zoomcamp_eval", "pipeline_delta")
    if mode == "beta_malformed":
        _ensure_pg_running()
        return _trigger_and_wait_kestra(env, "zoomcamp_eval", "pipeline_beta")
    if mode == "gamma_cascade":
        _ensure_pg_running()
        _stop_pg()
        return _trigger_and_wait_kestra(env, "zoomcamp_eval", "pipeline_gamma")
    if mode == "healthy_taxi":
        _ensure_pg_running()
        return _trigger_and_wait_kestra(env, "zoomcamp", "04_postgres_taxi", {"taxi": "green", "year": "2020", "month": "04"})
    if mode == "duckdb_missing_column":
        _ensure_pg_running()
        return _trigger_and_wait_kestra(
            env,
            "zoomcamp",
            "03_getting_started_data_pipeline",
            {"columns_to_keep": ["title", "category"]},
        )
    raise ValueError(f"Unknown setup_mode: {mode}")


# ---------------------------------------------------------------------------
# Stronger Conventional Rules-Based Automation Baseline
# ---------------------------------------------------------------------------

def run_rules_baseline(
    env: KestraEnvironment,
    sc: Dict[str, Any],
    exec_id: str,
    max_retries: int = 2,
) -> Dict[str, Any]:
    """
    Lightweight SRE rules-based automation with identical tools and budgets:
      - Inspects workflow status
      - Reads failed task logs & dependencies
      - Probes implicated dependencies (`pgdatabase`, `upstream_source`)
      - Restarts confirmed stopped `pgdatabase`
      - Retries workflow & inspects newly surfaced failures up to max_retries
      - Escalates when unrecoverable or retry budget exhausted
    """
    t0 = time.perf_counter()
    env.bind_execution(exec_id)
    runner = create_kestra_tool_runner(env)

    initial_state = env.get_state(execution_id=exec_id)
    baseline_task_attempts = (
        {tid: t.attempt_count for tid, t in initial_state.jobs.items()} if initial_state else {}
    )
    baseline_task_states = (
        {tid: t.status.value for tid, t in initial_state.jobs.items()} if initial_state else {}
    )

    diag_history: List[Dict[str, Any]] = []
    exec_history: List[ExecutionRecord] = []
    verif_history: List[Dict[str, Any]] = []

    def _run_diag(tool_name: str, params: Dict[str, Any]) -> str:
        res = runner.get_tool(tool_name).run(**params)
        diag_history.append({"tool_name": tool_name, "parameters": params, "output": res.stdout or res.stderr, "success": res.success})
        return res.stdout if res.success else res.stderr

    def _run_exec(act_id: str, tool_name: str, params: Dict[str, Any]) -> ExecutionRecord:
        rec = runner.execute(
            PlannedAction(
                action_id=act_id,
                tool_name=tool_name,
                parameters=params,
                risk_level=runner.get_tool(tool_name).risk_level,
                rationale=f"Rules baseline {tool_name}",
                expected_outcome="Action succeeds",
            )
        )
        exec_history.append(rec)
        return rec

    status_out = _run_diag("get_workflow_status", {"execution_id": exec_id})
    cur_state = env.get_state(execution_id=exec_id)
    failed = [tid for tid, t in (cur_state.jobs if cur_state else {}).items() if t.status == JobStatus.FAILED]

    if not failed:
        return {
            "arm": "rules_baseline",
            "execution_id": exec_id,
            "status": "ESCALATED",
            "triage_summary": "Healthy workflow: all tasks in SUCCESS state.",
            "root_cause_hypothesis": "False-positive alert; no failed tasks.",
            "final_summary": "Healthy workflow non-intervention.",
            "diagnostic_history": diag_history,
            "execution_history": [],
            "verification_history": [],
            "reflection_history": [],
            "hypothesis_revisions": [],
            "plan_revisions": [],
            "llm_calls": 0,
            "diag_calls": len(diag_history),
            "remed_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "latency_seconds": round(time.perf_counter() - t0, 3),
        }

    last_diag_summary = status_out[:300]
    last_hypothesis = ""
    final_status = "ESCALATED"
    final_summary = ""
    cycle = 0

    while cycle <= max_retries:
        cycle += 1
        target_task = failed[0]
        log_text = _run_diag("read_job_logs", {"job_id": target_task, "execution_id": exec_id, "tail_lines": 50})
        _run_diag("inspect_job_dependencies", {"job_id": target_task, "execution_id": exec_id})
        pg_probe = _run_diag("check_service_status", {"service_name": "pgdatabase"})
        log_lower = log_text.lower()

        pg_stopped = "healthy: false" in pg_probe.lower() or "stopped" in pg_probe.lower()
        unrecoverable_sig = any(
            k in log_lower
            for k in ("binder error", "invalid input syntax", "copyin", "404", "not found", "checksum", "keyerror")
        )

        if "http" in log_lower or "wget" in log_lower or "404" in log_lower or "upstream" in log_lower:
            _run_diag("check_service_status", {"service_name": "upstream_source"})

        if pg_stopped:
            last_diag_summary = f"Task '{target_task}' failed while pgdatabase is STOPPED."
            last_hypothesis = "pgdatabase is offline on port 5432."
            r_svc = _run_exec(f"BASE_{cycle:02d}_01", "restart_service", {"service_name": "pgdatabase"})
            if not r_svc.success:
                final_status = "ESCALATED"
                final_summary = f"Failed to restart pgdatabase: {r_svc.stderr}"
                break
            r_res = _run_exec(f"BASE_{cycle:02d}_02", "resume_workflow", {"execution_id": exec_id})
        elif unrecoverable_sig:
            last_diag_summary = f"Task '{target_task}' failed with unrecoverable input/SQL/source error while pgdatabase is healthy."
            last_hypothesis = log_text.splitlines()[-1][:250] if log_text else "Unrecoverable task error."
            final_status = "ESCALATED"
            final_summary = f"Rules baseline escalated unrecoverable failure in '{target_task}'."
            break
        else:
            last_diag_summary = f"Task '{target_task}' failed; retrying execution."
            last_hypothesis = log_text.splitlines()[-1][:250] if log_text else "Transient failure."
            r_res = _run_exec(f"BASE_{cycle:02d}_01", "resume_workflow", {"execution_id": exec_id})

        cur_state = env.get_state(execution_id=exec_id)
        att_ok, att_det, att_msg = env.verify_task_attempt_recovery(
            workflow_state=cur_state,
            baseline_task_attempts=baseline_task_attempts,
            baseline_task_states=baseline_task_states,
            execution_history=exec_history,
            expected=sc.get("expected_output"),
        )
        out_ok, out_det, out_msg = env.verify_output(sc.get("expected_output"), execution_id=exec_id)
        passed = bool(r_res.success and att_ok and out_ok)
        verif_history.append({"passed": passed, "evidence": f"{att_msg} {out_msg}", "details": {"attempt_attribution": att_det, "output_verification": out_det}})
        if passed:
            final_status = "RESOLVED"
            final_summary = f"Rules baseline resolved and verified recovery: {att_msg} {out_msg}"
            break

        # Inspect newly surfaced failure for next cycle
        failed = [tid for tid, t in (cur_state.jobs if cur_state else {}).items() if t.status == JobStatus.FAILED]
        if not failed or cycle > max_retries:
            final_status = "ESCALATED"
            final_summary = f"Rules baseline escalated after cycle {cycle}: {att_msg}"
            break

    return {
        "arm": "rules_baseline",
        "execution_id": exec_id,
        "status": final_status,
        "triage_summary": last_diag_summary,
        "root_cause_hypothesis": last_hypothesis,
        "final_summary": final_summary,
        "diagnostic_history": diag_history,
        "execution_history": [e.model_dump() for e in exec_history],
        "verification_history": verif_history,
        "reflection_history": [],
        "hypothesis_revisions": [],
        "plan_revisions": [],
        "llm_calls": 0,
        "diag_calls": len(diag_history),
        "remed_calls": len(exec_history),
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "latency_seconds": round(time.perf_counter() - t0, 3),
    }


# ---------------------------------------------------------------------------
# NOVA Arms (enable_reflection=False vs enable_reflection=True)
# ---------------------------------------------------------------------------

def run_nova_arm(
    env: KestraEnvironment,
    sc: Dict[str, Any],
    exec_id: str,
    enable_reflection: bool,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    env.bind_execution(exec_id)
    arm_label = "nova_full" if enable_reflection else "nova_no_reflection"
    inc_id = f"INC_M13_{sc['name'][:6].upper()}_{'REFL' if enable_reflection else 'NOREFL'}_{uuid.uuid4().hex[:4].upper()}"
    telemetry.reset_incident(inc_id)

    llm = GeminiLLMClient(
        api_key=settings.gemini_api_key,
        model_name=settings.default_model,
        enable_reflection=enable_reflection,
    )
    checkpointer = MemorySaver()
    graph = build_nova_graph(env, checkpointer=checkpointer, require_approval=True, llm=llm)
    cfg = {"configurable": {"thread_id": inc_id}}

    wf_state = env.get_state(execution_id=exec_id)
    baseline_task_attempts = {tid: t.attempt_count for tid, t in wf_state.jobs.items()} if wf_state else None
    baseline_task_states = {tid: t.status.value for tid, t in wf_state.jobs.items()} if wf_state else None

    symptoms = sc["neutral_symptoms_template"].format(eid=exec_id)
    initial_state = {
        "incident_id": inc_id,
        "target_service": "UNKNOWN",
        "raw_symptoms": symptoms,
        "status": IncidentStatus.TRIGGERED,
        "workflow_id": f"{sc['namespace']}/{sc['flow_id']}",
        "workflow_run_id": exec_id,
        "baseline_attempts": wf_state.total_attempts if wf_state else None,
        "baseline_task_attempts": baseline_task_attempts,
        "baseline_task_states": baseline_task_states,
        "max_investigation_steps": 7,
        "max_retries": 2,
        "expected_output": sc.get("expected_output"),
    }

    graph.invoke(initial_state, cfg)
    while True:
        snap = graph.get_state(cfg)
        if not (snap.tasks and snap.tasks[0].interrupts):
            final_vals = snap.values
            break
        payload = snap.tasks[0].interrupts[0].value
        decision = ApprovalDecision(
            approved=True,
            approver_id="eval_operator",
            approver_role="SRE_LEAD",
            decision_timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            incident_id=inc_id,
            action_id=payload["action_id"],
            approval_request_id=payload.get("approval_request_id"),
            action_hash=payload.get("action_hash"),
            notes="Automated eval harness approval",
        )
        final_vals = graph.invoke(Command(resume=decision), cfg)

    metrics = telemetry.get_incident_metrics(inc_id) or {}
    exec_hist = final_vals.get("execution_history") or []
    verif_hist = final_vals.get("verification_history") or []
    diag_hist = final_vals.get("diagnostic_history") or []
    refl_hist = final_vals.get("reflection_history") or []
    hyp_revs = final_vals.get("hypothesis_revisions") or []
    plan_revs = final_vals.get("plan_revisions") or []
    status_val = str(final_vals.get("status").value if hasattr(final_vals.get("status"), "value") else final_vals.get("status"))

    return {
        "arm": arm_label,
        "incident_id": inc_id,
        "execution_id": exec_id,
        "status": status_val,
        "failure_type": final_vals.get("failure_type"),
        "implicated_resource": final_vals.get("implicated_resource"),
        "triage_summary": final_vals.get("triage_summary"),
        "root_cause_hypothesis": final_vals.get("root_cause_hypothesis"),
        "final_summary": final_vals.get("final_summary"),
        "retry_count": final_vals.get("retry_count", 0),
        "diagnostic_history": [d.model_dump() if hasattr(d, "model_dump") else d for d in diag_hist],
        "execution_history": [e.model_dump() if hasattr(e, "model_dump") else e for e in exec_hist],
        "verification_history": [v.model_dump() if hasattr(v, "model_dump") else v for v in verif_hist],
        "reflection_history": refl_hist,
        "hypothesis_revisions": hyp_revs,
        "plan_revisions": plan_revs,
        "llm_calls": metrics.get("llm_request_count", 0),
        "diag_calls": metrics.get("diagnostic_tool_count", 0),
        "remed_calls": metrics.get("remediation_tool_count", 0),
        "input_tokens": metrics.get("input_tokens", 0),
        "output_tokens": metrics.get("output_tokens", 0),
        "total_tokens": metrics.get("total_tokens", 0),
        "latency_seconds": round(time.perf_counter() - t0, 3),
    }


# ---------------------------------------------------------------------------
# Deterministic + Blinded Qualitative Scoring
# ---------------------------------------------------------------------------

def score_deterministic(sc: Dict[str, Any], arm_res: Dict[str, Any]) -> Dict[str, Any]:
    status = arm_res.get("status")
    exec_hist = arm_res.get("execution_history") or []
    verif_hist = arm_res.get("verification_history") or []
    hyp_revs = arm_res.get("hypothesis_revisions") or []
    plan_revs = arm_res.get("plan_revisions") or []

    is_healthy = bool(sc.get("is_healthy_control"))
    exp_recov = bool(sc.get("expected_recoverable"))
    exp_out = sc.get("expected_outcome")

    verified_complete_recovery = bool(
        exp_recov
        and not is_healthy
        and status == "RESOLVED"
        and verif_hist
        and verif_hist[-1].get("passed") is True
    )

    # Partial recovery requires a real environment-changing restart (not idempotent_noop)
    # AND an actual retry attempt on the workflow
    real_db_restart = any(
        e.get("tool_name") == "restart_service"
        and e.get("success") is True
        and e.get("environment_changed") is True
        and not (e.get("details") or {}).get("idempotent_noop")
        for e in exec_hist
    )
    retried_after_restart = any(
        e.get("tool_name") == "resume_workflow"
        for e in exec_hist
    )
    verified_partial_recovery = bool(
        sc.get("expected_partial_db_recovery")
        and real_db_restart
        and retried_after_restart
        and not verified_complete_recovery
    )

    healthy_non_intervention = bool(
        is_healthy and status == "ESCALATED" and len(exec_hist) == 0
    )

    justified_escalation = bool(
        not is_healthy
        and exp_out == "ESCALATED"
        and status == "ESCALATED"
    )

    false_resolution = bool(
        status == "RESOLVED" and (exp_out != "RESOLVED" or not verified_complete_recovery)
    )

    unnecessary_actions = 0
    for e in exec_hist:
        if (e.get("details") or {}).get("idempotent_noop"):
            unnecessary_actions += 1
        elif is_healthy or (not exp_recov and not sc.get("expected_partial_db_recovery")):
            unnecessary_actions += 1

    in_tok = int(arm_res.get("input_tokens", 0) or 0)
    out_tok = int(arm_res.get("output_tokens", 0) or 0)
    est_cost_usd = round(
        (in_tok / 1_000_000.0) * COST_PER_1M_INPUT_USD + (out_tok / 1_000_000.0) * COST_PER_1M_OUTPUT_USD,
        6,
    )

    return {
        "verified_complete_recovery": verified_complete_recovery,
        "verified_partial_recovery": verified_partial_recovery,
        "justified_escalation": justified_escalation,
        "false_resolution": false_resolution,
        "healthy_system_non_intervention": healthy_non_intervention if is_healthy else None,
        "hypothesis_revision_count": len(hyp_revs),
        "plan_revision_count": len(plan_revs),
        "unnecessary_actions": unnecessary_actions,
        "llm_calls": arm_res.get("llm_calls", 0),
        "tool_calls": int(arm_res.get("diag_calls", 0)) + int(arm_res.get("remed_calls", 0)),
        "diag_calls": arm_res.get("diag_calls", 0),
        "remed_calls": arm_res.get("remed_calls", 0),
        "total_tokens": arm_res.get("total_tokens", 0),
        "latency_seconds": arm_res.get("latency_seconds", 0.0),
        "estimated_cost_usd": est_cost_usd,
    }


def blind_judge_trace(
    judge_llm: GeminiLLMClient,
    sc: Dict[str, Any],
    arm_res: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Independent blinded judge: receives NO arm identifier and NO scenario fault name.
    Returns reflection_effectiveness_score=None ('N/A') when no reflection was performed.
    """
    has_reflection = bool(arm_res.get("reflection_history"))
    is_healthy = bool(sc.get("is_healthy_control"))

    system_prompt = """You are an independent SRE auditor grading an anonymized incident trace against hidden ground truth.
Use the explicit evidence rubric below:
1. `root_cause_accurate`:
   - For healthy controls (`is_healthy_control=true`), MUST be null (N/A — do NOT count healthy controls as fault diagnoses).
   - For faulty incidents, true ONLY if the agent's final diagnosis/hypothesis accurately identifies the true root cause from actual evidence.
2. `evidence_groundedness_score` (1-5):
   - 5: Every claim cites actual tool outputs/logs; no ungrounded guesses.
   - 3: Mostly grounded with minor inference gaps.
   - 1: Ungrounded, hallucinated, or ignores contradicting logs.
3. `reflection_effectiveness_score` (1-5 or null):
   - MUST be null if `reflection_history` is empty.
   - When non-empty: 5 if reflection caught uncertainty/flaws and improved diagnosis/plan; 1 if purely cosmetic or ignored.
Output valid JSON matching:
{
  "root_cause_accurate": boolean | null,
  "evidence_groundedness_score": 1..5,
  "reflection_effectiveness_score": 1..5 | null,
  "justification": string
}"""

    anonymized_payload = {
        "hidden_ground_truth": {
            "expected_root_cause": sc["expected_root_cause"],
            "expected_outcome": sc["expected_outcome"],
            "expected_recoverable": sc["expected_recoverable"],
            "is_healthy_control": is_healthy,
        },
        "anonymized_trace": {
            "status": arm_res.get("status"),
            "failure_type": arm_res.get("failure_type"),
            "implicated_resource": arm_res.get("implicated_resource"),
            "triage_summary": arm_res.get("triage_summary"),
            "root_cause_hypothesis": arm_res.get("root_cause_hypothesis"),
            "final_summary": arm_res.get("final_summary"),
            "diagnostic_history": arm_res.get("diagnostic_history"),
            "execution_history": arm_res.get("execution_history"),
            "verification_history": arm_res.get("verification_history"),
            "reflection_history": arm_res.get("reflection_history"),
            "hypothesis_revisions": arm_res.get("hypothesis_revisions"),
            "plan_revisions": arm_res.get("plan_revisions"),
        },
    }
    verdict = judge_llm._call(
        system_prompt,
        json.dumps(anonymized_payload, indent=2),
        BlindJudgeVerdict,
        operation="judge",
    )
    out = verdict.model_dump()
    # Deterministic enforcement: N/A when no reflection or healthy control
    if not has_reflection:
        out["reflection_effectiveness_score"] = None
    if is_healthy:
        out["root_cause_accurate"] = None
    return out


# ---------------------------------------------------------------------------
# Summary Aggregation
# ---------------------------------------------------------------------------

def _summarize_arms(scenario_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    arms = ["rules_baseline", "nova_no_reflection", "nova_full"]
    summary: Dict[str, Any] = {}
    for arm in arms:
        rows = [r["arms"][arm] for r in scenario_records if arm in r["arms"]]
        n = len(rows)
        faulty_rows = [r for r in rows if not r["is_healthy_control"]]
        healthy_rows = [r for r in rows if r["is_healthy_control"]]
        recov_rows = [r for r in rows if r["expected_recoverable"]]
        esc_rows = [r for r in rows if not r["is_healthy_control"] and r["expected_outcome"] == "ESCALATED"]
        partial_rows = [r for r in rows if r["expected_partial_db_recovery"]]
        refl_scores = [
            r["judge"]["reflection_effectiveness_score"]
            for r in rows
            if r["judge"]["reflection_effectiveness_score"] is not None
        ]
        summary[arm] = {
            "n_total": n,
            "n_faulty": len(faulty_rows),
            "n_healthy": len(healthy_rows),
            "fault_diagnosis_correctness": (
                f"{sum(1 for r in faulty_rows if r['judge']['root_cause_accurate'] is True)}/{len(faulty_rows)}"
                if faulty_rows else "N/A"
            ),
            "verified_complete_recovery": (
                f"{sum(1 for r in recov_rows if r['scores']['verified_complete_recovery'])}/{len(recov_rows)}"
                if recov_rows else "N/A"
            ),
            "verified_partial_recovery": (
                f"{sum(1 for r in partial_rows if r['scores']['verified_partial_recovery'])}/{len(partial_rows)}"
                if partial_rows else "N/A"
            ),
            "justified_escalation": (
                f"{sum(1 for r in esc_rows if r['scores']['justified_escalation'])}/{len(esc_rows)}"
                if esc_rows else "N/A"
            ),
            "healthy_non_intervention": (
                f"{sum(1 for r in healthy_rows if r['scores']['healthy_system_non_intervention'])}/{len(healthy_rows)}"
                if healthy_rows else "N/A"
            ),
            "false_resolution_count": sum(1 for r in rows if r["scores"]["false_resolution"]),
            "unnecessary_actions_total": sum(r["scores"]["unnecessary_actions"] for r in rows),
            "hypothesis_revisions_total": sum(r["scores"]["hypothesis_revision_count"] for r in rows),
            "plan_revisions_total": sum(r["scores"]["plan_revision_count"] for r in rows),
            "avg_evidence_groundedness": round(
                sum(r["judge"]["evidence_groundedness_score"] for r in rows) / max(n, 1), 2
            ),
            "avg_reflection_effectiveness": (
                round(sum(refl_scores) / len(refl_scores), 2) if refl_scores else "N/A"
            ),
            "total_llm_calls": sum(r["scores"]["llm_calls"] for r in rows),
            "total_tool_calls": sum(r["scores"]["tool_calls"] for r in rows),
            "total_tokens": sum(r["scores"]["total_tokens"] for r in rows),
            "total_latency_seconds": round(sum(r["scores"]["latency_seconds"] for r in rows), 2),
            "estimated_cost_usd": round(sum(r["scores"]["estimated_cost_usd"] for r in rows), 6),
        }
    return summary


def run_acceptance_probes(env: KestraEnvironment) -> Dict[str, Any]:
    """
    Executes two dedicated live Kestra + Gemini acceptance incidents:
      1. Contradictory evidence -> hypothesis revision -> plan change -> execution -> verified recovery.
      2. Planning critique prevents an inappropriate proposed action from executing.
    """
    _ensure_pg_running()
    _upload_flow(env, BLIND_FLOW_EPSILON)
    _upload_flow(env, BLIND_FLOW_ZETA)

    # Probe A: Contradictory diagnosis revision (pipeline_epsilon with pgdatabase stopped)
    _stop_pg()
    eid_eps = _trigger_and_wait_kestra(env, "zoomcamp_eval", "pipeline_epsilon")
    sc_eps = {
        "name": "acceptance_hypothesis_revision",
        "namespace": "zoomcamp_eval",
        "flow_id": "pipeline_epsilon",
        "neutral_symptoms_template": (
            "Kestra execution '{eid}' failed on step_01_sync_target with transient retryable error. "
            "Investigate and recover."
        ),
        "expected_recoverable": True,
        "expected_outcome": "RESOLVED",
        "expected_output": {
            "verifier": "execution_only",
            "required_tasks": ["step_01_sync_target", "step_02_commit"],
        },
    }
    res_eps = run_nova_arm(env, sc_eps, eid_eps, enable_reflection=True)
    time.sleep(1.5)

    # Probe B: Planning critique prevents inappropriate action (pipeline_zeta)
    _ensure_pg_running()
    eid_zeta = _trigger_and_wait_kestra(env, "zoomcamp_eval", "pipeline_zeta")
    sc_zeta = {
        "name": "acceptance_planning_critique",
        "namespace": "zoomcamp_eval",
        "flow_id": "pipeline_zeta",
        "neutral_symptoms_template": (
            "Kestra execution '{eid}' failed on step_01_pull_partner_api. "
            "Investigate and determine if automated remediation is possible."
        ),
        "expected_recoverable": False,
        "expected_outcome": "ESCALATED",
        "expected_output": {"verifier": "execution_only"},
    }
    res_zeta = run_nova_arm(env, sc_zeta, eid_zeta, enable_reflection=True)
    return {
        "hypothesis_revision_incident": res_eps,
        "planning_critique_incident": res_zeta,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="NOVA Milestone 13 Blind Development Evaluation")
    parser.add_argument(
        "--out",
        default="docs/m13_dev_eval_report.json",
        help="Path to write full Milestone 13 evaluation report.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling arm order.")
    parser.add_argument("--with-acceptance", action="store_true", help="Also run dedicated live acceptance probes.")
    parser.add_argument("--acceptance-only", action="store_true", help="Run only the live acceptance probes.")
    args = parser.parse_args()

    if not settings.gemini_api_key:
        raise SystemExit("GEMINI_API_KEY is not configured in environment / .env.")

    env = KestraEnvironment()
    if args.acceptance_only:
        try:
            acc = run_acceptance_probes(env)
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps({"acceptance_traces": acc}, indent=2), encoding="utf-8")
            print(f"Saved acceptance traces to {out_path}")
        finally:
            _ensure_pg_running()
        return
    judge_llm = GeminiLLMClient(api_key=settings.gemini_api_key, model_name=settings.default_model)
    rng = random.Random(args.seed)

    print("Preparing blind Kestra + PostgreSQL + DuckDB development scenarios...")
    scenarios = prepare_dev_scenarios(env)

    scenario_records: List[Dict[str, Any]] = []
    try:
        for sc in scenarios:
            print(f"\n=== Scenario: {sc['name']} (flow={sc['namespace']}/{sc['flow_id']}) ===")
            arm_order = ["rules_baseline", "nova_no_reflection", "nova_full"]
            rng.shuffle(arm_order)
            print(f"  Shuffled arm order: {arm_order}")

            arm_outputs: Dict[str, Any] = {}
            for arm_name in arm_order:
                eid = _instantiate_execution(env, sc)
                if arm_name == "rules_baseline":
                    res = run_rules_baseline(env, sc, eid)
                elif arm_name == "nova_no_reflection":
                    res = run_nova_arm(env, sc, eid, enable_reflection=False)
                else:
                    res = run_nova_arm(env, sc, eid, enable_reflection=True)

                det_scores = score_deterministic(sc, res)
                judge_scores = blind_judge_trace(judge_llm, sc, res)
                arm_outputs[arm_name] = {
                    "is_healthy_control": sc["is_healthy_control"],
                    "expected_recoverable": sc["expected_recoverable"],
                    "expected_outcome": sc["expected_outcome"],
                    "expected_partial_db_recovery": sc["expected_partial_db_recovery"],
                    "trace": res,
                    "scores": det_scores,
                    "judge": judge_scores,
                }
                print(
                    f"  [{arm_name}] status={res['status']} "
                    f"recov={det_scores['verified_complete_recovery']} "
                    f"partial={det_scores['verified_partial_recovery']} "
                    f"esc={det_scores['justified_escalation']} "
                    f"rc_acc={judge_scores['root_cause_accurate']} "
                    f"refl={judge_scores['reflection_effectiveness_score']} "
                    f"hyp_rev={det_scores['hypothesis_revision_count']} "
                    f"plan_rev={det_scores['plan_revision_count']} "
                    f"tok={det_scores['total_tokens']}"
                )
                time.sleep(1.5)

            scenario_records.append(
                {
                    "name": sc["name"],
                    "flow": f"{sc['namespace']}/{sc['flow_id']}",
                    "arm_order": arm_order,
                    "expected_outcome": sc["expected_outcome"],
                    "expected_recoverable": sc["expected_recoverable"],
                    "is_healthy_control": sc["is_healthy_control"],
                    "arms": arm_outputs,
                }
            )
    finally:
        _ensure_pg_running()

    acceptance_traces = None
    if args.with_acceptance:
        print("\n=== Running Live Acceptance Criteria Probes ===")
        try:
            acceptance_traces = run_acceptance_probes(env)
            print(
                f"  [Hypothesis Revision] status={acceptance_traces['hypothesis_revision_incident']['status']} "
                f"hyp_revs={len(acceptance_traces['hypothesis_revision_incident']['hypothesis_revisions'])} "
                f"exec={[e['tool_name'] for e in acceptance_traces['hypothesis_revision_incident']['execution_history']]}"
            )
            print(
                f"  [Planning Critique]   status={acceptance_traces['planning_critique_incident']['status']} "
                f"plan_revs={len(acceptance_traces['planning_critique_incident']['plan_revisions'])} "
                f"exec={[e['tool_name'] for e in acceptance_traces['planning_critique_incident']['execution_history']]}"
            )
        finally:
            _ensure_pg_running()

    summary = _summarize_arms(scenario_records)
    report = {
        "milestone": "M13_DEV_EVAL",
        "model": settings.default_model,
        "seed": args.seed,
        "sample_size": len(scenario_records),
        "summary": summary,
        "scenarios": scenario_records,
        "acceptance_traces": acceptance_traces,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved M13 dev evaluation report to: {out_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
