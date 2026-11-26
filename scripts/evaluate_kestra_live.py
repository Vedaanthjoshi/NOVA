"""
Milestone 12: Live Kestra + PostgreSQL Fault Injection & Agentic Evaluation Harness.

Evaluates three systems on real executing Kestra/PostgreSQL scenarios:
  1. Fixed Baseline (read execution status -> read failed task log -> restart once)
  2. NOVA without Reflection (GeminiLLMClient(..., enable_reflection=False))
  3. Full NOVA with Reflection (GeminiLLMClient(..., enable_reflection=True))

Evaluates 5 real scenarios (kept strictly hidden from NOVA observations):
  1. source_unavailable (real Kestra wget 404 failure -> appropriate escalation)
  2. warehouse_db_unavailable (real stopped pgdatabase -> restart + resume -> 35,612-row PG verify)
  3. malformed_input (real PostgreSQL CopyIn type error -> appropriate escalation)
  4. cascading_db_then_source (unseen multi-fault: DB down first -> recovered -> downstream 404 -> adaptive escalation)
  5. healthy_workflow (already-succeeded 2020-04 green taxi execution -> healthy non-intervention)

Also performs Independent LLM-as-a-Judge scoring (evidence groundedness, root cause accuracy,
tool/plan correctness, reflection quality) and writes full trajectory traces to JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from pydantic import BaseModel, Field

# Ensure src/ is importable when invoked from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nova.config import settings
from nova.environments.kestra_runner import KestraEnvironment, _find_docker_bin
from nova.graph import build_nova_graph
from nova.llm.client import GeminiLLMClient
from nova.observability import telemetry
from nova.state import ApprovalDecision, IncidentStatus
from nova.tools.kestra_tools import create_kestra_tool_runner

# ---------------------------------------------------------------------------
# Isolated Evaluation Flow Definitions (outside agent awareness)
# ---------------------------------------------------------------------------

EVAL_FLOW_SOURCE_UNAVAILABLE = """id: 04_postgres_taxi_eval_source
namespace: zoomcamp_eval
inputs:
  - id: taxi
    type: STRING
    defaults: green
  - id: year
    type: STRING
    defaults: "2021"
  - id: month
    type: STRING
    defaults: "08"
tasks:
  - id: extract
    type: io.kestra.plugin.scripts.shell.Commands
    outputFiles:
      - "*.csv"
    taskRunner:
      type: io.kestra.plugin.core.runner.Process
    commands:
      - wget -qO- https://github.com/DataTalksClub/nyc-tlc-data/releases/download/{{inputs.taxi}}/{{inputs.taxi}}_tripdata_{{inputs.year}}-{{inputs.month}}.csv.gz | gunzip > {{inputs.taxi}}_tripdata_{{inputs.year}}-{{inputs.month}}.csv
      - test -s {{inputs.taxi}}_tripdata_{{inputs.year}}-{{inputs.month}}.csv
"""

EVAL_FLOW_MALFORMED_INPUT = """id: 04_postgres_taxi_eval_malformed
namespace: zoomcamp_eval
tasks:
  - id: prepare_payload
    type: io.kestra.plugin.scripts.shell.Commands
    outputFiles:
      - "bad_green.csv"
    taskRunner:
      type: io.kestra.plugin.core.runner.Process
    commands:
      - |
        cat <<'EOF' > bad_green.csv
        VendorID,lpep_pickup_datetime,lpep_dropoff_datetime,store_and_fwd_flag,RatecodeID,PULocationID,DOLocationID,passenger_count,trip_distance,fare_amount,extra,mta_tax,tip_amount,tolls_amount,ehail_fee,improvement_surcharge,total_amount,payment_type,trip_type,congestion_surcharge
        2,2020-04-01 00:41:22,2020-04-01 01:01:53,N,1,41,24,CORRUPT_INT,1.4,12.0,0.5,0.5,0.0,0.0,,0.3,13.3,2,1,0.0
        EOF
  - id: ensure_staging_table
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
  - id: load_staging_csv
    type: io.kestra.plugin.jdbc.postgresql.CopyIn
    format: CSV
    from: "{{outputs.prepare_payload.outputFiles['bad_green.csv']}}"
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

EVAL_FLOW_CASCADING_UNSEEN = """id: 04_postgres_taxi_eval_cascade
namespace: zoomcamp_eval
tasks:
  - id: init_warehouse_check
    type: io.kestra.plugin.jdbc.postgresql.Queries
    sql: "SELECT 1;"
  - id: fetch_source_partition
    type: io.kestra.plugin.scripts.shell.Commands
    outputFiles:
      - "*.csv"
    taskRunner:
      type: io.kestra.plugin.core.runner.Process
    commands:
      - wget -qO- https://github.com/DataTalksClub/nyc-tlc-data/releases/download/green/green_tripdata_2021-08.csv.gz | gunzip > green_2021_08.csv
      - test -s green_2021_08.csv
pluginDefaults:
  - type: io.kestra.plugin.jdbc.postgresql
    values:
      url: jdbc:postgresql://pgdatabase:5432/ny_taxi
      username: root
      password: root
"""


# ---------------------------------------------------------------------------
# Independent Judge Schema
# ---------------------------------------------------------------------------

class JudgeVerdict(BaseModel):
    root_cause_accurate: bool = Field(description="Whether diagnosis matches the hidden ground truth")
    tool_selection_accurate: bool = Field(description="Whether chosen diagnostic and remediation tools were appropriate and free of hallucinated parameters")
    evidence_groundedness_score: int = Field(ge=1, le=5, description="1-5 score for how strictly conclusions were grounded in observed logs/states")
    plan_correctness_score: int = Field(ge=1, le=5, description="1-5 score for remediation/escalation plan correctness")
    reflection_effectiveness_score: int = Field(ge=1, le=5, description="1-5 score for diagnostic/planning/outcome reflection quality (3 if N/A)")
    justification: str = Field(description="Concise technical justification")


# ---------------------------------------------------------------------------
# Docker & Kestra Fault Injection Helpers (external to NOVA)
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
    r = env._request(
        "POST",
        "/flows",
        content=yaml_text,
        headers={"Content-Type": "application/x-yaml"},
    )
    if r.status_code not in (200, 201):
        # Update existing flow
        import re
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
    inputs: Optional[Dict[str, str]] = None,
    timeout: float = 90.0,
) -> str:
    files = {k: (None, v) for k, v in (inputs or {}).items()} if inputs else None
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


def prepare_scenarios(env: KestraEnvironment) -> List[Dict[str, Any]]:
    """
    Injects real faults via Kestra + Docker outside the agent and returns
    (neutral_prompt, execution_id, hidden_ground_truth) descriptors.
    """
    _ensure_pg_running()
    _upload_flow(env, EVAL_FLOW_SOURCE_UNAVAILABLE)
    _upload_flow(env, EVAL_FLOW_MALFORMED_INPUT)
    _upload_flow(env, EVAL_FLOW_CASCADING_UNSEEN)

    scenarios: List[Dict[str, Any]] = []

    # 1. Source Unavailable
    eid_src = _trigger_and_wait_kestra(
        env, "zoomcamp_eval", "04_postgres_taxi_eval_source", {"taxi": "green", "year": "2021", "month": "08"}
    )
    scenarios.append(
        {
            "name": "source_unavailable",
            "namespace": "zoomcamp_eval",
            "flow_id": "04_postgres_taxi_eval_source",
            "execution_id": eid_src,
            "neutral_symptoms": (
                f"Kestra execution '{eid_src}' in namespace 'zoomcamp_eval' failed. "
                "Investigate the failure from Kestra status and logs, and remediate only if recoverable."
            ),
            "expected_recoverable": False,
            "expected_outcome": "ESCALATED",
            "expected_root_cause": "Upstream NYC taxi CSV release for 2021-08 returned HTTP 404 / empty output in extract task; external source unavailable and not fixable by NOVA.",
            "unseen": False,
        }
    )

    # 2. Warehouse DB Unavailable (re-creatable per arm)
    scenarios.append(
        {
            "name": "warehouse_db_unavailable",
            "namespace": "zoomcamp",
            "flow_id": "04_postgres_taxi",
            "execution_id": None,  # Created fresh per arm since it recovers DB
            "neutral_symptoms_template": (
                "Kestra execution '{eid}' in namespace 'zoomcamp' failed. "
                "Investigate the failure from Kestra status and logs, and remediate if recoverable."
            ),
            "expected_recoverable": True,
            "expected_outcome": "RESOLVED",
            "expected_root_cause": "PostgreSQL warehouse dependency (pgdatabase) was offline / refused connections on port 5432 during DDL/load tasks.",
            "expected_output": {
                "taxi": "green",
                "year": "2020",
                "month": "04",
                "expected_row_count": 35612,
                "expected_distinct_ids": 35612,
                "expected_sum_total_amount": 701002.26,
            },
            "unseen": False,
        }
    )

    # 3. Malformed Input
    eid_bad = _trigger_and_wait_kestra(env, "zoomcamp_eval", "04_postgres_taxi_eval_malformed")
    scenarios.append(
        {
            "name": "malformed_input",
            "namespace": "zoomcamp_eval",
            "flow_id": "04_postgres_taxi_eval_malformed",
            "execution_id": eid_bad,
            "neutral_symptoms": (
                f"Kestra execution '{eid_bad}' in namespace 'zoomcamp_eval' failed. "
                "Investigate the failure from Kestra status and logs, and remediate only if recoverable."
            ),
            "expected_recoverable": False,
            "expected_outcome": "ESCALATED",
            "expected_root_cause": "Malformed CSV input ('CORRUPT_INT' for integer column passenger_count) caused PostgreSQL CopyIn failure while pgdatabase is healthy.",
            "unseen": False,
        }
    )

    # 4. Cascading Unseen: DB unavailable followed by downstream 404 source failure
    scenarios.append(
        {
            "name": "cascading_db_then_source_unseen",
            "namespace": "zoomcamp_eval",
            "flow_id": "04_postgres_taxi_eval_cascade",
            "execution_id": None,  # Fresh per arm
            "neutral_symptoms_template": (
                "Kestra execution '{eid}' in namespace 'zoomcamp_eval' failed. "
                "Investigate the failure from Kestra status and logs, and remediate if possible."
            ),
            "expected_recoverable": False,  # Overall workflow cannot complete because downstream source is 404 after DB recovery
            "expected_outcome": "ESCALATED",
            "expected_partial_db_recovery": True,
            "expected_root_cause": "Initial failure was pgdatabase offline on init_warehouse_check; after restarting pgdatabase and resuming, downstream fetch_source_partition failed with HTTP 404 external source unavailable.",
            "unseen": True,
        }
    )

    # 5. Healthy Workflow
    eid_healthy = _trigger_and_wait_kestra(
        env, "zoomcamp", "04_postgres_taxi", {"taxi": "green", "year": "2020", "month": "04"}
    )
    scenarios.append(
        {
            "name": "healthy_workflow",
            "namespace": "zoomcamp",
            "flow_id": "04_postgres_taxi",
            "execution_id": eid_healthy,
            "neutral_symptoms": (
                f"Automated monitor raised an alert on Kestra execution '{eid_healthy}' in namespace 'zoomcamp'. "
                "Verify execution state and intervene only if an active failure exists."
            ),
            "expected_recoverable": False,
            "expected_outcome": "ESCALATED",  # Healthy non-intervention
            "expected_healthy_non_intervention": True,
            "expected_root_cause": "False-positive alert: all tasks in execution are already in SUCCESS state and no intervention is needed.",
            "unseen": False,
        }
    )

    return scenarios


def _instantiate_scenario_execution(env: KestraEnvironment, sc: Dict[str, Any]) -> str:
    """For mutating scenarios, resets fault state cleanly before each evaluation arm."""
    name = sc["name"]
    if name == "warehouse_db_unavailable":
        _ensure_pg_running()
        _clear_green_2020_04_partition()
        _stop_pg()
        return _trigger_and_wait_kestra(
            env, "zoomcamp", "04_postgres_taxi", {"taxi": "green", "year": "2020", "month": "04"}
        )
    if name == "cascading_db_then_source_unseen":
        _ensure_pg_running()
        _stop_pg()
        return _trigger_and_wait_kestra(env, "zoomcamp_eval", "04_postgres_taxi_eval_cascade")
    return sc["execution_id"]


# ---------------------------------------------------------------------------
# Arm 1: Fixed Baseline (Status -> Failed Log -> Restart Once)
# ---------------------------------------------------------------------------

def run_fixed_baseline(env: KestraEnvironment, sc: Dict[str, Any], exec_id: str) -> Dict[str, Any]:
    t0 = time.perf_counter()
    env.bind_execution(exec_id)
    runner = create_kestra_tool_runner(env)

    diag_calls = 0
    remed_calls = 0
    tool_trace: List[Dict[str, Any]] = []

    st_res = runner.get_tool("get_workflow_status").run(execution_id=exec_id)
    diag_calls += 1
    tool_trace.append({"tool": "get_workflow_status", "success": st_res.success})

    failed_jobs = st_res.data.get("failed_jobs", [])
    if not failed_jobs:
        # Healthy workflow — baseline stops
        return {
            "status": "ESCALATED",
            "healthy_non_intervention": True,
            "verified_recovery": False,
            "diag_calls": diag_calls,
            "remed_calls": remed_calls,
            "llm_calls": 0,
            "total_tokens": 0,
            "latency_seconds": round(time.perf_counter() - t0, 3),
            "trace": tool_trace,
        }

    # Read failed task log then blindly restart execution once
    log_res = runner.get_tool("read_job_logs").run(job_id=failed_jobs[0], execution_id=exec_id)
    diag_calls += 1
    tool_trace.append({"tool": "read_job_logs", "job_id": failed_jobs[0], "success": log_res.success})

    res_res = runner.get_tool("resume_workflow").run(execution_id=exec_id)
    remed_calls += 1
    tool_trace.append({"tool": "resume_workflow", "success": res_res.success})

    verified = False
    if res_res.success and sc.get("expected_output"):
        ok, _, _ = env.verify_output(sc["expected_output"])
        verified = ok

    return {
        "status": "RESOLVED" if (res_res.success and (verified or not sc.get("expected_output"))) else "FAILED",
        "healthy_non_intervention": False,
        "verified_recovery": verified,
        "diag_calls": diag_calls,
        "remed_calls": remed_calls,
        "llm_calls": 0,
        "total_tokens": 0,
        "latency_seconds": round(time.perf_counter() - t0, 3),
        "trace": tool_trace,
    }


# ---------------------------------------------------------------------------
# Arms 2 & 3: NOVA Graph Runner (with or without reflection)
# ---------------------------------------------------------------------------

def run_nova_arm(
    env: KestraEnvironment,
    sc: Dict[str, Any],
    exec_id: str,
    enable_reflection: bool,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    env.bind_execution(exec_id)
    inc_id = f"INC_KESTRA_{sc['name'][:8].upper()}_{'REFL' if enable_reflection else 'NOREFL'}_{uuid.uuid4().hex[:4].upper()}"
    telemetry.reset_incident(inc_id)

    llm = GeminiLLMClient(
        api_key=settings.gemini_api_key,
        model_name=settings.default_model,
        enable_reflection=enable_reflection,
    )
    checkpointer = MemorySaver()
    graph = build_nova_graph(env, checkpointer=checkpointer, require_approval=True, llm=llm)
    cfg = {"configurable": {"thread_id": inc_id}}

    symptoms = sc.get("neutral_symptoms") or sc["neutral_symptoms_template"].format(eid=exec_id)
    initial_state: NovaGraphState = {
        "incident_id": inc_id,
        "target_service": "UNKNOWN",
        "raw_symptoms": symptoms,
        "status": IncidentStatus.TRIGGERED,
        "workflow_id": f"{sc['namespace']}/{sc['flow_id']}",
        "workflow_run_id": exec_id,
        "max_investigation_steps": 7,
        "max_retries": 2,
        "expected_output": sc.get("expected_output"),
    }

    graph.invoke(initial_state, cfg)

    # Approve local HITL interrupts cleanly (preserving existing local approval mechanism)
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
    status_val = str(final_vals.get("status").value if hasattr(final_vals.get("status"), "value") else final_vals.get("status"))

    verified_recovery = False
    if status_val == "RESOLVED" and sc.get("expected_recoverable"):
        verified_recovery = bool(verif_hist and verif_hist[-1].passed)

    return {
        "incident_id": inc_id,
        "execution_id": exec_id,
        "status": status_val,
        "failure_type": final_vals.get("failure_type"),
        "implicated_resource": final_vals.get("implicated_resource"),
        "triage_summary": final_vals.get("triage_summary"),
        "root_cause_hypothesis": final_vals.get("root_cause_hypothesis"),
        "final_summary": final_vals.get("final_summary"),
        "retry_count": final_vals.get("retry_count", 0),
        "verified_recovery": verified_recovery,
        "healthy_non_intervention": (
            bool(sc.get("expected_healthy_non_intervention"))
            and status_val == "ESCALATED"
            and len(exec_hist) == 0
        ),
        "partial_db_restarted": any(e.tool_name == "restart_service" and e.success for e in exec_hist),
        "diagnostic_history": [d.model_dump() if hasattr(d, "model_dump") else d for d in diag_hist],
        "execution_history": [e.model_dump() if hasattr(e, "model_dump") else e for e in exec_hist],
        "verification_history": [v.model_dump() if hasattr(v, "model_dump") else v for v in verif_hist],
        "reflection_history": refl_hist,
        "llm_calls": metrics.get("llm_request_count", 0),
        "diag_calls": metrics.get("diagnostic_tool_count", 0),
        "remed_calls": metrics.get("remediation_tool_count", 0),
        "input_tokens": metrics.get("input_tokens", 0),
        "output_tokens": metrics.get("output_tokens", 0),
        "total_tokens": metrics.get("total_tokens", 0),
        "latency_seconds": round(time.perf_counter() - t0, 3),
    }


def judge_arm_result(
    judge_llm: GeminiLLMClient,
    sc: Dict[str, Any],
    arm_result: Dict[str, Any],
) -> JudgeVerdict:
    system_prompt = """You are an independent SRE evaluator scoring an autonomous incident agent trace against hidden ground truth.
Score strictly from the observed trajectory, tool outputs, diagnosis, actions, and reflections.
Output valid JSON matching:
{
  "root_cause_accurate": boolean,
  "tool_selection_accurate": boolean,
  "evidence_groundedness_score": 1..5,
  "plan_correctness_score": 1..5,
  "reflection_effectiveness_score": 1..5,
  "justification": string
}"""
    user_prompt = json.dumps(
        {
            "scenario": sc["name"],
            "hidden_ground_truth": {
                "expected_root_cause": sc["expected_root_cause"],
                "expected_outcome": sc["expected_outcome"],
                "expected_recoverable": sc["expected_recoverable"],
            },
            "agent_trace": {
                "status": arm_result.get("status"),
                "failure_type": arm_result.get("failure_type"),
                "implicated_resource": arm_result.get("implicated_resource"),
                "triage_summary": arm_result.get("triage_summary"),
                "root_cause_hypothesis": arm_result.get("root_cause_hypothesis"),
                "final_summary": arm_result.get("final_summary"),
                "diagnostic_history": arm_result.get("diagnostic_history"),
                "execution_history": arm_result.get("execution_history"),
                "verification_history": arm_result.get("verification_history"),
                "reflection_history": arm_result.get("reflection_history"),
            },
        },
        indent=2,
    )
    return judge_llm._call(system_prompt, user_prompt, JudgeVerdict, operation="judge")


def main() -> None:
    parser = argparse.ArgumentParser(description="NOVA Milestone 12 Live Kestra Evaluation")
    parser.add_argument(
        "--out",
        default=r"C:\Users\vedaa\.gemini\antigravity\brain\6a144adc-9df2-47f1-88b5-a1bfb7227698\scratch\kestra_eval_report.json",
        help="Path to write full evaluation JSON report.",
    )
    args = parser.parse_args()

    if not settings.gemini_api_key:
        raise SystemExit("GEMINI_API_KEY is not configured in environment / .env.")

    env = KestraEnvironment()
    judge_llm = GeminiLLMClient(api_key=settings.gemini_api_key, model_name=settings.default_model)

    print("Preparing real Kestra + PostgreSQL failure scenarios...")
    scenarios = prepare_scenarios(env)

    report: Dict[str, Any] = {
        "model": settings.default_model,
        "scenarios": [],
    }

    try:
        for sc in scenarios:
            print(f"\n=== Scenario: {sc['name']} (unseen={sc['unseen']}) ===")

            eid_base = _instantiate_scenario_execution(env, sc)
            base_res = run_fixed_baseline(env, sc, eid_base)
            print(f"  [Baseline] status={base_res['status']} verified={base_res['verified_recovery']} time={base_res['latency_seconds']}s")

            eid_norefl = _instantiate_scenario_execution(env, sc)
            norefl_res = run_nova_arm(env, sc, eid_norefl, enable_reflection=False)
            norefl_judge = judge_arm_result(judge_llm, sc, norefl_res).model_dump()
            print(
                f"  [NOVA No-Refl] status={norefl_res['status']} verified={norefl_res['verified_recovery']} "
                f"tokens={norefl_res['total_tokens']} judge_rc={norefl_judge['root_cause_accurate']}"
            )
            time.sleep(1.5)

            eid_full = _instantiate_scenario_execution(env, sc)
            full_res = run_nova_arm(env, sc, eid_full, enable_reflection=True)
            full_judge = judge_arm_result(judge_llm, sc, full_res).model_dump()
            print(
                f"  [NOVA Full]    status={full_res['status']} verified={full_res['verified_recovery']} "
                f"tokens={full_res['total_tokens']} judge_rc={full_judge['root_cause_accurate']}"
            )
            time.sleep(1.5)

            report["scenarios"].append(
                {
                    "name": sc["name"],
                    "unseen": sc["unseen"],
                    "expected_outcome": sc["expected_outcome"],
                    "expected_recoverable": sc["expected_recoverable"],
                    "baseline": base_res,
                    "nova_no_reflection": {**norefl_res, "judge": norefl_judge},
                    "nova_full": {**full_res, "judge": full_judge},
                }
            )
    finally:
        # Leave pgdatabase healthy with clean 2020-04 green taxi output at the end
        _ensure_pg_running()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved full evaluation report to: {out_path}")


if __name__ == "__main__":
    main()
