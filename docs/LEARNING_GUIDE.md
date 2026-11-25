# NOVA Technical Ownership & Learning Guide

This guide is designed to help you deeply understand **how NOVA actually works in code**, trace every line of execution across processes and layers, defend architectural trade-offs in an enterprise technical interview (such as at BMC Software), and practice modifying the codebase independently.

---

## 1. Repository Structure

```text
c:\NOVA\
├── .env.example                    # Safe configuration template (no secrets)
├── .gitignore                      # Excludes .env, .venv, *.sqlite, *.db, mock_state/
├── AGENTS.md                       # Permanent engineering principles
├── pyproject.toml                  # Package metadata, runtime deps, pytest config
├── README.md                       # Project overview, setup, results, boundaries
├── docs/
│   ├── ARCHITECTURE.md             # LangGraph topology & architectural deep dives
│   ├── DECISIONS.md                # Architecture Decision Records (ADRs 001–006)
│   ├── DEVELOPMENT_PLAN.md         # Original 3-week milestone roadmap
│   ├── INTERVIEW_PREP.md           # Concise whiteboard pitch & Q&A
│   ├── LEARNING_GUIDE.md           # This technical ownership & study guide
│   ├── PROGRESS.md                 # Milestone log & verified test counts
│   └── PROJECT_OVERVIEW.md         # Problem framing & BMC alignment
├── scripts/
│   ├── start_nova.py               # Unified local FastAPI + Streamlit launcher
│   ├── test_gemini_connection.py   # Standalone Gemini JSON-mode smoke check
│   ├── evaluate_gemini_live.py     # 10-scenario isolated LLM evaluation harness
│   ├── evaluate_adversarial.py     # 8-scenario compiled-graph stress test
│   └── demonstrate_milestone9.py   # 5-phase API + SQLite walkthrough script
├── src/nova/
│   ├── config.py                   # Pydantic Settings (settings singleton)
│   ├── state.py                    # NovaState, NovaGraphState, domain records, RBAC roles
│   ├── graph.py                    # StateGraph wiring & conditional edge routers
│   ├── observability.py            # Structured JSON logging, correlation IDs, telemetry
│   ├── environments/
│   │   └── mock_workflow_runner.py # Thread-safe disk-backed batch pipeline simulator
│   ├── llm/
│   │   ├── schemas.py              # Pydantic contracts for LLM outputs
│   │   └── client.py               # BaseLLMClient, MockLLMClient, GeminiLLMClient
│   ├── tools/
│   │   ├── base.py                 # BaseTool ABC, ToolResult, parameter validation
│   │   ├── workflow_tools.py       # 8 concrete tools (5 READ_ONLY, 1 LOW_RISK, 2 DESTRUCTIVE)
│   │   └── runner.py               # ToolRunner whitelist registry & execution timer
│   ├── nodes/
│   │   ├── triage.py               # Autonomous read-only investigation loop
│   │   ├── planner.py              # Remediation plan synthesis & action_hash generation
│   │   ├── executor.py             # Ground-truth risk gate, HITL interrupt(), tool execution
│   │   ├── verifier.py             # Independent downstream post-condition check
│   │   └── replanner.py            # Evidence-based re-diagnosis after verification failure
│   ├── storage/
│   │   ├── sqlite_saver.py         # Granular row-level LangGraph SqliteSaver (WAL mode)
│   │   └── incident_store.py       # Durable incident_metadata, approval_audit, telemetry
│   ├── api/
│   │   ├── schemas.py              # FastAPI request/response Pydantic DTOs
│   │   ├── service.py              # IncidentService (ThreadPoolExecutor, replay/RBAC guards)
│   │   └── main.py                 # FastAPI endpoints & exception handlers
│   └── ui/
│       ├── api_client.py           # NovaApiClient (pure httpx REST client)
│       ├── components/theme.py     # Enterprise dark CSS & HTML status/risk badges
│       └── app.py                  # 4-view Streamlit Operator Dashboard
└── tests/
    ├── unit/                       # Fast deterministic tests (state, tools, LLM, observability)
    └── integration/                # Graph, HITL, security, persistence, API, UI, hardening tests
```

---

## 2. Recommended Source-Code Reading Order

Read the codebase from **innermost contracts outward to HTTP/UI**:

1. `src/nova/config.py` — How `.env` settings are loaded into `settings`.
2. `src/nova/state.py` — Every enum (`IncidentStatus`, `RiskLevel`), domain record (`PlannedAction`, `ApprovalDecision`, `ExecutionRecord`, `VerificationRecord`, `DiagnosticRecord`), and `NovaGraphState` reducers (`operator.add`).
3. `src/nova/environments/mock_workflow_runner.py` — How the 4-job DAG (`JOB_01_EXTRACT` → `JOB_04_REPORT`), lockfiles, logs, and `workflow_state.json` behave on disk under `threading.RLock`.
4. `src/nova/tools/base.py`, `workflow_tools.py`, `runner.py` — How `BaseTool.run()` validates Pydantic args and how `ToolRunner.execute()` wraps execution into `ExecutionRecord`.
5. `src/nova/llm/schemas.py` and `src/nova/llm/client.py` — How `MockLLMClient` and `GeminiLLMClient` share identical Pydantic return contracts and retry/rate-limit handling.
6. `src/nova/nodes/triage.py`, `planner.py`, `executor.py`, `verifier.py`, `replanner.py` — How each node reads/returns partial state dicts and enforces deterministic safety boundaries.
7. `src/nova/graph.py` — How `build_nova_graph()` wires the 5 nodes and 5 conditional routers (`route_after_*`).
8. `src/nova/storage/sqlite_saver.py` and `incident_store.py` — How checkpoints (`checkpoints`, `checkpoint_blobs`, `checkpoint_writes`) and operational metadata/audits persist across restarts.
9. `src/nova/observability.py` — How `incident_context` (`contextvars`), `redact_secrets`, `StructuredJsonFormatter`, and `TelemetryRegistry` work without external infrastructure.
10. `src/nova/api/schemas.py`, `service.py`, `main.py` — How `IncidentService` runs graphs asynchronously, enforces duplicate/replay protection, and exposes REST endpoints.
11. `src/nova/ui/api_client.py` and `app.py` — How Streamlit polls FastAPI and submits operator decisions over HTTP.

---

## 3. Important Classes and Functions

| Symbol | File | Purpose |
|---|---|---|
| `NovaGraphState` | [`state.py`](../src/nova/state.py) | `TypedDict` state passed between LangGraph nodes; list fields use `Annotated[List[...], operator.add]` so nodes append rather than overwrite history. |
| `MockWorkflowRunner` | [`mock_workflow_runner.py`](../src/nova/environments/mock_workflow_runner.py) | Thread-safe disk simulator (`RLock` + atomic `os.replace`) managing `workflow_state.json`, job logs, service health, and lockfiles (`is_lock_active()`). |
| `BaseTool` / `ToolRunner` | [`base.py`](../src/nova/tools/base.py) / [`runner.py`](../src/nova/tools/runner.py) | Whitelist registry enforcing `args_schema` Pydantic validation before `_run()`, timing execution, and returning standardized `ExecutionRecord`. |
| `ClearLockfileTool` | [`workflow_tools.py`](../src/nova/tools/workflow_tools.py) | `DESTRUCTIVE` tool with idempotent no-op if already absent AND active worker lockfile protection (`is_lock_active`). |
| `BaseLLMClient` / `GeminiLLMClient` | [`client.py`](../src/nova/llm/client.py) | Structured JSON-mode LLM client with exponential/parsed `429` retry backoff and telemetry recording. |
| `build_nova_graph()` | [`graph.py`](../src/nova/graph.py) | Compiles the 5-node `StateGraph` with injected `env`, `tool_runner`, `checkpointer`, `require_approval`, and `llm`. |
| `create_executor_node()` | [`executor.py`](../src/nova/nodes/executor.py) | Looks up registry ground-truth `RiskLevel`, emits deterministic `approval_request_id` + `interrupt()`, verifies `ApprovalDecision` bindings/RBAC on resume, and advances `current_action_index`. |
| `SqliteSaver` | [`sqlite_saver.py`](../src/nova/storage/sqlite_saver.py) | Custom `BaseCheckpointSaver` persisting checkpoints, channel blobs, and `__interrupt__` writes in SQLite WAL mode. |
| `IncidentMetadataStore` | [`incident_store.py`](../src/nova/storage/incident_store.py) | Persists `incident_metadata`, `approval_audit` (for replay defense & compliance history), and `incident_telemetry` across backend restarts. |
| `IncidentService` | [`service.py`](../src/nova/api/service.py) | Manages background `ThreadPoolExecutor`, non-blocking `trigger_incident`, `get_incident_state`, `submit_approval` (RBAC + 409 conflict/replay guards), and restart recovery. |
| `TelemetryRegistry` | [`observability.py`](../src/nova/observability.py) | Thread-safe counters for LLM calls/retries/rate-limits, diagnostic vs. remediation tools, durations, and terminal reasons. |

---

## 4. End-to-End Execution Trace of One Incident

Let's trace **`STALE_LOCKFILE` with `target_service="UNKNOWN"`**:

1. **Scenario Setup (`POST /scenarios/inject`)**:
   - `main.py:inject_scenario()` calls `service.env.inject_scenario(ScenarioType.STALE_LOCKFILE)` (writes `locks/transform_job.lock` with `STATUS=ORPHANED_PROCESS` and `workflow_state.json`), then `service.env.run_workflow()`.
   - `JOB_01_EXTRACT` succeeds; `JOB_02_TRANSFORM` detects `transform_job.lock`, writes `FATAL` to `logs/JOB_02_TRANSFORM.log`, sets `status=FAILED`, and halts before `JOB_03_LOAD`.

2. **Incident Submission (`POST /incidents/trigger`)**:
   - FastAPI validates `IncidentTriggerRequest` and calls `IncidentService.trigger_incident()`.
   - `IncidentService` generates `incident_id` (e.g., `INC_92899795`), writes initial metadata to `incident_metadata` in SQLite, resets `telemetry`, submits `_execute()` to `ThreadPoolExecutor`, and immediately returns HTTP `202 Accepted`.

3. **Autonomous Investigation (`triage_node` loop)**:
   - Worker enters `with incident_context(incident_id):` and calls `self.graph.invoke(initial_state, thread_config)`.
   - **Step 1**: `target_service == "UNKNOWN"`. `llm.investigate()` returns `DIAGNOSTIC_ACTION(tool_name="get_workflow_status")`. `triage_node` verifies `tool.risk_level == RiskLevel.READ_ONLY`, runs it, appends `DiagnosticRecord` + evidence, sets `status=TRIAGING`. `route_after_triage` loops back to `triage`.
   - **Step 2**: `llm.investigate()` sees `JOB_02_TRANSFORM: status=FAILED` and requests `read_job_logs(job_id="JOB_02_TRANSFORM")`. `triage_node` runs it and loops.
   - **Step 3**: `llm.investigate()` sees `transform_job.lock` in the log and requests `inspect_lockfiles(lock_name="transform_job.lock")` to confirm presence and non-active ownership.
   - **Step 4**: `llm.investigate()` returns `DIAGNOSIS_READY` with `failure_type="STALE_LOCKFILE"`, `implicated_resource="transform_job.lock"`, `is_recoverable=True`. `triage_node` updates `target_service="JOB_02_TRANSFORM"`, `status=PLANNING`. `route_after_triage` routes to `planner`.

4. **Remediation Planning (`planner_node`)**:
   - `planner_node` builds `TriageResult` from typed state fields and calls `llm.plan()`.
   - `_schema_to_domain()` validates each `PlannedActionSchema`, assigns `action_id="ACT_01_01"`, and computes deterministic SHA256 `action_hash`. Sets `status=EXECUTING` and routes to `executor`.

5. **HITL Interrupt & Persistence (`executor_node`)**:
   - `executor_node` ignores `action.risk_level` and queries `runner.get_tool("clear_lockfile").risk_level` → `RiskLevel.DESTRUCTIVE`.
   - Computes deterministic `approval_request_id = f"APPR_{incident_id}_{action.action_id}_{action_hash[:8]}"` and calls `interrupt(approval_payload)`.
   - LangGraph pauses execution and `SqliteSaver` persists the checkpoint + `__interrupt__` row in `checkpoint_writes`.
   - When Streamlit polls `GET /incidents/{id}/state`, `IncidentService.get_incident_state()` sees the pending interrupt and returns `status="AWAITING_APPROVAL"` with `pending_approval`.

6. **Operator Approval & Resumption (`POST /incidents/{id}/approve`)**:
   - Operator clicks **Authorize Action** in Streamlit → `NovaApiClient.submit_approval()` sends `ApprovalSubmitRequest` (`approved=True`, `approver_id`, `approver_role="SRE_LEAD"`, `action_id`, `action_hash`).
   - Under `self._lock`, `IncidentService.submit_approval()` verifies RBAC, ensures no worker future is active, verifies `action_id` and `action_hash` match the pending interrupt, checks `approval_audit` to prevent replay, inserts `APPROVED_RESUMED` into `approval_audit`, and submits `_resume()` calling `self.graph.invoke(Command(resume=decision), thread_config)`.
   - LangGraph re-enters `executor_node` up to `interrupt()`, receives `decision`, re-verifies `approved is True`, `action_id`, `incident_id`, `action_hash`, and `approver_role`, then executes `runner.execute(action)` → `ClearLockfileTool._run()` verifies `is_lock_active()` is `False` and deletes `transform_job.lock`.
   - `executor_node` appends `ExecutionRecord` (with `approved_by`, `approver_role`, `approval_id`) and sets `status=VERIFYING`.

7. **Independent Post-Condition Verification (`verifier_node`)**:
   - `verifier_node` calls `env.run_workflow()`. All 4 jobs (`JOB_01_EXTRACT` through `JOB_04_REPORT`) run to `JobStatus.SUCCESS`.
   - Appends `VerificationRecord(passed=True)` and sets `status=RESOLVED`. `route_after_verifier` routes to `END`.
   - Worker `finally` block persists final telemetry to `incident_telemetry` in SQLite.

---

## 5. Data Flow Between Components

```text
Streamlit (app.py)
  │  HTTP JSON (IncidentTriggerRequest / ApprovalSubmitRequest)
  ▼
FastAPI (main.py) ──► IncidentService (service.py)
                         │  Initial NovaGraphState / Command(resume=ApprovalDecision)
                         ▼
                      LangGraph (graph.py)
                         ├── triage_node ──► BaseLLMClient.investigate() + READ_ONLY Tools
                         ├── planner_node ──► BaseLLMClient.plan() ──► List[PlannedAction]
                         ├── executor_node ──► interrupt() ──► ToolRunner.execute() ──► MockWorkflowRunner
                         ├── verifier_node ──► MockWorkflowRunner.run_workflow() ──► VerificationRecord
                         └── replanner_node ──► BaseLLMClient.replan() ──► Updated diagnosis + retry_count+1
                         │
                         ▼
                      SqliteSaver (checkpoints) + IncidentMetadataStore (metadata, audits, telemetry)
```

---

## 6. Architectural Trade-Offs

1. **Single Cyclic `StateGraph` vs. Multi-Agent Conversational Swarm**:
   - *Choice*: Single graph with 5 specialized nodes.
   - *Trade-off*: Lower flexibility for open-ended brainstorming, but deterministic transitions, lower token cost, native `interrupt()`, and clean auditability.
2. **FastAPI + Streamlit Separation vs. Direct LangGraph Import in Streamlit**:
   - *Choice*: Streamlit communicates strictly via HTTP (`NovaApiClient`).
   - *Trade-off*: Extra API/schema layer, but prevents Streamlit script reruns from interrupting background workers or locking SQLite, and allows CLI/external integrations.
3. **Custom Row-Level `SqliteSaver` + `IncidentMetadataStore` vs. `MemorySaver` or Postgres/Redis**:
   - *Choice*: Local SQLite WAL database.
   - *Trade-off*: Single-host file lock limits horizontal scaling, but supports zero-config local persistence and cross-process HITL recovery.
4. **Simulated `run_workflow()` Oracle vs. Production Passive Probing**:
   - *Choice*: Re-running `MockWorkflowRunner.run_workflow()` in `verifier_node`.
   - *Trade-off*: Great deterministic proof inside a local simulator; unsafe in production without idempotent replay or non-mutating health probes.
5. **Self-Asserted Local RBAC vs. OIDC/JWT Authentication**:
   - *Choice*: Role whitelist validation (`ALLOWED_APPROVER_ROLES`) on operator-supplied payload fields.
   - *Trade-off*: Demonstrates RBAC enforcement and audit logging mechanics cleanly for a local demo, but is **not** a security boundary against untrusted network callers without an IdP.

---

## 7. Important Tests and What They Prove

| Test File | What It Proves |
|---|---|
| `tests/unit/test_state.py` | Pydantic state serialization and `operator.add` reducer preservation. |
| `tests/unit/test_tools.py` | Tool whitelist enforcement (`UnknownToolError`) and Pydantic argument validation (`exit_code=126`). |
| `tests/unit/test_llm_client.py` | Structured schemas, planner helper priority (`SERVICE_OFFLINE` over `clear_lockfile` substring), prompt boundaries. |
| `tests/unit/test_autonomous_investigation.py` | Unknown-service investigation loop, `READ_ONLY` enforcement, unproductive loop detection, budget exhaustion. |
| `tests/unit/test_evidence_replanning.py` | Replanner revises hypothesis on action failure instead of assuming upstream success; escalates corruption. |
| `tests/unit/test_observability.py` | Secret redaction (`AIza...`, `Bearer ...`), `incident_id` correlation IDs, Gemini 429 retry/rate-limit telemetry. |
| `tests/integration/test_graph_execution.py` | End-to-end single-fault, data-corruption escalation, cascading replanning, and `max_retries=0` exhaustion. |
| `tests/integration/test_hitl_checkpoints.py` | Pause before destructive tools, resume on approval, halt on rejection, reload across `SqliteSaver` instances. |
| `tests/integration/test_security_hardening.py` | Deceptive LLM `READ_ONLY` risk override, `action_id`/`incident_id` binding mismatches, RBAC rejection, fail-closed unknown tool. |
| `tests/integration/test_persistence_hardening.py` | True separate-OS-PID `subprocess` resume against `SqliteSaver` and relational 3-table schema inspection. |
| `tests/integration/test_multistep_execution.py` | Sequential multi-step index advancement, halt on step failure, intermediate HITL pause/rejection between steps. |
| `tests/integration/test_idempotency_and_recovery.py` | Deterministic SHA256 `action_hash` and idempotent `clear_lockfile` / `restart_service` behavior. |
| `tests/integration/test_api_gateway.py` & `test_ui_gateway.py` | Non-blocking `202` trigger, `403` RBAC, `400` mismatch, `NovaApiClient` HTTP transport contracts. |
| `tests/integration/test_milestone10_hardening.py` | Active lockfile guard, false-positive non-intervention, `409` duplicate/replay rejection, backend restart with history retrieval, LLM failure `FAILED` transition, concurrent simulator thread safety. |
| `tests/integration/test_streamlit_ui.py` | Headless `AppTest` execution of `src/nova/ui/app.py` widget flows, refresh survival, and HITL buttons. |

---

## 8. Likely Technical Interview Questions

1. **Why use LangGraph instead of a `while` loop or a multi-agent framework?**
   - Explicit typed state, `operator.add` history reducers, deterministic conditional routing, and native `interrupt()` checkpointing across process restarts without conversational drift or extra LLM hops.
2. **How do you guarantee the LLM cannot bypass Human-in-the-Loop approval?**
   - `executor_node` discards the LLM-generated `PlannedAction.risk_level` and looks up `runner.get_tool(action.tool_name).risk_level`. Unknown tools fail closed to `ESCALATED`.
3. **Why is tool exit code `0` insufficient in operations automation?**
   - A restart command returning `0` only proves the command exited cleanly, not that the service stays healthy or downstream batch jobs pass. `verifier_node` decouples execution from independent post-condition verification.
4. **What happens if the backend restarts while an incident is awaiting approval?**
   - `SqliteSaver` persists the checkpoint and `__interrupt__` payload in WAL SQLite, `IncidentMetadataStore` persists the incident catalog and `approval_audit` log, and a restarted `IncidentService` hydrates metadata and resumes via `Command(resume=...)`.
5. **What happens if the backend crashes *mid-node* (not at an interrupt)?**
   - Because LangGraph checkpoints at superstep boundaries, the in-flight node is not resumed automatically; `IncidentService.get_incident_state()` detects a non-terminal state without an active worker (`interrupted_by_restart=True`) so operators see the interrupted state instead of silent hanging.
6. **How do you stop an operator from double-clicking Approve or replaying an old approval?**
   - `IncidentService.submit_approval()` acquires an `RLock`, rejects concurrent in-flight tasks with `409 Conflict`, checks `approval_audit` for already-consumed `(incident_id, action_id, approval_request_id)` tuples, and records consumption before launching `_resume()`.
7. **How does NOVA prevent an agent from deleting an active lockfile or looping forever during diagnosis?**
   - `triage_node` enforces `RiskLevel.READ_ONLY`, compares `(tool_name, parameters)` against `diagnostic_history` for loop detection, and bounds steps via `max_investigation_steps`. Additionally, `ClearLockfileTool` calls `env.is_lock_active()` to block deleting `STATUS=ACTIVE_WORKER` locks at execution time.
8. **How do you test an LLM-based system deterministically in CI?**
   - Dependency injection of `BaseLLMClient`: CI runs 100% deterministically against `MockLLMClient` returning the exact same Pydantic schemas, while `scripts/evaluate_gemini_live.py` and `evaluate_adversarial.py` benchmark live Gemini separately.
9. **What are the biggest gaps between NOVA today and a production BMC Control-M integration?**
   - Real OIDC/JWT auth + cryptographic signing, replacing `MockWorkflowRunner` with Control-M/service health adapters, replacing `env.run_workflow()` verification with non-mutating post-condition probes, and per-incident workspace isolation.
10. **How did you handle Gemini API rate limits and secret safety?**
    - `GeminiLLMClient._call()` parses retry delays on `429`/`ResourceExhausted`, logs backoff telemetry via `observability.py`, scrubs `AIza...`/`Bearer ...` tokens in `redact_secrets()`, and `IncidentService` catches terminal LLM failures to transition the incident to `FAILED` instead of leaving it stuck.

---

## 9. Small Independent Coding Exercises

1. **Add a Read-Only Disk Space Diagnostic Tool**:
   - Subclass `BaseTool` in `src/nova/tools/workflow_tools.py` with `RiskLevel.READ_ONLY`, register it in `create_default_tool_runner()`, expose it in `AVAILABLE_TOOLS_FOR_LLM` (`planner.py`), and write a unit test in `tests/unit/test_tools.py`.
2. **Add an Active-Worker Override Flag vs. Policy Block**:
   - Inspect how `ClearLockfileTool` uses `is_lock_active()`. Write a test verifying what happens if a lockfile contains both `STATUS=ACTIVE_WORKER` and `ORPHANED` vs. `HEARTBEAT=RECENT`.
3. **Add a `/telemetry` Summary Endpoint**:
   - Expose `telemetry.get_summary()` and per-incident `telemetry.get_incident_metrics(incident_id)` via a dedicated FastAPI route in `src/nova/api/main.py` and verify it in `tests/integration/test_api_gateway.py`.
4. **Simulate a Mid-Plan Step Failure with Replanning**:
   - Create an integration test where a 2-step plan succeeds on Step 1 (`inspect_lockfiles`), fails on Step 2 (`clear_lockfile` due to active lock), routes through `verifier` → `replanner`, and verifies `replanner_node` sets `diagnosis_revision_needed` rather than assuming Step 2 succeeded.
5. **Convert `verifier_node` to Passive-First Verification**:
   - Modify `verifier_node` so it runs `env.check_service_status()` and checks `env.locks_dir` *before* calling `env.run_workflow()`, skipping `run_workflow()` if a precondition is still visibly broken.
