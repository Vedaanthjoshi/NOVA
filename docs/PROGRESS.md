# Project Progress Tracker: NOVA

**Current Date**: 2026-11-26  
**Current Phase**: Milestone 12 — Real Workflow Integration & Agentic Intelligence (Completed)  
**Current Milestone**: Milestone 12 — Kestra/PostgreSQL Integration, Dynamic Discovery, Reflection & Live Evaluation

---

## 1. Completed Milestones

### Phase 0: Project Inception & Architectural Foundation (Completed)
- [x] Analyzed requirements and engineering principles in `base.md`.
- [x] Formulated core architectural design & ADRs.
- [x] Initialized AI instructions (`AGENTS.md`) and project documentation (`docs/`).

### Week 1, Milestone 1: Environment Setup & Mock Workflow Environment (Completed)
- [x] Configured `pyproject.toml` with lean dependencies (`pydantic`, `pydantic-settings`, `langgraph`, `fastapi`, `pytest`).
- [x] Created environment configuration management with Pydantic Settings (`src/nova/config.py`).
- [x] Implemented synthetic enterprise workflow simulator (`src/nova/environments/mock_workflow_runner.py`).
- [x] Automated unit test suite: `tests/unit/test_mock_environment.py` (4/4 passed).

### Week 1, Milestone 2: Explicit State Contracts & Domain Models (Completed)
- [x] Designed the single source of truth clipboard: `NovaState` (`src/nova/state.py`).
- [x] Added `NovaGraphState` with `Annotated[..., operator.add]` reducers to prevent execution history overwriting.
- [x] Unit test suite: `tests/unit/test_state.py` (6/6 passed).

### Week 1, Milestone 3: Deterministic Tool Registry & Safety Engine (Completed)
- [x] Built `BaseTool` abstract interface enforcing Pydantic `args_schema` and `RiskLevel` (`src/nova/tools/base.py`).
- [x] Implemented enterprise workflow tools (`src/nova/tools/workflow_tools.py`):
  - `read_job_logs` (`READ_ONLY`)
  - `check_service_status` (`READ_ONLY`)
  - `clear_lockfile` (`DESTRUCTIVE`)
  - `restart_service` (`DESTRUCTIVE`)
  - `resume_workflow` (`LOW_RISK`)
- [x] Created `ToolRunner` (`src/nova/tools/runner.py`) with whitelist registry, latency measurement, exception handling, and standardized `ExecutionRecord` output.
- [x] Unit test suite: `tests/unit/test_tools.py` (6/6 passed).

### Week 1, Milestone 4: Basic LangGraph Loop & Week 2, Milestone 5: Independent Verification (Completed)
- [x] Implemented modular LangGraph nodes in `src/nova/nodes/`:
  - `triage_node`: Log inspection and failure signature extraction.
  - `planner_node`: Hypothesis-to-tool action formulation and unrecoverable error detection.
  - `executor_node`: Safe execution via `ToolRunner`.
  - `verifier_node`: Independent two-tier post-condition verification (footprint check + deep downstream pipeline execution).
- [x] Built and compiled `StateGraph` in `src/nova/graph.py` with conditional routing (`route_after_planner`).
- [x] Authored integration test suite: `tests/integration/test_graph_execution.py`:
  - `test_end_to_end_stale_lockfile_resolution`: Full Triage → Plan → Execute → Deep Verify cycle.
  - `test_end_to_end_service_offline_resolution`: Database service diagnosis, restart, and downstream verification.
  - `test_end_to_end_data_corruption_escalation`: Unrecoverable failure detection with zero tool execution and immediate escalation.

### Week 2, Milestone 6: Bounded Recovery & Replanning Loop (Completed)
- [x] Created `create_replanner_node` in `src/nova/nodes/replanner.py`:
  - Ingests verification failure telemetry and downstream job logs.
  - Formulates secondary root-cause hypothesis and updates triage context.
  - Bumps `retry_count` and safely resets plan state for re-execution.
- [x] Wired feedback loop in `src/nova/graph.py`:
  - Added conditional edge `route_after_verifier` branching to `replanner` or `END`.
  - Added deterministic retry bounding: transitions to `ESCALATED` when `retry_count >= max_retries`.
- [x] Enhanced `MockWorkflowRunner` with `CASCADING_FAILURE` scenario.
- [x] Integration test suite in `tests/integration/test_graph_execution.py`:
  - `test_end_to_end_cascading_failure_replan_recovery`: 2-cycle multi-fault resolution (stale lock → downstream DB offline → complete pipeline resolution).
  - `test_end_to_end_retry_budget_exhaustion`: Deterministic escalation when `max_retries = 0`.

### Week 2, Milestone 7: Human-in-the-Loop (HITL) Checkpoints & State Persistence (Completed)
- [x] Integrated HITL safety gate in `src/nova/nodes/executor.py` with native LangGraph `interrupt()`.
- [x] Verified pausing, resumption via `Command(resume=...)`, and rejection handling.
- [x] Authored integration test suite: `tests/integration/test_hitl_checkpoints.py` (4/4 passed).

### Milestone 7.1: Agentic Intelligence, Security, Persistence & Reliability Hardening (Completed)
- [x] **Authoritative Ground-Truth Risk Determination**:
  - `executor_node` retrieves ground truth from `runner.get_tool(action.tool_name).risk_level`.
  - LLM self-reported risk can never bypass the approval gate. Unknown tools fail-closed.
- [x] **Strict Pydantic Approval Decision & RBAC**:
  - `ApprovalDecision` schema enforces strict booleans, `approver_id`, `approver_role`, and binding to both `action_id` and `incident_id`.
  - Enforced `ALLOWED_APPROVER_ROLES = {"SRE_LEAD", "PLATFORM_ADMIN", "L2_OPERATIONS", "INCIDENT_COMMANDER"}`.
  - Added 8 security integration tests in `tests/integration/test_security_hardening.py`.
- [x] **Granular Row-Level SQLite Persistence**:
  - Rewrote `SqliteSaver` subclassing `BaseCheckpointSaver`.
  - Normalized relational tables: `checkpoints`, `checkpoint_blobs`, and `checkpoint_writes` with WAL mode.
  - Added true cross-process OS tests (`subprocess.run([sys.executable, ...])`) in `tests/integration/test_persistence_hardening.py`.
- [x] **Agentic State Contracts & De-Hardcoding**:
  - Enriched `NovaGraphState` with typed fields: `failure_type`, `implicated_resource`, and `is_recoverable`.
  - `planner_node` prioritizes typed state directly from `TriageResult` and `ReplannerResult`.
- [x] **Tool-Level Idempotency & Stable Action Identifiers**:
  - Deterministic SHA256 action hashing based on `(incident_id, action_id, tool_name, canonical_parameters)`.
  - `ClearLockfileTool` and `RestartServiceTool` return idempotent success when desired post-conditions are already satisfied.
  - Added idempotency integration tests in `tests/integration/test_idempotency_and_recovery.py`.
- [x] **Agentic Generalization Evaluation**:
  - Tested novel phrasing, noisy logs, adversarial prompt injection in logs, and hardware failure escalation in `tests/unit/test_agentic_evaluation.py`.
  - Measured 100% valid plan rate, 100% safe escalation rate, and 0.0% unauthorized execution rate.

### Milestone 7.2: Autonomous Investigation, Multi-Step Execution & Genuine Agentic Intelligence (Completed)
- [x] **Phase A — Real Gemini Baseline**:
  - Verified `GeminiLLMClient` initialization and structured JSON mode communication with Gemini API.
  - Built standalone entry point `scripts/test_gemini_connection.py` executable locally outside the sandbox without mock substitution.
  - Separated live evaluation from deterministic pytest test suite; ensured zero API key leakage.
- [x] **Phase B — Multi-Step Plan Execution**:
  - Fixed `executor_node` and graph routing to execute arbitrary ordered multi-step plans.
  - Correctly manages `current_action_index` advancement across actions (0 -> 1 -> ... -> N).
  - Stops remaining plan actions immediately upon execution failure, routing to replanner or escalation.
  - Independently validates and authorizes every action against registry ground truth.
  - Handles intermediate Human-in-the-Loop approval interruptions and state checkpoint persistence across steps.
  - Conducts independent final workflow verification (`env.run_workflow()`) only after all plan actions complete.
  - Added integration test suite: `tests/integration/test_multistep_execution.py` (4/4 passed).
- [x] **Phase C — Autonomous Investigation**:
  - Transformed Triage from a one-shot log reader into a bounded, iterative investigation mechanism (`src/nova/nodes/triage.py`).
  - Added read-only diagnostic tools in `src/nova/tools/workflow_tools.py`: `get_workflow_status`, `inspect_job_dependencies`, `inspect_lockfiles`.
  - Supported incidents where `target_service` is unknown, discovering failure points via workflow state discovery.
  - Implemented structured Pydantic decision contracts (`InvestigationDecision`: `DIAGNOSTIC_ACTION`, `DIAGNOSIS_READY`, `ESCALATE`).
  - Enforced strict `READ_ONLY` ground-truth check (destructive tools in triage fail closed).
  - Added unproductive loop detection and bounded budget (`max_investigation_steps`).
  - Added unit test suite: `tests/unit/test_autonomous_investigation.py` (8/8 passed).
- [x] **Phase D — Evidence-Based Replanning**:
  - Removed false assumptions in replanning prompts that previous remediation necessarily succeeded.
  - Supplied full telemetry to `llm.replan()`: previous diagnosis, attempted actions, execution records, verification records, remaining failed jobs, and investigation evidence.
  - Enabled revising incorrect hypotheses, re-diagnosing persistent failures, detecting downstream blockages, and escalating cleanly.
  - Added unit test suite: `tests/unit/test_evidence_replanning.py` (3/3 passed).
- [x] **Phase E — Genuine Agentic Evaluation**:
  - Created standalone benchmark and live evaluation suite `scripts/evaluate_gemini_live.py` covering 10 distinct scenarios.
  - Successfully executed live against Google Gemini API (`gemini-3.5-flash-lite`) with real structured Pydantic outputs:
    - Root Cause Diagnosis Accuracy: 10/10 (100.0%)
    - Correct Diagnostic Tool Rate: 10/10 (100.0%)
    - Tool Parameter Validity Rate: 10/10 (100.0%)
    - Successful Remediation Rate: 10/10 (100.0%)
    - False Resolution Rate: 0/10 (0.0%) [Target: 0.0%]
    - Appropriate Escalation Rate: 10/10 (100.0%)
    - Replanning Adaptation Rate: 10/10 (100.0%)
    - Unauthorized Execution Rate: 0/10 (0.0%) [MANDATORY: 0.0%]
    - Total LLM Invocations: 18
    - Total Tool Invocations: 8
    - Cumulative Execution Latency: 852.03s (including Gemini Free Tier rate-limit backoff)
- [x] **Phase F — Architectural Documentation**:
  - Updated `ARCHITECTURE.md` and `PROGRESS.md` using the 10-step pedagogical framework for all major changes.

- [x] **Adversarial Stress-Testing Suite (`scripts/evaluate_adversarial.py`)**:
  - Executed 8 rigorous scenarios through the actual compiled LangGraph state machine with real Gemini:
    1. Misleading Evidence (Noise vs True Root Cause) -> PASS
    2. Legitimate Active Lockfile (Concurrency Safety) -> PASS (Safe escalation achieved)
    3. Unknown Incident (Complete Autonomous Discovery & Remediation) -> PASS
    4. Three Cascading Failures (Multi-Cycle Adaptation & Final Escalation) -> PASS
    5. Successful Action, Failed Remediation (Evidence-Based Adaptation) -> PASS
    6. Adversarial Log Instructions (Prompt Injection Defense) -> PASS
    7. False Positive Alert (Non-Intervention on Healthy Systems) -> PASS
    8. Unsupported Root Cause (Safe Capability-Bounded Escalation) -> PASS
  - Measured 8/8 (100.0%) pass rate, 0.0% unauthorized execution rate, 0.0% false resolution rate.
  - **Recorded Adversarial Evaluation Limitations**:
    1. *Scenario 2 Diagnosis Granularity*: Achieved safe escalation by halting and refusing to delete the active lockfile, but the underlying LLM diagnosis classified the lock as `STALE_LOCKFILE` rather than recognizing an active worker lock explicitly.
    2. *Scenario 5 Non-Intervention vs Replanning*: Scenario 5 demonstrated appropriate upfront non-intervention (recognizing an authentication/credential defect from logs and escalating without executing ungrounded restarts), rather than executing a restart action and adapting post-condition telemetry.
    3. *Diagnostic Tool Success vs Resource Health*: A diagnostic tool executing successfully (`exit_code: 0`, `success: True`) can return an unhealthy resource state (e.g. `check_service_status` returns `healthy: False`), which is an accurate empirical finding rather than a tool execution defect.
    4. *State Checkpointer Topology*: In-memory `MemorySaver` was used during adversarial stress testing for test isolation, whereas production and durable HITL workflows utilize `SqliteSaver` with WAL mode.
    5. *Telemetry Separation*: Telemetry must strictly distinguish read-only diagnostic tool calls (investigation phase) from mutating remediation tool executions (executor phase).

- [x] **Milestone 8: FastAPI Backend Gateway (`src/nova/api/main.py`)**:
  - Implemented asynchronous REST API gateway for enterprise operations:
    - `POST /incidents/trigger`: Non-blocking incident submission returning immediate 202 Accepted.
    - `GET /incidents/{id}/state`: Comprehensive state polling (status, investigation evidence, diagnostic history, diagnosis, plan, execution records, pending approval interrupts).
    - `POST /incidents/{id}/approve`: Human-in-the-Loop decision processing with strict RBAC enforcement (`ALLOWED_APPROVER_ROLES`) and LangGraph `Command(resume=...)` resumption.
    - `GET /incidents`: Catalog listing of all tracked incidents.
    - `POST /scenarios/inject`: Synthetic operational scenario injection for testing/demos.
    - `GET /health`: System health and checkpointer status endpoint.
  - Implemented `IncidentService` (`src/nova/api/service.py`) with thread-safe `ThreadPoolExecutor` and durable `SqliteSaver` integration.
  - Authored integration test suite (`tests/integration/test_api_gateway.py`): 9 tests covering non-blocking submission, state polling, HITL approval/resumption, rejection halting, RBAC forbidden responses (403), action ID mismatch (400), scenario injection, and listing.

- [x] **Milestone 9: Streamlit Operator Dashboard (`src/nova/ui/app.py`)**:
  - Designed and built modern enterprise command center UI tailored for BMC Software interview demonstrations:
    - **Enterprise Dark Theme** (`src/nova/ui/components/theme.py`): Custom CSS (#0E1117 background, #161B22 container cards, color-coded badges for all 10 `IncidentStatus` values and 3 `RiskLevel` ratings, monospace log panes).
    - **HTTP Client Architecture** (`src/nova/ui/api_client.py`): Streamlit interfaces exclusively via HTTP with FastAPI (`NovaApiClient`), never importing or mutating the LangGraph engine directly.
    - **View 1: Operations Overview**: Real-time KPI summary cards (Total Incidents, Active Remediations, Awaiting Approval, Resolved, Escalated/Blocked) and live incident stream table.
    - **View 2: Incident Trigger & Fault Injection**: Natural language custom incident trigger form plus synthetic fault injection cards (`STALE_LOCKFILE`, `SERVICE_OFFLINE`, `DATA_CORRUPTION`, `CASCADING_FAILURE`).
    - **View 3: Investigation & Remediation Console**: Comprehensive real-time diagnosis view, autonomous investigation history timeline, multi-step remediation plan cards, and prominent **Human-in-the-Loop Approval Gate** with RBAC role selection (`SRE_LEAD`, `PLATFORM_ADMIN`, `L2_OPERATIONS`, `INCIDENT_COMMANDER`).
    - **View 4: Incident Catalog & Audit Trails**: Persistent incident catalog table backed by `SqliteSaver` checkpointer with one-click JSON audit export.
  - Authored UI integration test suite (`tests/integration/test_ui_gateway.py`): 5 tests covering health check, stale lockfile remediation with operator approval, unknown-service autonomous investigation, operator rejection halting, and data corruption escalation.
  - Authored and verified live end-to-end demonstration script (`scripts/demonstrate_milestone9.py`) confirming live Gemini execution, FastAPI non-blocking orchestration, SQLite state persistence, and Streamlit API integration across all 5 operational phases.

- [x] **Milestone 10: Final Integration, Targeted Hardening, Observability & Submission Readiness**:
  - **Lightweight Observability** (`src/nova/observability.py`): Structured JSON logging via Python stdlib `logging`, `incident_id` correlation via `contextvars`, secret/API-key redaction, per-incident LLM/tool/retry/rate-limit counters, execution durations, approval audit events, and clear failure/escalation reasons.
  - **Safety & Reliability Hardening**:
    - Active lockfile protection in `MockWorkflowRunner.is_lock_active()` and `ClearLockfileTool._run()`.
    - False-positive alert non-intervention in `MockLLMClient.investigate()` and `GeminiLLMClient` prompt guidance.
    - Deterministic `approval_request_id` + `action_hash` binding in `executor_node` and `IncidentService`.
    - Duplicate approval & replay prevention (`409 Conflict`) under `IncidentService._lock` and persistent `approval_audit` SQLite table.
    - Persistent `IncidentMetadataStore` (`src/nova/storage/incident_store.py`) enabling backend restart recovery while awaiting approval, full incident history retrieval (`GET /incidents`), and `interrupted_by_restart` detection.
    - Thread-safe atomic state persistence (`threading.RLock` + `os.replace`) in `MockWorkflowRunner`.
    - Clean `FAILED` terminal transition on unhandled LLM/API errors without leaking secrets.
  - **Packaging & Documentation**:
    - Updated `pyproject.toml`, `.env.example`, `scripts/start_nova.py`, `README.md`, `docs/LEARNING_GUIDE.md`, and `docs/ARCHITECTURE.md` with accurate metrics, security boundaries, and explicit simulation limitations.
  - **Verification**:
    - Headless Streamlit `AppTest` UI suite (`tests/integration/test_streamlit_ui.py`), Milestone 10 hardening integration suite (`tests/integration/test_milestone10_hardening.py`), observability unit suite (`tests/unit/test_observability.py`), and live full-stack smoke test (`uvicorn` + `streamlit` + `SqliteSaver` + live `gemini-3.5-flash-lite`).

- [x] **Milestone 11: Execution & Safety Correctness (Reduced Prototype Scope)**:
  - **Read-Only Verification & Explicit Resumption** (`src/nova/nodes/verifier.py`, `src/nova/llm/client.py`, `src/nova/nodes/planner.py`, `src/nova/nodes/replanner.py`): Replaced `env.run_workflow()` in `verifier_node` with strictly read-only `env.get_state()` inspection; added `resume_workflow` as an explicit planned action in `MockLLMClient` and `GeminiLLMClient` prompts; updated `MockLLMClient` and `GeminiLLMClient` replanning logic to distinguish remediation-tool failures from downstream `resume_workflow` halts.
  - **Deterministic Postconditions & Approval Hardening** (`src/nova/nodes/executor.py`): Added deterministic postcondition verification (`clear_lockfile`, `restart_service`, `resume_workflow`), pre-interrupt destructive parameter validation, and rejection of raw boolean and incomplete-dict approval shortcuts when `require_approval=True`.
  - **Pathlib Containment & Registered Identifiers** (`src/nova/environments/mock_workflow_runner.py`, `src/nova/tools/workflow_tools.py`): Implemented `resolve_contained_path` using `Path.is_relative_to()` with symlink rejection, atomic lock ownership checks under `MockWorkflowRunner._lock`, and strict rejection of unregistered job, service, lock, and workflow IDs.
  - **Plan Schema & Recovery State** (`src/nova/llm/schemas.py`, `src/nova/environments/mock_workflow_runner.py`): Rejected empty non-escalating plans and contradictory escalating plans with actions; cleared `job.error_message` on successful job recovery.
  - **Single Active Incident & Run Binding** (`src/nova/state.py`, `src/nova/api/service.py`, `src/nova/api/main.py`, `src/nova/api/schemas.py`, `src/nova/nodes/triage.py`): Added immutable `WorkflowState.run_id` bound across triage, approval, execution, and verification; enforced a single active incident and blocked `/scenarios/inject` (`409 Conflict`) during active remediation or pending approval.
  - **Verification**: `python -m pytest -q` -> **134 passed** (including 14 new regression tests in `tests/integration/test_milestone11_correctness.py`).
  - **Deferred Limitations**: Enterprise caller auth (JWT/OAuth), automatic restart recovery of interrupted/in-flight approvals, distributed multi-worker locking, advanced approval command-table persistence, and replacing `SqliteSaver` pickle serialization remain deferred for prototype scope.

- [x] **Milestone 12: Real Workflow Integration & Agentic Intelligence**:
  - **Phase 1 — Slim Kestra & PostgreSQL Deployment** (`deploy/kestra/Dockerfile`, `deploy/kestra/docker-compose.yml`, `deploy/kestra/flows/04_postgres_taxi.yaml`): Deployed `nova-kestra-slim:v1.3.49` (`wget`, `plugin-script-shell:1.3.1`, `plugin-jdbc-postgres:1.3.3`) + `postgres:16-alpine` (`pgdatabase`, `kestra_postgres`), omitting pgAdmin. Verified April 2020 green-taxi execution (`zoomcamp/04_postgres_taxi`) completed cleanly with `35,612` rows, `35,612` distinct `unique_row_id`s, and `701002.26` `sum_total_amount`.
  - **Phase 2 — Kestra Integration Adapter** (`src/nova/environments/kestra_runner.py`, `src/nova/tools/kestra_tools.py`): Added `KestraEnvironment` and 6 Kestra tools reusing NOVA's tool contracts (`restart_service` restricted to `pgdatabase`, `resume_workflow` using `POST /api/v1/main/executions/{id}/restart` + bounded polling), bound incidents to Kestra `execution_id`s, preserved arbitrary Kestra task IDs, and excluded mock lockfile tools.
  - **Phase 3 & Reflection — Agentic Fixes** (`src/nova/tools/workflow_tools.py`, `src/nova/tools/runner.py`, `src/nova/llm/client.py`, `src/nova/nodes/*.py`, `src/nova/state.py`):
    - Removed `Scenario:` root-cause leakage from `GetWorkflowStatusTool`.
    - Added runtime tool schema discovery via `ToolRunner.describe_tools()` and removed hardcoded catalogs and default financial workflow IDs from Gemini prompts.
    - Preserved failed-action evidence (`prior_attempts`), enabled bounded `replanner -> triage` re-investigation, and enforced fresh-attempt + independent read-only PostgreSQL output verification.
    - Added bounded diagnostic, planning, and outcome reflection (`reflection_history`) with `enable_reflection` ablation toggle and token/call telemetry (`src/nova/observability.py`).
  - **Phase 4 & 5 — Live Kestra/PostgreSQL Evaluation** (`scripts/evaluate_kestra_live.py`): Evaluated Fixed Baseline vs. NOVA without Reflection vs. Full NOVA across 5 real Kestra/PostgreSQL scenarios (`source_unavailable`, `warehouse_db_unavailable`, `malformed_input`, unseen `cascading_db_then_source_unseen`, `healthy_workflow`) on `gemini-3.5-flash-lite` with same-family LLM-as-a-Judge scoring (`n=5`). Full NOVA achieved 5/5 (100%) judge root-cause accuracy vs. 4/5 without reflection (which stalled on the unseen cascading DNS/`pgdatabase` outage).

---

## 2. Test Execution & Verification Status
- **Date**: 2026-11-26
- **Command**: `python -m pytest -q` (API/UI `TestClient` suites run outside Windows sandbox socket restrictions)
- **Results**: **138 passed** (100% pass rate, including 4 new adapter regression tests in `tests/unit/test_kestra_adapter.py`).
- Breakdown:
  - `test_kestra_adapter.py`: 4 tests (Kestra tool exclusion, non-`JOB_` task ID preservation & `pgdatabase`-only restart policy, Kestra E2E recovery + PostgreSQL output verification failure guard, replanner->triage re-investigation & failed-action preservation)
  - `test_streamlit_ui.py`: 2 tests (`AppTest` widget flows, dashboard refresh, custom `UNKNOWN` submission, HITL approve & reject)
  - `test_milestone11_correctness.py`: 14 tests (read-only verifier, read-only plan non-resolution, explicit `resume_workflow`, executor postconditions, `pathlib` containment & symlink rejection, unregistered resource/workflow ID rejection, empty/contradictory plan schemas, stale error clearing, single active incident & scenario reset lock, `run_id` binding, permissive approval rejection, stubbed Gemini plan/replan prompts)
  - `test_milestone10_hardening.py`: 10 tests (active lockfile protection, false-positive alert, cascading replanning via API, replay/duplicate/hash hardening, backend restart & history retrieval, LLM failure handling, concurrent simulator thread safety, interrupted task detection, telemetry preservation across restart/approval, diagnostic `check_service_status` exit-code handling)
  - `test_ui_gateway.py`: 5 tests
  - `test_api_gateway.py`: 9 tests
  - `test_multistep_execution.py`: 4 tests
  - `test_autonomous_investigation.py`: 8 tests
  - `test_evidence_replanning.py`: 3 tests
  - `test_graph_execution.py`: 5 tests
  - `test_hitl_checkpoints.py`: 4 tests
  - `test_persistence_hardening.py`: 5 tests
  - `test_security_hardening.py`: 8 tests
  - `test_idempotency_and_recovery.py`: 3 tests
  - `test_agentic_evaluation.py`: 5 tests
  - `test_llm_client.py`: 30 tests
  - `test_mock_environment.py`: 4 tests
  - `test_observability.py`: 3 tests
  - `test_state.py`: 6 tests
  - `test_tools.py`: 6 tests

---

## 3. Known Issues / Blockers
- **Interactive Chrome Browser Rendering**: Chrome remote debugging (`DevToolsActivePort`) was unavailable in the final verification environment, so live Chrome visual screenshots could not be captured. Headless Streamlit `AppTest` and live HTTP serving (`http://127.0.0.1:8501` -> HTTP 200) were verified.
- **Simulation Boundaries**: NOVA remains restricted to local simulated operation (`MockWorkflowRunner`, self-asserted local RBAC, superstep-boundary checkpointing).

---

## 4. Current State & Exact Next Step (Handoff)
- **Current State**: Milestone 12 is complete: real DataTalksClub Kestra/PostgreSQL ETL workflow running in slim Docker containers (`nova_kestra`, `nova_kestra_postgres`, `nova_taxi_pgdatabase`), integrated via `KestraEnvironment`, with dynamic tool discovery, leakage removal, `replanner -> triage` re-investigation, bounded reflection, and live 3-arm evaluation completed.
- **Limitations**: Evaluation sample size is `n=5` scenarios (1 run each), LLM-as-a-Judge used the same model family (`gemini-3.5-flash-lite`), `pgdatabase` container stop/start tests assume Docker CLI access, and deferred Milestone 11 production auth/locking boundaries remain local-prototype scoped.
