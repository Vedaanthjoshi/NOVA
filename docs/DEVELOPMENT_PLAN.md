# Development Plan: NOVA (3-Week Roadmap)

## Overview & Approach
This roadmap is structured to ensure that every milestone produces a functional, tested slice of the system. We follow the 10-step pedagogical framework: **Explain $\rightarrow$ Design $\rightarrow$ Justify $\rightarrow$ Implement $\rightarrow$ Inspect $\rightarrow$ Test $\rightarrow$ Debug $\rightarrow$ Validate Understanding $\rightarrow$ Document $\rightarrow$ Hand Over**.

---

## Week 1: Core Engine, Deterministic Foundations & State
**Goal**: Build a fully functional, tested, single-run execution graph with strict typed state and simulated tools.

### Milestone 1: Environment & Project Foundation
- Setup project dependencies via `pyproject.toml` (LangGraph, Pydantic v2, pytest, etc.).
- Configure environment variables and configuration management (`config.py`).
- Implement the Mock Workflow Environment (`environments/mock_workflow_runner.py`) simulating batch jobs and failure modes.
- **Verification Gate**: Mock environment runs and reliably produces deterministic failure scenarios.

### Milestone 2: Explicit State Contracts & Domain Models
- Define `NovaState`, `PlannedAction`, `ExecutionRecord`, `VerificationRecord` using Pydantic and `TypedDict`.
- Implement validation routines and state reducers.
- **Verification Gate**: Unit tests in `tests/unit/test_state.py` passing with 100% coverage on state serialization.

### Milestone 3: Deterministic Tool Registry & Safety Engine
- Implement `BaseTool` abstract class with strict typing and `RiskLevel` (`READ_ONLY`, `LOW_RISK`, `DESTRUCTIVE`).
- Implement core mock tools: `check_service_status`, `read_job_logs`, `clear_lockfile`, `restart_service`.
- Implement execution runner with timeout and stdout/stderr capture.
- **Verification Gate**: Unit tests in `tests/unit/test_tools.py` verifying permission enforcement and timeout handling.

### Milestone 4: Basic LangGraph Loop (Triage $\rightarrow$ Plan $\rightarrow$ Execute)
- Assemble the initial LangGraph state graph.
- Implement basic Triage and Planner nodes using structured LLM output (GPT / Gemini API).
- **Verification Gate**: Integration test running a complete happy-path diagnosis and single-action execution.

---

## Week 2: Robustness, Verification, Recovery & API
**Goal**: Implement post-condition verification, replanning loops, human-in-the-loop breakpoints, and FastAPI backend.

### Milestone 5: Independent Post-Condition Verification
- Implement the Verifier node (`agents/verifier.py`).
- Distinguish between tool execution status and verified environmental post-condition.
- **Verification Gate**: Tests prove that a tool returning success when the condition is NOT met triggers a verification failure.

### Milestone 6: Bounded Recovery & Replanning Loop
- Connect conditional edges for verification failure to the Replanner node.
- Enforce strict retry bounds (`max_retries`) and state rollback logic.
- **Verification Gate**: Scenario tests showing successful recovery from a transient failure after 1 replan.

### Milestone 7: Human-in-the-Loop (HITL) Checkpoints
- Implement LangGraph `interrupt()` mechanism on `DESTRUCTIVE` operations.
- Persist state across interrupts using `SqliteSaver`.
- Resume execution when operator submits an approval/rejection token.
- **Verification Gate**: Test showing the graph pauses indefinitely on destructive actions and only resumes upon approval.

### Milestone 8: FastAPI Backend Gateway
- Implement REST API (`api/main.py`):
  - `POST /incidents/trigger`: Starts a new incident run.
  - `GET /incidents/{id}/state`: Retrieves current graph state and pending actions.
  - `POST /incidents/{id}/approve`: Approves or rejects pending HITL actions.
- **Verification Gate**: Automated API tests via `httpx`/`TestClient`.

---

## Week 3: Observability, Operator UI, Evaluation & Interview Prep
**Goal**: Build the Streamlit dashboard, comprehensive evaluation suite, and finalize interview defense prep.

### Milestone 9: Streamlit Operator Dashboard
- Live visualization of the active incident graph and state transitions.
- Interactive log viewer and pending approval dialog for human operators.
- Post-mortem report display.
- **Verification Gate**: End-to-end interactive manual test through the UI.

### Milestone 10: Evaluation & Benchmark Suite
- Create 6–8 automated test scenarios:
  1. *Simple Crash*: Service restart clears it.
  2. *Stale Lockfile*: Requires triage, diagnosis, lock deletion, and verification.
  3. *Unrecoverable Corruption*: Requires graceful failure and operator escalation.
  4. *Destructive Action*: Verifies human approval requirement is never bypassed.
- Measure: Resolution rate, token efficiency, MTTR, and safety adherence.
- **Verification Gate**: Automated eval script runs and generates a markdown scorecard.

### Milestone 10B: Staged Tool Synthesis & Sandbox Verification Pipeline (Committed Feature)
- **Problem Solved**: Automatically authoring tools for unforeseen errors without risking unverified production execution.
- **Implementation**:
  1. *Tool Authoring Engine*: LLM prompt generating valid Python subclasses of `BaseTool` with Pydantic `args_schema`.
  2. *Test Generator*: Automatically generates a corresponding `pytest` verification suite for the new tool.
  3. *Sandboxed Execution Runner*: Safely executes the generated tool and its tests in an isolated workspace without network access.
  4. *Human PR / Promotion Gate*: Operator reviews the generated code diff and sandbox test run in the Streamlit UI, approving it for dynamic injection into `TOOL_REGISTRY`.
- **Verification Gate**: End-to-end scenario where NOVA encounters an unknown error, synthesizes a tool, proves it in the sandbox, waits for human operator sign-off, and promotes it to solve the issue.

### Milestone 11: Final Documentation & Technical Interview Defense
- Complete `docs/DECISIONS.md` (ADRs).
- Complete `docs/INTERVIEW_PREP.md` with in-depth defense of trade-offs, architecture diagrams, and BMC alignment.
- Final rehearsal of technical walkthrough.

