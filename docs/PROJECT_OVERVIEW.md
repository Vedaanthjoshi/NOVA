# Project Overview: NOVA (Neural Orchestration and Virtual Agent)

## 1. Problem Statement
In enterprise IT operations and workload management (e.g., banking, telecom, supply chain), thousands of batch processes, ETL jobs, and distributed microservices run continuously. When workflows fail (e.g., job timeouts, deadlocks, disk exhaustion, downstream API connection failures), remediation typically falls to Site Reliability Engineers (SREs) and Operations Teams.

Manual incident remediation suffers from:
- **High Mean Time to Recovery (MTTR)**: Human operators must wake up, locate logs, decode stack traces, and manually execute runbook steps.
- **Human Error under Stress**: Operators running ad-hoc shell commands late at night can cause accidental data loss or cascading outages.
- **Superficial "Automation" Scripts**: Traditional scripts lack semantic context. They cannot reason about novel failure modes, diagnose dependent service failures, or adapt to partial failures.

---

## 2. The Solution: What NOVA Solves
**NOVA** is an autonomous, stateful Level 2 Operations & Incident Remediation Agent. It bridges the gap between static runbook automation and unguided human intervention:

1. **Intelligent Triage**: Ingests incident alerts, polls target telemetry, and isolates anomalous log signatures using LLM reasoning.
2. **Deterministic Governance**: Formulates diagnostic hypotheses and maps them to pre-validated recovery plans.
3. **Safety & Human-in-the-Loop (HITL)**: Enforces clear risk tiering. Low-risk actions (diagnostics, log reads, status checks) run autonomously; high-risk actions (service restart, lock clearing, file deletion) pause execution to require human sign-off.
4. **Post-Condition Verification**: Never trusts a tool's return code blindly. Executes independent verification probes (e.g., healthchecks, metric polling) before reporting success.
5. **Auditable Persistence**: Tracks every state transition, LLM decision, and tool invocation in an immutable SQLite audit log.

---

## 3. Alignment with BMC Software Context
BMC Software is a global leader in enterprise IT management, automated workload orchestration (Control-M), and cognitive service operations (Helix).

NOVA directly reflects BMC core engineering principles:
- **Workload & Batch Orchestration**: Awareness of task dependencies, retry budgets, and state machines.
- **Enterprise Safety**: Non-destructive defaults, principle of least privilege for tools, and operator-in-the-loop workflows.
- **Operational Reliability**: Focus on determinism, state recovery, and observability over brittle AI "magic".

---

## 4. Minimum Viable Product (MVP) Scope
### In Scope (v1 - 3 Weeks):
- Mock enterprise batch workflow generator that produces realistic failures (e.g., deadlocks, network timeouts, stale lockfiles).
- LangGraph-based state machine with explicit typed state (`TriageState`, `PlanState`, `ExecutionState`, `VerificationState`).
- Tool execution interface with safety classifications (`READ_ONLY`, `LOW_RISK`, `DESTRUCTIVE`).
- LangGraph native interrupts for human approval.
- Independent verification probe loop with bounded replanning.
- FastAPI backend for incident management.
- Streamlit dashboard providing operator visibility into the active graph, state, and approval checkpoints.
- Reproducible evaluation test suite measuring autonomous resolution rate, safety adherence, and verification accuracy.

### Out of Scope (Deliberately Excluded from v1):
- Multimodal audio/voice interfaces.
- Direct mutation of live production systems (all tests run in controlled mock environments).
- Distributed multi-agent swarms (unnecessary complexity; one well-governed graph is vastly superior).
- Premature vector database / RAG setups.

