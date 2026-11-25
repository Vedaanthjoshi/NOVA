# Architectural Decision Records (ADRs)

This document records the architectural choices, alternatives considered, and trade-offs made during the engineering of NOVA.

---

## ADR-001: Use LangGraph for Workflow Orchestration
- **Status**: Accepted
- **Context**: We need a framework to manage stateful, multi-step agent reasoning, tool invocation, failure loops, and human-in-the-loop approvals.
- **Alternatives Considered**:
  1. *Custom Python `while` loops*: High control, but re-invents state serialization, checkpointing, and branch visualization from scratch.
  2. *LangChain `AgentExecutor` (Legacy)*: Black-box ReAct loop, deprecated, poorly supports cyclic retries, difficult to pause for human approval.
  3. *AutoGen / CrewAI*: Multi-agent chat frameworks focused on LLM-to-LLM conversation rather than deterministic, auditable state machine transitions.
- **Decision**: Use `LangGraph` (`StateGraph`).
- **Consequences**:
  - *Pros*: Explicit state schema, native support for cyclic retries, first-class human-in-the-loop interrupts, built-in SQLite checkpointing.
  - *Cons*: Learning curve for LangGraph abstractions; requires disciplined schema definitions.

---

## ADR-002: Single Orchestration Graph vs. Multi-Agent Swarm
- **Status**: Accepted
- **Context**: In agentic AI, it is common to create multiple conversational agents (e.g., Planner Agent, Debugger Agent, Coder Agent) talking to each other.
- **Alternatives Considered**:
  1. *Multi-agent debate / conversational swarm*: Free-form conversational agents passing messages.
  2. *Single graph with specialized nodes*: A single state machine where each node executes a specific responsibility (some LLM-powered, some deterministic).
- **Decision**: Single graph with specialized nodes.
- **Consequences**:
  - *Pros*: Eliminates conversational drift, drastically lowers token usage and latency, provides deterministic state management, and is directly explainable in an enterprise interview.
  - *Cons*: Does not show off multiple persona chat UIs (which are largely demo toys rather than enterprise software).

---

## ADR-003: Strict Pydantic v2 Contracts for State & Tools
- **Status**: Accepted
- **Context**: LLMs naturally generate unstructured text. Enterprise tools require strictly typed inputs (e.g., integers for process IDs, valid file paths).
- **Alternatives Considered**:
  1. *String-based command parsing*: Fragile, vulnerable to prompt injection and syntax errors.
  2. *Pydantic v2 schemas*: Formal typing with automatic validation and JSON Schema generation.
- **Decision**: Use Pydantic v2 for all state representations, tool inputs, and LLM structured outputs.
- **Consequences**:
  - *Pros*: Fail-fast validation before tool execution. Zero runtime type ambiguity.
  - *Cons*: Slight overhead in defining explicit schemas for every tool.

---

## ADR-004: Explicit Post-Condition Verification Node
- **Status**: Accepted
- **Context**: An executed tool returning exit code `0` or HTTP `200` does not mean the system is fixed (e.g., a service restart command returns 0, but the service crashes 2 seconds later).
- **Alternatives Considered**:
  1. *Trust tool exit codes*: Assume exit code 0 means success.
  2. *Dedicated Verification Node*: Independent healthcheck or telemetry probe executed after every action.
- **Decision**: Dedicated Verification Node in the graph.
- **Consequences**:
  - *Pros*: Eliminates false-positive recoveries; aligns directly with enterprise reliability and SRE best practices.
  - *Cons*: Adds extra state transitions and test overhead.

---

## ADR-005: Local Controlled Mock Environment for v1
- **Status**: Accepted
- **Context**: We need to test incident remediation without risking production systems or incurring expensive cloud infrastructure costs.
- **Alternatives Considered**:
  1. *Live cloud integration (AWS/GCP/Kubernetes)*: Expensive, complex to test locally, high barrier to entry for reviewers.
  2. *Real BMC Control-M instance*: Inaccessible for independent local portfolio development.
  3. *Local Synthetic Workflow Engine*: Python-based environment simulating batch jobs, locks, disk usage, and network health.
- **Decision**: Local Synthetic Workflow Engine (`environments/mock_workflow_runner.py`).
- **Consequences**:
  - *Pros*: 100% reproducible tests, zero cloud costs, instant failure scenario injection, safe for automated evaluation.
  - *Cons*: Requires building and maintaining the mock environment.

---

## ADR-006: Staged Tool Synthesis via Sandboxed Unit Testing and HITL Promotion
- **Status**: Accepted (Committed Feature Roadmap)
- **Context**: While a curated tool registry prevents unauthorized execution, real-world enterprise operations encounter novel edge-case failures where no pre-built tool exists. Unsupervised dynamic code execution on production is dangerous and uncompliant.
- **Decision**: Implement an **Autonomous Tool Synthesis & Sandboxed Staging Pipeline**:
  1. When an incident diagnosis reveals that no registered tool exists, NOVA triggers a tool authoring workflow.
  2. The LLM generates a new Python tool inheriting from `BaseTool` (with explicit Pydantic `args_schema` and `RiskLevel`) along with a dedicated `pytest` suite.
  3. The proposed tool and tests are executed inside an isolated sandbox environment.
  4. The code, sandbox test results, and rationale are staged as a Change Request / Pull Request for human operator review in the UI.
  5. The tool is **never** deployed or executed against production until the human operator reviews, verifies, and explicitly promotes it into `TOOL_REGISTRY`.
- **Consequences**:
  - *Pros*: Combines dynamic adaptability to unforeseen errors with enterprise compliance, auditability, and safety.
  - *Cons*: Requires building isolated sandboxed subprocess execution, AST code inspection, and a review UI.

