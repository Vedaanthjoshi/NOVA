# Interview Preparation: Defending NOVA

This document prepares you to explain, defend, and walk through NOVA during technical interviews, with specific alignment for **BMC Software**.

---

## 1. The Elevator Pitch (30-45 Seconds)
> *"NOVA is an autonomous Level 2 Operations Agent designed to diagnose and remediate enterprise batch workflow failures. Instead of relying on brittle scripts or chaotic multi-agent swarms, NOVA uses a stateful, graph-based architecture built with LangGraph and Pydantic. It enforces enterprise safety with human-in-the-loop checkpoints for destructive actions, and critical to production reliability, it incorporates an independent post-condition verification loop—meaning an action is never declared successful until the system's health is empirically proven."*

---

## 2. Top Interview Questions & Model Answers

### Q1: Why didn't you build a multi-agent system with AutoGen or CrewAI?
**Model Answer**:
> *"In enterprise automation, predictability, auditability, and deterministic state transitions are paramount. Multi-agent systems with conversational debate loops introduce significant non-determinism, high token latency, and potential infinite conversational drift. 
> Instead, I architected NOVA as a single state machine using LangGraph. Each node has a strictly defined responsibility—some use LLMs for unstructured reasoning like log analysis and planning, while others use 100% deterministic Python for safety checks and tool execution. This gives us the power of generative reasoning with the predictability of enterprise software."*

### Q2: How do you prevent the agent from accidentally doing something destructive in production?
**Model Answer**:
> *"We implement a multi-layered safety guardrail:
> 1. **Tool Classification**: Every tool has a metadata risk rating (`READ_ONLY`, `LOW_RISK`, `DESTRUCTIVE`).
> 2. **Deterministic Policy Gate**: Before any tool is executed, a deterministic node checks the tool's rating.
> 3. **Human-in-the-Loop Interrupt**: For any `DESTRUCTIVE` action (such as terminating processes or clearing database locks), LangGraph triggers a native interrupt, checkpoints the state to SQLite, and halts execution until a human operator approves or rejects the step via the API or dashboard."*

### Q3: What is post-condition verification and why is it necessary?
**Model Answer**:
> *"In SRE and enterprise operations, a tool returning an exit code of `0` does not mean the system is healed. For example, executing a service restart command might return `0`, but the service might immediately fail on startup due to a port collision.
> In NOVA, we explicitly decouple 'Attempted Action' from 'Verified Outcome'. After an action is executed, a dedicated Verifier node executes independent health probes (checking HTTP endpoints, verifying queue depths, or inspecting process tables). Only if the empirical verification passes is the step marked successful."*

### Q4: How does NOVA align with enterprise tools like BMC Control-M?
**Model Answer**:
> *"BMC Control-M is the industry standard for enterprise workflow orchestration. Control-M excels at scheduling, dependency management, and executing complex job streams across hybrid environments. 
> NOVA is designed to sit alongside workload orchestrators as an intelligent operational assistant: ingesting job failure alerts, analyzing error output, reasoning across failure logs, and safely orchestrating remediation runbooks with human oversight."*

### Q5: How do you handle LLM hallucination or invalid tool arguments?
**Model Answer**:
> *"We use Pydantic v2 structured outputs. The LLM cannot output arbitrary strings; it is constrained to output schemas that adhere strictly to our Pydantic models. Furthermore, before any tool executes, arguments are validated deterministically against our type contracts. If validation fails, the error is immediately caught and routed to a replanning node with zero risk of running malformed commands."*

---

## 3. Key Architectural Concepts to Draw on a Whiteboard
1. **The Graph Lifecycle**: Triage $\rightarrow$ Plan $\rightarrow$ Safety Gate $\rightarrow$ (HITL Interrupt if Destructive) $\rightarrow$ Execute $\rightarrow$ Post-Condition Verification $\rightarrow$ (Loop/Replan if failed) $\rightarrow$ Incident Summary.
2. **State Separation**: Showing how `NovaState` flows through the graph as an immutable audit record.
3. **The 3 Tiers of Tool Safety**: Read-only, Low-risk, Destructive.

