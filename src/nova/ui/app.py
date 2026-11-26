"""
NOVA Enterprise Operations Command Center.
Streamlit Operator Dashboard for Autonomous Incident Remediation.

Adheres strictly to the BMC Software design standards:
- Modern enterprise dark theme.
- Exclusively communicates with the backend via the FastAPI REST Gateway.
- Live autonomous investigation visibility, diagnostic history, and multi-step plan inspection.
- Dedicated Human-in-the-Loop approval gate with simulated operator RBAC roles.
- Multi-cycle evidence-based replanning and post-condition verification tracking.
- Persistent audit history inspection via SqliteSaver.
"""

import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import streamlit as st

from nova.ui.api_client import NovaApiClient
from nova.ui.components.theme import (
    ENTERPRISE_DARK_CSS,
    render_badge,
    render_risk_pill,
)

# Page configuration
st.set_page_config(
    page_title="NOVA Operations Console",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject enterprise dark theme styles
st.markdown(ENTERPRISE_DARK_CSS, unsafe_allow_html=True)


# ==============================================================================
# State & Client Initialization
# ==============================================================================

if "api_url" not in st.session_state:
    st.session_state.api_url = "http://127.0.0.1:8000"

if "selected_incident_id" not in st.session_state:
    st.session_state.selected_incident_id = None

if "approval_in_flight" not in st.session_state:
    st.session_state.approval_in_flight = False

api = NovaApiClient(base_url=st.session_state.api_url)


# ==============================================================================
# Sidebar Navigation & Controls
# ==============================================================================

with st.sidebar:
    st.markdown("### 🛡️ **NOVA Operations**")
    st.caption("Neural Orchestration & Virtual Agent")
    st.divider()

    nav_options = [
        "📊 Operations Overview",
        "🚨 Incident Trigger & Fault Injection",
        "🔍 Investigation & Remediation Console",
        "📜 Incident Catalog & Audit Trails",
    ]
    if "nav_view" not in st.session_state:
        st.session_state.nav_view = nav_options[0]
    if st.session_state.get("view_mode_override") in nav_options:
        st.session_state.nav_view = st.session_state.pop("view_mode_override")

    view_mode = st.radio(
        "Navigation",
        nav_options,
        key="nav_view",
    )
    st.divider()

    st.markdown("#### **Gateway Connectivity**")
    health = api.check_health()
    if health.get("status") == "HEALTHY":
        st.success(f"Connected ({health.get('checkpointer', 'SqliteSaver')})")
    else:
        st.error(f"Offline: {health.get('detail', 'Gateway unreachable')}")

    api_url_input = st.text_input(
        "FastAPI Base URL",
        value=st.session_state.api_url,
        key="sidebar_api_url",
    )
    if api_url_input != st.session_state.api_url:
        st.session_state.api_url = api_url_input
        st.rerun()

    auto_refresh = st.checkbox("Auto-refresh active view (3s)", value=False, key="sidebar_auto_refresh")
    if st.button("🔄 Refresh Data Now", use_container_width=True):
        st.rerun()

    st.divider()
    st.caption("BMC Software Operations Interview Demo — Local Simulated Mode")


# ==============================================================================
# Helper Rendering Utilities
# ==============================================================================

def render_header(title: str, subtitle: str, extra_badge: str = ""):
    st.markdown(
        f"""
        <div class="nova-header">
            <div>
                <h1 class="nova-title">{title}</h1>
                <div class="nova-subtitle">{subtitle}</div>
            </div>
            <div>{extra_badge}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# VIEW 1: Operations Overview (Live Dashboard)
# ==============================================================================

if view_mode == "📊 Operations Overview":
    render_header(
        title="Operations Command Center",
        subtitle="Active Enterprise Batch Workload Health & Autonomous Operations",
    )

    incidents = api.list_incidents()
    total_count = len(incidents)
    active_count = sum(
        1
        for i in incidents
        if i.get("status")
        in (
            "TRIGGERED",
            "TRIAGING",
            "PLANNING",
            "EXECUTING",
            "VERIFYING",
            "REPLANNING",
            "AWAITING_APPROVAL",
        )
    )
    awaiting_approval = sum(
        1 for i in incidents if i.get("status") == "AWAITING_APPROVAL"
    )
    resolved_count = sum(1 for i in incidents if i.get("status") == "RESOLVED")
    escalated_count = sum(
        1 for i in incidents if i.get("status") in ("ESCALATED", "FAILED")
    )

    # Top KPI metric cards
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Incidents", total_count)
    with col2:
        st.metric("Active Remediations", active_count)
    with col3:
        st.metric("Awaiting Approval", awaiting_approval)
    with col4:
        st.metric("Resolved", resolved_count)
    with col5:
        st.metric("Escalated / Blocked", escalated_count)

    st.markdown("<br>", unsafe_allow_html=True)

    # Active incidents table
    st.markdown("#### **Active Incident Stream**")
    if not incidents:
        st.info("No incidents currently tracked. Inject a scenario or trigger an incident to begin.")
    else:
        for inc in incidents:
            st_class = inc.get("status", "UNKNOWN")
            inc_id = inc.get("incident_id")
            service = inc.get("target_service") or "UNKNOWN"
            fail_type = inc.get("failure_type") or "UNDER_INVESTIGATION"

            with st.container():
                c1, c2, c3, c4 = st.columns([2, 3, 2, 1.5])
                with c1:
                    st.markdown(f"**`{inc_id}`**")
                with c2:
                    st.caption(f"Target: `{service}` | Type: `{fail_type}`")
                with c3:
                    st.markdown(render_badge(st_class), unsafe_allow_html=True)
                with c4:
                    if st.button("Inspect Console", key=f"btn_inspect_{inc_id}"):
                        st.session_state.selected_incident_id = inc_id
                        st.session_state.view_mode_override = "🔍 Investigation & Remediation Console"
                        st.rerun()
                st.divider()


# ==============================================================================
# VIEW 2: Incident Trigger & Fault Injection
# ==============================================================================

elif view_mode == "🚨 Incident Trigger & Fault Injection":
    render_header(
        title="Incident Trigger & Synthetic Fault Injection",
        subtitle="Simulate Enterprise Batch Pipeline Failures & Trigger Autonomous Remediation",
    )

    tab_demo, tab_custom = st.tabs(
        ["🎯 Synthetic Scenario Fault Injection", "✍️ Custom Incident Submission"]
    )

    with tab_demo:
        st.markdown("Select a synthetic enterprise operational failure to inject into the mock pipeline environment:")

        sc1, sc2 = st.columns(2)
        with sc1:
            with st.container(border=True):
                st.markdown("##### 🔒 **Stale Lockfile Scenario**")
                st.caption("Orphaned process leaves a lockfile blocking `JOB_02_TRANSFORM`.")
                if st.button("Inject Stale Lockfile", use_container_width=True):
                    try:
                        res = api.inject_scenario("STALE_LOCKFILE")
                        st.success(f"Injected into {res['workflow_id']}: {res['error_message']}")
                        # Trigger corresponding incident
                        trig = api.trigger_incident(
                            raw_symptoms=f"FATAL: Lockfile 'transform_job.lock' detected on {res.get('failed_job')}.",
                            target_service=res.get("failed_job", "JOB_02_TRANSFORM"),
                        )
                        st.session_state.selected_incident_id = trig["incident_id"]
                        st.info(f"Incident triggered: `{trig['incident_id']}`. Navigate to Console to observe.")
                    except Exception as e:
                        st.error(f"Injection failed: {e}")

            with st.container(border=True):
                st.markdown("##### 🔄 **Cascading Multi-Fault**")
                st.caption("Stale lock on Job 2, followed by offline DB on Job 3.")
                if st.button("Inject Cascading Failure", use_container_width=True):
                    try:
                        res = api.inject_scenario("CASCADING_FAILURE")
                        st.success(f"Injected: {res['scenario']}")
                        trig = api.trigger_incident(
                            raw_symptoms="Pipeline stalled at step 2. Cascading failures suspected.",
                            target_service="JOB_02_TRANSFORM",
                        )
                        st.session_state.selected_incident_id = trig["incident_id"]
                        st.info(f"Incident triggered: `{trig['incident_id']}`")
                    except Exception as e:
                        st.error(f"Injection failed: {e}")

        with sc2:
            with st.container(border=True):
                st.markdown("##### 🔌 **Service Offline Scenario**")
                st.caption("Database warehouse service connection refused on port 5432.")
                if st.button("Inject Offline Service", use_container_width=True):
                    try:
                        res = api.inject_scenario("SERVICE_OFFLINE")
                        st.success(f"Injected into {res['workflow_id']}: {res['error_message']}")
                        trig = api.trigger_incident(
                            raw_symptoms=f"Connection refused to reporting_warehouse_service on port 5432.",
                            target_service=res.get("failed_job", "JOB_03_LOAD"),
                        )
                        st.session_state.selected_incident_id = trig["incident_id"]
                        st.info(f"Incident triggered: `{trig['incident_id']}`")
                    except Exception as e:
                        st.error(f"Injection failed: {e}")

            with st.container(border=True):
                st.markdown("##### 🛑 **Unrecoverable Data Corruption**")
                st.caption("Source feed contains zero-byte checksum mismatches. Must escalate.")
                if st.button("Inject Data Corruption", use_container_width=True):
                    try:
                        res = api.inject_scenario("DATA_CORRUPTION")
                        st.success(f"Injected: {res['scenario']}")
                        trig = api.trigger_incident(
                            raw_symptoms="CRITICAL: Checksum mismatch in input batch. Data corruption detected.",
                            target_service="JOB_01_EXTRACT",
                        )
                        st.session_state.selected_incident_id = trig["incident_id"]
                        st.info(f"Incident triggered: `{trig['incident_id']}`")
                    except Exception as e:
                        st.error(f"Injection failed: {e}")

    with tab_custom:
        st.markdown("Submit a natural language alert or log excerpt (target service may be left `UNKNOWN` for autonomous discovery):")
        with st.form("custom_incident_form"):
            custom_symptoms = st.text_area(
                "Symptom Description / Alert Content",
                value="The nightly financial reporting workflow did not complete. Investigate and resolve.",
                height=120,
            )
            col_t, col_r, col_s = st.columns(3)
            with col_t:
                custom_target = st.text_input("Target Service / Job ID", value="UNKNOWN")
            with col_r:
                custom_retries = st.number_input("Max Replanning Cycles", min_value=1, max_value=5, value=3)
            with col_s:
                custom_steps = st.number_input("Max Investigation Steps", min_value=1, max_value=10, value=5)

            col_env, col_eid = st.columns(2)
            with col_env:
                custom_env = st.selectbox("Execution Backend", ["mock", "kestra"], index=0)
            with col_eid:
                custom_eid = st.text_input("Kestra Execution ID (if kestra)", value="")

            submitted = st.form_submit_button("Submit Incident to NOVA", use_container_width=True)
            if submitted:
                try:
                    trigger_kwargs: Dict[str, Any] = {
                        "raw_symptoms": custom_symptoms,
                        "target_service": custom_target,
                        "max_retries": custom_retries,
                        "max_investigation_steps": custom_steps,
                    }
                    if custom_env == "kestra" or custom_eid.strip():
                        trigger_kwargs["environment"] = custom_env
                        if custom_eid.strip():
                            trigger_kwargs["execution_id"] = custom_eid.strip()
                    res = api.trigger_incident(**trigger_kwargs)
                    st.session_state.selected_incident_id = res["incident_id"]
                    st.success(f"Incident `{res['incident_id']}` accepted by Gateway!")
                except Exception as e:
                    st.error(f"Submission failed: {e}")


# ==============================================================================
# VIEW 3: Investigation & Remediation Console (Detail & Approvals)
# ==============================================================================

elif view_mode == "🔍 Investigation & Remediation Console":
    incidents = api.list_incidents()
    incident_ids = [i.get("incident_id") for i in incidents]

    selected_id = st.session_state.selected_incident_id
    if not selected_id and incident_ids:
        selected_id = incident_ids[0]
        st.session_state.selected_incident_id = selected_id

    # Top Selector Bar
    c_sel, c_stat, c_ref = st.columns([3, 2, 1])
    with c_sel:
        if incident_ids:
            cur_index = incident_ids.index(selected_id) if selected_id in incident_ids else 0
            chosen_id = st.selectbox(
                "Active Incident",
                incident_ids,
                index=cur_index,
                key="console_active_incident",
            )
            if chosen_id != selected_id:
                st.session_state.selected_incident_id = chosen_id
                st.rerun()
        else:
            st.info("No active incidents to display.")

    state = api.get_incident_state(st.session_state.selected_incident_id) if st.session_state.selected_incident_id else None

    if state:
        st_badge = render_badge(state.get("status", "UNKNOWN"))
        with c_stat:
            st.markdown(f"Status: {st_badge}", unsafe_allow_html=True)
        with c_ref:
            if st.button("Poll State"):
                st.rerun()

        st.divider()

        # Symptom Banner
        st.markdown(
            f"""
            <div class="nova-card">
                <div class="nova-card-header">
                    <span>Incident Context: <code>{state.get('incident_id')}</code></span>
                    <span>Target: <code>{state.get('target_service') or 'UNKNOWN'}</code></span>
                </div>
                <div style="font-size: 13px; color: #C9D1D9;"><b>Raw Symptoms:</b> {state.get('raw_symptoms')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ----------------------------------------------------------------------
        # HUMAN-IN-THE-LOOP APPROVAL GATE (Rendered prominently if paused)
        # ----------------------------------------------------------------------
        pending = state.get("pending_approval")
        if pending and state.get("status") == "AWAITING_APPROVAL":
            st.markdown(
                f"""
                <div class="approval-box">
                    <div style="font-size: 16px; font-weight: 700; color: #F59E0B; margin-bottom: 8px;">
                        ⚠️ Human-in-the-Loop Authorization Checkpoint
                    </div>
                    <div style="font-size: 13px; color: #FDE68A;">
                        NOVA has halted execution. A destructive tool call has been proposed and requires affirmative authorization.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.container(border=True):
                st.markdown("##### **Action Pending Authorization**")
                p1, p2, p3 = st.columns(3)
                with p1:
                    st.markdown(f"**Tool:** `{pending.get('tool_name')}`")
                with p2:
                    st.markdown(f"**Risk Level:** {render_risk_pill(pending.get('risk_level', 'DESTRUCTIVE'))}", unsafe_allow_html=True)
                with p3:
                    st.markdown(f"**Action ID:** `{pending.get('action_id')}`")

                st.markdown(f"**Parameters:** `{json.dumps(pending.get('parameters', {}))}`")
                if pending.get("action_hash") or pending.get("approval_request_id"):
                    st.caption(
                        f"Binding: `action_hash={pending.get('action_hash')}` | "
                        f"`approval_request_id={pending.get('approval_request_id')}`"
                    )
                st.markdown(f"**Rationale:** *{pending.get('rationale')}*")
                st.markdown(f"**Expected Outcome:** *{pending.get('expected_outcome')}*")

                st.divider()
                st.markdown("###### **Simulated Operator Decision (Local Demo Only — Self-Asserted RBAC)**")
                op1, op2 = st.columns(2)
                with op1:
                    approver_id = st.text_input(
                        "Operator Identifier",
                        value="SIMULATED_SRE_LEAD_01",
                        key="hitl_approver_id",
                    )
                with op2:
                    approver_role = st.selectbox(
                        "Authorized Role (RBAC)",
                        ["SRE_LEAD", "PLATFORM_ADMIN", "L2_OPERATIONS", "INCIDENT_COMMANDER"],
                        index=0,
                        key="hitl_approver_role",
                    )
                notes = st.text_input(
                    "Audit Notes",
                    value="Verified PID status; authorized operation for batch recovery.",
                    key="hitl_audit_notes",
                )

                btn_app, btn_rej = st.columns(2)
                with btn_app:
                    if st.button("✅ Authorize Action", use_container_width=True, type="primary"):
                        try:
                            api.submit_approval(
                                incident_id=state["incident_id"],
                                action_id=pending["action_id"],
                                approved=True,
                                approver_id=approver_id,
                                approver_role=approver_role,
                                notes=notes,
                                action_hash=pending.get("action_hash"),
                            )
                            st.success("Action approved! Graph resumed.")
                            time.sleep(1)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Approval failed: {e}")

                with btn_rej:
                    if st.button("❌ Reject & Halt Action", use_container_width=True):
                        try:
                            api.submit_approval(
                                incident_id=state["incident_id"],
                                action_id=pending["action_id"],
                                approved=False,
                                approver_id=approver_id,
                                approver_role=approver_role,
                                notes=notes,
                                action_hash=pending.get("action_hash"),
                            )
                            st.warning("Action rejected! Graph escalated.")
                            time.sleep(1)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Rejection failed: {e}")

        # ----------------------------------------------------------------------
        # Autonomous Investigation Timeline
        # ----------------------------------------------------------------------
        st.markdown("#### **1. Autonomous Investigation Telemetry**")
        i_col1, i_col2 = st.columns([1, 2])
        with i_col1:
            with st.container(border=True):
                st.markdown("##### **Current Diagnosis**")
                st.markdown(f"**Failure Type:** `{state.get('failure_type') or 'UNDER_INVESTIGATION'}`")
                st.markdown(f"**Implicated Resource:** `{state.get('implicated_resource') or 'UNKNOWN'}`")
                st.markdown(f"**Recoverable:** `{state.get('is_recoverable')}`")
                st.markdown(f"**Investigation Steps:** `{state.get('investigation_steps')}/{state.get('max_investigation_steps')}`")
                if state.get("triage_summary"):
                    st.caption(f"**Summary:** {state.get('triage_summary')}")
                if state.get("root_cause_hypothesis"):
                    st.caption(f"**Hypothesis:** {state.get('root_cause_hypothesis')}")

        with i_col2:
            with st.container(border=True):
                st.markdown("##### **Diagnostic Tool Execution History**")
                diag_history = state.get("diagnostic_history", [])
                if not diag_history:
                    st.caption("No diagnostic tools executed yet.")
                else:
                    for d in diag_history:
                        tname = d.get("tool_name", "tool")
                        params = json.dumps(d.get("parameters", {}))
                        output = d.get("output", "")
                        st.markdown(f"- **`{tname}`** params: `{params}`")
                        with st.expander(f"Output for {tname}"):
                            st.markdown(f'<div class="nova-mono">{output}</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ----------------------------------------------------------------------
        # Remediation Plan & Multi-Step Execution
        # ----------------------------------------------------------------------
        st.markdown("#### **2. Remediation Plan & Execution Records**")
        plan = state.get("plan", [])
        exec_history = state.get("execution_history", [])

        if not plan:
            st.info("No active remediation plan formulated yet.")
        else:
            for idx, action in enumerate(plan):
                act_id = action.get("action_id", f"ACT_{idx+1}")
                tool_name = action.get("tool_name")
                risk = action.get("risk_level", "LOW_RISK")
                rationale = action.get("rationale", "")
                params = json.dumps(action.get("parameters", {}))

                # Match strictly by action_id so prior-cycle actions do not alias current steps
                exec_record = next((e for e in exec_history if e.get("action_id") == act_id), None)

                with st.container(border=True):
                    h1, h2, h3 = st.columns([2, 3, 2])
                    with h1:
                        st.markdown(f"**Step {idx+1}: `{tool_name}`**")
                    with h2:
                        st.caption(f"Params: `{params}`")
                    with h3:
                        st.markdown(render_risk_pill(risk), unsafe_allow_html=True)

                    st.markdown(f"*{rationale}*")

                    if exec_record:
                        succ = exec_record.get("success", False)
                        exit_c = exec_record.get("exit_code", 0)
                        dur = exec_record.get("duration_seconds", 0.0)
                        stat_color = "green" if succ else "red"
                        st.markdown(f":{stat_color}[**Executed (exit code {exit_c}, {dur:.2f}s)**]")
                        with st.expander("View Output & Audit Details"):
                            st.markdown(f'<div class="nova-mono">{exec_record.get("stdout") or exec_record.get("stderr") or "No output."}</div>', unsafe_allow_html=True)
                    else:
                        st.caption("Status: Proposed (pending execution)")

        st.markdown("<br>", unsafe_allow_html=True)

        # ----------------------------------------------------------------------
        # Replanning & Independent Verification
        # ----------------------------------------------------------------------
        st.markdown("#### **3. Independent Verification & Replanning History**")
        verif_history = state.get("verification_history", [])
        if not verif_history:
            st.caption("Independent downstream workflow verification has not yet run.")
        else:
            for v in verif_history:
                v_pass = v.get("passed", False)
                v_color = "green" if v_pass else "orange"
                st.markdown(f":{v_color}[**Post-Condition Verification (Passed: {v_pass})**]")
                st.caption(f"Evidence: {v.get('evidence')}")
                if v.get("details"):
                    with st.expander("Verification Details"):
                        st.json(v.get("details"))

        if state.get("interrupted_by_restart"):
            st.warning(
                "Background worker is no longer active for this non-terminal incident "
                "(likely interrupted by a backend restart before reaching a checkpointed pause or terminal state)."
            )

        if state.get("final_summary"):
            if state.get("status") in ("ESCALATED", "FAILED"):
                st.error(f"**Escalation / Failure Summary:** {state.get('final_summary')}")
            else:
                st.success(f"**Final Resolution Summary:** {state.get('final_summary')}")

        # ----------------------------------------------------------------------
        # Lightweight Observability & Approval Audit Trail
        # ----------------------------------------------------------------------
        st.markdown("#### **4. Execution Telemetry & Approval Audit Trail**")
        t_col, a_col = st.columns(2)
        with t_col:
            telem = state.get("telemetry") or {}
            with st.container(border=True):
                st.markdown("##### **Operational Telemetry**")
                if not telem:
                    st.caption("No telemetry recorded for this incident yet.")
                else:
                    m1, m2, m3 = st.columns(3)
                    m1.metric("LLM Requests", telem.get("llm_request_count", 0))
                    m2.metric("Diagnostic Tools", telem.get("diagnostic_tool_count", 0))
                    m3.metric("Remediation Tools", telem.get("remediation_tool_count", 0))
                    st.caption(
                        f"Retries: `{telem.get('llm_retry_count', 0)}` | "
                        f"Rate Limits: `{telem.get('llm_rate_limit_count', 0)}` | "
                        f"Duration: `{telem.get('execution_duration_seconds', 0.0):.2f}s`"
                    )
        with a_col:
            audits = state.get("approval_audit_history") or []
            with st.container(border=True):
                st.markdown("##### **Approval Audit Records**")
                if not audits:
                    st.caption("No approval/rejection events recorded.")
                else:
                    for aud in audits:
                        st.markdown(
                            f"- `{aud.get('outcome')}` | `{aud.get('action_id')}` by "
                            f"`{aud.get('approver_id')}` (`{aud.get('approver_role')}`)"
                        )

    # Auto-refresh loop if enabled
    if auto_refresh and state and state.get("status") not in ("RESOLVED", "ESCALATED", "FAILED"):
        time.sleep(3)
        st.rerun()


# ==============================================================================
# VIEW 4: Incident Catalog & Audit Trails
# ==============================================================================

elif view_mode == "📜 Incident Catalog & Audit Trails":
    render_header(
        title="Incident Catalog & Persistent Audit Trails",
        subtitle="Historical Incident Registry from Durable SQLite Checkpoint Storage",
    )

    incidents = api.list_incidents()
    if not incidents:
        st.info("No recorded incidents found in checkpointer storage.")
    else:
        st.dataframe(
            [
                {
                    "Incident ID": i.get("incident_id"),
                    "Status": i.get("status"),
                    "Target Service": i.get("target_service"),
                    "Failure Type": i.get("failure_type"),
                    "Created At": i.get("created_at"),
                }
                for i in incidents
            ],
            use_container_width=True,
        )

        st.divider()
        st.markdown("##### **Download Complete Audit Payload**")
        if st.session_state.selected_incident_id:
            full_state = api.get_incident_state(st.session_state.selected_incident_id)
            if full_state:
                st.download_button(
                    label=f"Export JSON for {st.session_state.selected_incident_id}",
                    data=json.dumps(full_state, indent=2),
                    file_name=f"audit_{st.session_state.selected_incident_id}.json",
                    mime="application/json",
                    use_container_width=True,
                )
