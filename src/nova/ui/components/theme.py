"""
Enterprise Operations Theme & Styling for NOVA Streamlit Dashboard.

Adheres strictly to the BMC Software design direction:
- Professional enterprise dark theme.
- Clean typography and structured card layouts.
- Monospace font for telemetry, file paths, and hashes.
- No playful gradients, animations, or unnecessary decorative clutter.
"""

ENTERPRISE_DARK_CSS = """
<style>
/* Main container styling */
.main {
    background-color: #0E1117;
    color: #E6EDF3;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #161B22;
    border-right: 1px solid #30363D;
}

/* Header & Banner */
.nova-header {
    background: #161B22;
    border-bottom: 2px solid #30363D;
    padding: 18px 24px;
    margin-bottom: 24px;
    border-radius: 6px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.nova-title {
    font-size: 24px;
    font-weight: 700;
    color: #F0F6FC;
    letter-spacing: -0.5px;
    margin: 0;
}

.nova-subtitle {
    font-size: 13px;
    color: #8B949E;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Card components */
.nova-card {
    background-color: #161B22;
    border: 1px solid #30363D;
    border-radius: 6px;
    padding: 16px 20px;
    margin-bottom: 16px;
}

.nova-card-header {
    font-size: 14px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #8B949E;
    margin-bottom: 12px;
    border-bottom: 1px solid #21262D;
    padding-bottom: 8px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

/* Metric styling */
div[data-testid="stMetricValue"] {
    font-size: 26px !important;
    font-weight: 700 !important;
    color: #F0F6FC !important;
}

div[data-testid="stMetricLabel"] {
    font-size: 12px !important;
    color: #8B949E !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Monospace code / log view */
.nova-mono {
    font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace;
    font-size: 12px;
    background-color: #0D1117;
    border: 1px solid #30363D;
    border-radius: 4px;
    padding: 10px 14px;
    color: #C9D1D9;
    white-space: pre-wrap;
    word-break: break-all;
}

/* Status Badges */
.badge {
    display: inline-block;
    padding: 3px 10px;
    font-size: 11px;
    font-weight: 600;
    line-height: 1.4;
    border-radius: 12px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.badge-triggered { background-color: #21262D; color: #8B949E; border: 1px solid #30363D; }
.badge-triaging { background-color: #1F2937; color: #60A5FA; border: 1px solid #3B82F6; }
.badge-planning { background-color: #1E293B; color: #93C5FD; border: 1px solid #60A5FA; }
.badge-awaiting_approval { background-color: #422006; color: #FBBF24; border: 1px solid #F59E0B; }
.badge-executing { background-color: #082F49; color: #38BDF8; border: 1px solid #0EA5E9; }
.badge-verifying { background-color: #172554; color: #818CF8; border: 1px solid #6366F1; }
.badge-replanning { background-color: #3B0764; color: #C084FC; border: 1px solid #A855F7; }
.badge-resolved { background-color: #052E16; color: #4ADE80; border: 1px solid #22C55E; }
.badge-escalated { background-color: #450A0A; color: #F87171; border: 1px solid #EF4444; }
.badge-failed { background-color: #2B0B0B; color: #FCA5A5; border: 1px solid #DC2626; }

/* Risk ratings */
.risk-read_only { color: #60A5FA; font-weight: 600; font-size: 11px; }
.risk-low_risk { color: #34D399; font-weight: 600; font-size: 11px; }
.risk-destructive { color: #F87171; font-weight: 700; font-size: 11px; }

/* Approval warning box */
.approval-box {
    background-color: #2D1500;
    border: 1px solid #D97706;
    border-radius: 6px;
    padding: 16px 20px;
    margin: 16px 0;
}
</style>
"""


def render_badge(status_str: str) -> str:
    """Returns HTML markup for a styled incident status badge."""
    clean_status = (status_str or "UNKNOWN").upper().replace("INCIDENTSTATUS.", "")
    css_class = f"badge-{clean_status.lower()}"
    return f'<span class="badge {css_class}">{clean_status}</span>'


def render_risk_pill(risk_str: str) -> str:
    """Returns HTML markup for a tool safety risk level rating."""
    clean_risk = (risk_str or "LOW_RISK").upper().replace("RISKLEVEL.", "")
    css_class = f"risk-{clean_risk.lower()}"
    return f'<span class="{css_class}">[{clean_risk}]</span>'
