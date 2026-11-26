"""
Lightweight Structured Observability & Telemetry for NOVA.

Uses Python standard library (`logging`, `contextvars`, `json`, `threading`)
without external observability infrastructure.

Provides:
- Incident correlation IDs via `contextvars`.
- Secret-safe structured JSON logging (explicitly redacts API keys/tokens).
- Per-incident and global counters for LLM calls, retries, rate limits,
  diagnostic vs. remediation tool invocations, durations, approval audits,
  and clear failure/escalation reasons.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional

_incident_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "nova_incident_id", default=None
)

_SENSITIVE_KEY_RE = re.compile(
    r"(api[_-]?key|secret|token|password|credential|authorization)", re.IGNORECASE
)
_SECRET_VALUE_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z_-]{20,}"),
    re.compile(r"sk-[0-9A-Za-z_-]{20,}"),
    re.compile(r"(Bearer\s+)[A-Za-z0-9._-]+", re.IGNORECASE),
]


def redact_secrets(text: str) -> str:
    """Scrubs potential API keys or bearer tokens from log strings."""
    if not text:
        return text
    cleaned = text
    for pattern in _SECRET_VALUE_PATTERNS:
        if pattern.groups:
            cleaned = pattern.sub(r"\1[REDACTED]", cleaned)
        else:
            cleaned = pattern.sub("[REDACTED]", cleaned)
    return cleaned


def sanitize_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively redacts sensitive keys and secret patterns from structured fields."""
    sanitized: Dict[str, Any] = {}
    for key, val in data.items():
        if _SENSITIVE_KEY_RE.search(str(key)):
            sanitized[key] = "[REDACTED]"
        elif isinstance(val, dict):
            sanitized[key] = sanitize_fields(val)
        elif isinstance(val, str):
            sanitized[key] = redact_secrets(val)
        elif isinstance(val, list):
            sanitized[key] = [
                sanitize_fields(v)
                if isinstance(v, dict)
                else (redact_secrets(v) if isinstance(v, str) else v)
                for v in val
            ]
        else:
            sanitized[key] = val
    return sanitized


def get_current_incident_id() -> Optional[str]:
    """Returns the incident correlation ID bound to the current execution context."""
    return _incident_id_var.get()


@contextlib.contextmanager
def incident_context(incident_id: Optional[str]) -> Iterator[None]:
    """Binds an incident_id correlation token for the duration of the block."""
    if not incident_id:
        yield
        return
    token = _incident_id_var.set(incident_id)
    try:
        yield
    finally:
        _incident_id_var.reset(token)


class StructuredJsonFormatter(logging.Formatter):
    """Formats log records as single-line structured JSON with correlation ID."""

    def format(self, record: logging.LogRecord) -> str:
        incident_id = getattr(record, "incident_id", None) or get_current_incident_id()
        event = getattr(record, "event", None)
        fields = getattr(record, "fields", None)

        payload: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "incident_id": incident_id,
            "message": redact_secrets(record.getMessage()),
        }
        if event:
            payload["event"] = event
        if isinstance(fields, dict) and fields:
            payload["fields"] = sanitize_fields(fields)
        if record.exc_info:
            payload["exception"] = redact_secrets(self.formatException(record.exc_info))
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> logging.Logger:
    """Configures structured logging for the `nova` logger namespace idempotently."""
    root_nova = logging.getLogger("nova")
    numeric_level = getattr(logging, str(level).upper(), logging.INFO)
    root_nova.setLevel(numeric_level)

    has_structured_handler = any(
        isinstance(getattr(h, "formatter", None), StructuredJsonFormatter)
        for h in root_nova.handlers
    )
    if not has_structured_handler:
        handler = logging.StreamHandler()
        handler.setFormatter(StructuredJsonFormatter())
        root_nova.addHandler(handler)
    return root_nova


def log_event(
    logger: logging.Logger,
    level: int,
    event: str,
    message: str,
    incident_id: Optional[str] = None,
    **fields: Any,
) -> None:
    """Emits a structured log event with incident correlation and secret scrubbing."""
    corr_id = incident_id or get_current_incident_id()
    logger.log(
        level,
        redact_secrets(message),
        extra={
            "incident_id": corr_id,
            "event": event,
            "fields": sanitize_fields(fields),
        },
    )


@dataclass
class IncidentTelemetry:
    """Per-incident operational telemetry counters and timings."""

    incident_id: str
    llm_request_count: int = 0
    llm_retry_count: int = 0
    llm_rate_limit_count: int = 0
    llm_error_count: int = 0
    llm_duration_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    diagnostic_tool_count: int = 0
    remediation_tool_count: int = 0
    tool_failure_count: int = 0
    tool_duration_seconds: float = 0.0
    execution_duration_seconds: float = 0.0
    failure_reason: Optional[str] = None
    escalation_reason: Optional[str] = None
    approval_audit_count: int = 0
    started_at: Optional[float] = None
    events: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "llm_request_count": self.llm_request_count,
            "llm_retry_count": self.llm_retry_count,
            "llm_rate_limit_count": self.llm_rate_limit_count,
            "llm_error_count": self.llm_error_count,
            "llm_duration_seconds": round(self.llm_duration_seconds, 4),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "diagnostic_tool_count": self.diagnostic_tool_count,
            "remediation_tool_count": self.remediation_tool_count,
            "tool_failure_count": self.tool_failure_count,
            "tool_duration_seconds": round(self.tool_duration_seconds, 4),
            "execution_duration_seconds": round(self.execution_duration_seconds, 4),
            "failure_reason": self.failure_reason,
            "escalation_reason": self.escalation_reason,
            "approval_audit_count": self.approval_audit_count,
        }


class TelemetryRegistry:
    """Thread-safe in-process telemetry collector keyed by incident_id."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._incidents: Dict[str, IncidentTelemetry] = {}

    def _get_or_create(self, incident_id: str) -> IncidentTelemetry:
        entry = self._incidents.get(incident_id)
        if entry is None:
            entry = IncidentTelemetry(incident_id=incident_id, started_at=time.perf_counter())
            self._incidents[incident_id] = entry
        return entry

    def reset_incident(self, incident_id: str) -> None:
        with self._lock:
            self._incidents[incident_id] = IncidentTelemetry(
                incident_id=incident_id, started_at=time.perf_counter()
            )

    def hydrate_incident(self, incident_id: str, data: Dict[str, Any]) -> None:
        """Restores persisted telemetry counters into memory if not already present."""
        if not data:
            return
        with self._lock:
            if incident_id in self._incidents:
                return
            self._incidents[incident_id] = IncidentTelemetry(
                incident_id=incident_id,
                llm_request_count=int(data.get("llm_request_count", 0)),
                llm_retry_count=int(data.get("llm_retry_count", 0)),
                llm_rate_limit_count=int(data.get("llm_rate_limit_count", 0)),
                llm_error_count=int(data.get("llm_error_count", 0)),
                llm_duration_seconds=float(data.get("llm_duration_seconds", 0.0)),
                input_tokens=int(data.get("input_tokens", 0)),
                output_tokens=int(data.get("output_tokens", 0)),
                total_tokens=int(data.get("total_tokens", 0)),
                diagnostic_tool_count=int(data.get("diagnostic_tool_count", 0)),
                remediation_tool_count=int(data.get("remediation_tool_count", 0)),
                tool_failure_count=int(data.get("tool_failure_count", 0)),
                tool_duration_seconds=float(data.get("tool_duration_seconds", 0.0)),
                execution_duration_seconds=float(data.get("execution_duration_seconds", 0.0)),
                failure_reason=data.get("failure_reason"),
                escalation_reason=data.get("escalation_reason"),
                approval_audit_count=int(data.get("approval_audit_count", 0)),
                started_at=time.perf_counter(),
            )

    def record_llm_call(
        self,
        incident_id: Optional[str],
        operation: str,
        model_name: str,
        duration_seconds: float,
        retries: int = 0,
        rate_limits: int = 0,
        success: bool = True,
        error: Optional[str] = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: int = 0,
    ) -> None:
        key = incident_id or get_current_incident_id() or "_global"
        with self._lock:
            m = self._get_or_create(key)
            m.llm_request_count += 1
            m.llm_retry_count += retries
            m.llm_rate_limit_count += rate_limits
            m.input_tokens += max(0, input_tokens)
            m.output_tokens += max(0, output_tokens)
            m.total_tokens += max(0, total_tokens or (input_tokens + output_tokens))
            if not success:
                m.llm_error_count += 1
                if error:
                    m.failure_reason = redact_secrets(error)
            m.llm_duration_seconds += max(0.0, duration_seconds)
            m.events.append(
                {
                    "kind": "llm_call",
                    "operation": operation,
                    "model": model_name,
                    "duration_seconds": round(duration_seconds, 4),
                    "retries": retries,
                    "rate_limits": rate_limits,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens or (input_tokens + output_tokens),
                    "success": success,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

    def record_tool_call(
        self,
        incident_id: Optional[str],
        tool_name: str,
        risk_level: str,
        is_diagnostic: bool,
        duration_seconds: float,
        success: bool,
    ) -> None:
        key = incident_id or get_current_incident_id() or "_global"
        with self._lock:
            m = self._get_or_create(key)
            if is_diagnostic:
                m.diagnostic_tool_count += 1
            else:
                m.remediation_tool_count += 1
            if not success:
                m.tool_failure_count += 1
            m.tool_duration_seconds += max(0.0, duration_seconds)
            m.events.append(
                {
                    "kind": "tool_call",
                    "tool_name": tool_name,
                    "risk_level": risk_level,
                    "is_diagnostic": is_diagnostic,
                    "duration_seconds": round(duration_seconds, 4),
                    "success": success,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

    def record_approval_event(
        self,
        incident_id: Optional[str],
        outcome: str,
    ) -> None:
        key = incident_id or get_current_incident_id() or "_global"
        with self._lock:
            m = self._get_or_create(key)
            m.approval_audit_count += 1
            m.events.append(
                {
                    "kind": "approval_audit",
                    "outcome": outcome,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

    def record_execution_duration(
        self,
        incident_id: str,
        duration_seconds: float,
    ) -> None:
        with self._lock:
            m = self._get_or_create(incident_id)
            m.execution_duration_seconds += max(0.0, duration_seconds)

    def record_terminal_reason(
        self,
        incident_id: Optional[str],
        status: str,
        reason: Optional[str],
    ) -> None:
        key = incident_id or get_current_incident_id() or "_global"
        with self._lock:
            m = self._get_or_create(key)
            if reason:
                clean = redact_secrets(reason)
                if status == "ESCALATED":
                    m.escalation_reason = clean
                elif status == "FAILED":
                    m.failure_reason = clean

    def get_incident_metrics(self, incident_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            m = self._incidents.get(incident_id)
            return m.to_dict() if m else None

    def get_summary(self) -> Dict[str, Any]:
        with self._lock:
            incidents = [
                m.to_dict() for k, m in self._incidents.items() if k != "_global"
            ]
            return {
                "tracked_incidents": len(incidents),
                "total_llm_requests": sum(i["llm_request_count"] for i in incidents),
                "total_llm_retries": sum(i["llm_retry_count"] for i in incidents),
                "total_llm_rate_limits": sum(i["llm_rate_limit_count"] for i in incidents),
                "total_llm_errors": sum(i["llm_error_count"] for i in incidents),
                "total_diagnostic_tools": sum(i["diagnostic_tool_count"] for i in incidents),
                "total_remediation_tools": sum(i["remediation_tool_count"] for i in incidents),
                "total_approval_audits": sum(i["approval_audit_count"] for i in incidents),
            }


# Global telemetry registry singleton
telemetry = TelemetryRegistry()
