"""
Unit tests for NOVA lightweight observability, secret redaction, correlation IDs,
and Gemini rate-limit/retry telemetry.
"""

import logging
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from nova.llm.client import GeminiLLMClient
from nova.llm.schemas import TriageResult
from nova.observability import (
    StructuredJsonFormatter,
    incident_context,
    log_event,
    redact_secrets,
    sanitize_fields,
    telemetry,
)


def test_secret_redaction_and_field_sanitization():
    raw = "Auth failed with AIzaSyDummySecretKey1234567890abcdef and Bearer tok_abc123"
    scrubbed = redact_secrets(raw)
    assert "AIzaSyDummySecretKey" not in scrubbed
    assert "tok_abc123" not in scrubbed
    assert "[REDACTED]" in scrubbed

    fields = sanitize_fields(
        {
            "gemini_api_key": "AIzaSySecret99999999999999999999",
            "model": "gemini-3.5-flash-lite",
            "nested": {"token": "abc", "note": "Bearer xyz123"},
        }
    )
    assert fields["gemini_api_key"] == "[REDACTED]"
    assert fields["nested"]["token"] == "[REDACTED]"
    assert "xyz123" not in fields["nested"]["note"]
    assert fields["model"] == "gemini-3.5-flash-lite"


def test_structured_formatter_includes_incident_correlation_id():
    formatter = StructuredJsonFormatter()
    record = logging.LogRecord(
        name="nova.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="Testing AIzaSySecretKey12345678901234567890",
        args=(),
        exc_info=None,
    )
    with incident_context("INC_CORR_42"):
        formatted = formatter.format(record)
    assert '"incident_id": "INC_CORR_42"' in formatted
    assert "AIzaSySecretKey" not in formatted
    assert "[REDACTED]" in formatted


def test_gemini_client_rate_limit_retry_and_telemetry(monkeypatch):
    """Verifies GeminiLLMClient records rate-limit retries and masks secrets on errors."""
    fake_genai = MagicMock()
    fake_genai.GenerationConfig = lambda **kwargs: kwargs

    calls = {"count": 0}

    def fake_generate_content(prompt):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("429 Quota exceeded; retry_delay { seconds: 1 }")
        return SimpleNamespace(
            text=(
                '{"failure_type": "STALE_LOCKFILE", '
                '"implicated_resource": "transform_job.lock", '
                '"triage_summary": "Lockfile present.", '
                '"root_cause_hypothesis": "Orphaned lock.", '
                '"is_recoverable": true, '
                '"confidence": "HIGH"}'
            )
        )

    fake_model = MagicMock()
    fake_model.generate_content.side_effect = fake_generate_content
    fake_genai.GenerativeModel.return_value = fake_model

    monkeypatch.setitem(sys.modules, "google.generativeai", fake_genai)
    monkeypatch.setattr("nova.llm.client.time.sleep", lambda _: None)

    telemetry.reset_incident("INC_GEM_OBS_01")
    client = GeminiLLMClient(api_key="AIzaSyFakeTestKey123456789012345678")

    with incident_context("INC_GEM_OBS_01"):
        res = client.triage("JOB_02_TRANSFORM", "Lockfile detected")

    assert isinstance(res, TriageResult)
    assert res.failure_type == "STALE_LOCKFILE"

    metrics = telemetry.get_incident_metrics("INC_GEM_OBS_01")
    assert metrics is not None
    assert metrics["llm_request_count"] == 1
    assert metrics["llm_retry_count"] == 1
    assert metrics["llm_rate_limit_count"] == 1
    assert metrics["llm_error_count"] == 0
    assert metrics["llm_duration_seconds"] >= 0.0
