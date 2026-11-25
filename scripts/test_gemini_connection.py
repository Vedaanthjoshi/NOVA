"""
Phase A Live Gemini Baseline Verification Entry Point.

This script independently verifies that GeminiLLMClient can communicate with
the Google Gemini API and generate valid structured Pydantic models.

Usage (outside restricted sandbox):
    PowerShell:
        $env:GEMINI_API_KEY="your-api-key-here"
        python scripts/test_gemini_connection.py

    Bash:
        export GEMINI_API_KEY="your-api-key-here"
        python scripts/test_gemini_connection.py

Security Note:
    API keys are NEVER exposed, printed, or saved to disk.
"""
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path so scripts can run directly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

from nova.config import settings
from nova.llm.client import GeminiLLMClient
from nova.llm.schemas import PlannerResult, TriageResult
from nova.nodes.planner import AVAILABLE_TOOLS_FOR_LLM


def run_live_gemini_baseline() -> int:
    print("=" * 70)
    print("NOVA Milestone 7.2: Phase A — Live Gemini Baseline Verification")
    print("=" * 70)

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or settings.gemini_api_key

    if not api_key or not api_key.strip():
        print("\n[LIVE EVALUATION STATUS: PENDING]")
        print("Reason: No Gemini API key detected in GEMINI_API_KEY, GOOGLE_API_KEY, or .env.")
        print("Note: Live evaluation is deliberately separated from deterministic unit/integration tests.")
        print("\nTo execute this live baseline outside the restricted sandbox:")
        print("  1. In PowerShell:")
        print('     $env:GEMINI_API_KEY = "your-actual-gemini-api-key"')
        print("     python scripts/test_gemini_connection.py")
        print("  2. In Bash/Linux:")
        print('     export GEMINI_API_KEY="your-actual-gemini-api-key"')
        print("     python scripts/test_gemini_connection.py")
        print("\nSafety Confirmation: Zero mock results substituted. Real API call skipped.")
        return 0

    masked_key = api_key[:4] + "..." + api_key[-4:] if len(api_key) > 8 else "***"
    print(f"\n[INFO] Gemini API Key detected: {masked_key}")
    print(f"[INFO] Initializing GeminiLLMClient with default model: {settings.default_model}")

    try:
        client = GeminiLLMClient(api_key=api_key, model_name=settings.default_model)
    except Exception as exc:
        print(f"[ERROR] Failed to initialize GeminiLLMClient: {exc}")
        return 1

    sample_log = (
        "[2026-11-24 10:15:02] INFO: Starting batch transform job JOB_02_TRANSFORM\n"
        "[2026-11-24 10:15:03] ERROR: Cannot acquire exclusive filesystem lock.\n"
        "[2026-11-24 10:15:03] FATAL: Stale lockfile 'transform_job.lock' detected with PID 9182.\n"
        "[2026-11-24 10:15:03] FATAL: Process aborted with exit code 1.\n"
    )

    print("\n--- Step 1: Testing Live Structured Triage ---")
    start_t = time.perf_counter()
    try:
        triage_res = client.triage(target_service="JOB_02_TRANSFORM", log_content=sample_log)
        duration_triage = time.perf_counter() - start_t
        assert isinstance(triage_res, TriageResult)
        print(f"[SUCCESS] Triage completed in {duration_triage:.2f}s")
        print(f"  - Failure Type: {triage_res.failure_type}")
        print(f"  - Implicated Resource: {triage_res.implicated_resource}")
        print(f"  - Is Recoverable: {triage_res.is_recoverable}")
        print(f"  - Confidence: {triage_res.confidence}")
        print(f"  - Summary: {triage_res.triage_summary}")
    except Exception as exc:
        print(f"[FAIL] Triage API call failed: {exc}")
        return 1

    print("\n--- Step 2: Testing Live Structured Planning ---")
    start_p = time.perf_counter()
    try:
        plan_res = client.plan(
            triage=triage_res,
            available_tools=AVAILABLE_TOOLS_FOR_LLM,
            retry_count=0,
        )
        duration_plan = time.perf_counter() - start_p
        assert isinstance(plan_res, PlannerResult)
        print(f"[SUCCESS] Planning completed in {duration_plan:.2f}s")
        print(f"  - Is Escalation: {plan_res.is_escalation}")
        print(f"  - Planned Actions Count: {len(plan_res.actions)}")
        for idx, act in enumerate(plan_res.actions, 1):
            print(f"    Action {idx}: {act.tool_name} (Risk: {act.risk_level.value})")
            print(f"      Parameters: {act.parameters}")
            print(f"      Rationale: {act.rationale}")
    except Exception as exc:
        print(f"[FAIL] Planning API call failed: {exc}")
        return 1

    print("\n" + "=" * 70)
    print("PHASE A BASELINE VERIFICATION: PASSED (Live Gemini API verified)")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(run_live_gemini_baseline())
