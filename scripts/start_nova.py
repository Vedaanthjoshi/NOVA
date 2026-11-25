"""
Unified local startup script for NOVA (FastAPI Gateway + Streamlit Dashboard).

Usage:
    python scripts/start_nova.py

Starts:
1. FastAPI Backend Gateway on http://127.0.0.1:8000
2. Streamlit Operator Dashboard on http://127.0.0.1:8501

Press Ctrl+C to shut down both processes cleanly.
Note: Intended for local simulated operation only.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


def main() -> int:
    env = os.environ.copy()
    src_path = str(ROOT_DIR / "src")
    env["PYTHONPATH"] = (
        f"{src_path}{os.pathsep}{env['PYTHONPATH']}"
        if env.get("PYTHONPATH")
        else src_path
    )

    api_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "nova.api.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
    ]
    ui_cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(ROOT_DIR / "src" / "nova" / "ui" / "app.py"),
        "--server.port",
        "8501",
        "--server.address",
        "127.0.0.1",
        "--server.headless",
        "true",
    ]

    print("[NOVA] Starting FastAPI Gateway at http://127.0.0.1:8000 ...")
    api_proc = subprocess.Popen(api_cmd, cwd=str(ROOT_DIR), env=env)
    time.sleep(2.0)

    print("[NOVA] Starting Streamlit Dashboard at http://127.0.0.1:8501 ...")
    ui_proc = subprocess.Popen(ui_cmd, cwd=str(ROOT_DIR), env=env)

    print("[NOVA] Both services launched. Press Ctrl+C to stop.")
    try:
        while True:
            if api_proc.poll() is not None:
                print(f"[NOVA] FastAPI exited with code {api_proc.returncode}")
                break
            if ui_proc.poll() is not None:
                print(f"[NOVA] Streamlit exited with code {ui_proc.returncode}")
                break
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[NOVA] Shutting down services...")
    finally:
        for proc in (ui_proc, api_proc):
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
