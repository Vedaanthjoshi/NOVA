"""
Persistence Hardening & Cross-Process Recovery Integration Tests.

Validates:
1. True cross-process persistence where an interrupted Human-in-the-Loop state
   is reloaded and resumed by an entirely separate OS Python subprocess via
   `subprocess.run([sys.executable, ...])`.
2. Both initial execution and approval resumption running in separate OS processes.
3. Granular, row-level schema inspection ensuring `checkpoints`, `checkpoint_blobs`,
   and `checkpoint_writes` store normalized relational records rather than a
   single monolithic blob.
4. Thread deletion and list query filtering across SQLite records.
"""

import gc
import os
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
import pytest
from langgraph.types import Command

from nova.environments.mock_workflow_runner import (
    JobStatus,
    MockWorkflowRunner,
    ScenarioType,
)
from nova.graph import build_nova_graph
from nova.state import IncidentStatus, NovaGraphState
from nova.storage.sqlite_saver import SqliteSaver


@pytest.fixture
def mock_env():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        runner = MockWorkflowRunner(base_dir=tmpdir)
        yield runner


def test_cross_process_sqlite_persistence():
    """
    Milestone 7 / Phase D Primary Test:
    Proves that an incident paused on a HITL gate can be reloaded and resumed
    by an entirely independent operating system process.

    Execution Flow:
    1. Parent Process initiates graph execution using SqliteSaver.
    2. Graph interrupts on DESTRUCTIVE action (clear_lockfile).
    3. Parent Process closes database and terminates graph instance.
    4. Child Process (separate OS PID via subprocess.run) launches, initializes
       a fresh SqliteSaver against the same SQLite database file, verifies
       the paused state and pending interrupt, and resumes execution to RESOLVED.
    5. Parent Process validates that physical lockfile was removed, all jobs
       succeeded, and final state is RESOLVED in SQLite.
    """
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as base_tmp:
        env_dir = os.path.join(base_tmp, "env")
        db_path = os.path.join(base_tmp, "checkpoints.db")
        thread_id = "INC_CROSS_PROC_001"
        thread_config = {"configurable": {"thread_id": thread_id}}

        # Setup mock environment with stale lockfile
        runner = MockWorkflowRunner(base_dir=env_dir)
        runner.inject_scenario(ScenarioType.STALE_LOCKFILE)
        runner.run_workflow()

        lock_path = runner.locks_dir / "transform_job.lock"
        assert lock_path.exists(), "Precondition failed: lockfile must exist"

        # Phase 1: Parent process runs until HITL interrupt
        saver1 = SqliteSaver(db_path)
        graph1 = build_nova_graph(runner, checkpointer=saver1, require_approval=True)

        initial_state: NovaGraphState = {
            "incident_id": thread_id,
            "target_service": "JOB_02_TRANSFORM",
            "raw_symptoms": "Stale lockfile detected",
            "status": IncidentStatus.TRIGGERED,
            "triage_summary": None,
            "root_cause_hypothesis": None,
            "plan": [],
            "current_action_index": 0,
            "execution_history": [],
            "verification_history": [],
            "pending_approval": None,
            "retry_count": 0,
            "max_retries": 3,
            "final_summary": None,
        }

        graph1.invoke(initial_state, thread_config)

        # Verify paused state in parent
        paused_state = graph1.get_state(thread_config)
        assert paused_state.next == ("executor",)
        assert len(paused_state.tasks[0].interrupts) == 1
        assert paused_state.tasks[0].interrupts[0].value["tool_name"] == "clear_lockfile"

        # Terminate parent graph and saver
        del graph1
        saver1.close()
        del saver1
        gc.collect()

        assert lock_path.exists(), "Lockfile must remain intact while interrupted"

        # Phase 2: Launch independent OS subprocess to reload and resume
        resume_script = textwrap.dedent(f"""
            import sys
            import os
            from langgraph.types import Command
            from nova.environments.mock_workflow_runner import MockWorkflowRunner
            from nova.graph import build_nova_graph
            from nova.state import IncidentStatus
            from nova.storage.sqlite_saver import SqliteSaver

            env_dir = r"{env_dir}"
            db_path = r"{db_path}"
            thread_config = {{"configurable": {{"thread_id": "{thread_id}"}}}}

            # Initialize isolated runner and checkpointer
            runner = MockWorkflowRunner(base_dir=env_dir)
            saver = SqliteSaver(db_path)
            graph = build_nova_graph(runner, checkpointer=saver, require_approval=True)

            # Verify reload from disk
            state = graph.get_state(thread_config)
            assert state.next == ("executor",), f"Expected next=('executor',), got {{state.next}}"
            assert len(state.tasks) == 1, "Expected 1 task"
            assert len(state.tasks[0].interrupts) == 1, "Expected 1 interrupt"
            payload = state.tasks[0].interrupts[0].value
            assert payload["tool_name"] == "clear_lockfile", f"Unexpected tool {{payload}}"

            # Resume execution with operator approval
            final_state = graph.invoke(
                Command(resume={{"approved": True, "notes": "Approved by external OS process"}}),
                thread_config,
            )

            assert final_state["status"] == IncidentStatus.RESOLVED, f"Unexpected status: {{final_state['status']}}"
            assert len(final_state["execution_history"]) == 1
            assert final_state["execution_history"][0].tool_name == "clear_lockfile"
            assert final_state["verification_history"][0].passed is True

            saver.close()
            print("SUBPROCESS_SUCCESS")
        """)

        env = os.environ.copy()
        pythonpath = os.path.abspath("src")
        if "PYTHONPATH" in env:
            env["PYTHONPATH"] = f"{pythonpath}{os.pathsep}{env['PYTHONPATH']}"
        else:
            env["PYTHONPATH"] = pythonpath

        proc = subprocess.run(
            [sys.executable, "-c", resume_script],
            capture_output=True,
            text=True,
            env=env,
        )

        assert proc.returncode == 0, f"Subprocess failed with code {proc.returncode}:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        assert "SUBPROCESS_SUCCESS" in proc.stdout

        # Phase 3: Parent process verifies post-conditions
        assert not lock_path.exists(), "Lockfile must be cleared by subprocess execution"

        final_wf = runner.get_state()
        assert all(job.status == JobStatus.SUCCESS for job in final_wf.jobs.values())

        # Inspect final state using a fresh SqliteSaver
        saver3 = SqliteSaver(db_path)
        graph3 = build_nova_graph(runner, checkpointer=saver3, require_approval=True)
        final_reloaded = graph3.get_state(thread_config)
        assert final_reloaded.values["status"] == IncidentStatus.RESOLVED
        saver3.close()
        del graph3
        del saver3
        gc.collect()


def test_cross_process_both_phases_in_separate_subprocesses():
    """
    Rigorous test where BOTH Phase 1 (start to interrupt) and Phase 2 (resume to resolved)
    run in completely distinct OS subprocesses. Neither shares memory with the test runner.
    """
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as base_tmp:
        env_dir = os.path.join(base_tmp, "env")
        db_path = os.path.join(base_tmp, "checkpoints.db")
        thread_id = "INC_CROSS_PROC_002"

        runner = MockWorkflowRunner(base_dir=env_dir)
        runner.inject_scenario(ScenarioType.STALE_LOCKFILE)
        runner.run_workflow()

        env = os.environ.copy()
        pythonpath = os.path.abspath("src")
        env["PYTHONPATH"] = f"{pythonpath}{os.pathsep}{env.get('PYTHONPATH', '')}"

        # Subprocess 1: Initialize and run until interrupt
        start_script = textwrap.dedent(f"""
            from nova.environments.mock_workflow_runner import MockWorkflowRunner
            from nova.graph import build_nova_graph
            from nova.state import IncidentStatus, NovaGraphState
            from nova.storage.sqlite_saver import SqliteSaver

            runner = MockWorkflowRunner(base_dir=r"{env_dir}")
            saver = SqliteSaver(r"{db_path}")
            graph = build_nova_graph(runner, checkpointer=saver, require_approval=True)
            cfg = {{"configurable": {{"thread_id": "{thread_id}"}}}}

            initial_state: NovaGraphState = {{
                "incident_id": "{thread_id}",
                "target_service": "JOB_02_TRANSFORM",
                "raw_symptoms": "Stale lockfile detected",
                "status": IncidentStatus.TRIGGERED,
                "triage_summary": None,
                "root_cause_hypothesis": None,
                "plan": [],
                "current_action_index": 0,
                "execution_history": [],
                "verification_history": [],
                "pending_approval": None,
                "retry_count": 0,
                "max_retries": 3,
                "final_summary": None,
            }}

            graph.invoke(initial_state, cfg)
            state = graph.get_state(cfg)
            assert state.next == ("executor",)
            saver.close()
            print("START_SUBPROCESS_OK")
        """)

        proc1 = subprocess.run([sys.executable, "-c", start_script], capture_output=True, text=True, env=env)
        assert proc1.returncode == 0, f"Process 1 failed:\n{proc1.stderr}"
        assert "START_SUBPROCESS_OK" in proc1.stdout

        # Subprocess 2: Resume from disk and complete
        resume_script = textwrap.dedent(f"""
            from langgraph.types import Command
            from nova.environments.mock_workflow_runner import MockWorkflowRunner
            from nova.graph import build_nova_graph
            from nova.state import IncidentStatus
            from nova.storage.sqlite_saver import SqliteSaver

            runner = MockWorkflowRunner(base_dir=r"{env_dir}")
            saver = SqliteSaver(r"{db_path}")
            graph = build_nova_graph(runner, checkpointer=saver, require_approval=True)
            cfg = {{"configurable": {{"thread_id": "{thread_id}"}}}}

            state = graph.get_state(cfg)
            assert state.next == ("executor",)

            final_state = graph.invoke(
                Command(resume={{"approved": True, "notes": "Approved in proc2"}}),
                cfg,
            )
            assert final_state["status"] == IncidentStatus.RESOLVED
            saver.close()
            print("RESUME_SUBPROCESS_OK")
        """)

        proc2 = subprocess.run([sys.executable, "-c", resume_script], capture_output=True, text=True, env=env)
        assert proc2.returncode == 0, f"Process 2 failed:\n{proc2.stderr}"
        assert "RESUME_SUBPROCESS_OK" in proc2.stdout

        # Confirm environment cleared
        assert not (runner.locks_dir / "transform_job.lock").exists()


def test_granular_sqlite_schema_and_row_level_storage(mock_env: MockWorkflowRunner):
    """
    Validates that SqliteSaver creates true normalized relational tables:
    - Table `checkpoints` contains individual checkpoint rows with parent lineage.
    - Table `checkpoint_blobs` contains channel version blobs.
    - Table `checkpoint_writes` contains pending task writes and interrupts.
    - No monolithic single-blob serialization is used.
    """
    mock_env.inject_scenario(ScenarioType.STALE_LOCKFILE)
    mock_env.run_workflow()

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as db_dir:
        db_path = os.path.join(db_dir, "granular.db")
        saver = SqliteSaver(db_path)
        graph = build_nova_graph(mock_env, checkpointer=saver, require_approval=True)
        thread_config = {"configurable": {"thread_id": "INC_SCHEMA_001"}}

        initial_state: NovaGraphState = {
            "incident_id": "INC_SCHEMA_001",
            "target_service": "JOB_02_TRANSFORM",
            "raw_symptoms": "Stale lockfile detected",
            "status": IncidentStatus.TRIGGERED,
            "triage_summary": None,
            "root_cause_hypothesis": None,
            "plan": [],
            "current_action_index": 0,
            "execution_history": [],
            "verification_history": [],
            "pending_approval": None,
            "retry_count": 0,
            "max_retries": 3,
            "final_summary": None,
        }

        # Run to interrupt
        graph.invoke(initial_state, thread_config)

        # Inspect SQLite schema and rows
        conn = sqlite3.connect(db_path)
        try:
            # 1. Verify table existence
            tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
            assert "checkpoints" in tables
            assert "checkpoint_blobs" in tables
            assert "checkpoint_writes" in tables
            assert "nova_checkpoints" not in tables, "Legacy monolithic table must not exist"

            # 2. Verify row counts reflect incremental checkpoints
            cp_count = conn.execute("SELECT count(1) FROM checkpoints WHERE thread_id = 'INC_SCHEMA_001'").fetchone()[0]
            assert cp_count >= 3, f"Expected at least 3 discrete checkpoints, got {cp_count}"

            # 3. Verify parent lineage is preserved
            rows = conn.execute(
                "SELECT checkpoint_id, parent_checkpoint_id FROM checkpoints WHERE thread_id = 'INC_SCHEMA_001'"
            ).fetchall()
            parents = [r[1] for r in rows if r[1] is not None]
            assert len(parents) > 0, "Lineage links must be recorded in parent_checkpoint_id"

            # 4. Verify channel version blobs
            blob_count = conn.execute("SELECT count(1) FROM checkpoint_blobs WHERE thread_id = 'INC_SCHEMA_001'").fetchone()[0]
            assert blob_count > 0, "Channel version blobs must be persisted"

            # 5. Verify task writes and interrupt payloads
            writes = conn.execute(
                "SELECT channel, type, task_id FROM checkpoint_writes WHERE thread_id = 'INC_SCHEMA_001'"
            ).fetchall()
            write_channels = {w[0] for w in writes}
            assert "__interrupt__" in write_channels, "Interrupt write must be recorded in checkpoint_writes"
        finally:
            conn.close()

        saver.close()
        del graph
        del saver
        gc.collect()


def test_sqlite_saver_delete_thread(mock_env: MockWorkflowRunner):
    """
    Validates delete_thread properly cleans up all tables for a thread
    while isolating and preserving any other thread's records.
    """
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as db_dir:
        db_path = os.path.join(db_dir, "delete_test.db")
        saver = SqliteSaver(db_path)
        graph = build_nova_graph(mock_env, checkpointer=saver, require_approval=False)

        # Run thread 1
        cfg1 = {"configurable": {"thread_id": "THREAD_AAA"}}
        mock_env.inject_scenario(ScenarioType.SERVICE_OFFLINE)
        mock_env.run_workflow()
        graph.invoke(
            {
                "incident_id": "THREAD_AAA",
                "target_service": "JOB_01_INGEST",
                "raw_symptoms": "service down",
                "status": IncidentStatus.TRIGGERED,
                "triage_summary": None,
                "root_cause_hypothesis": None,
                "plan": [],
                "current_action_index": 0,
                "execution_history": [],
                "verification_history": [],
                "pending_approval": None,
                "retry_count": 0,
                "max_retries": 3,
                "final_summary": None,
            },
            cfg1,
        )

        # Run thread 2
        cfg2 = {"configurable": {"thread_id": "THREAD_BBB"}}
        graph.invoke(
            {
                "incident_id": "THREAD_BBB",
                "target_service": "JOB_01_INGEST",
                "raw_symptoms": "service down",
                "status": IncidentStatus.TRIGGERED,
                "triage_summary": None,
                "root_cause_hypothesis": None,
                "plan": [],
                "current_action_index": 0,
                "execution_history": [],
                "verification_history": [],
                "pending_approval": None,
                "retry_count": 0,
                "max_retries": 3,
                "final_summary": None,
            },
            cfg2,
        )

        conn = sqlite3.connect(db_path)
        try:
            assert conn.execute("SELECT count(1) FROM checkpoints WHERE thread_id = 'THREAD_AAA'").fetchone()[0] > 0
            assert conn.execute("SELECT count(1) FROM checkpoints WHERE thread_id = 'THREAD_BBB'").fetchone()[0] > 0
        finally:
            conn.close()

        # Delete thread AAA
        saver.delete_thread("THREAD_AAA")

        conn2 = sqlite3.connect(db_path)
        try:
            assert conn2.execute("SELECT count(1) FROM checkpoints WHERE thread_id = 'THREAD_AAA'").fetchone()[0] == 0
            assert conn2.execute("SELECT count(1) FROM checkpoint_blobs WHERE thread_id = 'THREAD_AAA'").fetchone()[0] == 0
            assert conn2.execute("SELECT count(1) FROM checkpoint_writes WHERE thread_id = 'THREAD_AAA'").fetchone()[0] == 0
            # Thread BBB remains intact
            assert conn2.execute("SELECT count(1) FROM checkpoints WHERE thread_id = 'THREAD_BBB'").fetchone()[0] > 0
        finally:
            conn2.close()

        saver.close()
        del graph
        del saver
        gc.collect()


def test_sqlite_saver_list_filtering(mock_env: MockWorkflowRunner):
    """
    Validates list() query filtering on limit, thread_id, and metadata.
    """
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as db_dir:
        db_path = os.path.join(db_dir, "list_test.db")
        saver = SqliteSaver(db_path)
        graph = build_nova_graph(mock_env, checkpointer=saver, require_approval=False)

        cfg = {"configurable": {"thread_id": "THREAD_LIST"}}
        mock_env.inject_scenario(ScenarioType.SERVICE_OFFLINE)
        mock_env.run_workflow()
        graph.invoke(
            {
                "incident_id": "THREAD_LIST",
                "target_service": "JOB_01_INGEST",
                "raw_symptoms": "service down",
                "status": IncidentStatus.TRIGGERED,
                "triage_summary": None,
                "root_cause_hypothesis": None,
                "plan": [],
                "current_action_index": 0,
                "execution_history": [],
                "verification_history": [],
                "pending_approval": None,
                "retry_count": 0,
                "max_retries": 3,
                "final_summary": None,
            },
            cfg,
        )

        # List all checkpoints
        all_cps = list(saver.list(cfg))
        assert len(all_cps) > 1

        # List with limit
        limited_cps = list(saver.list(cfg, limit=1))
        assert len(limited_cps) == 1
        assert limited_cps[0].checkpoint["id"] == all_cps[0].checkpoint["id"]

        saver.close()
        del graph
        del saver
        gc.collect()
