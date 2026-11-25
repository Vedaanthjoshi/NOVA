"""
SQLite-backed Operational Metadata & Approval Audit Store for NOVA.

Stores non-checkpoint operational records alongside LangGraph's SqliteSaver:
- `incident_metadata`: Incident catalog entries (survives backend restarts).
- `approval_audit`: Immutable audit log of HITL approval/rejection attempts
  and consumed approval_request_ids (prevents replay across restarts).
- `incident_telemetry`: Persisted per-incident counters, durations, and outcomes.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


class IncidentMetadataStore:
    """Thread-safe SQLite store for incident catalog metadata and approval audits."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self._lock = threading.Lock()
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_tables()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_tables(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS incident_metadata (
                    incident_id TEXT PRIMARY KEY,
                    target_service TEXT,
                    raw_symptoms TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS approval_audit (
                    audit_id TEXT PRIMARY KEY,
                    incident_id TEXT NOT NULL,
                    action_id TEXT NOT NULL,
                    approval_request_id TEXT,
                    action_hash TEXT,
                    tool_name TEXT,
                    approved INTEGER NOT NULL,
                    approver_id TEXT NOT NULL,
                    approver_role TEXT NOT NULL,
                    decision_timestamp TEXT NOT NULL,
                    notes TEXT,
                    outcome TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS incident_telemetry (
                    incident_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def upsert_incident(
        self,
        incident_id: str,
        target_service: Optional[str],
        raw_symptoms: str,
        created_at: str,
        updated_at: str,
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO incident_metadata
                    (incident_id, target_service, raw_symptoms, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(incident_id) DO UPDATE SET
                    updated_at = excluded.updated_at,
                    target_service = COALESCE(excluded.target_service, incident_metadata.target_service)
                """,
                (incident_id, target_service, raw_symptoms, created_at, updated_at),
            )

    def touch_incident(self, incident_id: str, updated_at: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE incident_metadata SET updated_at = ? WHERE incident_id = ?",
                (updated_at, incident_id),
            )

    def get_incident(self, incident_id: str) -> Optional[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT incident_id, target_service, raw_symptoms, created_at, updated_at
                FROM incident_metadata
                WHERE incident_id = ?
                """,
                (incident_id,),
            ).fetchone()
            if not row:
                return None
            return {
                "incident_id": row[0],
                "target_service": row[1],
                "raw_symptoms": row[2],
                "created_at": row[3],
                "updated_at": row[4],
            }

    def list_incidents(self) -> List[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT incident_id, target_service, raw_symptoms, created_at, updated_at
                FROM incident_metadata
                ORDER BY created_at DESC
                """
            ).fetchall()
            return [
                {
                    "incident_id": r[0],
                    "target_service": r[1],
                    "raw_symptoms": r[2],
                    "created_at": r[3],
                    "updated_at": r[4],
                }
                for r in rows
            ]

    def record_approval_audit(self, record: Dict[str, Any]) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO approval_audit (
                    audit_id, incident_id, action_id, approval_request_id,
                    action_hash, tool_name, approved, approver_id, approver_role,
                    decision_timestamp, notes, outcome
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["audit_id"],
                    record["incident_id"],
                    record["action_id"],
                    record.get("approval_request_id"),
                    record.get("action_hash"),
                    record.get("tool_name"),
                    1 if record.get("approved") else 0,
                    record["approver_id"],
                    record["approver_role"],
                    record["decision_timestamp"],
                    record.get("notes"),
                    record["outcome"],
                ),
            )

    def has_consumed_approval(
        self,
        incident_id: str,
        action_id: str,
        approval_request_id: Optional[str] = None,
    ) -> bool:
        """Checks whether an approval decision has already been accepted/consumed."""
        with self._lock, self._connect() as conn:
            if approval_request_id:
                row = conn.execute(
                    """
                    SELECT 1 FROM approval_audit
                    WHERE incident_id = ? AND approval_request_id = ?
                      AND outcome IN ('APPROVED_RESUMED', 'REJECTED_ESCALATED')
                    LIMIT 1
                    """,
                    (incident_id, approval_request_id),
                ).fetchone()
                if row:
                    return True
            row2 = conn.execute(
                """
                SELECT 1 FROM approval_audit
                WHERE incident_id = ? AND action_id = ?
                  AND outcome IN ('APPROVED_RESUMED', 'REJECTED_ESCALATED')
                LIMIT 1
                """,
                (incident_id, action_id),
            ).fetchone()
            return row2 is not None

    def get_approval_audits(self, incident_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            if incident_id:
                rows = conn.execute(
                    """
                    SELECT audit_id, incident_id, action_id, approval_request_id,
                           action_hash, tool_name, approved, approver_id, approver_role,
                           decision_timestamp, notes, outcome
                    FROM approval_audit
                    WHERE incident_id = ?
                    ORDER BY decision_timestamp ASC
                    """,
                    (incident_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT audit_id, incident_id, action_id, approval_request_id,
                           action_hash, tool_name, approved, approver_id, approver_role,
                           decision_timestamp, notes, outcome
                    FROM approval_audit
                    ORDER BY decision_timestamp DESC
                    """
                ).fetchall()
            return [
                {
                    "audit_id": r[0],
                    "incident_id": r[1],
                    "action_id": r[2],
                    "approval_request_id": r[3],
                    "action_hash": r[4],
                    "tool_name": r[5],
                    "approved": bool(r[6]),
                    "approver_id": r[7],
                    "approver_role": r[8],
                    "decision_timestamp": r[9],
                    "notes": r[10],
                    "outcome": r[11],
                }
                for r in rows
            ]

    def save_telemetry(self, incident_id: str, payload: Dict[str, Any], updated_at: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO incident_telemetry (incident_id, payload_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(incident_id) DO UPDATE SET
                    payload_json = excluded.payload_json,
                    updated_at = excluded.updated_at
                """,
                (incident_id, json.dumps(payload), updated_at),
            )

    def get_telemetry(self, incident_id: str) -> Optional[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM incident_telemetry WHERE incident_id = ?",
                (incident_id,),
            ).fetchone()
            if not row:
                return None
            return json.loads(row[0])
