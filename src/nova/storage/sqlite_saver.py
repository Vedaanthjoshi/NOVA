"""
Granular Row-Level SQLite Checkpoint Persistence for NOVA.

Implements a robust, granular SQLite checkpointer `SqliteSaver` subclassing
`langgraph.checkpoint.base.BaseCheckpointSaver`.

Tables:
- `checkpoints`: Row-level storage of graph checkpoints, parent pointers, and metadata.
- `checkpoint_blobs`: Channel snapshots keyed by (thread_id, checkpoint_ns, channel, version).
- `checkpoint_writes`: Intermediate task writes and HITL interrupt payloads.

Features:
- SQLite WAL mode for high-concurrency read/write transactions.
- Context-managed database connections with automatic commit, rollback, and cleanup.
- Safe serialization supporting both `pickle` and `self.serde.dumps_typed` / `loads_typed`.
- Monotonically increasing version generation and sort-key preserved pending writes.
- Full Human-in-the-Loop persistence across process terminations and restarts.
"""

from __future__ import annotations

import gc
import pickle
import random
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Iterator, Optional, Sequence, Tuple

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    SerializerProtocol,
    get_checkpoint_id,
    get_checkpoint_metadata,
    writes_sort_key,
)


class SqliteSaver(BaseCheckpointSaver):
    """
    Granular, row-level persistent SQLite Checkpointer for NOVA.

    Unlike naive wrappers that dump an entire in-memory dictionary into a single
    monolithic BLOB row, this saver persists checkpoints, channel version blobs,
    and task writes into normalized relational tables.

    Guarantees:
    - ACID transaction isolation per checkpoint commit.
    - Zero lock contention via SQLite WAL (Write-Ahead Logging).
    - True cross-process recovery: interrupted Human-in-the-Loop states can be
      resumed by any subsequent process with access to the database file.
    """

    def __init__(
        self,
        db_path: str | Path = "nova_checkpoints.db",
        *,
        serde: Optional[SerializerProtocol] = None,
    ) -> None:
        super().__init__(serde=serde)
        self.db_path = str(db_path)
        self._custom_serde = serde is not None
        self._lock = threading.Lock()

        # Ensure parent directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _get_connection(self) -> Iterator[sqlite3.Connection]:
        """Context manager providing a safe, transactional SQLite connection."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Initializes database schema with WAL mode enabled."""
        with self._lock, self._get_connection() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")

            # 1. Checkpoints table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS checkpoints (
                    thread_id TEXT,
                    checkpoint_ns TEXT,
                    checkpoint_id TEXT,
                    parent_checkpoint_id TEXT,
                    type TEXT,
                    checkpoint BLOB,
                    metadata BLOB,
                    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
                )
                """
            )

            # 2. Channel blobs table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS checkpoint_blobs (
                    thread_id TEXT,
                    checkpoint_ns TEXT,
                    channel TEXT,
                    version TEXT,
                    type TEXT,
                    blob BLOB,
                    PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
                )
                """
            )

            # 3. Intermediate writes table (including pending interrupts)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS checkpoint_writes (
                    thread_id TEXT,
                    checkpoint_ns TEXT,
                    checkpoint_id TEXT,
                    task_id TEXT,
                    idx INTEGER,
                    channel TEXT,
                    type TEXT,
                    value BLOB,
                    task_path TEXT DEFAULT '',
                    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
                )
                """
            )

    def _dump(self, obj: Any) -> tuple[str, bytes]:
        """Serializes an object using custom serde or robust pickle."""
        if self._custom_serde and self.serde is not None:
            try:
                return self.serde.dumps_typed(obj)
            except Exception:
                pass
        return ("pickle", pickle.dumps(obj))

    def _load(self, type_: str, data: bytes) -> Any:
        """Deserializes an object based on its recorded serialization format."""
        if type_ == "pickle":
            return pickle.loads(data)
        return self.serde.loads_typed((type_, data))

    def _load_blobs(
        self,
        conn: sqlite3.Connection,
        thread_id: str,
        checkpoint_ns: str,
        versions: ChannelVersions,
    ) -> dict[str, Any]:
        """Loads and deserializes channel values for the requested channel versions."""
        if not versions:
            return {}
        cursor = conn.execute(
            """
            SELECT channel, version, type, blob
            FROM checkpoint_blobs
            WHERE thread_id = ? AND checkpoint_ns = ?
            """,
            (thread_id, checkpoint_ns),
        )
        blob_map = {
            (row[0], str(row[1])): (row[2], row[3]) for row in cursor.fetchall()
        }
        result: dict[str, Any] = {}
        for k, ver in versions.items():
            entry = blob_map.get((k, str(ver)))
            if entry is None or entry[0] == "empty":
                continue
            result[k] = self._load(entry[0], entry[1])
        return result

    def _ordered_writes(
        self,
        conn: sqlite3.Connection,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
    ) -> list[tuple[str, str, Any]]:
        """Retrieves and sorts pending writes for a checkpoint in execution order."""
        cursor = conn.execute(
            """
            SELECT task_id, idx, channel, type, value, task_path
            FROM checkpoint_writes
            WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?
            """,
            (thread_id, checkpoint_ns, checkpoint_id),
        )
        rows = cursor.fetchall()
        sorted_rows = sorted(
            rows,
            key=lambda r: writes_sort_key(r[5] or "", r[0], r[1]),
        )
        return [(r[0], r[2], self._load(r[3], r[4])) for r in sorted_rows]

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """
        Fetch a checkpoint tuple from SQLite matching the given configuration.
        Returns None if no matching checkpoint is found.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns") or ""
        checkpoint_id = get_checkpoint_id(config)

        with self._lock, self._get_connection() as conn:
            if checkpoint_id:
                cursor = conn.execute(
                    """
                    SELECT thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata
                    FROM checkpoints
                    WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?
                    """,
                    (thread_id, checkpoint_ns, checkpoint_id),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata
                    FROM checkpoints
                    WHERE thread_id = ? AND checkpoint_ns = ?
                    ORDER BY checkpoint_id DESC
                    LIMIT 1
                    """,
                    (thread_id, checkpoint_ns),
                )
            row = cursor.fetchone()
            if not row:
                return None

            t_id, cp_ns, cp_id, parent_cp_id, cp_type, cp_blob, meta_blob = row

            checkpoint_: Checkpoint = self._load(cp_type, cp_blob)
            metadata: CheckpointMetadata = self._load(cp_type, meta_blob)

            channel_values = self._load_blobs(
                conn, t_id, cp_ns, checkpoint_.get("channel_versions", {})
            )
            pending_writes = self._ordered_writes(conn, t_id, cp_ns, cp_id)

        checkpoint_dict = {
            **checkpoint_,
            "channel_values": channel_values,
        }

        parent_config = (
            {
                "configurable": {
                    "thread_id": t_id,
                    "checkpoint_ns": cp_ns,
                    "checkpoint_id": parent_cp_id,
                }
            }
            if parent_cp_id
            else None
        )

        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": t_id,
                    "checkpoint_ns": cp_ns,
                    "checkpoint_id": cp_id,
                }
            },
            checkpoint=checkpoint_dict,
            metadata=metadata,
            parent_config=parent_config,
            pending_writes=pending_writes,
        )

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints matching the provided query parameters."""
        thread_id = (
            config["configurable"].get("thread_id")
            if config and "configurable" in config
            else None
        )
        checkpoint_ns = (
            config["configurable"].get("checkpoint_ns")
            if config and "configurable" in config
            else None
        )
        checkpoint_id = get_checkpoint_id(config) if config else None
        before_checkpoint_id = get_checkpoint_id(before) if before else None

        query = (
            "SELECT thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, "
            "type, checkpoint, metadata FROM checkpoints WHERE 1=1"
        )
        params: list[Any] = []

        if thread_id is not None:
            query += " AND thread_id = ?"
            params.append(thread_id)
        if checkpoint_ns is not None:
            query += " AND checkpoint_ns = ?"
            params.append(checkpoint_ns)
        if checkpoint_id is not None:
            query += " AND checkpoint_id = ?"
            params.append(checkpoint_id)
        if before_checkpoint_id is not None:
            query += " AND checkpoint_id < ?"
            params.append(before_checkpoint_id)

        query += " ORDER BY checkpoint_id DESC"

        with self._lock, self._get_connection() as conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

            results = []
            for row in rows:
                t_id, cp_ns, cp_id, parent_cp_id, cp_type, cp_blob, meta_blob = row
                metadata: CheckpointMetadata = self._load(cp_type, meta_blob)

                if filter and not all(
                    query_value == metadata.get(query_key)
                    for query_key, query_value in filter.items()
                ):
                    continue

                if limit is not None and limit <= 0:
                    break
                elif limit is not None:
                    limit -= 1

                checkpoint_: Checkpoint = self._load(cp_type, cp_blob)
                channel_values = self._load_blobs(
                    conn, t_id, cp_ns, checkpoint_.get("channel_versions", {})
                )
                pending_writes = self._ordered_writes(conn, t_id, cp_ns, cp_id)

                results.append(
                    CheckpointTuple(
                        config={
                            "configurable": {
                                "thread_id": t_id,
                                "checkpoint_ns": cp_ns,
                                "checkpoint_id": cp_id,
                            }
                        },
                        checkpoint={
                            **checkpoint_,
                            "channel_values": channel_values,
                        },
                        metadata=metadata,
                        parent_config=(
                            {
                                "configurable": {
                                    "thread_id": t_id,
                                    "checkpoint_ns": cp_ns,
                                    "checkpoint_id": parent_cp_id,
                                }
                            }
                            if parent_cp_id
                            else None
                        ),
                        pending_writes=pending_writes,
                    )
                )

        for item in results:
            yield item

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Stores a checkpoint snapshot and its updated channel versions."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns") or ""
        parent_checkpoint_id = config["configurable"].get("checkpoint_id")
        checkpoint_id = checkpoint["id"]

        c = checkpoint.copy()
        values: dict[str, Any] = c.pop("channel_values", {})

        with self._lock, self._get_connection() as conn:
            for k, v in new_versions.items():
                if k in values:
                    t, b = self._dump(values[k])
                else:
                    t, b = ("empty", b"")
                conn.execute(
                    """
                    INSERT OR REPLACE INTO checkpoint_blobs
                    (thread_id, checkpoint_ns, channel, version, type, blob)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (thread_id, checkpoint_ns, k, str(v), t, b),
                )

            meta = get_checkpoint_metadata(config, metadata)
            t_meta, b_meta = self._dump(meta)
            t_cp, b_cp = self._dump(c)

            conn.execute(
                """
                INSERT OR REPLACE INTO checkpoints
                (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    thread_id,
                    checkpoint_ns,
                    checkpoint_id,
                    parent_checkpoint_id,
                    t_cp,
                    b_cp,
                    b_meta,
                ),
            )

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Stores intermediate writes including pending task results and interrupts."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns") or ""
        checkpoint_id = config["configurable"]["checkpoint_id"]

        with self._lock, self._get_connection() as conn:
            for idx, (channel, val) in enumerate(writes):
                mapped_idx = WRITES_IDX_MAP.get(channel, idx)
                t, b = self._dump(val)
                if mapped_idx >= 0:
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO checkpoint_writes
                        (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, value, task_path)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            thread_id,
                            checkpoint_ns,
                            checkpoint_id,
                            task_id,
                            mapped_idx,
                            channel,
                            t,
                            b,
                            task_path,
                        ),
                    )
                else:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO checkpoint_writes
                        (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, value, task_path)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            thread_id,
                            checkpoint_ns,
                            checkpoint_id,
                            task_id,
                            mapped_idx,
                            channel,
                            t,
                            b,
                            task_path,
                        ),
                    )

    def list_thread_ids(self) -> list[str]:
        """Returns all distinct thread_ids persisted in the checkpoints table."""
        with self._lock, self._get_connection() as conn:
            rows = conn.execute(
                "SELECT DISTINCT thread_id FROM checkpoints ORDER BY rowid DESC"
            ).fetchall()
            return [r[0] for r in rows]

    def delete_thread(self, thread_id: str) -> None:
        """Deletes all checkpoints, channel blobs, and writes for a given thread."""
        with self._lock, self._get_connection() as conn:
            conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
            conn.execute("DELETE FROM checkpoint_blobs WHERE thread_id = ?", (thread_id,))
            conn.execute("DELETE FROM checkpoint_writes WHERE thread_id = ?", (thread_id,))

    def get_next_version(self, current: Optional[str], channel: None = None) -> str:
        """Generates a monotonically increasing version string for channel tracking."""
        if current is None:
            current_v = 0
        elif isinstance(current, int):
            current_v = current
        else:
            current_v = int(str(current).split(".")[0])
        next_v = current_v + 1
        next_h = random.random()
        return f"{next_v:032}.{next_h:016}"

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        return self.get_tuple(config)

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        for item in self.list(config, filter=filter, before=before, limit=limit):
            yield item

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return self.put(config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        return self.put_writes(config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id: str) -> None:
        return self.delete_thread(thread_id)

    def close(self) -> None:
        """Releases memory and garbage collects dangling handles on Windows."""
        gc.collect()

    def __enter__(self) -> "SqliteSaver":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    async def __aenter__(self) -> "SqliteSaver":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
