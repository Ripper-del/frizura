"""SQLite-based event store for time-travel debugging.

Provides persistent storage for all execution events and state snapshots,
enabling full replay, forking, and inspection of pipeline runs. Events are
stored as JSON blobs in SQLite via aiosqlite for async I/O.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Self
from uuid import uuid4

import asyncio
import aiosqlite

from frizura.core.events import Event, EventType, StateSnapshot

logger = logging.getLogger(__name__)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS events (
    id          TEXT PRIMARY KEY,
    pipeline_id TEXT NOT NULL,
    step_id     TEXT,
    step_name   TEXT,
    event_type  TEXT NOT NULL,
    sequence_number INTEGER NOT NULL,
    timestamp   TEXT NOT NULL,
    data        TEXT NOT NULL DEFAULT '{}',
    parent_event_id TEXT,
    UNIQUE(pipeline_id, sequence_number)
);

CREATE INDEX IF NOT EXISTS idx_events_pipeline
    ON events(pipeline_id, sequence_number);
CREATE INDEX IF NOT EXISTS idx_events_type
    ON events(pipeline_id, event_type);
CREATE INDEX IF NOT EXISTS idx_events_step
    ON events(pipeline_id, step_id);

CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id TEXT PRIMARY KEY,
    pipeline_id TEXT NOT NULL,
    step_id     TEXT NOT NULL,
    step_index  INTEGER NOT NULL,
    timestamp   TEXT NOT NULL,
    state       TEXT NOT NULL DEFAULT '{}',
    memory      TEXT NOT NULL DEFAULT '[]',
    budget_state TEXT NOT NULL DEFAULT '{}',
    metadata    TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_snapshots_pipeline
    ON snapshots(pipeline_id, step_index);
CREATE INDEX IF NOT EXISTS idx_snapshots_step
    ON snapshots(pipeline_id, step_id);

CREATE TABLE IF NOT EXISTS pipeline_forks (
    fork_id         TEXT PRIMARY KEY,
    source_pipeline TEXT NOT NULL,
    forked_pipeline TEXT NOT NULL,
    at_step_id      TEXT NOT NULL,
    at_sequence     INTEGER NOT NULL,
    created_at      TEXT NOT NULL
);
"""


def _serialize_datetime(obj: Any) -> Any:
    """JSON serializer for datetime objects."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class EventStore:
    """Async SQLite-backed event store for pipeline execution events.

    Supports appending events, saving/loading snapshots, querying by
    pipeline/step/type, forking pipelines, and generating summaries.

    Usage::

        async with EventStore(Path("events.db")) as store:
            await store.append(event)
            events = await store.get_events("pipeline-123")
    """

    def __init__(self, db_path: Path | str = ".frizura/events.db") -> None:
        self._db_path = Path(db_path)
        self._db: aiosqlite.Connection | None = None
        self._sequence_cache: dict[str, int] = {}
        self._lock = asyncio.Lock()

    # --- Lifecycle -----------------------------------------------------------

    async def init(self) -> None:
        """Open the database and create tables if they don't exist."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self._db_path))
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(_SCHEMA_SQL)
        await self._db.commit()
        logger.info("EventStore initialized at %s", self._db_path)

    async def close(self) -> None:
        """Close the database connection."""
        if self._db is not None:
            await self._db.close()
            self._db = None
            logger.debug("EventStore closed")

    async def __aenter__(self) -> Self:
        await self.init()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    @property
    def _conn(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("EventStore is not initialized — call init() or use as async context manager")
        return self._db

    # --- Sequence numbers ----------------------------------------------------

    async def _next_sequence(self, pipeline_id: str) -> int:
        """Get the next sequence number for a pipeline.

        Uses an in-memory cache, falling back to a DB query on cache miss.
        """
        if pipeline_id not in self._sequence_cache:
            cursor = await self._conn.execute(
                "SELECT MAX(sequence_number) FROM events WHERE pipeline_id = ?",
                (pipeline_id,),
            )
            row = await cursor.fetchone()
            self._sequence_cache[pipeline_id] = (row[0] or -1) + 1
        else:
            self._sequence_cache[pipeline_id] += 1
        return self._sequence_cache[pipeline_id]

    # --- Event storage -------------------------------------------------------

    async def append(self, event: Event) -> Event:
        """Append a single event, assigning an auto-incrementing sequence number.

        Returns a new Event instance with the assigned sequence_number.
        """
        async with self._lock:
            seq = await self._next_sequence(event.pipeline_id)
            # Events are frozen, so we create a new one with the sequence number
            stored_event = event.model_copy(update={"sequence_number": seq})

            await self._conn.execute(
                """INSERT INTO events
                   (id, pipeline_id, step_id, step_name, event_type,
                    sequence_number, timestamp, data, parent_event_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    stored_event.id,
                    stored_event.pipeline_id,
                    stored_event.step_id,
                    stored_event.step_name,
                    stored_event.event_type.value,
                    seq,
                    stored_event.timestamp.isoformat(),
                    json.dumps(stored_event.data, default=_serialize_datetime),
                    stored_event.parent_event_id,
                ),
            )
            await self._conn.commit()
            logger.debug(
                "Appended event %s [%s] seq=%d for pipeline %s",
                stored_event.id,
                stored_event.event_type,
                seq,
                stored_event.pipeline_id,
            )
            return stored_event

    async def append_batch(self, events: list[Event]) -> list[Event]:
        """Append multiple events in a single transaction.

        Returns a list of new Event instances with assigned sequence numbers.
        """
        if not events:
            return []

        async with self._lock:
            stored: list[Event] = []
            rows: list[tuple[str, str, str | None, str | None, str, int, str, str, str | None]] = []

            for event in events:
                seq = await self._next_sequence(event.pipeline_id)
                ev = event.model_copy(update={"sequence_number": seq})
                stored.append(ev)
                rows.append((
                    ev.id,
                    ev.pipeline_id,
                    ev.step_id,
                    ev.step_name,
                    ev.event_type.value,
                    seq,
                    ev.timestamp.isoformat(),
                    json.dumps(ev.data, default=_serialize_datetime),
                    ev.parent_event_id,
                ))

            await self._conn.executemany(
                """INSERT INTO events
                   (id, pipeline_id, step_id, step_name, event_type,
                    sequence_number, timestamp, data, parent_event_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
            await self._conn.commit()
            logger.debug("Batch appended %d events", len(stored))
            return stored

    # --- Event queries -------------------------------------------------------

    def _row_to_event(self, row: aiosqlite.Row) -> Event:
        """Convert a database row to an Event instance."""
        return Event(
            id=row["id"],
            pipeline_id=row["pipeline_id"],
            step_id=row["step_id"],
            step_name=row["step_name"],
            event_type=EventType(row["event_type"]),
            sequence_number=row["sequence_number"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            data=json.loads(row["data"]),
            parent_event_id=row["parent_event_id"],
        )

    async def get_events(
        self,
        pipeline_id: str,
        *,
        after_sequence: int | None = None,
        event_types: list[EventType] | None = None,
    ) -> list[Event]:
        """Retrieve events for a pipeline, optionally filtered.

        Args:
            pipeline_id: The pipeline to query.
            after_sequence: Only return events with sequence_number > this value.
            event_types: Only return events of these types.

        Returns:
            Events ordered by sequence_number ascending.
        """
        query = "SELECT * FROM events WHERE pipeline_id = ?"
        params: list[Any] = [pipeline_id]

        if after_sequence is not None:
            query += " AND sequence_number > ?"
            params.append(after_sequence)

        if event_types:
            placeholders = ",".join("?" for _ in event_types)
            query += f" AND event_type IN ({placeholders})"
            params.extend(et.value for et in event_types)

        query += " ORDER BY sequence_number ASC"

        cursor = await self._conn.execute(query, params)
        rows = await cursor.fetchall()
        return [self._row_to_event(row) for row in rows]

    async def get_event(self, event_id: str) -> Event | None:
        """Retrieve a single event by its ID."""
        cursor = await self._conn.execute(
            "SELECT * FROM events WHERE id = ?", (event_id,)
        )
        row = await cursor.fetchone()
        return self._row_to_event(row) if row else None

    # --- Snapshot storage ----------------------------------------------------

    async def save_snapshot(self, snapshot: StateSnapshot) -> None:
        """Persist a state snapshot, replacing any existing one for the same step."""
        await self._conn.execute(
            "DELETE FROM snapshots WHERE pipeline_id = ? AND step_id = ?",
            (snapshot.pipeline_id, snapshot.step_id),
        )
        await self._conn.execute(
            """INSERT OR REPLACE INTO snapshots
               (snapshot_id, pipeline_id, step_id, step_index,
                timestamp, state, memory, budget_state, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                snapshot.snapshot_id,
                snapshot.pipeline_id,
                snapshot.step_id,
                snapshot.step_index,
                snapshot.timestamp.isoformat(),
                json.dumps(snapshot.state, default=_serialize_datetime),
                json.dumps(snapshot.memory, default=_serialize_datetime),
                json.dumps(snapshot.budget_state, default=_serialize_datetime),
                json.dumps(snapshot.metadata, default=_serialize_datetime),
            ),
        )
        await self._conn.commit()
        logger.debug(
            "Saved snapshot %s for pipeline %s step %s",
            snapshot.snapshot_id,
            snapshot.pipeline_id,
            snapshot.step_id,
        )

    def _row_to_snapshot(self, row: aiosqlite.Row) -> StateSnapshot:
        """Convert a database row to a StateSnapshot instance."""
        return StateSnapshot(
            snapshot_id=row["snapshot_id"],
            pipeline_id=row["pipeline_id"],
            step_id=row["step_id"],
            step_index=row["step_index"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            state=json.loads(row["state"]),
            memory=json.loads(row["memory"]),
            budget_state=json.loads(row["budget_state"]),
            metadata=json.loads(row["metadata"]),
        )

    async def get_snapshot(
        self, pipeline_id: str, step_id: str
    ) -> StateSnapshot | None:
        """Retrieve a specific snapshot by pipeline and step IDs."""
        cursor = await self._conn.execute(
            "SELECT * FROM snapshots WHERE pipeline_id = ? AND step_id = ?",
            (pipeline_id, step_id),
        )
        row = await cursor.fetchone()
        return self._row_to_snapshot(row) if row else None

    async def get_latest_snapshot(
        self, pipeline_id: str
    ) -> StateSnapshot | None:
        """Retrieve the most recent snapshot for a pipeline."""
        cursor = await self._conn.execute(
            """SELECT * FROM snapshots
               WHERE pipeline_id = ?
               ORDER BY step_index DESC
               LIMIT 1""",
            (pipeline_id,),
        )
        row = await cursor.fetchone()
        return self._row_to_snapshot(row) if row else None

    # --- Pipeline queries ----------------------------------------------------

    async def list_pipelines(self, limit: int = 50) -> list[dict[str, Any]]:
        """List pipeline summaries, most recent first.

        Returns a list of dicts with: pipeline_id, event_count,
        first_event_at, last_event_at, status.
        """
        cursor = await self._conn.execute(
            """SELECT
                 pipeline_id,
                 COUNT(*)              AS event_count,
                 MIN(timestamp)        AS first_event_at,
                 MAX(timestamp)        AS last_event_at,
                 MAX(CASE WHEN event_type = 'pipeline.completed' THEN 1
                          WHEN event_type = 'pipeline.failed'    THEN 2
                          ELSE 0 END) AS status_code
               FROM events
               GROUP BY pipeline_id
               ORDER BY MAX(timestamp) DESC
               LIMIT ?""",
            (limit,),
        )
        rows = await cursor.fetchall()
        status_map = {0: "running", 1: "completed", 2: "failed"}
        return [
            {
                "pipeline_id": row["pipeline_id"],
                "event_count": row["event_count"],
                "first_event_at": row["first_event_at"],
                "last_event_at": row["last_event_at"],
                "status": status_map.get(row["status_code"], "unknown"),
            }
            for row in rows
        ]

    async def get_pipeline_summary(self, pipeline_id: str) -> dict[str, Any]:
        """Generate a detailed summary for a single pipeline.

        Returns: pipeline_id, step_count, status, duration_ms, total_cost_usd,
        event_count, models_used, first_event_at, last_event_at.
        """
        events = await self.get_events(pipeline_id)
        if not events:
            return {"pipeline_id": pipeline_id, "status": "not_found", "event_count": 0}

        step_ids: set[str] = set()
        models_used: set[str] = set()
        total_cost = 0.0
        status = "running"

        for ev in events:
            if ev.step_id:
                step_ids.add(ev.step_id)
            if ev.event_type == EventType.LLM_RESPONSE:
                model = ev.data.get("model", "")
                if model:
                    models_used.add(model)
                total_cost += ev.data.get("cost_usd", 0.0)
            if ev.event_type == EventType.PIPELINE_COMPLETED:
                status = "completed"
            elif ev.event_type == EventType.PIPELINE_FAILED:
                status = "failed"

        first_ts = events[0].timestamp
        last_ts = events[-1].timestamp
        duration_ms = (last_ts - first_ts).total_seconds() * 1000

        return {
            "pipeline_id": pipeline_id,
            "step_count": len(step_ids),
            "event_count": len(events),
            "status": status,
            "duration_ms": round(duration_ms, 2),
            "total_cost_usd": round(total_cost, 6),
            "models_used": sorted(models_used),
            "first_event_at": first_ts.isoformat(),
            "last_event_at": last_ts.isoformat(),
        }

    # --- Forking -------------------------------------------------------------

    async def fork(self, pipeline_id: str, at_step_id: str) -> str:
        """Create a fork of a pipeline at the given step.

        Copies all events up to and including the specified step into a new
        pipeline. Returns the new pipeline_id.

        Args:
            pipeline_id: Source pipeline to fork from.
            at_step_id: Step at which to fork — events up to the completion
                of this step are copied.

        Returns:
            The new forked pipeline_id.

        Raises:
            ValueError: If no events exist for the source pipeline or step.
        """
        all_events = await self.get_events(pipeline_id)
        if not all_events:
            raise ValueError(f"No events found for pipeline '{pipeline_id}'")

        # Find the last event for the target step
        cutoff_seq: int | None = None
        for ev in reversed(all_events):
            if ev.step_id == at_step_id:
                cutoff_seq = ev.sequence_number
                break

        if cutoff_seq is None:
            raise ValueError(
                f"Step '{at_step_id}' not found in pipeline '{pipeline_id}'"
            )

        # Create forked pipeline
        fork_id = uuid4().hex[:12]
        forked_pipeline_id = f"{pipeline_id}__fork_{fork_id}"

        # Copy events up to cutoff
        events_to_copy = [
            ev for ev in all_events if ev.sequence_number <= cutoff_seq
        ]
        forked_events: list[Event] = []
        for ev in events_to_copy:
            forked_events.append(
                Event(
                    id=uuid4().hex[:16],
                    pipeline_id=forked_pipeline_id,
                    step_id=ev.step_id,
                    step_name=ev.step_name,
                    event_type=ev.event_type,
                    timestamp=ev.timestamp,
                    data=ev.data,
                    parent_event_id=ev.parent_event_id,
                )
            )

        await self.append_batch(forked_events)

        # Copy relevant snapshots
        for ev_step_id in {e.step_id for e in events_to_copy if e.step_id}:
            snap = await self.get_snapshot(pipeline_id, ev_step_id)
            if snap:
                forked_snap = StateSnapshot(
                    pipeline_id=forked_pipeline_id,
                    step_id=snap.step_id,
                    step_index=snap.step_index,
                    timestamp=snap.timestamp,
                    state=snap.state,
                    memory=snap.memory,
                    budget_state=snap.budget_state,
                    metadata={**snap.metadata, "forked_from": pipeline_id},
                )
                await self.save_snapshot(forked_snap)

        # Record the fork relationship
        await self._conn.execute(
            """INSERT INTO pipeline_forks
               (fork_id, source_pipeline, forked_pipeline, at_step_id, at_sequence, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                fork_id,
                pipeline_id,
                forked_pipeline_id,
                at_step_id,
                cutoff_seq,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        await self._conn.commit()

        logger.info(
            "Forked pipeline %s at step %s → %s (%d events copied)",
            pipeline_id,
            at_step_id,
            forked_pipeline_id,
            len(forked_events),
        )
        return forked_pipeline_id
