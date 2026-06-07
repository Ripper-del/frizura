"""Replay engine for re-constructing pipeline state from stored events.

Enables time-travel debugging by replaying the event log forward from a
snapshot (or from scratch) to any point in a pipeline's history. Also
supports forking with modified state for "what-if" exploration.
"""

from __future__ import annotations

import copy
import logging
from datetime import datetime, timezone
from typing import Any

from frizura.core.events import Event, EventType, StateSnapshot
from frizura.core.exceptions import ReplayError, SnapshotNotFoundError
from frizura.models.budget import BudgetConstraint
from frizura.models.execution import (
    LLMResponse,
    Message,
    StepResult,
    TokenUsage,
)
from frizura.timetravel.snapshot import SnapshotManager
from frizura.timetravel.store import EventStore

logger = logging.getLogger(__name__)


class ReplayContext:
    """Mutable context built up during event replay.

    Accumulates state, memory, step results, and budget info as events
    are processed sequentially.
    """

    def __init__(self) -> None:
        self.state: dict[str, Any] = {}
        self.memory: list[Message] = []
        self.step_results: list[StepResult] = []
        self.budget_constraint: BudgetConstraint | None = None
        self.pipeline_id: str = ""
        self.current_step_id: str | None = None
        self.current_step_name: str | None = None
        self.events_processed: int = 0

        # Tracking for in-progress steps
        self._step_start_times: dict[str, datetime] = {}
        self._step_models: dict[str, str] = {}
        self._step_costs: dict[str, float] = {}
        self._step_outputs: dict[str, Any] = {}
        self._step_healing: dict[str, int] = {}

    def to_dict(self) -> dict[str, Any]:
        """Export the reconstructed context as a plain dict."""
        return {
            "pipeline_id": self.pipeline_id,
            "state": copy.deepcopy(self.state),
            "memory": [m.model_dump(mode="json") for m in self.memory],
            "step_results": [sr.model_dump(mode="json") for sr in self.step_results],
            "current_step_id": self.current_step_id,
            "events_processed": self.events_processed,
        }


def _apply_event(ctx: ReplayContext, event: Event) -> None:
    """Apply a single event to the replay context, mutating it in place.

    This is the core replay logic: each event type updates the context
    state in the same way the real execution engine would.
    """
    ctx.events_processed += 1
    ctx.pipeline_id = event.pipeline_id

    match event.event_type:
        case EventType.PIPELINE_STARTED:
            ctx.state = event.data.get("initial_state", {})
            ctx.pipeline_id = event.pipeline_id

        case EventType.STEP_STARTED:
            ctx.current_step_id = event.step_id
            ctx.current_step_name = event.step_name
            if event.step_id:
                ctx._step_start_times[event.step_id] = event.timestamp
                ctx._step_costs[event.step_id] = 0.0
                ctx._step_healing[event.step_id] = 0

        case EventType.LLM_REQUEST:
            # Record the messages sent to the LLM
            messages = event.data.get("messages", [])
            for msg_data in messages:
                try:
                    ctx.memory.append(Message.model_validate(msg_data))
                except Exception:
                    pass

        case EventType.LLM_RESPONSE:
            model = event.data.get("model", "")
            cost = event.data.get("cost_usd", 0.0)
            content = event.data.get("content", "")
            if event.step_id:
                ctx._step_models[event.step_id] = model
                ctx._step_costs[event.step_id] = (
                    ctx._step_costs.get(event.step_id, 0.0) + cost
                )
            # Add assistant response to memory
            if content:
                ctx.memory.append(
                    Message(role="assistant", content=content)
                )

        case EventType.STEP_COMPLETED:
            step_id = event.step_id or ""
            step_name = event.step_name or ""
            start = ctx._step_start_times.get(step_id, event.timestamp)
            duration_ms = (event.timestamp - start).total_seconds() * 1000

            result = StepResult(
                step_id=step_id,
                step_name=step_name,
                status="completed",
                output=event.data.get("output"),
                duration_ms=duration_ms,
                cost_usd=ctx._step_costs.get(step_id, 0.0),
                model_used=ctx._step_models.get(step_id, ""),
                healing_attempts=ctx._step_healing.get(step_id, 0),
                started_at=start,
                completed_at=event.timestamp,
            )
            ctx.step_results.append(result)

            # Merge step output into state
            output = event.data.get("output")
            if isinstance(output, dict):
                ctx.state.update(output)
            elif output is not None:
                ctx.state[f"step_{step_name}_output"] = output

        case EventType.STEP_FAILED:
            step_id = event.step_id or ""
            step_name = event.step_name or ""
            start = ctx._step_start_times.get(step_id, event.timestamp)
            duration_ms = (event.timestamp - start).total_seconds() * 1000

            result = StepResult(
                step_id=step_id,
                step_name=step_name,
                status="failed",
                error=event.data.get("error", "Unknown error"),
                duration_ms=duration_ms,
                cost_usd=ctx._step_costs.get(step_id, 0.0),
                model_used=ctx._step_models.get(step_id, ""),
                started_at=start,
                completed_at=event.timestamp,
            )
            ctx.step_results.append(result)

        case EventType.STEP_SKIPPED:
            result = StepResult(
                step_id=event.step_id or "",
                step_name=event.step_name or "",
                status="skipped",
                output=event.data.get("reason", "Condition not met"),
            )
            ctx.step_results.append(result)

        case EventType.STATE_MODIFIED:
            patches = event.data.get("patches", {})
            ctx.state.update(patches)

        case EventType.SCHEMA_HEAL_ATTEMPT:
            if event.step_id:
                ctx._step_healing[event.step_id] = (
                    ctx._step_healing.get(event.step_id, 0) + 1
                )

        case EventType.STATE_SNAPSHOT:
            # Snapshot events are informational during replay
            pass

        case _:
            # Other event types don't affect replay state
            pass


class ReplayEngine:
    """Replays recorded events to reconstruct pipeline state.

    Uses the ``EventStore`` for event retrieval and ``SnapshotManager``
    for accelerated restore from checkpoints.

    Example::

        engine = ReplayEngine(db_path=".frizura/events.db")
        ctx = await engine.replay_to("pipeline-123", "step-2")
        print(ctx.state)
    """

    def __init__(
        self,
        store: EventStore | None = None,
        snapshot_manager: SnapshotManager | None = None,
        *,
        db_path: str | None = None,
    ) -> None:
        self._owns_store = store is None
        if store is None:
            from frizura.models.config import FrizuraConfig
            path = db_path or str(FrizuraConfig().timetravel.db_path)
            self._store = EventStore(db_path=path)
        else:
            self._store = store
        self._snapshots = snapshot_manager or SnapshotManager()

    async def replay_to(
        self,
        pipeline_id: str,
        step_id: str,
    ) -> ReplayContext:
        """Replay events up to the completion of the given step.

        Attempts to start from the nearest snapshot before the target step
        to minimize the number of events that need to be replayed.

        Args:
            pipeline_id: Pipeline to replay.
            step_id: Step to replay up to (inclusive).

        Returns:
            A ``ReplayContext`` with the reconstructed state at that step.

        Raises:
            ReplayError: If the pipeline or step cannot be found.
        """
        logger.info(
            "Replaying pipeline %s to step %s", pipeline_id, step_id
        )

        await self._store.init()
        try:
            all_events = await self._store.get_events(pipeline_id)
            if not all_events:
                raise ReplayError(
                    f"No events found for pipeline '{pipeline_id}'"
                )

            # Find the last event for the target step
            target_seq: int | None = None
            for ev in reversed(all_events):
                if ev.step_id == step_id:
                    target_seq = ev.sequence_number
                    break

            if target_seq is None:
                raise ReplayError(
                    f"Step '{step_id}' not found in pipeline '{pipeline_id}'"
                )

            # Try to restore from a snapshot before the target
            ctx = ReplayContext()
            replay_after_seq: int = -1

            snapshot = await self._store.get_snapshot(pipeline_id, step_id)
            if snapshot and snapshot.step_index >= 0:
                state, memory, budget = self._snapshots.restore(snapshot)
                ctx = ReplayContext()
                ctx.state = state
                ctx.memory = memory
                ctx.budget_constraint = budget
                ctx.pipeline_id = pipeline_id
                ctx.current_step_id = step_id
                for ev in reversed(all_events):
                    if ev.step_id == step_id:
                        ctx.events_processed = sum(1 for e in all_events if e.sequence_number <= ev.sequence_number)
                        break
                logger.info("Restored exact snapshot at step %s directly", step_id)
                return ctx

            # Find the closest snapshot BEFORE the target step
            # We iterate events to find step order
            step_order: list[str] = []
            for ev in all_events:
                if (
                    ev.step_id
                    and ev.event_type == EventType.STEP_STARTED
                    and ev.step_id not in step_order
                ):
                    step_order.append(ev.step_id)

            target_step_index = (
                step_order.index(step_id) if step_id in step_order else -1
            )

            # Try to find a snapshot for a step before the target
            if target_step_index > 0:
                for prior_step in reversed(step_order[:target_step_index]):
                    prior_snap = await self._store.get_snapshot(
                        pipeline_id, prior_step
                    )
                    if prior_snap:
                        state, memory, budget = self._snapshots.restore(
                            prior_snap
                        )
                        ctx.state = state
                        ctx.memory = memory
                        ctx.budget_constraint = budget
                        # Find the sequence number of the snapshot step's last event
                        for ev in reversed(all_events):
                            if ev.step_id == prior_step:
                                replay_after_seq = ev.sequence_number
                                break
                        logger.debug(
                            "Restored snapshot at step %s, replaying from seq %d",
                            prior_step,
                            replay_after_seq,
                        )
                        break

            # Replay events from after the snapshot (or from the start)
            events_to_replay = [
                ev
                for ev in all_events
                if ev.sequence_number > replay_after_seq
                and ev.sequence_number <= target_seq
            ]

            for event in events_to_replay:
                _apply_event(ctx, event)

            logger.info(
                "Replay complete: %d events processed, state keys: %s",
                ctx.events_processed,
                list(ctx.state.keys()),
            )
            return ctx
        finally:
            if self._owns_store:
                await self._store.close()

    async def replay_full(self, pipeline_id: str) -> list[StepResult]:
        """Replay an entire pipeline and return all step results.

        Args:
            pipeline_id: Pipeline to replay.

        Returns:
            Ordered list of ``StepResult`` objects from the replay.

        Raises:
            ReplayError: If the pipeline cannot be found.
        """
        logger.info("Full replay of pipeline %s", pipeline_id)
        await self._store.init()
        try:
            all_events = await self._store.get_events(pipeline_id)
            if not all_events:
                raise ReplayError(
                    f"No events found for pipeline '{pipeline_id}'"
                )

            ctx = ReplayContext()
            for event in all_events:
                _apply_event(ctx, event)

            logger.info(
                "Full replay complete: %d steps, %d events",
                len(ctx.step_results),
                ctx.events_processed,
            )
            return ctx.step_results
        finally:
            if self._owns_store:
                await self._store.close()

    async def fork_and_modify(
        self,
        pipeline_id: str,
        step_id: str,
        state_patches: dict[str, Any],
    ) -> str:
        """Fork a pipeline at a step and apply state modifications.

        Creates a new pipeline by forking at ``step_id``, then applies
        ``state_patches`` to the forked pipeline's state. The patches are
        recorded as a ``STATE_MODIFIED`` event in the forked pipeline.

        Args:
            pipeline_id: Source pipeline.
            step_id: Step at which to fork.
            state_patches: Dict of state keys to modify in the fork.

        Returns:
            The new forked pipeline_id.

        Raises:
            ReplayError: If forking fails.
        """
        logger.info(
            "Forking pipeline %s at step %s with patches: %s",
            pipeline_id,
            step_id,
            list(state_patches.keys()),
        )
        await self._store.init()
        try:
            try:
                forked_id = await self._store.fork(pipeline_id, step_id)
            except ValueError as exc:
                raise ReplayError(str(exc)) from exc

            # Apply state patches as a new event in the forked pipeline
            if state_patches:
                patch_event = Event(
                    event_type=EventType.STATE_MODIFIED,
                    pipeline_id=forked_id,
                    step_id=step_id,
                    step_name=f"fork_patch_at_{step_id}",
                    data={
                        "patches": state_patches,
                        "source_pipeline": pipeline_id,
                        "fork_reason": "manual_modification",
                    },
                )
                await self._store.append(patch_event)

                # Update the snapshot if one exists
                snapshot = await self._store.get_snapshot(forked_id, step_id)
                if snapshot:
                    state, memory, budget = self._snapshots.restore(snapshot)
                    state.update(state_patches)
                    new_snap = self._snapshots.capture(
                        pipeline_id=forked_id,
                        step_id=step_id,
                        step_index=snapshot.step_index,
                        state=state,
                        memory=memory,
                        budget_constraint=budget,
                        extra_metadata={
                            "forked_from": pipeline_id,
                            "patches_applied": list(state_patches.keys()),
                        },
                    )
                    await self._store.save_snapshot(new_snap)

            logger.info("Fork created: %s", forked_id)
            return forked_id
        finally:
            if self._owns_store:
                await self._store.close()
