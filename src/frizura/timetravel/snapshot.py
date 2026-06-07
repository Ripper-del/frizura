"""Snapshot manager for capturing and restoring execution state.

Provides the bridge between live execution state (ExecutionContext) and
persistent snapshots (StateSnapshot). Handles serialization of complex
Python objects into JSON-safe dicts and reconstruction on restore.
"""

from __future__ import annotations

import copy
import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from pydantic import BaseModel

from frizura.core.events import StateSnapshot
from frizura.models.budget import Budget, BudgetConstraint
from frizura.models.execution import Message, StepResult

logger = logging.getLogger(__name__)


def _make_serializable(obj: Any) -> Any:
    """Recursively convert an object to a JSON-serializable form.

    Handles Pydantic models, Decimals, datetimes, sets, and nested structures.
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, set):
        return [_make_serializable(v) for v in sorted(obj, key=str)]
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    # Last resort: str representation
    return str(obj)


class SnapshotManager:
    """Captures and restores execution state snapshots.

    The manager serializes the current execution context — pipeline state,
    conversation memory, budget tracker, and step results — into a
    ``StateSnapshot`` that can be persisted by the ``EventStore`` and later
    used to restore state for replay or forking.

    Example::

        manager = SnapshotManager()
        snapshot = manager.capture(state, memory, budget, step_id, step_index, pipeline_id)
        restored_state, restored_memory, restored_budget = manager.restore(snapshot)
    """

    def capture(
        self,
        *,
        pipeline_id: str,
        step_id: str,
        step_index: int,
        state: dict[str, Any],
        memory: list[Message] | list[dict[str, Any]] | None = None,
        budget_constraint: BudgetConstraint | None = None,
        step_results: list[StepResult] | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> StateSnapshot:
        """Capture the current execution state into a snapshot.

        Args:
            pipeline_id: The pipeline being executed.
            step_id: The current step identifier.
            step_index: Zero-based index of the current step.
            state: The mutable pipeline state dict.
            memory: Conversation history (list of Messages or raw dicts).
            budget_constraint: Current budget tracking state.
            step_results: Results of steps completed so far.
            extra_metadata: Arbitrary extra data to include in the snapshot.

        Returns:
            A new ``StateSnapshot`` ready for persistence.
        """
        serialized_state = _make_serializable(copy.deepcopy(state))

        # Serialize memory
        serialized_memory: list[dict[str, Any]] = []
        if memory:
            for msg in memory:
                if isinstance(msg, Message):
                    serialized_memory.append(msg.model_dump(mode="json"))
                elif isinstance(msg, dict):
                    serialized_memory.append(msg)

        # Serialize budget
        serialized_budget: dict[str, Any] = {}
        if budget_constraint is not None:
            serialized_budget = budget_constraint.model_dump(mode="json")

        # Build metadata
        metadata: dict[str, Any] = {
            "captured_at": datetime.now(timezone.utc).isoformat(),
        }
        if step_results:
            metadata["completed_steps"] = [
                {
                    "step_id": sr.step_id,
                    "step_name": sr.step_name,
                    "status": sr.status,
                    "model_used": sr.model_used,
                    "cost_usd": sr.cost_usd,
                    "duration_ms": sr.duration_ms,
                }
                for sr in step_results
            ]
        if extra_metadata:
            metadata.update(_make_serializable(extra_metadata))

        snapshot = StateSnapshot(
            pipeline_id=pipeline_id,
            step_id=step_id,
            step_index=step_index,
            state=serialized_state,
            memory=serialized_memory,
            budget_state=serialized_budget,
            metadata=metadata,
        )
        logger.debug(
            "Captured snapshot %s for pipeline %s at step %s (index %d)",
            snapshot.snapshot_id,
            pipeline_id,
            step_id,
            step_index,
        )
        return snapshot

    def restore(
        self, snapshot: StateSnapshot
    ) -> tuple[dict[str, Any], list[Message], BudgetConstraint | None]:
        """Restore execution state from a snapshot.

        Args:
            snapshot: The snapshot to restore from.

        Returns:
            A tuple of ``(state, memory, budget_constraint)`` reconstructed
            from the snapshot data.
        """
        # Deep-copy state so callers can mutate without affecting the snapshot
        state = copy.deepcopy(snapshot.state)

        # Reconstruct messages
        memory: list[Message] = []
        for msg_data in snapshot.memory:
            try:
                memory.append(Message.model_validate(msg_data))
            except Exception:
                logger.warning(
                    "Could not reconstruct Message from snapshot memory entry: %s",
                    msg_data,
                )

        # Reconstruct budget
        budget_constraint: BudgetConstraint | None = None
        if snapshot.budget_state:
            try:
                budget_constraint = BudgetConstraint.model_validate(
                    snapshot.budget_state
                )
            except Exception:
                logger.warning(
                    "Could not reconstruct BudgetConstraint from snapshot: %s",
                    snapshot.budget_state,
                )

        logger.debug(
            "Restored state from snapshot %s (step=%s, %d memory messages)",
            snapshot.snapshot_id,
            snapshot.step_id,
            len(memory),
        )
        return state, memory, budget_constraint

    def restore_step_results(
        self, snapshot: StateSnapshot
    ) -> list[dict[str, Any]]:
        """Extract the step result summaries from a snapshot's metadata.

        Returns a list of dicts with step_id, step_name, status, etc. These
        are summaries — not full ``StepResult`` objects — since we don't
        persist every field.
        """
        return snapshot.metadata.get("completed_steps", [])
