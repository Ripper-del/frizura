"""Execution context — shared mutable state passed through every pipeline step.

The ``ExecutionContext`` is the single object that flows through the pipeline
DAG.  It carries:

* **state** — an arbitrary key-value dict that steps read/write.
* **memory** — the conversation history (list of ``Message`` objects).
* **budget** — a live ``BudgetConstraint`` tracker (optional).
* **metadata** — per-pipeline metadata such as tags, run-id, user-id, etc.

The context can be serialised to a ``StateSnapshot`` for time-travel
debugging and restored from one via the ``from_snapshot`` class method.
"""

from __future__ import annotations

import copy
import logging
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from frizura.core.events import StateSnapshot
from frizura.models.budget import BudgetConstraint
from frizura.models.config import FrizuraConfig
from frizura.models.execution import Message

logger = logging.getLogger(__name__)


class ExecutionContext(BaseModel):
    """Mutable execution context threaded through every pipeline step.

    Parameters
    ----------
    pipeline_id:
        Unique identifier for this pipeline run.
    pipeline_name:
        Human-readable name of the pipeline.
    state:
        Shared mutable key-value store.  Steps read and write arbitrary
        data here via :meth:`get` / :meth:`set`.
    budget:
        Optional budget tracker that is decremented as tokens/cost/time
        are consumed.
    memory:
        Ordered conversation history.  Steps append to this list so
        subsequent steps can reference earlier messages.
    metadata:
        Arbitrary metadata attached to this run (e.g. user-id, tags).
    current_step_index:
        Zero-based index of the step that is currently executing.
    config:
        Global Frizura configuration.
    """

    model_config = {"arbitrary_types_allowed": True}

    pipeline_id: str = Field(default_factory=lambda: uuid4().hex[:12])
    pipeline_name: str = ""
    state: dict[str, Any] = Field(default_factory=dict)
    budget: BudgetConstraint | None = None
    memory: list[Message] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    current_step_index: int = 0
    config: FrizuraConfig = Field(default_factory=FrizuraConfig)

    # ------------------------------------------------------------------ #
    # State helpers
    # ------------------------------------------------------------------ #

    def set(self, key: str, value: Any) -> None:  # noqa: A003
        """Store *value* under *key* in the shared state dict."""
        self.state[key] = value
        logger.debug("state[%r] = %s", key, _ellipsis(value))

    def get(self, key: str, default: Any = None) -> Any:
        """Return ``state[key]``, falling back to *default*."""
        return self.state.get(key, default)

    # ------------------------------------------------------------------ #
    # Memory helpers
    # ------------------------------------------------------------------ #

    def add_message(self, message: Message) -> None:
        """Append a message to the conversation history."""
        self.memory.append(message)
        logger.debug("memory += %s (%d chars)", message.role, len(message.content))

    # ------------------------------------------------------------------ #
    # Snapshot / restore (time-travel)
    # ------------------------------------------------------------------ #

    def to_snapshot(self, step_id: str, step_index: int) -> StateSnapshot:
        """Serialise the current context into an immutable snapshot.

        The snapshot captures ``state``, ``memory`` and ``budget`` so that
        the context can be fully restored later (e.g. during replay).
        """
        budget_state: dict[str, Any] = {}
        if self.budget is not None:
            budget_state = self.budget.model_dump(mode="json")

        return StateSnapshot(
            pipeline_id=self.pipeline_id,
            step_id=step_id,
            step_index=step_index,
            state=copy.deepcopy(self.state),
            memory=[m.model_dump(mode="json") for m in self.memory],
            budget_state=budget_state,
            metadata=copy.deepcopy(self.metadata),
        )

    @classmethod
    def from_snapshot(
        cls,
        snapshot: StateSnapshot,
        config: FrizuraConfig | None = None,
    ) -> ExecutionContext:
        """Reconstruct an ``ExecutionContext`` from a previously saved snapshot.

        Parameters
        ----------
        snapshot:
            The snapshot to restore from.
        config:
            Optional config override.  If ``None`` a default config is used.
        """
        budget: BudgetConstraint | None = None
        if snapshot.budget_state:
            budget = BudgetConstraint.model_validate(snapshot.budget_state)

        memory = [Message.model_validate(m) for m in snapshot.memory]

        return cls(
            pipeline_id=snapshot.pipeline_id,
            pipeline_name=snapshot.metadata.get("pipeline_name", ""),
            state=copy.deepcopy(snapshot.state),
            budget=budget,
            memory=memory,
            metadata=copy.deepcopy(snapshot.metadata),
            current_step_index=snapshot.step_index,
            config=config or FrizuraConfig(),
        )

    # ------------------------------------------------------------------ #
    # Cloning (for forking / parallel branches)
    # ------------------------------------------------------------------ #

    def clone(self) -> ExecutionContext:
        """Return a deep copy of this context.

        Useful when forking execution into parallel branches — each branch
        gets its own independent context that starts in the same state.
        """
        return ExecutionContext(
            pipeline_id=self.pipeline_id,
            pipeline_name=self.pipeline_name,
            state=copy.deepcopy(self.state),
            budget=(
                self.budget.model_copy(deep=True)
                if self.budget is not None
                else None
            ),
            memory=[m.model_copy(deep=True) for m in self.memory],
            metadata=copy.deepcopy(self.metadata),
            current_step_index=self.current_step_index,
            config=self.config.model_copy(deep=True),
        )

    # ------------------------------------------------------------------ #
    # Dunder helpers
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return (
            f"ExecutionContext(pipeline={self.pipeline_name!r}, "
            f"step_idx={self.current_step_index}, "
            f"state_keys={list(self.state.keys())}, "
            f"memory_len={len(self.memory)})"
        )


# ---------------------------------------------------------------------- #
# Internal helpers
# ---------------------------------------------------------------------- #

def _ellipsis(value: Any, max_len: int = 80) -> str:
    """Return a short repr of *value*, truncated with '…' if needed."""
    s = repr(value)
    return s if len(s) <= max_len else s[: max_len - 1] + "…"
