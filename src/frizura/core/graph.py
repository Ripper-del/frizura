"""Pipeline graph builder — DAG construction, validation, and compilation.

A ``Pipeline`` is an ordered collection of ``Step`` nodes that may have
explicit dependency edges.  Steps can be added linearly, in parallel, or
conditionally (branch).  ``Pipeline.compile()`` performs a topological sort,
validates the DAG, and returns a ``CompiledPipeline`` whose
``steps_order`` contains lists of steps that can run concurrently.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from enum import StrEnum
from typing import Any, Callable
from uuid import uuid4

from pydantic import BaseModel, Field

from frizura.core.exceptions import GraphError
from frizura.models.budget import Budget
from frizura.models.providers import CompletionConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------- #
# Step definition
# ---------------------------------------------------------------------- #

class StepType(StrEnum):
    """Classifies the kind of work a step performs."""

    LLM = "llm"
    TRANSFORM = "transform"
    BRANCH = "branch"
    PARALLEL = "parallel"
    HUMAN = "human"


class Step(BaseModel):
    """A single step in a pipeline.

    Parameters
    ----------
    id:
        Unique identifier (auto-generated).
    name:
        Human-readable name.
    step_type:
        What kind of step this is.
    handler:
        The callable that implements the step's logic.  For ``LLM`` steps
        this is typically ``None`` — the engine will call the provider.
        For ``TRANSFORM`` steps this must be a sync or async function.
    system_prompt:
        Optional system prompt injected before the step's messages.
    output_schema:
        If set, the LLM output will be validated (and possibly healed)
        against this Pydantic model.
    model:
        Explicit model identifier (``"provider:model_id"``).  ``None``
        means the smart-router picks the model.
    budget:
        Per-step budget constraints (independent of the pipeline budget).
    privacy:
        ``"auto"`` (let the classifier decide), ``"public"``, or
        ``"confidential"``.
    depends_on:
        Step IDs that must complete before this step starts.
    condition:
        Optional callable ``(ctx) -> bool``.  If it returns ``False`` the
        step is skipped.
    config:
        Extra ``CompletionConfig`` overrides for this step.
    """

    model_config = {"arbitrary_types_allowed": True}

    id: str = Field(default_factory=lambda: uuid4().hex[:10])
    name: str
    step_type: StepType = StepType.LLM
    handler: Any | None = None  # Callable — stored as Any for Pydantic compat
    system_prompt: str | None = None
    output_schema: Any | None = None  # type[BaseModel] — stored as Any
    model: str | None = None
    budget: Budget | None = None
    privacy: str = "auto"
    depends_on: list[str] = Field(default_factory=list)
    condition: Any | None = None  # Callable[[ExecutionContext], bool]
    config: CompletionConfig = Field(default_factory=CompletionConfig)

    def __repr__(self) -> str:
        return f"Step(id={self.id!r}, name={self.name!r}, type={self.step_type})"


# ---------------------------------------------------------------------- #
# Compiled pipeline
# ---------------------------------------------------------------------- #

class CompiledPipeline(BaseModel):
    """Result of ``Pipeline.compile()``.

    Attributes
    ----------
    steps_order:
        A list of *groups*.  Each group is a ``list[Step]`` whose members
        have no mutual dependencies and may therefore run concurrently.
        Groups themselves are ordered — group *i* finishes before group
        *i+1* starts.
    is_valid:
        ``True`` if the DAG has no issues.
    warnings:
        Non-fatal observations (e.g. unused steps).
    """

    model_config = {"arbitrary_types_allowed": True}

    steps_order: list[list[Step]] = Field(default_factory=list)
    is_valid: bool = True
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------- #
# Pipeline builder
# ---------------------------------------------------------------------- #

class Pipeline:
    """Fluent pipeline builder.

    Usage::

        pipeline = (
            Pipeline("summarise-and-tag")
            .add_step(Step(name="summarise", system_prompt="Summarise this."))
            .add_step(Step(name="tag", system_prompt="Extract tags.", depends_on=["<id>"]))
        )
        compiled = pipeline.compile()
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.steps: list[Step] = []
        self._step_index: dict[str, Step] = {}  # id → Step

    # ------------------------------------------------------------------ #
    # Fluent API
    # ------------------------------------------------------------------ #

    def add_step(self, step: Step) -> Pipeline:
        """Append a step to the pipeline.

        If the step has no explicit ``depends_on`` and there is already at
        least one step in the pipeline, the new step will implicitly depend
        on the *last* added step (creating a simple linear chain).

        Returns *self* for chaining.
        """
        if step.id in self._step_index:
            raise GraphError(f"Duplicate step id: {step.id!r}")

        # Auto-chain: if no explicit deps, depend on the last step.
        if not step.depends_on and self.steps:
            last = self.steps[-1]
            step.depends_on = [last.id]

        self.steps.append(step)
        self._step_index[step.id] = step
        return self

    def add_parallel(self, *steps: Step) -> Pipeline:
        """Add multiple steps that should execute concurrently.

        All added steps share the same ``depends_on`` — namely the previous
        step in the pipeline (if any).  A synthetic *join* is **not**
        inserted automatically; subsequent ``add_step`` calls will depend
        on the *last* of the parallel steps.  To join them, add a step
        that explicitly ``depends_on`` all their IDs.
        """
        if not steps:
            raise GraphError("add_parallel requires at least one step")

        prev_id = self.steps[-1].id if self.steps else None
        for s in steps:
            if s.id in self._step_index:
                raise GraphError(f"Duplicate step id: {s.id!r}")
            # All parallel siblings depend on the same predecessor (if any)
            if not s.depends_on and prev_id:
                s.depends_on = [prev_id]
            s.step_type = StepType.PARALLEL
            self.steps.append(s)
            self._step_index[s.id] = s
        return self

    def add_branch(
        self,
        condition_fn: Callable[..., str],
        branches: dict[str, Step],
    ) -> Pipeline:
        """Add a conditional branch.

        ``condition_fn(ctx) -> str`` should return a key from *branches*.
        Each branch step is added with its ``condition`` set so that only
        the selected branch actually executes at runtime.

        Returns *self* for chaining.
        """
        if not branches:
            raise GraphError("add_branch requires at least one branch")

        prev_id = self.steps[-1].id if self.steps else None

        for key, step in branches.items():
            if step.id in self._step_index:
                raise GraphError(f"Duplicate step id: {step.id!r}")
            step.step_type = StepType.BRANCH
            if not step.depends_on and prev_id:
                step.depends_on = [prev_id]
            # Wrap condition: runs the routing function and checks the key
            _key = key  # capture
            _fn = condition_fn

            def _make_cond(k: str, fn: Callable[..., str]) -> Callable[..., bool]:
                def _cond(ctx: Any) -> bool:
                    return fn(ctx) == k
                return _cond

            step.condition = _make_cond(_key, _fn)
            self.steps.append(step)
            self._step_index[step.id] = step
        return self

    # ------------------------------------------------------------------ #
    # Compilation / DAG validation
    # ------------------------------------------------------------------ #

    def compile(self) -> CompiledPipeline:
        """Validate the DAG and topologically sort the steps.

        Raises :class:`GraphError` on cycles or missing dependencies.
        """
        warnings: list[str] = []

        if not self.steps:
            return CompiledPipeline(
                steps_order=[],
                is_valid=True,
                warnings=["Pipeline has no steps."],
            )

        # --- validate dependency references ----
        all_ids = {s.id for s in self.steps}
        for s in self.steps:
            for dep in s.depends_on:
                if dep not in all_ids:
                    raise GraphError(
                        f"Step {s.name!r} ({s.id}) depends on unknown "
                        f"step {dep!r}"
                    )

        # --- Kahn's algorithm for topological sort ----
        in_degree: dict[str, int] = {s.id: 0 for s in self.steps}
        dependents: dict[str, list[str]] = defaultdict(list)
        for s in self.steps:
            for dep in s.depends_on:
                in_degree[s.id] += 1
                dependents[dep].append(s.id)

        queue: deque[str] = deque()
        for sid, deg in in_degree.items():
            if deg == 0:
                queue.append(sid)

        groups: list[list[Step]] = []
        visited = 0

        while queue:
            # All items currently in the queue have no unsatisfied deps —
            # they form a parallelisable group.
            group_ids = list(queue)
            queue.clear()
            group: list[Step] = [self._step_index[sid] for sid in group_ids]
            groups.append(group)
            visited += len(group)

            for sid in group_ids:
                for child in dependents[sid]:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        queue.append(child)

        if visited != len(self.steps):
            # Remaining nodes with non-zero in-degree form a cycle.
            cycle_members = [
                self._step_index[sid].name
                for sid, deg in in_degree.items()
                if deg > 0
            ]
            raise GraphError(
                f"Cycle detected in pipeline DAG involving steps: "
                f"{cycle_members}"
            )

        logger.info(
            "Pipeline %r compiled: %d steps in %d groups",
            self.name,
            len(self.steps),
            len(groups),
        )
        return CompiledPipeline(
            steps_order=groups,
            is_valid=True,
            warnings=warnings,
        )

    def get_execution_order(self) -> list[list[Step]]:
        """Convenience wrapper — compile and return the step groups."""
        return self.compile().steps_order

    # ------------------------------------------------------------------ #
    # Dunder helpers
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return f"Pipeline(name={self.name!r}, steps={len(self.steps)})"

    def __len__(self) -> int:
        return len(self.steps)
