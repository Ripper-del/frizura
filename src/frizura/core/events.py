"""Event types for the event-sourcing system."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class EventType(StrEnum):
    """All event types recorded by the Frizura event store."""

    # Pipeline lifecycle
    PIPELINE_STARTED = "pipeline.started"
    PIPELINE_COMPLETED = "pipeline.completed"
    PIPELINE_FAILED = "pipeline.failed"

    # Step lifecycle
    STEP_STARTED = "step.started"
    STEP_COMPLETED = "step.completed"
    STEP_FAILED = "step.failed"
    STEP_SKIPPED = "step.skipped"

    # LLM interactions
    LLM_REQUEST = "llm.request"
    LLM_RESPONSE = "llm.response"
    LLM_STREAM_START = "llm.stream.start"
    LLM_STREAM_END = "llm.stream.end"

    # Tool calling
    TOOL_CALL = "tool.call"
    TOOL_RESULT = "tool.result"

    # Schema validation & healing
    SCHEMA_VALIDATION_OK = "schema.validation.ok"
    SCHEMA_VALIDATION_FAILED = "schema.validation.failed"
    SCHEMA_HEAL_ATTEMPT = "schema.heal.attempt"
    SCHEMA_HEAL_SUCCESS = "schema.heal.success"
    SCHEMA_HEAL_FAILED = "schema.heal.failed"

    # Routing decisions
    ROUTING_DECISION = "routing.decision"
    ROUTING_FALLBACK = "routing.fallback"

    # Budget
    BUDGET_CHECK = "budget.check"
    BUDGET_EXHAUSTED = "budget.exhausted"

    # State management
    STATE_SNAPSHOT = "state.snapshot"
    STATE_MODIFIED = "state.modified"

    # Swarm / Privacy
    PRIVACY_CLASSIFICATION = "privacy.classification"
    PII_MASKED = "pii.masked"
    PII_UNMASKED = "pii.unmasked"
    SWARM_ROUTED_LOCAL = "swarm.routed.local"
    SWARM_ROUTED_CLOUD = "swarm.routed.cloud"

    # Optimization
    FEEDBACK_RECORDED = "optimization.feedback"
    OPTIMIZATION_STARTED = "optimization.started"
    OPTIMIZATION_COMPLETED = "optimization.completed"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Event(BaseModel):
    """A single event in the event-sourced execution log.
    
    Events are immutable records of everything that happens during pipeline
    execution. They enable time-travel debugging, replay, and auditing.
    """

    id: str = Field(default_factory=lambda: uuid4().hex[:16])
    timestamp: datetime = Field(default_factory=_utcnow)
    event_type: EventType
    pipeline_id: str
    step_id: str | None = None
    step_name: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    parent_event_id: str | None = None
    sequence_number: int = 0  # Auto-incremented within a pipeline

    model_config = {"frozen": True}  # Events are immutable


class StateSnapshot(BaseModel):
    """A snapshot of the full execution state at a point in time.
    
    Used for time-travel: replay events up to this point to restore state.
    """

    snapshot_id: str = Field(default_factory=lambda: uuid4().hex[:16])
    pipeline_id: str
    step_id: str
    step_index: int
    timestamp: datetime = Field(default_factory=_utcnow)
    state: dict[str, Any] = Field(default_factory=dict)
    memory: list[dict[str, Any]] = Field(default_factory=list)  # Conversation history
    budget_state: dict[str, Any] = Field(default_factory=dict)  # Budget tracker state
    metadata: dict[str, Any] = Field(default_factory=dict)
