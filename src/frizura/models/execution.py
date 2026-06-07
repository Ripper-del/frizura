"""Execution-related data models — messages, responses, results."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class MessageRole(StrEnum):
    """Role of a message in a conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    """A single message in a conversation."""

    role: MessageRole
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class TokenUsage(BaseModel):
    """Token usage for a single LLM call."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0


class LLMResponse(BaseModel):
    """Response from an LLM provider."""

    content: str
    model: str
    provider: str
    usage: TokenUsage = Field(default_factory=TokenUsage)
    finish_reason: str = "stop"
    latency_ms: float = 0.0
    raw: dict[str, Any] = Field(default_factory=dict, exclude=True)
    tool_calls: list[dict[str, Any]] | None = None


class StreamChunk(BaseModel):
    """A single chunk from a streaming LLM response."""

    content: str = ""
    is_final: bool = False
    usage: TokenUsage | None = None
    finish_reason: str | None = None


class StepResult(BaseModel):
    """Result of a single pipeline step execution."""

    step_id: str
    step_name: str
    status: str = "completed"  # "completed", "failed", "skipped", "healed"
    output: Any = None
    llm_response: LLMResponse | None = None
    duration_ms: float = 0.0
    cost_usd: float = 0.0
    model_used: str = ""
    routing_reason: str = ""
    healing_attempts: int = 0
    events_count: int = 0
    error: str | None = None
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None


class PipelineResult(BaseModel):
    """Result of a full pipeline execution."""

    pipeline_id: str = Field(default_factory=lambda: uuid4().hex[:12])
    pipeline_name: str = ""
    status: str = "completed"  # "completed", "failed", "partial"
    output: Any = None
    steps: list[StepResult] = Field(default_factory=list)
    total_duration_ms: float = 0.0
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    models_used: list[str] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
    error: str | None = None
