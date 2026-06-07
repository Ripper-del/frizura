"""Provider-related models — model info, completion config."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    """Information about a specific LLM model."""

    model_id: str  # e.g. "gpt-4o-mini", "claude-3-5-sonnet", "gemini-2.0-flash"
    provider: str  # "openai", "anthropic", "google", "ollama"
    display_name: str = ""
    context_window: int = 128_000
    max_output_tokens: int = 4096
    input_price_per_1m: Decimal = Decimal("0")
    output_price_per_1m: Decimal = Decimal("0")
    supports_json_mode: bool = True
    supports_tool_calling: bool = True
    supports_vision: bool = False
    supports_streaming: bool = True
    is_local: bool = False
    tier: str = "standard"  # "cheap", "standard", "premium"

    @property
    def cost_per_input_token(self) -> Decimal:
        return self.input_price_per_1m / Decimal("1000000")

    @property
    def cost_per_output_token(self) -> Decimal:
        return self.output_price_per_1m / Decimal("1000000")

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> Decimal:
        """Estimate cost for given token counts."""
        return (
            self.cost_per_input_token * input_tokens
            + self.cost_per_output_token * output_tokens
        )


class CompletionConfig(BaseModel):
    """Configuration for a single LLM completion request."""

    model: str | None = None  # Override model, or None for auto-routing
    temperature: float = 0.7
    max_tokens: int | None = None
    top_p: float | None = None
    stop: list[str] | None = None
    json_mode: bool = False
    json_schema: dict[str, Any] | None = None  # JSON Schema for structured output
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | None = None  # "auto", "none", or specific tool name
    seed: int | None = None
    extra: dict[str, Any] = Field(default_factory=dict)
