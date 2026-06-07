"""Frizura configuration models."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class ModelConfig(BaseModel):
    """Configuration for a single LLM model."""

    model_id: str
    provider: str  # "openai", "anthropic", "google", "ollama"
    display_name: str = ""
    context_window: int = 128_000
    max_output_tokens: int = 4096
    input_price_per_1m: Decimal = Decimal("0")  # USD per 1M input tokens
    output_price_per_1m: Decimal = Decimal("0")  # USD per 1M output tokens
    supports_json_mode: bool = True
    supports_tool_calling: bool = True
    supports_vision: bool = False
    supports_streaming: bool = True
    is_local: bool = False
    tier: str = "standard"  # "cheap", "standard", "premium"
    extra: dict[str, Any] = Field(default_factory=dict)


class ProviderConfig(BaseModel):
    """Configuration for a LLM provider."""

    name: str
    api_key: str | None = None
    base_url: str | None = None
    timeout: float = 30.0
    max_retries: int = 3
    extra: dict[str, Any] = Field(default_factory=dict)


class TimeTravelConfig(BaseModel):
    """Configuration for time-travel debugging."""

    enabled: bool = True
    db_path: Path = Path(".frizura/events.db")
    snapshot_every_n_steps: int = 1
    max_events_per_pipeline: int = 10_000


class RouterConfig(BaseModel):
    """Configuration for smart routing."""

    enabled: bool = True
    default_strategy: str = "cascade"  # "cascade", "cheapest", "fastest", "best"
    complexity_threshold_cheap: float = 0.3
    complexity_threshold_premium: float = 0.7


class SwarmConfig(BaseModel):
    """Configuration for hybrid swarm."""

    enabled: bool = True
    local_first: bool = True
    ollama_hosts: list[str] = Field(default_factory=lambda: ["http://localhost:11434"])
    auto_discover_lan: bool = False
    privacy_mode: str = "auto"  # "auto", "local_only", "cloud_only"


class FrizuraConfig(BaseSettings):
    """Main Frizura configuration, loaded from env vars and .env files."""

    model_config = {"env_prefix": "FRIZURA_", "env_file": ".env", "extra": "ignore"}

    # Provider API keys
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    google_api_key: str | None = None

    # Default model
    default_model: str = "openai:gpt-4o-mini"

    # Sub-configs
    timetravel: TimeTravelConfig = Field(default_factory=TimeTravelConfig)
    router: RouterConfig = Field(default_factory=RouterConfig)
    swarm: SwarmConfig = Field(default_factory=SwarmConfig)

    # Logging
    log_level: str = "INFO"
    verbose: bool = False
