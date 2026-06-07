"""Model registry — the single source of truth for available LLM models.

The ``ModelRegistry`` holds ``ModelInfo`` entries for every known model and
provides lookup, filtering, and provider instantiation.

Key features:

* Pre-populated ``DEFAULT_MODELS`` with 2025 pricing for OpenAI, Anthropic,
  Google, and a placeholder for Ollama local models.
* ``get_provider("openai:gpt-4o-mini")`` parses the ``provider:model``
  format and returns a ready-to-use ``LLMProvider`` instance.
* ``find_by_capability()`` queries models by feature flags.
* ``cheapest_for()`` selects the cheapest model satisfying constraints.
* Auto-registration from ``FrizuraConfig`` API keys.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any

from frizura.core.exceptions import (
    ModelNotFoundError,
    ProviderNotFoundError,
)
from frizura.models.config import FrizuraConfig
from frizura.models.providers import CompletionConfig, ModelInfo
from frizura.providers.base import LLMProvider

logger = logging.getLogger("frizura.providers.registry")


# ---------------------------------------------------------------------------
# Default model catalogue (2025 pricing)
# ---------------------------------------------------------------------------

DEFAULT_MODELS: dict[str, ModelInfo] = {
    # ── OpenAI ────────────────────────────────────────────────────────
    "gpt-4o": ModelInfo(
        model_id="gpt-4o",
        provider="openai",
        display_name="GPT-4o",
        context_window=128_000,
        max_output_tokens=16_384,
        input_price_per_1m=Decimal("2.50"),
        output_price_per_1m=Decimal("10.00"),
        supports_json_mode=True,
        supports_tool_calling=True,
        supports_vision=True,
        supports_streaming=True,
        tier="standard",
    ),
    "gpt-4o-mini": ModelInfo(
        model_id="gpt-4o-mini",
        provider="openai",
        display_name="GPT-4o mini",
        context_window=128_000,
        max_output_tokens=16_384,
        input_price_per_1m=Decimal("0.15"),
        output_price_per_1m=Decimal("0.60"),
        supports_json_mode=True,
        supports_tool_calling=True,
        supports_vision=True,
        supports_streaming=True,
        tier="cheap",
    ),
    "gpt-4-turbo": ModelInfo(
        model_id="gpt-4-turbo",
        provider="openai",
        display_name="GPT-4 Turbo",
        context_window=128_000,
        max_output_tokens=4096,
        input_price_per_1m=Decimal("10.00"),
        output_price_per_1m=Decimal("30.00"),
        supports_json_mode=True,
        supports_tool_calling=True,
        supports_vision=True,
        supports_streaming=True,
        tier="premium",
    ),
    "o1": ModelInfo(
        model_id="o1",
        provider="openai",
        display_name="o1",
        context_window=200_000,
        max_output_tokens=100_000,
        input_price_per_1m=Decimal("15.00"),
        output_price_per_1m=Decimal("60.00"),
        supports_json_mode=True,
        supports_tool_calling=True,
        supports_vision=True,
        supports_streaming=True,
        tier="premium",
    ),
    "o3-mini": ModelInfo(
        model_id="o3-mini",
        provider="openai",
        display_name="o3-mini",
        context_window=200_000,
        max_output_tokens=100_000,
        input_price_per_1m=Decimal("1.10"),
        output_price_per_1m=Decimal("4.40"),
        supports_json_mode=True,
        supports_tool_calling=True,
        supports_vision=False,
        supports_streaming=True,
        tier="standard",
    ),
    # ── Anthropic ─────────────────────────────────────────────────────
    "claude-3-5-sonnet-latest": ModelInfo(
        model_id="claude-3-5-sonnet-latest",
        provider="anthropic",
        display_name="Claude 3.5 Sonnet",
        context_window=200_000,
        max_output_tokens=8192,
        input_price_per_1m=Decimal("3.00"),
        output_price_per_1m=Decimal("15.00"),
        supports_json_mode=True,
        supports_tool_calling=True,
        supports_vision=True,
        supports_streaming=True,
        tier="standard",
    ),
    "claude-3-5-haiku-latest": ModelInfo(
        model_id="claude-3-5-haiku-latest",
        provider="anthropic",
        display_name="Claude 3.5 Haiku",
        context_window=200_000,
        max_output_tokens=8192,
        input_price_per_1m=Decimal("0.80"),
        output_price_per_1m=Decimal("4.00"),
        supports_json_mode=True,
        supports_tool_calling=True,
        supports_vision=False,
        supports_streaming=True,
        tier="cheap",
    ),
    "claude-4-opus-latest": ModelInfo(
        model_id="claude-4-opus-latest",
        provider="anthropic",
        display_name="Claude 4 Opus",
        context_window=200_000,
        max_output_tokens=32_000,
        input_price_per_1m=Decimal("15.00"),
        output_price_per_1m=Decimal("75.00"),
        supports_json_mode=True,
        supports_tool_calling=True,
        supports_vision=True,
        supports_streaming=True,
        tier="premium",
    ),
    # ── Google ────────────────────────────────────────────────────────
    "gemini-2.0-flash": ModelInfo(
        model_id="gemini-2.0-flash",
        provider="google",
        display_name="Gemini 2.0 Flash",
        context_window=1_000_000,
        max_output_tokens=8192,
        input_price_per_1m=Decimal("0.10"),
        output_price_per_1m=Decimal("0.40"),
        supports_json_mode=True,
        supports_tool_calling=True,
        supports_vision=True,
        supports_streaming=True,
        tier="cheap",
    ),
    "gemini-2.5-pro": ModelInfo(
        model_id="gemini-2.5-pro",
        provider="google",
        display_name="Gemini 2.5 Pro",
        context_window=1_000_000,
        max_output_tokens=65_536,
        input_price_per_1m=Decimal("1.25"),
        output_price_per_1m=Decimal("10.00"),
        supports_json_mode=True,
        supports_tool_calling=True,
        supports_vision=True,
        supports_streaming=True,
        tier="standard",
    ),
    "gemini-2.5-flash": ModelInfo(
        model_id="gemini-2.5-flash",
        provider="google",
        display_name="Gemini 2.5 Flash",
        context_window=1_000_000,
        max_output_tokens=65_536,
        input_price_per_1m=Decimal("0.15"),
        output_price_per_1m=Decimal("0.60"),
        supports_json_mode=True,
        supports_tool_calling=True,
        supports_vision=True,
        supports_streaming=True,
        tier="cheap",
    ),
}


# ---------------------------------------------------------------------------
# Provider factory callables
# ---------------------------------------------------------------------------

# Maps provider name → (module_path, class_name) for lazy import
_PROVIDER_FACTORIES: dict[str, tuple[str, str]] = {
    "openai": ("frizura.providers.openai", "OpenAIProvider"),
    "anthropic": ("frizura.providers.anthropic", "AnthropicProvider"),
    "google": ("frizura.providers.google", "GoogleProvider"),
    "ollama": ("frizura.providers.ollama", "OllamaProvider"),
}


def _import_provider_class(provider_name: str) -> type[LLMProvider]:
    """Lazily import a provider class by name."""
    if provider_name not in _PROVIDER_FACTORIES:
        raise ProviderNotFoundError(
            f"Unknown provider '{provider_name}'. "
            f"Available: {', '.join(sorted(_PROVIDER_FACTORIES))}"
        )
    module_path, class_name = _PROVIDER_FACTORIES[provider_name]
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, class_name)


# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------


class ModelRegistry:
    """Central registry of known LLM models and their provider adapters.

    Usage::

        registry = ModelRegistry.from_config(config)
        provider = registry.get_provider("openai:gpt-4o-mini")
        response = await provider.complete(messages, config)
    """

    _shared_state: dict[str, Any] = {}

    def __init__(self) -> None:
        self.__dict__ = self._shared_state
        if not hasattr(self, "_models"):
            self._models: dict[str, ModelInfo] = {}
            self._providers: dict[str, LLMProvider] = {}  # cached instances
            self._api_keys: dict[str, str] = {}  # provider → api_key
            self._base_urls: dict[str, str] = {}  # provider → base_url
            self._timeouts: dict[str, float] = {}  # provider → timeout

    # ------------------------------------------------------------------
    # Model registration
    # ------------------------------------------------------------------

    def register(self, model: ModelInfo) -> None:
        """Register a model in the catalogue.

        If a model with the same ``model_id`` already exists, it is
        silently replaced.
        """
        self._models[model.model_id] = model
        logger.debug("Registered model: %s (%s)", model.model_id, model.provider)

    def register_many(self, models: dict[str, ModelInfo]) -> None:
        """Register multiple models at once."""
        for info in models.values():
            self.register(info)

    def unregister(self, model_id: str) -> None:
        """Remove a model from the catalogue."""
        self._models.pop(model_id, None)
        self._providers.pop(model_id, None)

    # ------------------------------------------------------------------
    # Model queries
    # ------------------------------------------------------------------

    def get(self, model_id: str) -> ModelInfo:
        """Look up a model by ID.

        Raises ``ModelNotFoundError`` if the model is not registered.
        """
        info = self._models.get(model_id)
        if info is None:
            raise ModelNotFoundError(
                f"Model '{model_id}' not found in registry. "
                f"Available models: {', '.join(sorted(self._models))}"
            )
        return info

    def list_models(
        self,
        *,
        provider: str | None = None,
        tier: str | None = None,
    ) -> list[ModelInfo]:
        """List all registered models, optionally filtered by provider/tier."""
        models = list(self._models.values())
        if provider:
            models = [m for m in models if m.provider == provider]
        if tier:
            models = [m for m in models if m.tier == tier]
        return models

    def find_by_capability(
        self,
        *,
        json_mode: bool = False,
        tool_calling: bool = False,
        vision: bool = False,
        streaming: bool = False,
        local_only: bool = False,
    ) -> list[ModelInfo]:
        """Find models matching a set of capability requirements."""
        result: list[ModelInfo] = []
        for model in self._models.values():
            if json_mode and not model.supports_json_mode:
                continue
            if tool_calling and not model.supports_tool_calling:
                continue
            if vision and not model.supports_vision:
                continue
            if streaming and not model.supports_streaming:
                continue
            if local_only and not model.is_local:
                continue
            result.append(model)
        return result

    def cheapest_for(
        self,
        *,
        estimated_input_tokens: int = 1000,
        estimated_output_tokens: int = 1000,
        json_mode: bool = False,
        tool_calling: bool = False,
        vision: bool = False,
        local_only: bool = False,
        exclude_providers: set[str] | None = None,
        exclude_models: set[str] | None = None,
    ) -> ModelInfo | None:
        """Find the cheapest model satisfying the given constraints.

        Returns ``None`` if no model matches.
        """
        candidates = self.find_by_capability(
            json_mode=json_mode,
            tool_calling=tool_calling,
            vision=vision,
            local_only=local_only,
        )

        if exclude_providers:
            candidates = [c for c in candidates if c.provider not in exclude_providers]
        if exclude_models:
            candidates = [c for c in candidates if c.model_id not in exclude_models]

        # Filter to providers that have API keys configured (or are local)
        candidates = [
            c
            for c in candidates
            if c.is_local or c.provider in self._api_keys
        ]

        if not candidates:
            return None

        return min(
            candidates,
            key=lambda m: m.estimate_cost(estimated_input_tokens, estimated_output_tokens),
        )

    # ------------------------------------------------------------------
    # Provider instantiation
    # ------------------------------------------------------------------

    def get_provider(self, model_spec: str) -> LLMProvider:
        """Get (or create) a provider instance for the given model spec.

        ``model_spec`` can be:
        * ``"provider:model_id"`` — e.g. ``"openai:gpt-4o-mini"``
        * ``"model_id"`` — looked up in the registry to find the provider

        Raises ``ModelNotFoundError`` or ``ProviderNotFoundError`` on failure.
        """
        provider_name, model_id = self._parse_model_spec(model_spec)

        # Check cache
        cache_key = f"{provider_name}:{model_id}"
        if cache_key in self._providers:
            return self._providers[cache_key]

        # Look up model info
        model_info = self._models.get(model_id)
        if model_info is None:
            raise ModelNotFoundError(
                f"Model '{model_id}' not found in registry. "
                f"Available: {', '.join(sorted(self._models))}"
            )

        # Verify provider matches
        if provider_name and model_info.provider != provider_name:
            raise ModelNotFoundError(
                f"Model '{model_id}' is registered under provider "
                f"'{model_info.provider}', not '{provider_name}'"
            )

        provider_name = model_info.provider

        # Instantiate the provider class
        cls = _import_provider_class(provider_name)
        provider = cls(
            model_info=model_info,
            api_key=self._api_keys.get(provider_name),
            base_url=self._base_urls.get(provider_name),
            timeout=self._timeouts.get(provider_name, 60.0),
        )

        # Cache and return
        self._providers[cache_key] = provider
        return provider

    def set_api_key(self, provider: str, api_key: str) -> None:
        """Configure an API key for a provider."""
        self._api_keys[provider] = api_key
        # Invalidate cached providers for this provider
        to_remove = [k for k in self._providers if k.startswith(f"{provider}:")]
        for k in to_remove:
            del self._providers[k]

    def set_base_url(self, provider: str, base_url: str) -> None:
        """Configure a custom base URL for a provider."""
        self._base_urls[provider] = base_url
        to_remove = [k for k in self._providers if k.startswith(f"{provider}:")]
        for k in to_remove:
            del self._providers[k]

    def set_timeout(self, provider: str, timeout: float) -> None:
        """Configure request timeout for a provider."""
        self._timeouts[provider] = timeout

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def with_defaults(cls) -> ModelRegistry:
        """Create a registry pre-populated with the default model catalogue."""
        registry = cls()
        registry.register_many(DEFAULT_MODELS)
        return registry

    @classmethod
    def from_config(cls, config: FrizuraConfig) -> ModelRegistry:
        """Create a registry from a ``FrizuraConfig``, registering API keys
        and pre-populating models.

        Only providers with configured API keys (or Ollama, which is
        key-less) will be usable.
        """
        registry = cls.with_defaults()

        # Set API keys from config
        if config.openai_api_key:
            registry.set_api_key("openai", config.openai_api_key)
        if config.anthropic_api_key:
            registry.set_api_key("anthropic", config.anthropic_api_key)
        if config.google_api_key:
            registry.set_api_key("google", config.google_api_key)

        # Configure Ollama base URLs from swarm config
        if config.swarm.enabled and config.swarm.ollama_hosts:
            primary_host = config.swarm.ollama_hosts[0]
            registry.set_base_url("ollama", primary_host)

        logger.info(
            "Registry initialised: %d models, providers with keys: %s",
            len(registry._models),
            ", ".join(sorted(registry._api_keys)) or "(none)",
        )

        return registry

    async def register_ollama_models(
        self,
        host: str | None = None,
    ) -> list[ModelInfo]:
        """Discover and register all models from a running Ollama instance.

        Returns the list of newly registered ``ModelInfo`` objects.
        """
        from frizura.providers.ollama import discover_ollama_models

        effective_host = host or self._base_urls.get("ollama", "http://localhost:11434")
        discovered = await discover_ollama_models(host=effective_host)
        for info in discovered:
            self.register(info)
        if discovered:
            logger.info(
                "Discovered %d Ollama models: %s",
                len(discovered),
                ", ".join(m.model_id for m in discovered),
            )
        return discovered

    # ------------------------------------------------------------------
    # Available providers summary
    # ------------------------------------------------------------------

    def available_providers(self) -> list[str]:
        """Return list of provider names that have API keys configured or are local."""
        providers = set()
        for model in self._models.values():
            if model.is_local or model.provider in self._api_keys:
                providers.add(model.provider)
        return sorted(providers)

    def available_models(self) -> list[ModelInfo]:
        """Return only models whose provider has a configured API key (or is local)."""
        return [
            m
            for m in self._models.values()
            if m.is_local or m.provider in self._api_keys
        ]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _parse_model_spec(self, spec: str) -> tuple[str | None, str]:
        """Parse ``"provider:model"`` or plain ``"model"`` format.

        Returns ``(provider_name_or_None, model_id)``.
        """
        if ":" in spec:
            provider, _, model_id = spec.partition(":")
            return provider.strip(), model_id.strip()
        return None, spec.strip()

    def __repr__(self) -> str:
        return (
            f"<ModelRegistry models={len(self._models)} "
            f"providers={self.available_providers()}>"
        )

    def __len__(self) -> int:
        return len(self._models)

    def __contains__(self, model_id: str) -> bool:
        return model_id in self._models
