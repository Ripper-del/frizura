"""Ollama local-model provider adapter for the Frizura orchestrator.

Connects to a running Ollama instance via the ``ollama`` Python SDK to
run LLMs locally.  All costs are zero and ``is_local=True``.

Features:

* Auto-discovery of locally available models via ``ollama.list()``
* JSON mode via ``format="json"``
* Streaming with ``StreamChunk`` emission
* Dynamic ``ModelInfo`` creation for any model Ollama is serving
* Graceful handling when Ollama is not running or SDK not installed
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from decimal import Decimal
from typing import Any

from frizura.core.exceptions import (
    FrizuraError,
    ProviderError,
)
from frizura.models.execution import (
    LLMResponse,
    Message,
    MessageRole,
    StreamChunk,
    TokenUsage,
)
from frizura.models.providers import ModelInfo, CompletionConfig
from frizura.providers.base import LLMProvider

logger = logging.getLogger("frizura.providers.ollama")

# ---------------------------------------------------------------------------
# Lazy-import guard for the ollama SDK
# ---------------------------------------------------------------------------

_ollama_available = True
_ollama_import_error: str | None = None

try:
    import ollama as _ollama_sdk
    from ollama import AsyncClient as _AsyncClient
    from ollama import ResponseError as _ResponseError
except ImportError as _exc:
    _ollama_available = False
    _ollama_import_error = (
        f"The 'ollama' package is required for the Ollama provider. "
        f"Install it with:  pip install ollama\n"
        f"Original error: {_exc}"
    )


def _require_sdk() -> None:
    if not _ollama_available:
        raise ImportError(_ollama_import_error)


# ---------------------------------------------------------------------------
# Message conversion helpers
# ---------------------------------------------------------------------------


def _messages_to_ollama(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert Frizura ``Message`` objects to Ollama chat format.

    Ollama uses a simplified dict format identical to OpenAI's chat format.
    """
    out: list[dict[str, Any]] = []
    for msg in messages:
        m: dict[str, str] = {
            "role": msg.role.value,
            "content": msg.content,
        }
        out.append(m)
    return out


# ---------------------------------------------------------------------------
# Helper to build ModelInfo for a discovered local model
# ---------------------------------------------------------------------------


def make_ollama_model_info(
    model_name: str,
    *,
    context_window: int = 128_000,
    max_output_tokens: int = 4096,
) -> ModelInfo:
    """Create a ``ModelInfo`` for a locally-discovered Ollama model.

    All prices are zero and ``is_local=True``.
    """
    return ModelInfo(
        model_id=model_name,
        provider="ollama",
        display_name=f"Ollama: {model_name}",
        context_window=context_window,
        max_output_tokens=max_output_tokens,
        input_price_per_1m=Decimal("0"),
        output_price_per_1m=Decimal("0"),
        supports_json_mode=True,
        supports_tool_calling=False,  # varies by model; conservative default
        supports_vision=False,
        supports_streaming=True,
        is_local=True,
        tier="cheap",
    )


# ---------------------------------------------------------------------------
# Provider class
# ---------------------------------------------------------------------------


class OllamaProvider(LLMProvider):
    """Ollama local-model adapter."""

    def __init__(
        self,
        model_info: ModelInfo,
        *,
        api_key: str | None = None,  # unused for Ollama, kept for interface compat
        base_url: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 2,
        extra: dict[str, Any] | None = None,
    ) -> None:
        _require_sdk()
        super().__init__(
            model_info,
            api_key=api_key,
            base_url=base_url or "http://localhost:11434",
            timeout=timeout,
            max_retries=max_retries,
            extra=extra,
        )

        self._client = _AsyncClient(host=self._base_url, timeout=timeout)

    # ------------------------------------------------------------------
    # Core completion
    # ------------------------------------------------------------------

    async def _do_complete(
        self,
        messages: list[Message],
        config: CompletionConfig,
    ) -> LLMResponse:
        params = self._build_params(messages, config, stream=False)
        response = await self._client.chat(**params)
        return self._parse_response(response)

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    async def _do_stream(
        self,
        messages: list[Message],
        config: CompletionConfig,
    ) -> AsyncIterator[StreamChunk]:
        params = self._build_params(messages, config, stream=True)
        stream = await self._client.chat(**params)

        async for chunk in stream:
            message = chunk.get("message", {})
            content = message.get("content", "")
            done = chunk.get("done", False)

            sc = StreamChunk(content=content, is_final=done)

            if done:
                # Final chunk contains usage stats
                sc.finish_reason = "stop"
                sc.usage = TokenUsage(
                    input_tokens=chunk.get("prompt_eval_count", 0),
                    output_tokens=chunk.get("eval_count", 0),
                    total_tokens=(
                        chunk.get("prompt_eval_count", 0)
                        + chunk.get("eval_count", 0)
                    ),
                )

            yield sc

    # ------------------------------------------------------------------
    # Healthcheck
    # ------------------------------------------------------------------

    async def _do_healthcheck(self) -> bool:
        """Check if Ollama is reachable by listing models."""
        try:
            result = await self._client.list()
            return True
        except Exception:  # noqa: BLE001
            return False

    # ------------------------------------------------------------------
    # Error mapping
    # ------------------------------------------------------------------

    def _map_error(self, exc: Exception) -> FrizuraError:
        if isinstance(exc, FrizuraError):
            return exc
        if _ollama_available and isinstance(exc, _ResponseError):
            msg = str(exc)
            if "not found" in msg.lower():
                return ProviderError(
                    provider=self.provider_name,
                    model=self.model_id,
                    message=f"Model '{self.model_id}' not found. Run: ollama pull {self.model_id}",
                )
            return ProviderError(
                provider=self.provider_name,
                model=self.model_id,
                message=msg,
            )
        if isinstance(exc, ConnectionError | OSError):
            return ProviderError(
                provider=self.provider_name,
                model=self.model_id,
                message=(
                    f"Cannot connect to Ollama at {self._base_url}. "
                    f"Make sure Ollama is running: ollama serve"
                ),
            )
        return super()._map_error(exc)

    # ------------------------------------------------------------------
    # Auto-discovery
    # ------------------------------------------------------------------

    async def list_local_models(self) -> list[str]:
        """Return the names of all models available in the local Ollama instance."""
        try:
            result = await self._client.list()
            models = result.get("models", [])
            return [m.get("name", m.get("model", "")) for m in models if m]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to list Ollama models: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_params(
        self,
        messages: list[Message],
        config: CompletionConfig,
        *,
        stream: bool,
    ) -> dict[str, Any]:
        """Build keyword arguments for ``ollama.chat()``."""
        params: dict[str, Any] = {
            "model": config.model or self.model_id,
            "messages": _messages_to_ollama(messages),
            "stream": stream,
        }

        # Ollama options
        options: dict[str, Any] = {}
        if config.temperature is not None:
            options["temperature"] = config.temperature
        if config.max_tokens:
            options["num_predict"] = config.max_tokens
        if config.top_p is not None:
            options["top_p"] = config.top_p
        if config.stop:
            options["stop"] = config.stop
        if config.seed is not None:
            options["seed"] = config.seed

        # Extra provider-specific params
        if config.extra:
            options.update(config.extra)

        if options:
            params["options"] = options

        # JSON mode
        if config.json_mode or config.json_schema:
            params["format"] = "json"

        return params

    def _parse_response(self, response: dict[str, Any]) -> LLMResponse:
        """Parse an Ollama response dict into ``LLMResponse``."""
        message = response.get("message", {})
        content = message.get("content", "")

        input_tokens = response.get("prompt_eval_count", 0)
        output_tokens = response.get("eval_count", 0)

        usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )

        return LLMResponse(
            content=content,
            model=response.get("model", self.model_id),
            provider=self.provider_name,
            usage=usage,
            finish_reason="stop" if response.get("done", True) else "length",
            raw={
                k: v
                for k, v in response.items()
                if k not in ("message", "done")
            },
        )


# ---------------------------------------------------------------------------
# Convenience: auto-discover and create providers
# ---------------------------------------------------------------------------


async def discover_ollama_models(
    host: str = "http://localhost:11434",
    timeout: float = 10.0,
) -> list[ModelInfo]:
    """Probe a running Ollama instance and return ``ModelInfo`` for each model.

    Returns an empty list if Ollama is not reachable.
    """
    _require_sdk()
    try:
        client = _AsyncClient(host=host, timeout=timeout)
        result = await client.list()
        models_data = result.get("models", [])
        infos: list[ModelInfo] = []
        for m in models_data:
            name = m.get("name", m.get("model", ""))
            if not name:
                continue
            # Try to extract context window from model details
            details = m.get("details", {})
            ctx = 128_000  # safe default
            param_size = details.get("parameter_size", "")

            infos.append(make_ollama_model_info(name, context_window=ctx))
        return infos
    except Exception as exc:  # noqa: BLE001
        logger.debug("Ollama discovery failed at %s: %s", host, exc)
        return []
