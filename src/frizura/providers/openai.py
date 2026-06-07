"""OpenAI provider adapter for the Frizura orchestrator.

Supports GPT-4o, GPT-4o-mini, GPT-4-turbo, o1, and o3-mini via the
official ``openai`` async SDK.  Features:

* JSON mode (``response_format={"type": "json_object"}``)
* Structured output with JSON schema (``response_format={"type": "json_schema", …}``)
* Tool / function calling
* Streaming with incremental ``StreamChunk`` emission
* Graceful degradation when the ``openai`` package is not installed
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from frizura.core.exceptions import (
    AuthenticationError,
    FrizuraError,
    ProviderError,
    RateLimitError,
)
from frizura.models.execution import (
    LLMResponse,
    Message,
    StreamChunk,
    TokenUsage,
)
from frizura.models.providers import ModelInfo, CompletionConfig
from frizura.providers.base import LLMProvider

logger = logging.getLogger("frizura.providers.openai")

# ---------------------------------------------------------------------------
# Lazy-import guard for the openai SDK
# ---------------------------------------------------------------------------

_openai_available = True
_openai_import_error: str | None = None

try:
    import openai as _openai_sdk
    from openai import (
        APIConnectionError as _ConnErr,
        APIStatusError as _StatusErr,
        APITimeoutError as _TimeoutErr,
        AuthenticationError as _OAIAuthErr,
        RateLimitError as _OAIRateErr,
    )
except ImportError as _exc:
    _openai_available = False
    _openai_import_error = (
        f"The 'openai' package is required for the OpenAI provider. "
        f"Install it with:  pip install openai\n"
        f"Original error: {_exc}"
    )


def _require_sdk() -> None:
    if not _openai_available:
        raise ImportError(_openai_import_error)


# ---------------------------------------------------------------------------
# Message conversion helpers
# ---------------------------------------------------------------------------


def _messages_to_openai(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert Frizura ``Message`` objects to the OpenAI chat format."""
    out: list[dict[str, Any]] = []
    for msg in messages:
        m: dict[str, Any] = {"role": msg.role.value, "content": msg.content}
        if msg.name:
            m["name"] = msg.name
        if msg.tool_call_id:
            m["tool_call_id"] = msg.tool_call_id
        if msg.tool_calls:
            m["tool_calls"] = msg.tool_calls
        out.append(m)
    return out


def _extract_tool_calls(choice: Any) -> list[dict[str, Any]] | None:
    """Extract tool calls from an OpenAI chat completion choice."""
    tc = getattr(choice.message, "tool_calls", None)
    if not tc:
        return None
    result: list[dict[str, Any]] = []
    for call in tc:
        result.append(
            {
                "id": call.id,
                "type": call.type,
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
        )
    return result


# ---------------------------------------------------------------------------
# Provider class
# ---------------------------------------------------------------------------


class OpenAIProvider(LLMProvider):
    """OpenAI chat-completion adapter."""

    def __init__(
        self,
        model_info: ModelInfo,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        extra: dict[str, Any] | None = None,
    ) -> None:
        _require_sdk()
        super().__init__(
            model_info,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            extra=extra,
        )

        client_kwargs: dict[str, Any] = {
            "timeout": timeout,
            "max_retries": 0,  # we do retries ourselves in the base class
        }
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url

        self._client = _openai_sdk.AsyncOpenAI(**client_kwargs)

    # ------------------------------------------------------------------
    # Core completion
    # ------------------------------------------------------------------

    async def _do_complete(
        self,
        messages: list[Message],
        config: CompletionConfig,
    ) -> LLMResponse:
        params = self._build_params(messages, config, stream=False)
        response = await self._client.chat.completions.create(**params)
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
        # Request usage in the final chunk (supported by OpenAI SDK)
        params["stream_options"] = {"include_usage": True}
        stream = await self._client.chat.completions.create(**params)

        async for chunk in stream:
            if not chunk.choices:
                # Final chunk with usage only
                usage = self._extract_stream_usage(chunk)
                if usage:
                    yield StreamChunk(is_final=True, usage=usage)
                continue

            delta = chunk.choices[0].delta
            finish = chunk.choices[0].finish_reason

            content = delta.content or ""
            is_final = finish is not None

            sc = StreamChunk(
                content=content,
                is_final=is_final,
                finish_reason=finish,
            )

            if is_final:
                usage = self._extract_stream_usage(chunk)
                if usage:
                    sc.usage = usage

            yield sc

    # ------------------------------------------------------------------
    # Healthcheck
    # ------------------------------------------------------------------

    async def _do_healthcheck(self) -> bool:
        """List models to verify connectivity + auth."""
        result = await self._client.models.list()
        # If we get here without error the connection is good
        return bool(result)

    # ------------------------------------------------------------------
    # Error mapping
    # ------------------------------------------------------------------

    def _map_error(self, exc: Exception) -> FrizuraError:
        if isinstance(exc, FrizuraError):
            return exc
        if isinstance(exc, _OAIAuthErr):
            return AuthenticationError(
                provider=self.provider_name,
                model=self.model_id,
                message="Invalid or missing OpenAI API key",
            )
        if isinstance(exc, _OAIRateErr):
            retry_after: float | None = None
            headers = getattr(exc, "response", None)
            if headers is not None:
                ra = getattr(headers, "headers", {}).get("retry-after")
                if ra:
                    try:
                        retry_after = float(ra)
                    except (ValueError, TypeError):
                        pass
            return RateLimitError(
                provider=self.provider_name,
                model=self.model_id,
                retry_after=retry_after,
            )
        if isinstance(exc, _TimeoutErr):
            return ProviderError(
                provider=self.provider_name,
                model=self.model_id,
                message=f"Request timed out after {self._timeout}s",
            )
        if isinstance(exc, (_ConnErr, _StatusErr)):
            return ProviderError(
                provider=self.provider_name,
                model=self.model_id,
                message=str(exc),
            )
        return super()._map_error(exc)

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
        """Build the ``create()`` keyword arguments."""
        params: dict[str, Any] = {
            "model": config.model or self.model_id,
            "messages": _messages_to_openai(messages),
            "stream": stream,
        }

        # Temperature — note: o1/o3-mini don't support temperature
        model_name = (config.model or self.model_id).lower()
        is_reasoning = any(t in model_name for t in ("o1", "o3"))
        if not is_reasoning:
            params["temperature"] = config.temperature

        if config.max_tokens:
            # o-series uses max_completion_tokens
            key = "max_completion_tokens" if is_reasoning else "max_tokens"
            params[key] = config.max_tokens

        if config.top_p is not None and not is_reasoning:
            params["top_p"] = config.top_p
        if config.stop:
            params["stop"] = config.stop
        if config.seed is not None:
            params["seed"] = config.seed

        # JSON mode / structured output
        if config.json_schema:
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": config.json_schema.get("title", "response"),
                    "strict": True,
                    "schema": config.json_schema,
                },
            }
        elif config.json_mode:
            params["response_format"] = {"type": "json_object"}

        # Tools / function calling
        if config.tools:
            params["tools"] = config.tools
            if config.tool_choice:
                params["tool_choice"] = config.tool_choice

        # Extra provider-specific params
        if config.extra:
            params.update(config.extra)

        return params

    def _parse_response(self, response: Any) -> LLMResponse:
        """Parse a non-streaming ``ChatCompletion`` into ``LLMResponse``."""
        choice = response.choices[0]
        content = choice.message.content or ""
        tool_calls = _extract_tool_calls(choice)

        usage = TokenUsage()
        if response.usage:
            cached = 0
            if hasattr(response.usage, "prompt_tokens_details"):
                details = response.usage.prompt_tokens_details
                if details and hasattr(details, "cached_tokens"):
                    cached = details.cached_tokens or 0

            usage = TokenUsage(
                input_tokens=response.usage.prompt_tokens or 0,
                output_tokens=response.usage.completion_tokens or 0,
                total_tokens=response.usage.total_tokens or 0,
                cached_tokens=cached,
            )

        return LLMResponse(
            content=content,
            model=response.model or self.model_id,
            provider=self.provider_name,
            usage=usage,
            finish_reason=choice.finish_reason or "stop",
            tool_calls=tool_calls,
            raw={"id": response.id, "system_fingerprint": getattr(response, "system_fingerprint", None)},
        )

    @staticmethod
    def _extract_stream_usage(chunk: Any) -> TokenUsage | None:
        """Try to pull ``usage`` from a streaming chunk."""
        u = getattr(chunk, "usage", None)
        if u is None:
            return None
        return TokenUsage(
            input_tokens=u.prompt_tokens or 0,
            output_tokens=u.completion_tokens or 0,
            total_tokens=u.total_tokens or 0,
        )
