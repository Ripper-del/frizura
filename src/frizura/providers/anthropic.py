"""Anthropic provider adapter for the Frizura orchestrator.

Supports Claude 3.5 Sonnet, Claude 3.5 Haiku, and Claude 4 Opus via the
official ``anthropic`` async SDK.  Handles Anthropic's unique message
format (system prompt as a top-level parameter rather than a message).

Features:

* Tool / function calling (Anthropic native format)
* JSON mode via tool_use workaround (Anthropic doesn't have a native
  json_mode flag — we set the system prompt to request JSON)
* Streaming with ``StreamChunk`` emission
* Graceful degradation when the ``anthropic`` package is not installed
"""

from __future__ import annotations

import json
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
    MessageRole,
    StreamChunk,
    TokenUsage,
)
from frizura.models.providers import ModelInfo, CompletionConfig
from frizura.providers.base import LLMProvider

logger = logging.getLogger("frizura.providers.anthropic")

# ---------------------------------------------------------------------------
# Lazy-import guard
# ---------------------------------------------------------------------------

_anthropic_available = True
_anthropic_import_error: str | None = None

try:
    import anthropic as _anthropic_sdk
    from anthropic import (
        APIConnectionError as _ConnErr,
        APIStatusError as _StatusErr,
        APITimeoutError as _TimeoutErr,
        AuthenticationError as _AnthAuthErr,
        RateLimitError as _AnthRateErr,
    )
except ImportError as _exc:
    _anthropic_available = False
    _anthropic_import_error = (
        f"The 'anthropic' package is required for the Anthropic provider. "
        f"Install it with:  pip install anthropic\n"
        f"Original error: {_exc}"
    )


def _require_sdk() -> None:
    if not _anthropic_available:
        raise ImportError(_anthropic_import_error)


# ---------------------------------------------------------------------------
# Message conversion helpers
# ---------------------------------------------------------------------------

_JSON_MODE_SYSTEM_SUFFIX = (
    "\n\nIMPORTANT: You MUST respond with valid JSON only. "
    "Do not include any text outside the JSON object."
)


def _split_system_and_messages(
    messages: list[Message],
    json_mode: bool = False,
) -> tuple[str | None, list[dict[str, Any]]]:
    """Split a unified message list into Anthropic's (system, messages) form.

    Anthropic requires the system prompt to be passed as a top-level
    ``system`` parameter, not as a message.
    """
    system_parts: list[str] = []
    converted: list[dict[str, Any]] = []

    for msg in messages:
        if msg.role == MessageRole.SYSTEM:
            system_parts.append(msg.content)
            continue

        role = "user" if msg.role == MessageRole.USER else "assistant"

        # Tool results are sent as role=user with a special content block
        if msg.role == MessageRole.TOOL:
            converted.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.tool_call_id or "",
                            "content": msg.content,
                        }
                    ],
                }
            )
            continue

        m: dict[str, Any] = {"role": role, "content": msg.content}
        converted.append(m)

    system_text = "\n\n".join(system_parts) if system_parts else None

    if json_mode and system_text:
        system_text += _JSON_MODE_SYSTEM_SUFFIX
    elif json_mode:
        system_text = _JSON_MODE_SYSTEM_SUFFIX.strip()

    # Anthropic requires that the first message is always role=user.
    # If there are no messages at all or the first is assistant, prepend.
    if not converted:
        converted.append({"role": "user", "content": "Hello"})
    elif converted[0]["role"] != "user":
        converted.insert(0, {"role": "user", "content": "Please continue."})

    return system_text, converted


def _convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI-style tool definitions to Anthropic format."""
    anthropic_tools: list[dict[str, Any]] = []
    for tool in tools:
        if tool.get("type") == "function":
            fn = tool["function"]
            anthropic_tools.append(
                {
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
                }
            )
        else:
            # Pass through if already in Anthropic format
            anthropic_tools.append(tool)
    return anthropic_tools


def _extract_tool_calls_anthropic(content_blocks: list[Any]) -> list[dict[str, Any]] | None:
    """Extract tool_use blocks from Anthropic response content."""
    calls: list[dict[str, Any]] = []
    for block in content_blocks:
        block_type = block.type if hasattr(block, "type") else block.get("type")
        if block_type == "tool_use":
            block_id = block.id if hasattr(block, "id") else block.get("id", "")
            block_name = block.name if hasattr(block, "name") else block.get("name", "")
            block_input = block.input if hasattr(block, "input") else block.get("input", {})
            calls.append(
                {
                    "id": block_id,
                    "type": "function",
                    "function": {
                        "name": block_name,
                        "arguments": json.dumps(block_input) if isinstance(block_input, dict) else str(block_input),
                    },
                }
            )
    return calls or None


# ---------------------------------------------------------------------------
# Provider class
# ---------------------------------------------------------------------------


class AnthropicProvider(LLMProvider):
    """Anthropic Claude chat adapter."""

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
            "max_retries": 0,  # retries handled by base class
        }
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url

        self._client = _anthropic_sdk.AsyncAnthropic(**client_kwargs)

    # ------------------------------------------------------------------
    # Core completion
    # ------------------------------------------------------------------

    async def _do_complete(
        self,
        messages: list[Message],
        config: CompletionConfig,
    ) -> LLMResponse:
        params = self._build_params(messages, config)
        response = await self._client.messages.create(**params)
        return self._parse_response(response)

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    async def _do_stream(
        self,
        messages: list[Message],
        config: CompletionConfig,
    ) -> AsyncIterator[StreamChunk]:
        params = self._build_params(messages, config)
        params["stream"] = True

        async with self._client.messages.stream(**{k: v for k, v in params.items() if k != "stream"}) as stream:
            async for event in stream:
                event_type = getattr(event, "type", None)

                if event_type == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    text = ""
                    if delta and hasattr(delta, "text"):
                        text = delta.text or ""
                    if text:
                        yield StreamChunk(content=text)

                elif event_type == "message_delta":
                    # Final event with stop_reason and usage
                    delta = getattr(event, "delta", None)
                    finish_reason = getattr(delta, "stop_reason", "end_turn") if delta else "end_turn"
                    usage_data = getattr(event, "usage", None)
                    usage = None
                    if usage_data:
                        usage = TokenUsage(
                            output_tokens=getattr(usage_data, "output_tokens", 0),
                        )
                    yield StreamChunk(
                        is_final=True,
                        finish_reason=finish_reason,
                        usage=usage,
                    )

                elif event_type == "message_start":
                    # Contains input token count in message.usage
                    msg = getattr(event, "message", None)
                    if msg:
                        u = getattr(msg, "usage", None)
                        if u:
                            input_tokens = getattr(u, "input_tokens", 0)
                            if input_tokens:
                                logger.debug("Stream input tokens: %d", input_tokens)

    # ------------------------------------------------------------------
    # Healthcheck
    # ------------------------------------------------------------------

    async def _do_healthcheck(self) -> bool:
        """Send a tiny request to verify connectivity and auth."""
        response = await self._client.messages.create(
            model=self.model_id,
            max_tokens=1,
            messages=[{"role": "user", "content": "Hi"}],
        )
        return bool(response)

    # ------------------------------------------------------------------
    # Error mapping
    # ------------------------------------------------------------------

    def _map_error(self, exc: Exception) -> FrizuraError:
        if isinstance(exc, FrizuraError):
            return exc
        if isinstance(exc, _AnthAuthErr):
            return AuthenticationError(
                provider=self.provider_name,
                model=self.model_id,
                message="Invalid or missing Anthropic API key",
            )
        if isinstance(exc, _AnthRateErr):
            retry_after: float | None = None
            resp = getattr(exc, "response", None)
            if resp is not None:
                ra = getattr(resp, "headers", {}).get("retry-after")
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
    ) -> dict[str, Any]:
        """Build ``create()`` keyword arguments for the Anthropic API."""
        system_text, converted = _split_system_and_messages(
            messages, json_mode=config.json_mode
        )

        params: dict[str, Any] = {
            "model": config.model or self.model_id,
            "messages": converted,
            "max_tokens": config.max_tokens or self._model_info.max_output_tokens,
        }

        if system_text:
            params["system"] = system_text

        if config.temperature is not None:
            params["temperature"] = config.temperature
        if config.top_p is not None:
            params["top_p"] = config.top_p
        if config.stop:
            params["stop_sequences"] = config.stop

        # Tool calling
        if config.tools:
            params["tools"] = _convert_tools(config.tools)
            if config.tool_choice == "auto":
                params["tool_choice"] = {"type": "auto"}
            elif config.tool_choice == "none":
                # Anthropic doesn't have "none" — just omit tools
                del params["tools"]
            elif config.tool_choice:
                params["tool_choice"] = {"type": "tool", "name": config.tool_choice}

        # Extra provider-specific params
        if config.extra:
            params.update(config.extra)

        return params

    def _parse_response(self, response: Any) -> LLMResponse:
        """Parse an Anthropic ``Message`` response into ``LLMResponse``."""
        # Extract text content
        text_parts: list[str] = []
        for block in response.content:
            if hasattr(block, "text"):
                text_parts.append(block.text)
        content = "".join(text_parts)

        # Extract tool calls
        tool_calls = _extract_tool_calls_anthropic(response.content)

        # Token usage
        usage = TokenUsage()
        if response.usage:
            cache_read = getattr(response.usage, "cache_read_input_tokens", 0) or 0
            cache_create = getattr(response.usage, "cache_creation_input_tokens", 0) or 0
            input_tokens = response.usage.input_tokens or 0
            output_tokens = response.usage.output_tokens or 0
            usage = TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                cached_tokens=cache_read + cache_create,
            )

        # Map Anthropic stop reasons to standard ones
        finish_reason = response.stop_reason or "end_turn"
        finish_map = {"end_turn": "stop", "tool_use": "tool_calls", "max_tokens": "length"}
        finish_reason = finish_map.get(finish_reason, finish_reason)

        return LLMResponse(
            content=content,
            model=response.model or self.model_id,
            provider=self.provider_name,
            usage=usage,
            finish_reason=finish_reason,
            tool_calls=tool_calls,
            raw={"id": response.id, "type": response.type},
        )
