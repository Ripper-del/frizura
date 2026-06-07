"""Google Gemini provider adapter for the Frizura orchestrator.

Supports Gemini 2.0 Flash, Gemini 2.5 Pro, and Gemini 2.5 Flash via the
official ``google-genai`` SDK (the new unified SDK, *not* the legacy
``google-generativeai`` package).

Features:

* JSON mode via ``response_mime_type="application/json"``
* JSON schema-constrained output via ``response_schema``
* Proper token usage extraction from ``usage_metadata``
* Streaming with incremental ``StreamChunk`` emission
* Graceful degradation when the ``google-genai`` package is not installed
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
    MessageRole,
    StreamChunk,
    TokenUsage,
)
from frizura.models.providers import ModelInfo, CompletionConfig
from frizura.providers.base import LLMProvider

logger = logging.getLogger("frizura.providers.google")

# ---------------------------------------------------------------------------
# Lazy-import guard for the google-genai SDK
# ---------------------------------------------------------------------------

_google_available = True
_google_import_error: str | None = None

try:
    from google import genai as _genai
    from google.genai import types as _types
    from google.api_core.exceptions import (
        GoogleAPIError as _GoogleAPIError,
        PermissionDenied as _PermDenied,
        ResourceExhausted as _ResExhausted,
        Unauthenticated as _Unauth,
        DeadlineExceeded as _DeadlineExceeded,
    )
except ImportError as _exc:
    _google_available = False
    _google_import_error = (
        f"The 'google-genai' package is required for the Google provider. "
        f"Install it with:  pip install google-genai\n"
        f"Original error: {_exc}"
    )


def _require_sdk() -> None:
    if not _google_available:
        raise ImportError(_google_import_error)


# ---------------------------------------------------------------------------
# Message conversion helpers
# ---------------------------------------------------------------------------


def _messages_to_gemini(
    messages: list[Message],
) -> tuple[str | None, list[_types.Content]]:
    """Convert Frizura messages to Gemini's (system_instruction, contents).

    Gemini, like Anthropic, separates the system instruction from the
    conversation contents.
    """
    system_parts: list[str] = []
    contents: list[_types.Content] = []

    for msg in messages:
        if msg.role == MessageRole.SYSTEM:
            system_parts.append(msg.content)
            continue

        role = "user" if msg.role in (MessageRole.USER, MessageRole.TOOL) else "model"
        contents.append(
            _types.Content(
                role=role,
                parts=[_types.Part(text=msg.content)],
            )
        )

    system_text = "\n\n".join(system_parts) if system_parts else None

    # Gemini requires at least one content entry
    if not contents:
        contents.append(
            _types.Content(role="user", parts=[_types.Part(text="Hello")])
        )

    return system_text, contents


# ---------------------------------------------------------------------------
# Provider class
# ---------------------------------------------------------------------------


class GoogleProvider(LLMProvider):
    """Google Gemini chat adapter using google-genai SDK."""

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

        client_kwargs: dict[str, Any] = {}
        if api_key:
            client_kwargs["api_key"] = api_key

        self._client = _genai.Client(**client_kwargs)

    # ------------------------------------------------------------------
    # Core completion
    # ------------------------------------------------------------------

    async def _do_complete(
        self,
        messages: list[Message],
        config: CompletionConfig,
    ) -> LLMResponse:
        model_name, gen_config, system_instruction, contents = self._build_params(
            messages, config
        )
        response = await self._client.aio.models.generate_content(
            model=model_name,
            contents=contents,
            config=_types.GenerateContentConfig(
                **gen_config,
                **({"system_instruction": system_instruction} if system_instruction else {}),
            ),
        )
        return self._parse_response(response)

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    async def _do_stream(
        self,
        messages: list[Message],
        config: CompletionConfig,
    ) -> AsyncIterator[StreamChunk]:
        model_name, gen_config, system_instruction, contents = self._build_params(
            messages, config
        )

        stream = self._client.aio.models.generate_content_stream(
            model=model_name,
            contents=contents,
            config=_types.GenerateContentConfig(
                **gen_config,
                **({"system_instruction": system_instruction} if system_instruction else {}),
            ),
        )

        async for chunk in stream:
            text = ""
            if chunk.candidates:
                for part in chunk.candidates[0].content.parts:
                    if hasattr(part, "text") and part.text:
                        text += part.text

            # Check if this is the final chunk
            is_final = False
            finish_reason = None
            if chunk.candidates:
                fr = chunk.candidates[0].finish_reason
                if fr is not None:
                    is_final = True
                    # Map Gemini finish reasons
                    reason_map = {
                        "STOP": "stop",
                        "MAX_TOKENS": "length",
                        "SAFETY": "content_filter",
                    }
                    finish_reason = reason_map.get(str(fr), str(fr).lower())

            # Extract usage from the final chunk
            usage = None
            if is_final:
                usage = self._extract_usage(chunk)

            yield StreamChunk(
                content=text,
                is_final=is_final,
                finish_reason=finish_reason,
                usage=usage,
            )

    # ------------------------------------------------------------------
    # Healthcheck
    # ------------------------------------------------------------------

    async def _do_healthcheck(self) -> bool:
        """List models to verify connectivity and auth."""
        result = self._client.models.list(config={"page_size": 1})
        # Iterate to trigger the request
        for _ in result:
            return True
        return True

    # ------------------------------------------------------------------
    # Error mapping
    # ------------------------------------------------------------------

    def _map_error(self, exc: Exception) -> FrizuraError:
        if isinstance(exc, FrizuraError):
            return exc

        if not _google_available:
            return super()._map_error(exc)

        if isinstance(exc, (_Unauth, _PermDenied)):
            return AuthenticationError(
                provider=self.provider_name,
                model=self.model_id,
                message="Invalid or missing Google API key",
            )
        if isinstance(exc, _ResExhausted):
            return RateLimitError(
                provider=self.provider_name,
                model=self.model_id,
                retry_after=None,
            )
        if isinstance(exc, _DeadlineExceeded):
            return ProviderError(
                provider=self.provider_name,
                model=self.model_id,
                message=f"Request timed out after {self._timeout}s",
            )
        if isinstance(exc, _GoogleAPIError):
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
    ) -> tuple[str, dict[str, Any], str | None, list[Any]]:
        """Build parameters for generate_content.

        Returns (model_name, gen_config_dict, system_instruction, contents).
        """
        system_instruction, contents = _messages_to_gemini(messages)

        model_name = config.model or self.model_id

        gen_config: dict[str, Any] = {}

        if config.temperature is not None:
            gen_config["temperature"] = config.temperature
        if config.max_tokens:
            gen_config["max_output_tokens"] = config.max_tokens
        if config.top_p is not None:
            gen_config["top_p"] = config.top_p
        if config.stop:
            gen_config["stop_sequences"] = config.stop
        if config.seed is not None:
            gen_config["seed"] = config.seed

        # JSON mode / structured output
        if config.json_schema:
            gen_config["response_mime_type"] = "application/json"
            gen_config["response_schema"] = config.json_schema
        elif config.json_mode:
            gen_config["response_mime_type"] = "application/json"

        # Extra provider-specific params
        if config.extra:
            gen_config.update(config.extra)

        return model_name, gen_config, system_instruction, contents

    def _parse_response(self, response: Any) -> LLMResponse:
        """Parse a Gemini ``GenerateContentResponse`` into ``LLMResponse``."""
        # Extract text
        text = ""
        if response.candidates:
            for part in response.candidates[0].content.parts:
                if hasattr(part, "text") and part.text:
                    text += part.text

        # Finish reason
        finish_reason = "stop"
        if response.candidates:
            fr = response.candidates[0].finish_reason
            if fr is not None:
                reason_map = {
                    "STOP": "stop",
                    "MAX_TOKENS": "length",
                    "SAFETY": "content_filter",
                }
                finish_reason = reason_map.get(str(fr), str(fr).lower())

        # Usage
        usage = self._extract_usage(response) or TokenUsage()

        return LLMResponse(
            content=text,
            model=self.model_id,
            provider=self.provider_name,
            usage=usage,
            finish_reason=finish_reason,
            raw={},
        )

    @staticmethod
    def _extract_usage(response: Any) -> TokenUsage | None:
        """Extract token usage from ``usage_metadata``."""
        meta = getattr(response, "usage_metadata", None)
        if meta is None:
            return None
        input_tokens = getattr(meta, "prompt_token_count", 0) or 0
        output_tokens = getattr(meta, "candidates_token_count", 0) or 0
        cached = getattr(meta, "cached_content_token_count", 0) or 0
        total = getattr(meta, "total_token_count", 0) or (input_tokens + output_tokens)
        return TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total,
            cached_tokens=cached,
        )
