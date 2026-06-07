"""Abstract base class for LLM provider adapters.

Every provider (OpenAI, Anthropic, Google, Ollama) inherits from
``LLMProvider`` and implements provider-specific completion / streaming
logic.  The base class handles cross-cutting concerns:

* Latency measurement (wall-clock ``time.perf_counter``)
* Retries with exponential back-off & jitter
* Provider-error → Frizura-error mapping
* Cost estimation via the associated ``ModelInfo``
"""

from __future__ import annotations

import abc
import asyncio
import logging
import random
import time
from collections.abc import AsyncIterator
from decimal import Decimal
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
)
from frizura.models.providers import ModelInfo, CompletionConfig

logger = logging.getLogger("frizura.providers")

# Default retry parameters
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_BASE_DELAY = 1.0  # seconds
_DEFAULT_MAX_DELAY = 30.0  # seconds
_JITTER_FACTOR = 0.5  # ±50 %


class LLMProvider(abc.ABC):
    """Unified interface every LLM provider adapter must implement.

    Subclasses are expected to override:
    * ``_do_complete`` — the raw provider call (no retry / timing).
    * ``_do_stream``  — the raw streaming call.
    * ``_do_healthcheck`` — a lightweight connectivity probe.
    * ``_map_error``   — translate provider SDK errors into Frizura ones.

    The public ``complete()`` wrapper takes care of retries, latency
    measurement, and error normalisation.
    """

    def __init__(
        self,
        model_info: ModelInfo,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self._model_info = model_info
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = timeout
        self._max_retries = max_retries
        self._extra = extra or {}

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def model_info(self) -> ModelInfo:
        """Return the ``ModelInfo`` for the model this provider serves."""
        return self._model_info

    @property
    def provider_name(self) -> str:
        return self._model_info.provider

    @property
    def model_id(self) -> str:
        return self._model_info.model_id

    # ------------------------------------------------------------------
    # Cost estimation (delegated to ModelInfo)
    # ------------------------------------------------------------------

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> Decimal:
        """Estimate the cost in USD for the given token counts."""
        return self._model_info.estimate_cost(input_tokens, output_tokens)

    # ------------------------------------------------------------------
    # Public async interface
    # ------------------------------------------------------------------

    async def complete(
        self,
        messages: list[Message],
        config: CompletionConfig | None = None,
    ) -> LLMResponse:
        """Send a completion request with automatic retries & latency tracking.

        Raises
        ------
        AuthenticationError
            API key is missing or invalid.
        RateLimitError
            Quota exceeded – contains optional ``retry_after`` hint.
        ProviderError
            Any other provider-side failure.
        """
        config = config or CompletionConfig()
        last_error: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            try:
                t0 = time.perf_counter()
                response = await self._do_complete(messages, config)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

                # Enrich with timing + provider metadata
                response.latency_ms = elapsed_ms
                response.provider = self.provider_name
                if not response.model:
                    response.model = self.model_id

                logger.debug(
                    "LLM complete [%s:%s] %d→%d tokens  %.0f ms  attempt=%d",
                    self.provider_name,
                    self.model_id,
                    response.usage.input_tokens,
                    response.usage.output_tokens,
                    elapsed_ms,
                    attempt,
                )
                return response

            except (AuthenticationError, FrizuraError) as exc:
                # Auth errors are never retried
                if isinstance(exc, AuthenticationError):
                    raise
                last_error = exc
                raise

            except Exception as exc:  # noqa: BLE001
                mapped = self._map_error(exc)
                last_error = mapped

                if isinstance(mapped, AuthenticationError):
                    raise mapped from exc

                if attempt >= self._max_retries:
                    logger.warning(
                        "LLM complete failed after %d attempts: %s",
                        attempt,
                        mapped,
                    )
                    raise mapped from exc

                delay = self._backoff_delay(attempt)

                # For rate-limits, honour ``retry_after`` if provided
                if isinstance(mapped, RateLimitError) and mapped.retry_after:
                    delay = max(delay, mapped.retry_after)

                logger.info(
                    "LLM complete attempt %d/%d failed (%s), retrying in %.1fs",
                    attempt,
                    self._max_retries,
                    type(mapped).__name__,
                    delay,
                )
                await asyncio.sleep(delay)

        # Should be unreachable, but just in case:
        assert last_error is not None  # noqa: S101
        raise last_error

    async def stream(
        self,
        messages: list[Message],
        config: CompletionConfig | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a completion response chunk by chunk.

        Streaming does **not** retry automatically — the caller should
        decide whether to restart the stream on failure.
        """
        config = config or CompletionConfig()
        try:
            async for chunk in self._do_stream(messages, config):
                yield chunk
        except Exception as exc:  # noqa: BLE001
            raise self._map_error(exc) from exc

    async def healthcheck(self) -> bool:
        """Return ``True`` if the provider is reachable and authenticated."""
        try:
            return await self._do_healthcheck()
        except Exception:  # noqa: BLE001
            return False

    # ------------------------------------------------------------------
    # Abstract methods — subclasses MUST implement these
    # ------------------------------------------------------------------

    @abc.abstractmethod
    async def _do_complete(
        self,
        messages: list[Message],
        config: CompletionConfig,
    ) -> LLMResponse:
        """Perform the raw completion call (no retry wrapper)."""
        ...

    @abc.abstractmethod
    async def _do_stream(
        self,
        messages: list[Message],
        config: CompletionConfig,
    ) -> AsyncIterator[StreamChunk]:
        """Perform the raw streaming call."""
        ...
        # ``yield`` is needed so Python treats this as an async generator
        yield  # type: ignore[misc]  # pragma: no cover

    @abc.abstractmethod
    async def _do_healthcheck(self) -> bool:
        """Lightweight connectivity / auth check."""
        ...

    # ------------------------------------------------------------------
    # Error mapping — subclasses SHOULD override for provider SDK errors
    # ------------------------------------------------------------------

    def _map_error(self, exc: Exception) -> FrizuraError:
        """Map a provider SDK exception to the Frizura error hierarchy.

        The default implementation wraps anything unknown as a generic
        ``ProviderError``.  Subclasses should intercept SDK-specific
        exceptions (e.g. ``openai.AuthenticationError``) *before*
        calling ``super()``.
        """
        if isinstance(exc, FrizuraError):
            return exc
        return ProviderError(
            provider=self.provider_name,
            model=self.model_id,
            message=str(exc),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _backoff_delay(attempt: int) -> float:
        """Compute exponential back-off with jitter.

        delay = base * 2^(attempt-1) ± jitter
        """
        base = _DEFAULT_BASE_DELAY * (2 ** (attempt - 1))
        base = min(base, _DEFAULT_MAX_DELAY)
        jitter = base * _JITTER_FACTOR * (2 * random.random() - 1)  # noqa: S311
        return max(0.1, base + jitter)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.provider_name}:{self.model_id}>"
