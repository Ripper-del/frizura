"""Self-healing loop for schema validation failures.

When LLM output fails Pydantic validation, the healer sends focused repair
prompts back to the LLM (or escalates to a more powerful model) to fix the
output. Tracks all healing attempts as events for observability.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

from frizura.core.events import Event, EventType
from frizura.core.exceptions import SchemaHealingFailed, BudgetExhaustedError
from frizura.models.budget import BudgetConstraint
from frizura.models.execution import (
    LLMResponse,
    Message,
)
from frizura.models.providers import CompletionConfig
from frizura.schema.strategies import (
    HealingStrategy,
    extract_partial,
    get_healing_prompt,
    get_simplified_healing_prompt,
)
from frizura.schema.validator import SchemaValidator, ValidationResult

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for an LLM provider that can handle completion requests.

    Any object implementing ``complete`` can be used with ``SchemaHealer``.
    """

    async def complete(
        self,
        messages: list[Message],
        config: CompletionConfig,
    ) -> LLMResponse: ...


@runtime_checkable
class EventEmitter(Protocol):
    """Protocol for emitting events during healing."""

    async def emit(self, event: Event) -> None: ...


class _NullEmitter:
    """No-op event emitter for when events aren't being tracked."""

    async def emit(self, event: Event) -> None:
        pass


# Model escalation tiers — when healing fails on a cheap model, try a better one
_ESCALATION_MAP: dict[str, str] = {
    # OpenAI
    "gpt-4o-mini": "gpt-4o",
    "gpt-4o": "gpt-4o",
    "gpt-3.5-turbo": "gpt-4o",
    # Anthropic
    "claude-3-5-haiku-latest": "claude-sonnet-4-20250514",
    "claude-sonnet-4-20250514": "claude-sonnet-4-20250514",
    "claude-3-5-haiku-20241022": "claude-sonnet-4-20250514",
    # Google
    "gemini-2.0-flash": "gemini-2.5-pro-preview-06-05",
    "gemini-2.5-flash-preview-05-20": "gemini-2.5-pro-preview-06-05",
    # Ollama (stay local)
    "llama3.2": "llama3.1:70b",
    "mistral": "mixtral",
}


class SchemaHealer:
    """Self-healing loop that repairs invalid LLM output.

    Coordinates multiple repair attempts using different strategies:
    1. Retry with the same model using a focused repair prompt
    2. Escalate to a more powerful model
    3. Try with a simplified schema
    4. Extract whatever partial data is valid

    Example::

        healer = SchemaHealer()
        result = await healer.heal(
            raw_output="{ bad json }",
            schema=MyModel,
            validation_errors=[...],
            provider=my_provider,
            config=CompletionConfig(model="gpt-4o-mini"),
        )
    """

    def __init__(
        self,
        validator: SchemaValidator | None = None,
        event_emitter: EventEmitter | None = None,
    ) -> None:
        self._validator = validator or SchemaValidator()
        self._emitter: EventEmitter = event_emitter or _NullEmitter()

    async def heal(
        self,
        raw_output: str,
        schema: type[T],
        validation_errors: list[dict[str, Any]],
        provider: LLMProvider,
        config: CompletionConfig,
        *,
        max_attempts: int = 3,
        pipeline_id: str = "",
        step_id: str = "",
        budget: BudgetConstraint | None = None,
    ) -> T:
        """Run the self-healing loop to fix invalid LLM output.

        Tries multiple strategies in sequence:
        1. ``RETRY_SAME`` — send repair prompt to the same model
        2. ``ESCALATE_MODEL`` — switch to a more capable model
        3. ``SIMPLIFY_SCHEMA`` — retry with simpler schema
        4. ``EXTRACT_PARTIAL`` — last resort, extract whatever works

        Args:
            raw_output: The original invalid LLM output.
            schema: The Pydantic model class to validate against.
            validation_errors: Errors from the initial validation attempt.
            provider: LLM provider to use for repair completions.
            config: Completion configuration.
            max_attempts: Maximum number of LLM calls to make.
            pipeline_id: For event tracking.
            step_id: For event tracking.

        Returns:
            A validated instance of ``schema``.

        Raises:
            SchemaHealingFailed: If all attempts fail.
        """
        schema_name = schema.__name__
        current_errors = validation_errors
        current_output = raw_output
        current_config = config.model_copy()

        strategies = self._plan_strategies(max_attempts, config.model)

        for attempt, strategy in enumerate(strategies, 1):
            logger.info(
                "Healing attempt %d/%d for schema '%s' using strategy %s",
                attempt,
                len(strategies),
                schema_name,
                strategy.value,
            )

            if budget:
                budget.consume_retry()
                if budget.is_exhausted:
                    raise BudgetExhaustedError(
                        "retries",
                        float(budget.budget.max_retries),
                        float(budget.retries_used),
                    )

            await self._emit_heal_attempt(
                attempt=attempt,
                strategy=strategy,
                errors=current_errors,
                pipeline_id=pipeline_id,
                step_id=step_id,
                schema_name=schema_name,
            )

            match strategy:
                case HealingStrategy.RETRY_SAME:
                    messages = get_healing_prompt(
                        current_output, current_errors, schema
                    )
                    result = await self._try_completion(
                        provider, messages, current_config, schema
                    )

                case HealingStrategy.ESCALATE_MODEL:
                    escalated_model = self._get_escalation_model(
                        current_config.model
                    )
                    if escalated_model and escalated_model != current_config.model:
                        escalated_config = current_config.model_copy(
                            update={"model": escalated_model}
                        )
                        logger.info(
                            "Escalating from %s to %s",
                            current_config.model,
                            escalated_model,
                        )
                        messages = get_healing_prompt(
                            current_output, current_errors, schema
                        )
                        result = await self._try_completion(
                            provider, messages, escalated_config, schema
                        )
                    else:
                        # No escalation available, retry with same model
                        messages = get_healing_prompt(
                            current_output, current_errors, schema
                        )
                        result = await self._try_completion(
                            provider, messages, current_config, schema
                        )

                case HealingStrategy.SIMPLIFY_SCHEMA:
                    messages = get_simplified_healing_prompt(
                        current_output, current_errors, schema
                    )
                    # Use lower temperature for more predictable output
                    simple_config = current_config.model_copy(
                        update={"temperature": 0.1, "json_mode": True}
                    )
                    result = await self._try_completion(
                        provider, messages, simple_config, schema
                    )

                case HealingStrategy.EXTRACT_PARTIAL:
                    partial = extract_partial(current_output, schema)
                    if partial is not None:
                        await self._emit_heal_success(
                            attempt=attempt,
                            strategy=strategy,
                            pipeline_id=pipeline_id,
                            step_id=step_id,
                            schema_name=schema_name,
                        )
                        return partial
                    result = None

            if result is not None:
                # Validate the repair output
                validation = self._validator.validate(
                    result.content, schema
                )
                if validation.success and validation.parsed_object is not None:
                    await self._emit_heal_success(
                        attempt=attempt,
                        strategy=strategy,
                        pipeline_id=pipeline_id,
                        step_id=step_id,
                        schema_name=schema_name,
                    )
                    return schema.model_validate(validation.parsed_object)

                # Update for next attempt
                current_errors = validation.errors
                current_output = result.content
                logger.debug(
                    "Attempt %d failed with %d errors",
                    attempt,
                    len(current_errors),
                )

        # All attempts failed
        await self._emit_heal_failed(
            attempts=len(strategies),
            pipeline_id=pipeline_id,
            step_id=step_id,
            schema_name=schema_name,
        )
        raise SchemaHealingFailed(schema_name, len(strategies))

    def _plan_strategies(
        self, max_attempts: int, model: str | None
    ) -> list[HealingStrategy]:
        """Plan the sequence of healing strategies to try.

        Strategy order:
        1. Always start with RETRY_SAME
        2. ESCALATE_MODEL if a better model is available
        3. SIMPLIFY_SCHEMA for structural issues
        4. EXTRACT_PARTIAL as last resort
        """
        strategies: list[HealingStrategy] = []

        # Attempt 1: simple retry
        strategies.append(HealingStrategy.RETRY_SAME)

        # Attempt 2: escalate if possible
        if max_attempts >= 2:
            escalation = self._get_escalation_model(model)
            if escalation and escalation != model:
                strategies.append(HealingStrategy.ESCALATE_MODEL)
            else:
                strategies.append(HealingStrategy.SIMPLIFY_SCHEMA)

        # Attempt 3+: try remaining strategies
        if max_attempts >= 3:
            if HealingStrategy.SIMPLIFY_SCHEMA not in strategies:
                strategies.append(HealingStrategy.SIMPLIFY_SCHEMA)
            else:
                strategies.append(HealingStrategy.RETRY_SAME)

        # Always end with partial extraction as final fallback
        if max_attempts >= 4:
            strategies.append(HealingStrategy.EXTRACT_PARTIAL)
        elif len(strategies) < max_attempts:
            strategies.append(HealingStrategy.EXTRACT_PARTIAL)

        return strategies[:max_attempts]

    def _get_escalation_model(self, model: str | None) -> str | None:
        """Look up the escalation target for a model."""
        if model is None:
            return None
        # Strip provider prefix if present (e.g. "openai:gpt-4o-mini")
        bare_model = model.split(":")[-1] if ":" in model else model
        escalated = _ESCALATION_MAP.get(bare_model)
        if escalated is None:
            return None
        # Re-add provider prefix if it was present
        if ":" in model:
            prefix = model.split(":")[0]
            return f"{prefix}:{escalated}"
        return escalated

    async def _try_completion(
        self,
        provider: LLMProvider,
        messages: list[Message],
        config: CompletionConfig,
        schema: type[BaseModel],
    ) -> LLMResponse | None:
        """Attempt an LLM completion, returning None on failure."""
        try:
            response = await provider.complete(messages, config)
            return response
        except Exception as exc:
            logger.warning(
                "Healing completion failed: %s: %s",
                type(exc).__name__,
                exc,
            )
            return None

    # --- Event emission ------------------------------------------------------

    async def _emit_heal_attempt(
        self,
        *,
        attempt: int,
        strategy: HealingStrategy,
        errors: list[dict[str, Any]],
        pipeline_id: str,
        step_id: str,
        schema_name: str,
    ) -> None:
        event = Event(
            event_type=EventType.SCHEMA_HEAL_ATTEMPT,
            pipeline_id=pipeline_id or "unknown",
            step_id=step_id or None,
            data={
                "attempt": attempt,
                "strategy": strategy.value,
                "schema_name": schema_name,
                "error_count": len(errors),
                "errors_summary": [
                    f"{e.get('loc', '?')}: {e.get('msg', '?')}"
                    for e in errors[:3]
                ],
            },
        )
        await self._emitter.emit(event)

    async def _emit_heal_success(
        self,
        *,
        attempt: int,
        strategy: HealingStrategy,
        pipeline_id: str,
        step_id: str,
        schema_name: str,
    ) -> None:
        event = Event(
            event_type=EventType.SCHEMA_HEAL_SUCCESS,
            pipeline_id=pipeline_id or "unknown",
            step_id=step_id or None,
            data={
                "attempt": attempt,
                "strategy": strategy.value,
                "schema_name": schema_name,
            },
        )
        await self._emitter.emit(event)
        logger.info(
            "Schema '%s' healed on attempt %d via %s",
            schema_name,
            attempt,
            strategy.value,
        )

    async def _emit_heal_failed(
        self,
        *,
        attempts: int,
        pipeline_id: str,
        step_id: str,
        schema_name: str,
    ) -> None:
        event = Event(
            event_type=EventType.SCHEMA_HEAL_FAILED,
            pipeline_id=pipeline_id or "unknown",
            step_id=step_id or None,
            data={
                "attempts": attempts,
                "schema_name": schema_name,
            },
        )
        await self._emitter.emit(event)
        logger.warning(
            "Schema '%s' healing FAILED after %d attempts",
            schema_name,
            attempts,
        )
