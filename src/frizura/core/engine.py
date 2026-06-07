"""FrizuraEngine — the main execution engine (THE HEART).

This module wires together *every* subsystem of Frizura:

* **Pipeline compiler** — validates the DAG.
* **Smart router** — selects the best model for each step.
* **Schema validator / healer** — guarantees structured output.
* **Privacy classifier / PII masker** — gates cloud vs. local.
* **Event store** — records every action for time-travel.
* **Hybrid gateway / Local pool** — dispatches to providers.
* **Budget tracker** — enforces cost / token / time limits.

All integrations are lazily imported so that missing optional packages
(e.g. ``rich``) don't crash the import.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, TYPE_CHECKING
from uuid import uuid4

from frizura.core.context import ExecutionContext
from frizura.core.events import Event, EventType, StateSnapshot
from frizura.core.exceptions import (
    BudgetExhaustedError,
    GraphError,
    PipelineError,
    ProviderError,
    SchemaHealingFailed,
    SchemaValidationError,
    StepError,
    ReplayError,
    SnapshotNotFoundError,
)
from frizura.core.graph import CompiledPipeline, Pipeline, Step, StepType
from frizura.models.budget import Budget, BudgetConstraint, CostEntry
from frizura.models.config import FrizuraConfig
from frizura.models.execution import (
    LLMResponse,
    Message,
    MessageRole,
    PipelineResult,
    StepResult,
    TokenUsage,
)
from frizura.models.providers import CompletionConfig, ModelInfo

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------- #
# Engine
# ---------------------------------------------------------------------- #

class FrizuraEngine:
    """Central orchestration engine.

    The engine is the single integration point that ties together every
    Frizura subsystem.  All subsystems are initialised lazily on first use
    so importing the engine does not pull in heavy dependencies.

    Usage::

        engine = FrizuraEngine()
        result = await engine.run(pipeline, input_data, budget=Budget(max_cost=0.10))
    """

    def __init__(self, config: FrizuraConfig | None = None) -> None:
        self.config = config or FrizuraConfig()
        self._event_sequence: int = 0

        # Lazy-init slots — populated on first use
        self._event_store: Any | None = None
        self._model_registry: Any | None = None
        self._router: Any | None = None
        self._schema_validator: Any | None = None
        self._schema_healer: Any | None = None
        self._privacy_classifier: Any | None = None
        self._pii_masker: Any | None = None
        self._local_pool: Any | None = None
        self._hybrid_gateway: Any | None = None

        # In-memory event log (used when event store is not available)
        self._events: list[Event] = []
        # In-memory snapshot store
        self._snapshots: dict[str, list[StateSnapshot]] = {}
        # Background tasks tracking
        self._pending_tasks: list[asyncio.Task] = []

        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper(), logging.INFO),
        )
        logger.info("FrizuraEngine initialised (default_model=%s)", self.config.default_model)

    # ------------------------------------------------------------------ #
    # Lazy subsystem accessors
    # ------------------------------------------------------------------ #

    def _get_event_store(self) -> Any | None:
        """Try to load the EventStore from frizura.timetravel."""
        if self._event_store is not None:
            return self._event_store
        try:
            from frizura.timetravel.store import EventStore  # type: ignore[import-untyped]
            self._event_store = EventStore(
                db_path=str(self.config.timetravel.db_path),
            )
            return self._event_store
        except (ImportError, Exception) as exc:
            logger.debug("EventStore not available: %s", exc)
            return None

    def _get_router(self) -> Any | None:
        """Try to load the SmartRouter from frizura.router."""
        if self._router is not None:
            return self._router
        try:
            from frizura.router.smart_router import SmartRouter  # type: ignore[import-untyped]
            self._router = SmartRouter(config=self.config.router)
            return self._router
        except (ImportError, Exception) as exc:
            logger.debug("SmartRouter not available: %s", exc)
            return None

    def _get_model_registry(self) -> Any | None:
        """Try to load the ModelRegistry from frizura.router."""
        if self._model_registry is not None:
            return self._model_registry
        try:
            from frizura.router.registry import ModelRegistry  # type: ignore[import-untyped]
            self._model_registry = ModelRegistry()
            return self._model_registry
        except (ImportError, Exception) as exc:
            logger.debug("ModelRegistry not available: %s", exc)
            return None

    def _get_schema_validator(self) -> Any | None:
        if self._schema_validator is not None:
            return self._schema_validator
        try:
            from frizura.schema.validator import SchemaValidator  # type: ignore[import-untyped]
            self._schema_validator = SchemaValidator()
            return self._schema_validator
        except (ImportError, Exception) as exc:
            logger.debug("SchemaValidator not available: %s", exc)
            return None

    def _get_schema_healer(self) -> Any | None:
        if self._schema_healer is not None:
            return self._schema_healer
        try:
            from frizura.schema.healer import SchemaHealer  # type: ignore[import-untyped]
            self._schema_healer = SchemaHealer()
            return self._schema_healer
        except (ImportError, Exception) as exc:
            logger.debug("SchemaHealer not available: %s", exc)
            return None

    def _get_privacy_classifier(self) -> Any | None:
        if self._privacy_classifier is not None:
            return self._privacy_classifier
        try:
            from frizura.swarm.privacy import PrivacyClassifier  # type: ignore[import-untyped]
            self._privacy_classifier = PrivacyClassifier()
            return self._privacy_classifier
        except (ImportError, Exception) as exc:
            logger.debug("PrivacyClassifier not available: %s", exc)
            return None

    def _get_pii_masker(self) -> Any | None:
        if self._pii_masker is not None:
            return self._pii_masker
        try:
            from frizura.swarm.pii import PIIMasker  # type: ignore[import-untyped]
            self._pii_masker = PIIMasker()
            return self._pii_masker
        except (ImportError, Exception) as exc:
            logger.debug("PIIMasker not available: %s", exc)
            return None

    def _get_hybrid_gateway(self) -> Any | None:
        if self._hybrid_gateway is not None:
            return self._hybrid_gateway
        try:
            from frizura.swarm.gateway import HybridGateway  # type: ignore[import-untyped]
            self._hybrid_gateway = HybridGateway(config=self.config.swarm)
            return self._hybrid_gateway
        except (ImportError, Exception) as exc:
            logger.debug("HybridGateway not available: %s", exc)
            return None

    # ------------------------------------------------------------------ #
    # Event helpers
    # ------------------------------------------------------------------ #

    def _emit(
        self,
        event_type: EventType,
        pipeline_id: str,
        *,
        step_id: str | None = None,
        step_name: str | None = None,
        data: dict[str, Any] | None = None,
        parent_event_id: str | None = None,
    ) -> Event:
        """Create and record an event."""
        self._event_sequence += 1
        event = Event(
            event_type=event_type,
            pipeline_id=pipeline_id,
            step_id=step_id,
            step_name=step_name,
            data=data or {},
            parent_event_id=parent_event_id,
            sequence_number=self._event_sequence,
        )
        self._events.append(event)
        store = self._get_event_store()
        if store is not None:
            try:
                # EventStore may expose sync or async API; handle both
                _maybe_call = getattr(store, "append", None)
                if _maybe_call and asyncio.iscoroutinefunction(_maybe_call):
                    task = asyncio.get_event_loop().create_task(_maybe_call(event))
                    self._pending_tasks.append(task)
                elif _maybe_call:
                    _maybe_call(event)
            except Exception:
                logger.debug("Failed to persist event %s", event.id, exc_info=True)
        logger.debug("event %s  %s", event_type, event.id)
        return event

    def _save_snapshot(self, snapshot: StateSnapshot) -> None:
        """Persist a state snapshot."""
        pid = snapshot.pipeline_id
        if pid not in self._snapshots:
            self._snapshots[pid] = []
        self._snapshots[pid].append(snapshot)
        store = self._get_event_store()
        if store is not None:
            try:
                _maybe_call = getattr(store, "save_snapshot", None)
                if _maybe_call and asyncio.iscoroutinefunction(_maybe_call):
                    task = asyncio.get_event_loop().create_task(_maybe_call(snapshot))
                    self._pending_tasks.append(task)
                elif _maybe_call:
                    _maybe_call(snapshot)
            except Exception:
                logger.debug("Failed to persist snapshot %s", snapshot.snapshot_id, exc_info=True)

    # ------------------------------------------------------------------ #
    # Model resolution
    # ------------------------------------------------------------------ #

    def _parse_model_spec(self, spec: str) -> tuple[str, str]:
        """Parse ``"provider:model_id"`` into ``(provider, model_id)``.

        If no colon, the default provider from the config is assumed.
        """
        if ":" in spec:
            provider, model_id = spec.split(":", 1)
            return provider, model_id
        # Assume default provider from config
        default = self.config.default_model
        if ":" in default:
            provider = default.split(":", 1)[0]
        else:
            provider = "openai"
        return provider, spec

    async def _resolve_model(
        self,
        step: Step,
        context: ExecutionContext,
    ) -> tuple[str, str, str]:
        """Determine which model+provider to use for a step.

        Returns ``(provider, model_id, reason)``.
        """
        if step.model:
            provider, model_id = self._parse_model_spec(step.model)
            return provider, model_id, f"explicit: {step.model}"

        # Try smart router
        router = self._get_router()
        if router is not None:
            try:
                route_fn = getattr(router, "route", None)
                if route_fn:
                    if asyncio.iscoroutinefunction(route_fn):
                        decision = await route_fn(
                            messages=context.memory,
                            budget=context.budget,
                            config=step.config,
                        )
                    else:
                        decision = route_fn(
                            messages=context.memory,
                            budget=context.budget,
                            config=step.config,
                        )
                    if decision and hasattr(decision, "model_id"):
                        provider = getattr(decision, "provider", "openai")
                        reason = getattr(decision, "reason", "router")
                        return provider, decision.model_id, f"routed: {reason}"
            except Exception as exc:
                logger.warning("Router failed, falling back to default: %s", exc)

        # Fallback to config default
        provider, model_id = self._parse_model_spec(self.config.default_model)
        return provider, model_id, f"default: {self.config.default_model}"

    # ------------------------------------------------------------------ #
    # LLM provider dispatch
    # ------------------------------------------------------------------ #

    async def _call_llm(
        self,
        provider: str,
        model_id: str,
        messages: list[Message],
        config: CompletionConfig,
        pipeline_id: str,
        step_id: str,
    ) -> LLMResponse:
        """Dispatch a completion request to the appropriate provider.

        Tries the HybridGateway first, then falls back to a direct
        provider call.
        """
        self._emit(
            EventType.LLM_REQUEST,
            pipeline_id,
            step_id=step_id,
            data={
                "provider": provider,
                "model": model_id,
                "message_count": len(messages),
                "temperature": config.temperature,
            },
        )

        t0 = time.perf_counter()

        # --- Try HybridGateway ---
        gateway = self._get_hybrid_gateway()
        if gateway is not None:
            try:
                complete_fn = getattr(gateway, "complete", None)
                if complete_fn:
                    if asyncio.iscoroutinefunction(complete_fn):
                        resp = await complete_fn(
                            provider=provider,
                            model=model_id,
                            messages=messages,
                            config=config,
                        )
                    else:
                        resp = complete_fn(
                            provider=provider,
                            model=model_id,
                            messages=messages,
                            config=config,
                        )
                    if isinstance(resp, LLMResponse):
                        latency = (time.perf_counter() - t0) * 1000
                        resp = resp.model_copy(update={"latency_ms": latency})
                        self._emit(
                            EventType.LLM_RESPONSE,
                            pipeline_id,
                            step_id=step_id,
                            data={
                                "provider": provider,
                                "model": model_id,
                                "latency_ms": latency,
                                "tokens": resp.usage.total_tokens,
                            },
                        )
                        return resp
            except Exception as exc:
                logger.warning("HybridGateway failed: %s — trying direct provider", exc)

        # --- Direct provider fallback ---
        try:
            provider_mod = self._load_provider(provider)
            if provider_mod is not None:
                complete_fn = getattr(provider_mod, "complete", None)
                if complete_fn is None:
                    complete_fn = getattr(provider_mod, "acomplete", None)
                if complete_fn:
                    msg_dicts = [m.model_dump(exclude_none=True) for m in messages]
                    kwargs: dict[str, Any] = {
                        "model": model_id,
                        "messages": msg_dicts,
                        "temperature": config.temperature,
                    }
                    if config.max_tokens is not None:
                        kwargs["max_tokens"] = config.max_tokens
                    if config.json_mode:
                        kwargs["json_mode"] = True
                    if config.tools:
                        kwargs["tools"] = config.tools
                    if config.seed is not None:
                        kwargs["seed"] = config.seed

                    if asyncio.iscoroutinefunction(complete_fn):
                        result = await complete_fn(**kwargs)
                    else:
                        result = complete_fn(**kwargs)

                    latency = (time.perf_counter() - t0) * 1000

                    # Normalise into LLMResponse
                    if isinstance(result, LLMResponse):
                        resp = result.model_copy(update={"latency_ms": latency})
                    elif isinstance(result, dict):
                        resp = LLMResponse(
                            content=result.get("content", ""),
                            model=model_id,
                            provider=provider,
                            usage=TokenUsage(**(result.get("usage", {}))),
                            finish_reason=result.get("finish_reason", "stop"),
                            latency_ms=latency,
                            raw=result,
                        )
                    else:
                        resp = LLMResponse(
                            content=str(result),
                            model=model_id,
                            provider=provider,
                            latency_ms=latency,
                        )

                    self._emit(
                        EventType.LLM_RESPONSE,
                        pipeline_id,
                        step_id=step_id,
                        data={
                            "provider": provider,
                            "model": model_id,
                            "latency_ms": latency,
                            "tokens": resp.usage.total_tokens,
                        },
                    )
                    return resp
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(
                provider, model_id, f"LLM call failed: {exc}"
            ) from exc

        raise ProviderError(provider, model_id, "No usable provider found")

    def _load_provider(self, provider_name: str) -> Any | None:
        """Attempt to import a provider adapter module."""
        try:
            import importlib
            mod = importlib.import_module(f"frizura.providers.{provider_name}")
            return mod
        except ImportError:
            logger.debug("Provider module frizura.providers.%s not found", provider_name)
            return None

    # ------------------------------------------------------------------ #
    # Schema validation / healing
    # ------------------------------------------------------------------ #

    async def _validate_and_heal(
        self,
        raw_content: str,
        output_schema: type,
        step: Step,
        context: ExecutionContext,
        provider: str,
        model_id: str,
        pipeline_id: str,
        max_attempts: int = 3,
    ) -> Any:
        """Validate LLM output against *output_schema*, healing if needed.

        Returns the validated Pydantic model instance.
        """
        schema_name = getattr(output_schema, "__name__", str(output_schema))

        for attempt in range(max_attempts):
            try:
                # Try JSON extraction — the output may have markdown fences
                json_str = _extract_json(raw_content)
                parsed = json.loads(json_str)
                instance = output_schema.model_validate(parsed)

                self._emit(
                    EventType.SCHEMA_VALIDATION_OK,
                    pipeline_id,
                    step_id=step.id,
                    step_name=step.name,
                    data={"schema": schema_name, "attempt": attempt},
                )
                return instance

            except (json.JSONDecodeError, Exception) as exc:
                if attempt == 0:
                    self._emit(
                        EventType.SCHEMA_VALIDATION_FAILED,
                        pipeline_id,
                        step_id=step.id,
                        step_name=step.name,
                        data={
                            "schema": schema_name,
                            "error": str(exc),
                            "attempt": attempt,
                        },
                    )

                if attempt >= max_attempts - 1:
                    raise SchemaHealingFailed(schema_name, attempts=max_attempts) from exc

                # --- Healing attempt ---
                self._emit(
                    EventType.SCHEMA_HEAL_ATTEMPT,
                    pipeline_id,
                    step_id=step.id,
                    step_name=step.name,
                    data={"schema": schema_name, "attempt": attempt + 1},
                )

                if context.budget:
                    context.budget.consume_retry()
                    if context.budget.is_exhausted:
                        raise BudgetExhaustedError(
                            "retries",
                            float(context.budget.budget.max_retries),
                            float(context.budget.retries_used),
                        ) from exc

                # Build healing prompt
                schema_json = json.dumps(
                    output_schema.model_json_schema(), indent=2
                )
                heal_messages = [
                    Message(
                        role=MessageRole.SYSTEM,
                        content=(
                            "You are a JSON repair assistant. The following LLM "
                            "output failed to validate against the expected JSON "
                            "schema. Fix the output so it validates correctly. "
                            "Return ONLY valid JSON, no explanation."
                        ),
                    ),
                    Message(
                        role=MessageRole.USER,
                        content=(
                            f"Expected schema:\n```json\n{schema_json}\n```\n\n"
                            f"Invalid output:\n```\n{raw_content}\n```\n\n"
                            f"Error: {exc}\n\n"
                            f"Please output the corrected JSON only."
                        ),
                    ),
                ]

                heal_config = CompletionConfig(
                    temperature=0.0,
                    json_mode=True,
                    max_tokens=step.config.max_tokens,
                )

                try:
                    heal_resp = await self._call_llm(
                        provider,
                        model_id,
                        heal_messages,
                        heal_config,
                        pipeline_id,
                        step.id,
                    )
                    raw_content = heal_resp.content
                    # Update budget
                    if context.budget:
                        cost = _estimate_cost_simple(
                            heal_resp.usage.input_tokens,
                            heal_resp.usage.output_tokens,
                        )
                        context.budget.consume(
                            cost=cost,
                            tokens=heal_resp.usage.total_tokens,
                        )
                except ProviderError:
                    logger.warning("Healing LLM call failed on attempt %d", attempt + 1)
                    continue

        raise SchemaHealingFailed(schema_name, attempts=max_attempts)

    # ------------------------------------------------------------------ #
    # Step execution
    # ------------------------------------------------------------------ #

    async def execute_step(
        self,
        step: Step,
        context: ExecutionContext,
    ) -> StepResult:
        """Execute a single pipeline step.

        This is the main per-step logic. Steps may be skipped (via
        ``step.condition``), delegated to an LLM, or handled by a custom
        transform function.
        """
        t0 = time.perf_counter()
        pipeline_id = context.pipeline_id

        # --- Emit STEP_STARTED ---
        self._emit(
            EventType.STEP_STARTED,
            pipeline_id,
            step_id=step.id,
            step_name=step.name,
            data={"step_type": step.step_type, "model": step.model or "auto"},
        )

        # --- Condition check ---
        if step.condition is not None:
            try:
                cond = step.condition
                if asyncio.iscoroutinefunction(cond):
                    should_run = await cond(context)
                else:
                    should_run = cond(context)
            except Exception as exc:
                logger.warning("Condition check failed for %s: %s", step.name, exc)
                should_run = False

            if not should_run:
                self._emit(
                    EventType.STEP_SKIPPED,
                    pipeline_id,
                    step_id=step.id,
                    step_name=step.name,
                    data={"reason": "condition_false"},
                )
                elapsed = (time.perf_counter() - t0) * 1000
                return StepResult(
                    step_id=step.id,
                    step_name=step.name,
                    status="skipped",
                    duration_ms=elapsed,
                    completed_at=_utcnow(),
                )

        # --- Budget pre-check ---
        if context.budget and context.budget.is_exhausted:
            raise BudgetExhaustedError(
                "budget",
                context.budget.budget.max_cost or 0.0,
                float(context.budget.spent_cost),
            )

        try:
            # --- TRANSFORM steps: call the handler directly ---
            if step.step_type == StepType.TRANSFORM:
                return await self._execute_transform(step, context, t0)

            # --- HUMAN steps: placeholder ---
            if step.step_type == StepType.HUMAN:
                return await self._execute_human(step, context, t0)

            # --- LLM / PARALLEL / BRANCH steps ---
            return await self._execute_llm_step(step, context, t0)

        except (BudgetExhaustedError, SchemaHealingFailed):
            raise
        except StepError:
            raise
        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000
            self._emit(
                EventType.STEP_FAILED,
                pipeline_id,
                step_id=step.id,
                step_name=step.name,
                data={"error": str(exc)},
            )
            return StepResult(
                step_id=step.id,
                step_name=step.name,
                status="failed",
                error=str(exc),
                duration_ms=elapsed,
                completed_at=_utcnow(),
            )

    async def _execute_transform(
        self, step: Step, context: ExecutionContext, t0: float
    ) -> StepResult:
        """Execute a TRANSFORM step by calling its handler."""
        if step.handler is None:
            raise StepError(step.id, step.name, "TRANSFORM step has no handler")

        handler = step.handler
        if asyncio.iscoroutinefunction(handler):
            output = await handler(context)
        elif callable(handler):
            output = handler(context)
        else:
            raise StepError(step.id, step.name, "Handler is not callable")

        context.set(step.name, output)
        elapsed = (time.perf_counter() - t0) * 1000

        self._emit(
            EventType.STEP_COMPLETED,
            context.pipeline_id,
            step_id=step.id,
            step_name=step.name,
            data={"duration_ms": elapsed},
        )
        return StepResult(
            step_id=step.id,
            step_name=step.name,
            status="completed",
            output=output,
            duration_ms=elapsed,
            completed_at=_utcnow(),
        )

    async def _execute_human(
        self, step: Step, context: ExecutionContext, t0: float
    ) -> StepResult:
        """Execute a HUMAN step (placeholder — returns pending status)."""
        elapsed = (time.perf_counter() - t0) * 1000
        self._emit(
            EventType.STEP_COMPLETED,
            context.pipeline_id,
            step_id=step.id,
            step_name=step.name,
            data={"status": "human_pending"},
        )
        return StepResult(
            step_id=step.id,
            step_name=step.name,
            status="pending_human",
            duration_ms=elapsed,
            completed_at=_utcnow(),
        )

    async def _execute_llm_step(
        self, step: Step, context: ExecutionContext, t0: float
    ) -> StepResult:
        """Execute a step that requires an LLM call."""
        pipeline_id = context.pipeline_id

        # 1) Build messages
        messages: list[Message] = []
        if step.system_prompt:
            messages.append(
                Message(role=MessageRole.SYSTEM, content=step.system_prompt)
            )
        # Add conversation memory
        messages.extend(context.memory)

        # If there's a handler, call it to produce the user prompt
        if step.handler is not None:
            handler = step.handler
            if asyncio.iscoroutinefunction(handler):
                user_content = await handler(context)
            elif callable(handler):
                user_content = handler(context)
            else:
                user_content = str(handler)
            if isinstance(user_content, str):
                messages.append(
                    Message(role=MessageRole.USER, content=user_content)
                )
            elif isinstance(user_content, Message):
                messages.append(user_content)

        # 2) Resolve model
        provider, model_id, routing_reason = await self._resolve_model(step, context)
        self._emit(
            EventType.ROUTING_DECISION,
            pipeline_id,
            step_id=step.id,
            step_name=step.name,
            data={
                "provider": provider,
                "model": model_id,
                "reason": routing_reason,
            },
        )

        # 3) Schema instructions
        config = step.config.model_copy()
        if step.output_schema is not None:
            schema_json = json.dumps(
                step.output_schema.model_json_schema(), indent=2
            )
            schema_instruction = (
                f"\n\nYou MUST respond with valid JSON matching this schema:\n"
                f"```json\n{schema_json}\n```\n"
                f"Return ONLY the JSON object, nothing else."
            )
            # Append to last user message or create one
            if messages and messages[-1].role == MessageRole.USER:
                messages[-1] = Message(
                    role=MessageRole.USER,
                    content=messages[-1].content + schema_instruction,
                )
            else:
                messages.append(
                    Message(role=MessageRole.USER, content=schema_instruction)
                )
            config.json_mode = True

        # 4) Call LLM
        llm_response = await self._call_llm(
            provider, model_id, messages, config, pipeline_id, step.id
        )

        # 5) Update budget
        cost = _estimate_cost_simple(
            llm_response.usage.input_tokens,
            llm_response.usage.output_tokens,
        )
        if context.budget:
            context.budget.consume(
                cost=cost,
                tokens=llm_response.usage.total_tokens,
                time=(time.perf_counter() - t0),
            )

        # 6) Schema validation + healing
        output: Any = llm_response.content
        healing_attempts = 0
        status = "completed"

        if step.output_schema is not None:
            try:
                output = await self._validate_and_heal(
                    llm_response.content,
                    step.output_schema,
                    step,
                    context,
                    provider,
                    model_id,
                    pipeline_id,
                )
                # If we got here, validation succeeded (possibly after healing)
            except SchemaHealingFailed as exc:
                healing_attempts = exc.attempts
                status = "failed"
                output = llm_response.content  # raw content
                raise
            except BudgetExhaustedError:
                raise

        # 7) Update context state
        if isinstance(output, BaseModel):
            context.set(step.name, output.model_dump())
        else:
            context.set(step.name, output)

        # Also add assistant response to memory
        context.add_message(
            Message(role=MessageRole.ASSISTANT, content=llm_response.content)
        )

        # 8) Save snapshot
        snapshot = context.to_snapshot(step.id, context.current_step_index)
        self._save_snapshot(snapshot)
        self._emit(
            EventType.STATE_SNAPSHOT,
            pipeline_id,
            step_id=step.id,
            step_name=step.name,
            data={"snapshot_id": snapshot.snapshot_id},
        )

        # 9) Emit STEP_COMPLETED
        elapsed = (time.perf_counter() - t0) * 1000
        self._emit(
            EventType.STEP_COMPLETED,
            pipeline_id,
            step_id=step.id,
            step_name=step.name,
            data={
                "duration_ms": elapsed,
                "model": f"{provider}:{model_id}",
                "cost_usd": float(cost),
                "tokens": llm_response.usage.total_tokens,
                "output": output.model_dump() if isinstance(output, BaseModel) else output,
            },
        )

        return StepResult(
            step_id=step.id,
            step_name=step.name,
            status=status,
            output=output,
            llm_response=llm_response,
            duration_ms=elapsed,
            cost_usd=float(cost),
            model_used=f"{provider}:{model_id}",
            routing_reason=routing_reason,
            healing_attempts=healing_attempts,
            completed_at=_utcnow(),
        )

    # ------------------------------------------------------------------ #
    # Pipeline execution
    # ------------------------------------------------------------------ #

    async def run(
        self,
        pipeline: Pipeline,
        input_data: Any = None,
        budget: Budget | None = None,
        *,
        input: Any = None,  # noqa: A002
    ) -> PipelineResult:
        """Execute a full pipeline.

        Parameters
        ----------
        pipeline:
            The pipeline to run.
        input_data:
            Initial input — stored in ``context.state["input"]`` and, if
            it is a string, also added to memory as a USER message.
        budget:
            Optional budget constraints.

        Returns
        -------
        PipelineResult
            Aggregated result with per-step details.
        """
        actual_input = input_data if input_data is not None else input
        pipeline_id = uuid4().hex[:12]
        t0 = time.perf_counter()
        started_at = _utcnow()

        # Initialize EventStore if available
        store = self._get_event_store()
        if store is not None:
            await store.init()

        try:
            # 1) Create context
            budget_constraint: BudgetConstraint | None = None
            if budget is not None:
                budget_constraint = BudgetConstraint(budget=budget)

            context = ExecutionContext(
                pipeline_id=pipeline_id,
                pipeline_name=pipeline.name,
                budget=budget_constraint,
                config=self.config,
            )

            # Set initial input
            if actual_input is not None:
                context.set("input", actual_input)
                if isinstance(actual_input, str):
                    context.add_message(
                        Message(role=MessageRole.USER, content=actual_input)
                    )

            # 2) Emit PIPELINE_STARTED
            self._emit(
                EventType.PIPELINE_STARTED,
                pipeline_id,
                data={
                    "pipeline_name": pipeline.name,
                    "step_count": len(pipeline.steps),
                    "has_budget": budget is not None,
                },
            )

            step_results: list[StepResult] = []
            models_used: set[str] = set()
            status = "completed"
            error_msg: str | None = None
            final_output: Any = None

            try:
                # 3) Compile pipeline (validate DAG)
                compiled = pipeline.compile()

                # 4) Execute step groups
                for group_idx, group in enumerate(compiled.steps_order):
                    if len(group) == 1:
                        # Single step — run directly
                        step = group[0]
                        context.current_step_index = group_idx
                        result = await self.execute_step(step, context)
                        step_results.append(result)
                        if result.model_used:
                            models_used.add(result.model_used)
                        if result.status == "failed":
                            status = "partial"
                            error_msg = result.error
                    else:
                        # Parallel group — run concurrently
                        async with asyncio.TaskGroup() as tg:
                            tasks: list[asyncio.Task[StepResult]] = []
                            for step in group:
                                ctx_clone = context.clone()
                                ctx_clone.current_step_index = group_idx
                                task = tg.create_task(
                                    self.execute_step(step, ctx_clone),
                                    name=step.name,
                                )
                                tasks.append(task)

                        for task in tasks:
                            result = task.result()
                            step_results.append(result)
                            if result.model_used:
                                models_used.add(result.model_used)
                            # Merge parallel results back into main context
                            if result.output is not None:
                                context.set(result.step_name, result.output)

                # Final output = output of last completed step
                for sr in reversed(step_results):
                    if sr.status in ("completed", "healed") and sr.output is not None:
                        final_output = sr.output
                        break

            except GraphError as exc:
                status = "failed"
                error_msg = str(exc)
                logger.error("Pipeline graph error: %s", exc)
            except BudgetExhaustedError as exc:
                status = "failed"
                error_msg = str(exc)
                self._emit(
                    EventType.BUDGET_EXHAUSTED,
                    pipeline_id,
                    data={"error": str(exc)},
                )
                logger.warning("Budget exhausted: %s", exc)
            except (SchemaHealingFailed, StepError) as exc:
                status = "failed"
                error_msg = str(exc)
                logger.error("Step execution failed: %s", exc)
            except Exception as exc:
                status = "failed"
                error_msg = str(exc)
                logger.exception("Unexpected pipeline error")

            # 5) Emit PIPELINE_COMPLETED / FAILED
            elapsed_total = (time.perf_counter() - t0) * 1000
            total_cost = sum(sr.cost_usd for sr in step_results)
            total_tokens = sum(
                (sr.llm_response.usage.total_tokens if sr.llm_response else 0)
                for sr in step_results
            )

            if status == "completed":
                self._emit(
                    EventType.PIPELINE_COMPLETED,
                    pipeline_id,
                    data={
                        "duration_ms": elapsed_total,
                        "cost_usd": total_cost,
                        "steps": len(step_results),
                    },
                )
            else:
                self._emit(
                    EventType.PIPELINE_FAILED,
                    pipeline_id,
                    data={"error": error_msg, "steps_completed": len(step_results)},
                )

            return PipelineResult(
                pipeline_id=pipeline_id,
                pipeline_name=pipeline.name,
                status=status,
                output=final_output,
                steps=step_results,
                total_duration_ms=elapsed_total,
                total_cost_usd=total_cost,
                total_tokens=total_tokens,
                models_used=sorted(models_used),
                started_at=started_at,
                completed_at=_utcnow(),
                error=error_msg,
            )

        finally:
            if hasattr(self, "_pending_tasks") and self._pending_tasks:
                active_tasks = [t for t in self._pending_tasks if not t.done()]
                if active_tasks:
                    await asyncio.gather(*active_tasks, return_exceptions=True)
                self._pending_tasks.clear()
            if store is not None:
                await store.close()

    # ------------------------------------------------------------------ #
    # Single-shot shortcut
    # ------------------------------------------------------------------ #

    async def run_single(
        self,
        prompt: str,
        *,
        model: str | None = None,
        output_schema: type | None = None,
        budget: Budget | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.7,
    ) -> Any:
        """Shortcut for a single LLM call — no pipeline needed.

        Parameters
        ----------
        prompt:
            The user prompt.
        model:
            Model spec (``"provider:model_id"``).
        output_schema:
            Optional Pydantic model for structured output.
        budget:
            Optional budget constraints.
        system_prompt:
            Optional system prompt.
        temperature:
            Sampling temperature.

        Returns
        -------
        Any
            The parsed output if *output_schema* is given, otherwise the
            raw content string.
        """
        step = Step(
            name="single",
            step_type=StepType.LLM,
            model=model,
            system_prompt=system_prompt,
            output_schema=output_schema,
            config=CompletionConfig(temperature=temperature),
        )
        pipe = Pipeline("single-shot")
        pipe.steps = [step]
        pipe._step_index = {step.id: step}

        result = await self.run(pipe, prompt, budget=budget)

        if result.status == "failed":
            raise PipelineError(result.error or "Single-shot execution failed")

        return result.output

    # ------------------------------------------------------------------ #
    # Replay (time-travel)
    # ------------------------------------------------------------------ #

    async def replay(
        self,
        pipeline_id: str,
        until_step: str | None = None,
    ) -> ExecutionContext:
        """Replay a pipeline up to a given step by restoring from snapshots.

        Parameters
        ----------
        pipeline_id:
            The pipeline run to replay.
        until_step:
            If given, restore the snapshot taken at this step ID.
            Otherwise restore the latest snapshot.

        Returns
        -------
        ExecutionContext
            The restored context.
        """
        snapshots = self._snapshots.get(pipeline_id)
        if not snapshots:
            raise SnapshotNotFoundError(pipeline_id)

        if until_step is not None:
            matching = [s for s in snapshots if s.step_id == until_step]
            if not matching:
                raise SnapshotNotFoundError(pipeline_id, until_step)
            snapshot = matching[-1]
        else:
            snapshot = snapshots[-1]

        context = ExecutionContext.from_snapshot(snapshot, config=self.config)
        logger.info(
            "Replayed pipeline %s to step %s (index %d)",
            pipeline_id,
            snapshot.step_id,
            snapshot.step_index,
        )
        return context

    # ------------------------------------------------------------------ #
    # Inspector
    # ------------------------------------------------------------------ #

    async def inspect(self, pipeline_id: str) -> None:
        """Print a rich inspection of a pipeline run.

        Falls back to plain-text if ``rich`` is not installed.
        """
        events = [e for e in self._events if e.pipeline_id == pipeline_id]
        if not events:
            print(f"No events found for pipeline {pipeline_id}")
            return

        try:
            from rich.console import Console
            from rich.table import Table

            console = Console()
            table = Table(title=f"Pipeline: {pipeline_id}")
            table.add_column("#", style="dim", width=4)
            table.add_column("Type", style="cyan")
            table.add_column("Step", style="green")
            table.add_column("Data", style="white", max_width=60)
            table.add_column("Time", style="yellow", width=12)

            for e in events:
                table.add_row(
                    str(e.sequence_number),
                    e.event_type.value,
                    e.step_name or "",
                    str(e.data)[:60] if e.data else "",
                    e.timestamp.strftime("%H:%M:%S.%f")[:12],
                )
            console.print(table)

        except ImportError:
            # Plain-text fallback
            print(f"\n{'='*60}")
            print(f"Pipeline: {pipeline_id}  ({len(events)} events)")
            print(f"{'='*60}")
            for e in events:
                print(
                    f"  [{e.sequence_number:3d}] {e.event_type.value:30s} "
                    f"step={e.step_name or '-':15s} "
                    f"{str(e.data)[:50]}"
                )
            print()


# ---------------------------------------------------------------------- #
# Module-level helpers
# ---------------------------------------------------------------------- #

def _extract_json(text: str) -> str:
    """Extract the first JSON object/array from *text*.

    Handles markdown code-fenced output like ``​```json\n{...}\n```​``.
    """
    stripped = text.strip()

    # Remove markdown fences
    if stripped.startswith("```"):
        lines = stripped.split("\n")
        # Drop first line (```json) and last line (```)
        inner_lines: list[str] = []
        started = False
        for line in lines:
            if not started and line.strip().startswith("```"):
                started = True
                continue
            if started and line.strip() == "```":
                break
            if started:
                inner_lines.append(line)
        if inner_lines:
            stripped = "\n".join(inner_lines).strip()

    # Find first { or [
    for i, ch in enumerate(stripped):
        if ch in "{[":
            # Find matching closing bracket
            depth = 0
            open_ch = ch
            close_ch = "}" if ch == "{" else "]"
            for j in range(i, len(stripped)):
                if stripped[j] == open_ch:
                    depth += 1
                elif stripped[j] == close_ch:
                    depth -= 1
                    if depth == 0:
                        return stripped[i : j + 1]
            # No matching close found — return from the opening bracket
            return stripped[i:]

    return stripped


def _estimate_cost_simple(
    input_tokens: int,
    output_tokens: int,
    input_price_per_1m: float = 0.15,
    output_price_per_1m: float = 0.60,
) -> Decimal:
    """Rough cost estimate when we don't have model-specific pricing."""
    cost = (
        (input_tokens / 1_000_000) * input_price_per_1m
        + (output_tokens / 1_000_000) * output_price_per_1m
    )
    return Decimal(str(round(cost, 8)))


# Pydantic BaseModel import for isinstance check
from pydantic import BaseModel as _BaseModel  # noqa: E402
BaseModel = _BaseModel  # re-export so the reference in _execute_llm_step resolves
