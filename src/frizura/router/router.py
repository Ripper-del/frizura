"""Budget-aware smart model router.

The :class:`SmartRouter` is the main entry-point for automatic model
selection.  It combines complexity analysis, budget checking, capability
filtering, and pluggable strategies to choose the best model for each call.

It also supports *cascade routing*: try the cheapest suitable model first,
and if the call fails or produces a low-quality result, transparently
escalate to the next tier.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from decimal import Decimal
from typing import Any, Protocol

from pydantic import BaseModel, Field

from frizura.core.events import Event, EventType
from frizura.core.exceptions import (
    BudgetExhaustedError,
    NoSuitableModelError,
    RoutingError,
)
from frizura.models.budget import BudgetConstraint
from frizura.models.config import RouterConfig
from frizura.models.execution import LLMResponse, Message
from frizura.models.providers import CompletionConfig, ModelInfo
from frizura.router.analyzer import ComplexityAnalyzer, TaskComplexity
from frizura.router.calculator import CostCalculator
from frizura.router.strategies import BaseStrategy, get_strategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Registry protocol — the router only needs list/get from the registry.
# ---------------------------------------------------------------------------

class ModelRegistry(Protocol):
    """Minimal protocol a model registry must satisfy."""

    def list_models(self) -> Sequence[ModelInfo]: ...
    def get_model(self, model_id: str) -> ModelInfo | None: ...


# ---------------------------------------------------------------------------
# Public models
# ---------------------------------------------------------------------------

class RoutingDecision(BaseModel):
    """The result of a routing decision — which model to use and why."""

    model_info: ModelInfo
    reason: str = ""
    estimated_cost: Decimal = Decimal("0")
    alternatives: list[ModelInfo] = Field(default_factory=list)
    complexity: TaskComplexity | None = None
    strategy_used: str = ""


# ---------------------------------------------------------------------------
# Provider callback type
# ---------------------------------------------------------------------------

ProviderFn = Callable[
    [list[Message], CompletionConfig, ModelInfo],
    Awaitable[LLMResponse],
]


# ---------------------------------------------------------------------------
# SmartRouter
# ---------------------------------------------------------------------------

class SmartRouter:
    """Budget-aware model router with cascading support.

    Parameters
    ----------
    registry:
        Object that lists available models.
    config:
        Routing configuration (thresholds, default strategy, …).
    event_collector:
        Optional list to which routing events are appended.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        config: RouterConfig | None = None,
        *,
        event_collector: list[Event] | None = None,
    ) -> None:
        self._registry = registry
        self._config = config or RouterConfig()
        self._analyzer = ComplexityAnalyzer(
            threshold_cheap=self._config.complexity_threshold_cheap,
            threshold_premium=self._config.complexity_threshold_premium,
        )
        self._calculator = CostCalculator()
        self._events = event_collector
        self._strategy: BaseStrategy = get_strategy(self._config.default_strategy)

    # -- public API ---------------------------------------------------------

    async def route(
        self,
        prompt: str,
        budget: BudgetConstraint | None = None,
        schema: type[BaseModel] | None = None,
        tools: list[Any] | None = None,
        *,
        strategy: str | None = None,
        pipeline_id: str = "",
        step_id: str | None = None,
    ) -> RoutingDecision:
        """Select the best model for the given task.

        Steps
        -----
        1. Analyse complexity of *prompt* + *schema* + *tools*.
        2. Filter models by hard capability requirements.
        3. Filter by budget constraints.
        4. Apply the configured strategy to rank and pick.
        5. Return a :class:`RoutingDecision`.

        Raises
        ------
        NoSuitableModelError
            If no model passes the filters.
        BudgetExhaustedError
            If the budget is completely exhausted before routing.
        """

        # 0. Budget pre-check
        if budget is not None and budget.is_exhausted:
            raise BudgetExhaustedError(
                resource="budget",
                limit=float(budget.budget.max_cost or 0),
                spent=float(budget.spent_cost),
            )

        # 1. Complexity analysis
        complexity = self._analyzer.analyze(prompt, schema=schema, tools=tools)

        # 2. Get all models
        all_models = list(self._registry.list_models())
        if not all_models:
            raise NoSuitableModelError("No models registered in the registry")

        # 3. Capability filter
        candidates = self._filter_capabilities(all_models, schema=schema, tools=tools)
        if not candidates:
            raise NoSuitableModelError(
                "No models match the required capabilities "
                f"(json_mode={schema is not None}, tools={tools is not None})"
            )

        # 4. Budget filter
        if budget is not None:
            budget_ok = self._filter_budget(
                candidates, complexity.estimated_tokens, budget,
            )
            if budget_ok:
                candidates = budget_ok
            else:
                logger.warning(
                    "No models fit budget, proceeding with all %d candidates",
                    len(candidates),
                )

        # 5. Strategy selection
        if strategy:
            active_strategy = get_strategy(strategy)
            strategy_name = strategy
        elif budget and budget.budget.prefer:
            pref = budget.budget.prefer
            if pref == "quality":
                strategy_name = "best_quality"
            elif pref == "speed":
                strategy_name = "fastest"
            elif pref == "cost":
                strategy_name = "cheapest"
            else:
                strategy_name = self._config.default_strategy
            active_strategy = get_strategy(strategy_name)
        else:
            active_strategy = self._strategy
            strategy_name = self._config.default_strategy

        selected = active_strategy.select(candidates, complexity, budget)

        if selected is None:
            raise NoSuitableModelError(
                f"Strategy '{active_strategy.__class__.__name__}' returned no model"
            )

        # Build alternatives list (up to 3, excluding the selected one)
        alternatives = [m for m in candidates if m.model_id != selected.model_id][:3]

        est_cost = self._calculator.estimate(
            selected,
            complexity.estimated_tokens,
            complexity.estimated_tokens,  # rough output estimate
        )

        # strategy_name already resolved above

        decision = RoutingDecision(
            model_info=selected,
            reason=(
                f"Strategy '{strategy_name}' selected {selected.model_id} "
                f"(tier={selected.tier}) for complexity={complexity.score:.2f}"
            ),
            estimated_cost=est_cost,
            alternatives=alternatives,
            complexity=complexity,
            strategy_used=strategy_name,
        )

        # Emit routing event
        self._emit(
            EventType.ROUTING_DECISION,
            pipeline_id=pipeline_id,
            step_id=step_id,
            data={
                "model": selected.model_id,
                "provider": selected.provider,
                "tier": selected.tier,
                "strategy": strategy_name,
                "complexity_score": complexity.score,
                "estimated_cost": str(est_cost),
                "alternatives": [m.model_id for m in alternatives],
            },
        )

        logger.info(
            "Routed to %s (tier=%s, strategy=%s, score=%.2f, est=$%s)",
            selected.model_id,
            selected.tier,
            strategy_name,
            complexity.score,
            est_cost,
        )

        return decision

    async def route_with_cascade(
        self,
        prompt: str,
        budget: BudgetConstraint | None,
        schema: type[BaseModel] | None,
        provider_fn: ProviderFn,
        *,
        messages: list[Message] | None = None,
        config: CompletionConfig | None = None,
        max_attempts: int = 3,
        pipeline_id: str = "",
        step_id: str | None = None,
    ) -> LLMResponse:
        """Try the cheapest suitable model first, cascading upward on failure.

        Parameters
        ----------
        provider_fn:
            ``async def fn(messages, config, model_info) -> LLMResponse``
            that actually calls the LLM.
        messages:
            Conversation messages.  If not given, a single user message with
            *prompt* is constructed.
        config:
            Base completion config (temperature, etc.).
        max_attempts:
            Maximum number of models to try before giving up.

        Returns
        -------
        LLMResponse
            Response from the first model that succeeds.

        Raises
        ------
        RoutingError
            If all cascade attempts fail.
        """
        if messages is None:
            from frizura.models.execution import MessageRole
            messages = [Message(role=MessageRole.USER, content=prompt)]
        if config is None:
            config = CompletionConfig()

        # Get sorted candidates
        complexity = self._analyzer.analyze(prompt, schema=schema)
        all_models = list(self._registry.list_models())
        candidates = self._filter_capabilities(all_models, schema=schema)

        # Sort by tier order: cheap → standard → premium, then by cost within tier
        tier_rank = {"cheap": 0, "standard": 1, "premium": 2}
        candidates.sort(
            key=lambda m: (
                tier_rank.get(m.tier, 1),
                m.input_price_per_1m + m.output_price_per_1m,
            ),
        )

        if budget is not None:
            budget_filtered = self._filter_budget(
                candidates, complexity.estimated_tokens, budget,
            )
            if budget_filtered:
                candidates = budget_filtered

        errors: list[str] = []

        for attempt, model_info in enumerate(candidates[:max_attempts]):
            try:
                # Inject model into config
                call_config = config.model_copy(
                    update={"model": model_info.model_id},
                )

                t0 = time.perf_counter()
                response = await provider_fn(messages, call_config, model_info)
                elapsed = (time.perf_counter() - t0) * 1000

                # Track cost
                cost_entry = self._calculator.track(
                    model_info, response,
                    step_id=step_id, pipeline_id=pipeline_id,
                )
                if budget is not None:
                    budget.consume(
                        cost=cost_entry.cost_usd,
                        tokens=response.usage.total_tokens,
                        time=elapsed / 1000.0,
                    )

                logger.info(
                    "Cascade attempt %d/%d succeeded: model=%s cost=$%s",
                    attempt + 1, max_attempts, model_info.model_id,
                    cost_entry.cost_usd,
                )
                return response

            except Exception as exc:
                err_msg = f"{model_info.model_id}: {exc}"
                errors.append(err_msg)
                logger.warning(
                    "Cascade attempt %d/%d failed: %s",
                    attempt + 1, max_attempts, err_msg,
                )

                self._emit(
                    EventType.ROUTING_FALLBACK,
                    pipeline_id=pipeline_id,
                    step_id=step_id,
                    data={
                        "failed_model": model_info.model_id,
                        "attempt": attempt + 1,
                        "error": str(exc),
                    },
                )

                if budget is not None:
                    budget.consume_retry()

        raise RoutingError(
            f"All {len(errors)} cascade attempts failed: "
            + "; ".join(errors)
        )

    # -- filtering helpers --------------------------------------------------

    @staticmethod
    def _filter_capabilities(
        models: Sequence[ModelInfo],
        *,
        schema: type[BaseModel] | None = None,
        tools: list[Any] | None = None,
    ) -> list[ModelInfo]:
        """Keep only models that have the required capabilities."""
        result: list[ModelInfo] = []
        for m in models:
            if schema is not None and not m.supports_json_mode:
                continue
            if tools and not m.supports_tool_calling:
                continue
            result.append(m)
        return result

    def _filter_budget(
        self,
        models: Sequence[ModelInfo],
        estimated_input: int,
        budget: BudgetConstraint,
    ) -> list[ModelInfo]:
        """Keep only models whose estimated cost fits the remaining budget."""
        return [
            m for m in models
            if self._calculator.fits_budget(
                m, estimated_input, estimated_input, budget,
            )
        ]

    # -- event helpers ------------------------------------------------------

    def _emit(
        self,
        event_type: EventType,
        *,
        pipeline_id: str,
        step_id: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        if self._events is None:
            return
        event = Event(
            event_type=event_type,
            pipeline_id=pipeline_id,
            step_id=step_id,
            data=data or {},
        )
        self._events.append(event)
