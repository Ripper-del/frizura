"""Routing strategies for model selection.

Each strategy implements the same interface: given a list of candidate models,
a complexity analysis, and an optional budget constraint, select the single
best model.

Strategies can be composed — for example :class:`CascadeStrategy` iterates
through tiers from cheap to premium, while :class:`CheapestStrategy` always
picks the lowest-cost option.
"""

from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from decimal import Decimal
from enum import StrEnum

from frizura.models.budget import BudgetConstraint
from frizura.models.providers import ModelInfo
from frizura.router.analyzer import TaskComplexity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Strategy enum
# ---------------------------------------------------------------------------

class RoutingStrategy(StrEnum):
    """Available built-in routing strategies."""

    CASCADE = "cascade"
    CHEAPEST = "cheapest"
    FASTEST = "fastest"
    BEST_QUALITY = "best_quality"
    ROUND_ROBIN = "round_robin"


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseStrategy(ABC):
    """Base class that all routing strategies implement."""

    @abstractmethod
    def select(
        self,
        candidates: list[ModelInfo],
        complexity: TaskComplexity,
        budget: BudgetConstraint | None = None,
    ) -> ModelInfo | None:
        """Pick the best model from *candidates*, or ``None`` if no model
        is suitable.

        Implementations may assume that *candidates* have already been
        filtered for hard capability requirements (e.g. tool_calling).
        """


# ---------------------------------------------------------------------------
# Concrete strategies
# ---------------------------------------------------------------------------

_TIER_ORDER = {"cheap": 0, "standard": 1, "premium": 2}
_TIER_ORDER_REV = {"premium": 0, "standard": 1, "cheap": 2}


class CascadeStrategy(BaseStrategy):
    """Try the cheapest tier that matches the complexity, then escalate.

    Order: cheap → standard → premium (for low-complexity tasks the first
    tier tried is cheap; for high-complexity tasks it's premium).
    """

    def select(
        self,
        candidates: list[ModelInfo],
        complexity: TaskComplexity,
        budget: BudgetConstraint | None = None,
    ) -> ModelInfo | None:
        recommended = complexity.recommended_tier

        # Build tier groups, sorted cheapest-first within each tier
        tiers: dict[str, list[ModelInfo]] = {"cheap": [], "standard": [], "premium": []}
        for m in candidates:
            tier_key = m.tier if m.tier in tiers else "standard"
            tiers[tier_key].append(m)

        # Sort each tier by cost (input_price as proxy)
        for group in tiers.values():
            group.sort(key=lambda m: m.input_price_per_1m)

        # Determine tier order starting from the recommended one
        tier_names = ["cheap", "standard", "premium"]
        start_idx = tier_names.index(recommended) if recommended in tier_names else 0
        ordered = tier_names[start_idx:] + tier_names[:start_idx]

        for tier_name in ordered:
            for model in tiers[tier_name]:
                if self._fits_budget(model, complexity, budget):
                    logger.debug(
                        "Cascade selected %s (tier=%s)", model.model_id, tier_name,
                    )
                    return model

        # Fallback: return anything
        return candidates[0] if candidates else None

    @staticmethod
    def _fits_budget(
        model: ModelInfo,
        complexity: TaskComplexity,
        budget: BudgetConstraint | None,
    ) -> bool:
        if budget is None:
            return True
        est = model.estimate_cost(
            complexity.estimated_tokens,
            complexity.estimated_tokens,  # rough output estimate = same as input
        )
        remaining = budget.remaining_cost
        if remaining is not None and est > remaining:
            return False
        remaining_tok = budget.remaining_tokens
        if remaining_tok is not None and complexity.estimated_tokens * 2 > remaining_tok:
            return False
        return True


class CheapestStrategy(BaseStrategy):
    """Always pick the cheapest model that fits the budget."""

    def select(
        self,
        candidates: list[ModelInfo],
        complexity: TaskComplexity,
        budget: BudgetConstraint | None = None,
    ) -> ModelInfo | None:
        if not candidates:
            return None

        sorted_models = sorted(
            candidates,
            key=lambda m: m.input_price_per_1m + m.output_price_per_1m,
        )

        if budget is None:
            return sorted_models[0]

        for model in sorted_models:
            est = model.estimate_cost(
                complexity.estimated_tokens,
                complexity.estimated_tokens,
            )
            remaining = budget.remaining_cost
            if remaining is None or est <= remaining:
                return model

        # If nothing fits, return cheapest anyway (caller decides)
        return sorted_models[0]


class FastestStrategy(BaseStrategy):
    """Pick the model expected to have the lowest latency.

    Heuristic: local models are fastest, then cheap cloud tiers (smaller
    models tend to be faster), then premium.
    """

    def select(
        self,
        candidates: list[ModelInfo],
        complexity: TaskComplexity,
        budget: BudgetConstraint | None = None,
    ) -> ModelInfo | None:
        if not candidates:
            return None

        def _speed_key(m: ModelInfo) -> tuple[int, Decimal]:
            # Local models are fastest (0), then cheap(1), standard(2), premium(3)
            locality = 0 if m.is_local else 1
            tier_rank = _TIER_ORDER.get(m.tier, 1)
            return (locality, Decimal(tier_rank))

        sorted_models = sorted(candidates, key=_speed_key)
        return sorted_models[0]


class BestQualityStrategy(BaseStrategy):
    """Always pick the highest-tier (premium) model available."""

    def select(
        self,
        candidates: list[ModelInfo],
        complexity: TaskComplexity,
        budget: BudgetConstraint | None = None,
    ) -> ModelInfo | None:
        if not candidates:
            return None

        sorted_models = sorted(
            candidates,
            key=lambda m: _TIER_ORDER_REV.get(m.tier, 1),
        )
        return sorted_models[0]


class RoundRobinStrategy(BaseStrategy):
    """Rotate through candidates for load distribution.

    Maintains a simple counter internally. Not thread-safe — fine for the
    single-event-loop asyncio model Frizura uses.
    """

    def __init__(self) -> None:
        self._counter: int = 0

    def select(
        self,
        candidates: list[ModelInfo],
        complexity: TaskComplexity,
        budget: BudgetConstraint | None = None,
    ) -> ModelInfo | None:
        if not candidates:
            return None

        # Deterministic ordering so the round-robin is stable
        ordered = sorted(candidates, key=lambda m: m.model_id)
        idx = self._counter % len(ordered)
        self._counter += 1
        return ordered[idx]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_STRATEGY_MAP: dict[str, type[BaseStrategy]] = {
    RoutingStrategy.CASCADE: CascadeStrategy,
    RoutingStrategy.CHEAPEST: CheapestStrategy,
    RoutingStrategy.FASTEST: FastestStrategy,
    RoutingStrategy.BEST_QUALITY: BestQualityStrategy,
    RoutingStrategy.ROUND_ROBIN: RoundRobinStrategy,
}

# Keep singleton round-robin so counter persists across calls
_round_robin_instance = RoundRobinStrategy()


def get_strategy(name: str) -> BaseStrategy:
    """Return a strategy instance by name (see :class:`RoutingStrategy`).

    The :class:`RoundRobinStrategy` is a shared singleton so that its counter
    persists across routing calls.
    """
    if name == RoutingStrategy.ROUND_ROBIN:
        return _round_robin_instance

    cls = _STRATEGY_MAP.get(name)
    if cls is None:
        logger.warning("Unknown strategy '%s', falling back to cascade", name)
        cls = CascadeStrategy
    return cls()
