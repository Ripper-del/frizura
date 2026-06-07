"""Cost calculator for model routing decisions.

Uses :pyclass:`decimal.Decimal` for all monetary arithmetic so that budget
comparisons stay exact.  Provides helpers for estimation, tracking, budget
checks, and sorting models by cost.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import TYPE_CHECKING

from frizura.models.budget import BudgetConstraint, CostEntry
from frizura.models.execution import LLMResponse
from frizura.models.providers import ModelInfo

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


class CostCalculator:
    """Stateless helper that computes / tracks token-level costs.

    All monetary values use :class:`decimal.Decimal` to avoid floating-point
    drift that could let a pipeline silently exceed its budget.
    """

    # -- estimation ---------------------------------------------------------

    @staticmethod
    def estimate(
        model_info: ModelInfo,
        input_tokens: int,
        output_tokens: int,
    ) -> Decimal:
        """Return the estimated cost (USD) for the given token counts.

        Delegates to :meth:`ModelInfo.estimate_cost` which already uses
        ``Decimal`` arithmetic.
        """
        return model_info.estimate_cost(input_tokens, output_tokens)

    # -- tracking -----------------------------------------------------------

    @staticmethod
    def track(
        model_info: ModelInfo,
        response: LLMResponse,
        *,
        step_id: str | None = None,
        pipeline_id: str | None = None,
    ) -> CostEntry:
        """Create a :class:`CostEntry` from a completed LLM response.

        This is meant to be called *after* an LLM call completes so that the
        actual token usage (not an estimate) is recorded.
        """
        cost = model_info.estimate_cost(
            response.usage.input_tokens,
            response.usage.output_tokens,
        )
        entry = CostEntry(
            model=model_info.model_id,
            provider=model_info.provider,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cost_usd=cost,
            timestamp=datetime.now(timezone.utc),
            step_id=step_id,
            pipeline_id=pipeline_id,
        )
        logger.debug(
            "Tracked cost: model=%s cost=$%s in=%d out=%d",
            model_info.model_id,
            cost,
            response.usage.input_tokens,
            response.usage.output_tokens,
        )
        return entry

    # -- budget helpers -----------------------------------------------------

    @staticmethod
    def fits_budget(
        model_info: ModelInfo,
        estimated_input: int,
        estimated_output: int,
        constraint: BudgetConstraint,
    ) -> bool:
        """Return *True* if the *estimated* call fits within ``constraint``.

        Checks both cost and token limits (whichever are set).
        """
        # Token check
        total_tokens = estimated_input + estimated_output
        remaining_tokens = constraint.remaining_tokens
        if remaining_tokens is not None and total_tokens > remaining_tokens:
            return False

        # Cost check
        remaining_cost = constraint.remaining_cost
        if remaining_cost is not None:
            est_cost = model_info.estimate_cost(estimated_input, estimated_output)
            if est_cost > remaining_cost:
                return False

        return True

    @staticmethod
    def cheapest_models(
        models: Sequence[ModelInfo],
        input_tokens: int,
        output_tokens: int,
        budget: BudgetConstraint | None = None,
    ) -> list[ModelInfo]:
        """Return *models* sorted cheapest-first, optionally filtered to those
        that still fit within *budget*.

        Parameters
        ----------
        models:
            Candidate models (e.g. from a registry).
        input_tokens / output_tokens:
            Estimated token counts used for cost comparison.
        budget:
            If provided, models whose estimated cost exceeds the remaining
            budget are excluded.

        Returns
        -------
        list[ModelInfo]
            Sorted list (cheapest first).  May be empty if nothing fits.
        """
        candidates: list[tuple[Decimal, ModelInfo]] = []

        for model in models:
            est = model.estimate_cost(input_tokens, output_tokens)

            if budget is not None:
                remaining = budget.remaining_cost
                if remaining is not None and est > remaining:
                    continue
                remaining_tok = budget.remaining_tokens
                if (
                    remaining_tok is not None
                    and (input_tokens + output_tokens) > remaining_tok
                ):
                    continue

            candidates.append((est, model))

        # Stable sort by cost
        candidates.sort(key=lambda pair: pair[0])

        return [model for _, model in candidates]
