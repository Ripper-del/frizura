"""Budget and cost tracking models."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, Field


class Budget(BaseModel):
    """User-specified budget constraints for a task or pipeline."""

    max_cost: float | None = None  # Maximum cost in USD
    max_time: float | None = None  # Maximum execution time in seconds
    max_tokens: int | None = None  # Maximum total tokens
    max_retries: int = 3  # Maximum retry attempts (for schema healing, etc.)
    prefer: str = "cost"  # "cost", "speed", "quality"


class BudgetConstraint(BaseModel):
    """Internal budget tracker that decrements as resources are consumed."""

    budget: Budget
    spent_cost: Decimal = Decimal("0")
    spent_tokens: int = 0
    elapsed_time: float = 0.0
    retries_used: int = 0

    @property
    def remaining_cost(self) -> Decimal | None:
        if self.budget.max_cost is None:
            return None
        return Decimal(str(self.budget.max_cost)) - self.spent_cost

    @property
    def remaining_tokens(self) -> int | None:
        if self.budget.max_tokens is None:
            return None
        return self.budget.max_tokens - self.spent_tokens

    @property
    def remaining_time(self) -> float | None:
        if self.budget.max_time is None:
            return None
        return self.budget.max_time - self.elapsed_time

    @property
    def is_exhausted(self) -> bool:
        """Check if any budget limit has been exceeded."""
        if self.remaining_cost is not None and self.remaining_cost <= 0:
            return True
        if self.remaining_tokens is not None and self.remaining_tokens <= 0:
            return True
        if self.remaining_time is not None and self.remaining_time <= 0:
            return True
        if self.retries_used >= self.budget.max_retries:
            return True
        return False

    def consume(
        self,
        cost: Decimal = Decimal("0"),
        tokens: int = 0,
        time: float = 0.0,
    ) -> None:
        """Record resource consumption."""
        self.spent_cost += cost
        self.spent_tokens += tokens
        self.elapsed_time += time

    def consume_retry(self) -> None:
        """Record a retry attempt."""
        self.retries_used += 1


class CostEntry(BaseModel):
    """A single cost tracking entry."""

    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    cost_usd: Decimal
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    step_id: str | None = None
    pipeline_id: str | None = None


class CostReport(BaseModel):
    """Aggregated cost report."""

    entries: list[CostEntry] = Field(default_factory=list)
    total_cost_usd: Decimal = Decimal("0")
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    by_model: dict[str, Decimal] = Field(default_factory=dict)
    by_provider: dict[str, Decimal] = Field(default_factory=dict)
