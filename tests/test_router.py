"""Unit tests for the Frizura smart router."""

from __future__ import annotations

from decimal import Decimal
import pytest

from frizura.models.budget import Budget, BudgetConstraint
from frizura.models.providers import ModelInfo
from frizura.router.analyzer import ComplexityAnalyzer
from frizura.router.calculator import CostCalculator
from frizura.router.router import SmartRouter, RouterConfig
from frizura.providers.registry import ModelRegistry


def test_complexity_analyzer() -> None:
    """Test complexity analyzer rules and score calculations."""
    analyzer = ComplexityAnalyzer()
    
    # Simple prompt
    res1 = analyzer.analyze("hello")
    assert res1.score < 0.3
    assert res1.recommended_tier == "cheap"
    
    # Complex prompt (reasoning, codes, length)
    complex_prompt = (
        "Evaluate and analyze the performance of the following sorting algorithm. "
        "Explain step-by-step how to optimize the memory footprint. Provide Python code."
    )
    res2 = analyzer.analyze(complex_prompt)
    assert res2.score > res1.score
    assert "reasoning_complexity" in res2.factors


def test_cost_calculator() -> None:
    """Test cost estimation and budget fit verification."""
    calc = CostCalculator()
    
    model = ModelInfo(
        model_id="premium-model",
        provider="mock",
        display_name="Premium Model",
        context_window=100000,
        max_output_tokens=4096,
        input_price_per_1m=Decimal("10.0"),  # $10 per 1M tokens
        output_price_per_1m=Decimal("30.0"), # $30 per 1M tokens
        is_local=False,
    )
    
    # Cost estimation
    # 1000 input tokens = $0.01. 2000 output tokens = $0.06. Total = $0.07
    cost = calc.estimate(model, 1000, 2000)
    assert cost == Decimal("0.07")
    
    # Budget fit
    budget = Budget(max_cost=0.05)
    constraint = BudgetConstraint(budget=budget)
    
    assert not calc.fits_budget(model, 1000, 2000, constraint)
    
    cheap_budget = Budget(max_cost=0.10)
    cheap_constraint = BudgetConstraint(budget=cheap_budget)
    assert calc.fits_budget(model, 1000, 2000, cheap_constraint)


@pytest.mark.asyncio
async def test_smart_router_decision(register_mock_provider) -> None:
    """Test router selecting the best model based on budget preference."""
    registry = ModelRegistry()
    registry._models.clear()
    registry._providers.clear()
    
    # Register a cheap model and a premium model
    cheap_model = ModelInfo(
        model_id="cheap-model",
        provider="mock",
        display_name="Cheap Model",
        context_window=8000,
        max_output_tokens=2048,
        input_price_per_1m=Decimal("0.5"),
        output_price_per_1m=Decimal("1.5"),
        tier="cheap",
    )
    premium_model = ModelInfo(
        model_id="premium-model",
        provider="mock",
        display_name="Premium Model",
        context_window=100000,
        max_output_tokens=4096,
        input_price_per_1m=Decimal("10.0"),
        output_price_per_1m=Decimal("30.0"),
        tier="premium",
    )
    
    # Register mock providers for them
    from .conftest import MockLLMProvider
    prov_cheap = MockLLMProvider("cheap-model")
    prov_premium = MockLLMProvider("premium-model")
    registry.register(cheap_model)
    registry.register(premium_model)
    registry._providers["mock:cheap-model"] = prov_cheap
    registry._providers["mock:premium-model"] = prov_premium
    
    router = SmartRouter(registry=registry, config=RouterConfig())
    
    # Case 1: Low budget max_cost = $0.001 -> must pick cheap model
    budget1 = BudgetConstraint(budget=Budget(max_cost=0.001, prefer="cost"))
    decision1 = await router.route("Summarize this text.", budget=budget1)
    assert decision1.model_info.model_id == "cheap-model"
    
    # Case 2: Large budget + Quality preference -> should pick premium model
    budget2 = BudgetConstraint(budget=Budget(max_cost=1.0, prefer="quality"))
    decision2 = await router.route("Analyze this complex code.", budget=budget2)
    assert decision2.model_info.model_id == "premium-model"
