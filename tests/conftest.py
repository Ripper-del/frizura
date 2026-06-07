"""Shared fixtures for Frizura tests."""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import AsyncIterator
import pytest

from frizura.models.execution import LLMResponse, Message, MessageRole, TokenUsage
from frizura.models.providers import CompletionConfig, ModelInfo
from frizura.providers.base import LLMProvider
from frizura.providers.registry import ModelRegistry


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing."""

    def __init__(self, model_id: str = "mock-model") -> None:
        self._model_id = model_id
        super().__init__(model_info=self.model_info)
        self.responses: list[str] = ["Mock response"]
        self.call_count = 0

    async def _do_complete(
        self,
        messages: list[Message],
        config: CompletionConfig,
    ) -> LLMResponse:
        self.call_count += 1
        # Pick the response or fallback to default
        idx = min(self.call_count - 1, len(self.responses) - 1)
        content = self.responses[idx]
        
        # If schema is expected, return mock JSON (unless custom responses were provided)
        if (config.json_mode or config.json_schema) and self.responses == ["Mock response"]:
            # Simple heuristic: if schema name has MovieReview, return valid JSON
            if "MovieReview" in str(config.json_schema):
                content = '{"title": "Inception", "sentiment": "positive", "rating": 9.5, "summary": "Great movie"}'
            elif "Sentiment" in str(config.json_schema):
                content = '{"sentiment": "positive", "score": 0.95}'
            else:
                content = '{"status": "ok", "value": "mocked"}'

        return LLMResponse(
            content=content,
            model=self._model_id,
            provider="mock",
            usage=TokenUsage(input_tokens=10, output_tokens=20, total_tokens=30),
            finish_reason="stop",
            latency_ms=15.0,
        )

    async def _do_stream(
        self,
        messages: list[Message],
        config: CompletionConfig,
    ) -> AsyncIterator[Any]:
        yield "chunk 1"
        yield "chunk 2"

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> Decimal:
        return Decimal("0.0001")

    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo(
            model_id=self._model_id,
            provider="mock",
            display_name="Mock Model",
            context_window=8192,
            max_output_tokens=2048,
            input_price_per_1m=Decimal("1.0"),
            output_price_per_1m=Decimal("2.0"),
            supports_json_mode=True,
            supports_tool_calling=True,
            is_local=False,
            tier="cheap",
        )

    async def _do_healthcheck(self) -> bool:
        return True


@pytest.fixture
def mock_provider() -> MockLLMProvider:
    """Fixture providing a mock LLM provider."""
    return MockLLMProvider()


@pytest.fixture(autouse=True)
def register_mock_provider(mock_provider) -> None:
    """Auto-register mock provider in registry."""
    registry = ModelRegistry()
    
    # 1. Register "mock-model" ModelInfo
    info1 = mock_provider.model_info
    registry.register(info1)
    
    # 2. Register "model" ModelInfo
    info2 = ModelInfo(
        model_id="model",
        provider="mock",
        display_name="Mock Model Generic",
        context_window=8192,
        max_output_tokens=2048,
        input_price_per_1m=Decimal("1.0"),
        output_price_per_1m=Decimal("2.0"),
        supports_json_mode=True,
        supports_tool_calling=True,
        is_local=False,
        tier="cheap",
    )
    registry.register(info2)

    # 3. Cache provider instances
    registry._providers["mock:mock-model"] = mock_provider
    registry._providers["mock-model"] = mock_provider
    registry._providers["mock:model"] = mock_provider
    registry._providers["model"] = mock_provider
    
    # Also register default catalog models so the router test can look them up
    from frizura.providers.registry import DEFAULT_MODELS
    registry.register_many(DEFAULT_MODELS)

