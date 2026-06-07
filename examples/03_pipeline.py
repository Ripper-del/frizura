"""Frizura Example 03: Pipeline DAG builder.

Demonstrates how to build a multi-step pipeline using the Pipeline DAG builder
and register steps.
"""

from __future__ import annotations

import os
from decimal import Decimal
from frizura.core.graph import Pipeline, Step, StepType

# Register mock provider if running without real API keys
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "mock-key-for-example"
    from frizura.providers.base import LLMProvider
    from frizura.models.providers import ModelInfo
    from frizura.providers.registry import ModelRegistry

    class SimpleMockProvider(LLMProvider):
        def __init__(self) -> None:
            super().__init__(model_info=self.model_info)
        async def _do_complete(self, messages, config):
            from frizura.models.execution import LLMResponse, TokenUsage
            return LLMResponse(
                content="Mock response from comedy pipeline.",
                model="gpt-4o-mini",
                provider="openai",
                usage=TokenUsage(total_tokens=10),
            )
        async def _do_stream(self, messages, config):
            yield "mock"
        def estimate_cost(self, in_t, out_t):
            return Decimal("0")
        @property
        def model_info(self):
            return ModelInfo(model_id="gpt-4o-mini", provider="openai", input_price_per_1m=Decimal("0"), output_price_per_1m=Decimal("0"))
        async def _do_healthcheck(self):
            return True

    prov = SimpleMockProvider()
    registry = ModelRegistry()
    registry.register(prov.model_info)
    registry._providers["openai:gpt-4o-mini"] = prov
    registry._providers["mock:model"] = prov

# Step 1: Generate a short story
step1 = Step(
    name="joke-generator",
    system_prompt="You are a funny comedian.",
    handler=lambda ctx: "Tell me a short programming joke.",
    model="mock:model",
)

# Step 2: Rate/Critique the generated joke
step2 = Step(
    name="critic",
    system_prompt="You are a critical reviewer.",
    handler=lambda ctx: f"Provide a brief 1-line review of this joke: {ctx.get('joke-generator')}",
    model="mock:model",
)

# Build pipeline (adds steps sequentially)
pipeline = Pipeline("comedy-pipeline").add_step(step1).add_step(step2)
