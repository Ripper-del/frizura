"""Frizura Example 01: Standalone tasks with decorators.

Demonstrates how to use the `@task` decorator to define standalone LLM tasks.
Frizura automatically creates a single-step pipeline under the hood.
"""

from __future__ import annotations

import asyncio
import os
from dotenv import load_dotenv
import frizura

# Load API keys from .env file
load_dotenv()

# Verify that at least OpenAI API key is set for cloud, or we can use mock
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "mock-key-for-example"
    print("OPENAI_API_KEY not found. Using mock/test settings.")


@frizura.task(
    model="openai:gpt-4o-mini",
    system_prompt="You are a poet that writes short 2-line rhymes.",
)
async def write_rhyme(topic: str) -> str:
    """Write a short rhyme about {topic}."""
    # The docstring is treated as a prompt template and formatted with arguments!
    pass


@frizura.task(
    model="openai:gpt-4o-mini",
)
def write_rhyme_sync(topic: str) -> str:
    """Write a short rhyme about {topic} in sync mode."""
    pass


async def main() -> None:
    # 1. Run async task
    print("--- Running Async Task ---")
    rhyme = await write_rhyme("space exploration")
    print(f"Result:\n{rhyme}\n")


def main_sync() -> None:
    # 2. Run sync task
    print("--- Running Sync Task ---")
    rhyme_sync = write_rhyme_sync("cats")
    print(f"Result:\n{rhyme_sync}\n")


if __name__ == "__main__":
    # If running with mock keys, register the mock provider so it doesn't crash on connection error
    if os.environ.get("OPENAI_API_KEY") == "mock-key-for-example":
        from frizura.providers.base import LLMProvider
        from frizura.models.providers import ModelInfo
        from frizura.providers.registry import ModelRegistry
        from decimal import Decimal
        
        class SimpleMockProvider(LLMProvider):
            def __init__(self) -> None:
                super().__init__(model_info=self.model_info)
            async def _do_complete(self, messages, config):
                from frizura.models.execution import LLMResponse, TokenUsage
                return LLMResponse(
                    content="Rhyme: Stars shine bright in deep dark space,\nExploring is a human race.",
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

    asyncio.run(main())
    main_sync()
