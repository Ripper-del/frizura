"""Frizura Example 02: Structured Output & Schema Healing.

Demonstrates how to enforce structured JSON output matching a Pydantic model.
If the model output fails to validate, Frizura will automatically trigger a self-healing loop
to ask the LLM to fix the validation errors.
"""

from __future__ import annotations

import asyncio
import os
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import frizura

load_dotenv()


class MovieReview(BaseModel):
    title: str = Field(description="Title of the movie")
    sentiment: str = Field(description="Positive, negative, or neutral")
    rating: float = Field(description="Rating from 0.0 to 10.0")
    summary: str = Field(description="Short summary of the review")


@frizura.task(
    model="openai:gpt-4o-mini",
    output_schema=MovieReview,
)
async def extract_review(text: str) -> MovieReview:
    """Extract movie review details from the following text: {text}"""
    pass


async def main() -> None:
    # Example review text
    review_text = (
        "Just saw Inception again. What a masterpiece of cinema! The storytelling, "
        "acting, and soundtrack are all top notch. Easily a 9.5 out of 10 for me. "
        "Christopher Nolan does not disappoint."
    )

    print("Extracting structured review...")
    try:
        result = await extract_review(review_text)
        print("\n[bold green]Success![/bold] Structured output received:")
        print(f"Title: {result.title}")
        print(f"Sentiment: {result.sentiment}")
        print(f"Rating: {result.rating}/10.0")
        print(f"Summary: {result.summary}")
    except Exception as exc:
        print(f"Extraction failed: {exc}")


if __name__ == "__main__":
    # If running with mock keys, register the mock provider so it doesn't crash
    if not os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY") == "mock-key-for-example":
        os.environ["OPENAI_API_KEY"] = "mock-key-for-example"
        from frizura.providers.base import LLMProvider
        from frizura.models.providers import ModelInfo
        from frizura.providers.registry import ModelRegistry
        from decimal import Decimal
        
        class SchemaMockProvider(LLMProvider):
            def __init__(self) -> None:
                super().__init__(model_info=self.model_info)
                self.called = False
            async def _do_complete(self, messages, config):
                from frizura.models.execution import LLMResponse, TokenUsage
                # First response simulates invalid JSON (missing rating)
                # Second response (healing attempt) will be correct
                if self.called:
                    content = '{"title": "Inception", "sentiment": "Positive", "rating": 9.5, "summary": "A cinematic masterpiece."}'
                else:
                    self.called = True
                    content = '{"title": "Inception", "sentiment": "Positive", "summary": "A cinematic masterpiece."}' # MISSING rating
                
                return LLMResponse(
                    content=content,
                    model="gpt-4o-mini",
                    provider="openai",
                    usage=TokenUsage(total_tokens=15),
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
        
        prov = SchemaMockProvider()
        registry = ModelRegistry()
        registry.register(prov.model_info)
        registry._providers["openai:gpt-4o-mini"] = prov

    asyncio.run(main())
