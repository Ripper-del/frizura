"""Unit tests for the Frizura execution engine."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from frizura.core.context import ExecutionContext
from frizura.core.engine import FrizuraEngine
from frizura.core.graph import Pipeline, Step, StepType
from frizura.models.budget import Budget


class SentimentResult(BaseModel):
    sentiment: str
    score: float


@pytest.mark.asyncio
async def test_engine_simple_run(register_mock_provider) -> None:
    """Test running a simple single-step pipeline."""
    engine = FrizuraEngine()
    
    # Create simple pipeline
    pipeline = Pipeline("test-pipeline").add_step(
        Step(
            name="step-1",
            system_prompt="You are a helper.",
            handler=lambda ctx: "Hello, model!",
            model="mock:model",
        )
    )

    result = await engine.run(pipeline, input_data="test input")
    
    assert result.status == "completed"
    assert len(result.steps) == 1
    assert result.steps[0].status == "completed"
    assert result.steps[0].step_name == "step-1"
    assert result.steps[0].llm_response is not None
    assert "Mock response" in result.steps[0].llm_response.content


@pytest.mark.asyncio
async def test_engine_with_transform(register_mock_provider) -> None:
    """Test a pipeline with an LLM step followed by a TRANSFORM step."""
    engine = FrizuraEngine()

    def transform_fn(ctx: ExecutionContext) -> str:
        # Read the output of the first step
        first_step_out = ctx.get("step-1")
        return f"Transformed: {first_step_out}"

    pipeline = (
        Pipeline("transform-pipeline")
        .add_step(
            Step(
                name="step-1",
                model="mock:model",
                handler=lambda ctx: "Hello",
            )
        )
        .add_step(
            Step(
                name="step-2",
                step_type=StepType.TRANSFORM,
                handler=transform_fn,
            )
        )
    )

    result = await engine.run(pipeline, input_data="input")
    
    assert result.status == "completed"
    assert len(result.steps) == 2
    assert result.steps[0].status == "completed"
    assert result.steps[1].status == "completed"
    assert result.steps[1].output == "Transformed: Mock response"


@pytest.mark.asyncio
async def test_engine_structured_output(mock_provider, register_mock_provider) -> None:
    """Test schema validation via LLM step."""
    engine = FrizuraEngine()
    
    # Pre-populate provider with valid JSON for SentimentResult
    mock_provider.responses = ['{"sentiment": "positive", "score": 0.99}']

    pipeline = Pipeline("schema-pipeline").add_step(
        Step(
            name="sentiment-step",
            model="mock:model",
            output_schema=SentimentResult,
            handler=lambda ctx: "Analyze this",
        )
    )

    result = await engine.run(pipeline, input_data="input")
    
    assert result.status == "completed"
    assert len(result.steps) == 1
    assert isinstance(result.steps[0].output, SentimentResult)
    assert result.steps[0].output.sentiment == "positive"
    assert result.steps[0].output.score == 0.99
