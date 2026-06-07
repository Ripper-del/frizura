"""A/B evaluator — compare performance of different prompt templates on test data.

Runs A/B tests between two prompts on a test dataset to measure which version is
superior based on scores and latency.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable
from pydantic import BaseModel, Field

from frizura.models.execution import Message, MessageRole
from frizura.models.providers import CompletionConfig
from frizura.optimizer.collector import TrainingExample

logger = logging.getLogger(__name__)


class ExampleResult(BaseModel):
    """Evaluation result for a single test example."""

    input_data: str
    expected: str | None = None
    output_a: str
    output_b: str
    score_a: float
    score_b: float
    latency_a_ms: float
    latency_b_ms: float


class ComparisonResult(BaseModel):
    """Result of an A/B prompt comparison test."""

    winner: str  # "a", "b", or "tie"
    score_a: float
    score_b: float
    avg_latency_a_ms: float
    avg_latency_b_ms: float
    per_example: list[ExampleResult] = Field(default_factory=list)
    confidence: float = 1.0


class EvalResult(BaseModel):
    """Evaluation result for a single prompt over a dataset."""

    prompt: str
    avg_score: float
    avg_latency_ms: float
    outputs: list[str] = Field(default_factory=list)
    scores: list[float] = Field(default_factory=list)


class ABEvaluator:
    """Evaluates and compares different prompt templates."""

    def __init__(self, provider: Any, model: str) -> None:
        self.provider = provider
        self.model = model

    async def evaluate_single(
        self,
        prompt_template: str,
        test_data: list[TrainingExample],
        metric_fn: Callable[[str, TrainingExample], float | bool],
    ) -> EvalResult:
        """Run a single prompt template against a test dataset."""
        scores = []
        latencies = []
        outputs = []

        for ex in test_data:
            formatted = prompt_template
            if "{input}" in prompt_template:
                formatted = prompt_template.format(input=ex.input_data)
            elif "{text}" in prompt_template:
                formatted = prompt_template.format(text=ex.input_data)
            else:
                formatted = f"{prompt_template}\n\nInput:\n{ex.input_data}"

            messages = [Message(role=MessageRole.USER, content=formatted)]
            config = CompletionConfig(temperature=0.0)
            
            t0 = time.perf_counter()
            try:
                resp = await self.provider.complete(messages, config)
                dt = (time.perf_counter() - t0) * 1000
                output_content = resp.content
                score_val = metric_fn(output_content, ex)
                score = 1.0 if score_val is True else (0.0 if score_val is False else float(score_val))
                scores.append(score)
                outputs.append(output_content)
                latencies.append(dt)
            except Exception as exc:
                logger.warning("Eval failed: %s", exc)
                scores.append(0.0)
                outputs.append(f"ERROR: {str(exc)}")
                latencies.append(0.0)

        avg_score = sum(scores) / len(scores) if scores else 0.0
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        return EvalResult(
            prompt=prompt_template,
            avg_score=avg_score,
            avg_latency_ms=avg_latency,
            outputs=outputs,
            scores=scores,
        )

    async def compare(
        self,
        prompt_a: str,
        prompt_b: str,
        test_data: list[TrainingExample],
        metric_fn: Callable[[str, TrainingExample], float | bool],
    ) -> ComparisonResult:
        """Compare prompt A vs prompt B side-by-side."""
        res_a = await self.evaluate_single(prompt_a, test_data, metric_fn)
        res_b = await self.evaluate_single(prompt_b, test_data, metric_fn)

        per_example = []
        for i, ex in enumerate(test_data):
            per_example.append(
                ExampleResult(
                    input_data=ex.input_data,
                    expected=ex.expected_output,
                    output_a=res_a.outputs[i] if i < len(res_a.outputs) else "",
                    output_b=res_b.outputs[i] if i < len(res_b.outputs) else "",
                    score_a=res_a.scores[i] if i < len(res_a.scores) else 0.0,
                    score_b=res_b.scores[i] if i < len(res_b.scores) else 0.0,
                    latency_a_ms=res_a.avg_latency_ms,  # Approximate
                    latency_b_ms=res_b.avg_latency_ms,
                )
            )

        if res_a.avg_score > res_b.avg_score:
            winner = "a"
        elif res_b.avg_score > res_a.avg_score:
            winner = "b"
        else:
            winner = "tie"

        return ComparisonResult(
            winner=winner,
            score_a=res_a.avg_score,
            score_b=res_b.avg_score,
            avg_latency_a_ms=res_a.avg_latency_ms,
            avg_latency_b_ms=res_b.avg_latency_ms,
            per_example=per_example,
            confidence=1.0,
        )
