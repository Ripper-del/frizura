"""Prompt optimizer — DSPy-inspired automatic prompt generation and optimization.

Evaluates multiple candidate prompt templates against a training dataset and selects
the best-performing variant based on a metric.
"""

from __future__ import annotations

import logging
from typing import Any, Callable
from pydantic import BaseModel, Field

from frizura.models.execution import Message, MessageRole
from frizura.models.providers import CompletionConfig
from frizura.optimizer.collector import TrainingExample

logger = logging.getLogger(__name__)


class CandidatePrompt(BaseModel):
    """A candidate prompt variant and its evaluation score."""

    prompt: str
    score: float
    outputs: list[str] = Field(default_factory=list)


class OptimizationResult(BaseModel):
    """Result of prompt optimization."""

    best_prompt: str
    score: float
    all_candidates: list[CandidatePrompt] = Field(default_factory=list)
    improvement_pct: float = 0.0


class PromptOptimizer:
    """Optimises prompt templates by generating and evaluating variants."""

    def __init__(self, provider: Any, model: str) -> None:
        self.provider = provider
        self.model = model

    async def generate_variants(self, current_prompt: str, training_data: list[TrainingExample], n: int = 5) -> list[str]:
        """Generate N improved prompt variants using the LLM."""
        examples_str = ""
        for i, ex in enumerate(training_data[:3]):
            examples_str += f"Example {i+1}:\nInput: {ex.input_data}\nExpected Output: {ex.expected_output or 'N/A'}\n"
            if ex.feedback:
                examples_str += f"Feedback: {ex.feedback}\n"
            examples_str += "\n"

        system_msg = Message(
            role=MessageRole.SYSTEM,
            content=(
                "You are an expert prompt engineer. Your job is to improve a given "
                "prompt template to make the LLM produce better outputs. You will "
                "be given the current prompt, some training examples, and feedback."
            ),
        )

        user_msg = Message(
            role=MessageRole.USER,
            content=(
                f"Current prompt template:\n```\n{current_prompt}\n```\n\n"
                f"Training dataset examples:\n{examples_str}"
                f"Please generate {n} distinct and improved variations of the current prompt template. "
                "Focus on clarity, formatting instructions, and handling edge cases. "
                "Provide your response as a JSON list of strings, like this:\n"
                '["variant 1", "variant 2", ...]\n'
                "Output ONLY the JSON list, nothing else."
            ),
        )

        config = CompletionConfig(temperature=0.7, json_mode=True)
        try:
            resp = await self.provider.complete([system_msg, user_msg], config)
            # Parse response
            import json
            from frizura.schema.validator import extract_json
            extracted = extract_json(resp.content)
            if extracted:
                variants = json.loads(extracted)
                if isinstance(variants, list) and all(isinstance(v, str) for v in variants):
                    # Ensure original is included
                    if current_prompt not in variants:
                        variants.append(current_prompt)
                    return variants
        except Exception as exc:
            logger.error("Failed to generate prompt variants: %s", exc)
        
        # Fallback to simple variations if parsing failed
        return [
            current_prompt,
            current_prompt + "\nBe concise and exact.",
            current_prompt + "\nReason step-by-step before answering.",
        ]

    async def evaluate_prompt(
        self,
        prompt_template: str,
        training_data: list[TrainingExample],
        metric_fn: Callable[[str, TrainingExample], float | bool],
    ) -> CandidatePrompt:
        """Evaluate a single prompt template against all training examples."""
        scores = []
        outputs = []

        for ex in training_data:
            # Format prompt with input data
            # Assume prompt has '{input}' or we format it directly
            formatted = prompt_template
            if "{input}" in prompt_template:
                formatted = prompt_template.format(input=ex.input_data)
            elif "{text}" in prompt_template:
                formatted = prompt_template.format(text=ex.input_data)
            else:
                formatted = f"{prompt_template}\n\nInput:\n{ex.input_data}"

            messages = [Message(role=MessageRole.USER, content=formatted)]
            config = CompletionConfig(temperature=0.0)  # Deterministic evaluation
            
            try:
                resp = await self.provider.complete(messages, config)
                output_content = resp.content
                score_val = metric_fn(output_content, ex)
                # Convert bool to float
                score = 1.0 if score_val is True else (0.0 if score_val is False else float(score_val))
                scores.append(score)
                outputs.append(output_content)
            except Exception as exc:
                logger.warning("Failed evaluation on example: %s", exc)
                scores.append(0.0)
                outputs.append(f"ERROR: {str(exc)}")

        avg_score = sum(scores) / len(scores) if scores else 0.0
        return CandidatePrompt(prompt=prompt_template, score=avg_score, outputs=outputs)

    async def optimize(
        self,
        current_prompt: str,
        training_data: list[TrainingExample],
        metric_fn: Callable[[str, TrainingExample], float | bool],
        n_candidates: int = 5,
    ) -> OptimizationResult:
        """Run the full prompt optimization loop."""
        if not training_data:
            return OptimizationResult(best_prompt=current_prompt, score=0.0)

        # 1. Generate variant prompts
        variants = await self.generate_variants(current_prompt, training_data, n=n_candidates)
        
        # 2. Evaluate all candidates
        candidates = []
        for var in variants:
            logger.info("Evaluating prompt candidate: %s...", var[:50].replace("\n", " "))
            cand = await self.evaluate_prompt(var, training_data, metric_fn)
            candidates.append(cand)

        # 3. Find the best candidate
        candidates.sort(key=lambda c: c.score, reverse=True)
        best = candidates[0]
        
        # 4. Calculate improvement
        original_score = 0.0
        for cand in candidates:
            if cand.prompt == current_prompt:
                original_score = cand.score
                break

        improvement = 0.0
        if original_score > 0:
            improvement = ((best.score - original_score) / original_score) * 100.0
        elif best.score > 0:
            improvement = 100.0  # From 0 to positive

        return OptimizationResult(
            best_prompt=best.prompt,
            score=best.score,
            all_candidates=candidates,
            improvement_pct=improvement,
        )
