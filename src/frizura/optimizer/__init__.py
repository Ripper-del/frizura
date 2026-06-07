"""Self-optimization package — DSPy-inspired prompt optimization."""

from frizura.optimizer.collector import FeedbackCollector, TrainingExample
from frizura.optimizer.optimizer import PromptOptimizer, OptimizationResult, CandidatePrompt
from frizura.optimizer.evaluator import ABEvaluator, ComparisonResult, EvalResult, ExampleResult

__all__ = [
    "FeedbackCollector",
    "TrainingExample",
    "PromptOptimizer",
    "OptimizationResult",
    "CandidatePrompt",
    "ABEvaluator",
    "ComparisonResult",
    "EvalResult",
    "ExampleResult",
]
