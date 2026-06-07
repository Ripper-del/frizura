"""Complexity analyzer for smart routing decisions.

Analyzes prompts, schemas, and tool configurations to produce a complexity
score (0.0–1.0) that drives model tier selection.  The score is a weighted
combination of several independent factors so that even simple heuristics
compose into a useful signal.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Factor weights — must sum to 1.0
# ---------------------------------------------------------------------------
_WEIGHTS: dict[str, float] = {
    "prompt_length": 0.15,
    "reasoning_complexity": 0.25,
    "schema_complexity": 0.20,
    "code_generation": 0.15,
    "language_complexity": 0.10,
    "tool_usage": 0.15,
}

# ---------------------------------------------------------------------------
# Keyword lists used by individual factor extractors
# ---------------------------------------------------------------------------
_REASONING_KEYWORDS: list[str] = [
    "analyze", "analyse", "compare", "evaluate", "contrast", "synthesize",
    "summarize", "critique", "assess", "justify", "explain why",
    "step by step", "step-by-step", "reason", "trade-off", "tradeoff",
    "pros and cons", "implications", "infer", "deduce", "derive",
    "break down", "consider", "weigh", "differentiate", "prioritize",
    "categorize", "classify", "rank", "debate", "argue",
]

_CODE_KEYWORDS: list[str] = [
    "code", "function", "class", "implement", "algorithm", "program",
    "script", "debug", "refactor", "optimize", "compile", "runtime",
    "syntax", "api", "endpoint", "sql", "query", "regex", "html", "css",
    "python", "javascript", "typescript", "java", "rust", "go ", "golang",
    "c++", "cpp", "ruby", "swift", "kotlin", "bash", "shell",
]

# Rough heuristic for non-Latin script detection (Cyrillic, CJK, Arabic, …)
_NON_LATIN_RE = re.compile(
    r"[\u0400-\u04FF"       # Cyrillic
    r"\u4E00-\u9FFF"        # CJK Unified
    r"\u3040-\u309F"        # Hiragana
    r"\u30A0-\u30FF"        # Katakana
    r"\u0600-\u06FF"        # Arabic
    r"\u0590-\u05FF"        # Hebrew
    r"\uAC00-\uD7AF]",     # Hangul
)


# ---------------------------------------------------------------------------
# Public models
# ---------------------------------------------------------------------------

class TaskComplexity(BaseModel):
    """Result of a complexity analysis."""

    score: float = Field(ge=0.0, le=1.0, description="Overall complexity 0‥1")
    factors: dict[str, float] = Field(
        default_factory=dict,
        description="Individual factor scores (each 0‥1)",
    )
    recommended_tier: str = Field(
        description="'cheap', 'standard', or 'premium'",
    )
    estimated_tokens: int = Field(
        ge=0,
        description="Rough token estimate for the prompt",
    )


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class ComplexityAnalyzer:
    """Analyse a task to decide how complex (and therefore expensive) a model
    it needs.

    The analyser inspects the raw prompt text, the output schema (if any), and
    the available tools to compute a scalar complexity score together with the
    individual factor breakdown.
    """

    def __init__(
        self,
        *,
        threshold_cheap: float = 0.3,
        threshold_premium: float = 0.7,
        weights: dict[str, float] | None = None,
    ) -> None:
        self._threshold_cheap = threshold_cheap
        self._threshold_premium = threshold_premium
        self._weights = weights or dict(_WEIGHTS)

        # Normalise weights so they always sum to 1.0
        total = sum(self._weights.values())
        if total > 0:
            self._weights = {k: v / total for k, v in self._weights.items()}

    # -- public API ---------------------------------------------------------

    def analyze(
        self,
        prompt: str,
        schema: type[BaseModel] | None = None,
        tools: list[Any] | None = None,
    ) -> TaskComplexity:
        """Return a :class:`TaskComplexity` for the given task."""

        estimated_tokens = self._estimate_tokens(prompt)

        factors: dict[str, float] = {
            "prompt_length": self._factor_prompt_length(prompt, estimated_tokens),
            "reasoning_complexity": self._factor_reasoning(prompt),
            "schema_complexity": self._factor_schema(schema),
            "code_generation": self._factor_code(prompt),
            "language_complexity": self._factor_language(prompt),
            "tool_usage": self._factor_tools(tools),
        }

        score = sum(
            self._weights.get(name, 0.0) * value
            for name, value in factors.items()
        )
        score = max(0.0, min(1.0, score))  # clamp

        tier = self._tier_from_score(score)

        logger.debug(
            "Complexity analysis: score=%.3f tier=%s factors=%s estimated_tokens=%d",
            score, tier, factors, estimated_tokens,
        )

        return TaskComplexity(
            score=score,
            factors=factors,
            recommended_tier=tier,
            estimated_tokens=estimated_tokens,
        )

    # -- factor extractors --------------------------------------------------

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token count: words / 0.75 (≈ 1.33 tokens per word)."""
        word_count = len(text.split())
        return max(1, int(word_count / 0.75))

    @staticmethod
    def _factor_prompt_length(prompt: str, estimated_tokens: int) -> float:
        """Longer prompts generally need more capable models.

        Score curve:
        - <100 tokens  → 0.0‥0.1
        - 100–500      → 0.1‥0.4
        - 500–2000     → 0.4‥0.7
        - 2000–8000    → 0.7‥0.9
        - >8000        → 0.9‥1.0
        """
        if estimated_tokens < 100:
            return estimated_tokens / 1000.0  # 0 → 0.1
        if estimated_tokens < 500:
            return 0.1 + (estimated_tokens - 100) / 400 * 0.3
        if estimated_tokens < 2000:
            return 0.4 + (estimated_tokens - 500) / 1500 * 0.3
        if estimated_tokens < 8000:
            return 0.7 + (estimated_tokens - 2000) / 6000 * 0.2
        return min(1.0, 0.9 + (estimated_tokens - 8000) / 50000 * 0.1)

    @staticmethod
    def _factor_reasoning(prompt: str) -> float:
        """Detect multi-step reasoning keywords in the prompt."""
        lower = prompt.lower()
        hits = sum(1 for kw in _REASONING_KEYWORDS if kw in lower)
        # 0 hits → 0.0, 1 → 0.2, 2 → 0.4, 3 → 0.6, 4 → 0.8, 5+ → 1.0
        return min(1.0, hits * 0.2)

    @staticmethod
    def _factor_schema(schema: type[BaseModel] | None) -> float:
        """Score schema complexity by counting fields, nesting, optionality."""
        if schema is None:
            return 0.0

        try:
            json_schema = schema.model_json_schema()
        except Exception:
            return 0.1  # schema exists but we can't introspect it

        total_fields = 0
        max_depth = 0
        optional_count = 0
        required_count = 0

        def _walk(node: dict[str, Any], depth: int = 0) -> None:
            nonlocal total_fields, max_depth, optional_count, required_count

            if depth > max_depth:
                max_depth = depth

            properties = node.get("properties", {})
            required_set = set(node.get("required", []))

            for name, prop in properties.items():
                total_fields += 1
                if name in required_set:
                    required_count += 1
                else:
                    optional_count += 1

                # Recurse into nested objects
                if prop.get("type") == "object":
                    _walk(prop, depth + 1)

                # Recurse into array items
                items = prop.get("items", {})
                if isinstance(items, dict) and items.get("type") == "object":
                    _walk(items, depth + 1)

            # Handle $defs / definitions for referenced schemas
            for defn in node.get("$defs", {}).values():
                if isinstance(defn, dict) and defn.get("type") == "object":
                    _walk(defn, depth + 1)

        _walk(json_schema)

        # Scoring: more fields + deeper nesting = higher complexity
        field_score = min(1.0, total_fields / 20.0)
        depth_score = min(1.0, max_depth / 4.0)
        mix_score = (
            0.3 if optional_count > 0 and required_count > 0 else 0.0
        )

        return min(1.0, field_score * 0.5 + depth_score * 0.3 + mix_score * 0.2)

    @staticmethod
    def _factor_code(prompt: str) -> float:
        """Detect code-generation / programming keywords."""
        lower = prompt.lower()
        hits = sum(1 for kw in _CODE_KEYWORDS if kw in lower)
        return min(1.0, hits * 0.15)

    @staticmethod
    def _factor_language(prompt: str) -> float:
        """Detect non-English / multilingual content."""
        if not prompt:
            return 0.0
        non_latin_chars = len(_NON_LATIN_RE.findall(prompt))
        ratio = non_latin_chars / max(1, len(prompt))
        if ratio > 0.3:
            return 1.0
        if ratio > 0.05:
            return 0.6
        if non_latin_chars > 0:
            return 0.3
        return 0.0

    @staticmethod
    def _factor_tools(tools: list[Any] | None) -> float:
        """More tools available → more complex orchestration needed."""
        if not tools:
            return 0.0
        count = len(tools)
        # 1 tool → 0.2, 3 → 0.6, 5+ → 1.0
        return min(1.0, count * 0.2)

    # -- helpers ------------------------------------------------------------

    def _tier_from_score(self, score: float) -> str:
        if score < self._threshold_cheap:
            return "cheap"
        if score > self._threshold_premium:
            return "premium"
        return "standard"
