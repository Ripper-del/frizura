"""Privacy classifier — detect PII and classify data sensitivity.

Uses regex-based pattern matching to find personally identifiable
information (PII) in text.  Supports international formats including
Russian-specific documents (ИНН, СНИЛС, паспорт).

Classification levels:

- **PUBLIC** — no PII detected
- **SENSITIVE** — 1–3 low-risk entities (emails, phones, IPs)
- **CONFIDENTIAL** — 4+ entities, or any high-risk entity (passport, SSN,
  credit card)
"""

from __future__ import annotations

import logging
import re
from enum import StrEnum

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public models
# ---------------------------------------------------------------------------

class PrivacyLevel(StrEnum):
    """Data sensitivity level."""

    PUBLIC = "public"
    SENSITIVE = "sensitive"
    CONFIDENTIAL = "confidential"


class DetectedEntity(BaseModel):
    """A single PII entity found in the text."""

    entity_type: str
    value: str
    start: int
    end: int


class PrivacyClassification(BaseModel):
    """Result of privacy classification."""

    level: PrivacyLevel
    detected_entities: list[DetectedEntity] = Field(default_factory=list)
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in the classification (0‥1)",
    )
    entity_summary: dict[str, int] = Field(
        default_factory=dict,
        description="Count of each entity type detected",
    )


# ---------------------------------------------------------------------------
# High-risk entity types — any detection → CONFIDENTIAL
# ---------------------------------------------------------------------------

_HIGH_RISK_TYPES: frozenset[str] = frozenset({
    "credit_card",
    "ssn",
    "passport_ru",
    "snils",
})


# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------

class _Pattern:
    """Wrapper for a compiled regex + metadata."""

    __slots__ = ("name", "regex", "validator")

    def __init__(
        self,
        name: str,
        pattern: str,
        flags: int = 0,
        validator: _Validator | None = None,
    ) -> None:
        self.name = name
        self.regex = re.compile(pattern, flags)
        self.validator = validator


# Optional secondary validator (e.g. Luhn check)
_Validator = type(lambda m: True)  # Callable[[re.Match], bool]


def _luhn_check(number_str: str) -> bool:
    """Validate a number string with the Luhn algorithm."""
    digits = [int(d) for d in number_str if d.isdigit()]
    if len(digits) < 13:
        return False
    checksum = 0
    reverse = digits[::-1]
    for i, d in enumerate(reverse):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


def _validate_credit_card(match: re.Match[str]) -> bool:
    raw = match.group(0)
    return _luhn_check(raw)


def _validate_inn(match: re.Match[str]) -> bool:
    """Basic ИНН length check (10 or 12 digits)."""
    digits = re.sub(r"\D", "", match.group(0))
    return len(digits) in (10, 12)


def _validate_snils(match: re.Match[str]) -> bool:
    """Basic СНИЛС check — should be 11 digits."""
    digits = re.sub(r"\D", "", match.group(0))
    return len(digits) == 11


# The patterns — order matters for overlapping matches
_PATTERNS: list[_Pattern] = [
    # Email
    _Pattern(
        "email",
        r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
    ),
    # Credit card (13-19 digits, optional separators)
    _Pattern(
        "credit_card",
        r"\b(?:\d[ \-]?){13,19}\b",
        validator=_validate_credit_card,
    ),
    # SSN (US)
    _Pattern(
        "ssn",
        r"\b\d{3}[\- ]?\d{2}[\- ]?\d{4}\b",
    ),
    # Phone — international formats
    _Pattern(
        "phone",
        r"(?<!\d)"  # negative lookbehind
        r"(?:\+?\d{1,3}[\s\-.]?)?"
        r"(?:\(?\d{2,4}\)?[\s\-.]?)"
        r"\d{3,4}[\s\-.]?"
        r"\d{2,4}"
        r"(?!\d)",  # negative lookahead
    ),
    # Russian passport — серия и номер
    _Pattern(
        "passport_ru",
        r"(?:паспорт|passport)[\s:]*(?:серия\s*)?(\d{2}[\s]?\d{2})[\s,]*"
        r"(?:(?:№|номер|no\.?)\s*)?(\d{6})",
        re.IGNORECASE | re.UNICODE,
    ),
    # ИНН (10 or 12 digits, optionally prefixed)
    _Pattern(
        "inn_ru",
        r"(?:ИНН|инн|INN)[\s:]*(\d{10,12})",
        re.IGNORECASE | re.UNICODE,
        validator=_validate_inn,
    ),
    # СНИЛС (11 digits with optional dashes)
    _Pattern(
        "snils",
        r"(?:СНИЛС|снилс|SNILS)[\s:]*(\d{3}[\- ]?\d{3}[\- ]?\d{3}[\- ]?\d{2})",
        re.IGNORECASE | re.UNICODE,
        validator=_validate_snils,
    ),
    # IP address (v4)
    _Pattern(
        "ip_address",
        r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
        r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b",
    ),
    # URL with auth tokens (e.g. ?token=..., ?api_key=..., Bearer …)
    _Pattern(
        "auth_url",
        r"https?://[^\s]+[?&](?:token|api_key|access_token|key|secret|password)"
        r"=[^\s&]+",
        re.IGNORECASE,
    ),
]


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class PrivacyClassifier:
    """Classify text by its privacy/PII sensitivity.

    The classifier scans the text with a battery of regex patterns, applies
    optional validators (e.g. Luhn), and produces a :class:`PrivacyClassification`
    with the detected entities and overall level.

    Parameters
    ----------
    extra_patterns:
        Additional :class:`_Pattern` objects to include in scanning.
    """

    def __init__(
        self,
        *,
        extra_patterns: list[_Pattern] | None = None,
    ) -> None:
        self._patterns = list(_PATTERNS)
        if extra_patterns:
            self._patterns.extend(extra_patterns)

    def classify(self, text: str) -> PrivacyClassification:
        """Scan *text* for PII and return a classification."""
        if not text:
            return PrivacyClassification(
                level=PrivacyLevel.PUBLIC,
                confidence=1.0,
            )

        entities = self._scan(text)

        # De-duplicate overlapping matches (keep the longer / higher-priority one)
        entities = self._deduplicate(entities)

        # Summarise by type
        summary: dict[str, int] = {}
        for ent in entities:
            summary[ent.entity_type] = summary.get(ent.entity_type, 0) + 1

        level = self._decide_level(entities)
        confidence = self._compute_confidence(entities, level)

        result = PrivacyClassification(
            level=level,
            detected_entities=entities,
            confidence=confidence,
            entity_summary=summary,
        )

        logger.debug(
            "Privacy classification: level=%s entities=%d summary=%s",
            level, len(entities), summary,
        )

        return result

    # -- internals ----------------------------------------------------------

    def _scan(self, text: str) -> list[DetectedEntity]:
        """Run all patterns against *text*."""
        found: list[DetectedEntity] = []

        for pat in self._patterns:
            for match in pat.regex.finditer(text):
                # Apply optional validator
                if pat.validator is not None:
                    try:
                        if not pat.validator(match):
                            continue
                    except Exception:
                        continue

                found.append(DetectedEntity(
                    entity_type=pat.name,
                    value=match.group(0),
                    start=match.start(),
                    end=match.end(),
                ))

        return found

    @staticmethod
    def _deduplicate(entities: list[DetectedEntity]) -> list[DetectedEntity]:
        """Remove overlapping entities, keeping the longer match."""
        if len(entities) <= 1:
            return entities

        # Sort by start position, then by length descending
        entities.sort(key=lambda e: (e.start, -(e.end - e.start)))

        result: list[DetectedEntity] = []
        last_end = -1

        for ent in entities:
            if ent.start >= last_end:
                result.append(ent)
                last_end = ent.end

        return result

    @staticmethod
    def _decide_level(entities: list[DetectedEntity]) -> PrivacyLevel:
        """Decide the privacy level based on detected entities."""
        if not entities:
            return PrivacyLevel.PUBLIC

        # Any high-risk entity → CONFIDENTIAL
        for ent in entities:
            if ent.entity_type in _HIGH_RISK_TYPES:
                return PrivacyLevel.CONFIDENTIAL

        # 4+ entities of any kind → CONFIDENTIAL
        if len(entities) >= 4:
            return PrivacyLevel.CONFIDENTIAL

        # 1–3 entities → SENSITIVE
        return PrivacyLevel.SENSITIVE

    @staticmethod
    def _compute_confidence(
        entities: list[DetectedEntity],
        level: PrivacyLevel,
    ) -> float:
        """Heuristic confidence in the classification.

        Higher when we have more evidence (more entities detected) or when a
        high-risk entity is found.
        """
        if level == PrivacyLevel.PUBLIC:
            return 1.0  # No entities → very confident it's public

        # Base confidence
        confidence = 0.6

        # More entities → higher confidence
        confidence += min(0.3, len(entities) * 0.05)

        # High-risk entity → higher confidence
        has_high_risk = any(e.entity_type in _HIGH_RISK_TYPES for e in entities)
        if has_high_risk:
            confidence += 0.1

        return min(1.0, confidence)
