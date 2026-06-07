"""PII masker / demasker — replace PII with deterministic tokens.

The masker scans text using the same pattern engine as the privacy
classifier, replaces each match with a numbered placeholder like
``[EMAIL_1]``, and records the mapping so the original values can be
restored after the LLM call.

Deterministic: the same input text always produces the same masked output
and the same mapping dict, making tests reproducible and caching safe.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from pydantic import BaseModel, Field

from frizura.swarm.classifier import PrivacyClassifier, DetectedEntity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public models
# ---------------------------------------------------------------------------

class MaskedText(BaseModel):
    """Result of masking PII in a text string."""

    masked_text: str
    mapping: dict[str, str] = Field(
        default_factory=dict,
        description="token → original value, e.g. {'[EMAIL_1]': 'user@example.com'}",
    )
    entity_count: int = 0


# ---------------------------------------------------------------------------
# Masker
# ---------------------------------------------------------------------------

class PIIMasker:
    """Replace PII entities with deterministic placeholder tokens.

    Placeholder format: ``[TYPE_N]`` where *TYPE* is the uppercased entity
    type and *N* is a 1-based counter within that type.

    Parameters
    ----------
    classifier:
        A :class:`PrivacyClassifier` instance used to detect entities.
        If not provided a default one is created.
    """

    def __init__(self, classifier: PrivacyClassifier | None = None) -> None:
        self._classifier = classifier or PrivacyClassifier()

    # -- masking ------------------------------------------------------------

    def mask(self, text: str) -> MaskedText:
        """Scan *text* for PII and replace each match with a placeholder.

        Returns a :class:`MaskedText` containing the masked string, the
        reverse mapping, and the entity count.

        The replacement is applied from right-to-left so that earlier offsets
        remain valid as we splice in tokens of different lengths.
        """
        if not text:
            return MaskedText(masked_text=text, mapping={}, entity_count=0)

        classification = self._classifier.classify(text)
        entities = classification.detected_entities

        if not entities:
            return MaskedText(masked_text=text, mapping={}, entity_count=0)

        # Sort by start position ascending (for counter assignment),
        # then we'll apply replacements in reverse order.
        entities_sorted = sorted(entities, key=lambda e: e.start)

        # Assign deterministic counters per entity type
        type_counters: dict[str, int] = {}
        token_assignments: list[tuple[DetectedEntity, str]] = []

        for ent in entities_sorted:
            counter = type_counters.get(ent.entity_type, 0) + 1
            type_counters[ent.entity_type] = counter
            token = f"[{ent.entity_type.upper()}_{counter}]"
            token_assignments.append((ent, token))

        # Build mapping
        mapping: dict[str, str] = {}
        for ent, token in token_assignments:
            mapping[token] = ent.value

        # Apply replacements right-to-left to preserve offsets
        masked = text
        for ent, token in reversed(token_assignments):
            masked = masked[:ent.start] + token + masked[ent.end:]

        result = MaskedText(
            masked_text=masked,
            mapping=mapping,
            entity_count=len(entities_sorted),
        )

        logger.debug(
            "Masked %d entities: types=%s",
            result.entity_count,
            list(type_counters.keys()),
        )

        return result

    # -- unmasking ----------------------------------------------------------

    @staticmethod
    def unmask(masked_text: str, mapping: dict[str, str]) -> str:
        """Restore original PII values from a masked string.

        Replaces every ``[TYPE_N]`` token found in *masked_text* with the
        corresponding value from *mapping*.
        """
        if not mapping:
            return masked_text

        result = masked_text
        # Sort tokens by length descending so longer tokens are replaced first
        # (avoids partial-match issues, e.g. [EMAIL_10] before [EMAIL_1]).
        for token in sorted(mapping, key=len, reverse=True):
            result = result.replace(token, mapping[token])

        return result

    @staticmethod
    def unmask_response(response: str, mapping: dict[str, str]) -> str:
        """Unmask an LLM response that may echo back placeholder tokens.

        This is identical to :meth:`unmask` but exists as a semantically
        distinct entry point — the caller's intent is to restore PII in the
        model's *output* (which may or may not contain our placeholders).
        """
        if not mapping:
            return response

        result = response
        for token in sorted(mapping, key=len, reverse=True):
            result = result.replace(token, mapping[token])

        return result

    # -- convenience --------------------------------------------------------

    def mask_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, str]]:
        """Mask PII across a list of message dicts (role/content).

        Returns the masked messages and a merged mapping that can be used
        to unmask the LLM's response.
        """
        merged_mapping: dict[str, str] = {}
        masked_messages: list[dict[str, Any]] = []

        # We need globally unique counters across all messages
        type_counters: dict[str, int] = {}

        for msg in messages:
            content = msg.get("content", "")
            if not isinstance(content, str) or not content:
                masked_messages.append(dict(msg))
                continue

            classification = self._classifier.classify(content)
            entities = sorted(classification.detected_entities, key=lambda e: e.start)

            if not entities:
                masked_messages.append(dict(msg))
                continue

            # Assign tokens with global counters
            assignments: list[tuple[DetectedEntity, str]] = []
            for ent in entities:
                counter = type_counters.get(ent.entity_type, 0) + 1
                type_counters[ent.entity_type] = counter
                token = f"[{ent.entity_type.upper()}_{counter}]"
                assignments.append((ent, token))
                merged_mapping[token] = ent.value

            # Replace right-to-left
            masked_content = content
            for ent, token in reversed(assignments):
                masked_content = (
                    masked_content[:ent.start] + token + masked_content[ent.end:]
                )

            new_msg = dict(msg)
            new_msg["content"] = masked_content
            masked_messages.append(new_msg)

        return masked_messages, merged_mapping
