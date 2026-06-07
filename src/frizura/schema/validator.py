"""Schema validator for guaranteeing structured LLM output.

Extracts JSON from messy LLM responses (handling markdown code blocks,
trailing text, partial output) and validates it against Pydantic models.
Produces detailed ``ValidationResult`` objects for downstream consumption
by the healing loop.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, TypeVar

from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Regex patterns for JSON extraction, ordered by priority
_JSON_PATTERNS: list[re.Pattern[str]] = [
    # ```json ... ``` code block (most common LLM format)
    re.compile(r"```json\s*\n?(.*?)```", re.DOTALL),
    # ``` ... ``` generic code block
    re.compile(r"```\s*\n?(.*?)```", re.DOTALL),
    # Raw JSON object (greedy, finds the largest {...} block)
    re.compile(r"(\{.*\})", re.DOTALL),
    # Raw JSON array
    re.compile(r"(\[.*\])", re.DOTALL),
]


class ValidationResult(BaseModel):
    """Result of validating LLM output against a Pydantic schema.

    Attributes:
        success: Whether validation succeeded.
        parsed_object: The validated Pydantic model (as a dict), or None.
        errors: List of validation error dicts (Pydantic format).
        raw_output: The original LLM output string.
        json_extracted: The JSON string that was extracted, or None.
        schema_name: Name of the target schema class.
    """

    success: bool = False
    parsed_object: dict[str, Any] | None = None
    errors: list[dict[str, Any]] = Field(default_factory=list)
    raw_output: str = ""
    json_extracted: str | None = None
    schema_name: str = ""


def extract_json(text: str) -> str | None:
    """Extract JSON from LLM output that may contain surrounding text.

    Tries multiple strategies in order:
    1. Markdown ``json`` code blocks
    2. Generic code blocks
    3. The largest ``{...}`` block in the text
    4. The largest ``[...]`` block in the text
    5. The raw text itself (if it looks like JSON after stripping)

    Args:
        text: Raw LLM output text.

    Returns:
        The extracted JSON string, or ``None`` if no valid JSON is found.
    """
    if not text or not text.strip():
        return None

    stripped = text.strip()

    # Fast path: if the text is already valid JSON
    if stripped.startswith(("{", "[")):
        try:
            json.loads(stripped)
            return stripped
        except json.JSONDecodeError:
            pass

    # Try each extraction pattern
    for pattern in _JSON_PATTERNS:
        matches = pattern.findall(text)
        for match in matches:
            candidate = match.strip()
            if not candidate:
                continue
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                # Try to fix common issues: trailing commas, missing braces
                fixed = _attempt_json_fix(candidate)
                if fixed is not None:
                    return fixed

    # Last resort: try stripping non-JSON prefix/suffix
    # Look for the first { and last } (or [ and ])
    for open_char, close_char in [("{", "}"), ("[", "]")]:
        start = stripped.find(open_char)
        end = stripped.rfind(close_char)
        if start != -1 and end != -1 and end > start:
            candidate = stripped[start : end + 1]
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                fixed = _attempt_json_fix(candidate)
                if fixed is not None:
                    return fixed

    return None


def _attempt_json_fix(text: str) -> str | None:
    """Try to fix common JSON issues in LLM output.

    Handles:
    - Trailing commas before } or ]
    - Single quotes instead of double quotes (simple cases)
    - Unquoted keys (simple cases)
    """
    if not text:
        return None

    # Remove trailing commas before closing braces/brackets
    fixed = re.sub(r",\s*([}\]])", r"\1", text)

    # Try parsing after trailing comma fix
    try:
        json.loads(fixed)
        return fixed
    except json.JSONDecodeError:
        pass

    # Try replacing single quotes with double quotes (only for simple cases)
    # This is intentionally conservative to avoid breaking strings
    if "'" in fixed and '"' not in fixed:
        double_quoted = fixed.replace("'", '"')
        try:
            json.loads(double_quoted)
            return double_quoted
        except json.JSONDecodeError:
            pass

    return None


class SchemaValidator:
    """Validates LLM output against Pydantic schemas.

    Combines JSON extraction with Pydantic validation to handle the
    full range of messy LLM outputs.

    Example::

        validator = SchemaValidator()
        result = validator.validate(llm_output, MyResponseModel)
        if result.success:
            data = result.parsed_object
        else:
            # Pass result.errors to healer
            ...
    """

    def validate(
        self, raw_output: str, schema: type[T]
    ) -> ValidationResult:
        """Validate raw LLM output against a Pydantic model.

        Args:
            raw_output: The raw string from the LLM.
            schema: The Pydantic model class to validate against.

        Returns:
            A ``ValidationResult`` with success status, parsed object,
            and detailed error information.
        """
        schema_name = schema.__name__
        result = ValidationResult(
            raw_output=raw_output,
            schema_name=schema_name,
        )

        # Step 1: Extract JSON
        json_str = extract_json(raw_output)
        result.json_extracted = json_str

        if json_str is None:
            result.errors = [
                {
                    "type": "json_extraction",
                    "loc": (),
                    "msg": "Could not extract valid JSON from LLM output",
                    "input": raw_output[:200],
                }
            ]
            logger.debug(
                "JSON extraction failed for schema %s: %s",
                schema_name,
                raw_output[:100],
            )
            return result

        # Step 2: Parse JSON
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            result.errors = [
                {
                    "type": "json_parse",
                    "loc": (),
                    "msg": f"JSON parse error: {exc.msg}",
                    "input": json_str[:200],
                    "position": exc.pos,
                }
            ]
            logger.debug(
                "JSON parse failed for schema %s: %s at pos %d",
                schema_name,
                exc.msg,
                exc.pos,
            )
            return result

        # Step 3: Validate against schema
        try:
            validated = schema.model_validate(data)
            result.success = True
            result.parsed_object = validated.model_dump(mode="json")
            logger.debug(
                "Validation succeeded for schema %s", schema_name
            )
        except ValidationError as exc:
            result.errors = [
                {
                    "type": err["type"],
                    "loc": list(err["loc"]),
                    "msg": err["msg"],
                    "input": err.get("input"),
                }
                for err in exc.errors()
            ]
            logger.debug(
                "Pydantic validation failed for schema %s: %d errors",
                schema_name,
                len(result.errors),
            )

        return result

    def validate_json_string(
        self, json_str: str, schema: type[T]
    ) -> ValidationResult:
        """Validate an already-extracted JSON string against a schema.

        Use this when you've already extracted JSON and just need
        Pydantic validation.
        """
        result = ValidationResult(
            raw_output=json_str,
            json_extracted=json_str,
            schema_name=schema.__name__,
        )

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            result.errors = [
                {
                    "type": "json_parse",
                    "loc": (),
                    "msg": f"JSON parse error: {exc.msg}",
                    "input": json_str[:200],
                }
            ]
            return result

        try:
            validated = schema.model_validate(data)
            result.success = True
            result.parsed_object = validated.model_dump(mode="json")
        except ValidationError as exc:
            result.errors = [
                {
                    "type": err["type"],
                    "loc": list(err["loc"]),
                    "msg": err["msg"],
                    "input": err.get("input"),
                }
                for err in exc.errors()
            ]

        return result

    def get_schema_description(self, schema: type[BaseModel]) -> str:
        """Generate a human-readable description of a Pydantic schema.

        Produces a JSON Schema representation that can be included in
        prompts to guide LLM output.
        """
        json_schema = schema.model_json_schema()
        return json.dumps(json_schema, indent=2)
