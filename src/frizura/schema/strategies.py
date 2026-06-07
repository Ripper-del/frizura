"""Healing strategies and prompt builders for schema repair.

Provides the building blocks used by ``SchemaHealer`` to fix invalid
LLM output: prompt templates for repair requests, schema simplification,
and partial data extraction.
"""

from __future__ import annotations

import json
import logging
from enum import StrEnum
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from frizura.models.execution import Message, MessageRole

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class HealingStrategy(StrEnum):
    """Strategies for fixing invalid schema output."""

    RETRY_SAME = "retry_same"
    """Resend to the same model with a focused repair prompt."""

    ESCALATE_MODEL = "escalate_model"
    """Switch to a more capable model for the repair."""

    SIMPLIFY_SCHEMA = "simplify_schema"
    """Retry with a simplified schema that has fewer required fields."""

    EXTRACT_PARTIAL = "extract_partial"
    """Try to extract whatever valid partial data is available."""


def get_healing_prompt(
    raw_output: str,
    errors: list[dict[str, Any]],
    schema: type[BaseModel],
) -> list[Message]:
    """Build a repair prompt that tells the LLM how to fix its output.

    The prompt includes:
    - The original (broken) output
    - Specific validation errors with field locations
    - The full expected JSON schema
    - Clear instructions to output ONLY valid JSON

    Args:
        raw_output: The original LLM output that failed validation.
        errors: List of validation error dicts (from ``ValidationResult``).
        schema: The target Pydantic model class.

    Returns:
        A list of ``Message`` objects forming the repair conversation.
    """
    schema_json = json.dumps(schema.model_json_schema(), indent=2)

    # Format errors for the prompt
    error_lines: list[str] = []
    for i, err in enumerate(errors, 1):
        loc = err.get("loc", [])
        loc_str = " → ".join(str(p) for p in loc) if loc else "(root)"
        msg = err.get("msg", "Unknown error")
        err_type = err.get("type", "unknown")
        error_lines.append(f"  {i}. [{err_type}] at `{loc_str}`: {msg}")

    errors_text = "\n".join(error_lines) if error_lines else "  (no specific errors)"

    # Truncate raw output if extremely long
    truncated_output = raw_output
    if len(truncated_output) > 2000:
        truncated_output = truncated_output[:2000] + "\n... (truncated)"

    system_prompt = (
        "You are a JSON repair assistant. Your ONLY job is to fix the JSON "
        "output so it matches the required schema exactly. "
        "Respond with ONLY the corrected JSON — no explanations, no "
        "markdown, no code blocks, just raw valid JSON."
    )

    user_prompt = f"""The following output failed schema validation:

--- ORIGINAL OUTPUT ---
{truncated_output}
--- END ORIGINAL OUTPUT ---

Validation errors:
{errors_text}

Expected JSON Schema:
```json
{schema_json}
```

Please output ONLY the corrected JSON that conforms to the schema above.
Fix all validation errors while preserving the original intent and data.
Do NOT wrap the JSON in markdown code blocks or add any explanation."""

    return [
        Message(role=MessageRole.SYSTEM, content=system_prompt),
        Message(role=MessageRole.USER, content=user_prompt),
    ]


def get_simplified_healing_prompt(
    raw_output: str,
    errors: list[dict[str, Any]],
    schema: type[BaseModel],
) -> list[Message]:
    """Build a simpler repair prompt for stubborn cases.

    Uses a more direct, less verbose prompt style that focuses on the
    schema structure rather than the full error details.
    """
    simplified = simplify_schema(schema)
    schema_str = json.dumps(simplified, indent=2)

    system_prompt = (
        "Output valid JSON matching this schema. No explanation, no markdown. "
        "Just the JSON object."
    )

    user_prompt = f"""Fix this JSON to match the schema:

Input:
{raw_output[:1000]}

Schema:
{schema_str}

Output ONLY the fixed JSON:"""

    return [
        Message(role=MessageRole.SYSTEM, content=system_prompt),
        Message(role=MessageRole.USER, content=user_prompt),
    ]


def simplify_schema(schema: type[BaseModel]) -> dict[str, Any]:
    """Create a simplified representation of a Pydantic schema.

    Produces a streamlined JSON Schema that strips away metadata, examples,
    descriptions (keeping just types and required fields) to reduce prompt
    size and cognitive load on the LLM.

    Args:
        schema: The Pydantic model class.

    Returns:
        A simplified JSON Schema dict.
    """
    full_schema = schema.model_json_schema()
    return _simplify_schema_node(full_schema, full_schema.get("$defs", {}))


def _simplify_schema_node(
    node: dict[str, Any],
    defs: dict[str, Any],
) -> dict[str, Any]:
    """Recursively simplify a JSON Schema node."""
    result: dict[str, Any] = {}

    # Resolve $ref
    if "$ref" in node:
        ref_path = node["$ref"]
        ref_name = ref_path.split("/")[-1]
        if ref_name in defs:
            return _simplify_schema_node(defs[ref_name], defs)

    # Handle anyOf / oneOf (union types) — pick the first non-null
    for union_key in ("anyOf", "oneOf"):
        if union_key in node:
            options = node[union_key]
            non_null = [
                o for o in options if o.get("type") != "null"
            ]
            if non_null:
                simplified = _simplify_schema_node(non_null[0], defs)
                # Mark as optional if null is one of the options
                if len(non_null) < len(options):
                    simplified["nullable"] = True
                return simplified
            return {"type": "null"}

    # Type
    if "type" in node:
        result["type"] = node["type"]

    # Properties (object)
    if "properties" in node:
        result["type"] = "object"
        result["properties"] = {
            k: _simplify_schema_node(v, defs)
            for k, v in node["properties"].items()
        }
        if "required" in node:
            result["required"] = node["required"]

    # Items (array)
    if "items" in node:
        result["type"] = "array"
        result["items"] = _simplify_schema_node(node["items"], defs)

    # Enum
    if "enum" in node:
        result["enum"] = node["enum"]

    # Default
    if "default" in node:
        result["default"] = node["default"]

    # Title (keep as a minimal hint)
    if "title" in node and "properties" in node:
        result["title"] = node["title"]

    return result


def extract_partial(
    raw_output: str, schema: type[T]
) -> T | None:
    """Try to extract valid partial data from raw output.

    Attempts to parse the output and fill in missing required fields with
    sensible defaults (empty strings, empty lists, zeros). This is a
    last-resort strategy when full validation fails.

    Args:
        raw_output: The raw LLM output string.
        schema: The target Pydantic model class.

    Returns:
        A validated instance if partial extraction succeeds, or ``None``.
    """
    from frizura.schema.validator import extract_json

    json_str = extract_json(raw_output)
    if json_str is None:
        return None

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None

    # Try direct validation first
    try:
        return schema.model_validate(data)
    except ValidationError:
        pass

    # Fill in missing required fields with defaults
    json_schema = schema.model_json_schema()
    required_fields = json_schema.get("required", [])
    properties = json_schema.get("properties", {})

    for field_name in required_fields:
        if field_name not in data:
            prop = properties.get(field_name, {})
            data[field_name] = _default_for_type(prop)

    # Try validation again with filled defaults
    try:
        return schema.model_validate(data)
    except ValidationError as exc:
        logger.debug(
            "Partial extraction failed for %s: %s",
            schema.__name__,
            exc.error_count(),
        )
        return None


def _default_for_type(prop: dict[str, Any]) -> Any:
    """Generate a sensible default value for a JSON Schema property."""
    if "default" in prop:
        return prop["default"]

    # Handle $ref / anyOf / oneOf
    if "$ref" in prop or "anyOf" in prop or "oneOf" in prop:
        return {}

    prop_type = prop.get("type", "string")

    match prop_type:
        case "string":
            enum = prop.get("enum")
            return enum[0] if enum else ""
        case "integer":
            return 0
        case "number":
            return 0.0
        case "boolean":
            return False
        case "array":
            return []
        case "object":
            return {}
        case "null":
            return None
        case _:
            return ""
