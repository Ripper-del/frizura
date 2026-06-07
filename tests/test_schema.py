"""Unit tests for Frizura schema validation and healing loops."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from frizura.schema.validator import SchemaValidator, extract_json
from frizura.schema.healer import SchemaHealer
from frizura.core.context import ExecutionContext
from frizura.models.budget import Budget, BudgetConstraint
from frizura.models.config import FrizuraConfig


class Person(BaseModel):
    name: str
    age: int
    tags: list[str] = Field(default_factory=list)


def test_json_extraction() -> None:
    """Test extracting JSON from messy strings."""
    # Markdown block
    text1 = "Here is your JSON:\n```json\n{\"name\": \"Alice\", \"age\": 30}\n```\nHope it helps!"
    assert extract_json(text1) == '{"name": "Alice", "age": 30}'

    # Missing json prefix
    text2 = "```\n{\"name\": \"Bob\", \"age\": 25}\n```"
    assert extract_json(text2) == '{"name": "Bob", "age": 25}'

    # Messy surrounding text without code fences
    text3 = "Random prefix { \"name\": \"Charlie\", \"age\": 40 } random suffix"
    assert extract_json(text3) == '{ "name": "Charlie", "age": 40 }'

    # Trailing comma fixing
    text4 = '{"name": "David", "age": 35,}'
    assert extract_json(text4) == '{"name": "David", "age": 35}'


def test_schema_validator() -> None:
    """Test validation of raw output and error reporting."""
    validator = SchemaValidator()
    
    # Valid output
    res1 = validator.validate('{"name": "Alice", "age": 30}', Person)
    assert res1.success
    assert res1.parsed_object == {"name": "Alice", "age": 30, "tags": []}

    # Missing field
    res2 = validator.validate('{"age": 30}', Person)
    assert not res2.success
    assert len(res2.errors) == 1
    assert res2.errors[0]["loc"] == ["name"]

    # Wrong type
    res3 = validator.validate('{"name": "Alice", "age": "not-an-int"}', Person)
    assert not res3.success
    assert "age" in str(res3.errors[0]["loc"])


@pytest.mark.asyncio
async def test_schema_healer(mock_provider) -> None:
    """Test schema healing automatically repairs invalid JSON."""
    healer = SchemaHealer()
    
    # Configure mock provider to return valid JSON on the next call (healing step)
    mock_provider.responses = ['{"name": "Bob", "age": 25}']
    
    # Simulate a context
    budget = BudgetConstraint(budget=Budget(max_retries=3))
    ctx = ExecutionContext(
        pipeline_id="test-heal",
        pipeline_name="test-heal",
        state={},
        budget=budget,
        memory=[],
        metadata={},
        current_step_index=0,
        config=FrizuraConfig(),
    )

    # Initial invalid output
    invalid_output = '{"name": "Bob"}'  # Missing 'age'
    validator = SchemaValidator()
    val_res = validator.validate(invalid_output, Person)
    
    assert not val_res.success
    
    # Heal the output
    from frizura.core.graph import Step
    step = Step(name="test-step")
    
    from frizura.models.providers import CompletionConfig
    healed_obj = await healer.heal(
        raw_output=invalid_output,
        schema=Person,
        validation_errors=val_res.errors,
        provider=mock_provider,
        config=CompletionConfig(model="mock-model"),
        max_attempts=2,
        pipeline_id=ctx.pipeline_id,
        step_id=step.id,
        budget=ctx.budget,
    )
    
    assert isinstance(healed_obj, Person)
    assert healed_obj.name == "Bob"
    assert healed_obj.age == 25
    assert ctx.budget.retries_used == 1
