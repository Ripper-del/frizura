"""Unit tests for Frizura hybrid swarm (PII detection, masking, routing)."""

from __future__ import annotations

import pytest

from frizura.swarm.classifier import PrivacyClassifier, PrivacyLevel
from frizura.swarm.masker import PIIMasker
from frizura.swarm.gateway import HybridGateway
from frizura.swarm.pool import LocalPool, LocalModel
from frizura.core.exceptions import PrivacyViolationError, NoLocalModelError
from frizura.models.execution import Message, MessageRole
from frizura.models.providers import CompletionConfig
from frizura.models.config import SwarmConfig


def test_privacy_classification() -> None:
    """Test detecting PII and classifying data sensitivity."""
    classifier = PrivacyClassifier()
    
    # Public
    res1 = classifier.classify("Hello, how can I help you today?")
    assert res1.level == PrivacyLevel.PUBLIC
    assert len(res1.detected_entities) == 0

    # Sensitive (Phone/Email)
    res2 = classifier.classify("My email is user@example.com, call me.")
    assert res2.level == PrivacyLevel.SENSITIVE
    assert len(res2.detected_entities) == 1
    assert res2.detected_entities[0].entity_type == "email"

    # Confidential (Russian Passport, Credit Card, or 4+ low risk entities)
    res3 = classifier.classify("паспорт серия 4508 № 123456")
    assert res3.level == PrivacyLevel.CONFIDENTIAL
    
    res4 = classifier.classify("Phone1: +7-999-111-22-33, Phone2: +7-999-222-33-44, Email: a@b.com, IP: 127.0.0.1")
    assert res4.level == PrivacyLevel.CONFIDENTIAL  # 4+ entities


def test_pii_masking_unmasking() -> None:
    """Test PII masking replaces values and unmasking restores them."""
    masker = PIIMasker()
    
    text = "Call me at +7-999-123-45-67 or write to bob@example.com."
    masked_res = masker.mask(text)
    
    assert "[PHONE_1]" in masked_res.masked_text
    assert "[EMAIL_1]" in masked_res.masked_text
    assert "+7-999-123-45-67" not in masked_res.masked_text
    assert "bob@example.com" not in masked_res.masked_text
    
    # Unmask
    restored = masker.unmask(masked_res.masked_text, masked_res.mapping)
    assert restored == text


@pytest.mark.asyncio
async def test_hybrid_gateway_routing(register_mock_provider) -> None:
    """Test hybrid gateway routes or masks based on privacy classification."""
    config = SwarmConfig(privacy_mode="auto", local_first=False)
    
    # Mock local pool that returns a model
    class MockPool(LocalPool):
        async def discover(self):
            return [LocalModel(name="llama3", host="http://localhost:11434")]
    
    gateway = HybridGateway(
        config=config,
        local_pool=MockPool(),
    )
    
    # 1. PUBLIC request -> goes directly to cloud provider without changes
    msg_pub = [Message(role=MessageRole.USER, content="Explain quantum computing.")]
    resp1 = await gateway.complete("mock", "model", msg_pub, CompletionConfig())
    assert "Mock response" in resp1.content

    # 2. SENSITIVE request -> should mask PII, call cloud, then unmask response
    # Setup mock provider to echo back the user message (to test unmasking of output)
    from .conftest import MockLLMProvider
    mock_prov = MockLLMProvider("echo-model")
    mock_prov.responses = ["Hello [EMAIL_1], we received your request."]
    
    # Re-register
    from frizura.providers.registry import ModelRegistry
    ModelRegistry().register(mock_prov.model_info)
    ModelRegistry()._providers["mock:echo-model"] = mock_prov
    
    msg_sens = [Message(role=MessageRole.USER, content="Send to alice@example.com.")]
    resp2 = await gateway.complete("mock", "echo-model", msg_sens, CompletionConfig())
    
    assert "alice@example.com" in resp2.content
    assert "[EMAIL_1]" not in resp2.content
