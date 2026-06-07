"""Hybrid gateway — gates cloud vs. local execution based on privacy level.

Classifies privacy of messages, masks PII before cloud requests, redirects
confidential queries to local models, and de-masks LLM responses.
"""

from __future__ import annotations

import logging
from typing import Any

from frizura.core.exceptions import PrivacyViolationError, NoLocalModelError
from frizura.models.execution import LLMResponse, Message, MessageRole, TokenUsage
from frizura.models.providers import CompletionConfig
from frizura.models.config import SwarmConfig
from frizura.swarm.classifier import PrivacyClassifier, PrivacyLevel
from frizura.swarm.masker import PIIMasker
from frizura.swarm.pool import LocalPool

logger = logging.getLogger(__name__)


class HybridGateway:
    """Gateway for routing LLM calls between local and cloud providers.

    Implements PII classification, masking/unmasking, and local-first fallback.
    """

    def __init__(
        self,
        config: SwarmConfig | None = None,
        classifier: PrivacyClassifier | None = None,
        masker: PIIMasker | None = None,
        local_pool: LocalPool | None = None,
    ) -> None:
        self.config = config or SwarmConfig()
        self.classifier = classifier or PrivacyClassifier()
        self.masker = masker or PIIMasker(classifier=self.classifier)
        self.local_pool = local_pool or LocalPool(hosts=self.config.ollama_hosts)

    async def complete(
        self,
        provider: str,
        model: str,
        messages: list[Message],
        config: CompletionConfig,
    ) -> LLMResponse:
        """Process messages, apply privacy rules, and route to cloud or local.

        Args:
            provider: Target provider name.
            model: Target model name.
            messages: List of input messages.
            config: Completion configuration.

        Returns:
            Normalised LLMResponse.
        """
        # 1. Privacy classification
        combined_text = "\n".join(m.content for m in messages if m.content)
        classification = self.classifier.classify(combined_text)

        policy = self.config.privacy_mode
        # If policy is local_only, force CONFIDENTIAL
        if policy == "local_only":
            level = PrivacyLevel.CONFIDENTIAL
        elif policy == "cloud_only":
            level = PrivacyLevel.PUBLIC
        else:
            level = classification.level

        # 2. Route based on privacy level
        if level == PrivacyLevel.CONFIDENTIAL:
            # Must run locally
            logger.info("Privacy level CONFIDENTIAL: routing to local pool")
            if provider != "ollama" and not self.config.local_first:
                raise PrivacyViolationError(
                    level.value, f"cloud provider '{provider}'"
                )
            return await self.complete_local(messages, config)

        elif level == PrivacyLevel.SENSITIVE:
            # Mask PII and run cloud
            logger.info("Privacy level SENSITIVE: masking PII and routing to cloud")
            msg_dicts = [m.model_dump(exclude_none=True) for m in messages]
            masked_dicts, mapping = self.masker.mask_messages(msg_dicts)

            masked_messages = [Message(**m) for m in masked_dicts]

            # Direct call to provider or call_llm
            resp = await self._call_provider(provider, model, masked_messages, config)

            # Unmask response
            unmasked_content = self.masker.unmask_response(resp.content, mapping)
            return resp.model_copy(update={"content": unmasked_content})

        else:
            # PUBLIC: Route normally
            logger.debug("Privacy level PUBLIC: routing to cloud")
            return await self._call_provider(provider, model, messages, config)

    async def complete_local(
        self,
        messages: list[Message],
        config: CompletionConfig,
    ) -> LLMResponse:
        """Route call to local Ollama pool."""
        local_model = await self.local_pool.select_model()
        if not local_model:
            raise NoLocalModelError("No healthy local models found in pool")

        logger.info(
            "Selected local model '%s' on %s", local_model.name, local_model.host
        )

        # Build completion request for Ollama
        from frizura.providers.ollama import OllamaProvider
        ollama_prov = OllamaProvider(
            model_id=local_model.name,
            host=local_model.host,
        )
        return await ollama_prov.complete(messages, config)

    async def _call_provider(
        self,
        provider: str,
        model: str,
        messages: list[Message],
        config: CompletionConfig,
    ) -> LLMResponse:
        """Helper to invoke a provider directly."""
        from frizura.providers.registry import ModelRegistry
        registry = ModelRegistry()
        try:
            # Try to get from registry (including mock providers)
            prov_instance = registry.get_provider(f"{provider}:{model}")
            return await prov_instance.complete(messages, config)
        except Exception as registry_exc:
            logger.debug("Registry lookup failed: %s. Trying direct import fallback...", registry_exc)
            
        # Fallback: manually import the provider module and class
        import importlib
        try:
            mod_path = f"frizura.providers.{provider}"
            mod = importlib.import_module(mod_path)
            
            # Find provider class or function
            provider_cls = None
            cls_name = f"{provider.capitalize()}Provider"
            if hasattr(mod, cls_name):
                provider_cls = getattr(mod, cls_name)

            if provider_cls:
                try:
                    model_info = registry.get(model)
                except Exception:
                    from frizura.swarm.pool import make_ollama_model_info
                    model_info = make_ollama_model_info(model)
                inst = provider_cls(model_info=model_info)
                return await inst.complete(messages, config)

            # Fallback to direct module complete function if available
            complete_fn = getattr(mod, "complete", None) or getattr(mod, "acomplete", None)
            if complete_fn:
                msg_dicts = [m.model_dump(exclude_none=True) for m in messages]
                kwargs: dict[str, Any] = {
                    "model": model,
                    "messages": msg_dicts,
                    "temperature": config.temperature,
                }
                if config.max_tokens is not None:
                    kwargs["max_tokens"] = config.max_tokens
                if config.json_mode:
                    kwargs["json_mode"] = True
                if config.tools:
                    kwargs["tools"] = config.tools

                import asyncio
                if asyncio.iscoroutinefunction(complete_fn):
                    res = await complete_fn(**kwargs)
                else:
                    res = complete_fn(**kwargs)

                if isinstance(res, LLMResponse):
                    return res
                elif isinstance(res, dict):
                    return LLMResponse(
                        content=res.get("content", ""),
                        model=model,
                        provider=provider,
                        usage=TokenUsage(**res.get("usage", {})),
                        finish_reason=res.get("finish_reason", "stop"),
                        raw=res,
                    )
        except Exception as exc:
            logger.error("Failed to complete with provider %s: %s", provider, exc)
            raise

        raise ValueError(f"Could not load provider implementation for {provider}")

    async def close(self) -> None:
        """Close local pool."""
        await self.local_pool.close()
