"""Local model pool manager for Ollama integration.

Discovers, health-checks, and selects local models running on localhost or
other hosts in the LAN.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from pydantic import BaseModel, Field
import httpx

logger = logging.getLogger(__name__)


class LocalModel(BaseModel):
    """Information about a locally running model."""

    name: str
    host: str
    size: int = 0
    quantization: str = ""
    capabilities: list[str] = Field(default_factory=list)
    is_available: bool = True


class PoolStatus(BaseModel):
    """Status of the local model pool."""

    total_hosts: int = 0
    healthy_hosts: int = 0
    models_available: int = 0
    details: dict[str, Any] = Field(default_factory=dict)


class LocalPool:
    """Manages a pool of local Ollama instances."""

    def __init__(self, hosts: list[str] | None = None) -> None:
        self.hosts = hosts or ["http://localhost:11434"]
        self._client = httpx.AsyncClient(timeout=5.0)

    async def discover(self) -> list[LocalModel]:
        """Query Ollama hosts to find available local models."""
        models: list[LocalModel] = []
        for host in self.hosts:
            try:
                resp = await self._client.get(f"{host}/api/tags")
                if resp.status_code == 200:
                    data = resp.json()
                    for item in data.get("models", []):
                        name = item.get("name")
                        details = item.get("details", {})
                        models.append(
                            LocalModel(
                                name=name,
                                host=host,
                                size=item.get("size", 0),
                                quantization=details.get("quantization_level", "unknown"),
                                capabilities=["text"],  # Default
                                is_available=True,
                            )
                        )
            except Exception as exc:
                logger.debug("Ollama host %s not reachable: %s", host, exc)
        return models

    async def healthcheck(self) -> PoolStatus:
        """Check all configured hosts and return status."""
        healthy = 0
        details = {}
        total_models = 0
        for host in self.hosts:
            try:
                resp = await self._client.get(f"{host}/api/tags")
                if resp.status_code == 200:
                    healthy += 1
                    models_count = len(resp.json().get("models", []))
                    total_models += models_count
                    details[host] = {"status": "ok", "models": models_count}
                else:
                    details[host] = {"status": f"error_code_{resp.status_code}"}
            except Exception as exc:
                details[host] = {"status": f"error: {str(exc)}"}

        return PoolStatus(
            total_hosts=len(self.hosts),
            healthy_hosts=healthy,
            models_available=total_models,
            details=details,
        )

    async def select_model(self, required_capability: str | None = None) -> LocalModel | None:
        """Select a local model from the pool, preferring localhost if available."""
        models = await self.discover()
        if not models:
            return None
        # Prefer localhost first
        localhosts = [m for m in models if "localhost" in m.host or "127.0.0.1" in m.host]
        if localhosts:
            return localhosts[0]
        return models[0]

    async def discover_lan(self, port: int = 11434) -> list[str]:
        """Simple LAN discovery for Ollama endpoints (stub/mock for Phase 1)."""
        logger.info("Scanning LAN for Ollama instances on port %d...", port)
        # In Phase 1 we return localhost + hosts in self.hosts
        return list(self.hosts)

    async def close(self) -> None:
        """Close HTTP client."""
        await self._client.aclose()
