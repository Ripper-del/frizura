"""Custom exception hierarchy for Frizura."""

from __future__ import annotations


class FrizuraError(Exception):
    """Base exception for all Frizura errors."""


# --- Engine errors ---

class PipelineError(FrizuraError):
    """Error during pipeline execution."""


class StepError(FrizuraError):
    """Error during step execution."""

    def __init__(self, step_id: str, step_name: str, message: str):
        self.step_id = step_id
        self.step_name = step_name
        super().__init__(f"Step '{step_name}' ({step_id}): {message}")


class GraphError(FrizuraError):
    """Error in pipeline graph construction or validation."""


# --- Provider errors ---

class ProviderError(FrizuraError):
    """Error from an LLM provider."""

    def __init__(self, provider: str, model: str, message: str):
        self.provider = provider
        self.model = model
        super().__init__(f"[{provider}:{model}] {message}")


class ProviderNotFoundError(FrizuraError):
    """Provider not registered or not available."""


class ModelNotFoundError(FrizuraError):
    """Model not found in registry."""


class AuthenticationError(ProviderError):
    """API key missing or invalid."""


class RateLimitError(ProviderError):
    """Rate limit exceeded."""

    def __init__(self, provider: str, model: str, retry_after: float | None = None):
        self.retry_after = retry_after
        msg = "Rate limit exceeded"
        if retry_after:
            msg += f" (retry after {retry_after}s)"
        super().__init__(provider, model, msg)


# --- Budget errors ---

class BudgetExhaustedError(FrizuraError):
    """Budget limit has been reached."""

    def __init__(self, resource: str, limit: float, spent: float):
        self.resource = resource
        self.limit = limit
        self.spent = spent
        super().__init__(
            f"Budget exhausted: {resource} limit={limit}, spent={spent}"
        )


# --- Schema errors ---

class SchemaValidationError(FrizuraError):
    """LLM output failed schema validation."""

    def __init__(self, schema_name: str, errors: list[dict], raw_output: str):
        self.schema_name = schema_name
        self.validation_errors = errors
        self.raw_output = raw_output
        error_summary = "; ".join(
            f"{e.get('loc', '?')}: {e.get('msg', '?')}" for e in errors[:3]
        )
        super().__init__(
            f"Schema '{schema_name}' validation failed: {error_summary}"
        )


class SchemaHealingFailed(FrizuraError):
    """All healing attempts failed."""

    def __init__(self, schema_name: str, attempts: int):
        self.schema_name = schema_name
        self.attempts = attempts
        super().__init__(
            f"Schema '{schema_name}': healing failed after {attempts} attempts"
        )


# --- Routing errors ---

class NoSuitableModelError(FrizuraError):
    """No model matches the budget/capability constraints."""


class RoutingError(FrizuraError):
    """Error during model routing."""


# --- Time-travel errors ---

class ReplayError(FrizuraError):
    """Error during execution replay."""


class SnapshotNotFoundError(FrizuraError):
    """Requested snapshot/checkpoint not found."""

    def __init__(self, pipeline_id: str, step_id: str | None = None):
        self.pipeline_id = pipeline_id
        self.step_id = step_id
        msg = f"Snapshot not found for pipeline '{pipeline_id}'"
        if step_id:
            msg += f" at step '{step_id}'"
        super().__init__(msg)


# --- Swarm errors ---

class SwarmError(FrizuraError):
    """Error in hybrid swarm."""


class NoLocalModelError(SwarmError):
    """No local models available."""


class PrivacyViolationError(SwarmError):
    """Attempted to send confidential data to cloud."""

    def __init__(self, privacy_level: str, target: str):
        self.privacy_level = privacy_level
        self.target = target
        super().__init__(
            f"Cannot send {privacy_level} data to {target}"
        )
