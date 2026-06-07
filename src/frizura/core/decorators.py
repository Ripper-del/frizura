"""Decorators API for Frizura tasks and pipeline steps.

Enables defining LLM tasks and pipeline steps using Python decorators,
supporting both sync and async functions.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
from typing import Any, Callable, TypeVar, overload

from pydantic import BaseModel

from frizura.core.engine import FrizuraEngine
from frizura.core.graph import Pipeline, Step, StepType
from frizura.models.budget import Budget
from frizura.models.execution import Message, MessageRole

F = TypeVar("F", bound=Callable[..., Any])


def _format_prompt_from_docstring(docstring: str | None, args: tuple[Any, ...], kwargs: dict[str, Any], func: Callable[..., Any]) -> str:
    """Helper to format a docstring template using function arguments."""
    if not docstring:
        return ""
    
    # Strip common docstring whitespace
    lines = [line.strip() for line in docstring.strip().split("\n")]
    template = "\n".join(lines)
    
    # Try to bind arguments to function signature to get a parameter mapping
    sig = inspect.signature(func)
    try:
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        return template.format(**bound.arguments)
    except Exception:
        # Fallback to simple formatting or returning raw template
        try:
            return template.format(*args, **kwargs)
        except Exception:
            return template


def task(
    model: str | None = None,
    output_schema: type[BaseModel] | None = None,
    budget: Budget | None = None,
    privacy: str = "auto",
    system_prompt: str | None = None,
) -> Callable[[F], F]:
    """Decorator to define a standalone LLM task.

    When the decorated function is called, Frizura automatically creates
    and runs a single-step pipeline.

    Usage::

        @task(model="openai:gpt-4o-mini", output_schema=Sentiment)
        async def analyze_sentiment(text: str) -> Sentiment:
            '''Analyze the sentiment of: {text}'''
    """
    def decorator(func: F) -> F:
        # Determine if the decorated function is async
        is_async = asyncio.iscoroutinefunction(func)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # 1. Resolve user prompt
            # Execute the function to see if it returns a prompt string
            prompt = ""
            if is_async:
                # We cannot easily run async inside sync wrapper if not in event loop,
                # so we delegate to an async runner or handle it if we are called in async context.
                # Since tasks are usually awaited, the wrapper itself must be async.
                pass
            else:
                result = func(*args, **kwargs)
                if isinstance(result, str):
                    prompt = result
                elif isinstance(result, Message):
                    prompt = result.content
                elif result is None:
                    # Try docstring
                    prompt = _format_prompt_from_docstring(func.__doc__, args, kwargs, func)
            
            # Create the async execution wrapper
            async def async_run() -> Any:
                nonlocal prompt
                if is_async:
                    result = await func(*args, **kwargs)
                    if isinstance(result, str):
                        prompt = result
                    elif isinstance(result, Message):
                        prompt = result.content
                    elif result is None:
                        prompt = _format_prompt_from_docstring(func.__doc__, args, kwargs, func)
                
                # If prompt is still empty, fallback to docstring template
                if not prompt and func.__doc__:
                    prompt = _format_prompt_from_docstring(func.__doc__, args, kwargs, func)

                # Create the single-step pipeline
                # The step handler returns the prompt we just resolved
                step = Step(
                    name=func.__name__,
                    step_type=StepType.LLM,
                    system_prompt=system_prompt,
                    output_schema=output_schema,
                    model=model,
                    budget=budget,
                    privacy=privacy,
                    handler=lambda ctx: prompt,
                )

                pipeline = Pipeline(name=func.__name__).add_step(step)
                engine = FrizuraEngine()
                pipeline_result = await engine.run(pipeline, input_data=prompt, budget=budget)
                
                if pipeline_result.status == "failed":
                    raise RuntimeError(f"Task failed: {pipeline_result.error}")
                
                step_res = pipeline_result.steps[0]
                if output_schema is not None:
                    return step_res.output
                return step_res.llm_response.content if step_res.llm_response else step_res.output

            if is_async:
                return async_run()
            else:
                # If the decorated function is sync, we run the async loop synchronously
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None
                
                if loop and loop.is_running():
                    # We are in an running event loop (e.g. inside another async task),
                    # so we must return a coroutine, making the sync wrapper act like an async one.
                    return async_run()
                else:
                    return asyncio.run(async_run())

        return wrapper  # type: ignore[return-value]

    return decorator


def step(
    step_type: StepType = StepType.LLM,
    system_prompt: str | None = None,
    output_schema: type[BaseModel] | None = None,
    model: str | None = None,
    budget: Budget | None = None,
    privacy: str = "auto",
) -> Callable[[F], F]:
    """Decorator to mark a function as a pipeline step.

    Used with the Pipeline builder. The decorated function defines the user prompt
    generator or custom transform logic for the step.
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        # Attach Frizura step metadata to the function
        wrapper.__frizura_step__ = {  # type: ignore[attr-defined]
            "name": func.__name__,
            "step_type": step_type,
            "system_prompt": system_prompt,
            "output_schema": output_schema,
            "model": model,
            "budget": budget,
            "privacy": privacy,
            "handler": func,
        }
        return wrapper  # type: ignore[return-value]

    return decorator
