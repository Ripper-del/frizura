# frizura

Next-gen LLM orchestrator: smart cost routing, guaranteed schema output, time-travel debugging, and local-first hybrid swarm.

**Status:** Alpha · **Python:** 3.12+ · **License:** Apache 2.0

---

## Table of Contents

- [What is it](#what-is-it)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Concepts](#concepts)
  - [Task — single call](#task--single-call)
  - [Pipeline — DAG of steps](#pipeline--dag-of-steps)
  - [Budget — cost control](#budget--cost-control)
  - [Smart Router](#smart-router)
  - [Schema Guard](#schema-guard)
  - [Time-Travel Debugging](#time-travel-debugging)
  - [Hybrid Swarm — local models](#hybrid-swarm--local-models)
- [Providers & Models](#providers--models)
- [Configuration](#configuration)
- [CLI](#cli)
- [Examples](#examples)
- [Architecture](#architecture)
- [Development](#development)

---

## What is it

Frizura is a Python library for orchestrating LLM calls. It sits between your code and language models and handles:

- automatic model selection by price, quality, and speed
- guaranteed structured output via Pydantic with self-healing
- recording every step for later debugging and replay
- routing sensitive data to local models (Ollama)
- budget enforcement at the task, step, or pipeline level

No external services required — works as a plain Python library. SQLite for event storage is set up automatically.

---

## Features

| Feature | Description |
|---|---|
| `@frizura.task` | Decorator — turns any function into an LLM task |
| `Pipeline` + `Step` | DAG builder: linear, parallel, and conditional steps |
| Smart Router | Picks a model by complexity score, budget, and strategy |
| Cascade Routing | Tries the cheapest model first, escalates on failure |
| Schema Guard | Validates LLM output via Pydantic, self-heals up to 3 times |
| Time-Travel | Every step → event in SQLite, replay to any point |
| Hybrid Swarm | Local Ollama models + cloud APIs, PII masking |
| Budget Control | `Budget(max_cost=0.01)` — hard cap on cost and time |
| Rich TUI | `frizura inspect` — beautiful terminal debugger |

---

## Installation

```bash
# Full install (all providers + CLI)
pip install frizura[all]

# Only the providers you need
pip install frizura[anthropic]
pip install frizura[google]
pip install frizura[openai]
pip install frizura[ollama]

# From source
git clone https://github.com/Ripper-del/frizura.git
cd frizura
pip install -e ".[all]"
```

---

## Quick Start

**1. Set API keys** (env vars with `FRIZURA_` prefix):

```bash
export FRIZURA_ANTHROPIC_API_KEY=sk-ant-...
export FRIZURA_GOOGLE_API_KEY=AIza...
export FRIZURA_OPENAI_API_KEY=sk-...
# or use a .env file (see .env.example)
```

**2. Minimal example:**

```python
import asyncio
import frizura

@frizura.task(model="anthropic:claude-sonnet-4-6")
async def explain(topic: str) -> str:
    """Explain in simple terms: {topic}"""

result = asyncio.run(explain("quantum entanglement"))
print(result)
```

**3. With structured output:**

```python
from pydantic import BaseModel
import frizura

class Review(BaseModel):
    sentiment: str
    score: float
    summary: str

@frizura.task(
    model="openai:gpt-4o-mini",
    output_schema=Review,
    budget=frizura.Budget(max_cost=0.01),
)
async def analyze(text: str) -> Review:
    """Analyze the sentiment of this review: {text}"""

result = asyncio.run(analyze("Great movie, highly recommend!"))
print(result.sentiment, result.score)
```

---

## Concepts

### Task — single call

`@frizura.task` is the simplest way to call an LLM. Internally it creates a single-step Pipeline and runs it through the engine.

```python
@frizura.task(
    model="anthropic:claude-sonnet-4-6",  # explicit model or None (auto-routing)
    output_schema=MyModel,                 # optional — Pydantic schema
    budget=frizura.Budget(max_cost=0.05),  # optional — cost limit
    privacy="auto",                        # "auto" | "local_only" | "cloud_only"
    system_prompt="You are an expert.",    # optional
)
async def my_task(arg: str) -> str:
    """Prompt template with {arg} substituted from function arguments."""
```

The prompt is built from the docstring — `{arg}` is filled in from the function's arguments automatically.

---

### Pipeline — DAG of steps

For complex multi-step tasks with dependencies between steps:

```python
from frizura.core.graph import Pipeline, Step, StepType

# Linear chain
pipeline = (
    Pipeline("my-pipeline")
    .add_step(Step(
        name="draft",
        system_prompt="Write a draft.",
        handler=lambda ctx: ctx.get("input"),
    ))
    .add_step(Step(
        name="revise",
        system_prompt="Improve the text.",
        handler=lambda ctx: f"Improve this: {ctx.get('draft')}",
    ))
)

# Parallel steps (run concurrently)
pipeline.add_parallel(
    Step(name="translate-en", handler=lambda ctx: "Translate to English"),
    Step(name="translate-de", handler=lambda ctx: "Translate to German"),
)

# Conditional branch
pipeline.add_branch(
    condition_fn=lambda ctx: "formal" if ctx.get("is_business") else "casual",
    branches={
        "formal": Step(name="formal-tone", ...),
        "casual": Step(name="casual-tone", ...),
    }
)

# Run it
engine = frizura.FrizuraEngine()
result = await engine.run(pipeline, input_data="Source text")

print(result.output)
print(f"Cost: ${result.total_cost_usd:.5f}")
print(f"Time: {result.total_duration_ms:.0f}ms")
print(f"Pipeline ID: {result.pipeline_id}")
```

#### Step types

| Type | Description |
|---|---|
| `LLM` | Language model call (default) |
| `TRANSFORM` | Custom Python function, no LLM |
| `PARALLEL` | Concurrent execution of multiple steps |
| `BRANCH` | Conditional branch selection |
| `HUMAN` | Human-in-the-loop pause point |

---

### Budget — cost control

Enforce spending limits at the task or pipeline level:

```python
from frizura.models.budget import Budget

budget = Budget(
    max_cost=0.10,     # hard cap at $0.10
    max_tokens=50000,  # max 50k tokens
    max_time=30.0,     # max 30 seconds
    max_retries=3,     # max 3 retries (for schema healing)
    prefer="cost",     # "cost" | "quality" | "speed" — influences routing strategy
)

result = await engine.run(pipeline, budget=budget)
```

`BudgetExhaustedError` is raised if any limit is exceeded.

---

### Smart Router

When `model=None`, the router picks a model automatically.

**How it works:**
1. Analyses request complexity — `complexity score` from 0.0 to 1.0
2. Filters models by required capabilities (JSON mode, tool calling)
3. Filters by remaining budget
4. Applies the configured strategy

**Strategies:**

| Strategy | Logic |
|---|---|
| `cascade` _(default)_ | Tries cheapest model first, escalates on failure |
| `cheapest` | Always the cheapest suitable model |
| `fastest` | Lowest latency |
| `best_quality` | Premium tier only |

**Model tiers:** `cheap` → `standard` → `premium`

```python
from frizura.models.config import RouterConfig

config = frizura.FrizuraConfig(
    router=RouterConfig(
        default_strategy="cheapest",
        complexity_threshold_cheap=0.3,    # score < 0.3 → cheap tier
        complexity_threshold_premium=0.7,  # score > 0.7 → premium tier
    )
)
```

---

### Schema Guard

Frizura guarantees that LLM output will always be a valid Pydantic object.

**Process:**
1. LLM returns text
2. JSON is extracted (including from markdown fences ` ```json ... ``` `)
3. Validated via `Model.model_validate()`
4. If invalid → healing loop: the error is sent back to the model with a fix request
5. Up to 3 attempts. If all fail → `SchemaHealingFailed`

```python
from pydantic import BaseModel

class Answer(BaseModel):
    text: str
    confidence: float
    sources: list[str]

@frizura.task(output_schema=Answer)
async def query(question: str) -> Answer:
    """Answer this question with sources: {question}"""
```

---

### Time-Travel Debugging

Every run saves all events to SQLite (`.frizura/events.db`). You can inspect and replay any pipeline execution.

```bash
# Inspect pipeline execution in Rich TUI
frizura inspect <pipeline-id>

# Replay up to a specific step
frizura replay <pipeline-id> --to step-name
```

**What `frizura inspect` shows:**
- Summary panel: ID, status, cost, duration, models used
- Step timeline: status icons, duration, cost, heal attempts, output preview
- Event tree grouped by step
- Cost breakdown by model and step

**Event types:**

| Event | When |
|---|---|
| `pipeline.started / completed / failed` | Pipeline lifecycle |
| `step.started / completed / failed / skipped` | Step lifecycle |
| `llm.request / llm.response` | LLM call |
| `routing.decision` | Router chose a model |
| `schema.validation.ok / failed` | Schema check |
| `schema.heal.attempt` | Self-healing attempt |
| `state.snapshot` | Context state snapshot |
| `budget.exhausted` | Budget limit hit |

```python
# Replay in code
engine = frizura.FrizuraEngine()
result = await engine.run(pipeline, "input data")

# Restore context to a specific step
ctx = await engine.replay(result.pipeline_id, until_step="step-id")
print(ctx.state)
```

---

### Hybrid Swarm — local models

Frizura supports Ollama for running models fully locally. The router automatically sends requests to local models when `privacy="local_only"` or when data is detected as sensitive.

```bash
# Check local pool status
frizura swarm status

# Discover models on the local network
frizura swarm discover
```

```python
from frizura.models.config import SwarmConfig

config = frizura.FrizuraConfig(
    swarm=SwarmConfig(
        enabled=True,
        local_first=True,                          # prefer local models
        ollama_hosts=["http://localhost:11434"],    # Ollama hosts
        privacy_mode="auto",                       # "auto" | "local_only" | "cloud_only"
    )
)
```

---

## Providers & Models

### Supported providers

| Provider | Package | Env variable |
|---|---|---|
| OpenAI | `frizura[openai]` | `FRIZURA_OPENAI_API_KEY` |
| Anthropic | `frizura[anthropic]` | `FRIZURA_ANTHROPIC_API_KEY` |
| Google | `frizura[google]` | `FRIZURA_GOOGLE_API_KEY` |
| Ollama | `frizura[ollama]` | — (local, no key needed) |

### Model format

Models are specified as `"provider:model_id"`:

```
openai:gpt-4o
openai:gpt-4o-mini
anthropic:claude-sonnet-4-6
anthropic:claude-opus-4-6
google:gemini-2.0-flash
google:gemini-2.5-pro
ollama:llama3.3
ollama:mistral
ollama:phi4
ollama:qwen2.5
```

---

## Configuration

All settings via environment variables with the `FRIZURA_` prefix or a `.env` file in the project root.

```env
# Providers (at least one required)
FRIZURA_OPENAI_API_KEY=sk-...
FRIZURA_ANTHROPIC_API_KEY=sk-ant-...
FRIZURA_GOOGLE_API_KEY=AIza...

# Default model (used when model=None in task/step)
FRIZURA_DEFAULT_MODEL=anthropic:claude-sonnet-4-6

# Logging
FRIZURA_LOG_LEVEL=INFO
FRIZURA_VERBOSE=false
```

Or via config object in code:

```python
from frizura.models.config import FrizuraConfig, RouterConfig, SwarmConfig, TimeTravelConfig

config = FrizuraConfig(
    default_model="anthropic:claude-sonnet-4-6",
    log_level="DEBUG",
    router=RouterConfig(
        default_strategy="cascade",
        complexity_threshold_cheap=0.3,
        complexity_threshold_premium=0.7,
    ),
    swarm=SwarmConfig(
        local_first=True,
        ollama_hosts=["http://localhost:11434"],
    ),
    timetravel=TimeTravelConfig(
        enabled=True,
        db_path=".frizura/events.db",
        snapshot_every_n_steps=1,
    ),
)

engine = frizura.FrizuraEngine(config=config)
```

---

## CLI

```bash
# Run a pipeline from a Python file
frizura run my_pipeline.py
frizura run my_pipeline.py --budget-cost 0.10 --budget-time 60

# Time-travel debugger — Rich TUI in terminal
frizura inspect <pipeline-id>

# Replay to a specific step
frizura replay <pipeline-id> --to <step-id>

# Manage local model pool
frizura swarm status
frizura swarm discover
```

`frizura run` looks for a `Pipeline` object in the given file and runs it through `FrizuraEngine`.

---

## Examples

### Single task with auto-routing

```python
import frizura

@frizura.task(budget=frizura.Budget(max_cost=0.005, prefer="cost"))
async def summarize(text: str) -> str:
    """Summarize in 3 sentences: {text}"""

result = await summarize(long_article)
```

### Structured output with self-healing

```python
from pydantic import BaseModel
import frizura

class CodeReview(BaseModel):
    issues: list[str]
    suggestions: list[str]
    score: int  # 1–10

@frizura.task(model="anthropic:claude-sonnet-4-6", output_schema=CodeReview)
async def review_code(code: str) -> CodeReview:
    """Review this Python code for quality and bugs:\n\n{code}"""

review = await review_code(my_function)
print(f"Score: {review.score}/10")
print(f"Issues: {review.issues}")
```

### Multi-step pipeline

```python
from frizura.core.graph import Pipeline, Step

pipeline = (
    Pipeline("article-pipeline")
    .add_step(Step(
        name="research",
        system_prompt="You are a researcher.",
        handler=lambda ctx: f"Research this topic: {ctx.get('input')}",
    ))
    .add_step(Step(
        name="write",
        system_prompt="You are a journalist.",
        handler=lambda ctx: f"Write an article based on: {ctx.get('research')}",
    ))
    .add_step(Step(
        name="edit",
        system_prompt="You are an editor.",
        handler=lambda ctx: f"Edit and improve: {ctx.get('write')}",
        model="anthropic:claude-opus-4-6",
    ))
)

engine = frizura.FrizuraEngine()
result = await engine.run(pipeline, "quantum computing for beginners")

print(result.output)
print(f"Steps: {len(result.steps)}")
print(f"Total cost: ${result.total_cost_usd:.4f}")
# frizura inspect <result.pipeline_id>
```

### Parallel translation

```python
pipeline = (
    Pipeline("multi-lang")
    .add_step(Step(name="source", handler=lambda ctx: ctx.get("input")))
    .add_parallel(
        Step(name="en", handler=lambda ctx: f"Translate to English: {ctx.get('source')}"),
        Step(name="de", handler=lambda ctx: f"Translate to German: {ctx.get('source')}"),
        Step(name="ja", handler=lambda ctx: f"Translate to Japanese: {ctx.get('source')}"),
    )
)
```

---

## Architecture

```
src/frizura/
├── core/
│   ├── engine.py       # FrizuraEngine — main orchestrator
│   ├── graph.py        # Pipeline, Step, DAG compiler (Kahn's algorithm)
│   ├── context.py      # ExecutionContext — runtime state
│   ├── decorators.py   # @task, @step
│   └── events.py       # Event types
├── router/
│   ├── router.py       # SmartRouter — model selection
│   ├── analyzer.py     # ComplexityAnalyzer — request complexity scoring
│   ├── calculator.py   # CostCalculator — cost estimation
│   └── strategies.py   # cascade, cheapest, fastest, best_quality
├── providers/
│   ├── base.py         # LLMProvider — abstract base class
│   ├── anthropic.py    # Anthropic SDK
│   ├── google.py       # Google GenAI
│   ├── openai.py       # OpenAI SDK
│   ├── ollama.py       # Ollama (local models)
│   └── registry.py     # Provider registry
├── schema/
│   ├── validator.py    # Schema validation
│   └── healer.py       # Self-healing for invalid JSON
├── swarm/
│   ├── gateway.py      # HybridGateway — local + cloud dispatch
│   ├── pool.py         # LocalPool — Ollama pool
│   ├── classifier.py   # Privacy classifier
│   └── masker.py       # PII masking
├── timetravel/
│   ├── store.py        # EventStore (SQLite)
│   ├── snapshot.py     # State snapshots
│   ├── replay.py       # ReplayEngine
│   └── inspector.py    # Rich TUI visualizer
├── optimizer/
│   ├── collector.py    # Feedback collection
│   ├── evaluator.py    # Prompt variant evaluation
│   └── optimizer.py    # A/B prompt testing
├── models/
│   ├── config.py       # FrizuraConfig (pydantic-settings)
│   ├── budget.py       # Budget, BudgetConstraint
│   ├── execution.py    # PipelineResult, StepResult, LLMResponse
│   └── providers.py    # ModelInfo, CompletionConfig
└── cli/
    └── app.py          # Typer CLI
```

### Request flow

```
@task / engine.run()
        │
        ▼
  FrizuraEngine.run()
        │
        ├─ Pipeline.compile()      ← DAG validation, topological sort
        │
        ├─ for each step group:
        │   ├─ [parallel] asyncio.TaskGroup
        │   └─ execute_step()
        │         │
        │         ├─ SmartRouter.route()    ← if model not set explicitly
        │         │     ├─ ComplexityAnalyzer.analyze()
        │         │     ├─ filter by capability + budget
        │         │     └─ Strategy.select()
        │         │
        │         ├─ HybridGateway / Provider.complete()
        │         │
        │         ├─ Schema validation + healing loop (up to 3 attempts)
        │         │
        │         ├─ EventStore.append()   ← write to SQLite
        │         └─ StateSnapshot
        │
        └─ PipelineResult
```

---

## Development

```bash
# Install with dev dependencies
pip install -e ".[all]"

# Run tests
pytest

# Lint
ruff check src/

# Type check
mypy src/frizura/
```

### Test structure

```
tests/
├── conftest.py
├── test_engine.py      # FrizuraEngine — core scenarios
├── test_router.py      # SmartRouter — strategies, filtering
├── test_schema.py      # Schema Guard — validation, self-healing
├── test_swarm.py       # Hybrid Swarm — local/cloud routing
└── test_timetravel.py  # EventStore, ReplayEngine, Inspector
```
