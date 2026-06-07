# frizura

Next-gen LLM orchestrator: time-travel debugging, smart cost routing, guaranteed schemas, self-healing, hybrid swarm.

**Status:** Alpha · **Python:** 3.12+ · **License:** Apache 2.0

---

## Содержание

- [Что это](#что-это)
- [Возможности](#возможности)
- [Установка](#установка)
- [Быстрый старт](#быстрый-старт)
- [Концепции](#концепции)
  - [Task — одиночный вызов](#task--одиночный-вызов)
  - [Pipeline — DAG шагов](#pipeline--dag-шагов)
  - [Budget — бюджет](#budget--бюджет)
  - [Smart Router — авторотинг](#smart-router--авторотинг)
  - [Schema Guard — гарантия схемы](#schema-guard--гарантия-схемы)
  - [Time-Travel Debugging](#time-travel-debugging)
  - [Hybrid Swarm — локальные модели](#hybrid-swarm--локальные-модели)
- [Провайдеры и модели](#провайдеры-и-модели)
- [Конфигурация](#конфигурация)
- [CLI](#cli)
- [Примеры](#примеры)
- [Архитектура](#архитектура)
- [Разработка](#разработка)

---

## Что это

Frizura — Python-библиотека для оркестрации LLM-запросов. Она встаёт между вашим кодом и языковыми моделями и берёт на себя:

- выбор модели по цене/качеству/скорости автоматически
- гарантированный структурированный вывод через Pydantic
- сохранение каждого шага для последующей отладки и replay
- маршрутизацию чувствительных данных на локальные модели (Ollama)
- контроль бюджета на уровне задачи, шага или пайплайна

Никакого внешнего сервиса не нужно — работает как обычная Python-библиотека. SQLite для хранения событий поднимается автоматически.

---

## Возможности

| Фича | Описание |
|---|---|
| `@frizura.task` | Декоратор — превращает функцию в LLM-задачу |
| `Pipeline` + `Step` | DAG-builder: линейные, параллельные, условные шаги |
| Smart Router | Выбирает модель по complexity score, бюджету и стратегии |
| Cascade Routing | Пробует дешёвую модель сначала, эскалирует при неудаче |
| Schema Guard | Валидирует вывод LLM через Pydantic, само-исцеляет до 3 раз |
| Time-Travel | Каждый шаг → событие в SQLite, replay к любому шагу |
| Hybrid Swarm | Локальные Ollama-модели + облачные API, PII-маскирование |
| Budget Control | `Budget(max_cost=0.01)` — жёсткий лимит стоимости/времени |
| Rich TUI | `frizura inspect` — красивый дебаггер в терминале |

---

## Установка

```bash
# Полная установка (все провайдеры + CLI)
pip install frizura[all]

# Только нужные провайдеры
pip install frizura[anthropic]
pip install frizura[google]
pip install frizura[openai]
pip install frizura[ollama]

# Из исходников
git clone <repo>
cd orcestr
pip install -e ".[all]"
```

---

## Быстрый старт

**1. API-ключи** (env-переменные с префиксом `FRIZURA_`):

```bash
export FRIZURA_ANTHROPIC_API_KEY=sk-ant-...
export FRIZURA_GOOGLE_API_KEY=AIza...
export FRIZURA_OPENAI_API_KEY=sk-...
# или в .env файле
```

**2. Минимальный пример:**

```python
import asyncio
import frizura

@frizura.task(model="anthropic:claude-sonnet-4-6")
async def explain(topic: str) -> str:
    """Объясни простыми словами: {topic}"""

result = asyncio.run(explain("квантовая запутанность"))
print(result)
```

**3. Со структурированным выводом:**

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
    """Analyze review sentiment: {text}"""

result = asyncio.run(analyze("Отличный фильм, рекомендую!"))
print(result.sentiment, result.score)
```

---

## Концепции

### Task — одиночный вызов

`@frizura.task` — самый простой способ вызвать LLM. Под капотом создаёт Pipeline из одного шага.

```python
@frizura.task(
    model="anthropic:claude-sonnet-4-6",  # явная модель или None (авто-роутинг)
    output_schema=MyModel,                 # опционально — Pydantic-схема
    budget=frizura.Budget(max_cost=0.05),  # опционально — лимит стоимости
    privacy="auto",                        # "auto" | "local_only" | "cloud_only"
    system_prompt="You are an expert.",    # опционально
)
async def my_task(arg: str) -> str:
    """Prompt template with {arg} interpolation."""
```

Промпт формируется из docstring — `{arg}` подставляется из аргументов функции.

---

### Pipeline — DAG шагов

Для сложных многошаговых задач:

```python
from frizura.core.graph import Pipeline, Step, StepType

# Линейная цепочка
pipeline = (
    Pipeline("my-pipeline")
    .add_step(Step(
        name="draft",
        system_prompt="Write a draft.",
        handler=lambda ctx: ctx.get("input"),
    ))
    .add_step(Step(
        name="review",
        system_prompt="Review and improve.",
        handler=lambda ctx: f"Improve this: {ctx.get('draft')}",
    ))
)

# Параллельные шаги
pipeline.add_parallel(
    Step(name="translate-en", handler=lambda ctx: "Translate to EN"),
    Step(name="translate-de", handler=lambda ctx: "Translate to DE"),
)

# Условный branch
pipeline.add_branch(
    condition_fn=lambda ctx: "formal" if ctx.get("is_business") else "casual",
    branches={
        "formal": Step(name="formal-tone", ...),
        "casual": Step(name="casual-tone", ...),
    }
)

# Запуск
engine = frizura.FrizuraEngine()
result = await engine.run(pipeline, input_data="Исходный текст")
print(result.output)
print(f"Стоимость: ${result.total_cost_usd:.5f}")
print(f"Время: {result.total_duration_ms:.0f}ms")
```

#### Типы шагов (`StepType`)

| Тип | Описание |
|---|---|
| `LLM` | Вызов языковой модели (по умолчанию) |
| `TRANSFORM` | Кастомная Python-функция без LLM |
| `PARALLEL` | Параллельное выполнение группы шагов |
| `BRANCH` | Условный выбор ветки |
| `HUMAN` | Заглушка для human-in-the-loop (pending) |

---

### Budget — бюджет

Контроль расходов на уровне задачи или пайплайна:

```python
from frizura.models.budget import Budget

budget = Budget(
    max_cost=0.10,     # максимум $0.10
    max_tokens=50000,  # максимум 50k токенов
    max_time=30.0,     # максимум 30 секунд
    max_retries=3,     # максимум 3 повтора (для schema healing)
    prefer="cost",     # "cost" | "quality" | "speed" — влияет на стратегию
)

result = await engine.run(pipeline, budget=budget)
```

При превышении бюджета бросается `BudgetExhaustedError`.

---

### Smart Router — авторотинг

Если `model=None`, роутер выбирает модель автоматически.

**Как работает:**
1. Анализирует сложность запроса (`complexity score` от 0.0 до 1.0)
2. Фильтрует модели по нужным capability (JSON mode, tool calling)
3. Фильтрует по бюджету
4. Применяет стратегию выбора

**Стратегии:**

| Стратегия | Логика |
|---|---|
| `cascade` _(default)_ | Пробует дешёвую модель, эскалирует при ошибке |
| `cheapest` | Всегда самая дешёвая из подходящих |
| `fastest` | Минимальная задержка |
| `best_quality` | Топовый тир (premium) |

**Тиры моделей:** `cheap` → `standard` → `premium`

```python
# Явно задать стратегию
engine = frizura.FrizuraEngine(
    config=frizura.FrizuraConfig(
        router=RouterConfig(default_strategy="cheapest")
    )
)
```

---

### Schema Guard — гарантия схемы

Frizura гарантирует, что вывод LLM будет валидным Pydantic-объектом.

**Процесс:**
1. LLM возвращает текст
2. Извлекаем JSON (в том числе из markdown-фенсов ` ```json ... ``` `)
3. Валидируем через `Model.model_validate()`
4. Если невалидно → healing loop: отправляем ошибку обратно в ту же модель с просьбой исправить
5. До 3 попыток. Если все провалились → `SchemaHealingFailed`

```python
from pydantic import BaseModel

class Output(BaseModel):
    answer: str
    confidence: float
    sources: list[str]

@frizura.task(output_schema=Output)
async def query(question: str) -> Output:
    """Answer this question with sources: {question}"""
```

---

### Time-Travel Debugging

Каждый запуск сохраняет все события в SQLite (`.frizura/events.db`).

```bash
# Просмотр выполнения пайплайна
frizura inspect <pipeline-id>

# Replay до определённого шага
frizura replay <pipeline-id> --to step-name
```

**Что показывает `frizura inspect`:**
- Summary panel: ID, статус, стоимость, время, модели
- Step timeline: таблица шагов с иконками, длительностью, стоимостью
- Event tree: дерево событий сгруппированное по шагам
- Cost breakdown: расходы по моделям и шагам

**Типы событий:**

| Событие | Когда |
|---|---|
| `pipeline.started / completed / failed` | Старт/финиш пайплайна |
| `step.started / completed / failed / skipped` | Выполнение шага |
| `llm.request / llm.response` | Вызов модели |
| `routing.decision` | Выбор модели роутером |
| `schema.validation.ok / failed` | Проверка схемы |
| `schema.heal.attempt` | Попытка само-исцеления |
| `state.snapshot` | Снимок состояния контекста |
| `budget.exhausted` | Превышение бюджета |

```python
# Replay в коде
engine = frizura.FrizuraEngine()
result = await engine.run(pipeline, "input")

# Восстановить контекст к нужному шагу
ctx = await engine.replay(result.pipeline_id, until_step="step-id")
print(ctx.state)
```

---

### Hybrid Swarm — локальные модели

Frizura поддерживает Ollama для запуска моделей локально. Роутер автоматически маршрутизирует на локальные модели когда:
- задан `privacy="local_only"` или `privacy="auto"` и данные чувствительные
- бюджет ограничен и локальные модели дешевле

```bash
# Статус локального пула
frizura swarm status

# Обнаружить модели в локальной сети
frizura swarm discover
```

**Конфигурация Ollama:**

```python
from frizura.models.config import SwarmConfig

config = frizura.FrizuraConfig(
    swarm=SwarmConfig(
        enabled=True,
        local_first=True,                              # предпочитать локальные
        ollama_hosts=["http://localhost:11434"],        # хосты Ollama
        privacy_mode="auto",                           # "auto" | "local_only" | "cloud_only"
    )
)
```

---

## Провайдеры и модели

### Поддерживаемые провайдеры

| Провайдер | Пакет | Ключ |
|---|---|---|
| OpenAI | `frizura[openai]` | `FRIZURA_OPENAI_API_KEY` |
| Anthropic | `frizura[anthropic]` | `FRIZURA_ANTHROPIC_API_KEY` |
| Google | `frizura[google]` | `FRIZURA_GOOGLE_API_KEY` |
| Ollama | `frizura[ollama]` | — (локально) |

### Формат модели

Модель задаётся строкой `"provider:model_id"`:

```python
"openai:gpt-4o"
"openai:gpt-4o-mini"
"anthropic:claude-sonnet-4-6"
"anthropic:claude-opus-4-6"
"google:gemini-2.0-flash"
"google:gemini-2.5-pro"
"ollama:llama3.3"
"ollama:mistral"
"ollama:phi4"
"ollama:qwen2.5"
```

---

## Конфигурация

Все настройки через переменные окружения с префиксом `FRIZURA_` или `.env` файл.

```env
# Провайдеры
FRIZURA_OPENAI_API_KEY=sk-...
FRIZURA_ANTHROPIC_API_KEY=sk-ant-...
FRIZURA_GOOGLE_API_KEY=AIza...

# Дефолтная модель (если model=None в task/step)
FRIZURA_DEFAULT_MODEL=anthropic:claude-sonnet-4-6

# Логирование
FRIZURA_LOG_LEVEL=INFO
FRIZURA_VERBOSE=false
```

Или через объект конфига:

```python
from frizura.models.config import FrizuraConfig, RouterConfig, SwarmConfig, TimeTravelConfig

config = FrizuraConfig(
    default_model="anthropic:claude-sonnet-4-6",
    log_level="DEBUG",
    router=RouterConfig(
        default_strategy="cascade",
        complexity_threshold_cheap=0.3,    # score < 0.3 → cheap тир
        complexity_threshold_premium=0.7,  # score > 0.7 → premium тир
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
# Запустить пайплайн из Python-файла
frizura run examples/03_pipeline.py
frizura run my_pipeline.py --budget-cost 0.10 --budget-time 60

# Time-travel дебаггер
frizura inspect <pipeline-id>

# Replay до конкретного шага
frizura replay <pipeline-id> --to <step-id>

# Локальный пул моделей
frizura swarm status
frizura swarm discover
```

`frizura run` ищет объект `Pipeline` в указанном файле и запускает его.

---

## Примеры

### Простая задача с авто-роутингом

```python
import frizura

@frizura.task(budget=frizura.Budget(max_cost=0.005, prefer="cost"))
async def summarize(text: str) -> str:
    """Summarize in 3 sentences: {text}"""

result = await summarize(long_article)
```

### Структурированный вывод

```python
from pydantic import BaseModel
import frizura

class CodeReview(BaseModel):
    issues: list[str]
    suggestions: list[str]
    score: int  # 1-10

@frizura.task(model="anthropic:claude-sonnet-4-6", output_schema=CodeReview)
async def review_code(code: str) -> CodeReview:
    """Review this Python code for quality and bugs:\n\n{code}"""

review = await review_code(my_function)
print(f"Score: {review.score}/10")
print(f"Issues: {review.issues}")
```

### Многошаговый пайплайн

```python
from frizura.core.graph import Pipeline, Step

async def main():
    pipeline = (
        Pipeline("article-pipeline")
        .add_step(Step(
            name="research",
            system_prompt="You are a researcher.",
            handler=lambda ctx: f"Research this topic: {ctx.get('input')}",
        ))
        .add_step(Step(
            name="write",
            system_prompt="You are a writer.",
            handler=lambda ctx: f"Write an article based on: {ctx.get('research')}",
        ))
        .add_step(Step(
            name="edit",
            system_prompt="You are an editor.",
            handler=lambda ctx: f"Edit and improve: {ctx.get('write')}",
            model="anthropic:claude-opus-4-6",  # шаг с явной тяжёлой моделью
        ))
    )

    engine = frizura.FrizuraEngine()
    result = await engine.run(pipeline, "quantum computing for beginners")

    print(result.output)
    print(f"Шаги: {len(result.steps)}")
    print(f"Стоимость: ${result.total_cost_usd:.4f}")
    print(f"Pipeline ID: {result.pipeline_id}")
    # frizura inspect <pipeline_id>  ← для дебаггинга
```

### Параллельные шаги

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

## Архитектура

```
frizura/
├── core/
│   ├── engine.py       # FrizuraEngine — главный оркестратор
│   ├── graph.py        # Pipeline, Step, DAG-компилятор (Kahn's алгоритм)
│   ├── context.py      # ExecutionContext — состояние выполнения
│   ├── decorators.py   # @task, @step декораторы
│   └── events.py       # Типы событий
├── router/
│   ├── router.py       # SmartRouter — выбор модели
│   ├── analyzer.py     # ComplexityAnalyzer — оценка сложности запроса
│   ├── calculator.py   # CostCalculator — расчёт стоимости
│   └── strategies.py   # cascade, cheapest, fastest, best_quality
├── providers/
│   ├── base.py         # LLMProvider — абстракция
│   ├── anthropic.py    # Anthropic SDK
│   ├── google.py       # Google GenAI
│   ├── openai.py       # OpenAI SDK
│   └── ollama.py       # Ollama (локальные модели)
├── schema/
│   ├── validator.py    # SchemaValidator
│   └── healer.py       # SchemaHealer — само-исцеление
├── swarm/
│   ├── gateway.py      # HybridGateway — local + cloud
│   ├── pool.py         # LocalPool — Ollama-пул
│   ├── classifier.py   # PrivacyClassifier
│   └── masker.py       # PIIMasker — маскирование PII
├── timetravel/
│   ├── store.py        # EventStore (SQLite)
│   ├── snapshot.py     # StateSnapshot
│   ├── replay.py       # ReplayEngine
│   └── inspector.py    # Rich TUI инспектор
├── optimizer/
│   ├── collector.py    # Сбор обратной связи
│   ├── evaluator.py    # Оценка вариантов промптов
│   └── optimizer.py    # A/B тестирование промптов
├── models/
│   ├── config.py       # FrizuraConfig (pydantic-settings)
│   ├── budget.py       # Budget, BudgetConstraint
│   ├── execution.py    # PipelineResult, StepResult, LLMResponse
│   └── providers.py    # ModelInfo, CompletionConfig
└── cli/
    └── app.py          # Typer CLI
```

### Поток выполнения

```
@task / engine.run()
        │
        ▼
  FrizuraEngine.run()
        │
        ├─ Pipeline.compile()  ← DAG-валидация, топологическая сортировка
        │
        ├─ для каждой группы шагов:
        │   ├─ [параллельные] asyncio.TaskGroup
        │   └─ execute_step()
        │         │
        │         ├─ SmartRouter.route()  ← если model не задан явно
        │         │     ├─ ComplexityAnalyzer.analyze()
        │         │     ├─ фильтр по capability + budget
        │         │     └─ Strategy.select()
        │         │
        │         ├─ HybridGateway / Provider.complete()
        │         │
        │         ├─ Schema validation + healing loop
        │         │
        │         ├─ EventStore.append()  ← SQLite
        │         └─ StateSnapshot
        │
        └─ PipelineResult
```

---

## Разработка

```bash
# Установка с dev-зависимостями
pip install -e ".[all]"
pip install pytest pytest-asyncio ruff mypy

# Тесты
pytest

# Линтер
ruff check src/

# Типизация
mypy src/frizura/
```

### Структура тестов

```
tests/
├── conftest.py
├── test_engine.py      # FrizuraEngine — основные сценарии
├── test_router.py      # SmartRouter — стратегии, фильтрация
├── test_schema.py      # Schema Guard — валидация, healing
├── test_swarm.py       # Hybrid Swarm — local/cloud маршрутизация
└── test_timetravel.py  # EventStore, ReplayEngine, Inspector
```
