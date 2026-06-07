"""Feedback collector for gathering task evaluation datasets.

Saves user or auto-evaluator feedback (scores and text feedback) alongside
inputs/outputs in a SQLite database to build training datasets for optimization.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import aiosqlite
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TrainingExample(BaseModel):
    """An example of input, output, and rating for prompt optimization."""

    input_data: str
    expected_output: str | None = None
    score: float = 1.0
    feedback: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class FeedbackCollector:
    """Collects and stores evaluation feedback for pipeline steps."""

    def __init__(self, db_path: str | Path = ".frizura/events.db") -> None:
        self.db_path = Path(db_path)
        self._initialized = False

    async def init(self) -> None:
        """Initialise database tables for feedback collection."""
        if self._initialized:
            return
        
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS step_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pipeline_id TEXT,
                    step_id TEXT,
                    step_name TEXT,
                    score REAL,
                    feedback TEXT,
                    input_data TEXT,
                    output_data TEXT,
                    metadata TEXT,
                    timestamp TEXT
                )
                """
            )
            await db.commit()
        self._initialized = True

    async def record(
        self,
        pipeline_id: str,
        step_id: str,
        step_name: str,
        score: float,
        feedback: str | None = None,
        input_data: Any = None,
        output_data: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a feedback entry for a step execution."""
        await self.init()
        
        input_str = json.dumps(input_data) if input_data is not None else ""
        output_str = json.dumps(output_data) if output_data is not None else ""
        meta_str = json.dumps(metadata or {})
        ts_str = datetime.now(timezone.utc).isoformat()

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO step_feedback 
                (pipeline_id, step_id, step_name, score, feedback, input_data, output_data, metadata, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (pipeline_id, step_id, step_name, score, feedback, input_str, output_str, meta_str, ts_str),
            )
            await db.commit()
        logger.debug("Feedback recorded for step %s: score=%f", step_name, score)

    async def get_dataset(self, step_name: str, min_score: float | None = None) -> list[TrainingExample]:
        """Retrieve collected feedback as a training dataset."""
        await self.init()
        query = "SELECT input_data, output_data, score, feedback, metadata, timestamp FROM step_feedback WHERE step_name = ?"
        params: list[Any] = [step_name]

        if min_score is not None:
            query += " AND score >= ?"
            params.append(min_score)

        examples = []
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(query, params) as cursor:
                async for row in cursor:
                    try:
                        meta = json.loads(row[4]) if row[4] else {}
                    except Exception:
                        meta = {}
                    
                    # Clean up input/output from double json string serialization
                    inp = row[0]
                    try:
                        inp_parsed = json.loads(inp)
                        if isinstance(inp_parsed, str):
                            inp = inp_parsed
                    except Exception:
                        pass

                    out = row[1]
                    try:
                        out_parsed = json.loads(out)
                        if isinstance(out_parsed, str):
                            out = out_parsed
                    except Exception:
                        pass

                    examples.append(
                        TrainingExample(
                            input_data=str(inp),
                            expected_output=str(out),
                            score=row[2],
                            feedback=row[3],
                            metadata=meta,
                            timestamp=datetime.fromisoformat(row[5]) if row[5] else datetime.now(timezone.utc),
                        )
                    )
        return examples

    async def get_stats(self, step_name: str) -> dict[str, Any]:
        """Get feedback stats for a step."""
        await self.init()
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT COUNT(*), AVG(score) FROM step_feedback WHERE step_name = ?", (step_name,)
            ) as cursor:
                row = await cursor.fetchone()
                if row and row[0] > 0:
                    return {"count": row[0], "avg_score": row[1]}
        return {"count": 0, "avg_score": 0.0}
