"""Unit tests for Frizura time-travel debugging (Event Store, Replay, Fork)."""

from __future__ import annotations

from datetime import datetime, timezone
import pytest

from frizura.core.context import ExecutionContext
from frizura.core.events import Event, EventType, StateSnapshot
from frizura.models.config import FrizuraConfig
from frizura.timetravel.store import EventStore
from frizura.timetravel.replay import ReplayEngine


@pytest.mark.asyncio
async def test_event_store_lifecycle(tmp_path) -> None:
    """Test saving and loading events/snapshots in SQLite."""
    db_file = tmp_path / "test_events.db"
    store = EventStore(db_path=str(db_file))
    
    await store.init()
    
    # Append event
    pipeline_id = "test-pipe-123"
    event = Event(
        event_type=EventType.PIPELINE_STARTED,
        pipeline_id=pipeline_id,
        data={"name": "test"},
    )
    
    await store.append(event)
    
    # Retrieve events
    events = await store.get_events(pipeline_id)
    assert len(events) == 1
    assert events[0].event_type == EventType.PIPELINE_STARTED
    assert events[0].data == {"name": "test"}
    
    # Save snapshot
    snapshot = StateSnapshot(
        pipeline_id=pipeline_id,
        step_id="step-1",
        step_index=0,
        state={"x": 42},
        memory=[],
    )
    await store.save_snapshot(snapshot)
    
    # Get snapshot
    saved_snap = await store.get_snapshot(pipeline_id, "step-1")
    assert saved_snap is not None
    assert saved_snap.state == {"x": 42}
    
    await store.close()


@pytest.mark.asyncio
async def test_replay_engine_reconstruction(tmp_path, register_mock_provider) -> None:
    """Test using ReplayEngine to reconstruct execution state from events."""
    db_file = tmp_path / "replay_test.db"
    
    # Build a config pointing to the temp database
    config = FrizuraConfig()
    config.timetravel.db_path = db_file
    
    store = EventStore(db_path=str(db_file))
    await store.init()
    
    pipeline_id = "test-replay-999"
    
    # Emit events simulating a pipeline run
    # 1. Pipeline started
    await store.append(Event(event_type=EventType.PIPELINE_STARTED, pipeline_id=pipeline_id))
    # 2. Step 1 started
    await store.append(Event(event_type=EventType.STEP_STARTED, pipeline_id=pipeline_id, step_id="step-1", step_name="step-1"))
    # 3. Step 1 completed (saves state snapshot)
    await store.append(Event(event_type=EventType.STEP_COMPLETED, pipeline_id=pipeline_id, step_id="step-1", step_name="step-1"))
    await store.save_snapshot(StateSnapshot(
        pipeline_id=pipeline_id,
        step_id="step-1",
        step_index=0,
        state={"result": "step 1 finished", "score": 10},
        memory=[],
    ))
    
    await store.close()
    
    # Replay state using ReplayEngine
    replay_engine = ReplayEngine(db_path=str(db_file))
    ctx = await replay_engine.replay_to(pipeline_id, "step-1")
    
    assert ctx.state.get("result") == "step 1 finished"
    assert ctx.state.get("score") == 10
    
    # Test forking
    fork_id = await replay_engine.fork_and_modify(pipeline_id, "step-1", {"score": 20})
    
    # Verify fork has modified state
    fork_ctx = await replay_engine.replay_to(fork_id, "step-1")
    assert fork_ctx.state.get("result") == "step 1 finished"
    assert fork_ctx.state.get("score") == 20
