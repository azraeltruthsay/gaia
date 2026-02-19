"""Tests for MessageQueue file-backed persistence."""

import asyncio
import json
import pytest
import tempfile
from pathlib import Path

from gaia_web.queue.message_queue import MessageQueue, QueuedMessage


@pytest.fixture
def queue_file(tmp_path):
    return str(tmp_path / "test_queue.json")


@pytest.mark.asyncio
async def test_enqueue_persists_to_disk(queue_file):
    """Messages should be written to disk on enqueue."""
    mq = MessageQueue(core_url="http://fake:6415", queue_file=queue_file)

    msg = QueuedMessage(
        message_id="msg-1",
        content="Hello",
        source="test",
        session_id="test-session",
    )
    await mq.enqueue(msg)

    # Verify file exists and contains the message
    data = json.loads(Path(queue_file).read_text())
    assert len(data) == 1
    assert data[0]["message_id"] == "msg-1"
    assert data[0]["content"] == "Hello"


@pytest.mark.asyncio
async def test_dequeue_updates_disk(queue_file):
    """Dequeue should update the persisted file."""
    mq = MessageQueue(core_url="http://fake:6415", queue_file=queue_file)

    msg1 = QueuedMessage(message_id="msg-1", content="A", source="test", session_id="s1")
    msg2 = QueuedMessage(message_id="msg-2", content="B", source="test", session_id="s2")
    await mq.enqueue(msg1)
    await mq.enqueue(msg2)

    dequeued = await mq.dequeue()
    assert dequeued.message_id == "msg-1"

    # Disk should now have only msg-2
    data = json.loads(Path(queue_file).read_text())
    assert len(data) == 1
    assert data[0]["message_id"] == "msg-2"


@pytest.mark.asyncio
async def test_restore_on_init(queue_file):
    """A new MessageQueue should restore messages from an existing file."""
    # Seed the file manually
    seed = [
        {
            "message_id": "restored-1",
            "content": "I survived a restart",
            "source": "discord",
            "session_id": "sess-1",
            "priority": 0,
            "queued_at": "2026-02-19T12:00:00+00:00",
            "metadata": {},
        }
    ]
    Path(queue_file).write_text(json.dumps(seed))

    mq = MessageQueue(core_url="http://fake:6415", queue_file=queue_file)

    status = await mq.get_queue_status()
    assert status["count"] == 1

    msg = await mq.dequeue()
    assert msg.message_id == "restored-1"
    assert msg.content == "I survived a restart"


@pytest.mark.asyncio
async def test_empty_queue_clears_file(queue_file):
    """Dequeuing all messages should leave an empty list in the file."""
    mq = MessageQueue(core_url="http://fake:6415", queue_file=queue_file)

    msg = QueuedMessage(message_id="msg-1", content="X", source="test", session_id="s1")
    await mq.enqueue(msg)
    await mq.dequeue()

    data = json.loads(Path(queue_file).read_text())
    assert data == []


def test_queued_message_roundtrip():
    """QueuedMessage should serialize/deserialize correctly."""
    msg = QueuedMessage(
        message_id="rt-1",
        content="roundtrip test",
        source="web",
        session_id="s1",
        priority=5,
        metadata={"key": "value"},
    )
    d = msg.to_dict()
    restored = QueuedMessage.from_dict(d)
    assert restored.message_id == msg.message_id
    assert restored.content == msg.content
    assert restored.priority == msg.priority
    assert restored.metadata == msg.metadata
