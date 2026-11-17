"""vdaj: after an inline tool call only the reply she gave is kept, not also the
raw text with the call markup in it."""
import pytest

from gaia_core.config import get_config
from gaia_core.memory.session_manager import SessionManager


@pytest.fixture
def sm():
    return SessionManager(get_config())


@pytest.fixture
def sid(sm):
    _sid = "test_raw_tool_call_reply_vdaj"
    sm.sessions.pop(_sid, None)
    yield _sid
    sm.sessions.pop(_sid, None)


def _turns(sm, sid):
    return [(t["role"], t["content"]) for t in sm.get_or_create_session(sid).history]


def test_raw_call_reply_is_replaced(sm, sid):
    sm.add_message(sid, "user", "who are you?")
    sm.add_message(sid, "assistant", 'I\'m GAIA. <tool_call>{"tool": "introspect"}</tool_call>')
    assert sm.drop_raw_tool_call_reply(sid) is True
    sm.add_message(sid, "assistant", "I'm GAIA, built by Azrael.")
    assert _turns(sm, sid) == [("user", "who are you?"), ("assistant", "I'm GAIA, built by Azrael.")]


def test_plain_reply_is_kept(sm, sid):
    sm.add_message(sid, "user", "hi")
    sm.add_message(sid, "assistant", "hello")
    assert sm.drop_raw_tool_call_reply(sid) is False
    assert len(_turns(sm, sid)) == 2


def test_user_turn_is_never_dropped(sm, sid):
    sm.add_message(sid, "user", "what does <tool_call> mean?")
    assert sm.drop_raw_tool_call_reply(sid) is False
    assert len(_turns(sm, sid)) == 1
