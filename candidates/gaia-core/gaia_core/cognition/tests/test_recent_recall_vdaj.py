"""vdaj: a question about the conversation right now is answered by the turns
on screen, never by a search for the phrase."""
import pytest

from gaia_core.cognition.nlu.intent_detection import is_recent_recall


@pytest.mark.parametrize("text", [
    "what were we just talking about?",
    "What were we talking about?",
    "what have we been discussing?",
    "where were we?",
    "ok, where were we before that?",
    "what did I just ask you?",
    "what did you just say?",
    "what was I saying?",
    "what did I ask you a moment ago?",
    "can you recap this conversation?",
])
def test_recent_recall_detected(text):
    assert is_recent_recall(text) is True


@pytest.mark.parametrize("text", [
    "what were the causes of World War One?",
    "what are we going to do about the taxes?",
    "what did Einstein say about time?",
    "what did you have in mind for dinner",
    "What did we talk about regarding the memory palace last week?",
    "tell me about Canberra",
    "",
])
def test_not_recent_recall(text):
    assert is_recent_recall(text) is False
