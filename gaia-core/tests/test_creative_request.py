"""im6c: a creative request is composed, not grounded."""
import pytest

from gaia_core.cognition.nlu.intent_detection import is_creative_request


@pytest.mark.parametrize("text", [
    "Write me a haiku about autumn rain.",
    "Compose a limerick about a cat who hates Mondays.",
    "Tell me a joke about programmers.",
    "Write a short poem about the sea.",
    "Create a riddle whose answer is a shadow.",
    "Write a four-line song verse about a road trip.",
    "Can you write me a funny story about a dragon?",
    "please make up a bedtime story",
])
def test_creative(text):
    assert is_creative_request(text)


@pytest.mark.parametrize("text", [
    "Who betrayed King Arthur?",
    "Write a summary of the article at https://example.com/a",
    "Write a file called notes.txt with hello",
    "Tell me about the poem The Raven.",
    "What is a haiku?",
    "Recite the first stanza of The Raven",
    "write the poem's second verse from memory: Ozymandias",
])
def test_not_creative(text):
    assert not is_creative_request(text)
