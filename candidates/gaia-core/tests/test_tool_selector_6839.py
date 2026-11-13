"""Regression test for GAIA_Project-6839: asked mid-chat to note or file
something she'd just found, select_tool() saw only the bare instruction
("please file that as an issue") with no conversation history -- nothing
to resolve "that" against, and no evidence to put in issue(file)'s params
-- so she reached for file(write) instead (rejected: "Path not allowed").

Pins: recent turns (packet.context.relevant_history_snippet) now reach the
selection prompt, and the worked-example block names "issue"/"note" instead
of always anchoring on "file"/"read" alone.
"""
import json

from gaia_common.protocols.cognition_packet import (
    CognitionPacket, Content, Context, Header, Intent, RelevantHistorySnippet,
)
from gaia_core.cognition.tool_selector import _recent_conversation, select_tool


def _packet(user_text: str, snippets=None):
    return CognitionPacket(
        header=Header(session_id="s1", packet_id="p1"),
        content=Content(original_prompt=user_text),
        intent=Intent(user_intent="tool_use"),
        context=Context(relevant_history_snippet=snippets or []),
    )


class _CapturingModel:
    """Not named Llama/VLLMRemoteModel -> select_tool() calls
    create_chat_completion(**kwargs) directly, no grammar/timeout path."""
    def __init__(self, selected_tool="issue", params=None):
        self.last_messages = None
        self._selected_tool = selected_tool
        self._params = params or {}

    def create_chat_completion(self, messages, **kwargs):
        self.last_messages = messages
        body = {
            "selected_tool": self._selected_tool,
            "params": json.dumps(self._params),
            "reasoning": "test",
            "confidence": 0.8,
        }
        return {"choices": [{"message": {"content": json.dumps(body)}}]}


def test_recent_conversation_formats_history_as_text():
    snippets = [
        RelevantHistorySnippet(id="1", role="user", summary="What's in the vector index?"),
        RelevantHistorySnippet(id="2", role="assistant", summary="Found a missing vector index entry for embed_documents."),
    ]
    text = _recent_conversation(_packet("file that", snippets))
    assert "missing vector index" in text
    assert text.startswith("user:")


def test_recent_conversation_empty_when_no_history():
    assert _recent_conversation(_packet("file that")) == ""


def test_select_tool_prompt_carries_recent_history_for_pronoun_resolution():
    snippets = [RelevantHistorySnippet(
        id="1", role="assistant",
        summary="I found a missing vector index entry for embed_documents.",
    )]
    model = _CapturingModel(selected_tool="issue", params={"action": "file", "title": "x", "symptom": "y", "evidence": "z"})
    select_tool(_packet("Please file that as an issue", snippets), "Please file that as an issue", model)
    prompt_text = model.last_messages[-1]["content"]
    assert "missing vector index entry" in prompt_text
    assert "RECENT CONVERSATION" in prompt_text


def test_select_tool_prompt_omits_history_block_when_none():
    model = _CapturingModel(selected_tool="", params={})
    select_tool(_packet("What's the weather like today?"), "What's the weather like today?", model)
    prompt_text = model.last_messages[-1]["content"]
    assert "RECENT CONVERSATION" not in prompt_text


def test_select_tool_prompt_example_names_issue_note_not_only_file():
    model = _CapturingModel(selected_tool="", params={})
    select_tool(_packet("What's the weather like today?"), "What's the weather like today?", model)
    prompt_text = model.last_messages[-1]["content"]
    assert 'selected_tool="issue"' in prompt_text
    assert '"action": "note"' in prompt_text
