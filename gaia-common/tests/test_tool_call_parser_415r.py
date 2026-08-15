"""Tests for gaia_common.utils.tool_call_parser.ToolCallParser.

No test file existed for this module before GAIA_Project-415r, so this
covers both the new <tool_calling: domain.action> variant (the actual bug
fix) and baseline regression coverage for the pre-existing <tool_call> JSON
and meta-verb formats, which had zero automated coverage.

Context (415r): live 2026-08-15, Prime emitted
'<tool_calling: knowledge.search> {"query": "King Arthur"} </tool_calling>'
— a 4th tag format (domain.action embedded IN the open tag, bare params
JSON body) that ToolCallParser didn't recognize at all, so it leaked to the
user as raw text instead of executing. Same failure class as GAIA_Project-pfdw.
"""

from gaia_common.utils.tool_call_parser import (
    ToolCallParser,
    ParseEventType,
)


def _feed_all(parser, text):
    events = list(parser.feed(text))
    events += parser.flush()
    return events


def _feed_char_by_char(parser, text):
    """Simulate true token-level streaming — one character per feed() call."""
    events = []
    for ch in text:
        events.extend(parser.feed(ch))
    events.extend(parser.flush())
    return events


class TestToolCallingVariant:
    """The new <tool_calling: domain.action>{params}</tool_calling> format."""

    def test_basic_detection_single_feed(self):
        p = ToolCallParser()
        text = '<tool_calling: knowledge.search> {"query": "King Arthur"} </tool_calling>'
        events = _feed_all(p, text)
        detected = [e for e in events if e.type == ParseEventType.TOOL_CALL_DETECTED]
        assert len(detected) == 1
        assert detected[0].tool_name == "knowledge"
        assert detected[0].tool_action == "search"
        assert detected[0].tool_params == {"query": "King Arthur"}

    def test_preamble_text_before_tag_is_preserved(self):
        p = ToolCallParser()
        text = (
            "I'll search the conversation history for mentions of King Arthur.\n"
            '<tool_calling: knowledge.search> {"query": "King Arthur"} </tool_calling>'
        )
        events = _feed_all(p, text)
        text_events = [e for e in events if e.type == ParseEventType.TEXT]
        joined = "".join(e.text for e in text_events)
        assert "I'll search the conversation history" in joined
        assert "<tool_calling" not in joined

    def test_no_raw_tag_text_ever_reaches_a_text_event(self):
        """The literal opening/closing tag markers must never appear in any
        TEXT event — that's the actual leak this bug fixes."""
        p = ToolCallParser()
        text = '<tool_calling: knowledge.search> {"query": "King Arthur"} </tool_calling>'
        events = _feed_all(p, text)
        for e in events:
            if e.type == ParseEventType.TEXT:
                assert "<tool_calling" not in e.text
                assert "</tool_calling>" not in e.text

    def test_hallucinated_trailing_tool_result_text_still_parses_cleanly(self):
        """Prime also fabricated a fake '[Tool result: ...]' block after the
        real tag in the live incident. The parser should still detect the
        real tool call correctly regardless (main.py's caller is responsible
        for discarding anything after the first TOOL_CALL_DETECTED event —
        this test only pins the parser's own event stream)."""
        p = ToolCallParser()
        text = (
            '<tool_calling: knowledge.search> {"query": "King Arthur"} </tool_calling>\n\n'
            '[Tool result: fake fabricated content]'
        )
        events = _feed_all(p, text)
        detected = [e for e in events if e.type == ParseEventType.TOOL_CALL_DETECTED]
        assert len(detected) == 1
        assert detected[0].tool_name == "knowledge"
        assert detected[0].tool_action == "search"

    def test_streamed_char_by_char(self):
        """The header ('<tool_calling: knowledge.search>') must be correctly
        assembled even when it arrives one character at a time, since real
        generation streams token-by-token, not as one string."""
        p = ToolCallParser()
        text = '<tool_calling: knowledge.search> {"query": "King Arthur"} </tool_calling>'
        events = _feed_char_by_char(p, text)
        detected = [e for e in events if e.type == ParseEventType.TOOL_CALL_DETECTED]
        assert len(detected) == 1
        assert detected[0].tool_name == "knowledge"
        assert detected[0].tool_action == "search"
        assert detected[0].tool_params == {"query": "King Arthur"}

    def test_domain_with_no_action_dot(self):
        p = ToolCallParser()
        text = '<tool_calling: worldstate> {} </tool_calling>'
        events = _feed_all(p, text)
        detected = [e for e in events if e.type == ParseEventType.TOOL_CALL_DETECTED]
        assert len(detected) == 1
        assert detected[0].tool_name == "worldstate"
        assert detected[0].tool_action == ""

    def test_malformed_params_json_yields_tool_error_not_a_leak(self):
        p = ToolCallParser()
        text = '<tool_calling: knowledge.search> {not valid json at all </tool_calling>'
        events = _feed_all(p, text)
        assert any(e.type == ParseEventType.TOOL_ERROR for e in events)
        # And critically: the raw tag text must not show up in a TEXT event.
        for e in events:
            if e.type == ParseEventType.TEXT:
                assert "<tool_calling" not in e.text

    def test_empty_params_body_defaults_to_empty_dict(self):
        p = ToolCallParser()
        text = '<tool_calling: worldstate.current></tool_calling>'
        events = _feed_all(p, text)
        detected = [e for e in events if e.type == ParseEventType.TOOL_CALL_DETECTED]
        assert len(detected) == 1
        assert detected[0].tool_params == {}

    def test_unclosed_header_flushes_as_text_not_dropped(self):
        """Generation cut off mid-header (never saw the closing '>') — must
        not silently vanish; flush() should surface it as text."""
        p = ToolCallParser()
        p.feed("some text <tool_calling: knowledge.sea")
        events = p.flush()
        joined = "".join(e.text for e in events if e.type == ParseEventType.TEXT)
        assert "knowledge.sea" in joined


class TestExistingFormatsRegression:
    """Baseline coverage for the pre-existing formats — none existed before."""

    def test_json_tool_call_format(self):
        p = ToolCallParser()
        text = '<tool_call>{"tool": "web", "action": "search", "query": "foo"}</tool_call>'
        events = _feed_all(p, text)
        detected = [e for e in events if e.type == ParseEventType.TOOL_CALL_DETECTED]
        assert len(detected) == 1
        assert detected[0].tool_name == "web"
        assert detected[0].tool_action == "search"
        assert detected[0].tool_params == {"query": "foo"}

    def test_meta_verb_format(self):
        p = ToolCallParser()
        text = '<|tool>search(query="current time")<tool|>'
        events = _feed_all(p, text)
        detected = [e for e in events if e.type == ParseEventType.TOOL_CALL_DETECTED]
        assert len(detected) == 1
        assert detected[0].tool_name == "search"
        assert detected[0].tool_params == {"query": "current time"}

    def test_plain_text_with_no_tags_passes_through(self):
        p = ToolCallParser()
        events = _feed_all(p, "Just a normal reply, nothing special.")
        assert all(e.type == ParseEventType.TEXT for e in events)
        assert "".join(e.text for e in events) == "Just a normal reply, nothing special."
