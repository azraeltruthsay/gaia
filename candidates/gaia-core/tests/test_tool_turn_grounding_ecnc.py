"""ecnc: the continuation gets the tool's result as plain text, ahead of the
request, not a replay of her own tool-call turn plus XML/JSON to imitate;
the prompt shows the question-level grounding and pre-run search results;
and no automatic stop at "</tool_call>" (it emptied tool-free replies)."""

from unittest.mock import patch

from gaia_common.utils.tool_call_parser import tool_result_text
from gaia_core.models.vllm_remote_model import VLLMRemoteModel

TOOL_SYS = {"role": "system", "content": 'TOOLS: Use <tool_call>{"tool":"name","action":"verb"}</tool_call> to call tools.'}
USER = {"role": "user", "content": "What's on https://httpbin.org/html?"}


def _model():
    m = VLLMRemoteModel({"endpoint": "http://engine:8092", "path": "core"})
    m._resolve_model_field = lambda: "core"
    return m


def _call(messages, reply, **kw):
    m = _model()
    sent = {}

    def fake_post(path, payload, **_):
        sent.update(payload)
        return {"choices": [{"message": {"content": reply}, "finish_reason": "stop"}]}
    with patch.object(m, "_post", side_effect=fake_post), patch.object(m, "_log_usage"):
        out = m.create_chat_completion(messages=messages, max_tokens=64, **kw)
    return sent, out["choices"][0]["message"]["content"]


def test_no_automatic_stop_text_after_a_suppressed_call_survives():
    # Tool-free intents keep what Core writes after a (suppressed) call; an
    # automatic stop at </tool_call> emptied "What is the current date?".
    reply = '<tool_call>{"tool": "worldbuild", "action": "status"}</tool_call> It is Wednesday, October 28.'
    sent, text = _call([TOOL_SYS, USER], reply)
    assert "stop" not in sent and text == reply


def test_explicit_stop_is_forwarded():
    sent, _ = _call([TOOL_SYS, USER], "ok", stop=["</tool_call>"], include_stop_str_in_output=True)
    assert sent["stop"] == ["</tool_call>"] and sent["include_stop_str_in_output"] is True


def test_tool_result_text_reads_as_plain_text():
    page = tool_result_text({"title": "Moby-Dick", "url": "https://httpbin.org/html",
                             "content": "Availing himself of the mild, summer-cool weather..."})
    assert page.splitlines() == ["Moby-Dick", "https://httpbin.org/html",
                                 "Availing himself of the mild, summer-cool weather..."]
    recall = tool_result_text({"ok": True, "lookups": [
        {"fact": "From the hand of the woman in the Lake.", "source": "Quizlet",
         "url": "https://q", "looked_up": "2026-10-28", "stale": True}],
        "results": [{"content": "technical|infrastructure|2026-10-27|journal"}]})
    assert recall == "- From the hand of the woman in the Lake. (Quizlet, looked up 2026-10-28, https://q) [stale]"
    assert tool_result_text({"ok": False, "error": "Unknown action", "hint": "Say so."}) == "Failed: Unknown action\nSay so."
    assert tool_result_text({"results": [], "note": "Nothing remembered."}) == "Nothing remembered."
    assert tool_result_text("x" * 5000).endswith("(truncated)")
    assert "<tool_result>" not in page + recall


def _packet(fields):
    from gaia_common.protocols.cognition_packet import CognitionPacket, Content, DataField
    return CognitionPacket(content=Content(
        original_prompt="Search the web for who wrote The Name of the Rose and tell me.",
        data_fields=[DataField(key=k, value=v, type="json") for k, v in fields]))


def _prompt_text(packet):
    from gaia_core.utils import prompt_builder
    return " ".join(str(m.get("content", "")) for m in prompt_builder.build_from_packet(packet))


PROBE = {"The Name": {"results": [{"title": "NameMC", "snippet": "Check the availability of Minecraft names.",
                                   "url": "https://namemc.com/"}], "source": "web_search", "query": "The Name"}}
ROUTER = {"q": {"results": [{"title": "web", "snippet": "The Name of the Rose is the 1980 debut novel by Umberto Eco.",
                             "url": "https://en.wikipedia.org/wiki/The_Name_of_the_Rose"}],
                "source": "knowledge_router:web", "query": "q"}}


def test_router_grounding_beats_the_probes_per_entity_fallback():
    # Live: the probe's "The Name" search came first and was the only one shown.
    text = _prompt_text(_packet([("web_grounding", PROBE), ("web_grounding", ROUTER)]))
    assert "Umberto Eco" in text and "Minecraft" not in text


def test_probe_grounding_still_renders_when_alone():
    assert "Minecraft" in _prompt_text(_packet([("web_grounding", PROBE)]))


def test_pre_formatted_tool_result_reaches_the_prompt():
    pre = ("--- Web Search Results (query: who wrote The Name of the Rose) ---\n"
           "  Title: The Name of the Rose\n  The 1980 novel by Italian author Umberto Eco.\n"
           "--- End of Web Search Results ---")
    assert "Umberto Eco" in _prompt_text(_packet([("tool_result", pre)]))
