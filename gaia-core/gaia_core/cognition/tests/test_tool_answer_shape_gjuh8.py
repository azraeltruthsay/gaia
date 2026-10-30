"""gjuh.8: no '---' rule between a tool's result and the request. Core read it
as a section break and wrote past it — echoing '---' in 10/20 replies vs 0/20
without, same answers (live engine A/B, 2026-10-29)."""

import re

from gaia_common.utils.tool_call_parser import tool_answer_messages


def test_tool_answer_has_no_section_rule():
    msgs = tool_answer_messages("What did we decide to name the cat?",
                                "Searching your record of earlier conversations with them",
                                "- they said: \"Pixel it is!\"")
    user = msgs[1]["content"]
    assert not re.search(r"^\s*---\s*$", user, re.M)
    assert user.startswith("Searching your record of earlier conversations with them returned the following.")
    assert user.endswith("Your reply to them:")


def test_is_tool_answer_recognises_only_the_tool_answer_shape():
    from gaia_common.utils.tool_call_parser import is_tool_answer
    assert is_tool_answer(tool_answer_messages("q", "file(read)", "text"))
    assert not is_tool_answer([{"role": "system", "content": "You are GAIA."},
                               {"role": "user", "content": "hi"}])
    assert not is_tool_answer([])


def test_a_tool_answered_turn_deliberates_without_the_adapter(monkeypatch):
    # The adapter recited the routed file's opening 6/6 (bare Core 0/6).
    from gaia_core.cognition import deliberation
    seen = {}

    def fake_deliberate(*a, **kw):
        seen.update(kw)
        return deliberation.DeliberationResult(
            thinking="", final_response="ok", raw_output="ok", voice_evidence={},
            forbidden_hits=[], elapsed_ms=0.0, fallback_used=False, model_used="core")
    monkeypatch.setattr(deliberation, "deliberate", fake_deliberate)
    cfg = {"DELIBERATION": {"adapter_name": "core_deliberation_v1", "retry_on_confabulation": "none"}}
    deliberation.run_deliberated_turn("q", tool_answer_messages("q", "file(read)", "text"), None, cfg)
    assert seen["adapter_name"] is None and seen["native_thinking"] is False and seen["casual"] is True
    seen.clear()
    deliberation.run_deliberated_turn("q", [{"role": "system", "content": "You are GAIA."},
                                            {"role": "user", "content": "q"}], None, cfg)
    assert seen["adapter_name"] == "core_deliberation_v1"
