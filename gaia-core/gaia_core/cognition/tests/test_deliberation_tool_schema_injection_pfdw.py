"""Regression test for GAIA_Project-pfdw — deliberate() never told the model
what tools actually exist, which is a plausible cause of Prime hallucinating
a nonexistent "discord" tool inside a <tool_call> that then leaked straight
to a real Discord channel (session discord_channel_1484003096337191067,
2026-08-15). build_tool_schema_injection() (gaia_common.utils.tool_call_parser)
already builds exactly the right system-prompt text — it just wasn't being
called anywhere. deliberate() now appends it to the system message the same
way the deliberation instructions addendum is appended.
"""

from gaia_core.cognition.deliberation import deliberate


class _CapturePool:
    def __init__(self):
        self.captured_messages = None

    def acquire_model(self, role):
        return object()

    def release_model(self, role):
        pass

    def forward_to_model(self, role, *, messages, **kwargs):
        self.captured_messages = messages
        return {"choices": [{"message": {"content": "<think>fine</think>Sure, here you go."}}]}


def _system_text(messages):
    assert messages and messages[0]["role"] == "system"
    return messages[0]["content"]


def test_deliberate_injects_real_tool_schema():
    pool = _CapturePool()
    base = [
        {"role": "system", "content": "You are GAIA."},
        {"role": "user", "content": "what's the weather"},
    ]
    deliberate(
        user_input="what's the weather",
        assembled_messages=base,
        model_pool=pool,
        persist=False,
    )
    sys_text = _system_text(pool.captured_messages)
    assert "You have these tools available:" in sys_text
    # A real registered domain must appear...
    assert "- web(action):" in sys_text
    # ...and the hallucinated one from the live incident must not be implied
    # as valid (it's simply absent from the real domain list).
    assert "- discord(action):" not in sys_text


def test_deliberate_schema_injection_is_fail_open(monkeypatch):
    """If schema building ever breaks, deliberation must still run — never
    block the turn over a missing tool list."""
    import gaia_common.utils.tool_call_parser as tcp

    def _boom():
        raise RuntimeError("schema build exploded")

    monkeypatch.setattr(tcp, "build_tool_schema_injection", _boom)

    pool = _CapturePool()
    base = [{"role": "system", "content": "You are GAIA."}, {"role": "user", "content": "hi"}]
    result = deliberate(
        user_input="hi",
        assembled_messages=base,
        model_pool=pool,
        persist=False,
    )
    assert result.final_response


def test_deliberate_leaves_tools_out_of_turns_that_dont_offer_them():
    """1ha9: on a turn the prompt builder answers without tools (time, chat)
    the schema made Core reply with only a tool call, which main.py then
    suppressed — an empty reply. Deliberation follows the same decision."""
    from gaia_core.utils.prompt_builder import offers_tools
    pool = _CapturePool()
    base = [{"role": "system", "content": "You are GAIA."},
            {"role": "user", "content": "What is the current date and your uptime?"}]
    deliberate(user_input=base[1]["content"], assembled_messages=base, model_pool=pool,
               persist=False, offer_tools=offers_tools("time"))
    assert "You have these tools available:" not in _system_text(pool.captured_messages)


def test_offers_tools_follows_what_the_runtime_suppresses():
    from gaia_core.utils.prompt_builder import offers_tools
    assert not offers_tools("time") and not offers_tools("Greeting") and not offers_tools("gratitude")
    # "chat" is the router's catch-all ("Fetch the page at …" lands there):
    # leaving tools out of it broke web fetch and knowledge updates.
    assert offers_tools("chat") and offers_tools("tool_routing") and offers_tools("other")
    assert offers_tools("")
