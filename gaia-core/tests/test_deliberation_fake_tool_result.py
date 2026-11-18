"""b9mp: a <tool_result> in the model's own deliberation answer is made up
(only the system writes one). It is dropped, and an answer that was nothing
else falls back like an empty one."""
from gaia_core.cognition.deliberation import deliberate


class _Pool:
    def __init__(self, content, reasoning="Let me work it out. 2x = 0.10, x = 0.05."):
        self.content, self.reasoning = content, reasoning

    def acquire_model(self, role):
        return object()

    def release_model(self, role):
        pass

    def forward_to_model(self, role, **kw):
        return {"choices": [{"message": {"content": self.content,
                                         "reasoning_content": self.reasoning}}]}


def _run(content):
    msgs = [{"role": "system", "content": "You are GAIA."},
            {"role": "user", "content": "How much is the ball?"}]
    return deliberate("How much is the ball?", msgs, _Pool(content), model_role="prime",
                      persist=False, offer_tools=False, native_thinking=True)


def test_fake_result_only_falls_back():
    r = _run("<tool_result>...</tool_result>")
    assert r.final_response == ""
    assert r.fallback_used


def test_fake_result_dropped_from_real_answer():
    r = _run("<tool_result>price: 0.05</tool_result>\nThe ball costs $0.05.")
    assert "tool_result" not in r.final_response
    assert "$0.05" in r.final_response
    assert not r.fallback_used


def test_unclosed_fake_result_dropped():
    r = _run("<tool_result>{\"answer\": 5}")
    assert r.final_response == ""
    assert r.fallback_used


def test_plain_answer_untouched():
    r = _run("The ball costs $0.05.")
    assert r.final_response == "The ball costs $0.05."
    assert not r.fallback_used
