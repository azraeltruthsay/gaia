"""8xr4.16: shape-only uncertainty signals (4: no numeric value, 7: very
short) used to escalate correct terse answers — "Excalibur." to "Can you tell
me the name of King Arthur's sword?" went to Prime, which confabulated. They
now only mark the reply as suspect; the model must quote the words that
answer, and the quote must occur in the reply."""

from unittest.mock import MagicMock

from gaia_core.cognition.agent_core import AgentCore

SWORD_Q = "Can you tell me the name of King Arthur's sword?"


def _core_with_judge(judge_output=None, raises=False):
    ai_manager = MagicMock()
    ai_manager.config.constants = {}
    ai_manager.config.SHARED_DIR = "/tmp/test_shared"
    ai_manager.config.config.SHARED_DIR = "/tmp/test_shared"
    ac = AgentCore(ai_manager)
    judge = MagicMock()
    if raises:
        judge.create_chat_completion.side_effect = RuntimeError("engine down")
    else:
        judge.create_chat_completion.return_value = {
            "choices": [{"message": {"content": judge_output}}]}
    ac.model_pool._resolve_model_name_for_role = lambda role: role
    ac.model_pool.models = {"nano": judge}
    return ac, judge


def test_correct_terse_answer_is_kept():
    ac, _ = _core_with_judge("Excalibur.")
    assert ac._should_escalate_for_uncertainty("Excalibur.", SWORD_Q, model_name="nano") is False


def test_terse_number_answer_is_kept():
    ac, _ = _core_with_judge("4.\nNo words in the reply answer it.")  # first line only
    assert ac._should_escalate_for_uncertainty("4.", "What is 2 plus 2?", model_name="nano") is False


def test_judge_answering_itself_means_reply_did_not_answer():
    # The reply has no answer; the model supplies its own, which isn't in the reply.
    ac, _ = _core_with_judge("Paris.")
    assert ac._should_escalate_for_uncertainty("Sure!", "What's the capital of France?",
                                               model_name="nano") is True


def test_none_verdict_escalates():
    ac, _ = _core_with_judge("NONE")
    assert ac._should_escalate_for_uncertainty(
        "I'm ready to share the time with you.", "What time is it? Tell me the time",
        model_name="nano") is True


def test_judge_unavailable_falls_back_to_escalating():
    ac, _ = _core_with_judge(raises=True)
    assert ac._should_escalate_for_uncertainty("Excalibur.", SWORD_Q, model_name="nano") is True


def test_no_model_name_keeps_old_behaviour():
    ac, judge = _core_with_judge("Excalibur.")
    assert ac._should_escalate_for_uncertainty("Excalibur.", SWORD_Q) is True
    judge.create_chat_completion.assert_not_called()


def test_classification_tag_escalates_without_judging():
    ac, judge = _core_with_judge("COMPLEX")
    assert ac._should_escalate_for_uncertainty("COMPLEX", "hi", model_name="nano") is True
    judge.create_chat_completion.assert_not_called()


def test_hedging_still_escalates_without_judging():
    ac, judge = _core_with_judge("whatever")
    reply = "I don't have access to that information right now, sorry about it."
    assert ac._should_escalate_for_uncertainty(reply, "What is the GPU temperature?",
                                               model_name="nano") is True
    judge.create_chat_completion.assert_not_called()


def test_long_substantive_reply_is_not_judged():
    ac, judge = _core_with_judge("x")
    reply = "Excalibur is the legendary sword of King Arthur in Arthurian legend."
    assert ac._should_escalate_for_uncertainty(reply, "Who was King Arthur?", model_name="nano") is False
    judge.create_chat_completion.assert_not_called()
