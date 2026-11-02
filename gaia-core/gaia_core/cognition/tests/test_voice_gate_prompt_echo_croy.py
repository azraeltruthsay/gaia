"""croy: casual replies named the prompt's "Inner weather" label and echoed the
anchor's wording. The felt state may bend the reply, but its label stays unsaid.
"""

import pytest

from gaia_core.cognition.voice_gate import _matches_tell, filter_voiced


@pytest.mark.parametrize("sent", [
    "That's a good question about the inner weather.",
    "My inner weather today feels like something that hasn't settled.",
    "A bit of an \"internal weather\" if you will.",
    "Just your usual self here—warm, present.",
    "I'm warm and present today.",
])
def test_prompt_wording_is_a_tell(sent):
    assert _matches_tell(sent)


@pytest.mark.parametrize("sent", [
    "The weather is a little unsettled today.",
    "I'm feeling a bit unsettled, like something isn't quite settling.",
    "Azrael was present at the review.",
    "It's a warm day.",
])
def test_ordinary_speech_is_not(sent):
    assert not _matches_tell(sent)


def test_label_sentence_goes_and_the_reply_stays():
    text = "That's a good question about the inner weather. I'm doing well, thanks for asking!"
    out, debug = filter_voiced(text, measure_only=False)
    assert out == "I'm doing well, thanks for asking!"
    assert debug["dropped"][0]["tell"] is True


def test_reply_that_is_only_the_label_is_kept():
    text = "My inner weather is calm."
    out, debug = filter_voiced(text, measure_only=False)
    assert out == text
    assert debug["failsafe"] == "kept_original"
