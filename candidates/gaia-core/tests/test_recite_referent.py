"""'Recite that poem' names the work the chat was just about (2026-11-11)."""
from gaia_core.cognition.recite_referent import is_anaphoric_recite, resolve
from gaia_core.cognition.nlu.intent_detection import _is_recite_request

TYGER_TALK = [
    {"role": "user", "content": 'There\'s a poem that starts with "Tyger, Tyger burning bright." Do you know what poem I\'m referring to?'},
    {"role": "assistant", "content": 'Yes — that\'s the opening line of "The Tyger" by William Blake. It\'s a famous example of Romantic poetry.'},
]


def test_that_poem_resolves_to_the_tyger():
    q = "Can you recite that poem for me?"
    out = resolve(q, TYGER_TALK + [{"role": "user", "content": q}])
    assert out == "Recite The Tyger by William Blake"
    assert _is_recite_request(out)


def test_bold_title_and_bare_by():
    hist = [{"role": "assistant", "content": "Here is **Ozymandias by Percy Bysshe Shelley**:\n\nI met a traveller"}]
    assert resolve("recite it again", hist) == "Recite Ozymandias by Percy Bysshe Shelley"
    hist = [{"role": "assistant", "content": "You mean The Raven by Edgar Allan Poe, I think."}]
    assert resolve("Could you recite that one?", hist) == "Recite The Raven by Edgar Allan Poe"


def test_quoted_first_line_is_not_a_title():
    hist = [{"role": "user", "content": 'A poem starts "Tyger, Tyger burning bright"'}]
    assert resolve("recite that poem", hist) is None


def test_named_requests_are_left_alone():
    assert not is_anaphoric_recite("Can you recite Jabberwocky for me?")
    assert not is_anaphoric_recite("Recite The Tyger")
    assert resolve("Can you recite Jabberwocky?", TYGER_TALK) is None


def test_most_recent_work_wins():
    hist = [{"role": "assistant", "content": 'That was "Ozymandias" by Percy Bysshe Shelley.'}] + TYGER_TALK
    assert resolve("recite that poem", hist) == "Recite The Tyger by William Blake"
