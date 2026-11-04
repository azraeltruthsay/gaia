"""Grounding says how far each thing can be trusted, and acts on it (3yjp)."""
import pytest

from gaia_common.utils.lookup_memory import TRUST_LEVELS, TRUST_WORDS, trust_rank
from gaia_core.cognition.knowledge_router import (
    GroundingContext,
    GroundingResult,
    _local_answers,
    _strong_memory,
    weakest_last,
)

NOTE = "A journal line that mentions the capital of Kazakhstan in passing, at some length."


def r(tier, content=NOTE, source="memory", **kw):
    return GroundingResult(source=source, trust_tier=tier, content=content, **kw)


def strong(tier):
    return r(tier, score=1.0, question_match=1.0, confirmed_match=1.0)


def test_every_level_has_words_and_a_place():
    assert set(TRUST_WORDS) == set(TRUST_LEVELS)
    assert trust_rank("told") < trust_rank("own_record") < trust_rank("looked_up") < trust_rank("corroborated")
    assert trust_rank("unheard of") == -1


@pytest.mark.parametrize("tier", ["told", "told_by_several", "disputed"])
def test_weak_memory_never_stands_in_for_a_fresh_look(tier):
    assert not _strong_memory([strong(tier)])


@pytest.mark.parametrize("tier", ["looked_up", "corroborated", "validated", "told_by_architect"])
def test_a_confirmed_memory_can(tier):
    assert _strong_memory([strong(tier)])


def test_her_notes_do_not_switch_off_the_web_for_questions_about_the_world():
    assert not _local_answers([r("own_record", source="journal")], "What is the capital of Kazakhstan?")


def test_her_notes_do_answer_questions_about_her():
    assert _local_answers([r("own_record", source="journal")], "What did you do yesterday?")


def test_what_she_measured_saves_a_search():
    assert _local_answers([r("observed", source="system_clock")], "What is the time in Paris?")


def test_the_weakest_go_last_and_the_rest_keep_their_order():
    journal, web, told = r("own_record", source="journal"), r("looked_up", source="web"), r("told")
    assert weakest_last([told, journal, web]) == [journal, web, told]


def test_the_prompt_reads_in_words_and_the_best_comes_first():
    ctx = GroundingContext(results=[r("told", "Told thing."), r("corroborated", "Agreed thing.")])
    out = ctx.format_for_prompt()
    assert out.index("Agreed thing") < out.index("Told thing")
    assert "(more than one source agrees)" in out and "VERIFIED" not in out.upper()
    assert ctx.best_result.content == "Agreed thing."


def test_a_lookup_gets_its_level_from_its_evidence():
    from gaia_common.utils.lookup_memory import trust_level
    assert trust_level({"url": "https://a.org/x", "also_at": ["https://b.org/y"]}) == "corroborated"
    assert trust_level({"url": "https://a.org/x", "disputed_at": "2026"}) == "disputed"
    assert trust_level({"kind": "search", "url": ""}) == "looked_up"
    assert trust_level({"kind": "told"}) == "told"
