"""0vuz: lookup memory keeps what GAIA looked up, verbatim and with
provenance, recalls it for later questions, and tracks when to look again."""

import json
from datetime import datetime, timedelta, timezone

from gaia_common.utils.lookup_memory import (
    ARCHIVE_UNUSED_DAYS, PIN_AFTER_CONFIRMATIONS, STABLE_DAYS, VOLATILE_DAYS, LookupMemory,
    assess_answer, content_terms, is_volatile, question_match)

T0 = datetime(2026, 10, 27, 12, 0, tzinfo=timezone.utc)

LADY = {
    "title": "Lady of the Lake - Wikipedia",
    "url": "https://en.wikipedia.org/wiki/Lady_of_the_Lake",
    "snippet": "The Lady of the Lake gives King Arthur the sword Excalibur and later "
               "receives it back when Bedivere throws it into the lake.",
    "domain": "en.wikipedia.org",
    "trust_tier": "trusted",
}
ONIMUSHA = {
    "title": "Onimusha: Way of the Sword",
    "url": "https://example.com/onimusha",
    "snippet": "Onimusha: Way of the Sword is an action game set in Edo-period Kyoto with demons.",
    "domain": "example.com",
}


def _mem(tmp_path):
    return LookupMemory(tmp_path / "lookups")


def test_remembers_verbatim_with_provenance(tmp_path):
    m = _mem(tmp_path)
    ids = m.remember("Where did King Arthur get Excalibur?", [LADY], now=T0)
    rec = json.loads((tmp_path / "lookups" / f"{ids[0]}.json").read_text())
    assert rec["snippet"] == LADY["snippet"]
    assert rec["url"] == LADY["url"] and rec["domain"] == "en.wikipedia.org"
    assert rec["trust_tier"] == "trusted" and rec["kind"] == "search"
    assert rec["queries"] == ["Where did King Arthur get Excalibur?"]
    assert rec["retrieved_at"].startswith("2026-10-27")
    assert rec["revalidate_after"].startswith(str((T0 + timedelta(days=STABLE_DAYS)).date()))


def test_skips_results_without_a_usable_snippet(tmp_path):
    m = _mem(tmp_path)
    assert m.remember("q", [{"title": "x", "url": "u", "snippet": "too short"}, "junk", None]) == []


def test_same_result_again_refreshes_instead_of_duplicating(tmp_path):
    m = _mem(tmp_path)
    first = m.remember("Where did Arthur get the sword?", [LADY], now=T0)
    later = T0 + timedelta(days=30)
    again = m.remember("Who gave Arthur Excalibur?", [LADY], now=later)
    assert first == again
    assert len(list((tmp_path / "lookups").glob("*.json"))) == 1
    rec = json.loads((tmp_path / "lookups" / f"{first[0]}.json").read_text())
    assert rec["queries"] == ["Where did Arthur get the sword?", "Who gave Arthur Excalibur?"]
    assert rec["retrieved_at"].startswith("2026-11-26")
    assert rec["first_retrieved_at"].startswith("2026-10-27")


def test_recall_finds_the_relevant_lookup_first(tmp_path):
    m = _mem(tmp_path)
    m.remember("King Arthur sword", [LADY, ONIMUSHA], now=T0)
    hits = m.recall("Where did he get the sword? King Arthur Excalibur", now=T0)
    assert hits[0]["url"] == LADY["url"]
    assert hits[0]["score"] >= 0.6 and hits[0]["stale"] is False
    assert all(h["url"] != ONIMUSHA["url"] for h in hits)


def test_recall_ignores_unrelated_questions(tmp_path):
    m = _mem(tmp_path)
    m.remember("King Arthur sword", [LADY], now=T0)
    assert m.recall("What is the capital of France?", now=T0) == []
    assert m.recall("how are you?", now=T0) == []  # stopwords only


def test_recall_counts_usage(tmp_path):
    m = _mem(tmp_path)
    rid = m.remember("King Arthur sword", [LADY], now=T0)[0]
    m.recall("King Arthur Excalibur", now=T0)
    m.recall("King Arthur Excalibur", now=T0)
    rec = json.loads((tmp_path / "lookups" / f"{rid}.json").read_text())
    assert rec["times_recalled"] == 2


def test_volatile_lookups_go_stale_quickly(tmp_path):
    m = _mem(tmp_path)
    news = dict(ONIMUSHA, snippet="The latest Onimusha release date was announced today for next spring.")
    m.remember("latest Onimusha release date", [news], now=T0)
    assert m.recall("Onimusha release date", now=T0 + timedelta(days=1))[0]["stale"] is False
    assert m.recall("Onimusha release date", now=T0 + timedelta(days=VOLATILE_DAYS + 1))[0]["stale"] is True
    assert is_volatile("What's the weather today?") and not is_volatile("Who forged Excalibur?")


def test_dispute_marks_stale_until_the_web_confirms_again(tmp_path):
    m = _mem(tmp_path)
    rid = m.remember("King Arthur sword", [LADY], now=T0)[0]
    assert m.mark_disputed([rid, "missing-id"], now=T0) == 1
    assert m.recall("King Arthur Excalibur", now=T0)[0]["stale"] is True
    assert m.stats() == {"lookups": 1, "disputed": 1, "pinned": 0, "confirmed": 0, "archived": 0}
    m.remember("Who gave Arthur Excalibur?", [LADY], now=T0)  # web returned it again
    assert m.recall("King Arthur Excalibur", now=T0)[0]["stale"] is False


def test_unreadable_record_is_skipped_not_fatal(tmp_path):
    m = _mem(tmp_path)
    m.remember("King Arthur sword", [LADY], now=T0)
    (tmp_path / "lookups" / "broken.json").write_text("{not json")
    assert m.recall("King Arthur Excalibur", now=T0)[0]["url"] == LADY["url"]


def test_content_terms_drop_stopwords():
    assert content_terms("Where did he get the sword?") == {"sword"}


def test_question_match_tells_same_subject_questions_apart():
    # Live 0vuz case: results remembered for the sword's name must not
    # answer "where did he get it" without a fresh lookup.
    followup = "Where did he get the sword? name King Arthur's Excalibur"
    assert question_match(followup, ["King Arthur's", "['King Arthur', 'sword']"]) < 0.6
    # ...but the same question asked again in a new session does match.
    assert question_match("Where did King Arthur get Excalibur?", [followup]) >= 0.6


def test_recall_reports_question_match(tmp_path):
    m = _mem(tmp_path)
    m.remember("Where did King Arthur get Excalibur?", [LADY], now=T0)
    hit = m.recall("Where did Arthur get Excalibur?", now=T0)[0]
    assert hit["question_match"] >= 0.6


def test_record_looked_up_for_this_question_outranks_generic_ones(tmp_path):
    # Live 0vuz case: generic "Excalibur is the sword of King Arthur" records
    # tied at relevance 1.0 and pushed the Lady of the Lake out of the top 3.
    m = _mem(tmp_path)
    for i in range(3):
        m.remember("Excalibur", [{"title": f"Excalibur {i}", "url": f"https://x/{i}",
                                  "snippet": f"Excalibur is the legendary sword of King Arthur, version {i}."}], now=T0)
    m.remember("Where did he get the sword? name King Arthur's Excalibur", [LADY], now=T0)
    hits = m.recall("Where did King Arthur get Excalibur?", now=T0)
    assert hits[0]["url"] == LADY["url"] and hits[0]["question_match"] >= 0.6


# ── 6afh: did the memory actually answer? ────────────────────────────

# The live 6afh records: about Excalibur, silent on where Arthur got it.
LEAD = {"title": "Excalibur - Wikipedia", "url": "https://en.wikipedia.org/wiki/Excalibur",
        "snippet": "Excalibur is the mythical sword of King Arthur that was purported to possess "
                   "magical powers. Its first reliably datable appearance is in Geoffrey of Monmouth."}
Q = "Where did King Arthur get Excalibur?"


def test_assess_answer_verdicts():
    src = [LEAD["snippet"], LADY["snippet"]]
    assert assess_answer(Q, "It depends on which version of the legend you read.", src)["verdict"] == "hedged"
    assert assess_answer(Q, "The sources don't say where.", src)["verdict"] == "hedged"
    invented = assess_answer(Q, "Merlin gave it to him after Lancelot tested him at Camelot.", src)
    assert invented["verdict"] == "unsupported"
    assert invented["unsupported_names"] == ["camelot", "lancelot", "merlin"]
    assert assess_answer(Q, "The Lady of the Lake gave it to him.", src)["verdict"] == "supported"
    assert assess_answer(Q, "", src)["verdict"] == "hedged"


def test_invented_answer_marks_memory_unhelpful_for_that_question(tmp_path):
    m = _mem(tmp_path)
    rid = m.remember(Q, [LEAD], now=T0)[0]
    out = m.feedback(Q, "Merlin gave it to him at Camelot.", [{"memory_id": rid, "snippet": LEAD["snippet"]}], now=T0)
    assert out["verdict"] == "unsupported" and out["records"] == 1
    hit = m.recall(Q, now=T0)[0]
    assert hit["unhelpful_match"] >= 0.6 and hit["confirmed_match"] == 0.0
    # A different question about the same sword isn't affected.
    assert m.recall("What powers did Excalibur possess?", now=T0)[0]["unhelpful_match"] < 0.6


def test_supported_answer_confirms_and_clears_unhelpful(tmp_path):
    m = _mem(tmp_path)
    rid = m.remember(Q, [LADY], now=T0)[0]
    src = [{"memory_id": rid, "snippet": LADY["snippet"]}]
    m.feedback(Q, "I'm not sure.", src, now=T0)
    m.feedback(Q, "The Lady of the Lake gave it to him.", src, now=T0)
    hit = m.recall("Where did Arthur get Excalibur?", now=T0)[0]
    assert hit["confirmed_match"] >= 0.6 and hit["unhelpful_match"] < 0.6


def test_feedback_finds_records_by_url_when_ids_were_lost(tmp_path):
    # Grounding renders web results as "title: snippet"; the url still finds them.
    m = _mem(tmp_path)
    m.remember(Q, [LADY], now=T0)
    out = m.feedback(Q, "The Lady of the Lake.", [{"url": LADY["url"], "snippet": f"web: {LADY['snippet']}"}], now=T0)
    assert out["records"] == 1


def test_dispute_drops_confirmations(tmp_path):
    m = _mem(tmp_path)
    rid = m.remember(Q, [LADY], now=T0)[0]
    m.feedback(Q, "The Lady of the Lake.", [{"memory_id": rid}], now=T0)
    m.mark_disputed([rid], now=T0)
    assert m.recall(Q, now=T0)[0]["confirmed_match"] == 0.0


def test_confirmed_often_is_pinned(tmp_path):
    m = _mem(tmp_path)
    rid = m.remember(Q, [LADY], now=T0)[0]
    for i in range(PIN_AFTER_CONFIRMATIONS):
        m.feedback(f"{Q} ({i})", "The Lady of the Lake.", [{"memory_id": rid}], now=T0)
    rec = json.loads((tmp_path / "lookups" / f"{rid}.json").read_text())
    assert rec["pinned"] is True and rec["times_confirmed"] == PIN_AFTER_CONFIRMATIONS


# ── lifecycle: pinned forever, the rest archived when unused ─────────

def test_sweep_archives_unused_keeps_pinned_and_used(tmp_path):
    m = _mem(tmp_path)
    unused = m.remember("sword", [ONIMUSHA], now=T0)[0]
    pinned = m.remember("lake", [dict(ONIMUSHA, url="https://p", snippet="Pinned forever: " + ONIMUSHA["snippet"])], now=T0)[0]
    used = m.remember(Q, [LADY], now=T0)[0]
    m.pin([pinned])
    later = T0 + timedelta(days=ARCHIVE_UNUSED_DAYS + 5)
    m.recall("King Arthur Excalibur", now=later)  # recalled → in use
    assert m.sweep(now=later, dry_run=True)["archived"][0]["id"] == unused
    assert (tmp_path / "lookups" / f"{unused}.json").exists()  # dry run moved nothing
    out = m.sweep(now=later)
    assert [a["id"] for a in out["archived"]] == [unused] and out["pinned"] == 1
    assert (tmp_path / "lookups" / "archive" / f"{unused}.json").exists()
    assert not (tmp_path / "lookups" / f"{unused}.json").exists()
    assert {p.stem for p in (tmp_path / "lookups").glob("*.json")} == {pinned, used}
    assert m.stats()["archived"] == 1


def test_archive_is_searchable_only_on_request(tmp_path):
    m = _mem(tmp_path)
    rid = m.remember("Onimusha", [ONIMUSHA], now=T0)[0]
    m.sweep(now=T0 + timedelta(days=ARCHIVE_UNUSED_DAYS + 1))
    assert m.recall("Onimusha Way of the Sword", now=T0) == []
    hits = m.recall("Onimusha Way of the Sword", now=T0, include_archive=True)
    assert hits[0]["id"] == rid and hits[0]["archived"] is True
    # Recalling from the archive doesn't move it back by itself…
    assert (tmp_path / "lookups" / "archive" / f"{rid}.json").exists()


def test_web_returning_an_archived_record_restores_it(tmp_path):
    m = _mem(tmp_path)
    rid = m.remember("Onimusha", [ONIMUSHA], now=T0)[0]
    later = T0 + timedelta(days=ARCHIVE_UNUSED_DAYS + 1)
    m.sweep(now=later)
    assert m.remember("Onimusha", [dict(ONIMUSHA)], now=later) == [rid]
    rec = json.loads((tmp_path / "lookups" / f"{rid}.json").read_text())
    assert "archived_at" not in rec and rec["first_retrieved_at"].startswith("2026-10-27")
    assert not (tmp_path / "lookups" / "archive" / f"{rid}.json").exists()


def test_pin_restores_and_restore_works(tmp_path):
    m = _mem(tmp_path)
    a = m.remember("Onimusha", [ONIMUSHA], now=T0)[0]
    m.sweep(now=T0 + timedelta(days=ARCHIVE_UNUSED_DAYS + 1))
    assert m.pin([a]) == 1
    assert (tmp_path / "lookups" / f"{a}.json").exists()
    assert m.sweep(now=T0 + timedelta(days=5000))["archived"] == []  # pinned: forever


def test_long_stale_records_are_archived(tmp_path):
    m = _mem(tmp_path)
    news = dict(ONIMUSHA, snippet="The latest Onimusha release date was announced today for next spring.")
    m.remember("latest Onimusha release date", [news], now=T0)
    later = T0 + timedelta(days=VOLATILE_DAYS + 31)
    m.recall("Onimusha release date", now=later - timedelta(days=1))  # used, but stale
    assert m.sweep(now=later)["archived"][0]["reason"].startswith("stale")


def test_sweep_caps_active_records(tmp_path):
    m = _mem(tmp_path)
    for i in range(4):
        m.remember(f"q{i}", [dict(ONIMUSHA, url=f"https://x/{i}", snippet=f"{ONIMUSHA['snippet']} {i}")],
                   now=T0 + timedelta(hours=i))
    out = m.sweep(now=T0, max_active=2)
    assert len(out["archived"]) == 2 and out["active"] == 2
    assert all(a["reason"] == "over capacity" for a in out["archived"])


def test_only_records_that_carry_the_answer_are_confirmed(tmp_path):
    # Live 6afh: the Excalibur page lead rode along with the Quizlet card that
    # actually says "the Lake"; only the card earned the confirmation.
    m = _mem(tmp_path)
    card = {"title": "King Arthur Flashcards | Quizlet", "url": "https://quizlet.com/x",
            "snippet": "Where did King Arthur get Excalibur? From the hand of the woman in the Lake of Enchantment."}
    lead_id, card_id = m.remember(Q, [LEAD, card], now=T0)
    out = m.feedback(Q, "From the hand of the woman in the Lake.",
                     [{"memory_id": lead_id}, {"memory_id": card_id}], now=T0)
    assert out["records"] == 1
    hits = {h["id"]: h for h in m.recall(Q, now=T0)}
    assert hits[card_id]["confirmed_match"] >= 0.6
    assert hits[lead_id]["confirmed_match"] == 0.0 and hits[lead_id]["unhelpful_match"] == 0.0
