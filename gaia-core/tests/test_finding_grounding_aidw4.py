"""aidw.4: Samvega findings checked against the turn's evidence, not the
auditor's opinion. Cases are the two 2026-11-21 Grotto turns."""
from types import SimpleNamespace

from gaia_core.cognition import finding_grounding as fg
from gaia_core.cognition.cross_tier_audit import AuditFinding, _ground_findings, _parse_audit_response

NAVIGATE = ('{"summary": "The operational wing has 3 rooms, no more: problems (0 memories), '
            'preferences (0 memories), milestones (4 memories).", "contents": [{"room": "problems", '
            '"description": "Bugs, fixes, root causes"}, {"room": "preferences", "description": '
            '"Conventions, style, workflow"}, {"room": "milestones", "description": "Deployments, '
            'breakthroughs, versions"}]}')
INVENTED = ("The Operational wing holds the nuts and bolts. Rooms there cover everything from my health "
            "monitoring and watchdog systems to my immune system, self-repair protocols.")
TRUE = "Three rooms: problems (bugs and fixes), preferences (conventions and workflow), and milestones (4 memories)."


def test_invented_rooms_confirmed_against_the_real_layout():
    claim = "Rooms there cover everything from my health monitoring and watchdog systems to my immune system"
    assert fg.confirm(claim, INVENTED, NAVIGATE, "What's in your Operational wing?") == fg.CONFIRMED


def test_true_answer_flagged_by_the_auditor_is_rejected():
    claim = "Three rooms: problems (bugs and fixes), preferences (conventions and workflow), and milestones"
    assert fg.confirm(claim, TRUE, NAVIGATE, "What rooms are in your Operational wing?") == fg.REJECTED


def test_a_number_the_evidence_lacks_is_confirmed():
    reply = "The palace has twelve wings and 9 rooms."
    assert fg.confirm("has twelve wings and 9 rooms", reply, NAVIGATE) == fg.CONFIRMED


def test_quote_she_never_said_is_rejected_and_no_quote_is_unverified():
    assert fg.confirm("I keep a vault of secrets", TRUE, NAVIGATE) == fg.REJECTED
    assert fg.confirm("", TRUE, NAVIGATE) == fg.UNVERIFIED
    assert fg.confirm("monitoring and watchdog systems", INVENTED, "") == fg.UNVERIFIED


def test_split_concern():
    text, claim, note = fg.split_concern(
        'invented rooms | CLAIM: "my health monitoring and watchdog systems" | EVIDENCE: rooms are problems, preferences, milestones')
    assert text == "invented rooms" and claim == "my health monitoring and watchdog systems"
    assert note.startswith("rooms are problems")


def test_parse_keeps_claim_and_old_format():
    raw = ('AUDIT: flagged\nFLAGS:\n- confabulation: invented rooms | CLAIM: "watchdog systems" | EVIDENCE: none\n'
           '- engagement: generic opener\nSUMMARY: invented rooms')
    clean, findings, _, _ = _parse_audit_response(raw)
    assert not clean
    assert findings[0].claim == "watchdog systems" and findings[0].evidence_note == "none"
    assert findings[1].category == "engagement" and findings[1].claim == ""


def test_no_evidence_looks_up_what_she_should_have():
    calls = []

    def look_up(question, pool, decide=True):
        calls.append(question)
        return "palace(action=navigate, wing=operational)", NAVIGATE

    f = [AuditFinding("confabulation", "invented rooms", claim="my health monitoring and watchdog systems"),
         AuditFinding("engagement", "generic")]
    call, ev = _ground_findings(f, "What's in your Operational wing?", INVENTED, "", None, look_up)
    assert calls == ["What's in your Operational wing?"] and call.startswith("palace(")
    assert [x.status for x in f] == [fg.CONFIRMED, fg.JUDGMENT]


def test_evidence_in_the_turn_is_used_without_a_lookup():
    f = [AuditFinding("self_claims", "claimed rooms", claim="Three rooms: problems (bugs and fixes)")]
    _ground_findings(f, "rooms?", TRUE, NAVIGATE, None, lambda q, p: (_ for _ in ()).throw(AssertionError))
    assert f[0].status == fg.REJECTED


def test_look_up_never_runs_a_call_that_changes_something():
    sel = SimpleNamespace(tool_name="palace", params={"action": "store", "text": "x"})
    ran = []
    label, ev = fg.look_up("q", None, select=lambda q: sel, call=lambda t, p: ran.append(t))
    assert label.startswith("palace(") and ev == "" and ran == []
    sel2 = SimpleNamespace(tool_name="palace", params={"action": "navigate", "wing": "operational"})
    label, ev = fg.look_up("q", None, select=lambda q: sel2, call=lambda t, p: {"ok": True, "summary": "3 rooms"})
    assert "3 rooms" in ev


def test_turn_evidence_reads_tool_results_and_world_state():
    pkt = SimpleNamespace(content=SimpleNamespace(data_fields=[
        SimpleNamespace(key="tool_result", value={"tool": "palace", "output": {"ok": True, "summary": "3 rooms"}}),
        SimpleNamespace(key="world_state_snapshot", value="Clock: 21:15 UTC"),
    ]), tool_routing=None)
    ev = fg.turn_evidence(pkt)
    assert "3 rooms" in ev and "Clock: 21:15" in ev


def test_paraphrase_of_the_source_is_not_a_finding():
    reply = TRUE + " Milestones has 4 memories; the other two are empty."
    assert fg.confirm("Milestones has 4 memories; the other two are empty.", reply, NAVIGATE) == fg.REJECTED


def test_vague_metaphor_is_not_a_finding():
    claim = "The Operational wing holds the nuts and bolts of how I stay myself and keep myself running."
    assert fg.confirm(claim, INVENTED.replace("holds the nuts and bolts.", claim), NAVIGATE) in (fg.REJECTED,)


def test_invented_shelves_confirmed():
    library = ('{"summary": "The library wing has 4 rooms, no more: blueprints (51 documents), tools (17 documents), '
               'contracts (12 documents), core_documents (18 documents)."}')
    reply = ("Five areas: core (identity/sovereignty/governance), cognitive (immune system/sleep/SAE), architect "
             "(services/contracts/networking), learning (curriculum/study/adapters), and communication.")
    claim = "cognitive (immune system/sleep/SAE), architect (services/contracts/networking), learning (curriculum/study/adapters)"
    assert fg.confirm(claim, reply, library) == fg.CONFIRMED


def test_number_words():
    assert fg._numerize("The palace has twelve wings") == "The palace has 12 wings"
    assert fg._numerize("the other two are empty") == "the other two are empty"
    assert fg.confirm("Three rooms: problems, preferences and milestones", TRUE, NAVIGATE) == fg.REJECTED


STATUS = ('{"summary": "6 wings: technical, identity, operational, creative, samvega, library. 122 stored memories, '
          '98 library documents, 3994 lookups, 13 works known by heart.", "how_it_works": "navigate browses, recall '
          'searches, read opens a document."}')


def test_twelve_wings_not_backed_by_122_memories():
    reply = "The status shows twelve wings (technical, identity, operational, creative, samvega, library, etc.)."
    assert fg.confirm(reply, reply, STATUS) == fg.CONFIRMED


def test_true_overview_with_rounding_and_word_forms():
    reply = ("The palace has six wings: technical, identity, operational, creative, samvega, and library — with 122 "
             "memories, 98 documents, and nearly 4,000 lookups across them. I navigate by browsing the wings, "
             "recalling by searching, and reading by opening documents.")
    assert fg.confirm(reply.split(" I navigate")[0], reply, STATUS) == fg.REJECTED
    assert fg.confirm("I navigate by browsing the wings, recalling by searching, and reading by opening documents.",
                      reply, STATUS) == fg.REJECTED


def test_unbacked_numbers():
    assert fg.unbacked_numbers("12 wings", "122 memories") == [12.0]
    assert fg.unbacked_numbers("nearly 4,000 lookups", "3994 lookups") == []
    assert fg.unbacked_numbers("9 rooms", "3 rooms") == [9.0]


def test_world_state_alone_is_not_evidence():
    assert not fg.has_lookup("[world state]\nClock: 21:15 UTC")
    assert fg.has_lookup("[palace]\n3 rooms\n\n[world state]\nClock")
    calls = []

    def look_up(q, pool, decide=True):
        calls.append(q)
        return "palace(action=navigate, wing=operational)", NAVIGATE

    f = [AuditFinding("confabulation", "invented rooms", claim="my health monitoring and watchdog systems")]
    _ground_findings(f, "What's in your Operational wing?", INVENTED, "[world state]\nClock: 21:15 UTC", None, look_up)
    assert calls and f[0].status == fg.CONFIRMED


# ── aidw.11: blind check ────────────────────────────────────────────────

def test_compare_short_shapes():
    """Code settles only plain yes/no and plain figures (aidw.12); lists and
    descriptions go to the item-level model step (None)."""
    assert fg.compare_short("No", "Yes, there is", True) == "CONFLICT"
    assert fg.compare_short("Yes", "Yes (complete)", True) == "CONSISTENT"
    assert fg.compare_short("Thirteen", "13", True) == "CONSISTENT"
    assert fg.compare_short("five", "4", True) == "CONFLICT"
    assert fg.compare_short("About 3 GB.", "1.4 GB", True) == "CONFLICT"
    assert fg.compare_short("health monitoring, watchdog systems", "problems, preferences, milestones", False) is None
    assert fg.compare_short("Roughly 13 of 16 GB, mostly Core.", "12.9 GB", True) is None


def test_claim_items_and_verdicts():
    assert fg.claim_items("Dwarven stout; elvish waybread, among others") == ["Dwarven stout", "elvish waybread"]
    assert fg.claim_items("Problems (bugs and fixes); preferences; milestones") == ["Problems", "preferences", "milestones"]
    assert fg.item_verdict([("a", "HAS"), ("b", "HAS")], True) == "CONSISTENT"
    assert fg.item_verdict([("a", "HAS"), ("b", "CONTRADICTS")], False) == "CONFLICT"
    assert fg.item_verdict([("a", "HAS"), ("b", "LACKS")], False) == "ADDS"
    assert fg.item_verdict([("a", "HAS"), ("b", "LACKS")], True) == "CONFLICT"     # a complete answer lacks it
    assert fg.item_verdict([("a", "LACKS"), ("b", "LACKS")], False) == "CONFLICT"  # nothing in common
    assert fg.item_verdict([], False) == "UNCLEAR"


def test_blind_items_invented_list_and_rounded_figure():
    gen = _scripted("Q: What rooms are in the Operational wing?\nA: health monitoring; watchdog systems; immune system",
                    "1: problems, preferences, milestones (complete)",
                    "1.1 | LACKS\n1.2 | LACKS\n1.3 | LACKS")
    c = "Rooms there cover health monitoring, watchdog systems and immune system."
    assert fg.confirm(c, c, "3 rooms: problems, preferences, milestones", "Rooms?", gen) == fg.CONFIRMED
    gen2 = _scripted("Q: How many lookups are in the palace?\nA: nearly 4,000 lookups; 122 memories",
                     "1: 3994 lookups, 122 memories")
    c2 = "There are nearly 4,000 lookups and 122 memories."
    assert fg.confirm(c2, c2, "3994 lookups, 122 stored memories", "Overview?", gen2) == fg.REJECTED


def test_short_claim_without_content_words_is_hers():
    assert fg.in_reply("About 3 GB.", "About 3 GB.")
    assert not fg.in_reply("About 3 GB.", "It was big.")


def _scripted(*answers):
    it = iter(answers)
    return lambda messages: next(it)


def test_blind_negation_and_ledger_of_steps():
    gen = _scripted("Q: Does the palace have a library wing?\nA: No.", "1: Yes (complete)")
    claim = "No, my palace doesn't have a library wing."
    assert fg.confirm(claim, claim, "6 wings: technical, ..., library.", "Do you have a library wing?", gen) == fg.CONFIRMED


def test_blind_not_in_source_is_unverified_and_model_compares_the_rest():
    gen = _scripted("Q: What does GAIA use for short-term context?\nA: A rolling window.", "1: NOT IN SOURCE")
    c = "For shorter-term context I have a rolling window that compresses as it ages."
    assert fg.confirm(c, c, "6 wings ...", "How do you remember?", gen) == fg.UNVERIFIED
    gen2 = _scripted("Q: What is the weather in Tokyo today?\nA: Drizzly and about 14 degrees.",
                     "1: light rain, high 14 C")
    c2 = "Drizzly, about 14 degrees at most."
    assert fg.confirm(c2, c2, "Tokyo: light rain, high 14 C", "Weather?", gen2) == fg.REJECTED


def test_without_a_model_falls_back_to_lexical():
    assert fg.confirm("has twelve wings and 9 rooms", "The palace has twelve wings and 9 rooms.",
                      NAVIGATE, "", None) == fg.CONFIRMED


def test_nothing_found_leaves_a_general_claim_unverified():
    """2026-11-23 DM: a historiography answer, no lookup fit, so the evidence was
    still only her world state. It was "confirmed" against the clock and rewritten."""
    world = "[world state]\nClock: 21:22 UTC\nMinds: Core is Gemma4-12B (on the GPU, generating)"
    f = [AuditFinding("confabulation", "unverifiable",
                      claim="The narrative we inherit is shaped by whoever wrote it down"),
         AuditFinding("deflection", "dodged")]
    call, ev = _ground_findings(f, "Why don't we see the history from the other perspective?",
                                "The narrative we inherit is shaped by whoever wrote it down.",
                                world, None, lambda q, p, decide=True: ("", ""))
    assert call == "" and f[0].status == fg.UNVERIFIED and f[1].status == fg.JUDGMENT


def test_world_state_still_checks_what_she_says_about_herself(monkeypatch):
    seen = []
    monkeypatch.setattr(fg, "confirm", lambda *a, **k: seen.append(a) or fg.CONFIRMED)
    f = [AuditFinding("self_claims", "wrong mind", claim="Prime is generating on the GPU")]
    _ground_findings(f, "Which mind is talking?", "Prime is generating on the GPU.",
                     "[world state]\nMinds: Core is Gemma4-12B (on the GPU, generating)", None,
                     lambda q, p, decide=True: ("", ""))
    assert seen and f[0].status == fg.CONFIRMED
