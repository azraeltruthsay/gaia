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
