"""
Characterization tests for the weighted intent classifier — GAIA_Project-21j.

Pins the CURRENT score→routing behaviour (incl. the LITE/Nano + TRIAGE zones,
where _nano_triage is consulted) before Nano retirement, so the Core-direct
rewrite is verifiable against captured behaviour. See test_router_characterization
for the routing-engine side.

The "LITE = Nano/Lite zone" + "TRIAGE = defer to _nano_triage" assertions are
the nano surface here: on 21j, LITE stays on Core and TRIAGE arbitration becomes
a Core (or pure-heuristic) decision, not a Nano LLM call.
"""

from gaia_core.cognition.nlu.intent_detection import WeightedIntentClassifier, Plan


WIC = WeightedIntentClassifier


def _routing_for(score: float) -> str:
    """Mirror the documented threshold map (the behaviour to preserve)."""
    if score > WIC.THRESHOLD_PRIME:
        return "PRIME"
    if score < WIC.THRESHOLD_LITE:
        return "LITE"
    return "TRIAGE"


def test_thresholds_pinned():
    assert WIC.THRESHOLD_PRIME == 0.8
    assert WIC.THRESHOLD_LITE == 0.3


def test_plan_defaults_to_triage_routing():
    # Plan.routing default is TRIAGE — the ambiguous zone that consults nano.
    assert Plan(intent="other").routing == "TRIAGE"


def test_score_is_in_unit_interval():
    for text, intent in [("hi", "greeting"), ("implement a distributed scheduler", "planning"),
                         ("read the file", "read_file"), ("", "other")]:
        s = WIC.score(text, embed_intent=intent, embed_confidence=0.8)
        assert 0.0 <= s <= 1.0


def test_greeting_scores_low_lite_zone_TODAY():
    # NANO→CORE on retire: trivial greetings score in the LITE (Nano/Lite) zone.
    s = WIC.score("hello there", embed_intent="greeting", embed_confidence=0.9)
    assert s < WIC.THRESHOLD_LITE
    assert _routing_for(s) == "LITE"


def test_planning_scores_above_lite_zone():
    # CHARACTERIZED REALITY: the weighted formula caps intent's contribution
    # (W_INTENT=0.40), so even a "planning" request lands in TRIAGE, NOT PRIME —
    # PRIME is reached via escalation markers / _resolve_prime_target, not raw
    # score. The invariant that matters: a complex request never falls to LITE.
    s = WIC.score("design and implement a multi-step migration pipeline architecture",
                  embed_intent="planning", embed_confidence=0.9)
    assert s >= WIC.THRESHOLD_LITE
    assert _routing_for(s) in ("TRIAGE", "PRIME")


def test_score_is_deterministic():
    a = WIC.score("what time is it", embed_intent="chat", embed_confidence=0.5)
    b = WIC.score("what time is it", embed_intent="chat", embed_confidence=0.5)
    assert a == b


def test_complexity_weights_ordering_pinned():
    # Relative ordering the router depends on: planning > chat/greeting.
    from gaia_core.cognition.nlu.intent_detection import _INTENT_COMPLEXITY
    assert _INTENT_COMPLEXITY["planning"] > _INTENT_COMPLEXITY["chat"]
    assert _INTENT_COMPLEXITY["greeting"] < _INTENT_COMPLEXITY["shell"]
