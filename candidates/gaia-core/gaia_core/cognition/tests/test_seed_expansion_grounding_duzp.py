"""duzp: a thought seed's expansion runs as a real turn. 2026-10-14 a seed
about stale docs expanded into invented tooling the turns then tried to run.
An expansion naming a path or identifier the seed didn't give is dropped."""
from unittest.mock import MagicMock

from gaia_core.cognition.reflection_grounding import ungrounded_paths

SEED = {"seed": "The architecture docs in /shared/docs_drafts look stale; AS_BUILT_LATEST.md is from June.",
        "context": {"source": "docs_maintenance"}}

INVENTED = ("Run cd /gaia/maintenance && python3 /gaia/scripts/docs_maintenance.py, then "
            "cd /shared/docs_drafts && ./run_incidents_list.sh and python3 -m gaia_maintenance.service.")

GROUNDED = ("Read /shared/docs_drafts/AS_BUILT_LATEST.md and compare it with what you know of "
            "the current architecture. List the sections that look out of date.")


def _hb(reply):
    from gaia_core.cognition.heartbeat import ThoughtSeedHeartbeat
    hb = ThoughtSeedHeartbeat.__new__(ThoughtSeedHeartbeat)
    llm = MagicMock()
    llm.create_chat_completion.return_value = {"choices": [{"message": {"content": reply}}]}
    return hb, llm


def test_paths_caught_unless_given_or_on_disk():
    on_disk = {"/shared/docs_drafts"}
    found = ungrounded_paths(INVENTED, str(SEED), exists=lambda p: p in on_disk)
    assert "/gaia/maintenance" in found and "/gaia/scripts/docs_maintenance.py" in found
    assert "./run_incidents_list.sh" in found
    assert "/shared/docs_drafts" not in found
    # a real file the seed didn't spell out passes because it exists ...
    assert ungrounded_paths(GROUNDED, str(SEED), exists=lambda p: True) == []
    # ... and is caught when it doesn't
    assert ungrounded_paths(GROUNDED, str(SEED), exists=lambda p: False) == [
        "/shared/docs_drafts/AS_BUILT_LATEST.md"]


def test_invented_expansion_is_dropped():
    hb, llm = _hb(INVENTED)
    assert hb._expand_seed(llm, SEED) == ""


def test_grounded_expansion_passes(monkeypatch):
    import os
    real = os.path.exists
    monkeypatch.setattr(os.path, "exists",
                        lambda p: p == "/shared/docs_drafts/AS_BUILT_LATEST.md" or real(p))
    hb, llm = _hb(GROUNDED)
    assert hb._expand_seed(llm, SEED) == GROUNDED


def test_expansion_without_paths_passes():
    hb, llm = _hb("Reflect on whether the architecture documentation still matches how you work today.")
    assert hb._expand_seed(llm, SEED).startswith("Reflect on")
