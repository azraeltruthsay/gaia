"""Sleep-task notices are not her thoughts, and aren't refiled (gb7y, 2026-11-06).

Run as turns, "10 stale docs... review and apply" and "curriculum pairs...
review and approve" read as instructions: she said she had committed the
docs and approved the pairs, with no tool for either. The docs notice was
planted every ~11 minutes: 13,668 archived.
"""
import json
from unittest.mock import MagicMock

from gaia_core.cognition.heartbeat import ThoughtSeedHeartbeat
from gaia_core.cognition.thought_seed import same_seed_waiting

NOTICE = ("Documentation maintenance: 10 stale docs detected. Targets: knowledge/system_reference/"
          "AS_BUILT_ARCHITECTURE.md. 10 draft updates saved to /shared/docs_drafts/. Review and apply during next session.")


def test_notices_are_archived_without_asking_the_model():
    hb = ThoughtSeedHeartbeat.__new__(ThoughtSeedHeartbeat)
    llm = MagicMock()
    for kind in ("docs_maintenance", "knowledge_ingestion"):
        decision, reason = hb._triage_seed(llm, {"seed": NOTICE, "seed_type": kind, "context": {}})
        assert decision == "archive" and "notice" in reason
    llm.create_chat_completion.assert_not_called()


def test_her_own_seeds_still_go_to_triage():
    hb = ThoughtSeedHeartbeat.__new__(ThoughtSeedHeartbeat)
    llm = MagicMock()
    llm.create_chat_completion.return_value = {"choices": [{"message": {"content": "ACT\nworth looking into"}}]}
    assert hb._triage_seed(llm, {"seed": "Why did the caching layer slow down?", "seed_type": "general",
                                 "context": {}})[0] == "act"
    assert hb._triage_seed(llm, {"seed": "What is a qubit?", "seed_type": "knowledge_gap"})[0] == "act"


def test_the_same_notice_is_not_planted_twice(tmp_path):
    (tmp_path / "pending").mkdir()
    assert not same_seed_waiting(NOTICE, tmp_path)
    (tmp_path / "pending" / "seed_1.json").write_text(json.dumps({"seed": NOTICE}))
    assert same_seed_waiting(NOTICE, tmp_path)
    assert not same_seed_waiting(NOTICE.replace("10 stale", "11 stale"), tmp_path)
    assert not same_seed_waiting("", tmp_path)
