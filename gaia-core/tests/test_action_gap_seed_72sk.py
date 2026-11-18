"""72sk: a real tool called with an action it lacks seeds a capability gap."""
import json

from gaia_core.main import action_gap_seed

ERR = "Unknown action state for domain worldbuild. Available: [campaigns, search, list, get, create, update]"


def _payload(seed):
    return json.loads(seed.split("Payload: ", 1)[1])


def test_action_on_another_tool_is_a_misleading_name():
    seed = action_gap_seed("worldbuild", "world", ERR, "how are you doing?", "s1")
    p = _payload(seed)
    assert p["kind"] == "misleading_name" and "introspect" in p["action_found_in"]
    assert "knowledge gap" in seed      # thought_seed types it knowledge_gap -> CodeMind


def test_action_nowhere_is_a_missing_action():
    p = _payload(action_gap_seed("worldbuild", "teleport", ERR, "x", "s1"))
    assert p["kind"] == "missing_action" and p["action_found_in"] == []


def test_other_errors_seed_nothing():
    assert action_gap_seed("web", "fetch", "timeout", "x", "s1") is None
