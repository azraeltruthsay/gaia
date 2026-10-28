"""578k: the user's message reaches Core as written. A fuzzy name corrector
at the agent boundary rewrote 'Who wrote Pride and Prejudice?' into
'…Prime and Prejudice' (difflib ratio 0.8 to the project name 'Prime')."""

import inspect

from gaia_core.cognition import agent_core


def test_run_turn_does_not_rewrite_user_input_against_project_names():
    src = inspect.getsource(agent_core.AgentCore.run_turn)
    assert "correct_text" not in src and "entity_validator" not in src


def test_validator_module_is_gone():
    import importlib.util
    assert importlib.util.find_spec("gaia_common.utils.entity_validator") is None
