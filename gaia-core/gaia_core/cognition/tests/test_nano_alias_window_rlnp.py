"""rlnp: 'nano' is an alias to Core's engine, so it has Core's window. Its
stale max_model_len of 2048 made the planning call's prompt overflow, which
cut the system message to 200 chars and the planning output to junk."""

import json
import os

import pytest

_CONSTANTS = "/gaia-common/gaia_common/constants/gaia_constants.json"


def _constants():
    if not os.path.exists(_CONSTANTS):
        pytest.skip("gaia_constants.json not mounted")
    return json.load(open(_CONSTANTS))


def test_nano_alias_has_cores_window():
    models = _constants()["MODEL_CONFIGS"]
    nano, core = models["nano"], models["core"]
    assert nano["endpoint"] == core["endpoint"]
    assert nano["max_model_len"] == core["max_model_len"]
