"""The resting gear is a setting (Azrael, 2026-11-10): every transition back
to rest reads the persisted choice when it happens; the endpoints set it and
shift now only from the everyday gears."""

from __future__ import annotations

import pytest

from gaia_common.lifecycle import states as st
from gaia_common.lifecycle.states import LifecycleState as L, TransitionTrigger as T


@pytest.fixture(autouse=True)
def gear_file(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "RESTING_GEAR_FILE", tmp_path / "resting_gear.json")
    monkeypatch.setattr(st, "RESTING_STATE", L.FOCUSING)


def test_default_then_persisted_choice():
    assert st.resting_state() == L.FOCUSING
    st.set_resting_state(L.AWAKE, set_by="test")
    assert st.resting_state() == L.AWAKE


def test_every_rest_transition_follows_the_setting():
    for state in (L.AWAKE, L.FOCUSING):
        st.set_resting_state(state)
        for current, trigger in st.RESTING_TRANSITIONS:
            assert st.validate_transition(current, trigger) == state


def test_rest_transitions_cover_the_table_defaults():
    defaults = {(c, t) for c, tab in st.TRANSITIONS.items() for t, v in tab.items() if v == st.RESTING_STATE}
    # With the default (focusing) every table entry that is FOCUSING by being
    # "rest" is listed; FOCUSING reached for other reasons (escalation) is not.
    rest_like = {(c, t) for c, t in defaults if t != T.ESCALATION_NEEDED}
    assert rest_like == st.RESTING_TRANSITIONS


def test_other_transitions_unchanged():
    st.set_resting_state(L.AWAKE)
    assert st.validate_transition(L.FOCUSING, T.TASK_COMPLETE) == L.AWAKE
    assert st.validate_transition(L.AWAKE, T.ESCALATION_NEEDED) == L.FOCUSING
    assert st.validate_transition(L.AWAKE, T.IDLE_TIMEOUT) == L.PARKED


def test_only_gear_1_or_2():
    with pytest.raises(ValueError):
        st.set_resting_state(L.SLEEP)


def test_corrupt_file_falls_back_to_default():
    st.RESTING_GEAR_FILE.write_text("{not json")
    assert st.resting_state() == L.FOCUSING
