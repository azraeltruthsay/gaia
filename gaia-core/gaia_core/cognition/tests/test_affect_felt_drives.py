"""felt-line renders the `drives` axis (GAIA_Project-7n3/d69/3rr).

Root cause of "affect never voices": affect_felt_line() read only feels/
curious_about/tired_of, but the appraiser writes primarily to `drives`
(competence/coherence). A genuinely-fed organ therefore rendered EMPTY.
These pin that drives now surface as number-free felt-facts.
"""
from gaia_core.cognition.affect_runtime import affect_felt_line


def test_competence_drive_renders():
    snap = {"feels": {}, "drives": {"competence": 0.30}, "curious_about": {}, "tired_of": {}}
    assert affect_felt_line(snap) == "a quiet pull to get this right"


def test_strong_competence_uses_strong_band():
    snap = {"feels": {}, "drives": {"competence": 0.75}, "curious_about": {}, "tired_of": {}}
    assert affect_felt_line(snap) == "a strong pull to get this right"


def test_coherence_drive_renders():
    snap = {"feels": {}, "drives": {"coherence": 0.50}, "curious_about": {}, "tired_of": {}}
    assert affect_felt_line(snap) == "something that won't quite settle"


def test_drive_composes_with_curious():
    snap = {"feels": {}, "drives": {"coherence": 0.50},
            "curious_about": {"the_sae_atlas": 0.55}, "tired_of": {}}
    line = affect_felt_line(snap)
    assert "something that won't quite settle" in line
    assert "drawn toward the sae atlas" in line


def test_only_one_drive_clause_kept_brief():
    snap = {"feels": {}, "drives": {"competence": 0.75, "coherence": 0.50},
            "curious_about": {}, "tired_of": {}}
    line = affect_felt_line(snap)
    # top drive only (competence), not both — keeps the felt-line short
    assert line == "a strong pull to get this right"


def test_below_threshold_drive_is_silent():
    # _PROMPT_THRESHOLD = 0.15 — a tiny drive shouldn't render.
    snap = {"feels": {}, "drives": {"competence": 0.05}, "curious_about": {}, "tired_of": {}}
    assert affect_felt_line(snap) == ""


def test_unknown_drive_name_does_not_render_raw():
    # An unmapped drive must NOT leak a mechanical name into the felt-line.
    snap = {"feels": {}, "drives": {"mystery_drive": 0.9}, "curious_about": {}, "tired_of": {}}
    assert "mystery_drive" not in affect_felt_line(snap)


def test_empty_snapshot_is_empty_line():
    assert affect_felt_line({"feels": {}, "drives": {}, "curious_about": {}, "tired_of": {}}) == ""
