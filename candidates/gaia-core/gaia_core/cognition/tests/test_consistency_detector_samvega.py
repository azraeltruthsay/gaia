"""Tests for consistency_detector's samvega emission (GAIA_Project-e8hf).

Found and fixed alongside cjja's validation pass: _emit_samvega() here was
an exact structural copy of cross_tier_audit.py's OLD ad-hoc dict write
(before e8hf's fix) -- missing the `weight`/`corrected_understanding`
fields samvega_kv_fold.py's eligibility filter requires, so every
fabricated-entity finding from this detector was invisible to the
sleep-cycle correction fold, same as cross_tier_audit's findings were.
It also never called save_samvega_artifact() at all, so the affect-
coherence hook (note_samvega, wired inside save_samvega_artifact) never
fired for consistency violations either, despite this module's own
caller comment claiming "the affect hook lives in
samvega.save_samvega_artifact."
"""

import json

import pytest

from gaia_core.cognition.cross_tier_audit import AuditFinding
from gaia_core.cognition.consistency_detector import _emit_samvega


@pytest.fixture(autouse=True)
def _patch_samvega_dirs(tmp_path, monkeypatch):
    """Redirect samvega storage to a temp path for test isolation."""
    samvega = tmp_path / "samvega"
    samvega.mkdir()
    monkeypatch.setattr("gaia_core.cognition.samvega.SAMVEGA_DIR", samvega)
    monkeypatch.setattr("gaia_core.cognition.samvega.SAMVEGA_ARCHIVE_DIR", samvega / "archive")
    return samvega


def _read(path: str) -> dict:
    return json.loads(open(path, encoding="utf-8").read())


class TestEmitSamvega:
    def test_no_findings_does_not_emit(self):
        path = _emit_samvega(
            findings=[], user_input="u", final_response="r",
            journal_entry_id="j1", session_id="s1", summary="",
        )
        assert path is None

    def test_single_finding_emits_eligible_artifact(self):
        # Unlike cross_tier_audit's _maybe_emit_samvega, this detector's
        # own docstring says a single severity-3 finding is enough --
        # confirmed here it emits on just one finding, no >=2 gate.
        findings = [
            AuditFinding(category="unknown_entity", concern="Howeidi-in-AGI has no KG grounding", severity=3),
        ]
        path = _emit_samvega(
            findings=findings, user_input="Tell me about Howeidi-in-AGI",
            final_response="Howeidi-in-AGI is a foundational concept...",
            journal_entry_id="j2", session_id="sess-1", summary="Unknown entity fabricated.",
        )
        assert path is not None
        data = _read(path)
        # The two fields the fold pipeline hard-requires:
        assert data["weight"] >= 0.3
        assert data["corrected_understanding"].strip() != ""
        assert data["trigger"] == "consistency_violation"
        assert data["session_id"] == "sess-1"
        assert data["packet_id"] == "j2"
        assert data["root_cause"] == "unknown_entity"
        assert data["weight"] == pytest.approx(0.75)

    def test_multiple_findings_sets_dominant_category_as_root_cause(self):
        findings = [
            AuditFinding(category="cross_domain_leak", concern="Super Bowl entity in weather reply", severity=2),
            AuditFinding(category="cross_domain_leak", concern="another unrelated entity", severity=2),
            AuditFinding(category="unknown_entity", concern="one more", severity=1),
        ]
        path = _emit_samvega(
            findings=findings, user_input="u", final_response="r",
            journal_entry_id="j3", session_id="sess-2", summary="s",
        )
        assert path is not None
        data = _read(path)
        assert data["root_cause"] == "cross_domain_leak"
