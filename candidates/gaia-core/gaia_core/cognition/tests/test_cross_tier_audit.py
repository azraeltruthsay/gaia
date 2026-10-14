"""Tests for cross_tier_audit's samvega emission (GAIA_Project-e8hf).

The bug: cross_tier_audit.py wrote an ad-hoc dict into the samvega/
directory that happened to glob-match the real correction-learning
pipeline's file pattern, but never set `weight` or `corrected_understanding`
— the two fields samvega_kv_fold.py's eligibility filter hard-requires.
Every audit-flagged confabulation was silently invisible to the one
mechanism designed to act on it. The fix: build a real SamvegaArtifact and
go through save_samvega_artifact() so the schema actually matches.
"""

import json

import pytest

from gaia_core.cognition.cross_tier_audit import AuditFinding, _maybe_emit_samvega


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


class TestMaybeEmitSamvega:
    def test_single_low_severity_does_not_emit(self):
        # Matches the pre-existing noise-avoidance gate: a lone severity-1
        # finding shouldn't escalate to a samvega artifact at all.
        findings = [AuditFinding(category="engagement", concern="minor", severity=1)]
        path = _maybe_emit_samvega(
            findings=findings, user_input="hi", final_response="hello",
            journal_entry_id="j1", summary="minor concern",
        )
        assert path is None

    def test_serious_finding_emits_eligible_artifact(self):
        findings = [
            AuditFinding(category="confabulation", concern="invented a sleep cycle window", severity=3),
        ]
        path = _maybe_emit_samvega(
            findings=findings, user_input="How are you?", final_response="Sleeping until 6:30 UTC.",
            journal_entry_id="j2", summary="Fabricated system state.", session_id="sess-1",
        )
        assert path is not None
        data = _read(path)
        # The two fields the fold pipeline hard-requires:
        assert data["weight"] >= 0.3
        assert data["corrected_understanding"].strip() != ""
        # Full schema plumbed through correctly:
        assert data["trigger"] == "cross_tier_audit"
        assert data["session_id"] == "sess-1"
        assert data["packet_id"] == "j2"
        assert data["root_cause"] == "confabulation"
        assert data["reviewed"] is False
        assert data["weight"] == pytest.approx(0.75)

    def test_multiple_findings_sets_dominant_category_as_root_cause(self):
        findings = [
            AuditFinding(category="identity", concern="confused voice", severity=2),
            AuditFinding(category="identity", concern="claimed user's state", severity=2),
            AuditFinding(category="deflection", concern="dodged the question", severity=1),
        ]
        path = _maybe_emit_samvega(
            findings=findings, user_input="u", final_response="r",
            journal_entry_id="j3", summary="s",
        )
        assert path is not None
        data = _read(path)
        assert data["root_cause"] == "identity"

    def test_no_findings_does_not_emit(self):
        path = _maybe_emit_samvega(
            findings=[], user_input="u", final_response="r",
            journal_entry_id="j4", summary="",
        )
        assert path is None
