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

from gaia_core.cognition.cross_tier_audit import (
    AuditFinding, _maybe_emit_samvega, _parse_audit_response,
)


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


class TestParseAuditResponseFollowup:
    """e8hf Phase B: a flagged turn can carry a FOLLOWUP line (the visible
    same-conversation self-correction); a clean one never should, even if
    Prime hallucinates one anyway."""

    def test_flagged_with_followup_extracts_it(self):
        raw = (
            "AUDIT: flagged\n"
            "FLAGS:\n"
            "- confabulation: invented a fictional training detail\n"
            "\n"
            "SUMMARY: Core made something up instead of answering directly.\n"
            "\n"
            "FOLLOWUP: Quick correction — the sword's name is Excalibur.\n"
        )
        clean, findings, summary, followup = _parse_audit_response(raw)
        assert clean is False
        assert len(findings) == 1
        assert followup == "Quick correction — the sword's name is Excalibur."

    def test_clean_ignores_followup_even_if_present(self):
        raw = "AUDIT: clean\nSUMMARY: No concerns.\nFOLLOWUP: should never surface.\n"
        clean, findings, summary, followup = _parse_audit_response(raw)
        assert clean is True
        assert followup == ""

    def test_flagged_without_followup_is_empty(self):
        raw = (
            "AUDIT: flagged\n"
            "FLAGS:\n"
            "- deflection: dodged a direct question\n"
            "SUMMARY: Core deflected.\n"
        )
        clean, findings, summary, followup = _parse_audit_response(raw)
        assert clean is False
        assert followup == ""


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


# ── Verdict-first (d12t follow-up) ──────────────────────────────────────

class _Pool:
    """Fake model pool: replies in order, records every call."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def forward_to_model(self, role, messages, **kw):
        self.calls.append({"role": role, "messages": [dict(m) for m in messages], **kw})
        return {"choices": [{"message": {"content": self.replies.pop(0)}}]}


def _run(monkeypatch, pool, cfg=None):
    from gaia_core.cognition import cross_tier_audit as cta
    notes = []
    monkeypatch.setattr(cta, "_claim_entry", lambda eid: True)
    monkeypatch.setattr(cta, "_lifecycle_allows_audit", lambda: (True, "ok"))
    monkeypatch.setattr(cta, "_maybe_emit_samvega", lambda **kw: None)
    import gaia_core.memory.journal as journal
    monkeypatch.setattr(journal, "annotate_entry", lambda eid, note, **kw: notes.append(note) or True)
    res = cta.run_cross_tier_audit_sync(
        user_input="What is the capital of France?", thinking="It's Paris.",
        final_response="Paris.", journal_entry_id="cognition-1", model_pool=pool,
        config={"CROSS_TIER_AUDIT": cfg or {}})
    return res, notes


def test_clean_verdict_ends_the_audit_after_one_short_call(monkeypatch):
    pool = _Pool("clean")
    res, notes = _run(monkeypatch, pool)
    assert res.clean and not res.findings and len(pool.calls) == 1
    call = pool.calls[0]
    assert call["max_tokens"] <= 8 and call["chat_template_kwargs"] == {"enable_thinking": False}
    assert call["temperature"] == 0.0  # deterministic gate
    assert call["messages"][-1]["content"].endswith("answer with one word, clean or flagged.")
    assert notes and notes[0].startswith("Cross-tier audit (clean, Prime)")  # still journaled


def test_flagged_verdict_gets_the_full_audit_extending_the_same_conversation(monkeypatch):
    detail = "AUDIT: flagged\nFLAGS:\n- confabulation: invented a component\nSUMMARY: made something up"
    pool = _Pool("AUDIT: flagged", detail)
    res, _ = _run(monkeypatch, pool)
    assert not res.clean and res.findings[0].category == "confabulation"
    first, second = pool.calls
    # Same prefix: the second call only appends turns (gaia_cpp reuses the KV).
    assert second["messages"][:2] == first["messages"]
    assert second["messages"][2] == {"role": "assistant", "content": "AUDIT: flagged"}
    assert "chat_template_kwargs" not in second  # details think as before


def test_full_audit_can_overturn_a_flagged_verdict(monkeypatch):
    res, _ = _run(monkeypatch, _Pool("flagged", "AUDIT: clean\nSUMMARY: fine on reflection"))
    assert res.clean


def test_unreadable_verdict_falls_through_to_the_full_audit(monkeypatch):
    pool = _Pool("hmm", "AUDIT: clean\nSUMMARY: ok")
    res, _ = _run(monkeypatch, pool)
    assert res.clean and len(pool.calls) == 2


def test_verdict_first_can_be_turned_off(monkeypatch):
    pool = _Pool("AUDIT: clean\nSUMMARY: ok")
    res, _ = _run(monkeypatch, pool, cfg={"verdict_first": False})
    assert res.clean and len(pool.calls) == 1
    assert pool.calls[0]["messages"][-1]["content"].endswith("Run the audit now.")
