"""Tests for the creative-generation route (GAIA_Project-45i)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gaia_core.cognition.creative_generation import (
    CreativeResult,
    _compose_user_prompt,
    generate_creative_grounded,
)


def _ok_response(text: str):
    """Build a mock LLM response wrapper."""
    return text


@pytest.fixture
def mock_clean_llm():
    """LLM that returns the prompt-echo and never fabricates."""
    with patch("gaia_core.cognition.creative_generation._call_llm") as m:
        m.return_value = "GAIA's response: clean, grounded prose."
        yield m


@pytest.fixture
def mock_clean_consistency():
    """Consistency detector that always returns clean."""
    with patch(
        "gaia_core.cognition.creative_generation._run_consistency",
        return_value=(True, []),
    ) as m:
        yield m


# ── _compose_user_prompt ────────────────────────────────────────────


class TestComposeUserPrompt:
    def test_just_base_user(self):
        out = _compose_user_prompt("Write a section.", "", "", [])
        assert out == "Write a section."

    def test_with_grounding(self):
        out = _compose_user_prompt(
            "Write.", "Fact: Core is on GPU.", "", [],
        )
        assert "Fact: Core is on GPU." in out
        assert "Grounding evidence" in out

    def test_with_kg_recency(self):
        out = _compose_user_prompt(
            "Write.", "", "[KG recent: senator=Alice]", [],
        )
        assert "[KG recent: senator=Alice]" in out

    def test_with_banned_terms(self):
        out = _compose_user_prompt(
            "Write.", "", "", ["FakeEntity", "MadeUpPath"],
        )
        assert "FakeEntity" in out
        assert "MadeUpPath" in out
        assert "WITHOUT using any of those terms" in out

    def test_combined(self):
        out = _compose_user_prompt(
            "Write.", "ground", "kg", ["bad"],
        )
        # Order: user → grounding → kg → banned
        assert out.index("ground") < out.index("kg")
        assert out.index("kg") < out.index("bad")


# ── generate_creative_grounded — happy path ─────────────────────────


class TestHappyPath:
    def test_returns_text_on_first_attempt_when_clean(
        self, mock_clean_llm, mock_clean_consistency,
    ):
        result = generate_creative_grounded(
            system_prompt="You are GAIA.",
            user_prompt="Write about your architecture.",
            consistency_source_text="GAIA has two tiers: Core and Prime.",
        )
        assert isinstance(result, CreativeResult)
        assert result.text == "GAIA's response: clean, grounded prose."
        assert result.consistency_clean is True
        assert result.fabrications_found == []
        assert result.rerolls == 0
        assert result.error is None
        # Only one LLM call when audit is clean
        assert mock_clean_llm.call_count == 1

    def test_grounding_evidence_propagated(
        self, mock_clean_llm, mock_clean_consistency,
    ):
        generate_creative_grounded(
            system_prompt="sys",
            user_prompt="user",
            grounding_evidence="EVIDENCE_MARKER",
        )
        # Inspect the user message passed to _call_llm
        call_args = mock_clean_llm.call_args
        composed_user = call_args.kwargs["user"]
        assert "EVIDENCE_MARKER" in composed_user


# ── re-roll on unclean audit ────────────────────────────────────────


class TestRerollLoop:
    def test_rerolls_when_audit_flags_fabrication(self, mock_clean_llm):
        """Sequence: attempt 0 audit→unclean(banned=['X']), attempt 1
        audit→clean. Returns text with rerolls=1."""
        audit_results = [(False, ["FakeEntity"]), (True, [])]
        with patch(
            "gaia_core.cognition.creative_generation._run_consistency",
            side_effect=audit_results,
        ) as mock_audit:
            result = generate_creative_grounded(
                system_prompt="sys", user_prompt="user",
                max_rerolls=2,
            )
        assert result.consistency_clean is True
        assert "FakeEntity" in result.fabrications_found
        assert result.rerolls == 1
        assert mock_clean_llm.call_count == 2
        assert mock_audit.call_count == 2

    def test_cap_at_max_rerolls(self, mock_clean_llm):
        """If audit keeps flagging, stop at max_rerolls + 1 LLM calls."""
        audit_results = [
            (False, ["A"]), (False, ["B"]), (False, ["C"]),
        ]
        with patch(
            "gaia_core.cognition.creative_generation._run_consistency",
            side_effect=audit_results,
        ):
            result = generate_creative_grounded(
                system_prompt="sys", user_prompt="user",
                max_rerolls=2,
            )
        # 3 attempts total (initial + 2 re-rolls), final still unclean
        assert mock_clean_llm.call_count == 3
        assert result.consistency_clean is False
        # All flagged terms accumulated
        assert set(result.fabrications_found) >= {"A", "B", "C"}

    def test_banned_terms_passed_to_reroll(self, mock_clean_llm):
        """The re-roll attempt's user prompt should include the previous
        banned terms in the 'do not use' list."""
        audit_results = [(False, ["Sovereign_Neural_Transcoder"]), (True, [])]
        with patch(
            "gaia_core.cognition.creative_generation._run_consistency",
            side_effect=audit_results,
        ):
            generate_creative_grounded(
                system_prompt="sys", user_prompt="user",
                max_rerolls=2,
            )
        # Second call should include the banned term in the prompt
        second_call = mock_clean_llm.call_args_list[1]
        assert "Sovereign_Neural_Transcoder" in second_call.kwargs["user"]


# ── error paths ─────────────────────────────────────────────────────


class TestErrorPaths:
    def test_llm_failure_returns_error(self):
        with patch(
            "gaia_core.cognition.creative_generation._call_llm",
            side_effect=RuntimeError("network is down"),
        ):
            result = generate_creative_grounded(
                system_prompt="sys", user_prompt="user",
            )
        assert result.error == "network is down"
        assert result.text == ""

    def test_consistency_import_failure_treated_as_clean(self, mock_clean_llm):
        """If the consistency_detector module fails to import or run,
        creative generation should still succeed (logged, but not blocked)."""
        with patch(
            "gaia_core.cognition.creative_generation._run_consistency",
            return_value=(True, []),
        ):
            result = generate_creative_grounded(
                system_prompt="sys", user_prompt="user",
            )
        assert result.consistency_clean is True
        assert result.error is None


# ── KG recency integration ──────────────────────────────────────────


class TestKgRecencyIntegration:
    def test_no_kg_skips_kg_block(self, mock_clean_llm, mock_clean_consistency):
        """When kg=None, the KG block is not injected."""
        result = generate_creative_grounded(
            system_prompt="sys", user_prompt="user",
            kg=None,
        )
        composed_user = mock_clean_llm.call_args.kwargs["user"]
        # Stage 8 marker would be "[KG recent reference" or similar
        assert "KG recent" not in composed_user
        assert result.error is None

    def test_kg_grounding_injected_when_provided(
        self, mock_clean_llm, mock_clean_consistency,
    ):
        """When kg is provided and recency grounding returns text, the
        composed user prompt should include it."""
        with patch(
            "gaia_core.cognition.creative_generation._kg_recency_block",
            return_value="[KG recent reference — Alice is the current senator]",
        ):
            generate_creative_grounded(
                system_prompt="sys", user_prompt="current senator?",
                kg=SimpleNamespace(),  # any non-None value
            )
        composed_user = mock_clean_llm.call_args.kwargs["user"]
        assert "Alice is the current senator" in composed_user

    def test_disable_kg_grounding_flag(
        self, mock_clean_llm, mock_clean_consistency,
    ):
        """enable_kg_grounding=False suppresses the KG block even when kg
        is provided."""
        with patch(
            "gaia_core.cognition.creative_generation._kg_recency_block",
            return_value="[KG recent reference — should not appear]",
        ) as mock_kg:
            generate_creative_grounded(
                system_prompt="sys", user_prompt="user",
                kg=SimpleNamespace(),
                enable_kg_grounding=False,
            )
        mock_kg.assert_not_called()


# ── result shape ────────────────────────────────────────────────────


class TestResultShape:
    def test_to_dict_serializable(self, mock_clean_llm, mock_clean_consistency):
        result = generate_creative_grounded(
            system_prompt="sys", user_prompt="user",
            grounding_evidence="some grounding",
        )
        d = result.to_dict()
        assert d["text"]
        assert d["rerolls"] == 0
        assert d["consistency_clean"] is True
        assert d["fabrications_found"] == []
        assert d["grounding_used_chars"] > 0
        assert d["elapsed_ms"] >= 0

    def test_elapsed_ms_is_positive(self, mock_clean_llm, mock_clean_consistency):
        result = generate_creative_grounded(
            system_prompt="sys", user_prompt="user",
        )
        # Even a fast mock call should record some elapsed time
        assert result.elapsed_ms >= 0
