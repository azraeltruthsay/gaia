"""Tests for the LLM body generator + sleep-cycle entry (99z)."""

from __future__ import annotations

from unittest.mock import patch


from gaia_core.cognition.skill_llm_body import (
    _build_user_prompt,
    _wrap_playbook,
    make_llm_body_generator,
    run_skill_creator_cycle,
)


def _failures(n: int, *, intent="research", query_prefix="latest python") -> list[dict]:
    return [
        {
            "t": f"2026-05-27T12:0{i}:00+00:00",
            "intent": intent,
            "query": f"{query_prefix} {i}",
            "pattern": f"{intent}:{query_prefix}",
        }
        for i in range(n)
    ]


def _mock_creative_result(text: str = "", error: str = None,
                          consistency_clean: bool = True,
                          fabrications: list = None, rerolls: int = 0):
    """Build a CreativeResult-shape mock."""
    from gaia_core.cognition.creative_generation import CreativeResult
    return CreativeResult(
        text=text,
        rerolls=rerolls,
        fabrications_found=fabrications or [],
        consistency_clean=consistency_clean,
        grounding_used="",
        elapsed_ms=10,
        error=error,
    )


# ── _build_user_prompt ──────────────────────────────────────────────


class TestBuildUserPrompt:
    def test_includes_pattern_and_failures(self):
        out = _build_user_prompt("research:latest python", _failures(3))
        assert "research:latest python" in out
        assert "n=3" in out
        for q in ["latest python 0", "latest python 1", "latest python 2"]:
            assert q in out

    def test_truncates_long_query(self):
        long_q = "x" * 500
        failures = [{"t": "", "intent": "x", "query": long_q, "pattern": ""}]
        out = _build_user_prompt("x:y", failures)
        # Should be truncated to 200 + "..."
        assert "..." in out
        assert long_q not in out

    def test_caps_failure_samples(self):
        out = _build_user_prompt("x:y", _failures(15))
        # Only first 10 in detail; rest summarized
        assert "10 more" in out or "5 more" in out


# ── _wrap_playbook ──────────────────────────────────────────────────


class TestWrapPlaybook:
    def test_contains_playbook_text(self):
        body = _wrap_playbook("x:y", _failures(3), "1. Do thing\n2. Then this")
        assert "1. Do thing" in body
        assert "2. Then this" in body

    def test_includes_failure_block(self):
        body = _wrap_playbook("x:y", _failures(3), "1. step")
        assert "Recent failure samples" in body
        assert "latest python 0" in body

    def test_status_draft_marker(self):
        body = _wrap_playbook("x:y", _failures(3), "1. step")
        assert "draft" in body.lower()


# ── make_llm_body_generator ─────────────────────────────────────────


class TestMakeLLMBodyGenerator:
    def test_returns_callable(self):
        gen = make_llm_body_generator()
        assert callable(gen)

    def test_calls_creative_generation(self):
        gen = make_llm_body_generator()
        with patch(
            "gaia_core.cognition.creative_generation.generate_creative_grounded",
            return_value=_mock_creative_result(text="1. Look it up\n2. Respond"),
        ) as mock_call:
            body = gen("research:python", _failures(3))
        assert mock_call.called
        # Body wraps the playbook
        assert "Look it up" in body
        assert "Recent failure samples" in body

    def test_falls_back_on_transport_error(self):
        gen = make_llm_body_generator()
        with patch(
            "gaia_core.cognition.creative_generation.generate_creative_grounded",
            side_effect=RuntimeError("prime unreachable"),
        ):
            body = gen("research:python", _failures(3))
        # Deterministic fallback marker
        assert "TODO" in body

    def test_falls_back_on_result_error(self):
        gen = make_llm_body_generator()
        with patch(
            "gaia_core.cognition.creative_generation.generate_creative_grounded",
            return_value=_mock_creative_result(error="timeout"),
        ):
            body = gen("research:python", _failures(3))
        assert "TODO" in body

    def test_falls_back_on_empty_text(self):
        gen = make_llm_body_generator()
        with patch(
            "gaia_core.cognition.creative_generation.generate_creative_grounded",
            return_value=_mock_creative_result(text=""),
        ):
            body = gen("research:python", _failures(3))
        assert "TODO" in body

    def test_strips_think_tags(self):
        gen = make_llm_body_generator()
        playbook_with_think = (
            "<think>let me think...</think>\n"
            "1. Look it up\n"
            "2. Respond"
        )
        with patch(
            "gaia_core.cognition.creative_generation.generate_creative_grounded",
            return_value=_mock_creative_result(text=playbook_with_think),
        ):
            body = gen("x:y", _failures(3))
        assert "<think>" not in body
        assert "Look it up" in body

    def test_accepts_unclean_with_fabrications(self):
        """Auto-drafts are human-reviewed — we still accept a body
        with fabricated terms rather than block on the consistency
        audit (the human filters before promotion)."""
        gen = make_llm_body_generator()
        with patch(
            "gaia_core.cognition.creative_generation.generate_creative_grounded",
            return_value=_mock_creative_result(
                text="1. Step\n2. Other",
                consistency_clean=False,
                fabrications=["FakeEntity"],
                rerolls=2,
            ),
        ):
            body = gen("x:y", _failures(3))
        assert "1. Step" in body


# ── run_skill_creator_cycle ─────────────────────────────────────────


class TestRunSkillCreatorCycle:
    def test_no_failures_returns_zero(self, tmp_path, monkeypatch):
        # Point the failure log at an empty file
        log_file = tmp_path / "empty.jsonl"
        log_file.write_text("")
        monkeypatch.setenv("GAIA_SKILL_FAILURE_LOG", str(log_file))
        out_dir = tmp_path / "auto"

        result = run_skill_creator_cycle(
            output_dir=out_dir, use_llm=False,
        )
        assert result["ok"] is True
        assert result["drafted"] == 0

    def test_drafts_when_threshold_hit(self, tmp_path, monkeypatch):
        log_file = tmp_path / "log.jsonl"
        monkeypatch.setenv("GAIA_SKILL_FAILURE_LOG", str(log_file))
        out_dir = tmp_path / "auto"
        from gaia_common.utils.skill_failures import record_failure
        for _ in range(3):
            record_failure(
                intent="research", query="latest python release",
                log_path=str(log_file),
            )
        result = run_skill_creator_cycle(
            output_dir=out_dir, use_llm=False,
        )
        assert result["drafted"] == 1
        # Skill file written
        assert any(out_dir.rglob("SKILL.md"))

    def test_reload_url_called_on_drafts(self, tmp_path, monkeypatch):
        log_file = tmp_path / "log.jsonl"
        from gaia_common.utils.skill_failures import record_failure
        for _ in range(3):
            record_failure(
                intent="x", query="some query", log_path=str(log_file),
            )
        out_dir = tmp_path / "auto"

        with patch("urllib.request.urlopen") as mock_open:
            # Mock context manager
            mock_open.return_value.__enter__ = lambda s: s
            mock_open.return_value.__exit__ = lambda *a: None
            mock_open.return_value.read.return_value = b'{"ok": true}'

            result = run_skill_creator_cycle(
                log_path=str(log_file),
                output_dir=out_dir,
                use_llm=False,
                reload_url="http://fake/reload",
            )
        assert result["drafted"] == 1
        assert result["reloaded"] is True

    def test_reload_failure_swallowed(self, tmp_path, monkeypatch):
        log_file = tmp_path / "log.jsonl"
        from gaia_common.utils.skill_failures import record_failure
        for _ in range(3):
            record_failure(
                intent="x", query="some query", log_path=str(log_file),
            )
        out_dir = tmp_path / "auto"

        with patch(
            "urllib.request.urlopen",
            side_effect=RuntimeError("mcp down"),
        ):
            result = run_skill_creator_cycle(
                log_path=str(log_file),
                output_dir=out_dir,
                use_llm=False,
                reload_url="http://fake/reload",
            )
        # Drafts still succeeded; reload failed gracefully
        assert result["drafted"] == 1
        assert result["reloaded"] is False
        assert "mcp down" in (result.get("reload_error") or "")
