"""Tests for Saṃvega — Semantic Discernment Artifacts."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gaia_core.cognition.samvega import (
    SAMVEGA_ARCHIVE_DIR,
    SAMVEGA_DIR,
    SamvegaArtifact,
    SamvegaTrigger,
    archive_artifact,
    compute_samvega_weight,
    list_artifacts_by_weight,
    list_unreviewed_artifacts,
    save_samvega_artifact,
    update_artifact,
)


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _patch_samvega_dirs(tmp_path, monkeypatch):
    """Redirect samvega directories to a temp path for test isolation."""
    samvega = tmp_path / "samvega"
    samvega.mkdir()
    monkeypatch.setattr("gaia_core.cognition.samvega.SAMVEGA_DIR", samvega)
    monkeypatch.setattr("gaia_core.cognition.samvega.SAMVEGA_ARCHIVE_DIR", samvega / "archive")
    return samvega


def _make_artifact(**overrides) -> SamvegaArtifact:
    """Create a test artifact with sensible defaults."""
    defaults = dict(
        timestamp="2026-02-27T12:00:00+00:00",
        session_id="test-session",
        packet_id="pkt-001",
        trigger=SamvegaTrigger.USER_CORRECTION.value,
        original_output_summary="Test output",
        what_went_wrong="Got it wrong",
        root_cause="Misunderstood context",
        values_misaligned=["accuracy"],
        corrected_understanding="Should have checked context first",
        weight=0.6,
    )
    defaults.update(overrides)
    return SamvegaArtifact(**defaults)


# ── Serialization ────────────────────────────────────────────────────────


class TestArtifactSerialization:
    def test_round_trip(self):
        artifact = _make_artifact()
        d = artifact.to_dict()
        restored = SamvegaArtifact.from_dict(d)
        assert restored.trigger == artifact.trigger
        assert restored.weight == artifact.weight
        assert restored.values_misaligned == ["accuracy"]
        assert restored.promoted_to_tier5 is False
        assert restored.reviewed is False

    def test_to_json(self):
        artifact = _make_artifact()
        j = artifact.to_json()
        data = json.loads(j)
        assert data["artifact_type"] == "samvega"
        assert data["trigger"] == "user_correction"


# ── Weight Calculation ───────────────────────────────────────────────────


class TestWeightCalculation:
    def test_user_correction_base(self):
        w = compute_samvega_weight("user_correction")
        assert w == pytest.approx(0.6)

    def test_confidence_mismatch(self):
        w = compute_samvega_weight(
            "confidence_mismatch",
            original_confidence=0.9,
            reflection_confidence=0.4,
        )
        assert w == pytest.approx(0.5)

    def test_pattern_detection_base(self):
        w = compute_samvega_weight("pattern_detection")
        assert w == pytest.approx(0.4)

    def test_observer_block_multiplier(self):
        w = compute_samvega_weight("user_correction", observer_severity="BLOCK")
        assert w == pytest.approx(0.6 * 1.5)

    def test_observer_caution_multiplier(self):
        w = compute_samvega_weight("user_correction", observer_severity="CAUTION")
        assert w == pytest.approx(0.6 * 1.2)

    def test_repeated_domain_multiplier(self):
        w = compute_samvega_weight("user_correction", is_repeated_domain=True)
        assert w == pytest.approx(0.6 * 1.3)

    def test_stacked_multipliers(self):
        w = compute_samvega_weight(
            "user_correction",
            observer_severity="BLOCK",
            is_repeated_domain=True,
        )
        # 0.6 * 1.5 * 1.3 = 1.17 → clamped to 1.0
        assert w == pytest.approx(1.0)

    def test_clamp_to_zero(self):
        w = compute_samvega_weight(
            "confidence_mismatch",
            original_confidence=0.3,
            reflection_confidence=0.8,
        )
        assert w == pytest.approx(0.0)

    def test_unknown_trigger(self):
        w = compute_samvega_weight("something_new")
        assert w == pytest.approx(0.3)

    def test_custom_multipliers(self):
        w = compute_samvega_weight(
            "user_correction",
            observer_severity="BLOCK",
            multipliers={"observer_block": 2.0, "observer_caution": 1.0, "repeated_domain": 1.0},
        )
        # 0.6 * 2.0 = 1.2 → clamped to 1.0
        assert w == pytest.approx(1.0)


# ── File CRUD ────────────────────────────────────────────────────────────


class TestFileCRUD:
    def test_save_and_list(self, tmp_path):
        artifact = _make_artifact()
        path = save_samvega_artifact(artifact)
        assert path is not None
        assert path.exists()

        unreviewed = list_unreviewed_artifacts()
        assert len(unreviewed) == 1
        assert unreviewed[0][1]["trigger"] == "user_correction"

    def test_list_unreviewed_excludes_reviewed(self, tmp_path):
        a1 = _make_artifact(reviewed=False, weight=0.8)
        a2 = _make_artifact(reviewed=True, weight=0.5)
        save_samvega_artifact(a1)
        # Manually save the reviewed one
        import time; time.sleep(0.01)  # ensure unique timestamp
        save_samvega_artifact(a2)

        # Mark the second as reviewed by updating
        all_files = list(Path(tmp_path / "samvega").glob("samvega_*.json"))
        for f in all_files:
            data = json.loads(f.read_text())
            if data.get("weight") == 0.5:
                data["reviewed"] = True
                f.write_text(json.dumps(data, indent=2))

        unreviewed = list_unreviewed_artifacts()
        assert len(unreviewed) == 1
        assert unreviewed[0][1]["weight"] == 0.8

    def test_list_by_weight(self, tmp_path):
        save_samvega_artifact(_make_artifact(weight=0.3))
        import time; time.sleep(0.01)
        save_samvega_artifact(_make_artifact(weight=0.7))
        import time; time.sleep(0.01)
        save_samvega_artifact(_make_artifact(weight=0.9))

        results = list_artifacts_by_weight(min_weight=0.5)
        assert len(results) == 2
        # Sorted by weight descending
        assert results[0][1]["weight"] == 0.9
        assert results[1][1]["weight"] == 0.7

    def test_update_artifact(self, tmp_path):
        artifact = _make_artifact()
        path = save_samvega_artifact(artifact)
        fname = path.name

        data = json.loads(path.read_text())
        data["reviewed"] = True
        data["reviewed_at"] = "2026-02-27T14:00:00+00:00"
        assert update_artifact(fname, data) is True

        updated = json.loads(path.read_text())
        assert updated["reviewed"] is True

    def test_update_nonexistent(self):
        assert update_artifact("nonexistent.json", {}) is False

    def test_archive_artifact(self, tmp_path):
        artifact = _make_artifact()
        path = save_samvega_artifact(artifact)
        fname = path.name

        assert archive_artifact(fname) is True
        assert not path.exists()
        archive_path = Path(tmp_path / "samvega" / "archive" / fname)
        assert archive_path.exists()

    def test_archive_nonexistent(self):
        assert archive_artifact("nonexistent.json") is False

    def test_empty_dir_returns_empty(self, tmp_path):
        assert list_unreviewed_artifacts() == []
        assert list_artifacts_by_weight(0.0) == []
