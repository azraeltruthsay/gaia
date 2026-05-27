"""Tests for the SAE atlas meta synthesis (GAIA_Project-874).

When the on-disk meta.json is stale or missing, the /api/activations/atlas
endpoint synthesizes layer info from the tail of the activation stream
so the UI doesn't need a manual file edit to keep up with the live
engine.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest


@pytest.fixture
def streamed_layers(tmp_path: Path):
    """Build a fake activation stream + atlas dir; rewire the route's
    module-level paths to point at them. Yields (atlas_dir, stream_path).
    Restores the originals on teardown.
    """
    atlas_dir = tmp_path / "atlas"
    atlas_dir.mkdir()
    stream_path = tmp_path / "activation_stream.jsonl"

    import gaia_web.routes.activations as mod
    orig_atlas = mod._ATLAS_DIR
    orig_stream = mod._LOG_PATH
    mod._ATLAS_DIR = str(atlas_dir)
    mod._LOG_PATH = str(stream_path)
    try:
        yield atlas_dir, stream_path
    finally:
        mod._ATLAS_DIR = orig_atlas
        mod._LOG_PATH = orig_stream


def _write_stream_records(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _rec(token: str, tier: str, layers: list[int]) -> dict:
    """Build one fake activation_stream record."""
    return {
        "ts": "2026-05-27T12:00:00",
        "tier": tier,
        "token": token,
        "token_idx": 0,
        "session_id": "test",
        "features": [
            {"idx": 100 + L, "strength": 1.0, "label": f"neuron_{L}", "layer": L}
            for L in layers
        ],
    }


# ── _discover_layers_from_stream ─────────────────────────────────────


class TestDiscoverLayersFromStream:
    def test_empty_stream_returns_empty(self, streamed_layers):
        atlas_dir, stream_path = streamed_layers
        # File doesn't exist
        from gaia_web.routes.activations import _discover_layers_from_stream
        info = _discover_layers_from_stream(str(stream_path))
        assert info["layers"] == []
        assert info["tier"] is None
        assert info["sample_count"] == 0

    def test_zero_byte_file(self, streamed_layers):
        atlas_dir, stream_path = streamed_layers
        stream_path.write_text("")
        from gaia_web.routes.activations import _discover_layers_from_stream
        info = _discover_layers_from_stream(str(stream_path))
        assert info["layers"] == []
        assert info["sample_count"] == 0

    def test_single_record_yields_layers(self, streamed_layers):
        atlas_dir, stream_path = streamed_layers
        _write_stream_records(stream_path, [
            _rec("hi", "core", [4, 8, 12, 16]),
        ])
        from gaia_web.routes.activations import _discover_layers_from_stream
        info = _discover_layers_from_stream(str(stream_path))
        assert info["layers"] == [4, 8, 12, 16]
        assert info["tier"] == "core"
        assert info["sample_count"] == 1

    def test_distinct_layers_merged(self, streamed_layers):
        atlas_dir, stream_path = streamed_layers
        _write_stream_records(stream_path, [
            _rec("a", "core", [4, 8]),
            _rec("b", "core", [12, 16]),
            _rec("c", "core", [4, 32]),  # overlap with first
        ])
        from gaia_web.routes.activations import _discover_layers_from_stream
        info = _discover_layers_from_stream(str(stream_path))
        assert info["layers"] == [4, 8, 12, 16, 32]
        assert info["tier"] == "core"

    def test_tier_none_when_mixed(self, streamed_layers):
        atlas_dir, stream_path = streamed_layers
        _write_stream_records(stream_path, [
            _rec("a", "core", [4]),
            _rec("b", "prime", [8]),
        ])
        from gaia_web.routes.activations import _discover_layers_from_stream
        info = _discover_layers_from_stream(str(stream_path))
        # Mixed tier set → no definitive tier
        assert info["tier"] is None
        # But layers from both ARE merged
        assert set(info["layers"]) == {4, 8}

    def test_malformed_json_skipped(self, streamed_layers):
        atlas_dir, stream_path = streamed_layers
        with open(stream_path, "w") as f:
            f.write(json.dumps(_rec("ok", "core", [4, 8])) + "\n")
            f.write("not valid json\n")
            f.write(json.dumps(_rec("ok2", "core", [12])) + "\n")
        from gaia_web.routes.activations import _discover_layers_from_stream
        info = _discover_layers_from_stream(str(stream_path))
        assert info["sample_count"] == 2  # the two valid lines
        assert info["layers"] == [4, 8, 12]

    def test_tail_window_drops_old_records(self, streamed_layers):
        """Older records past the tail window are not parsed.

        Build a stream long enough that the tail_bytes window excludes
        the first record; the layers from that record should not appear.
        """
        atlas_dir, stream_path = streamed_layers
        # First record will be at offset 0; tail will skip it
        records = [_rec(f"t{i}", "core", [4 + i]) for i in range(200)]
        _write_stream_records(stream_path, records)
        from gaia_web.routes.activations import _discover_layers_from_stream
        # 256 bytes window is small enough to drop most records
        info = _discover_layers_from_stream(
            str(stream_path), tail_bytes=512, max_lines=5,
        )
        # We should see SOME layers but not the full union
        assert 0 < len(info["layers"]) < 200

    def test_no_features_field_safe(self, streamed_layers):
        atlas_dir, stream_path = streamed_layers
        _write_stream_records(stream_path, [
            {"ts": "x", "tier": "core", "token": "y"},  # no features
        ])
        from gaia_web.routes.activations import _discover_layers_from_stream
        info = _discover_layers_from_stream(str(stream_path))
        assert info["layers"] == []
        assert info["tier"] == "core"
        assert info["sample_count"] == 1


# ── get_atlas integration ────────────────────────────────────────────


class TestGetAtlasSynthesis:
    @pytest.mark.asyncio
    async def test_missing_meta_synthesizes_from_stream(self, streamed_layers):
        atlas_dir, stream_path = streamed_layers
        _write_stream_records(stream_path, [
            _rec("a", "core", [4, 8, 12, 16, 32, 36, 40, 41]),
        ])
        from gaia_web.routes.activations import get_atlas
        result = await get_atlas()
        # Layer dict reflects synthesized set
        keys = sorted(int(k) for k in result["layers"].keys())
        assert keys == [4, 8, 12, 16, 32, 36, 40, 41]
        # Meta.json should have been written to disk
        meta = json.loads((atlas_dir / "meta.json").read_text())
        assert meta["layers"] == [4, 8, 12, 16, 32, 36, 40, 41]
        assert meta["source"] == "auto_from_activation_stream"

    @pytest.mark.asyncio
    async def test_stale_meta_gets_refreshed(self, streamed_layers):
        atlas_dir, stream_path = streamed_layers
        # On-disk meta.json says one thing
        (atlas_dir / "meta.json").write_text(json.dumps({
            "layers": [12, 23],
            "model": "/models/old",
            "timestamp": 100.0,
        }))
        # Live stream says another
        _write_stream_records(stream_path, [
            _rec("a", "core", [4, 8, 12, 16]),
        ])
        from gaia_web.routes.activations import get_atlas
        await get_atlas()
        meta = json.loads((atlas_dir / "meta.json").read_text())
        assert meta["layers"] == [4, 8, 12, 16]
        # Model field is preserved from on-disk (not all bases overwrite)
        assert meta["model"] == "/models/old"
        # Source is annotated
        assert meta["source"] == "auto_from_activation_stream"

    @pytest.mark.asyncio
    async def test_matching_meta_not_rewritten(self, streamed_layers):
        atlas_dir, stream_path = streamed_layers
        meta_payload = {
            "layers": [4, 8, 12, 16],
            "model": "/models/current",
            "timestamp": 100.0,
        }
        (atlas_dir / "meta.json").write_text(json.dumps(meta_payload))
        original_mtime = (atlas_dir / "meta.json").stat().st_mtime
        # Stream matches the on-disk meta
        _write_stream_records(stream_path, [
            _rec("a", "core", [4, 8, 12, 16]),
        ])
        from gaia_web.routes.activations import get_atlas
        await get_atlas()
        # File should NOT have been rewritten
        new_mtime = (atlas_dir / "meta.json").stat().st_mtime
        assert new_mtime == original_mtime

    @pytest.mark.asyncio
    async def test_no_stream_falls_back_to_disk(self, streamed_layers):
        atlas_dir, stream_path = streamed_layers
        (atlas_dir / "meta.json").write_text(json.dumps({
            "layers": {"4": {"features": {"0": "label_a"}}},
            "model": "/models/x",
            "timestamp": 100.0,
        }))
        # No stream file at all — should serve from disk
        from gaia_web.routes.activations import get_atlas
        result = await get_atlas()
        assert "4" in result["layers"]
        assert result["model"] == "/models/x"

    @pytest.mark.asyncio
    async def test_per_layer_label_files_still_merged(self, streamed_layers):
        atlas_dir, stream_path = streamed_layers
        # Stream advertises layers [4, 8]
        _write_stream_records(stream_path, [
            _rec("a", "core", [4, 8]),
        ])
        # Per-layer label file for layer 4
        (atlas_dir / "layer_4_labels.json").write_text(json.dumps({
            "0": "curiosity",
            "1": "skepticism",
        }))
        from gaia_web.routes.activations import get_atlas
        result = await get_atlas()
        assert "4" in result["layers"]
        feats = result["layers"]["4"].get("features", {})
        assert feats.get("0") == "curiosity"
        assert feats.get("1") == "skepticism"
