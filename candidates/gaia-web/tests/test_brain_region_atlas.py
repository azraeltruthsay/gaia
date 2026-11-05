"""One neuron visual, two minds: identity belongs to both (Azrael, 2026-11-05)."""
import json
from pathlib import Path

ATLAS = Path(__file__).resolve().parents[1] / "static" / "brain_region_atlas.json"


def _regions():
    return {r["name"]: r for r in json.loads(ATLAS.read_text())["regions"]}


def test_identity_is_shared_with_each_models_own_features():
    region = _regions()["Somatosensory"]
    assert region["tiers"] == ["core", "prime"] and "identity_self" in region["domains"]
    assert {f["tier"] for f in region["features"]} == {"core", "prime"}
    assert set(region["layerRanges"]) == {"core", "prime"}


def test_other_regions_keep_one_owner():
    for name, region in _regions().items():
        if name != "Somatosensory":
            assert region["tiers"] == [region["tier"]], name
