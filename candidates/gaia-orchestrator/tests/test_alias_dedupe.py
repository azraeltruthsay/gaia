"""nano is probed at Core's own address; when Core's VRAM moved a few MB between
the two probes, the (path, vram) dedupe missed and nano became a second GPU
holder (false single-holder violations, VRAM counted twice)."""

from gaia_common.lifecycle.snapshot import TierLiveStatus
from gaia_orchestrator.lifecycle_machine import LifecycleMachine


def _machine(endpoints, tiers):
    m = LifecycleMachine.__new__(LifecycleMachine)
    m._tier_endpoints = endpoints
    m._aliased_logged = set()

    class _Snap:
        pass
    m._snapshot = _Snap()
    m._snapshot.tiers = dict(tiers)
    return m


def _gpu(vram, path="/models/core-gpu.gguf"):
    return TierLiveStatus(device="gpu", model_loaded=True, model_path=path, vram_mb=vram, healthy=True)


def test_same_endpoint_is_an_alias_even_when_vram_moved():
    ep = "http://gaia-core:8092"
    m = _machine({"core": ep, "nano": ep, "prime": "http://gaia-prime:7777"},
                 {"core": _gpu(12880), "nano": _gpu(12885), "prime": TierLiveStatus(device="cpu", model_loaded=True,
                                                                                     model_path="/models/prime.gguf")})
    m._dedupe_aliased_tiers()
    assert m._snapshot.tiers["core"].vram_mb == 12880
    assert m._snapshot.tiers["nano"].device == "unloaded" and m._snapshot.tiers["nano"].vram_mb == 0
    assert m._snapshot.tiers["prime"].model_loaded


def test_forwarder_with_same_model_and_device_is_an_alias():
    m = _machine({"core": "http://gaia-core:8092", "nano": "http://gaia-nano:8080"},
                 {"core": _gpu(12880), "nano": _gpu(12884)})
    m._dedupe_aliased_tiers()
    assert m._snapshot.tiers["nano"].vram_mb == 0


def test_different_engines_both_kept():
    m = _machine({"core": "http://gaia-core:8092", "prime": "http://gaia-prime:7777"},
                 {"core": _gpu(12880), "prime": _gpu(4600, "/models/prime-nf4")})
    m._dedupe_aliased_tiers()
    assert m._snapshot.tiers["core"].vram_mb == 12880 and m._snapshot.tiers["prime"].vram_mb == 4600
