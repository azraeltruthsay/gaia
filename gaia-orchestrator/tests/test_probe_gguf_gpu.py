"""frlp: the tier probe classifies by device, not backend. A GGUF model run
by gaia_cpp with its layers on the GPU reports backend "cpp", device "cuda"
and is conscious; the same backend on the CPU is subconscious. Before, every
cpp backend read as subconscious, so a GGUF-on-GPU Core (the listening gear,
and gear 1 on llama.cpp) looked like a mismatch on every poll."""

from __future__ import annotations

import pytest

from gaia_orchestrator import consciousness_matrix as cmod
from gaia_orchestrator.consciousness_matrix import ConsciousnessLevel, ConsciousnessMatrix


@pytest.fixture
def matrix(monkeypatch):
    monkeypatch.setattr(
        ConsciousnessMatrix, "_load_config",
        staticmethod(lambda: {
            "inference_endpoints": {"nano": "http://n:0", "core": "http://c:0", "prime": "http://p:0"},
            "gpu_models": {"nano": "/m/n", "core": "/m/c", "prime": "/m/p"},
            "cpu_models": {"nano": "/m/n.gguf", "core": "/m/c.gguf", "prime": "/m/p.gguf"},
        }),
    )
    return ConsciousnessMatrix(lifecycle_machine=None)


def _fake_httpx(monkeypatch, payload):
    class Resp:
        status_code = 200

        def json(self):
            return payload

    class Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, **k):
            return Resp()

    monkeypatch.setattr(cmod.httpx, "AsyncClient", Client)


CPP_HEALTH = {"status": "ok", "engine": "gaia", "managed": True, "model_loaded": True,
              "mode": "active", "backend": "cpp", "has_gpu": False}


@pytest.mark.asyncio
@pytest.mark.parametrize("payload, expected", [
    ({**CPP_HEALTH, "device": "cuda"}, ConsciousnessLevel.CONSCIOUS),
    ({**CPP_HEALTH, "device": "cpu"}, ConsciousnessLevel.SUBCONSCIOUS),
    ({"managed": True, "model_loaded": True, "backend": "engine", "device": "cuda"}, ConsciousnessLevel.CONSCIOUS),
    ({"managed": True, "model_loaded": True, "backend": "engine", "device": "cpu"}, ConsciousnessLevel.SUBCONSCIOUS),
    ({"managed": True, "model_loaded": False, "mode": "standby", "backend": "none"}, ConsciousnessLevel.UNCONSCIOUS),
])
async def test_probe_classifies_by_device(matrix, monkeypatch, payload, expected):
    _fake_httpx(monkeypatch, payload)
    await matrix._probe_tier("core")
    assert matrix._tiers["core"].actual == expected
