"""A message during a gear shift waits for the engine instead of being
turned away with 'try again' (2026-11-17: three lost in a row)."""
import asyncio
import json

from gaia_core.main import wait_for_engine


def _run(states, wait_s=120):
    t = [0.0]
    seq = iter(states)

    async def health():
        s = next(seq, "up")
        if s == "error":
            raise ConnectionError
        return {"model_loaded": s == "up", "status": "ok" if s == "up" else "loading"}

    async def sleep(d):
        t[0] += d

    async def go():
        return [x async for x in wait_for_engine(health, wait_s, sleep, clock=lambda: t[0])]

    return asyncio.run(go())


def test_waits_then_goes_on():
    out = _run(["loading"] * 8 + ["up"])
    assert out[-1] is True
    waking = [json.loads(x) for x in out if isinstance(x, str)]
    assert waking and all(w["value"] == "waking" for w in waking)


def test_ready_at_once_says_nothing():
    assert _run(["up"]) == [True]


def test_gives_up_after_the_limit():
    out = _run(["error"] * 1000, wait_s=30)
    assert out[-1] is False and len([x for x in out if isinstance(x, str)]) == 3
