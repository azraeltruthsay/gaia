"""Her first reading of a message: what it asks of her, read by both minds (d3px).

Each mind reads the message alone (no system prompt, no history) through its
engine's POST /hidden/read, and a linear probe on its own hidden state says:
  act     talk | answer | make | recite | do | plan
  source  own | memory | live | conversation | web | theirs
  depth   quick | think
Questions about herself are answered from her memory and constitution
(source=memory), not inference in the moment (Azrael, 2026-11-27).

The two readings are compared, as with letter counts. Measured 2026-11-27 on
847 real messages held out by session (scripts/intent_probe_d3px.py):
             Core GPU  Prime CPU  both agree
  act          85%       87%      87% of messages: 92% right
  source       81%       79%      81%: 89%
  depth        94%       96%      95%: 97%
Where they disagree each is near a coin flip, so a disagreement takes the
careful route: both sources are consulted, she thinks first, and a tool is
used only when both minds read "do".

Probes live in /shared/intent_probes/<tier>_<gpu|cpu>.npz, fitted for one
model file (stamped in the probe); a mind on a model or device without a
matching probe doesn't vote. After a bake: re-extract and fit (minutes).
Reads go through the engine's reader context (its own small KV on the same
weights), so they don't wait for a generation: Core 12B on GPU ~40 ms during
one, Prime 8B on CPU ~0.5 s during one (~170 ms idle). Agreement figures
above are from the first fit; on the reader's true last token (Core's
earlier reads were one token early, through the MTP path): act 91% of
messages agree, 91% right; source 86%: 85%; depth 97%: 97%.
"""
import json
import logging
import os
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger("GAIA.Reading")

PROBE_DIR = os.environ.get("INTENT_PROBE_DIR", "/shared/intent_probes")
MINDS = {
    "core": os.environ.get("READING_CORE_ENDPOINT", "http://localhost:8092"),
    "prime": os.environ.get("READING_PRIME_ENDPOINT", "http://gaia-prime:7777"),
}
AXES = ("act", "source", "depth")
_HEALTH_TTL = 30.0
_MAX_CHARS = 1600   # the engine's reader context holds 512 tokens

_lock = threading.Lock()
_probes: Dict[str, Optional[dict]] = {}
_health: Dict[str, tuple] = {}
_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="reading")


def enabled() -> bool:
    return os.environ.get("READING_ENABLED", "1").lower() in ("1", "true", "yes", "on")


def _get_json(url: str, body: Optional[dict] = None, timeout: float = 5.0) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data, {"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _where(mind: str) -> Optional[dict]:
    """The mind's engine as it is now: {tag, model}, cached for a while."""
    now = time.monotonic()
    hit = _health.get(mind)
    if hit and now - hit[0] < _HEALTH_TTL:
        return hit[1]
    try:
        h = _get_json(MINDS[mind] + "/health", timeout=2.0)
        if not h.get("model_loaded") or h.get("backend") != "cpp":
            got = None
        else:
            got = {"tag": f"{mind}_{'gpu' if h.get('has_gpu') else 'cpu'}",
                   "model": os.path.realpath(h.get("model") or "")}
    except Exception:
        got = None
    _health[mind] = (now, got)
    return got


def _probe(tag: str) -> Optional[dict]:
    with _lock:
        if tag not in _probes:
            path = os.path.join(PROBE_DIR, f"{tag}.npz")
            try:
                d = np.load(path)
                _probes[tag] = {k: d[k] for k in d.files}
                logger.info("Reading: probe %s loaded (%s)", tag, str(_probes[tag].get("model", "")))
            except FileNotFoundError:
                _probes[tag] = None
            except Exception:
                logger.warning("Reading: probe %s unreadable", tag, exc_info=True)
                _probes[tag] = None
        return _probes[tag]


def apply_probe(probe: dict, layers: Dict[int, List[float]]) -> dict:
    """{axis: (label, confidence)} from one mind's hidden state."""
    x = np.concatenate([np.asarray(layers[int(n)], dtype=np.float32) for n in probe["layers"]])
    out = {}
    for ax in AXES:
        z = (x - probe[f"{ax}_mean"]) / probe[f"{ax}_scale"]
        logits = probe[f"{ax}_coef"] @ z + probe[f"{ax}_bias"]
        classes = [str(c) for c in probe[f"{ax}_classes"]]
        if len(classes) == 2 and logits.shape[0] == 1:
            p1 = float(1.0 / (1.0 + np.exp(-logits[0])))
            out[ax] = (classes[1], p1) if p1 >= 0.5 else (classes[0], 1.0 - p1)
        else:
            e = np.exp(logits - logits.max())
            p = e / e.sum()
            out[ax] = (classes[int(p.argmax())], float(p.max()))
    return out


def _read_one(mind: str, text: str, timeout: float) -> Optional[dict]:
    where = _where(mind)
    if not where:
        return None
    probe = _probe(where["tag"])
    if probe is None:
        return None
    if str(probe.get("model", "")) and str(probe["model"]) != where["model"]:
        logger.debug("Reading: %s probe fits %s, engine runs %s", where["tag"], probe["model"], where["model"])
        return None
    got = _get_json(MINDS[mind] + "/hidden/read",
                    {"messages": [{"role": "user", "content": text[:_MAX_CHARS]}],
                     "layers": [int(n) for n in probe["layers"]]}, timeout=timeout)
    return {"mind": where["tag"], **apply_probe(probe, {int(k): v for k, v in got["layers"].items()})}


def combine(readings: List[dict]) -> dict:
    """One reading from the minds' readings. Agreement routes; on disagreement
    the careful route: every source either read, think first, and "do" only
    when every mind read "do"."""
    act = [r["act"][0] for r in readings]
    src = [r["source"][0] for r in readings]
    dep = [r["depth"][0] for r in readings]
    # One mind alone is no agreement: the careful route, as on a split.
    agreed = {ax: len(readings) >= 2 and len({r[ax][0] for r in readings}) == 1 for ax in AXES}
    if agreed["act"] or len(set(act)) == 1 and act[0] != "do":
        a = act[0]
    else:
        rest = [x for x in act if x != "do"]
        a = "answer" if "answer" in rest or not rest else rest[0]
    return {
        "act": a,
        "sources": list(dict.fromkeys(src)),
        "depth": dep[0] if agreed["depth"] else "think",
        "agreed": agreed,
        "minds": len(readings),
        "readings": readings,
    }


def read_message(text: str, timeout: float = 3.0) -> Optional[dict]:
    """Both minds' reading of `text`, combined; None when no mind could read it."""
    if not enabled() or not (text or "").strip():
        return None
    t0 = time.monotonic()
    futs = {m: _pool.submit(_read_one, m, text, timeout) for m in MINDS}
    readings = []
    for m, f in futs.items():
        try:
            r = f.result(timeout=timeout + 0.5)
            if r:
                readings.append(r)
        except Exception as e:
            logger.debug("Reading: %s could not read: %s", m, e)
    if not readings:
        return None
    out = combine(readings)
    out["ms"] = round((time.monotonic() - t0) * 1000)
    return out
