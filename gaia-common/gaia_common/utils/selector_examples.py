"""Selector examples: questions and the call that answers them (aidw.7,
read by A2 / aidw.2).

Each record is a lesson in looking: a question she answered without a
lookup, and the read-only call that showed the answer (from a Samvega
finding confirmed against it), or a turn where a lookup grounded the
answer. A2 will put the nearest few in front of her tool selector.

/shared/selector_examples.jsonl, one JSON object per line:
  {"ts", "question", "call", "tool", "params", "source", "ref"}
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import List, Optional

_CALL_RE = re.compile(r"^\s*(\w+)\((.*)\)\s*$", re.DOTALL)


def path() -> Path:
    return Path(os.environ.get("SELECTOR_EXAMPLES", os.path.join(os.environ.get("SHARED_DIR", "/shared"),
                                                                 "selector_examples.jsonl")))


def parse_call(call: str) -> Optional[dict]:
    """'palace(action=navigate, wing=operational)' -> {'tool': 'palace', 'params': {...}}."""
    m = _CALL_RE.match(call or "")
    if not m:
        return None
    params = {}
    for part in re.split(r",\s*(?=\w+=)", m.group(2).strip()):
        if "=" in part:
            k, v = part.split("=", 1)
            params[k.strip()] = v.strip()
    return {"tool": m.group(1), "params": params}


def read(limit: int = 0) -> List[dict]:
    try:
        rows = [json.loads(line) for line in path().read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, ValueError):
        return []
    return rows[-limit:] if limit else rows


def add(question: str, call: str, source: str, ref: str = "") -> bool:
    """Record an example; False if it is a duplicate or unusable."""
    question = re.sub(r"^[^:\n]{1,64}:\s", "", (question or "").strip(), count=1)  # "Azrael: …"
    parsed = parse_call(call)
    if not question or not parsed:
        return False
    key = (question.lower(), call.strip())
    if any((r.get("question", "").lower(), r.get("call", "")) == key for r in read()):
        return False
    p = path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": time.time(), "question": question, "call": call.strip(), **parsed,
                            "source": source, "ref": ref}, ensure_ascii=False) + "\n")
    return True


# ── Retrieval (aidw.2): nearest examples by meaning, not by keywords ──────

MIN_SIMILARITY = 0.40  # measured on MiniLM 2026-11-22: unrelated 0.08-0.28, same kind 0.38-0.43, paraphrase 0.66+
SAMVEGA_BONUS = 0.05   # a call that corrected a confirmed mistake outranks a routine one
_cache = {"mtime": None, "rows": [], "vecs": None, "model": None}


def call_label(tool: str, params: dict) -> str:
    return f"{tool}({', '.join(f'{k}={v}' for k, v in (params or {}).items())})"


def _vectors(embed, texts):
    import numpy as np
    v = embed.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(v, dtype="float32")


def nearest(question: str, embed, k: int = 3, min_sim: float = MIN_SIMILARITY) -> List[dict]:
    """The k examples whose questions mean most nearly the same as `question`
    (cosine similarity of sentence embeddings), above `min_sim`. One per call."""
    question = re.sub(r"^[^:\n]{1,64}:\s", "", (question or "").strip(), count=1)
    if not question or embed is None:
        return []
    try:
        p = path()
        mtime = p.stat().st_mtime if p.exists() else None
        if mtime is None:
            return []
        if _cache["mtime"] != mtime or _cache["model"] is not embed:
            rows = read()
            _cache.update(mtime=mtime, rows=rows, model=embed,
                          vecs=_vectors(embed, [r["question"] for r in rows]) if rows else None)
        rows, vecs = _cache["rows"], _cache["vecs"]
        if not rows or vecs is None:
            return []
        q = _vectors(embed, [question])[0]
        scored = []
        for r, s in zip(rows, (vecs @ q).tolist()):
            s += SAMVEGA_BONUS if r.get("source") == "samvega" else 0.0
            if s >= min_sim:
                scored.append((s, r))
        scored.sort(key=lambda x: -x[0])
        out, seen = [], set()
        for s, r in scored:
            if r["call"] in seen:
                continue
            seen.add(r["call"])
            out.append({**r, "similarity": round(s, 3)})
            if len(out) >= k:
                break
        return out
    except Exception:
        import logging
        logging.getLogger("GAIA.SelectorExamples").debug("nearest failed", exc_info=True)
        return []


def prompt_block(examples: List[dict]) -> str:
    """The examples as the selector reads them."""
    if not examples:
        return ""
    lines = [f'- "{e["question"][:200]}" -> {e["call"]}' for e in examples]
    return ("EARLIER QUESTIONS LIKE THIS, AND THE CALL THAT ANSWERED THEM (from your own history; "
            "use one if this request asks the same kind of thing):\n" + "\n".join(lines) + "\n")
