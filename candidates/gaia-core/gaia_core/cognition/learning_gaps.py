"""What she didn't know, kept for her own study (dcrs/p3ta, 2026-11-20).

Each time code shows a gap in what she knows, it is noted here: the question,
how it showed, and the conversation. self_improve turns enough of them into
syllabus topics she studies, and possibly a training run.

  corrected    her look back found (and code confirmed) she was wrong
  not_found    she promised to look it up and found no trusted source
  disagreed    on an engineering turn, her two minds disagreed from memory
  unsure       calculations of a worked problem wouldn't agree
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, List


def _path() -> Path:
    return Path(os.environ.get("SHARED_DIR", "/shared")) / "learning" / "gaps.jsonl"


def note(question: str, kind: str, session_id: str = "", detail: str = "") -> None:
    question = (question or "").strip()
    if len(question) < 8:
        return
    try:
        p = _path()
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), "question": question[:500], "kind": kind,
                                "session_id": session_id, "detail": detail[:500]}, ensure_ascii=False) + "\n")
    except OSError:
        pass


def since(ts: float) -> List[Dict]:
    try:
        rows = [json.loads(line) for line in _path().read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, ValueError):
        return []
    seen, out = set(), []
    for r in rows:
        if r.get("ts", 0) <= ts:
            continue
        key = r["question"].lower()[:120]
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out
