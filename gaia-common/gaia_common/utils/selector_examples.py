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
