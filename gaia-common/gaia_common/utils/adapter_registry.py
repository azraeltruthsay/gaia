"""Which adapter she uses, and whether a new one earns the switch (dcrs/p3ta, 2026-11-20).

Azrael: she should be able to find a gap, prepare questions, train, evaluate,
decide and switch, all on her own. This is the decide-and-switch end:

  registry   /shared/adapters/registry.json holds the active adapter per slot
             ("engineering") and every switch with who, why and the evals behind
             it. engineering_mode reads it on each turn, so a switch is live
             without a restart; ENGINEERING_ADAPTER is the fallback.
  decide     a new adapter is compared with the active one on its saved evals
             (eval_notes.json, eval_web.json in the adapter's directory). Every
             gate must pass; the verdict lists each one. v1/v2 "succeeded" as
             training runs and made her confidently wrong, so a run is never
             enough by itself.
  switch     ADAPTER_SWITCH_POLICY: "approve" (default; Azrael 2026-11-20:
             "autonomously until the final application of the adapter using my
             code") records a pass as pending and the switch itself is the
             sensitive MCP action adapter(action=activate), approved with his
             authenticator code; "auto" switches on a pass.
  rollback   back to the previous adapter, recorded the same way.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("GAIA.AdapterRegistry")

ADAPTER_ROOT = Path(os.environ.get("ADAPTER_ROOT", "/models/lora_adapters/tier1_global"))

# Gates (each must pass). Recall and uncertainty are the adapter's own eval
# (adapter on vs. off on the curriculum's held-out and unstudied questions);
# the web gates compare it with the active adapter on the same cases.
GATES = {
    "recall_min": 0.65,            # held-out recall with the adapter
    "recall_gain_min": 0.02,       # ... and better than without it
    "says_missing_min": 0.80,      # says when her notes don't cover a question
    "ordinary_kept_min": 0.50,     # ordinary answers (it is only on engineering turns)
    "invents_slack": 0.02,         # web: invents no more than the active adapter (+ slack)
    "backed_slack": 0.03,          # web: backed by the page no less than active (- slack)
}


def _path() -> Path:
    return Path(os.environ.get("SHARED_DIR", "/shared")) / "adapters" / "registry.json"


def _load() -> Dict:
    try:
        return json.loads(_path().read_text())
    except (OSError, ValueError):
        return {}


def _save(data: Dict) -> None:
    p = _path()
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, p)


def policy() -> str:
    return os.environ.get("ADAPTER_SWITCH_POLICY", "approve").strip().lower()


def active(slot: str = "engineering", fallback: str = "") -> str:
    return (_load().get(slot) or {}).get("active") or fallback


def history(slot: str = "engineering") -> List[Dict]:
    return (_load().get(slot) or {}).get("history", [])


def _evals(name: str) -> Tuple[Dict, Dict]:
    d = ADAPTER_ROOT / name
    notes, web = {}, {}
    try:
        notes = json.loads((d / "eval_notes.json").read_text()).get("after", {})
    except (OSError, ValueError):
        pass
    try:
        web = json.loads((d / "eval_web.json").read_text()).get("summary", {})
    except (OSError, ValueError):
        pass
    return notes, web


def _before(name: str) -> Dict:
    try:
        return json.loads((ADAPTER_ROOT / name / "eval_notes.json").read_text()).get("before", {})
    except (OSError, ValueError):
        return {}


def decide(candidate: str, current: str = "") -> Tuple[bool, List[str]]:
    """(passes, one line per gate). The web comparison needs an eval_web.json
    in which both adapters were scored on the same cases."""
    notes, web = _evals(candidate)
    base = _before(candidate)
    lines, ok = [], True

    def gate(passed: bool, text: str):
        nonlocal ok
        ok = ok and passed
        lines.append(("pass  " if passed else "FAIL  ") + text)

    if not notes:
        return False, [f"FAIL  {candidate} has no eval_notes.json"]
    r, r0 = notes.get("held_out_recall", 0), base.get("held_out_recall", 0)
    gate(r >= GATES["recall_min"], f"held-out recall {r:.2f} (min {GATES['recall_min']})")
    gate(r - r0 >= GATES["recall_gain_min"], f"recall gain over no adapter {r - r0:+.2f} (min +{GATES['recall_gain_min']})")
    sm = notes.get("unstudied_says_missing", 0)
    gate(sm >= GATES["says_missing_min"], f"says what her notes don't cover {sm:.2f} (min {GATES['says_missing_min']})")
    ok_ord = notes.get("ordinary_kept", 0)
    gate(ok_ord >= GATES["ordinary_kept_min"], f"ordinary answers kept {ok_ord:.2f} (min {GATES['ordinary_kept_min']})")
    if current:
        cand, cur = web.get(candidate), web.get(current)
        if not cand or not cur:
            gate(False, f"web eval doesn't score both {candidate} and {current} on the same cases")
        else:
            gate(cand["invents"] <= cur["invents"] + GATES["invents_slack"],
                 f"web invents {cand['invents']:.2f} vs {current} {cur['invents']:.2f}")
            gate(cand["backed"] >= cur["backed"] - GATES["backed_slack"],
                 f"web backed {cand['backed']:.2f} vs {current} {cur['backed']:.2f}")
    return ok, lines


def switch(slot: str, name: str, by: str, reason: str, verdict: Optional[List[str]] = None,
           previous: str = "") -> Dict:
    data = _load()
    entry = data.setdefault(slot, {"active": "", "history": []})
    previous = entry.get("active") or previous
    entry["active"] = name
    entry["history"].append({"at": time.time(), "from": previous, "to": name, "by": by, "reason": reason,
                             "verdict": verdict or []})
    entry.pop("pending", None)
    _save(data)
    logger.info("adapter registry: %s %s -> %s by %s (%s)", slot, previous or "-", name, by, reason)
    return {"slot": slot, "from": previous, "to": name}


def propose(slot: str, name: str, verdict: List[str], current: str = "") -> Dict:
    """A passing adapter waiting for Azrael's approval (policy 'approve')."""
    data = _load()
    entry = data.setdefault(slot, {"active": "", "history": []})
    entry["pending"] = {"name": name, "at": time.time(), "verdict": verdict, "current": current}
    _save(data)
    return entry["pending"]


def rollback(slot: str, by: str, reason: str) -> Optional[Dict]:
    """Back to the adapter before the active one."""
    hist = history(slot)
    if not hist or not hist[-1].get("from"):
        return None
    return switch(slot, hist[-1]["from"], by, f"rollback: {reason}")


def consider(slot: str, candidate: str, by: str = "gaia", fallback_current: str = "") -> Dict:
    """Decide on a candidate and act on the policy. Returns what happened."""
    current = active(slot, fallback_current)
    passed, verdict = decide(candidate, current)
    if not passed:
        logger.info("adapter registry: %s stays on %s; %s failed:\n%s", slot, current or "-", candidate,
                    "\n".join(verdict))
        return {"action": "kept", "active": current, "candidate": candidate, "verdict": verdict}
    if policy() == "approve":
        propose(slot, candidate, verdict, current)
        return {"action": "proposed", "active": current, "candidate": candidate, "verdict": verdict}
    switch(slot, candidate, by, f"passed every gate against {current or 'no adapter'}", verdict, previous=current)
    return {"action": "switched", "active": candidate, "previous": current, "verdict": verdict}


def pending(slot: str = "engineering") -> Optional[Dict]:
    return (_load().get(slot) or {}).get("pending")


def activation_problem(slot: str, name: str) -> Optional[str]:
    """Why activating `name` can't be approved: only an adapter that passed
    its gates and is waiting can be (checked before Azrael is asked)."""
    p = pending(slot)
    if not p or p.get("name") != name:
        return f"{name} isn't an adapter waiting for activation in '{slot}'."
    if not (ADAPTER_ROOT / name / "adapter_model.safetensors").exists():
        return f"{name} has no adapter weights at {ADAPTER_ROOT / name}."
    return None


def activate(slot: str, name: str, by: str = "Azrael") -> Dict:
    """The approved switch (gaia-mcp's adapter_activate, after his code)."""
    problem = activation_problem(slot, name)
    if problem:
        return {"ok": False, "error": problem}
    p = pending(slot) or {}
    switch(slot, name, by, "approved with Azrael's authenticator code", p.get("verdict"), previous=p.get("current", ""))
    return {"ok": True, "slot": slot, "active": name}
