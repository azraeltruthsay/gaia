#!/usr/bin/env python3
"""Investigation practice from her own log calls (wcxz.1).

In both self-repair trials (c2kz, 24bs) her investigation never reached the
code: her first search hit the right log line and she didn't open it, or she
searched the evidence's data ("watt", "Highwayengineering") instead of its
fixed words. Every log call with a literal message in her candidate code is
an exercise in that missing step:

    evidence quotes the line  ->  SEARCH its fixed words  ->  READ the hit
    ->  PLACE the function that writes it

Each exercise is run through her real issue_investigation.investigate() in a
throwaway ledger, with a scripted mind playing the right actions, so the
search results, reads, step counts and the hint are exactly what she would
see. The saved conversation is the training example. About a third first
search the data in the line, get the real "no line contains" reply, and
recover. One or two unrelated log lines go in as decoys. ~15% of source
files are held out (split "eval"), so the adapter can be measured on code it
never practised on.

Runs inside gaia-core with the candidate packages first on the path:

    docker exec -w / -e PYTHONPATH=/gaia/GAIA_Project/candidates/gaia-core:/gaia/GAIA_Project/candidates/gaia-common \\
        gaia-core python3 /gaia/GAIA_Project/scripts/investigation_practice.py [--limit N]

Output (git-ignored, regeneratable): knowledge/curricula/codemind/investigation_practice/{train,eval}.jsonl
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import re
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

REPO = Path("/gaia/GAIA_Project")
OUT = REPO / "knowledge" / "curricula" / "codemind" / "investigation_practice"
ROOTS = (REPO / "candidates" / "gaia-core" / "gaia_core", REPO / "candidates" / "gaia-common" / "gaia_common")
MAX_FUNC_LINES = 150
MIN_LEAD, MAX_LEAD = 12, 60
EVAL_SHARE = 15            # percent of source files held out
RECOVER_SHARE = 0.33

_PLACEHOLDER = re.compile(r"%[-+ #0]*\d*(?:\.\d+)?[rsdifx]")
_LEVELS = {"info": "INFO", "warning": "WARNING", "warn": "WARNING", "error": "ERROR", "exception": "ERROR"}
SAMPLES = ["What is the SI unit of power?", "Which ancient wonder stood at Alexandria?", "web_search",
           "discord_dm_596925786208993283", "gi-9168e4", "core", "prime", "Tell me about the Raven",
           "gaia-mcp", "/models/core-gpu.gguf", "timeout", "palace", "the Lighthouse of Alexandria",
           "unit_of_power", "Highway_engineering", "awake", "focusing", "Write a haiku about frost"]
TITLES = ["Something went wrong around this log line", "A turn went wrong; this line was in my logs",
          "My logs show this when it goes wrong", "This message came up and the reply was wrong",
          "Find what decides this log message"]
SYMPTOMS = ["During a turn that went wrong, my logs had the lines below. I want to find the code that decides it.",
            "A reply came out wrong, and this was logged at the time. Where in my code is it decided?",
            "This log line keeps appearing when the answer is bad. I need the code behind it.",
            "Something misbehaved; the evidence is below."]


def log_calls():
    """(path, function name, call line, level, message, logger name) for every
    log call with a literal message, in functions of at most MAX_FUNC_LINES."""
    for root in ROOTS:
        for path in sorted(root.rglob("*.py")):
            if "tests" in path.parts or path.name.startswith("test_"):
                continue
            try:
                source = path.read_text(encoding="utf-8")
                tree = ast.parse(source)
            except (OSError, SyntaxError):
                continue
            m = re.search(r'getLogger\(\s*["\']([^"\']+)["\']', source)
            logger = m.group(1) if m else "GAIA." + "".join(w.title() for w in path.stem.split("_"))
            for fn in ast.walk(tree):
                if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if (fn.end_lineno or fn.lineno) - fn.lineno + 1 > MAX_FUNC_LINES:
                    continue
                for call in ast.walk(fn):
                    if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                            and call.func.attr in _LEVELS and call.args
                            and isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str)
                            and len(call.args[0].value) >= 20):
                        # the innermost def holding the call
                        inner = [f for f in ast.walk(fn) if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))
                                 and f.lineno <= call.lineno <= (f.end_lineno or f.lineno)]
                        owner = min(inner, key=lambda f: (f.end_lineno or f.lineno) - f.lineno)
                        if owner is fn:
                            yield path, fn.name, call.lineno, _LEVELS[call.func.attr], call.args[0].value, logger


def lead_of(message: str) -> str:
    """The longest fixed stretch of the message, as she should search it."""
    parts = [p.strip(" :;,.-()[]'\"=") for p in _PLACEHOLDER.split(message)]
    best = max(parts, key=len, default="")
    if len(best) > MAX_LEAD:
        best = best[:MAX_LEAD].rsplit(" ", 1)[0]
    return best if len(best) >= MIN_LEAD else ""


def render(message: str, rng: random.Random) -> tuple:
    """The line as her log prints it, and the data put into it."""
    used = []

    def fill(m):
        spec = m.group(0)
        if spec[-1] in "dix":
            v = str(rng.randint(2, 900))
        elif spec[-1] == "f":
            v = f"{rng.uniform(0.1, 99):.2f}"
        else:
            v = rng.choice(SAMPLES)
            if spec[-1] == "r":
                v = repr(v)
        used.append(v.strip("'"))
        return v
    return _PLACEHOLDER.sub(fill, message).replace("%%", "%"), used


def stamp(rng: random.Random) -> str:
    return f"2026-11-{rng.randint(10, 25):02d}T{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d}"


def split_of(rel: str) -> str:
    return "eval" if int(hashlib.sha1(rel.encode()).hexdigest(), 16) % 100 < EVAL_SHARE else "train"


def scripted(actions: List[str], record: dict):
    """A mind that plays `actions` in order and keeps the conversation."""
    queue = list(actions)

    def generate(messages):
        answer = queue.pop(0) if queue else "NOT_FOUND: out of script"
        record["messages"] = [dict(m) for m in messages] + [{"role": "assistant", "content": answer}]
        return answer
    return generate


def one(call, decoys, rng, L, I, C, view) -> Optional[dict]:
    path, func, line, level, message, logger = call
    rel = str(path.relative_to(REPO / "candidates"))
    lead = lead_of(message)
    if not lead:
        return None
    if C.resolve(func, REPO) != (path, func):
        return None                                   # name not unique: PLACE would be refused
    if f"{rel}:{line}:" not in view.search(lead):
        return None                                   # the lead doesn't find its own line (too common)
    text, data = render(message, rng)
    evidence = [f"{stamp(rng)} [gaia-core] {level}:{logger}:{text}"]
    # Decoys are context; the symptom says which line the issue is about, as
    # she would (without it, which of three log lines was "the bug" was a guess).
    for d in decoys:
        if f" {level}:{logger}:" not in d:
            evidence.insert(rng.randrange(len(evidence) + 1), d)
    symptom = rng.choice(SYMPTOMS)
    if len(evidence) > 1:
        symptom += f" The line that matters is the {level} from {logger}."
    actions = []
    wrong = next((d for d in data if len(d) >= 5 and "No line contains" in view.search(d)), None)
    if wrong and rng.random() < RECOVER_SHARE:
        actions.append(f"SEARCH: {wrong}")            # her mistake: the data, not the code's words
    actions += [f"SEARCH: {lead}", f"READ: {rel}:{line}",
                f"PLACE: {func}\nNOTE: {func}() writes this line; what decides it is the code in it before the call."]
    with tempfile.TemporaryDirectory() as base:
        filed = L.file_issue(rng.choice(TITLES), symptom, evidence, source="user: practice",
                             base=Path(base))
        if not filed.get("ok"):
            return None
        issue = L.issues(Path(base))[filed["id"]]
        record: dict = {}
        result = I.investigate(issue, scripted(actions, record), skills_dir=Path("/knowledge/skills"),
                               base=Path(base), repo_root=REPO)
    if result.get("place") != func or not record.get("messages"):
        return None
    return {"messages": record["messages"],
            "meta": {"file": rel, "function": func, "line": line, "lead": lead, "recovers": bool(actions[0] != f"SEARCH: {lead}"),
                     "decoys": len(decoys), "split": split_of(rel)}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=25)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    from gaia_common.utils import codemind_issues as C
    from gaia_common.utils import issue_ledger as L
    from gaia_core.cognition import issue_investigation as I
    rng = random.Random(a.seed)
    calls = list(log_calls())
    rng.shuffle(calls)
    if a.limit:
        calls = calls[:a.limit]
    view = I.CodeView(REPO)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    kept = {"train": 0, "eval": 0}
    skipped = 0
    with open(out / "train.jsonl", "w") as tr, open(out / "eval.jsonl", "w") as ev:
        for n, call in enumerate(calls, 1):
            others = [c for c in rng.sample(calls, min(4, len(calls))) if c[0] != call[0]][:rng.randint(1, 2)]
            decoys = [f"{stamp(rng)} [gaia-core] {c[3]}:{c[5]}:{render(c[4], rng)[0]}" for c in others]
            ex = one(call, decoys, rng, L, I, C, view)
            if not ex:
                skipped += 1
                continue
            kept[ex["meta"]["split"]] += 1
            (ev if ex["meta"]["split"] == "eval" else tr).write(json.dumps(ex, ensure_ascii=False) + "\n")
            if n % 100 == 0:
                print(f"{n}/{len(calls)}: kept {kept}, skipped {skipped}", flush=True)
    print(f"{len(calls)} log calls: kept {kept}, skipped {skipped} -> {out}")


if __name__ == "__main__":
    sys.exit(main())
