#!/usr/bin/env python3
"""Turn past bug fixes into self-repair training examples.

Two live trials (c2kz, 24bs; knowledge/curricula/codemind/self_repair/) gave
two worked examples of the loop's three steps. An adapter needs far more;
the history holds them: commits that fixed a bug bead in one or two
gaia-core/gaia-common source files and added tests. Each becomes the same
shape as c2kz.json:

  issue           the bug bead's title and description (as the loop would see
                  an issue). Flagged `issue_names_place` when it already names
                  the changed function: the investigation is given away there.
  place           the function(s) whose lines the fix changed, read from the
                  file as it was before the fix.
  investigation   reconstructed, not observed: a quoted string or a code name
                  from the bead that is in the changed file before the fix ->
                  SEARCH it, READ the hit, READ the function, PLACE it. None
                  when the bead names nothing that leads there (no guessing).
  regression_test the test functions the commit added (not re-run: whether they
                  fail on the parent is unverified, `test_verified: false`).
  patch           the diff of the candidate source files.

    python3 scripts/self_repair_from_history.py               # since 2026-09-01
    python3 scripts/self_repair_from_history.py --since 2026-06-01 --out DIR

Host script: git and a bead export only (`bd list --all --limit 0 --json`).
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(REPO, "knowledge", "curricula", "codemind", "self_repair", "history")
PKGS = ("candidates/gaia-core/gaia_core/", "candidates/gaia-common/gaia_common/")
MAX_SOURCE_FILES = 2
MAX_PATCH_LINES = 400
MAX_TEST_CHARS = 6000
MAX_LEAD_HITS = 3          # a lead that hits more lines of the file than this says little
MAX_PLACE_LINES = 150      # a function CodeMind can be shown whole; bigger places get no investigation


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True, check=False).stdout


def is_test(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return name.startswith("test_") or "/tests/" in path or path.endswith("conftest.py")


def is_source(path: str) -> bool:
    return path.endswith(".py") and path.startswith(PKGS) and not is_test(path)


def cv_rel(path: str) -> str:
    """The path as her investigation's search results show it."""
    return path.split("candidates/", 1)[-1]


# ── beads ────────────────────────────────────────────────────────────────

def load_beads() -> Dict[str, dict]:
    out = subprocess.run(["bd", "list", "--all", "--limit", "0", "--json"], capture_output=True, text=True,
                         cwd=REPO, check=False).stdout
    beads = {}
    for b in json.loads(out or "[]"):
        short = b["id"].split("-", 1)[-1]
        beads[short] = b
    return beads


_BEAD_PATTERNS = (re.compile(r"GAIA_Project-([a-z0-9]+(?:\.\d+)?)"),
                  re.compile(r"^([a-z0-9]{4}(?:\.\d+)?)\s*:", re.M),          # "c2kz: ..."
                  re.compile(r"\(((?:[a-z0-9]{4}(?:\.\d+)?)(?:,\s*[a-z0-9]{4}(?:\.\d+)?)*)\)"))  # "(a4rr)", "(aidw.1, aidw.3)"


def bead_ids(message: str) -> List[str]:
    ids: List[str] = []
    for rx in _BEAD_PATTERNS:
        for m in rx.findall(message):
            ids.extend(x.strip() for x in m.split(","))
    return list(dict.fromkeys(ids))


# ── where the fix is ─────────────────────────────────────────────────────

def old_ranges(parent: str, sha: str, path: str) -> List[Tuple[int, int]]:
    """Lines of the parent's file the fix touched (an insertion counts as the
    line it goes after)."""
    out = []
    for m in re.finditer(r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@", git("diff", "-U0", parent, sha, "--", path), re.M):
        start, count = int(m.group(1)), int(m.group(2) if m.group(2) is not None else 1)
        out.append((max(start, 1), max(start + count - 1, start)))
    return out


def functions(source: str) -> List[Tuple[int, int, str]]:
    """(first line, last line, name) of every def, innermost found by span."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    return [(n.lineno, n.end_lineno or n.lineno, n.name) for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


def enclosing(defs: List[Tuple[int, int, str]], line: int) -> Optional[Tuple[int, int, str]]:
    inside = [d for d in defs if d[0] <= line <= d[1]]
    return min(inside, key=lambda d: d[1] - d[0]) if inside else None


def changed_functions(parent: str, sha: str, path: str) -> Counter:
    source = git("show", f"{parent}:{path}")
    defs = functions(source)
    hits: Counter = Counter()
    for lo, hi in old_ranges(parent, sha, path):
        for line in range(lo, hi + 1):
            d = enclosing(defs, line)
            hits[d[2] if d else "<module>"] += 1
    return hits


# ── the investigation that would have found it ───────────────────────────

_QUOTED = re.compile(r"[\"'`“‘]([^\"'`”’\n]{10,120})[\"'`”’]")
_CODE_NAME = re.compile(r"\b(?:[a-z][a-z0-9]*_[a-z0-9_]+|[A-Z][a-z]+(?:[A-Z][a-z0-9]+)+)\b")


def leads(text: str) -> List[str]:
    """What a bead gives to search for: quoted text first, then code names."""
    quoted = [q.strip() for q in _QUOTED.findall(text or "") if len(q.strip()) >= 10]
    names = [n for n in _CODE_NAME.findall(text or "") if len(n) >= 6]
    return list(dict.fromkeys(quoted + names))


def hits(parent: str, path: str, lead: str) -> List[Tuple[int, str]]:
    out = git("grep", "-n", "-F", "-i", "-e", lead, parent, "--", path)
    found = []
    for line in out.splitlines():
        parts = line.split(":", 3)          # <rev>:<path>:<line>:<text>
        if len(parts) == 4:
            found.append((int(parts[2]), parts[3].strip()))
    return found


def module_stems() -> set:
    return {f.rsplit("/", 1)[-1][:-3] for f in git("ls-files", *PKGS).splitlines() if f.endswith(".py")}


def investigation(parent: str, path: str, place: str, note: str, bead_text: str, stems: set) -> Optional[dict]:
    """SEARCH a lead -> READ its hit -> (READ the place) -> PLACE, only when
    each step follows from the last: the hit is in the place itself, or in a
    function whose code names the place. A module name, or a hit in a comment
    elsewhere, leads nowhere in particular (diq7, ba6o in the first pass)."""
    source = git("show", f"{parent}:{path}")
    defs = functions(source)
    lines = source.splitlines()
    for lead in leads(bead_text):
        if lead == place or lead in stems or lead.lower() in ("gaia_core", "gaia_common"):
            continue            # the answer's own name, or a module's: not an investigation
        found = hits(parent, path, lead)
        if len(found) > MAX_LEAD_HITS:
            continue            # not distinctive: "AgentCore", "except ImportError"
        for line, text in found:
            if text.lstrip().startswith("#"):
                continue
            around = enclosing(defs, line)
            if not around:
                continue
            if around[2] != place:
                body = "\n".join(lines[around[0] - 1:around[1]])
                if not re.search(rf"\b{re.escape(place)}\b", body):
                    continue
            steps = [{"action": f"SEARCH: {lead}", "why": "a phrase or name from the issue, as it appears in code",
                      "result_line": f"{cv_rel(path)}:{line}: {text[:160]}"},
                     {"action": f"READ: {cv_rel(path)}:{line}", "why": f"the hit is in {around[2]}"}]
            if around[2] != place:
                steps.append({"action": f"READ: {place}", "why": f"{around[2]} calls it"})
            steps.append({"action": f"PLACE: {place}", "note": note})
            return {"format": "issue_investigation actions: SEARCH / READ / PLACE + NOTE",
                    "gold_steps": steps, "reconstructed": True, "lead": lead}
    return None


# ── the test and the patch ───────────────────────────────────────────────

def added_tests(parent: str, sha: str, paths: List[str]) -> str:
    """The test functions the commit added, with their file's imports."""
    modules = []
    for path in paths:
        new = git("show", f"{sha}:{path}")
        if not new:
            continue
        try:
            tree = ast.parse(new)
        except SyntaxError:
            continue
        old = git("show", f"{parent}:{path}")
        old_names = {d[2] for d in functions(old)} if old else set()
        lines = new.splitlines()
        tests = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and n.name.startswith("test_") and n.name not in old_names]
        if not tests:
            continue
        first_def = min((n.lineno for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                                                                       ast.ClassDef))), default=1)
        head = "\n".join(lines[:first_def - 1]).rstrip()
        helpers = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                   and not n.name.startswith("test_") and n.name not in old_names]
        body = ["\n".join(lines[(n.decorator_list[0].lineno if n.decorator_list else n.lineno) - 1:n.end_lineno])
                for n in helpers + tests]
        modules.append(f"# {path}\n{head}\n\n\n" + "\n\n\n".join(body))
    text = "\n\n".join(modules)
    return text if len(text) <= MAX_TEST_CHARS else ""


def cause_note(message: str) -> str:
    """The commit's first paragraph after the subject: usually the cause."""
    paras = [p.strip() for p in message.split("\n\n")[1:] if p.strip()]
    paras = [p for p in paras if not p.startswith(("Co-Authored-By", "Claude-Session"))]
    return " ".join(paras[0].split())[:400] if paras else ""


# ── one commit ───────────────────────────────────────────────────────────

def convert(sha: str, beads: Dict[str, dict], stems: set) -> Tuple[Optional[dict], str]:
    message = git("show", "-s", "--format=%B", sha)
    files = [f for f in git("show", "--name-only", "--format=", sha).splitlines() if f]
    sources = [f for f in files if is_source(f)]
    tests = [f for f in files if f.startswith("candidates/") and is_test(f) and f.endswith(".py")]
    if not sources:
        return None, "no candidate source file"
    if len(sources) > MAX_SOURCE_FILES:
        return None, "more than 2 source files"
    if not tests:
        return None, "no test added"
    bugs = [b for b in bead_ids(message) if beads.get(b, {}).get("issue_type") == "bug"]
    if not bugs:
        return None, "no bug bead"
    bead = beads[bugs[0]]
    parent = f"{sha}^"
    patch = git("diff", parent, sha, "--", *sources)
    if patch.count("\n") > MAX_PATCH_LINES:
        return None, "patch too long"
    regression = added_tests(parent, sha, tests)
    if not regression:
        return None, "no new test function"
    places: Counter = Counter()
    for path in sources:
        for name, n in changed_functions(parent, sha, path).items():
            places[(path, name)] += n
    real = [(p, n) for (p, n), _ in places.most_common() if n != "<module>"]
    if not real:
        return None, "only module-level changes"
    path, place = real[0]
    bead_text = f"{bead.get('title', '')}\n{bead.get('description', '')}"
    note = cause_note(message)
    span = next(((a, b) for a, b, n in functions(git("show", f"{parent}:{path}")) if n == place), (0, 0))
    inv = (investigation(parent, path, place, note, bead_text, stems)
           if span[1] - span[0] + 1 <= MAX_PLACE_LINES else None)
    return {
        "id": bugs[0], "bead": bead["id"], "commit": sha[:10], "date": git("show", "-s", "--format=%cs", sha).strip(),
        "source": "history",
        "issue": {"title": bead.get("title", ""), "symptom": bead.get("description", "")},
        "issue_names_place": bool(re.search(rf"\b{re.escape(place)}\b", bead_text)),
        "place": {"file": cv_rel(path), "function": place, "lines": span[1] - span[0] + 1,
                  "also_changed": [f"{cv_rel(p)}:{n}" for p, n in real[1:4]]},
        "investigation": inv,
        "regression_test": {"format": "pytest functions added by the fix", "gold": regression,
                            "test_verified": False},
        "patch": {"format": "unified diff of the candidate source files", "gold_diff": patch},
        "commit_subject": message.splitlines()[0] if message else "",
    }, "ok"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", default="2026-09-01")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    beads = load_beads()
    if not beads:
        sys.exit("no beads: is bd on the PATH with BEADS_DIR set?")
    shas = git("log", f"--since={a.since}", "--format=%H", "--", *[p.rstrip("/") for p in PKGS]).split()
    os.makedirs(a.out, exist_ok=True)
    why: Counter = Counter()
    index = []
    stems = module_stems()
    for sha in shas:
        ex, reason = convert(sha, beads, stems)
        why[reason] += 1
        if not ex:
            continue
        name = f"{ex['commit']}_{ex['id']}.json"
        with open(os.path.join(a.out, name), "w") as f:
            json.dump(ex, f, indent=1, ensure_ascii=False)
        index.append({"file": name, "bead": ex["bead"], "place": f"{ex['place']['file']}:{ex['place']['function']}",
                      "place_lines": ex["place"]["lines"], "investigation": bool(ex["investigation"]),
                      "issue_names_place": ex["issue_names_place"]})
    with open(os.path.join(a.out, "index.jsonl"), "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in index)
    print(f"{len(shas)} commits: " + ", ".join(f"{k} {v}" for k, v in why.most_common()))
    print(f"{len(index)} examples; with a reconstructed investigation: {sum(r['investigation'] for r in index)}; "
          f"issue already names the place: {sum(r['issue_names_place'] for r in index)}; "
          f"place over 150 lines: {sum(r['place_lines'] > 150 for r in index)}")


if __name__ == "__main__":
    main()
