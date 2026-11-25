#!/usr/bin/env python3
"""Held-out replay of her self-repair loop (wcxz.2).

Past bugs, each replayed against the code as it was before its fix, through
today's loop and a chosen mind, scored on what the loop has to get right:

  investigation  she names the function the real fix changed (no place given)
  test           her regression test fails on the pre-fix code and passes on
                 her patched code
  patch          her patch (given the right place, as CodeMind is once a place
                 is named) passes the real fix's own tests, and breaks none of
                 the module's existing tests

plus her investigation on the held-out investigation-practice cases (eval
split of scripts/investigation_practice.py), a larger sample of that step.

    python3 scripts/self_repair_eval.py prepare                       # host: worktrees + manifest
    docker exec -w / -e PYTHONPATH=/gaia/GAIA_Project/candidates/gaia-core:/gaia/GAIA_Project/candidates/gaia-common \\
        gaia-core python3 /gaia/GAIA_Project/scripts/self_repair_eval.py run --label base [--practice 60]
    python3 scripts/self_repair_eval.py score --label base            # host: docker test runs, report

`run --endpoint URL --adapter NAME` evaluates another mind (an adapter later);
the bugs and the scoring stay the same. The held-out bugs are listed in
knowledge/curricula/codemind/self_repair/heldout.json; training must leave
them out.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(os.environ.get("GAIA_REPO_ROOT_EVAL", "/gaia/GAIA_Project"))
SELF_REPAIR = REPO / "knowledge" / "curricula" / "codemind" / "self_repair"
HELDOUT = SELF_REPAIR / "heldout.json"
EVAL_DIR = REPO / ".worktrees" / "eval"
PRACTICE_EVAL = REPO / "knowledge" / "curricula" / "codemind" / "investigation_practice" / "eval.jsonl"
TEST_IMAGE_FROM = "gaia-core-candidate"
REG_TEST_NAME = "test_eval_regression.py"
TIKTOKEN_CACHE = REPO / ".worktrees" / "tiktoken-cache"   # copied from the candidate's /tmp/data-gym-cache


def sh(*cmd, cwd=None, timeout=600) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False)


# ── prepare (host) ───────────────────────────────────────────────────────

def prepare() -> None:
    held = json.loads(HELDOUT.read_text())
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for h in held["bugs"]:
        ex = json.loads((SELF_REPAIR / "history" / h["history"]).read_text())
        commit = sh("git", "-C", str(REPO), "rev-parse", ex["commit"]).stdout.strip()
        wt = EVAL_DIR / ex["commit"]
        if not wt.exists():
            r = sh("git", "-C", str(REPO), "worktree", "add", "--detach", str(wt), f"{commit}^")
            if r.returncode:
                print("worktree failed:", ex["commit"], r.stderr.strip()[:200])
                continue
        files = [f for f in sh("git", "-C", str(REPO), "show", "--name-only", "--format=", commit).stdout.split() if f]
        tests = [f for f in files if f.startswith("candidates/") and f.endswith(".py")
                 and (f.rsplit("/", 1)[-1].startswith("test_") or "/tests/" in f)]
        sources = [f for f in files if f.startswith(("candidates/gaia-core/gaia_core/", "candidates/gaia-common/gaia_common/"))
                   and f.endswith(".py") and f not in tests]
        # Source modules git keeps out (samvega.py, "instance-specific") aren't in
        # a worktree; without them gaia_core.cognition doesn't import. Today's copy.
        ignored = [f for f in sh("git", "-C", str(REPO), "ls-files", "--others", "--ignored", "--exclude-standard",
                                 "candidates/gaia-core/gaia_core", "candidates/gaia-common/gaia_common").stdout.split()
                   if f.endswith(".py") and "__pycache__" not in f]
        for f in ignored:
            if not (wt / f).exists():
                (wt / f).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(REPO / f, wt / f)
        issue = ex["issue"]
        if h.get("issue_from"):            # her ledger issue as filed, not the bead edited after the fix
            issue = json.loads((SELF_REPAIR / h["issue_from"]).read_text())["issue"]
        manifest.append({"id": h["id"], "commit": commit, "worktree": str(wt), "place": ex["place"],
                         "copied_ignored": ignored,
                         "issue": issue, "gold_tests": tests, "sources": sources})
    (EVAL_DIR / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"{len(manifest)} bugs ready in {EVAL_DIR}")


# ── run (inside gaia-core) ───────────────────────────────────────────────

def _chat(endpoint: str, adapter: str, max_tokens: int, temperature: float):
    import urllib.request

    def generate(messages):
        body = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature}
        if adapter:
            body["model"] = adapter
        req = urllib.request.Request(f"{endpoint}/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        out = json.loads(urllib.request.urlopen(req, timeout=600).read())
        return re.sub(r"<think>.*?</think>", "", out["choices"][0]["message"]["content"] or "", flags=re.S).strip()
    return generate


def _issue_in(base: Path, issue: dict) -> dict:
    from gaia_common.utils import issue_ledger as L
    evidence = issue.get("evidence") or [s for s in re.split(r"(?<=[.!?])\s+", issue.get("symptom", "")) if s][:2]
    filed = L.file_issue(issue["title"], issue.get("symptom", ""), evidence, source="user: eval", base=base,
                         suspected_cause=issue.get("suspected_cause", ""))
    return L.issues(base)[filed["id"]]


def _same_place(named: str, function: str) -> bool:
    last = re.split(r"[.:/\s]", (named or "").strip().strip("`'\"()"))
    return function in last or (named or "").strip().rstrip("()").endswith(function)


def run_bug(bug: dict, investigate_gen, code_gen) -> dict:
    from gaia_common.utils import codemind_issues as C, codemind_tests as CT, issue_ledger as L
    from gaia_common.utils.codemind_regression import extract_test_code, validate_test_code
    from gaia_common.utils.codemind_engine import CodeMindEngine
    from gaia_core.cognition import issue_investigation as I
    from gaia_core.cognition.sleep_task_scheduler import SleepTaskScheduler
    from gaia_core.config import Config

    wt = Path(bug["worktree"])
    out = {"id": bug["id"], "gold_place": bug["place"]}
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        issue = _issue_in(base, bug["issue"])
        t0 = time.time()
        inv = I.investigate(issue, investigate_gen, skills_dir=Path("/knowledge/skills"), base=base, repo_root=wt)
        log = [json.loads(x) for x in (base / I.LOG).read_text().splitlines()]
        out["investigation"] = {"place": inv.get("place"), "reason": inv.get("reason"), "steps": inv.get("steps"),
                                "found": _same_place(inv.get("place", ""), bug["place"]["function"]),
                                "seconds": round(time.time() - t0, 1), "transcript": log[-1].get("transcript")}
        # CodeMind, given the right place (as once her issue names one)
        L.note_issue(issue["id"], f"The bug is in {bug['place']['function']}.", base=base,
                     file=f"candidates/{bug['place']['file']}")
        L.note_issue(issue["id"], "Function to fix.", base=base, file=bug["place"]["function"]) \
            if C.resolve(bug["place"]["function"], wt) else None
        dets = C.detections(base=base, repo_root=wt, limit=1)
    if not dets:
        out["codemind"] = {"error": "no detection for the place"}
        return out
    det = dets[0]
    file_path = det["file_path"]
    original = Path(file_path).read_text(encoding="utf-8")
    shown = original
    if len(original) > 30000:
        shown = C.excerpt(original, (det.get("metadata") or {}).get("symbol", "")) or ""
    s = SleepTaskScheduler.__new__(SleepTaskScheduler)
    s.config = Config()
    s.model_pool = None
    s._codemind_propose = lambda prompt: s._clean_code_response(code_gen([{"role": "user", "content": prompt}]) or "") or None
    engine = CodeMindEngine(s.config.constants)
    cfg = dict(s.config.constants.get("CODEMIND", {}))
    t0 = time.time()
    proposed = s._codemind_propose_fix(engine, file_path, det["description"], original, cfg, prompt_content=shown)
    located = CT.locate(file_path, repo_root=wt)
    service, root, module = located if located else ("", None, "")
    test_text, test_ok, why = "", False, ""
    if module:
        resp = s._codemind_propose(engine.build_regression_test_prompt(
            file_path=file_path, module=module, issue_description=det["description"], file_content=shown))
        if resp and not resp.strip().startswith("NO_TEST"):
            test_text = extract_test_code(resp)
            test_ok, why = validate_test_code(test_text, module)
    found = CT.find_tests(file_path, repo_root=wt)
    out["codemind"] = {"file": str(Path(file_path).relative_to(wt)), "service": service, "module": module,
                       "declined": proposed is None, "changed": bool(proposed and proposed != original),
                       "proposed": proposed if proposed and proposed != original else "",
                       "test": test_text if test_ok else "", "test_rejected": "" if test_ok else (why or "none"),
                       "module_tests": (found[1] if found else []), "seconds": round(time.time() - t0, 1)}
    return out


def run_practice(n: int, investigate_gen, seed: int = 11) -> dict:
    from gaia_core.cognition import issue_investigation as I
    rows = [json.loads(x) for x in PRACTICE_EVAL.read_text().splitlines()]
    random.Random(seed).shuffle(rows)
    found = 0
    out = []
    for r in rows[:n]:
        with tempfile.TemporaryDirectory() as tmp:
            issue = _issue_in(Path(tmp), r["issue"])
            res = I.investigate(issue, investigate_gen, skills_dir=Path("/knowledge/skills"), base=Path(tmp),
                                repo_root=REPO)
        ok = _same_place(res.get("place", ""), r["meta"]["function"])
        found += ok
        out.append({"file": r["meta"]["file"], "function": r["meta"]["function"], "named": res.get("place"),
                    "found": ok, "steps": res.get("steps"), "reason": (res.get("reason") or "")[:160]})
        print(f"  practice {len(out)}/{n}: {'found' if ok else 'missed'} {r['meta']['function']} <- {res.get('place')!r}",
              flush=True)
    return {"n": len(out), "found": found, "cases": out}


def run(label: str, endpoint: str, adapter: str, practice: int, only: str) -> None:
    manifest = json.loads((EVAL_DIR / "manifest.json").read_text())
    res_dir = EVAL_DIR / "results" / label
    res_dir.mkdir(parents=True, exist_ok=True)
    # her sleep settings: investigation 400 tokens at 0.2; CodeMind 4096 at 0.1
    investigate_gen = _chat(endpoint, adapter, 400, 0.2)
    code_gen = _chat(endpoint, adapter, 4096, 0.1)
    for bug in manifest:
        if only and bug["id"] != only:
            continue
        print(f"{bug['id']}: ...", flush=True)
        r = run_bug(bug, investigate_gen, code_gen)
        (res_dir / f"{bug['id']}.json").write_text(json.dumps(r, indent=1, ensure_ascii=False))
        inv, cm = r["investigation"], r.get("codemind", {})
        print(f"{bug['id']}: investigation {'FOUND' if inv['found'] else 'missed'} ({inv['place'] or inv['reason'][:60]}); "
              f"patch {'declined' if cm.get('declined') else 'proposed' if cm.get('changed') else 'none'}; "
              f"test {'written' if cm.get('test') else 'rejected: ' + str(cm.get('test_rejected'))[:50]}", flush=True)
    if practice:
        p = run_practice(practice, investigate_gen)
        (res_dir / "practice.json").write_text(json.dumps(p, indent=1))
        print(f"practice: found {p['found']}/{p['n']}")


# ── score (host) ─────────────────────────────────────────────────────────

def _image() -> str:
    return sh("docker", "inspect", TEST_IMAGE_FROM, "--format", "{{.Config.Image}}").stdout.strip()


def _pytest(wt: Path, service: str, paths, image: str) -> dict:
    """Run tests from one service of the worktree in a throwaway container
    (root, no /shared mount: nothing reaches the live system)."""
    if not paths:
        return {"ran": False}
    svc = service if service.startswith("gaia-") else f"gaia-{service}"
    workdir = "/app" if svc == "gaia-core" else "/gaia-common"
    rel = [p.split(f"candidates/{svc}/", 1)[-1] for p in paths]
    cmd = ["docker", "run", "--rm", "--user", "root", "--network", "none",
           "-v", f"{wt}/candidates/gaia-core:/app", "-v", f"{wt}/candidates/gaia-common:/gaia-common",
           "-v", f"{TIKTOKEN_CACHE}:/tmp/data-gym-cache:ro",   # no network: tiktoken's GPT-2 files from here
           "-e", "PYTHONPATH=/app:/gaia-common", "-e", "SHARED_DIR=/tmp/shared", "-w", workdir,
           "--entrypoint", "python", image, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--no-header",
           "-o", "addopts=", *rel]
    r = sh(*cmd, timeout=900)
    tail = (r.stdout + r.stderr).strip().splitlines()
    summary = next((ln.strip("= ") for ln in reversed(tail) if re.search(r"\d+ (passed|failed|error)|no tests ran", ln)), "")
    return {"ran": True, "ok": r.returncode == 0, "summary": summary or f"rc={r.returncode}",
            "tail": "\n".join(tail[-25:])}


def _reset(wt: Path) -> None:
    """Back to the pre-fix commit. `checkout <commit> -- files` stages them, so
    a plain `checkout -- .` restored the staged fix and every condition after
    the control ran on fixed code. Ignored copies (samvega.py) are kept."""
    sh("git", "-C", str(wt), "reset", "-q", "--hard", "HEAD")
    sh("git", "-C", str(wt), "clean", "-fdq", "candidates")


def _by_service(paths):
    groups = {}
    for p in paths:
        svc = p.split("/")[1]
        groups.setdefault(svc, []).append(p)
    return groups


def _run_all(wt, paths, image):
    results = [_pytest(wt, svc, ps, image) for svc, ps in _by_service(paths).items()]
    return {"ran": bool(results), "ok": all(r.get("ok") for r in results) if results else False,
            "summary": "; ".join(r.get("summary", "") for r in results), "tail": "\n".join(r.get("tail", "") for r in results)}


def score(label: str) -> None:
    manifest = {b["id"]: b for b in json.loads((EVAL_DIR / "manifest.json").read_text())}
    res_dir = EVAL_DIR / "results" / label
    image = _image()
    rows = []
    for path in sorted(res_dir.glob("*.json")):
        if path.name in ("practice.json", "score.json"):
            continue
        r = json.loads(path.read_text())
        bug = manifest[r["id"]]
        wt = Path(bug["worktree"])
        cm = r.get("codemind", {})
        row = {"id": r["id"], "found_place": r["investigation"]["found"]}
        _reset(wt)
        # controls: the real fix's tests fail before it and pass with it
        sh("git", "-C", str(wt), "checkout", bug["commit"], "--", *bug["gold_tests"])
        before = _run_all(wt, bug["gold_tests"], image)
        sh("git", "-C", str(wt), "checkout", bug["commit"], "--", *bug["sources"])
        with_fix = _run_all(wt, bug["gold_tests"], image)
        row["control"] = {"gold_fail_before": not before["ok"], "gold_pass_with_fix": with_fix["ok"],
                          "before": before["summary"], "with_fix": with_fix["summary"]}
        _reset(wt)
        reg = wt / "candidates" / f"gaia-{cm.get('service', 'core').replace('gaia-', '')}" / "tests" / REG_TEST_NAME
        if cm.get("test"):
            reg.parent.mkdir(parents=True, exist_ok=True)
            reg.write_text(cm["test"])
            t = _run_all(wt, [str(reg.relative_to(wt))], image)
            row["test_fails_before"] = not t["ok"]
        else:
            row["test_fails_before"] = None
        if cm.get("proposed"):
            (wt / cm["file"]).write_text(cm["proposed"])
            if cm.get("test"):
                row["test_passes_after"] = _run_all(wt, [str(reg.relative_to(wt))], image)["ok"]
            else:
                row["test_passes_after"] = None
            existing = [f"candidates/{cm['service']}/{p}" if not p.startswith("candidates/") else p
                        for p in cm.get("module_tests", []) if p != REG_TEST_NAME]
            row["module_tests_ok"] = _run_all(wt, existing, image)["ok"] if existing else None
            sh("git", "-C", str(wt), "checkout", bug["commit"], "--", *bug["gold_tests"])
            g = _run_all(wt, bug["gold_tests"], image)
            row["patch_passes_gold"] = g["ok"]
            row["patch_gold_summary"] = g["summary"]
        else:
            row.update(test_passes_after=None, module_tests_ok=None, patch_passes_gold=None,
                       patch_gold_summary="declined" if cm.get("declined") else "no change")
        _reset(wt)
        rows.append(row)
        print(json.dumps(row), flush=True)
    valid = [r for r in rows if r["control"]["gold_fail_before"] and r["control"]["gold_pass_with_fix"]]
    practice = json.loads((res_dir / "practice.json").read_text()) if (res_dir / "practice.json").exists() else None

    def frac(xs):
        xs = [x for x in xs if x is not None]
        return f"{sum(xs)}/{len(xs)}" if xs else "-"
    summary = {
        "label": label, "bugs": len(rows), "bugs_with_valid_controls": len(valid),
        "investigation_found_place": frac([r["found_place"] for r in rows]),
        "test_fails_before": frac([r["test_fails_before"] for r in rows]),
        "test_fails_before_and_passes_after": frac([(r["test_fails_before"] and r["test_passes_after"])
                                                    if r["test_fails_before"] is not None else False for r in rows]),
        "patch_passes_gold_tests (valid controls)": frac([bool(r["patch_passes_gold"]) for r in valid]),
        "patch_breaks_no_module_test": frac([r["module_tests_ok"] for r in rows]),
        "practice_investigation": f"{practice['found']}/{practice['n']}" if practice else "-",
    }
    (res_dir / "score.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    r = sub.add_parser("run")
    r.add_argument("--label", required=True)
    r.add_argument("--endpoint", default="http://gaia-prime:7777")
    r.add_argument("--adapter", default="")
    r.add_argument("--practice", type=int, default=0)
    r.add_argument("--only", default="")
    s = sub.add_parser("score")
    s.add_argument("--label", required=True)
    a = ap.parse_args()
    if a.cmd == "prepare":
        prepare()
    elif a.cmd == "run":
        run(a.label, a.endpoint, a.adapter, a.practice, a.only)
    else:
        score(a.label)


if __name__ == "__main__":
    sys.exit(main())
