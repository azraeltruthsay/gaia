#!/usr/bin/env python3
"""The GPU half of her self-improvement loop (dcrs/p3ta, 2026-11-20).

What Claude ran by hand for v3..v6, as one run she can start herself (gaia-study
/pipeline/self-improve, called by gaia-core's self_improve when she is idle):

  meditation   maintenance on, /consciousness/training (study owns the GPU)
  review       blind_review.py: new pairs checked against their pages, answer hidden
  build        the curriculum's build.py: train / held-out split
  augment      augment_curriculum.py: rewordings + questions her notes don't answer
  train        train_notes_adapter.py --middle-ground: the next version, with its evals
  web eval     eval_engineering_web.py: the active adapter and the new one, same cases
  decide       adapter_registry.consider: every gate must pass; a pass waits for
               Azrael's authenticator code (policy "approve")
  restore      back to the gear she was in, maintenance off

Each step's outcome goes to /shared/self_improve/last_run.json; a failed step
stops the run (and still restores her gear).
"""
import argparse
import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ORCH = "http://gaia-orchestrator:6410"
DOCTOR = "http://gaia-doctor:6419"
REPO = Path("/gaia/GAIA_Project")
OUT = Path("/shared/self_improve")


def _post(url: str, body: dict = None, timeout: int = 60) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body or {}).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or b"{}")
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _state() -> str:
    try:
        with urllib.request.urlopen(f"{ORCH}/lifecycle/state", timeout=10) as r:
            return json.loads(r.read()).get("state", "")
    except Exception:
        return ""


def _vram_used() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
        return int(out.split()[0])
    except Exception:
        return 0


def next_name(current: str, subject_slug: str) -> str:
    m = re.match(r"^(.*_v)(\d+)$", current or "")
    if m:
        return f"{m.group(1)}{int(m.group(2)) + 1}"
    return f"{subject_slug}_v1"


def run(cur: str, study: str, slot: str, current: str, report: dict, log) -> dict:
    slug = Path(study).name
    name = next_name(current, slug)
    report.update({"candidate": name, "current": current, "steps": []})
    steps = [
        ("review", [sys.executable, "scripts/blind_review.py", cur, study, "--prime-gpu"]),
        ("build", [sys.executable, f"{cur}/build.py", f"{study}/pairs.jsonl"]),
        ("augment", [sys.executable, "scripts/augment_curriculum.py", cur, study, "--prime-gpu"]),
        ("train", [sys.executable, "scripts/train_notes_adapter.py", "--curriculum", cur, "--study", study,
                   "--name", name, "--middle-ground"]),
        ("web eval", [sys.executable, "scripts/eval_engineering_web.py", "--curriculum", cur, "--study", study,
                      "--adapters", f"{current},{name}" if current else name]),
    ]
    env = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    import os
    for label, cmd in steps:
        t0 = time.time()
        log(f"== {label}: {' '.join(cmd[1:])}")
        p = subprocess.run(cmd, cwd=REPO, env={**os.environ, **env}, capture_output=True, text=True)
        tail = "\n".join(line for line in (p.stdout + p.stderr).splitlines()
                         if "it/s]" not in line and "s/it]" not in line)[-1500:]
        report["steps"].append({"step": label, "ok": p.returncode == 0, "seconds": round(time.time() - t0),
                                "tail": tail})
        log(tail[-600:])
        if p.returncode != 0:
            report["outcome"] = f"stopped: {label} failed"
            return report
    sys.path.insert(0, "/gaia-common")
    from gaia_common.utils.adapter_registry import consider
    decision = consider(slot, name, by="gaia (self_improve)", fallback_current=current)
    report["decision"] = decision
    report["outcome"] = decision["action"]
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curriculum", required=True)
    ap.add_argument("--study", required=True)
    ap.add_argument("--slot", default="engineering")
    ap.add_argument("--current", default="")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    log_path = OUT / "run.log"

    def log(msg):
        with log_path.open("a") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}\n")

    report = {"started": time.time(), "curriculum": a.curriculum}
    before = _state()
    report["gear_before"] = before
    _post(f"{DOCTOR}/maintenance/enter", {"ttl_seconds": 14400, "reason": "self_improve: train and evaluate"})
    _post(f"{ORCH}/consciousness/training")
    for _ in range(60):
        if 0 < _vram_used() < 4500:
            break
        time.sleep(10)
    try:
        run(a.curriculum, a.study, a.slot, a.current, report, log)
    except Exception as e:
        report["outcome"] = f"error: {e}"
    finally:
        _post(f"{DOCTOR}/maintenance/exit")
        ep = {"parked": "park", "focusing": "focusing", "awake": "awake", "sleeping": "sleep",
              "asleep": "sleep", "deep_sleep": "deep-sleep"}.get(before, "park")
        report["restored_via"] = ep
        _post(f"{ORCH}/consciousness/{ep}")
        report["finished"] = time.time()
        (OUT / "last_run.json").write_text(json.dumps(report, indent=1))
        log(f"== done: {report.get('outcome')}")


if __name__ == "__main__":
    main()
