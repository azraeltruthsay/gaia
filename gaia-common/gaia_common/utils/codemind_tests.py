"""CodeMind's TEST step (a0aw.2): the tests that cover a touched module, run in
the candidate.

Syntax, ruff and diff size say a patch is well-formed, not that it works.
This picks the test files for the module a patch touched (named after it, or
importing it) and runs them in gaia-core-candidate through the orchestrator,
where the tests' writes land on the candidate's volumes (they write to
/shared, the knowledge graph and chat history; run inside gaia-core they
would touch production).

A module no test covers fails: a fix nothing checks is not validated, and
the answer is a regression test, not a pass.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(os.environ.get("GAIA_REPO_ROOT", "/gaia/GAIA_Project"))
# candidates/<service>/ -> its package, as the orchestrator's /candidate/test knows them
SERVICES = {"gaia-core": "gaia_core", "gaia-common": "gaia_common"}
MAX_FILES = 20


def locate(file_path: str, repo_root: Path = REPO_ROOT) -> Optional[Tuple[str, Path, str]]:
    """(service, service root, dotted module) for a candidate .py file, or None."""
    try:
        rel = Path(file_path).resolve().relative_to(repo_root.resolve())
    except ValueError:
        return None
    parts = rel.parts
    if len(parts) < 4 or parts[0] != "candidates" or parts[1] not in SERVICES or rel.suffix != ".py":
        return None
    service, inner = parts[1], parts[2:]
    if inner[0] != SERVICES[service]:
        return None
    module = ".".join(inner)[:-3]
    if module.endswith(".__init__"):
        module = module[: -len(".__init__")]
    return service, repo_root / "candidates" / service, module


def _test_files(root: Path, package: str) -> List[Path]:
    found = []
    for base in (root / "tests", root / package):
        if base.is_dir():
            found += [p for p in base.rglob("test_*.py") if "__pycache__" not in p.parts]
    return sorted(set(found))


def find_tests(file_path: str, repo_root: Path = REPO_ROOT,
               extra: Optional[List[str]] = None) -> Optional[Tuple[str, List[str]]]:
    """(service, test paths relative to the service root) covering file_path:
    tests named after the module, then tests that import it; plus `extra`
    (a regression test written with the fix). None if file_path isn't a
    candidate module of a service the candidate can test."""
    where = locate(file_path, repo_root)
    if where is None:
        return None
    service, root, module = where
    stem = module.rsplit(".", 1)[-1]
    parent, _, _ = module.rpartition(".")
    imports = re.compile(
        rf"(?:\bimport\s+{re.escape(module)}\b|\bfrom\s+{re.escape(module)}\s+import\b"
        rf"|\bfrom\s+{re.escape(parent)}\s+import\s+(?:\(?[\w\s,]*\b){re.escape(stem)}\b)")
    named, importing = [], []
    for path in _test_files(root, SERVICES[service]):
        if f"_{stem}_" in f"_{path.stem.removeprefix('test_')}_":
            named.append(path)
            continue
        try:
            if imports.search(path.read_text(encoding="utf-8", errors="ignore")):
                importing.append(path)
        except OSError:
            continue
    chosen = [str(p.relative_to(root)) for p in named + importing]
    for e in extra or []:
        rel = str(Path(e).resolve().relative_to(root.resolve())) if Path(e).is_absolute() else e
        if rel not in chosen:
            chosen.insert(0, rel)
    return service, chosen[:MAX_FILES]


def run_tests(service: str, paths: List[str], endpoint: Optional[str] = None,
              timeout: int = 300) -> Dict[str, object]:
    """Run them in the candidate via the orchestrator. Never raises."""
    if not paths:
        return {"ok": False, "summary": "no test covers this module; write a regression test", "output": ""}
    endpoint = (endpoint or os.environ.get("ORCHESTRATOR_ENDPOINT", "http://gaia-orchestrator:6410")).rstrip("/")
    body = json.dumps({"service": service, "paths": paths, "timeout": timeout}).encode()
    req = urllib.request.Request(f"{endpoint}/candidate/test", data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout + 30) as resp:
            out = json.loads(resp.read().decode())
    except Exception as e:  # noqa: BLE001 - an unreachable runner is a failed test step
        return {"ok": False, "summary": f"candidate tests could not run: {str(e)[:200]}", "output": ""}
    return {"ok": bool(out.get("ok")), "summary": str(out.get("summary", "")),
            "output": str(out.get("output", ""))}


def run_test_step(file_path: str, endpoint: Optional[str] = None, extra: Optional[List[str]] = None,
              timeout: int = 300) -> Dict[str, object]:
    """Find and run the tests for file_path: {ok, service, paths, summary, output}."""
    found = find_tests(file_path, extra=extra)
    if found is None:
        return {"ok": False, "service": "", "paths": [], "output": "",
                "summary": f"no candidate test target for {file_path}"}
    service, paths = found
    return {"service": service, "paths": paths, **run_tests(service, paths, endpoint, timeout)}
