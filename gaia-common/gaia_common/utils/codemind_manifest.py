"""CodeMind restart-manifest builder (nfi3) — the last mile of the loop.

detect -> patch (s4r2) -> validate -> [THIS] manifest -> promote ->
deadman-supervised deploy (kmcb).

Maps CodeMind's applied candidate files to logical services and renders
a restart manifest via the s4r2 `restart_manifest` scaffold, extended
with `promote_files` so the doctor promotes candidates->prod before the
restart. Pure functions — the sleep scheduler does the file I/O.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

# Deployable logical services (must stay in step with the doctor's
# _SELF_DEPLOY_SERVICES and the orchestrator's _PROD_SRC_DIRS).
# "doctor" is intentionally absent: the supervisor never supervises
# its own restart, and CodeMind must not try.
_DIR_TO_SERVICE = {
    "gaia-core": "core",
    "gaia-web": "web",
    "gaia-mcp": "mcp",
    "gaia-study": "study",
    "gaia-audio": "audio",
    "gaia-orchestrator": "orchestrator",
    "gaia-common": "common",
}

_CAND_RE = re.compile(r"^candidates/([^/]+)/")


def service_for_candidate_file(path: str) -> Optional[str]:
    """candidates/gaia-study/... -> 'study'; None for anything else."""
    m = _CAND_RE.match(path.replace("\\", "/").lstrip("./"))
    if not m:
        return None
    return _DIR_TO_SERVICE.get(m.group(1))


def build_manifest_for_files(
    applied_files: List[str],
    requested_by: str = "codemind",
    bead: str = "",
    evidence: Optional[List[str]] = None,
) -> Tuple[Optional[Dict], Optional[str]]:
    """Build a restart-manifest dict for CodeMind's applied candidate files.

    Returns (manifest, error). Files that don't map to a deployable
    service (non-candidate paths, unknown dirs, doctor) abort the build —
    a partially-deployable manifest would desync the trees.
    """
    if not applied_files:
        return None, "no applied files"

    services: List[str] = []
    normalized: List[str] = []
    for f in applied_files:
        norm = f.replace("\\", "/").lstrip("./")
        svc = service_for_candidate_file(norm)
        if svc is None:
            return None, f"not a deployable candidate file: {f!r}"
        if svc not in services:
            services.append(svc)
        normalized.append(norm)

    # "common" is a library: it restarts nothing by itself, so a
    # common-only manifest must ride with a service that consumes it.
    if services == ["common"]:
        services = ["common", "core"]

    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y-%m-%dT%H-%M-%S")
    manifest_id = f"{stamp}_codemind_{'_'.join(s for s in services if s != 'common')}"

    # Dogfood the s4r2 scaffold for the base document, then extend with
    # the phase-2 fields (multi-service + promote_files).
    def _jsafe(s: str) -> str:
        # Free text lands inside JSON string literals in the template —
        # escape it so a stray quote in test evidence can't break render.
        return json.dumps(str(s))[1:-1]

    try:
        from gaia_common.utils.scaffold import render
        base = json.loads(render("restart_manifest", {
            "manifest_id": manifest_id,
            "requested_by": _jsafe(requested_by),
            "bead": _jsafe(bead),
            "service": services[0],
            "change_summary": _jsafe(f"CodeMind autonomous fix ({len(normalized)} file(s))"),
            "tests_run": _jsafe("; ".join(evidence or ["validate_full + validate_diff_safety passed"])),
            "created_at": now.isoformat(),
        }))
    except Exception as exc:
        return None, f"scaffold render failed: {exc}"

    base["services"] = services
    base["promote_files"] = normalized
    return base, None
