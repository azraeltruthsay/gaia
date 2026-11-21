"""
Palace library (zbwe): reference shelves in the memory palace, kept in
sync with their sources — her blueprints, the tools she can call, the
service contracts, her core documents.

A memory room holds AAAK memories she stored. A *synced* room holds
documents mirrored from somewhere else: the source stays the truth, the
room is a readable copy refreshed whenever it is older than the sync
interval (and by the sleep-cycle sweep). She reads them whole, in parts,
with palace(read); they are never compressed, archived or swept.

Config (MEMPALACE):
  "wings": {"library": {"description": ..., "rooms": {"blueprints": ...}}}
  "sync_sources": {"library/blueprints": [{"glob": "/knowledge/blueprints/*.md"}],
                   "library/tools": [{"generator": "domain_tools"}]}
  "sync_interval_seconds": 600

Each synced room keeps a manifest (.library.json) of what it mirrors and
from where. A mirrored doc is pruned only when its source directory is
visible here and the file is gone from it, so a container that lacks a
mount never empties a shelf another container filled.
"""

import difflib
import glob as _glob
import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("GAIA.PalaceLibrary")

MANIFEST = ".library.json"
DEFAULT_SYNC_INTERVAL = 600
MAX_DOC_BYTES = 400_000          # a mirrored doc larger than this is skipped
PART_CHARS = 6000                # palace(read) returns a doc in parts this size


def _title_of(text: str, fallback: str) -> str:
    for line in text.splitlines()[:40]:
        s = line.strip()
        if s.startswith("#"):
            return s.lstrip("#").strip()[:120] or fallback
        m = re.match(r"^(?:service|name|title)\s*:\s*(.+)$", s, re.IGNORECASE)
        if m:
            return m.group(1).strip().strip("'\"")[:120]
    return fallback


def _gen_domain_tools() -> Dict[str, str]:
    """One doc per tool domain, plus an index — from the live catalog, so
    the shelf always lists exactly the actions the runtime accepts."""
    from gaia_common.utils.domain_tools import DOMAIN_ACTIONS, DOMAIN_TOOLS, SENSITIVE_ACTIONS

    docs: Dict[str, str] = {}
    index = ["# Your tools", "",
             "Every tool you can call, and its actions. One page per tool on this shelf.", ""]
    for domain, spec in DOMAIN_TOOLS.items():
        desc = spec.get("description", "")
        index.append(f"- **{domain}** — {desc}")
        lines = [f"# Tool: {domain}", "", desc, ""]
        if spec.get("dynamic"):
            lines += ["Called as fabric(pattern, input): a pattern name and the text to process."]
        else:
            lines.append("## Actions")
            for action in DOMAIN_ACTIONS.get(domain, []):
                adef = (spec.get("actions") or {}).get(action) or {}
                params = adef.get("params") or {}
                ptxt = ", ".join(f"{k}: {v}" for k, v in params.items()) or "no parameters"
                gate = " — needs approval" if (domain, action) in SENSITIVE_ACTIONS else ""
                lines.append(f"- `{domain}(action={action})` — {ptxt}{gate}")
        docs[f"{domain}.md"] = "\n".join(lines) + "\n"
    docs["_index.md"] = "\n".join(index) + "\n"
    return docs


GENERATORS: Dict[str, Callable[[], Dict[str, str]]] = {"domain_tools": _gen_domain_tools}


def readable_blueprint(text: str) -> str:
    """A service blueprint YAML as prose she can read and retell: role,
    purpose, interfaces, dependencies, failure modes. The YAML itself is
    for machines; quoted raw, "what does the doctor do?" got its outbound
    call list (zbwe). Returns the text unchanged if it isn't one."""
    try:
        import yaml
        d = yaml.safe_load(text)
    except Exception:
        return text
    if not isinstance(d, dict) or "id" not in d or not ("interfaces" in d or "intent" in d):
        return text
    intent = d.get("intent") or {}
    rt = d.get("runtime") or {}
    out = [f"# Blueprint: {d['id']}" + (f" — {d['role']}" if d.get("role") else ""), ""]
    if intent.get("purpose"):
        out += [f"## What {d['id']} does", str(intent["purpose"]).strip(), ""]
    if intent.get("cognitive_role"):
        out += ["## Cognitive role", str(intent["cognitive_role"]).strip(), ""]
    facts = [f"{k}: {rt[k]}" for k in ("port", "gpu", "base_image") if rt.get(k) not in (None, "")]
    if d.get("service_status"):
        facts.insert(0, f"status: {d['service_status']}")
    if facts:
        out += ["## Runtime", "; ".join(facts), ""]
    ifaces = d.get("interfaces") or []
    for direction, head in (("inbound", "What it offers"), ("outbound", "What it calls")):
        rows = [i for i in ifaces if isinstance(i, dict) and i.get("direction") == direction]
        if rows:
            out.append(f"## {head}")
            for i in rows:
                tr = i.get("transport") or {}
                where = " ".join(str(x) for x in (tr.get("method"), tr.get("path")) if x)
                out.append(f"- {i.get('id')}{f' ({where})' if where else ''}: {i.get('description') or ''}".rstrip(": "))
            out.append("")
    deps = (d.get("dependencies") or {}).get("services") or []
    if deps:
        out += ["## Depends on"] + [f"- {x.get('id')}: {x.get('role')}" + ("" if x.get("required") else " (optional)")
                                   for x in deps if isinstance(x, dict)] + [""]
    fails = d.get("failure_modes") or []
    if fails:
        out += ["## When things go wrong"] + [f"- {f.get('condition')}: {f.get('response')}"
                                             for f in fails if isinstance(f, dict)] + [""]
    if intent.get("design_decisions"):
        dd = intent["design_decisions"]
        out += ["## Design decisions"] + ([f"- {x}" for x in dd] if isinstance(dd, list) else [str(dd)]) + [""]
    return "\n".join(out)


RENDERERS: Dict[str, Callable[[str], str]] = {"blueprint_yaml": readable_blueprint}


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


class PalaceLibrary:
    """Synced shelves of the palace: sync, list, read, search."""

    def __init__(self, root_dir: Path, wings: dict, sources: dict,
                 interval: float = DEFAULT_SYNC_INTERVAL):
        self._root = Path(root_dir)
        self._wings = wings
        self._sources = {k.strip("/"): v for k, v in (sources or {}).items()}
        self._interval = float(interval)
        self._last_sync = 0.0

    # ── which rooms are synced ─────────────────────────────────────────

    def is_synced(self, wing: str, room: str) -> bool:
        return f"{wing}/{room}" in self._sources

    def synced_rooms(self) -> List[str]:
        return list(self._sources)

    def _room_dir(self, key: str) -> Path:
        return self._root.joinpath(*key.split("/"))

    def _manifest(self, room_dir: Path) -> dict:
        try:
            return json.loads((room_dir / MANIFEST).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    # ── sync ───────────────────────────────────────────────────────────

    def maybe_sync(self) -> Optional[dict]:
        """Sync when the last sync (by this process) is older than the
        interval. Cheap enough to call on every palace read."""
        if time.time() - self._last_sync < self._interval:
            return None
        return self.sync()

    def sync(self) -> dict:
        self._last_sync = time.time()
        report = {"ok": True, "rooms": {}}
        for key, specs in self._sources.items():
            try:
                report["rooms"][key] = self._sync_room(key, specs or [])
            except Exception as e:
                logger.warning("PalaceLibrary: sync of %s failed: %s", key, e)
                report["rooms"][key] = {"error": str(e)}
                report["ok"] = False
        changed = sum(r.get("added", 0) + r.get("updated", 0) + r.get("removed", 0)
                      for r in report["rooms"].values())
        if changed:
            logger.info("PalaceLibrary: synced %d change(s) across %d shelf/shelves", changed, len(self._sources))
        return report

    def _sync_room(self, key: str, specs: list) -> dict:
        room_dir = self._room_dir(key)
        room_dir.mkdir(parents=True, exist_ok=True)
        old = self._manifest(room_dir).get("docs", {})
        new: Dict[str, dict] = {}
        visible_roots: set = set()   # source dirs/generators this process can see
        stats = {"added": 0, "updated": 0, "removed": 0, "unchanged": 0}

        def _put(name: str, text: str, source: str) -> None:
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
            entry = {"source": source, "sha": digest, "title": _title_of(text, name),
                     "bytes": len(text.encode("utf-8"))}
            target = room_dir / name
            if old.get(name, {}).get("sha") == digest and target.exists():
                stats["unchanged"] += 1
            else:
                _write_atomic(target, text)
                stats["updated" if name in old else "added"] += 1
            new[name] = entry

        for spec in specs:
            if "generator" in spec:
                gen = GENERATORS.get(spec["generator"])
                if not gen:
                    logger.warning("PalaceLibrary: unknown generator %r for %s", spec["generator"], key)
                    continue
                visible_roots.add(f"generator:{spec['generator']}")
                for name, text in gen().items():
                    _put(name, text, f"generator:{spec['generator']}")
            elif "glob" in spec:
                pattern = spec["glob"]
                base = Path(pattern).parent
                if not base.is_dir():
                    continue
                visible_roots.add(str(base))
                for src in sorted(_glob.glob(pattern)):
                    p = Path(src)
                    if not p.is_file() or p.name.startswith("."):
                        continue
                    if p.stat().st_size > MAX_DOC_BYTES:
                        logger.info("PalaceLibrary: %s too large for the shelf, skipped", p)
                        continue
                    try:
                        text = p.read_text(encoding="utf-8")
                    except (OSError, UnicodeDecodeError) as e:
                        logger.debug("PalaceLibrary: cannot read %s: %s", p, e)
                        continue
                    render = RENDERERS.get(spec.get("render", ""))
                    _put(p.name, render(text) if render else text, str(p))

        # Keep what this process can't see; prune what vanished from a source it can.
        for name, entry in old.items():
            if name in new:
                continue
            src = entry.get("source", "")
            root = src if src.startswith("generator:") else str(Path(src).parent)
            if root in visible_roots:
                try:
                    (room_dir / name).unlink(missing_ok=True)
                except OSError:
                    pass
                stats["removed"] += 1
            else:
                new[name] = entry
        _write_atomic(room_dir / MANIFEST, json.dumps(
            {"synced_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "docs": new}, indent=1))
        return stats

    # ── list / read / search ───────────────────────────────────────────

    def list_docs(self, wing: str, room: str) -> List[dict]:
        room_dir = self._room_dir(f"{wing}/{room}")
        docs = self._manifest(room_dir).get("docs", {})
        return [{"doc": name, "title": e.get("title", name), "kb": round(e.get("bytes", 0) / 1024, 1)}
                for name, e in sorted(docs.items()) if (room_dir / name).exists()]

    def find_doc(self, doc: str, rooms: List[str]) -> dict:
        """Resolve a doc name in the given synced rooms: exact, then
        case/extension-insensitive, then a unique partial or title match."""
        want = (doc or "").strip().lower()
        if not want:
            return {"ok": False, "error": "Which document? Give doc=<name>."}
        want_stem = re.sub(r"\.(md|ya?ml|txt|json)$", "", want)
        cands = []
        for key in rooms:
            wing, room = key.split("/", 1)
            for d in self.list_docs(wing, room):
                cands.append((key, d))
        norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
        for test in (
            lambda k, d: d["doc"].lower() == want,
            lambda k, d: re.sub(r"\.(md|ya?ml|txt|json)$", "", d["doc"].lower()) == want_stem,
            lambda k, d: norm(d["doc"]).startswith(norm(want_stem)) or norm(want_stem) in norm(d["doc"]),
            lambda k, d: norm(want_stem) in norm(d["title"]),
        ):
            hits = [(k, d) for k, d in cands if test(k, d)]
            if len(hits) == 1:
                return {"ok": True, "room": hits[0][0], "doc": hits[0][1]["doc"]}
            if len(hits) > 1:
                return {"ok": False, "error": f"'{doc}' matches several documents; name one exactly.",
                        "matches": [f"{k}/{d['doc']}" for k, d in hits[:10]]}
        names = [d["doc"] for _, d in cands]
        close = difflib.get_close_matches(want, [n.lower() for n in names], n=5, cutoff=0.4)
        return {"ok": False, "error": f"No document named '{doc}' on these shelves.",
                "did_you_mean": [n for n in names if n.lower() in close]}

    def read(self, room_key: str, doc: str, part: int = 1) -> dict:
        path = self._room_dir(room_key) / doc
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return {"ok": False, "error": f"{room_key}/{doc} is not on the shelf."}
        entry = self._manifest(path.parent).get("docs", {}).get(doc, {})
        chunks = _split(text, PART_CHARS)
        part = max(1, min(int(part or 1), len(chunks)))
        out = {"ok": True, "location": room_key, "doc": doc, "title": entry.get("title", doc),
               "source": entry.get("source", ""), "part": part, "parts": len(chunks),
               "text": chunks[part - 1]}
        if part < len(chunks):
            out["next"] = f"palace(action=read, doc={doc}, part={part + 1})"
        return out

    def search(self, query: str, rooms: List[str], top_k: int = 3) -> List[dict]:
        words = set(re.findall(r"\b\w{3,}\b", (query or "").lower()))
        if not words:
            return []
        hits = []
        for key in rooms:
            room_dir = self._room_dir(key)
            for name, entry in self._manifest(room_dir).get("docs", {}).items():
                try:
                    text = (room_dir / name).read_text(encoding="utf-8")
                except OSError:
                    continue
                low = text.lower()
                title = (entry.get("title") or name).lower()
                score = sum(2 for w in words if w in title or w in name.lower()) + \
                    sum(1 for w in words if w in low)
                if score < max(2, (len(words) + 1) // 2):
                    continue
                paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
                best = max(paras, key=lambda p: sum(w in p.lower() for w in words)) if paras else ""
                hits.append({"location": key, "doc": name, "title": entry.get("title", name),
                             "score": score, "excerpt": best.strip()[:600],
                             "read": f"palace(action=read, doc={name})"})
        hits.sort(key=lambda h: -h["score"])
        return hits[:top_k]


def _split(text: str, size: int) -> List[str]:
    """Split on paragraph boundaries into parts of about `size` chars."""
    if len(text) <= size:
        return [text]
    parts, cur = [], ""
    for para in re.split(r"(\n\s*\n)", text):
        if len(cur) + len(para) > size and cur:
            parts.append(cur)
            cur = ""
        while len(para) > size:
            parts.append(para[:size])
            para = para[size:]
        cur += para
    if cur.strip():
        parts.append(cur)
    return parts or [text]


def asks_about_palace(text: str, config: Optional[dict] = None) -> bool:
    """Whether a message asks about her memory palace: names it, or names
    one of its wings or rooms *as* a wing or room ("your Operational wing",
    "the blueprints shelf", "what rooms are in ..."). "wing" or "room" alone
    is ordinary talk (a hospital wing, a spare room)."""
    low = (text or "").lower()
    if not low:
        return False
    if re.search(r"\b(?:memory|mind)\s+palace\b|\bpalace\b|\bpalatial\b", low):
        return True
    wings = (config or {}).get("wings", {}) or {}
    names = set()
    for w, wdef in wings.items():
        names.add(w.replace("_", " "))
        names.update(r.replace("_", " ") for r in (wdef or {}).get("rooms", {}))
    for n in sorted(names, key=len, reverse=True):
        if re.search(r"\b" + re.escape(n) + r"\s+(?:wing|room|shelf|shelves)\b", low):
            return True
    if re.search(r"\b(?:your|my)\s+(?:library|blueprints?|tool\s*list)\b", low) and \
            re.search(r"\b(?:wing|room|shelf|palace|library)\b", low):
        return True
    return bool(re.search(r"\b(?:wings?|rooms?)\b.*\byour\s+(?:memory|memories)\b"
                          r"|\byour\s+(?:memory|memories)\b.*\b(?:wings?|rooms?)\b", low))
