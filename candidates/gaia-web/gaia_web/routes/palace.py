"""Palace report: everything in her memory, for review (GAIA_Project-oxfy).

Read-only by construction: the knowledge graph is opened with mode=ro and the
palace is mounted read-only here. Management comes later, through gaia-mcp, so
every change has one writer and leaves a record.

The portal has no login. Azrael's call (2026-11-05): it is reachable from his
home network and WireGuard, both of which he secures.
"""

import json
import logging
import os
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger("GAIA.Web.Palace")

router = APIRouter(prefix="/api/palace", tags=["palace"])

_RE_SAFE = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-."
MAX_LIMIT = 200


def _root() -> Path:
    return Path(os.getenv("PALACE_ROOT", "/knowledge/mempalace"))


def _kg_path() -> str:
    return os.getenv("PALACE_KG_PATH", "/shared/knowledge_graph/gaia_kg.sqlite3")


def _json_files(directory: Path) -> Iterable[dict]:
    for path in sorted(directory.glob("*.json")) if directory.is_dir() else []:
        try:
            yield json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.warning("Palace report: unreadable %s", path)


def _page(items: list[dict], limit: int, offset: int) -> dict:
    return {"total": len(items), "offset": offset, "limit": limit, "items": items[offset:offset + limit]}


def _matches(q: str, *fields) -> bool:
    return not q or q.lower() in " ".join(str(f or "") for f in fields).lower()


def _people() -> list[dict]:
    return list(_json_files(_root() / "people"))


def _person_of(session_id: str, people: list[dict]) -> dict | None:
    discord = session_id[len("discord_dm_"):] if session_id.startswith("discord_dm_") else None
    for person in people:
        for account in person.get("accounts", []):
            if account.get("id") in (session_id, discord):
                return person
    return None


def _conversation(path: Path) -> list[dict]:
    records = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                records.append(json.loads(line))
            except ValueError:
                continue
    except OSError:
        logger.warning("Palace report: unreadable %s", path)
    return records


def _memories() -> list[dict]:
    out = []
    for path in _root().rglob("*.aaak"):
        try:
            head, _, body = path.read_text(encoding="utf-8").partition("\n")
        except OSError:
            continue
        parts = head.split("|")
        out.append({"id": path.stem, "wing": parts[0] if parts else "", "room": parts[1] if len(parts) > 1 else "",
                    "date": parts[2] if len(parts) > 2 else "", "source": parts[3] if len(parts) > 3 else "",
                    "intent": parts[4] if len(parts) > 4 else "", "text": body,
                    "archived": "archive" in path.parts})
    return sorted(out, key=lambda m: m["date"], reverse=True)


def _kg() -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{_kg_path()}?mode=ro", uri=True, timeout=2)
    conn.row_factory = sqlite3.Row
    return conn


@router.get("/overview")
async def overview() -> dict:
    """How much is in each part of her memory."""
    root = _root()
    counts = {"people": len(_people()),
              "lookups": sum(1 for _ in (root / "knowledge" / "lookups").glob("*.json")),
              "works": sum(1 for _ in (root / "knowledge" / "works").glob("*.json")),
              "memories": sum(1 for _ in root.rglob("*.aaak")),
              "conversations": sum(1 for _ in (root / "conversations").glob("*.jsonl"))}
    try:
        with _kg() as conn:
            counts["facts"] = conn.execute("select count(*) from triples").fetchone()[0]
            counts["entities"] = conn.execute("select count(*) from entities").fetchone()[0]
    except sqlite3.Error as exc:
        logger.warning("Palace report: knowledge graph unavailable: %s", exc)
        counts["facts"] = counts["entities"] = None
    return counts


@router.get("/people")
async def people(q: str = "", limit: int = Query(50, ge=1, le=MAX_LIMIT), offset: int = Query(0, ge=0)) -> dict:
    found = [p for p in _people() if _matches(q, p.get("names"), p.get("relationship"), p.get("facts"))]
    return _page(sorted(found, key=lambda p: (p.get("names") or [""])[0].lower()), limit, offset)


@router.get("/lookups")
async def lookups(q: str = "", limit: int = Query(50, ge=1, le=MAX_LIMIT), offset: int = Query(0, ge=0)) -> dict:
    """What she has looked up or been told, most used first."""
    found = [r for r in _json_files(_root() / "knowledge" / "lookups")
             if _matches(q, r.get("title"), r.get("snippet"), r.get("queries"), r.get("url"))]
    found.sort(key=lambda r: (r.get("times_recalled") or 0, r.get("retrieved_at") or ""), reverse=True)
    return _page(found, limit, offset)


@router.get("/works")
async def works(q: str = "", limit: int = Query(50, ge=1, le=MAX_LIMIT), offset: int = Query(0, ge=0)) -> dict:
    found = [w for w in _json_files(_root() / "knowledge" / "works")
             if _matches(q, w.get("title"), w.get("author"), w.get("text"))]
    return _page(sorted(found, key=lambda w: str(w.get("title") or "").lower()), limit, offset)


@router.get("/memories")
async def memories(q: str = "", wing: str = "", limit: int = Query(50, ge=1, le=MAX_LIMIT),
                   offset: int = Query(0, ge=0)) -> dict:
    found = [m for m in _memories() if (not wing or m["wing"] == wing)
             and _matches(q, m["text"], m["room"], m["source"], m["intent"])]
    return _page(found, limit, offset)


@router.get("/conversations")
async def conversations(q: str = "", limit: int = Query(50, ge=1, le=MAX_LIMIT), offset: int = Query(0, ge=0)) -> dict:
    everyone = _people()
    rows = []
    for path in (_root() / "conversations").glob("*.jsonl"):
        records = _conversation(path)
        person = _person_of(path.stem, everyone)
        name = (person.get("names") or [""])[0] if person else ""
        if not _matches(q, path.stem, name, [r.get("user") for r in records], [r.get("reply") for r in records]):
            continue
        rows.append({"session_id": path.stem, "person": name, "exchanges": len(records),
                     "first": records[0].get("ts", "") if records else "",
                     "last": records[-1].get("ts", "") if records else ""})
    return _page(sorted(rows, key=lambda r: r["last"], reverse=True), limit, offset)


@router.get("/conversations/{session_id}")
async def conversation(session_id: str) -> dict:
    if not session_id or any(c not in _RE_SAFE for c in session_id) or ".." in session_id:
        raise HTTPException(status_code=400, detail="Bad session id")
    path = _root() / "conversations" / f"{session_id}.jsonl"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="No such conversation")
    return {"session_id": session_id, "exchanges": _conversation(path)}


@router.get("/graph")
async def graph(q: str = "", world: str = "", fact_type: str = "", include_ended: bool = False,
                limit: int = Query(50, ge=1, le=MAX_LIMIT), offset: int = Query(0, ge=0)) -> dict:
    """Facts in the knowledge graph, newest first. A fact whose valid_to has
    passed has ended; she keeps it, and so does this view when asked."""
    where, args = [], []
    if q:
        where.append("(subject like ? or predicate like ? or object like ?)")
        args += [f"%{q}%"] * 3
    if world:
        where.append("world = ?")
        args.append(world)
    if fact_type:
        where.append("fact_type = ?")
        args.append(fact_type)
    if not include_ended:
        where.append("valid_to is null")
    clause = f" where {' and '.join(where)}" if where else ""
    try:
        with _kg() as conn:
            total = conn.execute(f"select count(*) from triples{clause}", args).fetchone()[0]
            rows = conn.execute(
                "select id, subject, predicate, object, valid_from, valid_to, confidence, source, extracted_at, "
                f"world, fact_type from triples{clause} order by extracted_at desc limit ? offset ?",
                args + [limit, offset]).fetchall()
            worlds = [dict(r) for r in conn.execute("select id, name, modality, lifecycle from worlds")]
            types = [r[0] for r in conn.execute(
                "select fact_type from triples where fact_type is not null group by fact_type order by count(*) desc")]
    except sqlite3.Error as exc:
        raise HTTPException(status_code=503, detail=f"Knowledge graph unavailable: {exc}")
    return {"total": total, "offset": offset, "limit": limit, "items": [dict(r) for r in rows],
            "worlds": worlds, "fact_types": types}
