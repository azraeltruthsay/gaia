"""She finds her way around a shared Google Drive folder (2026-11-21).

Andy shared a reference folder and asked her to open "Blogett" in the Welding
folder. She listed the folder, then looked for "Welding" on her own disk; asked
again, she used her browser tool, which shows names without links. Code now:
when a message asks to open / look in / find something and a Drive folder link
is in the conversation, it walks that folder (and its subfolders), finds the
item whose name matches what was asked (misspellings allowed), and opens it:
a folder lists its items, a document opens with its page count and how to
search inside it (web fetch with find=).
"""
from __future__ import annotations

import difflib
import json
import logging
import os
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("GAIA.DriveNav")

_FOLDER_RE = re.compile(r"https?://drive\.google\.com/drive/(?:u/\d+/)?folders/[A-Za-z0-9_-]{10,}")
_ASK_RE = re.compile(r"\b(?:open|look\s+(?:in|into|at|for|through)|go\s+(?:in|into|to)|get\s+(?:in|into)|find|read|"
                     r"pull\s+up|check|search|dive\s+into|navigate)\b", re.IGNORECASE)
_STOP = set("""the a an in into to for of and or please can you could would look open go get find read check search
folder folders file files document documents doc pdf drive google link that this there those them it its from
gaia your my me we us see if inside within through at pull up dive navigate""".split())
MAX_FOLDERS = 25
_IN_DOC_RE = re.compile(r"\b(?:find|look\s+(?:for|up)|search|go\s+to|turn\s+to|show\s+me|where(?:'s| is)|page|problem|"
                        r"example|figure|fig\.?|table|section|chapter)\b", re.IGNORECASE)


def _state_path(session_id: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", session_id or "none")
    return Path(os.environ.get("SHARED_DIR", "/shared")) / "drive_nav" / f"{safe}.json"


def _open_doc(session_id: str) -> Optional[Dict]:
    try:
        return json.loads(_state_path(session_id).read_text())
    except (OSError, ValueError):
        return None


def _remember_doc(session_id: str, name: str, url: str) -> None:
    p = _state_path(session_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"name": name, "url": url}))


def find_in_open_doc(user_input: str, session_id: str, call) -> Optional[str]:
    """Look for something in the document opened in this conversation."""
    doc = _open_doc(session_id)
    if not doc or not _IN_DOC_RE.search(user_input or ""):
        return None
    words = [w for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9,.'-]*", re.sub(r"^[^:\n]{1,64}:\s", "", user_input, count=1))
             if w.lower() not in _STOP and w.lower() not in {"page", "problem", "where", "show", "turn", "it", "is"}]
    if not words:
        return None
    r = _fetch(call, doc["url"], find=" ".join(words[:6]))
    text = r.get("content") or r.get("error") or ""
    pages = re.findall(r"--- page (\d+) ---\n(.*?)(?=\n\n--- page |\Z)", text, re.S)
    if not pages:
        return f"I searched **{doc['name']}** for that and didn't find it. Scanned text misspells words; try a number or a distinctive phrase from it."
    num, body = pages[0]
    body = re.sub(r"[ \t]{3,}", "   ", body.strip())[:1800]
    more = f" (it also comes up on page{'s' if len(pages) > 2 else ''} {', '.join(p for p, _ in pages[1:])})" if len(pages) > 1 else ""
    return f"From **{doc['name']}**, page {num}{more}:\n\n```\n{body}\n```"


def _fetch(call, url: str, **extra) -> Dict:
    r = call("web", {"action": "fetch", "url": url, **extra}, timeout=300) or {}
    r = r.get("response", r) if isinstance(r, dict) else {}
    return r.get("result", r) if isinstance(r, dict) else {}


def folder_in(history: List[Dict], user_input: str) -> Optional[str]:
    for text in [user_input] + [m.get("content") or "" for m in reversed(history[-30:])]:
        m = _FOLDER_RE.search(text)
        if m:
            return m.group(0)
    return None


def wanted_words(text: str) -> List[str]:
    body = re.sub(r"^[^:\n]{1,64}:\s", "", text or "", count=1)
    body = re.sub(r"https?://\S+", " ", body)      # the whole link, query string and all
    return [w for w in re.findall(r"[A-Za-z][A-Za-z0-9'-]{2,}", body) if w.lower() not in _STOP]


def _score(name: str, words: List[str]) -> float:
    toks = [t for t in re.findall(r"[A-Za-z][A-Za-z0-9'-]{2,}", name.lower())]
    if not toks or not words:
        return 0.0
    total = 0.0
    for w in words:
        best = max(difflib.SequenceMatcher(None, w.lower(), t).ratio() for t in toks)
        total += best if best >= 0.75 else 0.0
    return total / len(words)


def walk(root: str, call) -> List[Tuple[str, str, str, str]]:
    """(name, url, kind, parent) for every item under `root`, breadth first."""
    items, queue, seen = [], [(root, "")], set()
    while queue and len(seen) < MAX_FOLDERS:
        url, parent = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)
        listing = _fetch(call, url)
        for it in listing.get("items") or []:
            items.append((it["name"], it["url"], it["kind"], parent or listing.get("title", "")))
            if it["kind"] == "folder":
                queue.append((it["url"], it["name"]))
    return items


def handle(user_input: str, history: List[Dict], call=None, session_id: str = "") -> Optional[str]:
    """Her reply when asked to open something in a shared Drive folder, else None."""
    if call is None:
        from gaia_core.utils.mcp_client import call_jsonrpc as call
    if not _ASK_RE.search(user_input or "") and not _IN_DOC_RE.search(user_input or ""):
        return None
    root = folder_in(history, user_input)
    words = wanted_words(user_input)
    if not root:
        return find_in_open_doc(user_input, session_id, call) if session_id else None
    items = walk(root, call)
    if not items:
        return None
    if not words:
        return None
    scored = sorted(((_score(n, words), n, u, k, p) for n, u, k, p in items), reverse=True)
    best = scored[0]
    if best[0] < 0.5 and session_id and _open_doc(session_id):
        return find_in_open_doc(user_input, session_id, call)     # not a new item: look inside the open one
    if best[0] < 0.5:
        names = ", ".join(n for n, _, _, _ in items[:12])
        return f"I looked through the shared folder but didn't find anything matching that. What's in it: {names}."
    _, name, url, kind, parent = best
    if kind == "folder":
        inside = [n for n, _, _, p in items if p == name]
        listed = "; ".join(inside) if inside else "nothing I can see"
        return f"In **{name}** there's: {listed}."
    doc = _fetch(call, url)
    if not doc.get("ok", True) or not doc.get("content"):
        return f"I found **{name}** (in {parent}), but I couldn't open it: {doc.get('error') or 'no text came back'}."
    if session_id:
        _remember_doc(session_id, name, url)
    pages = re.search(r":\s*(\d+) pages", doc.get("content", ""))
    size = f"{pages.group(1)} pages" if pages else f"{doc.get('word_count', 0):,} words"
    return (f"I found **{name}** in {parent} and opened it: {size}. It's a long document, so tell me what to look "
            f"for, like a problem with a detail from it, a figure, or a heading, and I'll go to that page.")
