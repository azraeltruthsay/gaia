"""Reading a file someone sends her (2026-11-19).

Before this, an attached file's text never reached her turn: gaia-web cut it
to 2,000 characters and gaia-core only filed that into a knowledge base. Now
the file is part of what she reads:

  - A file that fits (LONG_CHARS) is in her turn whole, like pasted text.
  - A long one: her turn has its opening and knows she is still reading. What
    she says meanwhile is hers -- a first impression, or just that she's
    reading it. In the background her CFR store reads the rest (sections and
    summaries, whichever mind is loaded). When it's done she looks at what
    she was asked, the summaries and the sections that matter for it, and
    asks herself, in her inner voice, whether she has anything to add;
    otherwise nothing is posted. The file stays in her CFR store for cfr_focus later.

Her model proposes, code checks: section numbers she names are kept only if
they exist; whether she follows up is asked in her inner voice
(inner_monologue), and only a real SAY line is posted.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import urllib.request
from pathlib import Path
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("GAIA.LongRead")

LONG_CHARS = 6000
OPENING_CHARS = 3500
MAX_FOCUS = 2
WEB_URL = os.environ.get("GAIA_WEB_URL", "http://gaia-web:6414")
SOURCES = Path(os.environ.get("SHARED_DIR", "/shared")) / "gaia_state" / "cfr" / "sources"

PICK_SYSTEM = (
    "You are GAIA. You finished reading a file someone sent you. Below is what "
    "they said when they sent it and a summary of each section. Which sections "
    "do you need to look at closely to answer them? Reply with up to {n} section "
    "numbers, or NONE.")

def attached_text(value: Dict) -> str:
    """The file's extracted text: all of it when gaia-web saved it, else the preview."""
    path = (value or {}).get("text_path")
    if path:
        try:
            return Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            logger.info("long_read: %s unreadable, using the preview", path)
    return (value or {}).get("text_preview", "") or ""


def opening(text: str, most: int = OPENING_CHARS) -> str:
    """The start of a long text, cut at a paragraph (or line) break."""
    if len(text) <= most:
        return text
    head = text[:most]
    cut = max(head.rfind("\n\n"), head.rfind("\n"))
    return head[:cut] if cut > most // 2 else head


def reading_block(name: str, text: str) -> str:
    """What her turn carries of the file."""
    if len(text) <= LONG_CHARS:
        return f"\n\n[Attached file: {name}]\n{text}\n[End of {name}]"
    pages = max(2, round(len(text) / 3000))
    return (f"\n\n[Attached file: {name}, about {pages} pages. You have read its beginning, "
            f"below; you are reading the rest now and can come back to it when you've "
            f"finished. Meanwhile you can say what you make of it so far, or just that "
            f"you're reading it.]\n{opening(text)}\n[...the rest of {name} is still being read]")


def doc_id(text: str) -> str:
    return "att_" + hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:12]


def _sections(text: str, cfr) -> List[Dict]:
    did = doc_id(text)
    SOURCES.mkdir(parents=True, exist_ok=True)
    path = SOURCES / f"{did}.txt"
    path.write_text(text, encoding="utf-8")
    res = cfr.ingest(str(path), doc_id=did)
    if not res.get("ok"):
        raise RuntimeError(f"CFR ingest: {res.get('error')}")
    doc = cfr._load_tree(did)
    return [{"summary": sec.summary or "", "text": sec.full_text} for sec in doc.sections]


def pick_sections(asked: str, sections: List[Dict], generate: Callable[[List[dict]], str]) -> List[int]:
    summaries = "\n".join(f"{i}. {s['summary'][:400]}" for i, s in enumerate(sections, 1))
    answer = generate([{"role": "system", "content": PICK_SYSTEM.format(n=MAX_FOCUS)},
                       {"role": "user", "content": f"They said: {asked}\n\nSections:\n{summaries}"}]) or ""
    if answer.strip().upper().startswith("NONE"):
        return []
    picks = []
    for n in re.findall(r"\b(\d{1,3})\b", answer):
        i = int(n) - 1
        if 0 <= i < len(sections) and i not in picks:
            picks.append(i)
    return picks[:MAX_FOCUS]


def follow_up(name: str, asked: str, said: str, sections: List[Dict],
              generate: Callable[[List[dict]], str], session_id: str = "") -> Optional[str]:
    """What she adds once she's read it all, or None. Whether to speak is her
    own call, asked in her inner voice (inner_monologue): a model seldom
    volunteers to speak again, and the question must not read as a person's."""
    from gaia_core.cognition.inner_monologue import ask_myself
    picks = pick_sections(asked, sections, generate)
    m = re.match(r"^([^:\n]{1,64}):\s", asked or "")
    who, words = (m.group(1), asked[m.end():]) if m else ("someone", asked or "")
    summaries = "\n".join(f"Section {i}: {s['summary'][:500]}" for i, s in enumerate(sections, 1))
    close = "\n\n".join(f"Section {i + 1}, which I looked at closely:\n{sections[i]['text']}"
                         for i in picks)
    thought = (
        f"I've finished reading {name}, which {who} sent me. When it arrived I had only read "
        f"the beginning, and I told them: \"{said[:300]}\". When they sent it they said: "
        f"\"{words[:300]}\"\n\nWhat's in it, section by section:\n{summaries}\n\n{close}\n\n"
        f"Now that I've read all of it, do I have anything worth adding, like an answer I "
        f"couldn't give before, or something notable in it? If I do, I'll say it to {who}, "
        f"using only what the file says. If not, I'll leave it.")
    return ask_myself(thought, generate, about=f"finished reading {name}", session_id=session_id)["say"]


def push(session_id: str, text: str) -> bool:
    data = json.dumps({"text": text, "source": "long_read"}).encode()
    try:
        req = urllib.request.Request(f"{WEB_URL}/api/conversations/{session_id}/push", data=data,
                                     method="POST", headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return bool(json.loads(r.read() or b"{}").get("delivered"))
    except Exception as e:
        logger.warning("long_read: push to %s failed: %s", session_id, e)
        return False


def finish_reading(session_id: str, name: str, text: str, asked: str, said: Callable[[], str],
                   generate: Callable[[List[dict]], str], record: Callable[[str], None],
                   cfr=None, send: Callable[[str, str], bool] = push) -> Optional[str]:
    """Read the whole file, then post what she adds (if anything). `said` gives
    her reply from the live turn (read when the reading is done)."""
    if cfr is None:
        from gaia_common.utils.cfr_manager import CFRManager
        cfr = CFRManager()
    sections = _sections(text, cfr)
    out = follow_up(name, asked, said() or "", sections, generate, session_id)
    logger.info("long_read: finished %s (%d sections); %s", name, len(sections),
                "following up" if out else "nothing to add")
    if out and send(session_id, out):
        record(out)
    return out


def start(*args, **kwargs) -> threading.Thread:
    def run():
        try:
            finish_reading(*args, **kwargs)
        except Exception:
            logger.warning("long_read: reading failed", exc_info=True)
    t = threading.Thread(target=run, name="long-read", daemon=True)
    t.start()
    return t
