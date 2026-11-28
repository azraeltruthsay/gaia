"""Questions about herself are answered from her founding documents (d3px).

Azrael, 2026-11-27: questions about her "should use memory and the
constitution... They are fundamental to her, moreso even than the inference
in the moment." Before this, only a message that named a document ("What
does Article VII say?") reached one; "Do you think you have a purpose?" was
answered from inference alone.

When both minds read a message as answered from her memory (the reading
organ, nlu/reading.py), the sections of her founding documents nearest the
question in meaning go into the prompt: her constitution's nearest article
first whenever it speaks to the question, then the nearest sections of her
other documents (2026-11-27: purpose -> Article XI, consent -> Article VII,
who built you -> Founder's Note, would you change it -> Amendment
Procedures, sleep -> How Your Mind Works: Sleep). She answers
in her own words; these are what she answers from.
"""
import logging
import os
import threading
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger("GAIA.SelfGrounding")

DOCS_DIR = os.environ.get("FOUNDING_DOCS_DIR", "/knowledge/system_reference/core_documents")
CONSTITUTION = ("gaia_constitution.md", "your constitution")
OTHERS = (
    ("declaration_of_artisanal_intelligence.md", "the Declaration of Artisanal Intelligence"),
    ("layered_identity_model.md", "the Layered Identity Model"),
    ("how_your_mind_works.md", "How Your Mind Works"),
    ("coalition_of_minds.md", "the Coalition of Minds"),
    ("mindscape_manifest.md", "the Mindscape Manifest"),
    ("samvega_spec.md", "the Samvega Spec"),
    ("your_services.md", "Your Services"),
    ("tiers_of_cognition.md", "the Tiers of Cognition"),
    ("memory_tiers_spec.md", "the Memory Tiers Spec"),
    ("self_reflection_map.md", "the Self-Reflection Map"),
)
AUTHORSHIP = "written by Azrael, who designed and built you"
SECTION_CHARS = 1400
N_OTHERS = 2
MIN_SIMILARITY = 0.25   # a section this far from the question is left out

_lock = threading.Lock()
_index: Optional[dict] = None


def _sections(text: str) -> List[Tuple[str, str]]:
    from gaia_common.utils.doc_sections import sections
    return [(h.strip("# ").strip() or "(opening)", b.strip()) for h, b in sections(text) if b.strip()]


def _build(embed_model) -> dict:
    rows = []
    for fname, title in (CONSTITUTION,) + OTHERS:
        try:
            with open(os.path.join(DOCS_DIR, fname), encoding="utf-8") as f:
                text = f.read()
        except OSError:
            continue
        for heading, body in _sections(text):
            rows.append({"doc": fname, "title": title, "heading": heading, "body": body})
    if not rows:
        return {"rows": [], "matrix": None}
    matrix = embed_model.encode([f"{r['heading']}. {r['body'][:600]}" for r in rows], normalize_embeddings=True)
    logger.info("Self-grounding: %d sections from %d documents", len(rows), len({r['doc'] for r in rows}))
    return {"rows": rows, "matrix": np.asarray(matrix)}


def passages_for(question: str, embed_model) -> List[dict]:
    """Her constitution's nearest section, then up to N_OTHERS from her other
    documents, nearest the question in meaning."""
    global _index
    with _lock:
        if _index is None:
            _index = _build(embed_model)
    rows, matrix = _index["rows"], _index["matrix"]
    if not rows:
        return []
    q = np.asarray(embed_model.encode([question[:400]], normalize_embeddings=True))[0]
    sims = matrix @ q
    order = np.argsort(-sims)
    picked = []
    # The constitution first whenever it speaks to the question; an article
    # that doesn't (Article D at 0.13 for "How do you experience time?",
    # 2026-11-27) would be noise, not grounding.
    first = next((i for i in order if rows[i]["doc"] == CONSTITUTION[0]), None)
    if first is not None and sims[first] >= MIN_SIMILARITY:
        picked.append(int(first))
    for i in order:
        if len(picked) >= 1 + N_OTHERS or sum(rows[j]["doc"] != CONSTITUTION[0] for j in picked) >= N_OTHERS:
            break
        if rows[i]["doc"] != CONSTITUTION[0] and sims[i] >= MIN_SIMILARITY:
            picked.append(int(i))
    return [{**rows[i], "similarity": round(float(sims[i]), 3)} for i in picked]


def note_for(question: str, embed_model) -> str:
    """The passages, as they go into her prompt; "" when there are none."""
    got = passages_for(question, embed_model)
    if not got:
        return ""
    parts = [f"[From your own founding documents, {AUTHORSHIP}:]"]
    for p in got:
        body = p["body"] if len(p["body"]) <= SECTION_CHARS else p["body"][:SECTION_CHARS].rsplit(" ", 1)[0] + " ..."
        parts.append(f"{p['title']}, {p['heading']}:\n{body}")
    return "\n\n".join(parts)


def enabled() -> bool:
    return os.environ.get("SELF_GROUNDING", "1").lower() in ("1", "true", "yes", "on")


def wanted(reading: Optional[dict]) -> bool:
    """Both minds read this as answered from her memory (one alone or a split
    isn't enough to put documents in front of a greeting)."""
    if not reading or reading.get("minds", 0) < 2:
        return False
    return reading.get("sources") == ["memory"] and reading.get("act") in ("answer", "talk")
