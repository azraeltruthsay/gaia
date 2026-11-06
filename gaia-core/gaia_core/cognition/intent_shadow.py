"""What the person wants, read by meaning — logged, not acted on (d3px).

Keywords sent "how many containers run your architecture?" to a four-phase
planning pipeline and every general-knowledge question to small talk. On 166
real messages (2026-11-06): today's rules 47%; Core on the CPU, asked to
label a message, answered it instead (25%); the embedding model's nearest
labeled examples 63%; a rule when one fires (33/35 right), the embedding
otherwise, 73%.

Shadow mode: on every turn the combined reading is written beside what
routing did, so disagreements can be read and labeled before anything
routes by it. Runs off the turn's thread; a failure costs nothing.

The labeled examples are private conversations and live outside git, in
/shared/evals/intent_d3px/labeled.json. Without them, nothing is logged.
"""
import collections
import json
import logging
import os
import threading
import time
from typing import List, Optional

logger = logging.getLogger("GAIA.IntentShadow")

EXAMPLES_PATH = os.environ.get("INTENT_SHADOW_EXAMPLES", "/shared/evals/intent_d3px/labeled.json")
LOG_PATH = os.environ.get("INTENT_SHADOW_LOG", "/shared/evals/intent_d3px/shadow.jsonl")
_MAX_LOG_BYTES = 5 * 1024 * 1024
_K = 3

_KEYWORD_LABELS = {"recitation": "recite", "read_file": "tool", "write_file": "tool", "find_file": "tool",
                   "list_files": "tool", "list_tree": "tool", "list_tools": "tool", "shell": "tool",
                   "correction": "dispute"}
_SKIP_SOURCES = {"heartbeat", "initiative", "sleep", "system"}

_lock = threading.Lock()
_examples: Optional[list] = None
_matrix = None


def enabled() -> bool:
    return os.environ.get("INTENT_SHADOW", "1").lower() in ("1", "true", "yes", "on")


def by_rules(text: str, history: list) -> Optional[str]:
    """The label a precise rule gives, or None when none fires."""
    from gaia_core.cognition.nlu.intent_detection import (
        _keyword_intent_classify, is_conversation_followup, is_conversation_recall,
        is_factual_pushback, is_knowledge_followup)
    from gaia_core.cognition.told_facts import told_fact
    from gaia_core.cognition.earlier_lookup import refers_to_results
    from gaia_common.utils.founding_docs import named_document
    from gaia_common.utils.host_facts import asks_about_host
    if told_fact(text):
        return "tell"
    if asks_about_host(text):
        return "state"
    if named_document(text):
        return "self"
    if is_conversation_recall(text) or is_conversation_followup(text, history):
        return "recall"
    if is_factual_pushback(text, history):
        return "dispute"
    if refers_to_results(text) or is_knowledge_followup(text, history):
        return "followup"
    return _KEYWORD_LABELS.get(_keyword_intent_classify(text))


def _load(embed_model) -> bool:
    global _examples, _matrix
    if _examples is not None:
        return bool(_examples)
    try:
        with open(EXAMPLES_PATH, encoding="utf-8") as f:
            rows = [r for r in json.load(f) if r.get("label") and r.get("text")]
        _matrix = embed_model.encode([r["text"][:400] for r in rows], normalize_embeddings=True)
        _examples = rows
        logger.info("Intent shadow: %d labeled examples", len(rows))
    except FileNotFoundError:
        _examples = []
        logger.info("Intent shadow: no labeled examples at %s; nothing will be logged", EXAMPLES_PATH)
    except Exception:
        _examples = []
        logger.warning("Intent shadow: could not load examples", exc_info=True)
    return bool(_examples)


def by_embedding(text: str, embed_model) -> tuple:
    """(label, similarity of the nearest example) by the nearest labeled examples."""
    import numpy as np
    v = embed_model.encode([text[:400]], normalize_embeddings=True)[0]
    sims = _matrix @ v
    votes = collections.Counter()
    for j in np.argsort(-sims)[:_K]:
        votes[_examples[j]["label"]] += float(sims[j])
    return votes.most_common(1)[0][0], round(float(sims.max()), 3)


def read(text: str, history: list, embed_model) -> Optional[dict]:
    """The combined reading of one message, or None without examples."""
    with _lock:
        if not _load(embed_model):
            return None
    rule = by_rules(text, history)
    emb, sim = by_embedding(text, embed_model)
    return {"rule": rule, "embed": emb, "embed_sim": sim, "label": rule or emb}


def observe(text: str, history: List[dict], session_id: str, source: str, model_pool) -> None:
    """Log the reading of this turn in the background."""
    if not enabled() or not (text or "").strip() or (source or "") in _SKIP_SOURCES:
        return
    prev = [h for h in history or [] if isinstance(h, dict) and h.get("role") in ("user", "assistant")][-2:]
    ts = time.time()

    def _run():
        try:
            embed = model_pool.get_embed_model(timeout=0, lazy_load=True)
            if embed is None:
                return
            got = read(text, prev, embed)
            if got:
                _write({"ts": ts, "session_id": session_id, "source": source, "text": text[:400],
                        "prev_user": next((h.get("content", "")[:200] for h in reversed(prev) if h.get("role") == "user"), ""),
                        **got, "ms": round((time.time() - ts) * 1000)})
        except Exception:
            logger.debug("Intent shadow failed (non-fatal)", exc_info=True)

    threading.Thread(target=_run, name="intent-shadow", daemon=True).start()


def _write(entry: dict) -> None:
    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        if os.path.exists(LOG_PATH) and os.path.getsize(LOG_PATH) > _MAX_LOG_BYTES:
            os.replace(LOG_PATH, LOG_PATH + ".1")
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as e:
        logger.debug("Intent shadow: could not log: %s", e)
