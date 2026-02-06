
"""

Manages the creation, update, resolution, prioritization, and pruning of GAIA's topic cache.

Supports the Initiative Loop (GIL) by maintaining a prioritized, well-structured list of emergent discussion topics.

Location: /app/utils/topic_manager.py

"""

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any

logger = logging.getLogger("GAIA.TopicManager")

def _load_topic_cache(path: str) -> List[Dict[str, Any]]:
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"🟡 No existing topic cache found at {path}. Initializing new one.")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"❌ Failed to decode topic cache: {e}")
        return []

def _save_topic_cache(path: str, cache: List[Dict[str, Any]]):
    try:
        with open(path, 'w') as f:
            json.dump(cache, f, indent=2)
        logger.info(f"✅ Topic cache updated at {path} with {len(cache)} topics.")
    except Exception as e:
        logger.error(f"❌ Failed to write topic cache to {path}: {e}")

def add_topic(path: str, topic: Dict[str, Any]) -> None:
    cache = _load_topic_cache(path)
    topic.setdefault("topic_id", f"topic-{len(cache) + 1}")
    topic.setdefault("urgency", 0.5)
    topic.setdefault("emotional_weight", 0.5)
    topic.setdefault("relevance_to_current_context", 0.5)
    topic.setdefault("novelty_score", 0.5)
    topic.setdefault("surface", True)
    topic.setdefault("resolved", False)
    topic.setdefault("last_discussed", None)
    topic.setdefault("source", "self_reflection")
    cache.append(topic)
    _save_topic_cache(path, cache)
    logger.info(f"➕ Added new topic: {topic.get('topic_id')} :: {topic.get('topic')}")

def resolve_topic(path: str, topic_id: str) -> bool:
    cache = _load_topic_cache(path)
    for t in cache:
        if t.get("topic_id") == topic_id:
            t["resolved"] = True
            t["last_discussed"] = datetime.utcnow().isoformat()
            _save_topic_cache(path, cache)
            logger.info(f"✔️ Resolved topic {topic_id}")
            return True
    logger.warning(f"⚠️ Topic to resolve not found: {topic_id}")
    return False

def update_topic(path: str, topic_id: str, updates: Dict[str, Any]) -> bool:
    cache = _load_topic_cache(path)
    for t in cache:
        if t.get("topic_id") == topic_id:
            t.update(updates)
            _save_topic_cache(path, cache)
            logger.info(f"✏️ Updated topic {topic_id} with {updates}")
            return True
    logger.warning(f"⚠️ Topic to update not found: {topic_id}")
    return False

def prune_resolved_topics(path: str) -> None:
    cache = _load_topic_cache(path)
    original_count = len(cache)
    cache = [t for t in cache if not t.get("resolved", False)]
    _save_topic_cache(path, cache)
    logger.info(f"🧹 Pruned {original_count - len(cache)} resolved topics.")

def list_topics(path: str, include_resolved: bool = False) -> List[Dict[str, Any]]:
    cache = _load_topic_cache(path)
    if not include_resolved:
        cache = [t for t in cache if not t.get("resolved", False)]
    return cache

def prioritize_topics(path: str, top_n: Optional[int] = 5) -> List[Dict[str, Any]]:
    """
    Sort topics by weighted priority and return the top N entries.
    Priority is based on urgency, emotional weight, relevance, and novelty.
    """
    cache = list_topics(path, include_resolved=False)
    for topic in cache:
        topic["priority_score"] = (
            0.4 * topic.get("urgency", 0.5) +
            0.2 * topic.get("emotional_weight", 0.5) +
            0.3 * topic.get("relevance_to_current_context", 0.5) +
            0.1 * topic.get("novelty_score", 0.5)
        )
    sorted_cache = sorted(cache, key=lambda x: x["priority_score"], reverse=True)
    logger.debug(f"🔎 Prioritized {len(sorted_cache)} topics.")
    return sorted_cache[:top_n] if top_n else sorted_cache
