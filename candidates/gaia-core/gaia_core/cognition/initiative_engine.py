"""
Initiative Engine — ported from archive/gaia-assistant-monolith/run_gil.py.

Executes a single autonomous thought cycle: picks the highest-priority topic
from the topic cache and feeds a self-generated prompt to AgentCore.

Unlike the archived GIL which ran its own ``schedule`` loop, this version is a
callable invoked by :class:`SleepTaskScheduler` during the SLEEPING state.
The idle check is already handled by the sleep state machine.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("GAIA.InitiativeEngine")

GIL_SESSION_ID = "gaia_initiative_loop_session"
TOPIC_CACHE_PATH = "/knowledge/system_reference/topic_cache.json"


class InitiativeEngine:
    """Autonomous thought engine driven by the topic manager."""

    def __init__(self, config, agent_core=None) -> None:
        self.config = config
        self.agent_core = agent_core

    def execute_turn(self) -> Optional[Dict[str, Any]]:
        """Run one autonomous reflection cycle.

        Returns a dict with ``topic_id`` and ``status`` on success,
        or ``None`` when there are no topics (or ``agent_core`` is missing).
        """
        if self.agent_core is None:
            logger.warning("InitiativeEngine: agent_core not available, skipping turn")
            return None

        # 1. Select the top-priority unresolved topic
        from gaia_core.cognition.topic_manager import prioritize_topics

        top_topics = prioritize_topics(TOPIC_CACHE_PATH, top_n=1)
        if not top_topics:
            logger.info("InitiativeEngine: no active topics — nothing to do")
            return None

        topic = top_topics[0]
        topic_id = topic.get("topic_id", "unknown")
        topic_desc = topic.get("topic", "")
        logger.info("InitiativeEngine: selected topic [%s] — %s", topic_id, topic_desc)

        # 2. Build a self-prompt (ported from run_gil.py)
        self_prompt = self._build_self_prompt(topic)

        # 3. Run through AgentCore, consuming all events
        try:
            logger.info("InitiativeEngine: handing self-prompt to AgentCore...")
            for event in self.agent_core.run_turn(
                user_input=self_prompt,
                session_id=GIL_SESSION_ID,
            ):
                if isinstance(event, dict) and event.get("type") != "token":
                    logger.debug("GIL event: %s", event)

            logger.info("InitiativeEngine: turn complete for topic [%s]", topic_id)
            return {"topic_id": topic_id, "status": "complete"}

        except Exception:
            logger.error("InitiativeEngine: turn failed for topic [%s]", topic_id, exc_info=True)
            return {"topic_id": topic_id, "status": "error"}

    @staticmethod
    def _build_self_prompt(topic: Dict[str, Any]) -> str:
        """Build the autonomous reflection prompt from a topic dict."""
        topic_desc = topic.get("topic", "")
        return (
            "[Autonomous Reflection Cycle]\n"
            f"My current highest-priority, unresolved topic is: '{topic_desc}'.\n"
            f"The topic's metadata is: {topic}.\n"
            "\n"
            "My task is to analyze this topic and decide on the next step.\n"
            "- If I have enough information to resolve it, I will use the "
            "`resolve_topic` primitive.\n"
            "- If I need to do more work or break it down, I will use the "
            "`update_topic` primitive to update its status or the `add_topic` "
            "primitive to create sub-tasks.\n"
            "- If the task requires writing code or a document, I will use the "
            "`ai.write` primitive.\n"
            "\n"
            "Based on this, what is my next logical action?"
        )
