"""
Goal Detection Module — detects and carries user goals across conversation turns.

Three detection paths:
  1. Fast-path — self-evident intents map directly to a goal
  2. Session-carry — active goal carries forward with decay
  3. LLM inference — Lite model classifies ambiguous goals (~280 tokens)
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Optional

from gaia_common.protocols.cognition_packet import (
    CognitionPacket,
    DetectedGoal,
    GoalConfidence,
    GoalState,
)

logger = logging.getLogger("GAIA.GoalDetector")

# Maximum turns an active goal carries before decaying to LOW confidence
MAX_CARRY_TURNS = 8

# Fast-path: intent string → (goal_id, description)
FAST_PATH_MAP: dict[str, tuple[str, str]] = {
    "greeting": ("casual_conversation", "Casual greeting or social interaction"),
    "farewell": ("casual_conversation", "Wrapping up conversation"),
    "question": ("information_seeking", "Seeking specific information or answers"),
    "help_request": ("task_assistance", "Requesting help with a task"),
    "tool_use": ("task_execution", "Executing a specific tool-based task"),
    "clarification": ("information_seeking", "Clarifying previously discussed information"),
    "acknowledgement": ("casual_conversation", "Acknowledging or confirming understanding"),
}

# LLM prompt template — kept lean for ~280 token budget
_GOAL_PROMPT = """\
Given the user's message and detected intent, classify the user's overarching goal in 1-2 words (snake_case) and a brief description.

Intent: {intent}
User message: {user_input}

Respond in exactly this format:
GOAL_ID: <snake_case_id>
DESCRIPTION: <one sentence description>
CONFIDENCE: high|medium|low"""


class GoalDetector:
    """Detects and manages user goals across conversation turns."""

    def __init__(self, config=None):
        self.config = config

    def detect(
        self,
        packet: CognitionPacket,
        session_manager,
        session_id: str,
        model_pool=None,
    ) -> GoalState:
        """Run the three-path detection pipeline and return an updated GoalState."""
        intent_str = packet.intent.user_intent if packet.intent else ""
        user_input = packet.content.original_prompt if packet.content else ""

        # Path 1: Fast-path — direct intent→goal mapping
        fast_goal = self._fast_path_detect(intent_str, user_input)
        if fast_goal:
            state = GoalState(current_goal=fast_goal, turn_count=0)
            self._persist_goal(session_manager, session_id, state)
            logger.info(f"Goal fast-path: {fast_goal.goal_id} ({fast_goal.confidence.value})")
            return state

        # Path 2: Session carry — reuse active goal from previous turn
        carried = self._session_carry(session_manager, session_id)
        if carried:
            logger.info(
                f"Goal carry: {carried.current_goal.goal_id} "
                f"(turn {carried.turn_count}, {carried.current_goal.confidence.value})"
            )
            return carried

        # Path 3: LLM inference via Lite model
        llm_goal = self._llm_detect(packet, model_pool)
        if llm_goal:
            state = GoalState(current_goal=llm_goal, turn_count=0)
            self._persist_goal(session_manager, session_id, state)
            logger.info(f"Goal LLM-detected: {llm_goal.goal_id} ({llm_goal.confidence.value})")
            return state

        # No goal detected — return empty state
        return GoalState()

    # ── Fast-path detection ──────────────────────────────────────────

    def _fast_path_detect(self, intent: str, user_input: str) -> Optional[DetectedGoal]:
        """Map well-known intents directly to goals without LLM inference."""
        intent_lower = intent.lower().strip()
        if intent_lower in FAST_PATH_MAP:
            goal_id, description = FAST_PATH_MAP[intent_lower]
            return DetectedGoal(
                goal_id=goal_id,
                description=description,
                confidence=GoalConfidence.HIGH,
                detected_at=datetime.now(timezone.utc).isoformat(),
                source="fast_path",
            )
        return None

    # ── Session carry ────────────────────────────────────────────────

    def _session_carry(self, session_manager, session_id: str) -> Optional[GoalState]:
        """Carry forward the active goal from the previous turn with decay."""
        if session_manager is None:
            return None

        stored = session_manager.get_session_meta(session_id, "goal_state")
        if not stored or not stored.get("current_goal"):
            return None

        turn_count = stored.get("turn_count", 0) + 1
        goal_data = stored["current_goal"]

        # Decay: after MAX_CARRY_TURNS, drop confidence to LOW
        if turn_count >= MAX_CARRY_TURNS:
            confidence = GoalConfidence.LOW
        else:
            confidence = GoalConfidence(goal_data.get("confidence", "medium"))

        carried_goal = DetectedGoal(
            goal_id=goal_data["goal_id"],
            description=goal_data["description"],
            confidence=confidence,
            detected_at=goal_data["detected_at"],
            source="session_carry",
        )

        previous = [
            DetectedGoal(**pg)
            for pg in stored.get("previous_goals", [])
        ]

        state = GoalState(
            current_goal=carried_goal,
            previous_goals=previous,
            turn_count=turn_count,
            goal_shifts=stored.get("goal_shifts", 0),
        )
        # Persist updated turn count
        self._persist_goal(session_manager, session_id, state)
        return state

    # ── LLM inference ────────────────────────────────────────────────

    def _llm_detect(self, packet: CognitionPacket, model_pool) -> Optional[DetectedGoal]:
        """Use the Lite model for goal classification when fast-path fails."""
        if model_pool is None:
            return None

        try:
            llm = model_pool.get_model_for_role("lite")
        except Exception:
            llm = None
        if llm is None:
            return None

        intent_str = packet.intent.user_intent if packet.intent else "unknown"
        user_input = packet.content.original_prompt if packet.content else ""
        if not user_input.strip():
            return None

        prompt = _GOAL_PROMPT.format(intent=intent_str, user_input=user_input[:500])

        try:
            result = llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=128,
            )
            response_text = (
                result.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            return self._parse_llm_response(response_text)
        except Exception as e:
            logger.warning(f"Goal LLM inference failed: {e}")
            return None

    @staticmethod
    def _parse_llm_response(text: str) -> Optional[DetectedGoal]:
        """Parse the structured LLM response into a DetectedGoal."""
        goal_id_match = re.search(r"GOAL_ID:\s*(.+)", text)
        desc_match = re.search(r"DESCRIPTION:\s*(.+)", text)
        conf_match = re.search(r"CONFIDENCE:\s*(high|medium|low)", text, re.IGNORECASE)

        if not goal_id_match:
            return None

        goal_id = re.sub(r"[^a-z0-9_]", "_", goal_id_match.group(1).strip().lower())
        description = desc_match.group(1).strip() if desc_match else goal_id.replace("_", " ")
        confidence_str = conf_match.group(1).lower() if conf_match else "medium"

        return DetectedGoal(
            goal_id=goal_id,
            description=description,
            confidence=GoalConfidence(confidence_str),
            detected_at=datetime.now(timezone.utc).isoformat(),
            source="llm",
        )

    # ── Goal shift handling ──────────────────────────────────────────

    @staticmethod
    def handle_goal_shift(
        new_goal_desc: str,
        packet: CognitionPacket,
        session_manager,
        session_id: str,
    ):
        """Process a GOAL_SHIFT directive from the LLM response."""
        goal_id = re.sub(r"[^a-z0-9_]", "_", new_goal_desc.strip().lower())[:64]

        new_goal = DetectedGoal(
            goal_id=goal_id,
            description=new_goal_desc.strip(),
            confidence=GoalConfidence.MEDIUM,
            detected_at=datetime.now(timezone.utc).isoformat(),
            source="llm",
        )

        # Archive current goal if present
        previous = []
        if packet.goal_state and packet.goal_state.current_goal:
            previous = list(packet.goal_state.previous_goals or [])
            previous.append(packet.goal_state.current_goal)

        shifts = (packet.goal_state.goal_shifts if packet.goal_state else 0) + 1

        new_state = GoalState(
            current_goal=new_goal,
            previous_goals=previous,
            turn_count=0,
            goal_shifts=shifts,
        )

        packet.goal_state = new_state

        if session_manager is not None:
            GoalDetector._persist_goal_static(session_manager, session_id, new_state)

        logger.info(f"Goal shift → {goal_id} (shift #{shifts})")

    # ── Persistence helpers ──────────────────────────────────────────

    def _persist_goal(self, session_manager, session_id: str, state: GoalState):
        """Persist goal state to session meta."""
        GoalDetector._persist_goal_static(session_manager, session_id, state)

    @staticmethod
    def _persist_goal_static(session_manager, session_id: str, state: GoalState):
        """Static persistence — usable from handle_goal_shift."""
        if session_manager is None:
            return
        serialized = {
            "current_goal": state.current_goal.to_dict() if state.current_goal else None,
            "previous_goals": [pg.to_dict() for pg in (state.previous_goals or [])],
            "turn_count": state.turn_count,
            "goal_shifts": state.goal_shifts,
        }
        session_manager.set_session_meta(session_id, "goal_state", serialized)
