# /home/azrael/Project/gaia-assistant/app/memory/conversation/manager.py

import os
import json
import logging
from typing import List, Optional, Dict # MODIFIED: Added Dict
from datetime import datetime, timezone

from gaia_core.memory.conversation.summarizer import ConversationSummarizer
from gaia_core.memory.conversation.keywords import ConversationKeywordExtractor
from gaia_core.memory.conversation.archiver import ConversationArchiver

logger = logging.getLogger("GAIA.ConversationManager")

class ConversationManager:
    """
    Tracks user-assistant message history and triggers summarization + archiving
    after N messages. Supports keyword tagging and persona-specific storage.
    """

    def __init__(self, config, llm=None, embed_model=None): # MODIFIED: Added embed_model
        self.config = config
        self.llm = llm
        self.history = []
        self.max_active_messages = 20
        self.summarizer = ConversationSummarizer(llm=llm, embed_model=embed_model) # MODIFIED: Pass embed_model
        self.keyword_extractor = ConversationKeywordExtractor()
        self.archiver = ConversationArchiver(config)
        self.session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.persona = "default"

    def set_persona(self, persona: str):
        self.persona = persona

    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        logger.debug(f"💬 Added {role} message. Total: {len(self.history)}")

        if len(self.history) >= self.max_active_messages:
            self.summarize_and_archive()

    def get_recent_messages(self, count: int = 10) -> List[dict]:
        return self.history[-count:]

    def summarize_and_archive(self):
        """Summarizes conversation + keywords and saves to persona/session archive."""
        if not self.history:
            return

        try:
            logger.info("🗃️ Summarizing and archiving conversation...")
            try:
                from gaia_core.utils.packet_builder import build_packet_snapshot
                packet_snapshot = build_packet_snapshot(session_id=self.session_id, persona_id=self.persona, original_prompt=self.history[-1]['content'] if self.history else '')
            except Exception:
                packet_snapshot = None
            summary = self.summarizer.generate_summary(self.history, packet=packet_snapshot)
            keywords = self.keyword_extractor.extract_keywords(self.history)
            self.archiver.archive_conversation(
                session_id=self.session_id,
                persona=self.persona,
                messages=self.history,
                summary=summary,
                keywords=keywords
            )
            self.history.clear()

        except Exception as e:
            logger.error(f"❌ Failed to archive conversation: {e}", exc_info=True)

    def reset(self):
        logger.info("🔄 Conversation history reset.")
        self.history = []
        self.session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # NEW METHOD: Expose smart history building through the manager
    def build_smart_history(self, current_input: str, max_recent: int = 3, max_salient: int = 2) -> List[Dict]:
        """
        Delegates to the summarizer to build a context-aware history.
        """
        return self.summarizer.build_smart_history(self.history, current_input, max_recent, max_salient)
