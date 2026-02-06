"""
conversation/archiver.py

Handles saving and loading archived conversations in Markdown format.
Manages storage, file formatting, and retrieval operations.
"""

import os
import json
import logging
from datetime import datetime
from typing import List
from gaia_core.config import Config, get_config

logger = logging.getLogger("GAIA.ConversationArchiver")

class ConversationArchiver:
    """
    Saves a conversation history to disk, structured by persona + session ID.
    """

    def __init__(self, config):
        self.config = config

    def archive_conversation(self, session_id: str, persona: str, messages: List[dict], summary: str, keywords: List[str]):
        try:
            base_path = os.path.join(self.config.HISTORY_DIR, "conversations", persona)
            os.makedirs(base_path, exist_ok=True)
            file_path = os.path.join(base_path, f"{session_id}.json")

            archive = {
                "session_id": session_id,
                "persona": persona,
                "timestamp": datetime.utcnow().isoformat(),
                "summary": summary,
                "keywords": keywords,
                "messages": messages
            }

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(archive, f, indent=2)

            logger.info(f"💾 Conversation archived: {file_path}")

        except Exception as e:
            logger.error(f"❌ Failed to archive conversation: {e}", exc_info=True)
