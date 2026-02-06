import os
import json
import logging
from typing import Dict, Optional
from threading import Lock

logger = logging.getLogger("GAIA.KnowledgeIndex")

class KnowledgeIndex:
    """
    A lightweight persistent index that tracks knowledge hashes for duplicate detection,
    document embedding, and memory tier alignment.
    """

    def __init__(self, path="/app/utils/knowledge_index.json"):
        self.path = path
        self.index: Dict[str, Dict] = {}
        self.lock = Lock()
        self.load()

    def load(self):
        if not os.path.exists(self.path):
            logger.info(f"📘 No existing knowledge index at {self.path}. Starting fresh.")
            return
        try:
            with open(self.path, 'r', encoding='utf-8') as f:
                self.index = json.load(f)
                logger.info(f"📘 Loaded knowledge index from {self.path} with {len(self.index)} entries.")
        except Exception as e:
            logger.error(f"❌ Failed to load knowledge index: {e}", exc_info=True)
            self.index = {}

    def save(self):
        try:
            with open(self.path, 'w', encoding='utf-8') as f:
                json.dump(self.index, f, indent=2)
                logger.debug(f"📘 Saved knowledge index to {self.path}.")
        except Exception as e:
            logger.error(f"❌ Failed to save knowledge index: {e}", exc_info=True)

    def add(self, file_path: str, hash_value: str, tier: str):
        with self.lock:
            self.index[file_path] = {
                "hash": hash_value,
                "tier": tier
            }
            logger.debug(f"📝 Added to knowledge index: {file_path} (tier {tier})")
            self.save()

    def get(self, file_path: str) -> Optional[Dict]:
        return self.index.get(file_path)

    def get_hash(self, file_path: str) -> Optional[str]:
        return self.index.get(file_path, {}).get("hash")

    def was_already_processed(self, file_path: str, hash_value: str) -> bool:
        """
        Returns True if file was processed before with the same hash.
        """
        return self.get_hash(file_path) == hash_value

    def remove(self, file_path: str):
        with self.lock:
            if file_path in self.index:
                del self.index[file_path]
                logger.info(f"🗑️ Removed {file_path} from knowledge index.")
                self.save()

    def list_all(self):
        return list(self.index.keys())
