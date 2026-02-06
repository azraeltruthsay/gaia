"""
conversation/keywords.py

Handles keyword extraction from conversation history.
Lightweight heuristics for identifying important terms.
"""

import re
import logging
from typing import List

logger = logging.getLogger("GAIA.KeywordExtractor")

COMMON_WORDS = {
    "the", "and", "for", "that", "have", "with", "this", "from", "your",
    "are", "not", "but", "was", "you", "has", "can", "all", "will", "about",
    "just", "what", "get", "like", "when", "they", "how", "out", "use", "who",
    "more", "some", "now", "had", "any", "our", "why", "which", "she", "him",
    "her", "its", "his", "them", "been", "did", "than", "then", "were", "say",
    "said", "because"
}

class ConversationKeywordExtractor:
    """
    Extracts high-value keywords from a conversation history.
    """

    def extract_keywords(self, messages: List[dict], max_keywords: int = 10) -> List[str]:
        try:
            all_text = " ".join(msg['content'] for msg in messages if msg.get('content'))
            words = re.findall(r"\b[a-zA-Z]{4,}\b", all_text.lower())
            counts = {}

            for word in words:
                if word in COMMON_WORDS:
                    continue
                counts[word] = counts.get(word, 0) + 1

            sorted_words = sorted(counts.items(), key=lambda x: x[1], reverse=True)
            keywords = [word for word, count in sorted_words[:max_keywords]]
            logger.debug(f"🏷️ Extracted keywords: {keywords}")
            return keywords

        except Exception as e:
            logger.error(f"❌ Failed to extract keywords: {e}", exc_info=True)
            return []
