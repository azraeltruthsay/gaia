import logging
import re
import difflib
from typing import List, Any, Optional

logger = logging.getLogger("GAIA.EntityValidator")

class EntityValidator:
    """
    Validates and corrects project-specific nouns and entities in text
    using fuzzy matching against a canonical registry.
    """
    
    def __init__(self, canonical_entities: Optional[List[str]] = None):
        # Default project entities if none provided
        self.entities = canonical_entities or [
            "GAIA", "Azrael", "Core", "Prime", "Study", "MCP", "Orchestrator",
            "Samvega", "CognitionPacket", "Epistemic Drive", "SleepCycle",
            "BlueShot", "Mindscape", "Sovereign", "Artisanal", "Handcrafted",
            "Six-Tier Memory", "Immutable Self", "Cognitive Forge"
        ]
        
        # Common known misspellings mapping (high-confidence overrides)
        self.hard_mappings = {
            "asriel": "Azrael",
            "gaya": "GAIA",
            "m-c-pee": "MCP",
            "mcp-lite": "MCP",
            "v-l-l-m": "vLLM",
            "vllm": "vLLM",
        }

    def correct_text(self, text: str, threshold: float = 0.8) -> str:
        """
        Scans text for entities and applies fuzzy corrections.
        """
        if not text:
            return text

        # 1. Apply hard mappings first (case-insensitive)
        for wrong, right in self.hard_mappings.items():
            # Use regex with word boundaries to avoid partial matches
            pattern = re.compile(re.escape(wrong), re.IGNORECASE)
            text = pattern.sub(right, text)

        # 2. Fuzzy match potential nouns
        # Find capitalized words or sequences that might be entities
        # This is a simple heuristic; can be improved with NER later
        words = re.findall(r'\b[A-Z][a-z]*\b', text)
        
        corrections = {}
        for word in set(words):
            if word in self.entities:
                continue
                
            # Check for close matches in canonical list
            matches = difflib.get_close_matches(word, self.entities, n=1, cutoff=threshold)
            if matches:
                corrections[word] = matches[0]
                logger.debug(f"Entity correction: {word} -> {matches[0]}")

        # Apply fuzzy corrections
        for wrong, right in corrections.items():
            pattern = re.compile(r'\b' + re.escape(wrong) + r'\b')
            text = pattern.sub(right, text)

        return text

    @classmethod
    def from_config(cls, config: Any) -> 'EntityValidator':
        """
        Factory method to build a validator from GAIA Config/Constants.
        """
        entities = [
            "GAIA", "Azrael", "Core", "Prime", "Study", "MCP", "Orchestrator",
            "Samvega", "CognitionPacket", "Epistemic Drive", "SleepCycle",
            "BlueShot", "Mindscape", "Sovereign", "Artisanal", "Handcrafted"
        ]
        
        # Try to pull service names from constants
        try:
            endpoints = config.constants.get("SERVICE_ENDPOINTS", {})
            for service in endpoints.keys():
                entities.append(service.capitalize())
                entities.append(f"gaia-{service}")
        except Exception:
            pass
            
        return cls(list(set(entities)))
