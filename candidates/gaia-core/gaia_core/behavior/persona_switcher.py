# app/behavior/persona_switcher.py
"""
This module contains the logic for dynamically switching GAIA's persona based on user intent.
"""

import json
import logging
import os
import unicodedata
from typing import Tuple, Optional

logger = logging.getLogger("GAIA.PersonaSwitcher")


def _normalize_text(text: str) -> str:
    """Normalize Unicode text to ASCII for keyword matching.

    Converts special characters like æ → ae, ē → e, etc.
    """
    # NFKD normalization decomposes characters (e.g., ē → e + combining macron)
    normalized = unicodedata.normalize('NFKD', text)
    # Remove combining characters (accents, macrons, etc.) and keep base letters
    ascii_text = ''.join(c for c in normalized if not unicodedata.combining(c))
    # Handle special ligatures that NFKD doesn't decompose fully
    ascii_text = ascii_text.replace('æ', 'ae').replace('Æ', 'AE')
    ascii_text = ascii_text.replace('œ', 'oe').replace('Œ', 'OE')
    return ascii_text

# Keyword mappings for persona detection
PERSONA_KEYWORDS = {
    "dnd_player_assistant": [
        "d&d", "dnd", "character sheet", "spell", "strauthauk", "axuraud",
        "rupert", "dungeon master", "player character", "braeneage", "heimr",
        # Special character variants for Braeneage/Brænēage
        "brænēage", "brǣnēage", "braenēage", "brænage",
    ],
}

# Default personas directory - matches GAIA configuration
PERSONAS_DIR = os.environ.get("GAIA_PERSONAS_DIR", "/knowledge/personas")


def _load_persona_config(persona_name: str) -> Optional[dict]:
    """Load persona configuration from JSON file."""
    # Try nested structure first: personas/<name>/<name>_persona.json
    nested_path = os.path.join(PERSONAS_DIR, persona_name, f"{persona_name}_persona.json")
    # Then try simple structure: personas/<name>.json
    simple_path = os.path.join(PERSONAS_DIR, f"{persona_name}.json")

    for path in [nested_path, simple_path]:
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load persona config from {path}: {e}")
    return None


def get_knowledge_base_for_persona(persona_name: str) -> Optional[str]:
    """
    Get the knowledge_base_name from a persona's configuration file.

    Args:
        persona_name: The name of the persona.

    Returns:
        The knowledge_base_name if configured, None otherwise.
    """
    config = _load_persona_config(persona_name)
    if config:
        kb_name = config.get("knowledge_base_name")
        if kb_name:
            logger.debug(f"Persona '{persona_name}' uses knowledge base: {kb_name}")
            return kb_name
    return None


def get_persona_for_request(user_input: str) -> Tuple[str, Optional[str]]:
    """
    Determines the appropriate persona and knowledge base for a given user request.

    Args:
        user_input: The user's message.

    Returns:
        A tuple containing (persona_name, knowledge_base_name).
        knowledge_base_name is read from the persona's JSON configuration.
    """
    input_lower = user_input.lower()
    # Also create a normalized version for matching special characters (æ→ae, ē→e)
    input_normalized = _normalize_text(input_lower)

    # Check each persona's keywords
    for persona_name, keywords in PERSONA_KEYWORDS.items():
        # Check against both original (lowercased) and normalized input
        if any(keyword in input_lower or keyword in input_normalized for keyword in keywords):
            logger.info(f"Intent detected for persona: '{persona_name}'")
            # Load knowledge_base_name from the persona's configuration
            knowledge_base_name = get_knowledge_base_for_persona(persona_name)
            return persona_name, knowledge_base_name

    # Default to the "dev" persona and check if it has a knowledge base configured
    default_persona = "dev"
    default_kb = get_knowledge_base_for_persona(default_persona)
    return default_persona, default_kb
