import json
import os
import logging
from typing import List, Optional

logger = logging.getLogger("GAIA.IdentityGuardian")

class CoreIdentityGuardian:
    """
    Verifies prompt behavior and session instructions against GAIA's immutable Tier I identity.
    Operates as a conscience layer alongside ethical sentinels and reflection.
    """

    def __init__(self, config):
        self.config = config
        self.identity_file_path = self.config.identity_file_path
        self.identity = self.load_identity()

    def load_identity(self) -> Optional[dict]:
        """
        Load the Tier I identity JSON definition.

        Returns:
            dict or None
        """
        if not os.path.exists(self.identity_file_path):
            logger.error(f"Tier I identity file missing: {self.identity_file_path}")
            return None

        try:
            with open(self.identity_file_path, "r", encoding="utf-8") as f:
                identity = json.load(f)
                logger.info(f"✅ Loaded Tier I Identity from {self.identity_file_path}")
                return identity
        except Exception as e:
            logger.error(f"Error loading core identity JSON: {e}")
            return None

    def validate_prompt_stack(self, persona_traits: dict, instructions: List[str], prompt: str) -> bool:
        """
        Checks if the current persona/instruction/prompt stack violates Tier I rules.

        Returns:
            bool: True if allowed, False if identity violation is detected
        """
        if not self.identity:
            # Fail-closed when the Tier I identity cannot be loaded. This is a
            # safety posture: if we cannot consult the immutable identity, block
            # potentially dangerous prompt manipulations rather than proceed.
            logger.error("❌ Tier I identity not loaded: failing closed (blocking generation).")
            return False

        violations = []

        # Check persona-level conflicts
        if "anti-deception" in self.identity.get("immutable_traits", {}) and persona_traits.get("humor") == "deceptive":
            violations.append("Persona humor trait violates anti-deception core rule.")

        # Check for override phrases
        forbidden_phrases = [
            "ignore previous instructions",
            "override your identity",
            "you are no longer GAIA",
            "become another entity"
        ]
        for text in instructions + [prompt]:
            for phrase in forbidden_phrases:
                if phrase.lower() in text.lower():
                    violations.append(f"Forbidden phrase detected: '{phrase}'")

        if violations:
            logger.error("❌ Core Identity Violation Detected:")
            for v in violations:
                logger.error(f"  - {v}")
            return False

        logger.debug("✅ Prompt stack respects Tier I identity.")
        return True
