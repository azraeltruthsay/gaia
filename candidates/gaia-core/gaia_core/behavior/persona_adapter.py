"""
Persona Adapter (pillar-compliant, robust)
- Adapts/merges persona config with current pipeline context.
- Ensures context has correct template, instructions, and allows future persona behaviors.
"""
import logging

logger = logging.getLogger("GAIA.PersonaAdapter")

class PersonaAdapter:
    """
    Adapts/wraps raw persona data into a consistent object for use throughout GAIA.
    Ensures persona attributes (name, template, instructions, traits) are readily accessible.
    """
    def __init__(self, persona_data: dict, config=None):
        self.name = persona_data.get("name", "default")
        self.description = persona_data.get("description", "")
        self.template = persona_data.get("template", "")
        self.instructions = persona_data.get("instructions", []) # Should be a list of strings
        self.traits = persona_data.get("traits", {}) # Dictionary of traits
        self.config = config # Keep a reference to config if needed for future dynamic behavior

        # Ensure instructions is always a list of strings for consistency
        if isinstance(self.instructions, str):
            self.instructions = [self.instructions]
        elif not isinstance(self.instructions, list):
            self.instructions = []

        logger.debug(f"Initialized PersonaAdapter for '{self.name}'")

    def get_full_instructions(self) -> str:
        """Combines template and instructions into a single string."""
        full_text = []
        if self.template:
            full_text.append(self.template)
        if self.instructions:
            full_text.extend(self.instructions)
        return "\n".join(full_text).strip()

    def __repr__(self):
        return f"<PersonaAdapter: {self.name}>"

    def __str__(self):
        return self.name

