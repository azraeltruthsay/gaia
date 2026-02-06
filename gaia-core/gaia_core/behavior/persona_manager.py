# /home/azrael/Project/gaia-assistant/app/behavior/persona_manager.py

import os
import json
import logging
from typing import Optional, Dict, List

logger = logging.getLogger("GAIA.PersonaManager")

class PersonaManager:
    """
    Manages loading and listing of GAIA's personas from disk.
    This class is a stateless service for retrieving persona data.
    """
    def __init__(self, personas_dir: str):
        self.personas_dir = personas_dir
        if not os.path.isdir(self.personas_dir):
            logger.error(f"Personas directory not found: {self.personas_dir}. Personas cannot be loaded.")
            self.personas_dir = None # Prevent further errors

    def load_persona_data(self, name: str) -> Optional[Dict]:
        """
        Loads a persona's data from its JSON file based on the standard
        directory structure: /personas/<name>/<name>_persona.json
        """
        if not self.personas_dir:
            return None

        # This path construction is more robust for nested persona files.
        # Note: We assume the persona name is the directory name.
        persona_filepath = os.path.join(self.personas_dir, name, f"{name}_persona.json")

        if not os.path.exists(persona_filepath):
            # A common fallback is to check for a simple file if the directory structure fails
            simple_filepath = os.path.join(self.personas_dir, f"{name}.json")
            if os.path.exists(simple_filepath):
                persona_filepath = simple_filepath
            else:
                logger.warning(f"Persona file not found for '{name}' at {persona_filepath}")
                return None
        try:
            with open(persona_filepath, 'r', encoding='utf-8') as f:
                persona_data = json.load(f)
            logger.info(f"✅ Loaded persona data for '{name}' from {persona_filepath}")
            return persona_data
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding persona JSON for '{name}': {e}")
            return None
        except Exception as e:
            logger.error(f"Error loading persona '{name}': {e}")
            return None

    def list_personas(self) -> List[str]:
        """
        Lists all available persona names by scanning for valid persona directories
        or standalone .json files in the personas directory.
        """
        if not self.personas_dir:
            return []

        personas = set() # Use a set to avoid duplicates
        try:
            for entry in os.listdir(self.personas_dir):
                full_path = os.path.join(self.personas_dir, entry)
                # A valid persona can be a directory containing its own JSON definition file...
                if os.path.isdir(full_path) and os.path.exists(os.path.join(full_path, f"{entry}_persona.json")):
                    personas.add(entry)
                # ...or a simple .json file.
                elif entry.endswith(".json"):
                    personas.add(entry.replace(".json", ""))
        except Exception as e:
            logger.error(f"Failed to list personas from {self.personas_dir}: {e}", exc_info=True)

        return sorted(list(personas))
    
    def get_persona(self, name: str):
        """ Deprecated alias for load_persona_data()."""
        return self.load_persona_data(name)
