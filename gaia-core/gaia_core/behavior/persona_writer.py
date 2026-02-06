import os
import json
import logging
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger("GAIA.PersonaWriter")

class PersonaWriter:
    """
    Handles creation of persona folders and writing JSON + instruction overlays to disk.
    Used during interactive persona design and embedding workflows.
    """

    def __init__(self, vectordb_client, personas_dir="/personas"):
        """
        Args:
            vectordb_client: Optional embedding/indexing client
            personas_dir (str): Root path to save persona folders
        """
        self.personas_dir = personas_dir
        self.vectordb = vectordb_client

    def create_persona_from_template(self, template: Dict, instructions: Optional[Dict[str, str]] = None) -> bool:
        """
        Create a persona folder from a dict template and optional instructions.

        Args:
            template (Dict): Metadata dictionary for the persona
            instructions (Dict): Mapping of instruction name to .txt content

        Returns:
            bool: True if successful, False otherwise
        """
        persona_name = template.get("name")
        if not persona_name:
            logger.error("❌ Persona template missing required field: 'name'")
            return False

        persona_dir = os.path.join(self.personas_dir, persona_name)
        instructions_dir = os.path.join(persona_dir, "instructions")

        if os.path.exists(os.path.join(persona_dir, "persona.json")):
            logger.warning(f"⚠️ Persona '{persona_name}' already exists. Skipping creation.")
            return False

        try:
            os.makedirs(instructions_dir, exist_ok=True)
            with open(os.path.join(persona_dir, "persona.json"), "w", encoding="utf-8") as f:
                json.dump(template, f, indent=2)
            logger.info(f"✅ Saved persona.json for '{persona_name}'")

            if instructions:
                for fname, content in instructions.items():
                    out_path = os.path.join(instructions_dir, f"{fname}.txt")
                    with open(out_path, "w", encoding="utf-8") as f:
                        f.write(content)
                    logger.info(f"📝 Saved instruction: {fname}.txt")

            if self.vectordb:
                summary_text = self._summarize_persona(template)
                self._embed_to_vectordb(summary_text, persona_name)

            return True
        except Exception as e:
            logger.error(f"❌ Failed to create persona '{persona_name}': {e}", exc_info=True)
            return False

    def _summarize_persona(self, template: Dict) -> str:
        """
        Create a simple string summary of persona traits for embedding.

        Args:
            template (dict): Persona profile dictionary

        Returns:
            str: Summary block for indexing
        """
        name = template.get("name", "unknown")
        tone = template.get("tone", "default")
        context = template.get("context", "unspecified")
        traits = template.get("traits", {})
        return f"Persona: {name}\nTone: {tone}\nContext: {context}\nTraits: {json.dumps(traits, indent=2)}"

    def _embed_to_vectordb(self, summary_text: str, tag: str) -> None:
        """
        Embed a short summary of the persona into the vector store.

        Args:
            summary_text: Summary of persona attributes
            tag: Identifier (usually persona name)
        """
        try:
            self.vectordb.add_documents([{
                "text": summary_text,
                "metadata": {
                    "tier": "3_personas",
                    "tag": tag,
                    "type": "persona_definition",
                    "created": datetime.now().isoformat()
                }
            }])
            logger.info(f"📚 Embedded persona '{tag}' into vector store")
        except Exception as e:
            logger.error(f"❌ Failed to embed persona '{tag}': {e}")
