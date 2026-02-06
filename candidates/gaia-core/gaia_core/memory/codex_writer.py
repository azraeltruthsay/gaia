import logging
from pathlib import Path
from typing import List, Tuple, Optional

from gaia_core.memory.semantic_codex import SemanticCodex, CodexEntry
from gaia_core.config import Config
from gaia_common.protocols.cognition_packet import CognitionPacket, Model # Assuming CognitionPacket is accessible

logger = logging.getLogger("GAIA.CodexWriter")

class CodexWriter:
    def __init__(self, config: Config, semantic_codex: SemanticCodex):
        self.config = config
        self.semantic_codex = semantic_codex
        # For now, we'll keep the self-generated docs root here,
        # but it should ideally come from config.
        # Ensure this root is configured in semantic_codex's scan paths or it won't be loaded.
        self.self_generated_docs_root = Path(config.KNOWLEDGE_DIR) / "self_generated_docs"
        self.self_generated_docs_root.mkdir(parents=True, exist_ok=True)


    def document_information(
        self,
        packet: CognitionPacket,
        info_to_document: str,
        symbol: str,
        title: str,
        tags: Optional[List[str]] = None,
        llm_model: Optional[Model] = None, # LLM to use for summarization/refinement
    ) -> Optional[Path]:
        """
        Orchestrates the process of generating formal documentation from given information
        and saving it as a CodexEntry.

        Args:
            packet: The current CognitionPacket (for context, history, etc.).
            info_to_document: The raw information string to be documented.
            symbol: A unique identifier for the CodexEntry.
            title: A human-readable title for the documentation.
            tags: Optional list of tags for the entry.
            llm_model: The LLM to use for refining the information into a document body.

        Returns:
            The Path to the created Markdown file, or None if documentation failed.
        """
        logger.info(f"Attempting to document information for symbol: {symbol}")

        # Step 1: Refine the info_to_document into a formal body using LLM (if provided)
        document_body = info_to_document 
        if llm_model:
            logger.info("LLM model provided for refining document body.")
            try:
                document_body = self._refine_with_llm(info_to_document, llm_model, packet, symbol, title)
            except Exception as e:
                logger.warning(f"Failed to refine document body with LLM for '{symbol}': {e}. Using raw information.")
                document_body = info_to_document # Fallback

        # Step 2: Create a CodexEntry object
        entry = CodexEntry(
            symbol=symbol,
            title=title,
            body=document_body,
            tags=tuple(tags) if tags else (),
            version="1.0", # Default version for self-generated docs
            scope="gaia_generated" # Indicate this is self-generated
        )

        # Step 3: Write the CodexEntry using SemanticCodex
        try:
            file_path = self.semantic_codex.write_entry(entry)
            logger.info(f"Successfully documented '{symbol}' to {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Failed to write documentation for '{symbol}': {e}")
            return None

    def _refine_with_llm(
        self, raw_info: str, llm_model: Model, packet: CognitionPacket, symbol: str, title: str
    ) -> str:
        """
        Uses an LLM to refine raw information into a structured document body for a CodexEntry.
        """
        system_prompt = f"""You are an expert documentation specialist for GAIA. Your task is to transform raw information into a clear, concise, and structured Markdown document body.

The document is for a CodexEntry with the symbol '{symbol}' and title '{title}'.

Focus on extracting the core facts, concepts, and procedures. Organize the information logically with headings, bullet points, and code blocks where appropriate.

Rules:
- Start directly with the content of the document body. Do NOT include YAML front matter or a title heading at the very top.
- Be concise but comprehensive.
- Use Markdown formatting extensively (headings, bold, italics, lists, code blocks).
- Ensure the language is formal, objective, and easy to understand for future GAIA instances or developers.
- If the raw information is sparse, state that clearly but still provide any key takeaways.
- Do NOT generate information not present in the information, unless it's a structural element like a common section heading (e.g., 'Overview', 'Purpose').

Raw Information to Document:
"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": raw_info}
        ]

        try:
            # Assuming llm_model has a get_chat_completion method that returns a response object
            response = llm_model.get_chat_completion(messages=messages, max_tokens=self.config.max_tokens_lite)
            # Extract content from the response, handling potential streaming or non-streaming formats
            if hasattr(response, 'choices') and response.choices:
                return response.choices[0].message.content.strip()
            else: # Fallback for other response types
                return str(response).strip()
        except Exception as e:
            logger.error(f"LLM refinement failed for symbol '{symbol}': {e}")
            return raw_info # Fallback to raw info if LLM fails
