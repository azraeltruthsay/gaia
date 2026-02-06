import os
import logging
from datetime import datetime
from typing import List, Dict

from app.config import Config
from app.utils.code_analyzer.language_detector import detect_language
from app.utils.code_analyzer.docstring_extractor import extract_docstrings
from app.utils.code_analyzer.structure_extractor import extract_structure
from app.utils.code_analyzer.chunk_creator import create_chunks
from app.utils.code_analyzer.llm_analysis import summarize_chunks
from app.utils.code_analyzer.snapshot_manager import SnapshotManager
from app.utils.code_analyzer.file_loader import load_file_safely
from app.utils.code_analyzer.file_scanner import scan_code_directory

logger = logging.getLogger("GAIA.CodeAnalyzer")

class CodeAnalyzer:
    """
    Coordinates the code analysis pipeline.
    Tracks modified files, extracts structural data, summarizes with LLM,
    and prepares data for vector storage or knowledge tier elevation.
    """

    def __init__(self, config: Config, llm=None, doc_processor=None):
        self.config = config
        self.llm = llm
        self.doc_processor = doc_processor
        self.snapshot_manager = SnapshotManager(self.config)
        self.code_root = self.config.codebase_path or "/app"
        self.summary_output_path = self.config.system_reference_path("code_summaries")
        os.makedirs(self.summary_output_path, exist_ok=True)

    def refresh_code_tree(self, root_dir: str = None):
        """
        Rescan code files and refresh snapshots.
        """
        root = root_dir or self.code_root
        file_list = scan_code_directory(root)
        self.snapshot_manager.update_snapshot(file_list)

    def review_codebase(self):
        """
        Perform full reanalysis of changed files and write updated summaries.
        """
        logger.info("🧠 Starting codebase review cycle...")

        changed_files = self.snapshot_manager.get_modified_files()
        logger.info(f"🔍 {len(changed_files)} changed files to analyze.")

        for file_path in changed_files:
            try:
                abs_path = os.path.join(self.code_root, file_path)
                code = load_file_safely(abs_path)
                if not code:
                    continue

                language = detect_language(file_path, code)
                doc_info = extract_docstrings(code, language)
                structure = extract_structure(code, language)
                chunks = create_chunks(file_path, code, structure)

                summary = summarize_chunks(chunks, self.llm) if self.llm else "(No LLM summary)"

                summary_data = {
                    "file": file_path,
                    "language": language,
                    "docstrings": doc_info,
                    "structure": structure,
                    "summary": summary,
                    "analyzed_at": datetime.utcnow().isoformat()
                }

                output_file = os.path.join(self.summary_output_path, file_path.replace("/", "__") + ".json")
                os.makedirs(os.path.dirname(output_file), exist_ok=True)

                with open(output_file, "w", encoding="utf-8") as f:
                    import json
                    json.dump(summary_data, f, indent=2)

                logger.info(f"📄 Code summary saved: {file_path}")

            except Exception as e:
                logger.error(f"❌ Error analyzing {file_path}: {e}", exc_info=True)
