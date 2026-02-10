from __future__ import annotations
import os, json, hashlib, threading, logging
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from pathlib import Path
try:
    import yaml  # optional; used only if available
except Exception:
    yaml = None
 
logger = logging.getLogger("GAIA.SemanticCodex")
 
@dataclass(frozen=True)
class CodexEntry:
    symbol: str
    title: str
    body: str
    tags: Tuple[str, ...] = ()
    version: str = "v1"
    scope: str = "global"  # "global" | project | session
 
class SemanticCodex:
    """
    Side-car memory for semantically compressed concepts.
    Loads JSON/YAML files from Config.KNOWLEDGE_CODEX_DIR into an in-memory index.
    """
    _instance = None
    _lock = threading.Lock()
 
    def __init__(self, config):
        self.config = config
        self.root = Path(config.KNOWLEDGE_CODEX_DIR)
        self.exts = set(x.lower() for x in config.CODEX_FILE_EXTS)
        self._index: Dict[str, CodexEntry] = {}
        self._checksums: Dict[str, str] = {}
        self._load_all()
 
    @classmethod
    def instance(cls, config):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(config)
            return cls._instance
 
    def write_entry(self, entry: CodexEntry) -> Path:
        """
        Writes a CodexEntry to a Markdown file with YAML front matter in the
        self-generated documents directory.
        """
        if not yaml:
            raise RuntimeError("PyYAML is not installed. Cannot write CodexEntry to Markdown.")

        # Ensure the self-generated docs directory exists
        self_generated_docs_path = self.root / "self_generated_docs"
        self_generated_docs_path.mkdir(parents=True, exist_ok=True)

        # Construct filename from symbol
        filename = f"{entry.symbol}.md"
        file_path = self_generated_docs_path / filename

        # Prepare YAML front matter
        front_matter = {
            "symbol": entry.symbol,
            "title": entry.title,
            "tags": list(entry.tags), # Convert tuple to list for YAML serialization
            "version": entry.version,
            "scope": entry.scope,
        }
        yaml_front_matter = yaml.dump(front_matter, sort_keys=False, default_flow_style=False)

        # Combine front matter and body
        content = f"""---
{yaml_front_matter}---

{entry.body.strip()}
"""

        # Write to file
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info(f"CodexEntry '{entry.symbol}' written to {file_path}")
            # After writing, trigger a hot reload to update the in-memory index
            self.hot_reload()
            return file_path
        except Exception as e:
            logger.error(f"Failed to write CodexEntry '{entry.symbol}' to {file_path}: {e}")
            raise
 
    def _iter_files(self):
        if not self.root.is_dir():
            return []
        for p in self.root.rglob("*"):
            if p.is_file() and p.suffix.lower() in self.exts:
                yield p
 
    def _checksum(self, path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            h.update(f.read())
        return h.hexdigest()
 
    def _load_one(self, path: Path):
        try:
            # Check for Markdown with YAML front matter
            if path.suffix.lower() == ".md":
                if not yaml:
                    logger.warning(f"PyYAML not available, skipping Markdown codex file {path}")
                    return

                content = path.read_text(encoding="utf-8")
                parts = content.split("---", 2) # Split into [before_first_---, yaml_block, markdown_body]

                if len(parts) >= 3 and not parts[0].strip(): # Ensure it starts with "---" and has a YAML block
                    try:
                        front_matter = yaml.safe_load(parts[1])
                        body = parts[2].strip()

                        if front_matter:
                            sym = front_matter.get("symbol")
                            title = front_matter.get("title", sym)
                            tags = tuple(front_matter.get("tags", []) or [])
                            version = front_matter.get("version", "v1")
                            scope = front_matter.get("scope", "global")

                            if sym and body:
                                entry = CodexEntry(
                                    symbol=sym,
                                    title=title,
                                    body=body,
                                    tags=tags,
                                    version=version,
                                    scope=scope,
                                )
                                self._index[sym] = entry
                                return # Successfully loaded Markdown, exit
                            else:
                                logger.warning(f"Markdown codex file {path} missing 'symbol' or 'body' in front matter.")
                        else:
                            logger.warning(f"Markdown codex file {path} has empty or invalid YAML front matter.")

                    except yaml.YAMLError as e:
                        logger.warning(f"Failed to parse YAML front matter in {path}: {e}")
                    except Exception as e:
                        logger.warning(f"Error processing Markdown codex file {path}: {e}")
                # If not valid Markdown with front matter, or parsing failed, fall through to try as JSON/YAML
                
            # Existing JSON/YAML parsing logic
            payload = None
            if path.suffix.lower() == ".json":
                with open(path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            elif path.suffix.lower() in self.exts: # Assume other extensions are for YAML as per original logic
                if yaml is None:
                    return # PyYAML not installed, skip other YAML files
                with open(path, "r", encoding="utf-8") as f:
                    payload = yaml.safe_load(f)
            else:
                return # Not a supported file type (e.g., .txt, or other unknown)

            # Rest of the original logic for processing payload
            if not payload:
                return
            entries = payload if isinstance(payload, list) else [payload]
            for obj in entries:
                sym = obj.get("symbol")
                body = obj.get("body")
                if not sym or not body:
                    continue
                entry = CodexEntry(
                    symbol=sym,
                    title=obj.get("title", sym),
                    body=body,
                    tags=tuple(obj.get("tags", []) or []),
                    version=obj.get("version", "v1"),
                    scope=obj.get("scope", "global"),
                )
                self._index[sym] = entry
        except Exception as e:
            logger.warning(f"Failed loading codex file {path}: {e}")
 
    def _load_all(self):
        self._index.clear()
        self._checksums.clear()
        for path in self._iter_files():
            self._load_one(path)
            self._checksums[str(path)] = self._checksum(path)
 
    def hot_reload(self) -> bool:
        if not self.config.CODEX_ALLOW_HOT_RELOAD:
            return False
 
        changed = False
        current_file_paths = {str(p) for p in self._iter_files()}
        known_file_paths = set(self._checksums.keys())
 
        # If files were added or removed, a full reload is the safest way
        # to ensure the index is consistent and entries from deleted files are purged.
        if current_file_paths != known_file_paths:
            logger.info("Codex file structure changed (added/removed files). Performing full reload.")
            self._load_all()
            return True  # A change definitely occurred
 
        # If the file list is identical, check for modifications in existing files.
        for raw_path, prev_checksum in self._checksums.items():
            p = Path(raw_path)
            current_checksum = self._checksum(p)
            if current_checksum != prev_checksum:
                logger.info(f"Detected change in codex file: {raw_path}")
                self._load_one(p)  # Reload the single modified file
                self._checksums[raw_path] = current_checksum
                changed = True
 
        return changed
 
    def get(self, symbol: str) -> Optional[CodexEntry]:
        return self._index.get(symbol)
 
    def search(self, query: str, limit: int = 10) -> List[CodexEntry]:
        q = (query or "").lower()
        results = [e for e in self._index.values() if (q in e.symbol.lower() or q in e.title.lower())]
        return results[:limit]