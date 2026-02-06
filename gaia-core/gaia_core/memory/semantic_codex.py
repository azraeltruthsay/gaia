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
            if path.suffix.lower() == ".json":
                with open(path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            else:
                if yaml is None:
                    return
                with open(path, "r", encoding="utf-8") as f:
                    payload = yaml.safe_load(f)
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