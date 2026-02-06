"""MemoryManager facade to unify short-term, session, and long-term memory.

Short-term: in-process dict (fast cache)
Working: SessionManager (persistent session history)
Long-term: VectorIndexer via MCP embedding_query (or direct VectorIndexer)
"""
from __future__ import annotations
import logging
from typing import Any, Dict, List
from gaia_core.memory.session_manager import SessionManager
from gaia_core.config import Config, get_config
from gaia_core.utils import mcp_client

logger = logging.getLogger("GAIA.MemoryManager")


class MemoryManager:
    _instance = None

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.short_term: Dict[str, Any] = {}
        # SessionManager expects config and optional llm; we don't pass llm here
        self.session_mgr = SessionManager(self.config)

    @classmethod
    def instance(cls, config: Config = None) -> 'MemoryManager':
        if cls._instance is None:
            cls._instance = MemoryManager(config=config)
        return cls._instance

    # Short-term cache API
    def set_short(self, key: str, value: Any):
        self.short_term[key] = value

    def get_short(self, key: str, default=None):
        return self.short_term.get(key, default)

    # Working/session API (wrap SessionManager)
    def add_message(self, session_id: str, role: str, content: str):
        self.session_mgr.add_message(session_id, role, content)

    def get_history(self, session_id: str):
        return self.session_mgr.get_history(session_id)

    # Long-term vector query API (delegates to mcp_client.embedding_query)
    def query_long(self, query: str, top_k: int = 5):
        res = mcp_client.embedding_query(query, top_k=top_k)
        if res.get("ok"):
            return res.get("results", [])
        logger.error(f"MemoryManager.query_long failed: {res.get('error')}")
        return []
