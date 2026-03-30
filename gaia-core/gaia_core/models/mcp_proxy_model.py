"""MCP-backed model adapter.

Implements a minimal model-like interface expected by ModelPool consumers.
Delegates requests to the configured MCP-Lite server via JSON-RPC.
"""
from __future__ import annotations
import logging
import os
import requests
from typing import List, Dict

logger = logging.getLogger("GAIA.MCPProxyModel")


class MCPProxyModel:
    def __init__(self, config=None, role_name: str = "prime"):
        self.config = config
        self.role = role_name
        self.endpoint = None
        if config:
            self.endpoint = config.constants.get("MCP_LITE_ENDPOINT")
        if not self.endpoint:
            self.endpoint = os.environ.get("MCP_LITE_ENDPOINT")

    def _call_rpc(self, method: str, params: Dict) -> Dict:
        if not self.endpoint:
            raise RuntimeError("MCP endpoint not configured for MCPProxyModel")
        payload = {"jsonrpc": "2.0", "method": method, "params": params or {}, "id": f"mcpproxy-{self.role}"}
        logger.info(f"MCPProxyModel calling RPC {method} -> {self.endpoint}")
        r = requests.post(self.endpoint, json=payload, timeout=10)
        r.raise_for_status()
        return r.json()

    def create_chat_completion(self, messages: List[Dict], **kwargs) -> Dict:
        # Use a standard method name that the MCP server can map to a model-invoke tool
        params = {"messages": messages, "kwargs": kwargs, "role": self.role}
        resp = self._call_rpc("model_chat", params)
        # Expect result in resp['result'] with structure similar to other models
        if "result" in resp and isinstance(resp["result"], dict):
            return resp["result"]
        # Normalize older servers that return the inner dict directly
        return resp

    def create_completion(self, prompt: str, **kwargs) -> Dict:
        params = {"prompt": prompt, **kwargs, "role": self.role}
        resp = self._call_rpc("model_complete", params)
        if "result" in resp and isinstance(resp["result"], dict):
            return resp["result"]
        return resp

    def __repr__(self):
        return f"<MCPProxyModel role={self.role} endpoint={self.endpoint}>"
