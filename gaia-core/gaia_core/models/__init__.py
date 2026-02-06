"""
gaia_core.models - Model pool and LLM backend implementations.

This package provides:
- model_pool: Unified model pool interface (Prime/Lite/Embedding)
- model_manager: Model lifecycle management
- vllm_model: vLLM backend for GPU inference
- hf_model: HuggingFace Transformers backend
- gemini_model: Google Gemini API backend
- oracle_model: External API oracle backend
- mcp_proxy_model: MCP-proxied model backend
- dev_model: Development/testing model backend
- document: Document model definitions
"""

# Note: Explicit imports deferred until app.* dependencies are fully migrated.
# Once migration is complete, add convenience imports here.

__all__ = [
    "model_pool",
    "model_manager",
    "vllm_model",
    "hf_model",
    "gemini_model",
    "oracle_model",
    "mcp_proxy_model",
    "dev_model",
    "document",
]
