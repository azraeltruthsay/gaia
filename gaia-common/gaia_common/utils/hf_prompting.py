"""Helper utilities for building HF-compatible prompts and stop tokens.

This module provides a minimal templating layer so HF-style models can receive
compact prompts (e.g., short-answer templates) without changing the rest of
the pipeline.
"""
from typing import List, Dict, Any, Optional


DEFAULT_TEMPLATES = {
    "default": "{messages}",
    # Short-answer template: system instruction to be concise and answer directly
    "short_answer": (
        "You are GAIA-Prime. Answer concisely and directly. If the user asks for a "
        "factual question, reply with the brief factual answer only (one sentence).\n\n{messages}"
    ),
    # ChatML/im_start format - wraps content in <|im_start|>/<|im_end|> tokens
    # Use this for models trained on Qwen/ChatML format (including the Claude model)
    "chatml": None,  # Handled specially via _build_chatml
    "qwen_chat": None,  # Alias for chatml
    "im_start": None,  # Alias for chatml
    # raw passthrough - applies chatml tokens but preserves content structure
    # This is the recommended mode for GCP: content is preserved exactly,
    # but wrapped in the tokens the model needs to generate properly
    "raw": None,
    "none": None,
}


def build_hf_prompt(messages: List[Dict[str, Any]], template_name: str = "default", prompt_config: Optional[Dict] = None) -> str:
    """Convert a message list (role/content) into a single string using a template.

    messages: list of {role, content} or a raw string (for convenience)
    template_name: one of DEFAULT_TEMPLATES keys or a custom name in prompt_config
    prompt_config: optional dict with template overrides (e.g., from Config.prompt_config)

    Template modes:
    - "chatml", "qwen_chat", "im_start", "raw", "none", None, "":
      Use ChatML format with <|im_start|>/<|im_end|> tokens.
      This preserves your content exactly while wrapping it in the tokens
      the model needs to generate. Recommended for GCP packets.
    - "default": Simple concatenation with template wrapper
    - "short_answer": Adds concise instruction prefix
    - "simple": Legacy role: content format (no special tokens)
    """
    # Convenience: if messages is a raw string, wrap it in a user message
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    # ChatML-style templates: wrap in <|im_start|>/<|im_end|> tokens
    # This is the key fix: even "raw" mode needs these tokens for the model to work
    if template_name in (None, "", "raw", "none", "chatml", "qwen_chat", "im_start"):
        return _build_chatml(messages)

    # Legacy simple format (for debugging or models that don't need special tokens)
    if template_name == "simple":
        parts: List[str] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            parts.append(f"{role}: {content}")
        return "\n".join(parts)

    # Build raw messages concatenation for template substitution
    parts: List[str] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            parts.append(f"[SYSTEM]\n{content}\n")
        elif role == "assistant":
            parts.append(f"[ASSISTANT]\n{content}\n")
        else:
            parts.append(f"[USER]\n{content}\n")

    raw = "\n".join(parts).strip()

    # Resolve template
    template = DEFAULT_TEMPLATES.get(template_name)
    if not template and prompt_config:
        template = prompt_config.get("hf_templates", {}).get(template_name)
    if not template:
        template = DEFAULT_TEMPLATES["default"]

    return template.format(messages=raw)


def default_stop_tokens(prompt_config: Optional[Dict] = None) -> List[str]:
    """Return a prioritized list of stop tokens.

    If prompt_config provides 'stop_tokens', those are prepended.
    Includes ChatML end token for models using <|im_start|>/<|im_end|> format.

    Note: We intentionally do NOT include </think> as a stop token because:
    1. Models that use <think>...</think> for reasoning should complete their thought
    2. The actual response comes AFTER </think>
    3. We strip think tags in post-processing (output_router._strip_think_tags_robust)

    If the model gets stuck in think loops, that's a generation/prompt issue,
    not a stop token issue.
    """
    # ChatML end token should be first - it's the primary stop signal
    tokens = ["<|im_end|>", "<|endoftext|>", "\n\nUser:", "\n\nAssistant:"]
    if prompt_config:
        custom = prompt_config.get("stop_tokens") or prompt_config.get("hf_stop_tokens")
        if isinstance(custom, list):
            # put custom tokens first
            tokens = custom + tokens
    return tokens


def _build_chatml(messages: List[Dict[str, Any]]) -> str:
    """Render messages using ChatML format (<|im_start|>role ... <|im_end|>).

    This format is used by Qwen, the Claude-derived model, and many other
    instruction-tuned models. The content within each role block is preserved
    exactly as provided - this is critical for GCP packets.

    The model will only generate after seeing <|im_start|>assistant, so we
    append that as the final token to trigger generation.

    **Assistant prefill**: If the last message has role "assistant", it is
    treated as a generation prefix — the content is placed after
    ``<|im_start|>assistant\\n`` *without* a closing ``<|im_end|>``, so the
    model continues from that text rather than starting from scratch.  This
    steers small models into synthesis mode (e.g., "Based on the results,").
    """
    role_map = {
        "system": "system",
        "user": "user",
        "assistant": "assistant",
        "tool": "tool",
    }
    chunks: List[str] = []

    # Detect assistant prefill: last message is assistant → use as generation seed
    has_prefill = (
        messages
        and role_map.get(messages[-1].get("role", "").lower()) == "assistant"
    )
    main_messages = messages[:-1] if has_prefill else messages

    for msg in main_messages:
        role = role_map.get(msg.get("role", "user").lower(), "user")
        content = msg.get("content", "") or ""
        # Content is preserved exactly - no modifications
        chunks.append(f"<|im_start|>{role}\n{content}<|im_end|>")

    if has_prefill:
        # Open the assistant turn with the prefill content — no <|im_end|>
        # so the model continues generating from this point
        prefill = messages[-1].get("content", "") or ""
        chunks.append(f"<|im_start|>assistant\n{prefill}")
    else:
        # Append bare assistant prefix to trigger generation
        chunks.append("<|im_start|>assistant\n")
    return "\n".join(chunks)


# Alias for backwards compatibility
_build_qwen_chat = _build_chatml
