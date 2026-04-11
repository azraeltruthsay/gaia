"""Model-family-aware chat message formatting.

Lightweight formatter for use across GAIA services. Detects model family
from tokenizer special tokens and formats messages accordingly.

Supports:
- Qwen (ChatML): <|im_start|>role\n...<|im_end|>
- Gemma 4: <|turn>role<turn|>\n...
- Fallback: ChatML (safe default)

Usage:
    from gaia_common.utils.chat_format import ChatFormat

    fmt = ChatFormat.from_tokenizer(tokenizer)
    # or
    fmt = ChatFormat("gemma4")  # explicit family

    text = fmt.system("You are GAIA.")
    text = fmt.message("user", "Hello")
    text = fmt.assistant_prefix()
    text = fmt.conversation(messages)
"""

import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class ChatFormat:
    """Lightweight chat formatter — model-family-aware."""

    CHATML = "chatml"
    GEMMA4 = "gemma4"

    def __init__(self, family: str = CHATML):
        self.family = family

    @classmethod
    def from_tokenizer(cls, tokenizer) -> "ChatFormat":
        """Detect model family from tokenizer special tokens."""
        if getattr(tokenizer, "sot_token", None) == "<|turn>":
            return cls(cls.GEMMA4)
        # Check vocab for ChatML tokens
        vocab = getattr(tokenizer, "vocab", None) or {}
        if isinstance(vocab, dict) and "<|im_start|>" in vocab:
            return cls(cls.CHATML)
        try:
            ids = tokenizer.encode("<|im_start|>", add_special_tokens=False)
            if len(ids) == 1:
                return cls(cls.CHATML)
        except Exception:
            pass
        return cls(cls.CHATML)

    @classmethod
    def from_model_path(cls, model_path: str) -> "ChatFormat":
        """Detect model family from model path name."""
        path_lower = model_path.lower()
        if "gemma" in path_lower:
            return cls(cls.GEMMA4)
        return cls(cls.CHATML)

    def message(self, role: str, content: str) -> str:
        """Format a single message."""
        if self.family == self.GEMMA4:
            return f"<|turn>{role}<turn|>\n{content}"
        return f"<|im_start|>{role}\n{content}<|im_end|>"

    def system(self, content: str) -> str:
        """Format a system message."""
        return self.message("system", content)

    def assistant_prefix(self, enable_thinking: bool = True) -> str:
        """Return the assistant generation prefix."""
        if self.family == self.GEMMA4:
            prefix = "<|turn>assistant<turn|>\n"
            if not enable_thinking:
                prefix += "<|think|>\n\n<|think|>\n\n"
            return prefix
        prefix = "<|im_start|>assistant\n"
        if not enable_thinking:
            prefix += "<think>\n\n</think>\n\n"
        return prefix

    def conversation(self, messages: List[Dict[str, str]],
                     add_generation_prompt: bool = True,
                     enable_thinking: bool = True) -> str:
        """Format a full conversation."""
        parts = [self.message(m["role"], m.get("content", "")) for m in messages]
        if add_generation_prompt:
            parts.append(self.assistant_prefix(enable_thinking))
        return "\n".join(parts)

    @property
    def think_token(self) -> str:
        if self.family == self.GEMMA4:
            return "<|think|>"
        return "<think>"

    @property
    def think_close_token(self) -> str:
        if self.family == self.GEMMA4:
            return "<|think|>"
        return "</think>"

    def strip_special_tokens(self, text: str) -> str:
        """Remove chat format tokens from text."""
        if self.family == self.GEMMA4:
            import re
            text = re.sub(r"<\|turn>[a-z]+<turn\|>\n?", "", text)
            text = text.replace("<|think|>", "")
        else:
            text = text.replace("<|im_start|>", "").replace("<|im_end|>", "")
            text = text.replace("<think>", "").replace("</think>", "")
        return text.strip()

    @property
    def stop_tokens(self) -> List[str]:
        """Return stop token strings for generation."""
        if self.family == self.GEMMA4:
            return ["<turn|>", "<eos>"]
        return ["<|im_end|>", "<|endoftext|>"]
