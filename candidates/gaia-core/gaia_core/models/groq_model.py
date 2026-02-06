"""
Groq API model wrapper for GAIA.

Groq provides free, fast inference on open-source models via their custom LPU hardware.
This wrapper provides an OpenAI-compatible interface for use as a fallback when local
GPU inference is unavailable.

Environment:
    GROQ_API_KEY: API key from console.groq.com (required)
    GROQ_MODEL: Model to use (default: llama-3.3-70b-versatile)
    GROQ_TIMEOUT: Request timeout in seconds (default: 60)
"""

import logging
import os
import time
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger("GAIA.Groq")

# Lazy import to avoid dependency issues if groq not installed
Groq = None


def _ensure_groq_imported():
    """Lazy import of groq SDK."""
    global Groq
    if Groq is None:
        try:
            from groq import Groq as _Groq
            Groq = _Groq
        except ImportError:
            raise RuntimeError(
                "groq package not installed. Install with: pip install groq"
            )


class GroqAPIModel:
    """
    Groq API wrapper providing create_chat_completion interface.

    Compatible with GAIA's model pool and can serve as a fallback for gpu_prime.

    Attributes:
        model_name: Groq model identifier (e.g., "llama-3.3-70b-versatile")
        api_key: Groq API key
        timeout: Request timeout in seconds
    """

    # Available models and their characteristics
    AVAILABLE_MODELS = {
        "llama-3.3-70b-versatile": {
            "context_window": 128000,
            "description": "Best quality, general purpose",
            "tokens_per_minute": 6000,
        },
        "llama-3.1-70b-versatile": {
            "context_window": 128000,
            "description": "Previous gen, still excellent",
            "tokens_per_minute": 6000,
        },
        "llama-3.1-8b-instant": {
            "context_window": 128000,
            "description": "Fast, good for simple tasks",
            "tokens_per_minute": 20000,
        },
        "mixtral-8x7b-32768": {
            "context_window": 32768,
            "description": "MoE model, good balance",
            "tokens_per_minute": 5000,
        },
        "gemma2-9b-it": {
            "context_window": 8192,
            "description": "Google Gemma, instruction-tuned",
            "tokens_per_minute": 15000,
        },
    }

    def __init__(
        self,
        model_name: str = None,
        api_key: str = None,
        timeout: int = None,
    ):
        _ensure_groq_imported()

        self.model_name = model_name or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.timeout = timeout or int(os.getenv("GROQ_TIMEOUT", "60"))

        if not self.api_key:
            raise RuntimeError(
                "Groq API key not configured. Set GROQ_API_KEY environment variable. "
                "Get a free key at https://console.groq.com"
            )

        if self.model_name not in self.AVAILABLE_MODELS:
            logger.warning(
                f"Model '{self.model_name}' not in known models. "
                f"Available: {list(self.AVAILABLE_MODELS.keys())}"
            )

        self.client = Groq(api_key=self.api_key, timeout=self.timeout)
        self._request_count = 0
        self._total_tokens = 0

        logger.info(f"GroqAPIModel initialized with model={self.model_name}")

    def create_chat_completion(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stream: bool = False,
        **kwargs,
    ) -> Dict[str, Any] | Generator[Dict[str, Any], None, None]:
        """
        Create a chat completion using Groq API.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-2.0)
            top_p: Nucleus sampling parameter
            stream: If True, return a generator yielding chunks
            **kwargs: Additional arguments (ignored for compatibility)

        Returns:
            Dict with 'choices' key containing the response, or generator if streaming

        Raises:
            RuntimeError: If API call fails
        """
        self._request_count += 1
        start_time = time.time()

        # Sanitize messages - Groq is strict about format
        clean_messages = self._sanitize_messages(messages)

        # Clamp parameters to valid ranges
        temperature = max(0.0, min(2.0, float(temperature)))
        top_p = max(0.0, min(1.0, float(top_p)))
        max_tokens = max(1, min(32768, int(max_tokens)))

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=clean_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=stream,
            )

            duration = time.time() - start_time

            if stream:
                return self._stream_response(response, duration)

            # Extract content from response
            content = response.choices[0].message.content
            if content is None:
                content = ""

            # Log usage stats
            self._log_usage(response, duration)

            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                    "finish_reason": response.choices[0].finish_reason,
                }],
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                "model": self.model_name,
                "provider": "groq",
            }

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Groq API error after {duration:.2f}s: {e}")
            raise RuntimeError(f"Groq API call failed: {e}") from e

    def _sanitize_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Clean messages for Groq API compatibility.

        Groq is strict about message format:
        - Only 'role' and 'content' keys allowed
        - Role must be 'system', 'user', or 'assistant'
        - Content must be string (not None)
        """
        clean = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Normalize role
            if role not in ("system", "user", "assistant"):
                role = "user"

            # Ensure content is string
            if content is None:
                content = ""
            elif not isinstance(content, str):
                content = str(content)

            # Skip empty messages (except system)
            if not content.strip() and role != "system":
                continue

            clean.append({"role": role, "content": content})

        # Groq requires at least one user message
        if not any(m["role"] == "user" for m in clean):
            clean.append({"role": "user", "content": "(continue)"})

        return clean

    def _stream_response(
        self,
        response_stream,
        start_duration: float
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Process streaming response and yield chunks.

        Yields dicts compatible with GAIA's streaming interface.
        """
        content_buffer = []

        for chunk in response_stream:
            delta = chunk.choices[0].delta
            if delta.content:
                content_buffer.append(delta.content)
                yield {
                    "choices": [{
                        "delta": {"content": delta.content},
                        "finish_reason": None,
                    }]
                }

        # Final chunk with complete content
        full_content = "".join(content_buffer)
        logger.debug(f"Groq stream complete: {len(full_content)} chars")

        yield {
            "choices": [{
                "message": {"role": "assistant", "content": full_content},
                "finish_reason": "stop",
            }],
            "provider": "groq",
        }

    def _log_usage(self, response, duration: float):
        """Log token usage and request stats."""
        try:
            usage = response.usage
            self._total_tokens += usage.total_tokens

            logger.info(
                f"Groq [{self.model_name}] - "
                f"Prompt: {usage.prompt_tokens}, "
                f"Completion: {usage.completion_tokens}, "
                f"Total: {usage.total_tokens}, "
                f"Duration: {duration:.2f}s, "
                f"Session total: {self._total_tokens} tokens"
            )
        except Exception as e:
            logger.debug(f"Could not log Groq usage: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Return usage statistics for this session."""
        return {
            "model": self.model_name,
            "request_count": self._request_count,
            "total_tokens": self._total_tokens,
        }

    @classmethod
    def list_models(cls) -> Dict[str, Dict]:
        """Return available models and their characteristics."""
        return cls.AVAILABLE_MODELS.copy()
