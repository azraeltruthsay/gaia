"""
Nano-Refiner engine using gaia-nano HTTP endpoint.

Provides high-speed text refinement (spelling, diarization, formatting)
by calling the gaia-nano llama-server over its OpenAI-compatible API.
"""

import logging

import httpx

logger = logging.getLogger("GAIA.Audio.Refiner")

SYSTEM_PROMPT = (
    "You are GAIA's Transcript Refiner. Clean up raw audio transcripts, "
    "correct proper names like 'Azrael', and perform semantic diarization. "
    "Be concise and return ONLY the refined text."
)


class RefinerEngine:
    def __init__(self, endpoint: str = "http://gaia-nano:8080"):
        self.endpoint = endpoint.rstrip("/")
        self._ready: bool = False

    def load(self):
        """Verify gaia-nano is reachable."""
        if self._ready:
            return

        health_url = f"{self.endpoint}/health"
        logger.info(f"Checking gaia-nano health at {health_url}")
        try:
            resp = httpx.get(health_url, timeout=5.0)
            resp.raise_for_status()
            self._ready = True
            logger.info("gaia-nano is reachable — Nano-Refiner ready")
        except Exception as e:
            logger.warning(f"gaia-nano not reachable at {health_url}: {e} (will retry on first refine)")

    def refine(self, prompt: str, max_tokens: int = 2048) -> str:
        """Run refinement via gaia-nano HTTP API."""
        url = f"{self.endpoint}/v1/chat/completions"
        try:
            resp = httpx.post(
                url,
                json={
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": max_tokens,
                    "temperature": 0.1,
                },
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.error(f"Refinement failed: {e}")
            return f"Error: {e}"

    def unload(self):
        """No-op — model lives in gaia-nano container."""
        self._ready = False
        logger.info("Nano-Refiner detached from gaia-nano")
