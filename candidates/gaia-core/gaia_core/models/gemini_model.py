import logging
import os
import time
import requests
from typing import List, Dict, Any


class GeminiAPIModel:
    """
    Minimal Gemini chat wrapper using the REST API.
    Expects GOOGLE_API_KEY in env. Model name is taken from config MODEL_CONFIGS entry.
    No streaming; returns the full content as a single message.
    """

    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("Gemini API key missing (set GOOGLE_API_KEY)")
        self.logger = logging.getLogger(__name__)
        self.endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"

    def create_chat_completion(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        stream: bool = False,
        **kwargs,
    ):
        if stream:
            raise NotImplementedError("Gemini streaming is not implemented in this minimal wrapper.")

        # Flatten messages to a single prompt; Gemini supports role/content pairs via "parts".
        contents = []
        for m in messages:
            text = m.get("content", "") or ""
            if not text:
                continue
            contents.append({"role": m.get("role", "user"), "parts": [{"text": text}]})

        body = {
            "contents": contents or [{"role": "user", "parts": [{"text": ""}]}],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": float(temperature),
                "topP": float(top_p),
            },
        }

        start = time.time()
        resp = requests.post(
            self.endpoint,
            params={"key": self.api_key},
            json=body,
            timeout=90,
        )
        duration = time.time() - start
        if resp.status_code != 200:
            raise RuntimeError(f"Gemini API error {resp.status_code}: {resp.text}")
        data = resp.json()
        try:
            candidates = data.get("candidates") or []
            content = candidates[0]["content"]["parts"][0]["text"]
        except Exception:
            raise RuntimeError(f"Gemini response missing content: {data}")

        self.logger.debug("Gemini request duration: %.2fs", duration)
        return {"choices": [{"message": {"content": content}}]}
