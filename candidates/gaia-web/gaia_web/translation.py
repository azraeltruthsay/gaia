"""
Translation client for LibreTranslate integration.

Provides async detect + translate with graceful degradation.
All methods return None on failure — never raise, never block the Discord message flow.
"""

import logging
import os
import re
from typing import Optional, Tuple

import httpx

logger = logging.getLogger("GAIA.Web.Translate")

TRANSLATE_ENDPOINT = os.getenv("TRANSLATE_ENDPOINT", "http://gaia-translate:5000")
_DETECT_TIMEOUT = 5.0
_TRANSLATE_TIMEOUT = 15.0
_MIN_CONFIDENCE = 0.65

# Unicode ranges for script-based language hinting when LibreTranslate detection fails
_SCRIPT_HINTS = {
    "ja": re.compile(r"[\u3040-\u309F\u30A0-\u30FF]"),       # Hiragana + Katakana
    "zh": re.compile(r"[\u4E00-\u9FFF]"),                     # CJK Unified Ideographs
    "ko": re.compile(r"[\uAC00-\uD7AF\u1100-\u11FF]"),       # Hangul
    "ar": re.compile(r"[\u0600-\u06FF\u0750-\u077F]"),        # Arabic
    "ru": re.compile(r"[\u0400-\u04FF]"),                     # Cyrillic
}

_client: Optional[httpx.AsyncClient] = None

LANG_NAMES = {
    "en": "English", "ja": "Japanese", "ko": "Korean", "zh": "Chinese",
    "es": "Spanish", "fr": "French", "de": "German",
    "pt": "Portuguese", "ru": "Russian", "ar": "Arabic",
}


async def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(base_url=TRANSLATE_ENDPOINT, timeout=15.0)
    return _client


async def detect_language(text: str) -> Optional[Tuple[str, float]]:
    """Detect language. Returns (lang_code, confidence) or None on failure."""
    try:
        client = await _get_client()
        resp = await client.post("/detect", json={"q": text}, timeout=_DETECT_TIMEOUT)
        resp.raise_for_status()
        results = resp.json()
        if results and len(results) > 0:
            top = results[0]
            return (top["language"], top["confidence"])
    except Exception as exc:
        logger.warning("Language detection failed: %s", exc)
    return None


async def translate(text: str, source: str, target: str) -> Optional[str]:
    """Translate text. Returns translated string or None on failure."""
    try:
        client = await _get_client()
        resp = await client.post("/translate", json={
            "q": text,
            "source": source,
            "target": target,
        }, timeout=_TRANSLATE_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        return data.get("translatedText")
    except Exception as exc:
        logger.warning("Translation failed (%s->%s): %s", source, target, exc)
    return None


def _hint_language_by_script(text: str) -> Optional[str]:
    """Detect language from Unicode script when LibreTranslate fails on non-Latin text."""
    # Japanese detection must precede Chinese (they share CJK ideographs)
    for lang in ("ja", "ko", "ar", "ru", "zh"):
        pattern = _SCRIPT_HINTS[lang]
        if pattern.search(text):
            return lang
    return None


async def detect_and_translate_to_english(text: str) -> Optional[Tuple[str, str, str]]:
    """Detect language; if non-English, translate to English.

    Returns (translated_text, source_lang, source_lang_name) or None if:
      - text is already English
      - detection failed or confidence too low
      - translation failed
    """
    if not text or len(text.strip()) < 3:
        return None

    detection = await detect_language(text)

    # LibreTranslate detection is unreliable for CJK/Arabic/Cyrillic.
    # Fall back to script-based hinting when confidence is low or detection fails.
    lang, confidence = detection if detection else ("en", 0.0)
    if lang == "en" or confidence < _MIN_CONFIDENCE:
        # Check character scripts as a fallback
        hinted_lang = _hint_language_by_script(text)
        if hinted_lang:
            lang = hinted_lang
            confidence = 90.0  # High confidence from script match
        else:
            return None

    translated = await translate(text, source=lang, target="en")
    if translated is None:
        return None

    lang_name = LANG_NAMES.get(lang, lang.upper())
    return (translated, lang, lang_name)


async def close() -> None:
    """Close the persistent HTTP client."""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None
