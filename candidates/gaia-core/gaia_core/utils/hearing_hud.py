"""Her hearing sense, as gaia-core reads it: one line per clip, beside the audio.

gaia-audio's /hear measures a clip (length, loudness, separate sounds, tone or
noise, pitches and how they move) and names it with an AudioSet classifier.
Core's audio tower was trained on speech: asked about a dog, rain or a siren
it said "white noise" or "silence" (2026-11-17). The line is plain fact, put
before her question, so she can say what she hears and how she knows.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger("GAIA.HearingHUD")

AUDIO_ENDPOINT = os.environ.get("AUDIO_ENDPOINT", "http://gaia-audio:8080")
HEAR_TIMEOUT_S = float(os.environ.get("HEARING_TIMEOUT_S", "15"))


def hear(audio_base64: str, sample_rate: Optional[int] = None, labels: bool = True) -> Optional[dict]:
    """gaia-audio's /hear result, or None if it can't be reached."""
    if not audio_base64:
        return None
    try:
        import httpx
        from gaia_common.utils.service_auth import auth_headers
        r = httpx.post(f"{AUDIO_ENDPOINT}/hear",
                       json={"audio_base64": audio_base64, "sample_rate": sample_rate, "labels": labels},
                       headers=auth_headers("POST", "/hear"), timeout=HEAR_TIMEOUT_S)
        if r.status_code == 200:
            return r.json()
        logger.warning("Hearing: gaia-audio /hear returned %d", r.status_code)
    except Exception as e:
        logger.warning("Hearing: gaia-audio unreachable (%s)", e)
    return None


def hud_for(payload) -> str:
    """The line for one AudioPayload (object or dict); '' if none."""
    get = (payload.get if isinstance(payload, dict) else lambda k, d=None: getattr(payload, k, d))
    result = hear(get("data_base64", "") or "", get("sample_rate", None))
    return (result or {}).get("hud", "") or ""


def with_hud(prompt: str, huds) -> str:
    """The question with what her hearing measured before it. The battery
    builds the same text (cognitive_test_battery.py, query_engine_with_audio)."""
    lines = [h for h in (huds or []) if h]
    return "\n".join(lines) + "\n\n" + (prompt or "") if lines else (prompt or "")
