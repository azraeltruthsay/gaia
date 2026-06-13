"""
Locality — GAIA's sense of place and live environment.

Produces a short, factual "where I am" snapshot for prompt injection, in the
same *declarative* register as the architecture-fact blocks (NOT behavioral
instruction — Gemma4-E4B disowns "you feel rooted here" prompting but accepts
plain facts about itself; see CLAUDE.md "affect is capacity, not content").

Two halves:
  * PLACE is stable — the workstation she runs on (her body), the city/region
    she lives in, the operator who is physically near the machine.
  * ENVIRONMENT is live — the season, whether it is light or dark out, and the
    current weather. Weather comes from open-meteo (keyless) when latitude and
    longitude are configured; everything degrades gracefully:
        no lat/lon or no network  -> season + a clock-based light/dark guess
        no GAIA_LOCALE_CITY        -> city derived from the GAIA_USER_TZ name

Configuration (all optional, env vars):
  GAIA_USER_TZ        IANA tz, e.g. "America/Los_Angeles" (also used elsewhere)
  GAIA_LOCALE_CITY    Human place name, e.g. "the Los Angeles area"
  GAIA_LOCALE_LAT     Latitude  (float) — enables live weather when paired w/ lon
  GAIA_LOCALE_LON     Longitude (float)
  GAIA_LOCALE_MACHINE Body description, default "this workstation (RTX 5080, 16 GB)"
  GAIA_OPERATOR_NAME  Operator's name, default "Azrael"
  GAIA_LOCALE_WEATHER "0" to force-disable the weather fetch even with lat/lon
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from urllib.request import urlopen

logger = logging.getLogger("GAIA.Locality")

# Weather is cached so we hit the network at most once per TTL, not per prompt.
# A *failed* fetch is negative-cached for a shorter window: build_locality_block
# runs on the user request path, so without this a sustained open-meteo outage
# would make every turn pay the urlopen timeout.
_WEATHER_TTL_S = 900       # cache a good fetch for 15 minutes
_WEATHER_FAIL_TTL_S = 120  # after a failure, wait 2 minutes before retrying
_weather_cache: Dict[str, Any] = {"ts": 0.0, "key": "", "data": None}

# WMO weather interpretation codes -> short human phrase.
# https://open-meteo.com/en/docs (weather_code)
_WMO = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "freezing fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    56: "freezing drizzle", 57: "freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain",
    66: "freezing rain", 67: "freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "rain showers", 81: "rain showers", 82: "violent rain showers",
    85: "snow showers", 86: "heavy snow showers",
    95: "thunderstorm", 96: "thunderstorm w/ hail", 99: "thunderstorm w/ hail",
}

_NORTHERN_SEASONS = [
    # (start_month, label) — meteorological seasons
    (12, "winter"), (3, "spring"), (6, "summer"), (9, "autumn"),
]


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or "").strip() or default


def _local_now() -> datetime:
    """Current time in the operator's timezone (UTC if GAIA_USER_TZ unset)."""
    now = datetime.now(timezone.utc)
    tz = _env("GAIA_USER_TZ")
    if tz:
        try:
            from zoneinfo import ZoneInfo
            return now.astimezone(ZoneInfo(tz))
        except Exception:
            logger.debug("Bad GAIA_USER_TZ %r; falling back to UTC", tz, exc_info=True)
    return now


def _locale_city() -> str:
    """Human place name. Explicit config wins; else derive from the tz name."""
    city = _env("GAIA_LOCALE_CITY")
    if city:
        return city
    tz = _env("GAIA_USER_TZ")
    if tz and "/" in tz:
        # "America/Los_Angeles" -> "the Los Angeles area"
        leaf = tz.rsplit("/", 1)[-1].replace("_", " ")
        return f"the {leaf} area"
    return ""


def _latlon() -> Optional[tuple]:
    lat, lon = _env("GAIA_LOCALE_LAT"), _env("GAIA_LOCALE_LON")
    if not lat or not lon:
        return None
    try:
        return (float(lat), float(lon))
    except ValueError:
        logger.debug("Bad GAIA_LOCALE_LAT/LON %r/%r", lat, lon)
        return None


def _season(dt_local: datetime, lat: Optional[float]) -> str:
    """Meteorological season; flips hemisphere when latitude is southern."""
    m = dt_local.month
    season = "winter"
    for start, label in _NORTHERN_SEASONS:
        if (start == 12 and m in (12, 1, 2)) or (start <= m < start + 3):
            season = label
            break
    if lat is not None and lat < 0:
        flip = {"winter": "summer", "summer": "winter",
                "spring": "autumn", "autumn": "spring"}
        season = flip[season]
    return season


def _fetch_weather(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    """Fetch current conditions from open-meteo, cached for _WEATHER_TTL_S.

    Returns {desc, temp_c, is_day} or None on any failure. Never raises.
    """
    key = f"{lat:.3f},{lon:.3f}"
    now = time.time()
    cache = _weather_cache
    if cache["key"] == key and cache["ts"]:
        age = now - cache["ts"]
        if cache["data"] is not None and age < _WEATHER_TTL_S:
            return cache["data"]            # fresh success
        if cache["data"] is None and age < _WEATHER_FAIL_TTL_S:
            return None                     # recent failure — back off, no refetch

    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&current=temperature_2m,weather_code,is_day"
    )
    try:
        with urlopen(url, timeout=3) as resp:  # noqa: S310 (fixed host)
            payload = json.loads(resp.read().decode("utf-8"))
        cur = payload.get("current") or {}
        data = {
            "desc": _WMO.get(int(cur.get("weather_code", -1)), ""),
            "temp_c": cur.get("temperature_2m"),
            "is_day": bool(cur.get("is_day", 1)),
        }
        cache.update(ts=now, key=key, data=data)
        return data
    except Exception:
        logger.debug("Weather fetch failed for %s", key, exc_info=True)
        # Negative-cache so we don't pay the urlopen timeout on every prompt
        # build during an outage; weather simply drops off until it recovers.
        cache.update(ts=now, key=key, data=None)
        return None


def locality_snapshot() -> Dict[str, Any]:
    """Structured locality data (for callers that want the parts, not prose)."""
    dt_local = _local_now()
    latlon = _latlon()
    lat = latlon[0] if latlon else None

    weather = None
    if latlon and _env("GAIA_LOCALE_WEATHER", "1") != "0":
        weather = _fetch_weather(*latlon)

    # Daylight: prefer the live is_day flag; else guess from the local hour.
    if weather is not None:
        is_day = weather["is_day"]
    else:
        is_day = 6 <= dt_local.hour < 20

    return {
        "machine": _env("GAIA_LOCALE_MACHINE", "this workstation (RTX 5080, 16 GB)"),
        "operator": _env("GAIA_OPERATOR_NAME", "Azrael"),
        "city": _locale_city(),
        "season": _season(dt_local, lat),
        "is_day": is_day,
        "weather": weather,
    }


def build_locality_block() -> str:
    """One short factual line about place + live environment, or "" if nothing.

    Example:
        You run on this workstation (RTX 5080, 16 GB) in the Los Angeles area;
        Azrael is nearby. Out there now: clear sky, 18°C, daytime, late spring.
    """
    snap = locality_snapshot()

    # — Place —
    place = f"You run on {snap['machine']}"
    if snap["city"]:
        place += f" in {snap['city']}"
    if snap["operator"]:
        place += f"; {snap['operator']} is nearby"
    place += "."

    # — Live environment —
    env_bits = []
    w = snap["weather"]
    if w:
        if w["desc"]:
            env_bits.append(w["desc"])
        if w["temp_c"] is not None:
            env_bits.append(f"{round(w['temp_c'])}°C")
    env_bits.append("daytime" if snap["is_day"] else "after dark")
    env_bits.append(snap["season"])

    if env_bits:
        return f"{place} Out there now: {', '.join(env_bits)}."
    return place
