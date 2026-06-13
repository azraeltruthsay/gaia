"""Tests for the Locality organ (place + live environment).

All tests run offline — the only network path (weather) is exercised by
monkeypatching urlopen, never by hitting open-meteo.
"""

import json
from datetime import datetime, timezone

import gaia_common.utils.locality as loc


# ── season ────────────────────────────────────────────────────────────────

class TestSeason:
    def test_northern_winter(self):
        dt = datetime(2026, 2, 1, tzinfo=timezone.utc)
        assert loc._season(dt, lat=34.0) == "winter"

    def test_northern_summer(self):
        dt = datetime(2026, 7, 1, tzinfo=timezone.utc)
        assert loc._season(dt, lat=34.0) == "summer"

    def test_southern_hemisphere_flips(self):
        dt = datetime(2026, 7, 1, tzinfo=timezone.utc)  # northern summer
        assert loc._season(dt, lat=-34.0) == "winter"

    def test_no_latitude_assumes_northern(self):
        dt = datetime(2026, 4, 1, tzinfo=timezone.utc)
        assert loc._season(dt, lat=None) == "spring"


# ── city derivation ─────────────────────────────────────────────────────────

class TestLocaleCity:
    def test_explicit_city_wins(self, monkeypatch):
        monkeypatch.setenv("GAIA_LOCALE_CITY", "the Shire")
        monkeypatch.setenv("GAIA_USER_TZ", "America/Los_Angeles")
        assert loc._locale_city() == "the Shire"

    def test_derives_from_tz(self, monkeypatch):
        monkeypatch.delenv("GAIA_LOCALE_CITY", raising=False)
        monkeypatch.setenv("GAIA_USER_TZ", "America/Los_Angeles")
        assert loc._locale_city() == "the Los Angeles area"

    def test_empty_when_unconfigured(self, monkeypatch):
        monkeypatch.delenv("GAIA_LOCALE_CITY", raising=False)
        monkeypatch.delenv("GAIA_USER_TZ", raising=False)
        assert loc._locale_city() == ""


# ── block assembly (offline) ─────────────────────────────────────────────────

class TestBuildLocalityBlock:
    def test_place_and_environment_no_weather(self, monkeypatch):
        monkeypatch.setenv("GAIA_LOCALE_CITY", "the Los Angeles area")
        monkeypatch.setenv("GAIA_OPERATOR_NAME", "Azrael")
        monkeypatch.delenv("GAIA_LOCALE_LAT", raising=False)
        monkeypatch.delenv("GAIA_LOCALE_LON", raising=False)

        block = loc.build_locality_block()
        assert "Los Angeles area" in block
        assert "Azrael is nearby" in block
        assert "Out there now" in block
        # daylight word is always present even without weather
        assert ("daytime" in block) or ("after dark" in block)

    def test_no_network_call_without_latlon(self, monkeypatch):
        """build must not touch urlopen when lat/lon are unset."""
        def boom(*a, **k):
            raise AssertionError("network was called without lat/lon")
        monkeypatch.setattr(loc, "urlopen", boom)
        monkeypatch.delenv("GAIA_LOCALE_LAT", raising=False)
        monkeypatch.delenv("GAIA_LOCALE_LON", raising=False)
        loc.build_locality_block()  # must not raise

    def test_weather_disabled_flag(self, monkeypatch):
        def boom(*a, **k):
            raise AssertionError("weather fetched despite GAIA_LOCALE_WEATHER=0")
        monkeypatch.setattr(loc, "urlopen", boom)
        monkeypatch.setenv("GAIA_LOCALE_LAT", "34.05")
        monkeypatch.setenv("GAIA_LOCALE_LON", "-118.24")
        monkeypatch.setenv("GAIA_LOCALE_WEATHER", "0")
        loc.build_locality_block()  # must not raise


# ── weather fetch + cache (mocked network) ──────────────────────────────────

class _FakeResp:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class TestWeather:
    def setup_method(self):
        # reset the module-level cache so tests don't bleed into each other
        loc._weather_cache.update(ts=0.0, key="", data=None)

    def test_fetch_parses_and_renders(self, monkeypatch):
        payload = {"current": {"temperature_2m": 18.4, "weather_code": 0, "is_day": 1}}
        monkeypatch.setattr(loc, "urlopen", lambda *a, **k: _FakeResp(payload))
        monkeypatch.setenv("GAIA_LOCALE_LAT", "34.05")
        monkeypatch.setenv("GAIA_LOCALE_LON", "-118.24")
        monkeypatch.delenv("GAIA_LOCALE_WEATHER", raising=False)

        block = loc.build_locality_block()
        assert "clear sky" in block
        assert "18°C" in block      # rounded
        assert "daytime" in block   # is_day=1

    def test_cache_avoids_second_network_call(self, monkeypatch):
        calls = {"n": 0}
        payload = {"current": {"temperature_2m": 10, "weather_code": 3, "is_day": 0}}

        def counting(*a, **k):
            calls["n"] += 1
            return _FakeResp(payload)

        monkeypatch.setattr(loc, "urlopen", counting)
        a = loc._fetch_weather(34.05, -118.24)
        b = loc._fetch_weather(34.05, -118.24)
        assert a == b
        assert calls["n"] == 1  # second call served from cache

    def test_fetch_failure_returns_none(self, monkeypatch):
        def boom(*a, **k):
            raise OSError("no network")
        monkeypatch.setattr(loc, "urlopen", boom)
        assert loc._fetch_weather(34.05, -118.24) is None

    def test_failure_is_negative_cached(self, monkeypatch):
        """A failed fetch must back off — not retry (and re-block) every call."""
        calls = {"n": 0}

        def boom(*a, **k):
            calls["n"] += 1
            raise OSError("no network")

        monkeypatch.setattr(loc, "urlopen", boom)
        assert loc._fetch_weather(34.05, -118.24) is None
        assert loc._fetch_weather(34.05, -118.24) is None
        assert calls["n"] == 1  # second call served from the negative cache
