"""LibreTranslate's confidence is 0-100; a 0-1 threshold let a 57% French guess
on an English DM through ("beads" -> "beats", 2026-11-24)."""
import asyncio

from gaia_web import translation as tr


def _run(detected, monkeypatch):
    async def fake_detect(text):
        return detected

    async def fake_translate(text, source, target):
        return "translated"
    monkeypatch.setattr(tr, "detect_language", fake_detect)
    monkeypatch.setattr(tr, "translate", fake_translate)
    return asyncio.run(tr.detect_and_translate_to_english("Can you see your development beads?"))


def test_weak_guess_on_english_is_left_alone(monkeypatch):
    assert _run(("fr", 0.57), monkeypatch) is None


def test_confident_detection_still_translates(monkeypatch):
    assert _run(("fr", 1.0), monkeypatch) == ("translated", "fr", tr.LANG_NAMES.get("fr", "FR"))


def test_detect_language_reports_zero_to_one(monkeypatch):
    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return [{"language": "fr", "confidence": 57.0}]

    class _Client:
        async def post(self, *a, **k):
            return _Resp()

    async def client():
        return _Client()
    monkeypatch.setattr(tr, "_get_client", client)
    assert asyncio.run(tr.detect_language("x")) == ("fr", 0.57)
