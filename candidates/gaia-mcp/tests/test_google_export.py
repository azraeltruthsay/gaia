"""Shared Google Docs/Drive links are read through their plain export (2026-11-20)."""
from types import SimpleNamespace

from gaia_mcp import web_tools as wt

DOC = "https://docs.google.com/document/d/1AbCdEfGhIjKlMnOpQrStUvWxYz0123456789/edit?usp=sharing"


def test_export_urls():
    assert wt.google_export_url(DOC) == "https://docs.google.com/document/d/1AbCdEfGhIjKlMnOpQrStUvWxYz0123456789/export?format=txt"
    assert wt.google_export_url("https://drive.google.com/file/d/1xYz0123456789AbCdEf/view?usp=sharing") == \
        "https://drive.google.com/uc?export=download&id=1xYz0123456789AbCdEf"
    assert wt.google_export_url("https://drive.google.com/open?id=1xYz0123456789AbCdEf") == \
        "https://drive.google.com/uc?export=download&id=1xYz0123456789AbCdEf"
    assert wt.google_export_url("https://en.wikipedia.org/wiki/Weld").endswith("/Weld")


def _fake(monkeypatch, tmp_path, resp):
    monkeypatch.setenv("SHARED_DIR", str(tmp_path))
    (tmp_path / "web").mkdir()
    (tmp_path / "web" / "trusted_added.json").write_text('{"domains": [{"domain": "docs.google.com"}]}')
    wt._trust_config = None
    seen = []
    monkeypatch.setattr("requests.get", lambda url, **k: seen.append(url) or resp)
    monkeypatch.setattr(wt, "_remember_lookup", lambda *a, **k: None)
    return seen


def test_doc_read_as_plain_text(monkeypatch, tmp_path):
    resp = SimpleNamespace(content="﻿Problem 1\nDetermine the size of fillet weld for the bracket.".encode(),
                           headers={"Content-Type": "text/plain; charset=utf-8"}, url="https://docs.google.com/x",
                           raise_for_status=lambda: None)
    seen = _fake(monkeypatch, tmp_path, resp)
    out = wt.web_fetch({"url": DOC})
    assert seen[0].endswith("/export?format=txt") and out["ok"] and out["format"] == "text"
    assert out["title"] == "Problem 1" and "fillet weld" in out["content"]


def test_restricted_doc_says_so(monkeypatch, tmp_path):
    resp = SimpleNamespace(content=b"<html>Sign in</html>", headers={"Content-Type": "text/html"},
                           url="https://accounts.google.com/ServiceLogin?continue=...", raise_for_status=lambda: None)
    _fake(monkeypatch, tmp_path, resp)
    out = wt.web_fetch({"url": DOC})
    assert not out["ok"] and "isn't shared publicly" in out["error"]
