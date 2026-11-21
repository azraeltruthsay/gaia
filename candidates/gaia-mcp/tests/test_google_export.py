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


FOLDER_HTML = ('<html><title>Welding</title><div class="flip-entry"><a href="https://drive.google.com/file/d/1O25nbEEbZC64V98sj5B6wTxnw7t9buvN/view?usp=drive_web">'
               '<div class="flip-entry-title">Blodgett - Design of Welded Structures.pdf</div></a></div></html>')


def test_drive_folder_lists_items(monkeypatch, tmp_path):
    resp = SimpleNamespace(text=FOLDER_HTML, raise_for_status=lambda: None)
    seen = _fake(monkeypatch, tmp_path, resp)
    (tmp_path / "web" / "trusted_added.json").write_text('{"domains": [{"domain": "drive.google.com"}]}')
    wt._trust_config = None
    out = wt.web_fetch({"url": "https://drive.google.com/drive/folders/1pj3Xt4UWDOaS6sTZUjogx_J0hkvdOdwB?usp=sharing"})
    assert "embeddedfolderview?id=1pj3Xt4UWDOaS6sTZUjogx_J0hkvdOdwB" in seen[0]
    assert out["ok"] and out["items"][0]["kind"] == "file"
    assert "File: Blodgett - Design of Welded Structures.pdf -> https://drive.google.com/file/d/1O25nbEEbZC64V98sj5B6wTxnw7t9buvN/view" in out["content"]


def test_long_document_find(monkeypatch, tmp_path):
    monkeypatch.setenv("SHARED_DIR", str(tmp_path))
    text = "\f".join([f"page {i} about beams" for i in range(1, 50)] + ["Problem 1 Determine the size of required fillet weld for the bracket shown in Figure 15"])
    out = wt._long_document("https://x/doc.pdf", "Blodgett", text, "Problem 1 fillet weld")
    assert "--- page 50 ---" in out and "Figure 15" in out
    assert "Nothing on any page" in wt._long_document("https://x/doc.pdf", "Blodgett", text, "torsion spring")
    assert wt._long_document("https://x/doc.pdf", "Blodgett", text, "").startswith("Blodgett: 50 pages, too long")
