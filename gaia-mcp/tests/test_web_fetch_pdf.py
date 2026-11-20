"""web_fetch reads a PDF on a trusted site as text (2026-11-20)."""
from types import SimpleNamespace

from gaia_mcp import web_tools as wt


def test_pdf_is_read_as_text(monkeypatch):
    text = "Design Manual M 22-01 Chapter 1260 Sight Distance. " + "Stopping sight distance is the sum " * 20
    resp = SimpleNamespace(content=b"%PDF-1.7 ...", headers={"Content-Type": "application/pdf"},
                           raise_for_status=lambda: None)
    monkeypatch.setattr("requests.get", lambda *a, **k: resp)
    monkeypatch.setattr(wt, "_pdf_text", lambda data: text)
    monkeypatch.setattr(wt, "_remember_lookup", lambda *a, **k: None)
    out = wt.web_fetch({"url": "https://wsdot.wa.gov/publications/manuals/fulltext/M22-01/1260.pdf"})
    assert out["ok"] and out["format"] == "pdf" and out["title"].startswith("Design Manual")


def test_unreadable_pdf_says_so(monkeypatch):
    resp = SimpleNamespace(content=b"%PDF", headers={"Content-Type": "application/pdf"}, raise_for_status=lambda: None)
    monkeypatch.setattr("requests.get", lambda *a, **k: resp)
    monkeypatch.setattr(wt, "_pdf_text", lambda data: "")
    out = wt.web_fetch({"url": "https://wsdot.wa.gov/x.pdf"})
    assert not out["ok"] and "PDF" in out["error"]
