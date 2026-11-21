"""Finding an item in a shared Drive folder (2026-11-21)."""
from gaia_core.cognition import drive_nav as dn

ROOT = "https://drive.google.com/drive/folders/1oar_Umo-Nez5WxGyJA9JlFTqPDIA8fSS"
WELD = "https://drive.google.com/drive/folders/1pj3Xt4UWDOaS6sTZUjogx_J0hkvdOdwB"
BOOK = "https://drive.google.com/file/d/1O25nbEEbZC64V98sj5B6wTxnw7t9buvN/view"


def fake_call(method, params, timeout=0):
    url = params["url"]
    if url == ROOT:
        return {"result": {"title": "References", "items": [
            {"name": "Text References", "url": "https://drive.google.com/drive/folders/1m0AtXzpyfU7-3e50YT3ZtJdCN8", "kind": "folder"},
            {"name": "Welding", "url": WELD, "kind": "folder"}]}}
    if url == WELD:
        return {"result": {"title": "Welding", "items": [
            {"name": "Blodgett - Design of Welded Structures.pdf", "url": BOOK, "kind": "file"}]}}
    if url == BOOK and params.get("find"):
        return {"result": {"ok": True, "content": "Blodgett: 835 pages; the pages that match:\n\n--- page 752 ---\nDetermine the size of required fillet weld for the bracket shown in Figure 15, to carry a load of 18,000 lbs."}}
    if url == BOOK:
        return {"result": {"ok": True, "content": "THE JAMES F. LINCOLN ARC WELDING FOUNDATION: 835 pages, too long to read at once."}}
    return {"result": {"items": []}}


HIST = [{"role": "user", "content": f"Andy: Here is a drive with references: {ROOT}?usp=sharing"}]


def test_opens_the_misspelled_book():
    out = dn.handle("Azrael: Please look in the folders for the Blogett document", HIST, call=fake_call)
    assert "Blodgett - Design of Welded Structures.pdf" in out and "835 pages" in out and "in Welding" in out


def test_lists_a_folder():
    assert dn.handle("Azrael: Look in Welding", HIST, call=fake_call) == \
        "In **Welding** there's: Blodgett - Design of Welded Structures.pdf."


def test_not_a_drive_request():
    assert dn.handle("Azrael: look at that sunset", [], call=fake_call) is None


def test_link_in_the_same_message():
    out = dn.handle(f"Azrael: Please look in the folders for the Blogett document {ROOT}?usp=drive_link", [], call=fake_call)
    assert "Blodgett - Design of Welded Structures.pdf" in out


def test_find_inside_the_opened_book(tmp_path, monkeypatch):
    monkeypatch.setenv("SHARED_DIR", str(tmp_path))
    dn.handle("Azrael: open the Blogett document", HIST, call=fake_call, session_id="s1")
    out = dn.handle("Azrael: find the problem with the 18,000 lbs bracket", HIST, call=fake_call, session_id="s1")
    assert out.startswith("From **Blodgett - Design of Welded Structures.pdf**, page 752") and "Figure 15" in out
