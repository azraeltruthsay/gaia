"""The campaign's names go to gaia-core as hashes, never names (2026-11-17)."""

import gaia_mcp.kanka_tools as kt


class Pages:
    def get(self, path, params=None, cache_ttl=None):
        page = (params or {}).get("page", 1)
        data = {1: [{"name": "Leshen"}, {"name": "Dusk in Leshen"}], 2: [{"name": "The Hidden Traitor"}]}[page]
        return {"data": data, "links": {"next": "x" if page == 1 else None}}


def test_hashes_not_names(monkeypatch):
    monkeypatch.setattr(kt, "_get_client", lambda: Pages())
    r = kt.kanka_name_hashes({})
    assert r["ok"] and r["max_words"] == 3 and len(r["hashes"]) == 3
    assert "Traitor" not in str(r) and "Leshen" not in str(r)
    assert kt.name_hash("  dusk in   LESHEN ") in r["hashes"]
