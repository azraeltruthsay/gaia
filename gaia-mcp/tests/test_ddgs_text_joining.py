"""0vuz: ddgs glued words at the search engine's <b>highlights</b>
("Excaliburis the sword ofKingArthur"); web_tools patches the extractor."""

import pytest

ddgs_base = pytest.importorskip("ddgs.base")
ddgs_results = pytest.importorskip("ddgs.results")

from gaia_mcp import web_tools  # noqa: E402

HTML = """<html><body><ul>
<li><a href="https://en.wikipedia.org/wiki/Excalibur">Excalibur</a>
<p><b>Excalibur</b> is the legendary sword of <b>King Arthur</b>, given by the Lady of the Lake.</p></li>
</ul></body></html>"""


class _Engine(ddgs_base.BaseSearchEngine):
    name = "test"
    category = "text"
    provider = "test"
    search_url = "https://example.invalid"
    items_xpath = "//li"
    elements_xpath = {"body": ".//p//text()"}
    result_type = ddgs_results.TextResult

    def build_payload(self, *a, **k):
        return {}


def test_highlighted_terms_keep_their_spaces():
    web_tools._ddgs_patched = False
    web_tools._patch_ddgs_text_joining()
    body = _Engine().extract_results(HTML)[0].body
    assert body == "Excalibur is the legendary sword of King Arthur, given by the Lady of the Lake."


def test_page_title_falls_back_to_og_title_and_h1():
    # ecnc: httpbin /html has no <title>; its name is the <h1>.
    from gaia_mcp.web_tools import _extract_title_bs4, _title_regex
    page = "<html><body><h1>Herman Melville - Moby-Dick</h1><p>Availing himself...</p></body></html>"
    assert _extract_title_bs4(page) == "Herman Melville - Moby-Dick"
    assert _title_regex(page) == "Herman Melville - Moby-Dick"
    og = '<html><head><meta property="og:title" content="OG Name"></head><body><h1>H</h1></body></html>'
    assert _extract_title_bs4(og) == "OG Name"
    titled = "<html><head><title> Real </title></head><body><h1>H</h1></body></html>"
    assert _extract_title_bs4(titled) == "Real" and _title_regex(titled) == "Real"
    assert _extract_title_bs4("<p>nothing</p>") == ""
