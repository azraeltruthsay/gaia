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
