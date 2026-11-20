"""She asks to trust a site; it is added only once approved (2026-11-20)."""
import pytest

from gaia_mcp import web_tools as wt


@pytest.fixture(autouse=True)
def shared(tmp_path, monkeypatch):
    monkeypatch.setenv("SHARED_DIR", str(tmp_path))
    wt._trust_config = None
    yield
    wt._trust_config = None


def test_problems_checked_before_asking():
    assert "isn't a domain" in wt.trust_domain_problem({"domain": "not a domain", "reason": "x"})
    assert "needs a reason" in wt.trust_domain_problem({"domain": "codes.iccsafe.org"})
    assert "already" in wt.trust_domain_problem({"domain": "en.wikipedia.org", "reason": "x"})
    assert wt.trust_domain_problem({"domain": "https://www.dot.ca.gov/x", "reason": "Caltrans manuals"}) is None


def test_approved_domain_becomes_readable():
    trust, _, _ = wt._get_config()
    assert not trust.is_allowed("dot.ca.gov")
    out = wt.web_trust_domain({"domain": "https://www.dot.ca.gov/programs", "reason": "Caltrans design manual"})
    assert out["ok"] and out["domain"] == "dot.ca.gov"
    trust, _, _ = wt._get_config()
    assert trust.is_allowed("dot.ca.gov") and trust.tier_for_domain("dot.ca.gov") == "trusted"
    assert wt.approved_domains() == ["dot.ca.gov"]


def test_trust_domain_is_sensitive():
    from gaia_common.utils.domain_tools import is_sensitive
    assert is_sensitive("web", "trust_domain") and not is_sensitive("web", "fetch")
