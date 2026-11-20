"""A plain request to add a site goes to Azrael's approval (2026-11-20)."""
from gaia_core.cognition import site_request as sr


def test_requested_domain():
    assert sr.requested_domain("Azrael: Can you add scribd.com to your web search allow list now?") == "scribd.com"
    assert sr.requested_domain("Andy: please whitelist https://www.dot.ca.gov/programs") == "dot.ca.gov"
    assert sr.requested_domain("Azrael: is scribd on your allowed list of websites?") is None
    assert sr.requested_domain("Nick: add some salt to the soup") is None
    assert sr.requested_domain("Azrael: I visited scribd.com yesterday") is None


def test_handle_parks_for_approval():
    calls = []

    def call(method, params):
        calls.append((method, params))
        return {"ok": False, "pending_approval": True, "action_id": "a1"}

    class SM:
        def set_session_meta(self, *a, **k): pass
    out = sr.handle("Azrael: Can you add scribd.com to your web search allow list now?", "s", SM(), call=call)
    assert calls[0][1]["action"] == "trust_domain" and calls[0][1]["domain"] == "scribd.com"
    assert "authenticator code" in out and "scribd.com" in out


def test_handle_reports_problem():
    out = sr.handle("Azrael: add en.wikipedia.org to your trusted sites", "s", None,
                    call=lambda m, p: {"ok": False, "error": "Approval request failed: 'web' can't run: en.wikipedia.org is already a site you can read."})
    assert out == "I can't add that one: en.wikipedia.org is already a site you can read."
