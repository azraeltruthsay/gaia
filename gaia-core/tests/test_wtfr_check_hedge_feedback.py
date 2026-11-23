"""wtfr: the 'not sure' an answer check puts in place of her answer says
nothing about the pages she looked up, so lookup memory isn't told they
failed the question."""
from types import SimpleNamespace

from gaia_core.cognition import engineering_mode as em
from gaia_core.cognition.agent_core import AgentCore


def test_check_hedged():
    assert em.check_hedged(em.UNSURE) and em.check_hedged(em.UNSURE_PLAIN)
    assert em.check_hedged("[Prime]\n\n" + em.UNSURE_PLAIN)
    assert not em.check_hedged("I'm not sure, but the 2021 IBC is in effect per the SBCC timeline.")
    assert not em.check_hedged("")


def _packet():
    df = SimpleNamespace(key="web_grounding", value={"q": {
        "source": "knowledge_router", "query": "Which IBC year does Washington use?",
        "results": [{"memory_id": "m1", "url": "https://sbcc.wa.gov/", "snippet": "2021 IBC"}]}})
    return SimpleNamespace(content=SimpleNamespace(data_fields=[df]))


def test_feedback_skipped_after_a_check_hedge(monkeypatch):
    sent = []
    from gaia_core.utils import mcp_client
    monkeypatch.setattr(mcp_client, "call_jsonrpc", lambda m, p: sent.append(p) or {})
    started = []
    import threading
    monkeypatch.setattr(threading, "Thread", lambda target, **kw: SimpleNamespace(start=lambda: (started.append(1), target())))
    AgentCore._lookup_feedback(None, em.UNSURE_PLAIN, _packet())
    assert not sent and not started
    AgentCore._lookup_feedback(None, "Washington uses the 2021 IBC (SBCC).", _packet())
    assert len(sent) == 1 and sent[0]["question"].startswith("Which IBC")
