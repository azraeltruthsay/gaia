"""!gear1 / !gear2 / !gear in chat (Azrael, 2026-11-10): parse, set through
the orchestrator, word the reply; anything else is left to the conversation."""

from gaia_core.cognition import gear_switch as gs


def test_parse():
    assert gs.parse_command("!gear1") == ("gear", "1")
    assert gs.parse_command("!Gear2") == ("gear", "2")
    assert gs.parse_command("  !gear 1 ") == ("gear", "1")
    assert gs.parse_command("/gear2.") == ("gear", "2")
    assert gs.parse_command("!gear") == ("gear", None)
    for text in ("!gear3", "what gear are you in?", "!gears", "!gear1 please now", ""):
        assert gs.parse_command(text) is None


def test_set_shifts_and_replies(monkeypatch):
    calls = []

    def fake(method, path, body=None, timeout=10.0):
        calls.append((method, path, body))
        return {"ok": True, "gear": body["gear"], "shifting_to": "awake"}

    monkeypatch.setattr(gs, "_call", fake)
    reply = gs.handle_command("!gear1", set_by="chat:discord")
    assert calls == [("POST", "/lifecycle/resting_gear", {"gear": "1", "shift_now": True, "set_by": "chat:discord"})]
    assert reply.startswith("Gear 1: Core on the GPU") and "Shifting now." in reply


def test_report(monkeypatch):
    monkeypatch.setattr(gs, "_call", lambda m, p, b=None, timeout=10.0: {"gear": "2"})
    assert gs.handle_command("!gear").startswith("I rest in gear 2. Gear 2: Prime")


def test_not_a_command_is_none(monkeypatch):
    monkeypatch.setattr(gs, "_call", lambda *a, **k: (_ for _ in ()).throw(AssertionError("called")))
    assert gs.handle_command("tell me about gears") is None


def test_gearbox_down(monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("refused")
    monkeypatch.setattr(gs, "_call", boom)
    assert "couldn't reach the gearbox" in gs.handle_command("!gear2")
