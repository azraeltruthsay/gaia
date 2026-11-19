"""A note to herself ("THOUGHT_SEED: ...") never enters her history: several
reply paths keep raw text, and kept, she copies the line into every following
reply (2026-11-19, three Discord replies in a row)."""
from gaia_core.memory import session_manager as sm_mod


def _sm(tmp_path, monkeypatch):
    from gaia_core.config import get_config
    sm = sm_mod.SessionManager(get_config())
    monkeypatch.setattr(sm_mod, "SESSIONS_FILE", str(tmp_path / "s.json"), raising=False)
    monkeypatch.setattr(sm, "_save_state", lambda: None)
    monkeypatch.setattr(sm, "summarize_and_archive", lambda sid: None)
    return sm


def test_seed_line_cut(tmp_path, monkeypatch):
    sm = _sm(tmp_path, monkeypatch)
    sm.add_message("seed_t", "assistant", "THOUGHT_SEED: is it a joke?\n\nNick, it's yours.")
    assert sm.get_history("seed_t")[-1]["content"] == "Nick, it's yours."


def test_seed_only_reply_not_kept(tmp_path, monkeypatch):
    sm = _sm(tmp_path, monkeypatch)
    sm.add_message("seed_t2", "assistant", "*THOUGHT_SEED: nothing else*")
    assert not sm.get_history("seed_t2")


def test_plain_reply_untouched(tmp_path, monkeypatch):
    sm = _sm(tmp_path, monkeypatch)
    sm.add_message("seed_t3", "assistant", "A thought seed is a note.\nSecond line.")
    assert sm.get_history("seed_t3")[-1]["content"] == "A thought seed is a note.\nSecond line."


def test_goal_shift_line_cut(tmp_path, monkeypatch):
    sm = _sm(tmp_path, monkeypatch)
    sm.add_message("seed_t4", "assistant", "GOAL_SHIFT: clarify\nI'd like your last name for my notes.")
    assert sm.get_history("seed_t4")[-1]["content"] == "I'd like your last name for my notes."
