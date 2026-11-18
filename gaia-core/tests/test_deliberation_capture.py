"""w6in/b9mp: deliberation prompts can be kept as sent, for teacher data."""
import json

from gaia_core.cognition.deliberation import _capture_prompt


def test_off_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("DELIBERATION_CAPTURE_DIR", raising=False)
    _capture_prompt([{"role": "user", "content": "hi"}], "core", None, True)
    assert not list(tmp_path.iterdir())


def test_captures_and_caps(tmp_path, monkeypatch):
    monkeypatch.setenv("DELIBERATION_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("DELIBERATION_CAPTURE_MAX", "2")
    for _ in range(4):
        _capture_prompt([{"role": "user", "content": "hi"}], "prime", "prime_thinking_v3", True)
    (f,) = tmp_path.iterdir()
    lines = [json.loads(x) for x in f.read_text().splitlines()]
    assert len(lines) == 2 and lines[0]["model_role"] == "prime" and lines[0]["messages"][0]["content"] == "hi"
