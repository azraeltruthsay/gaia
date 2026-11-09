"""a0aw.4: the self_review sleep task — she reads her turns against her
playbooks and what she writes is filed in her ledger."""

import json
from datetime import datetime, timezone

from gaia_common.utils import issue_ledger as L
from gaia_core.cognition import self_review as S

NOW = datetime(2026, 11, 8, 18, 0, tzinfo=timezone.utc)


def _l(ts, level, logger, msg):
    return f"2026-11-08T{ts}.000000+00:00 [gaia-core] {level}:{logger}:{msg}"


def _turn(ts, pid, text):
    return [
        _l(ts, "INFO", "GAIA.Core.API", f"Processing packet {pid}: '{text}' from web"),
        _l(ts, "INFO", "GAIA.IntentDetection", "Intent detection: done intent=brainstorming complexity=0.3 routing=TRIAGE"),
        _l(ts, "INFO", "GAIA.AgentCore", "Tool routing triggered for intent: brainstorming"),
        _l(ts, "INFO", "GAIA.AgentCore", "Tool selector determined no tool needed"),
    ]


SKILL = """---
name: ts-misrouted-request
domain: troubleshooting
status: active
seen_in: [turns]
---
# Misrouted request

## Symptoms
- tool routing fired on a request that needed no tool
"""


def _setup(tmp_path, n=3):
    lines = []
    for i in range(n):
        lines += _turn(f"16:{10 + i:02d}:00", f"web_{i}", f"Write a haiku number {i}...")
    log = tmp_path / "gaia-core.log"
    log.write_text("\n".join(lines) + "\n")
    hist = tmp_path / "hist"
    hist.mkdir()
    skills = tmp_path / "skills" / "ts-misrouted-request"
    skills.mkdir(parents=True)
    (skills / "SKILL.md").write_text(SKILL)
    issues = tmp_path / "issues"
    return dict(log_file=log, history_dir=hist, skills_dir=tmp_path / "skills", base=issues, now=NOW)


ANSWER = """<think>looking</think>
ISSUE: Haiku routed to tools
PLAYBOOK: ts-misrouted-request
EVIDENCE: tool routing fired (intent brainstorming)
SYMPTOM: A haiku request went through tool routing.
CAUSE: unknown
"""


def test_parse_findings_needs_title_and_evidence():
    found = S.parse_findings(ANSWER + "\nISSUE: no evidence here\nSYMPTOM: x\n")
    assert len(found) == 1
    f = found[0]
    assert f["playbook"] == "ts-misrouted-request" and f["cause"] == ""
    assert f["evidence"] == ["tool routing fired (intent brainstorming)"]
    assert S.parse_findings("NONE") == []
    bold = S.parse_findings("**ISSUE:** t\n- **EVIDENCE:** `a line`\n")
    assert bold[0]["evidence"] == ["a line"] and bold[0]["symptom"] == "t"


def test_review_files_what_she_found_and_advances(tmp_path):
    kw = _setup(tmp_path)
    seen = []

    def gen(system, user):
        seen.append(user)
        return ANSWER

    out = S.run_self_review(gen, **kw)
    assert out["batches"] == 1 and out["turns"] == 3
    assert len(out["filed"]) == 1 and not out["refused"]
    assert "ts-misrouted-request" in seen[0] and "ISSUE:" in seen[0]
    issue = L.get_issue(out["filed"][0], base=kw["base"])["issue"]
    assert issue["source"] == "self_review" and issue["evidence_verified"] == [True]
    review = json.loads((kw["base"] / S.REVIEWS).read_text().splitlines()[0])
    assert review["answer"].startswith("<think>") and review["filed"] == out["filed"]
    # nothing new since: no second call
    again = S.run_self_review(gen, **kw)
    assert again["batches"] == 0 and len(seen) == 1


def test_invented_evidence_is_refused(tmp_path):
    kw = _setup(tmp_path)
    out = S.run_self_review(lambda s, u: ANSWER.replace(
        "tool routing fired (intent brainstorming)", "generate_code was sent to nano at 0.596"), **kw)
    assert not out["filed"] and out["refused"]


def test_duplicate_becomes_note(tmp_path):
    kw = _setup(tmp_path)
    first = S.run_self_review(lambda s, u: ANSWER, **kw)
    (kw["log_file"]).write_text(kw["log_file"].read_text() + "\n".join(
        _turn("16:30:00", "web_9", "Write a haiku about frost...")) + "\n")
    second = S.run_self_review(lambda s, u: ANSWER.replace(
        "tool routing fired (intent brainstorming)", "selector: no tool needed"), **kw)
    assert second["noted"] == first["filed"] and not second["filed"]
    assert L.get_issue(first["filed"][0], base=kw["base"])["issue"]["notes"][0]["text"].startswith("Seen again")


def test_no_answer_keeps_cursor(tmp_path):
    kw = _setup(tmp_path)
    out = S.run_self_review(lambda s, u: "", **kw)
    assert out["batches"] == 0 and not (kw["base"] / S.STATE).exists()


def test_cap_per_run(tmp_path, monkeypatch):
    kw = _setup(tmp_path)
    monkeypatch.setattr(S, "MAX_FILED_PER_BATCH", 1)
    two = ANSWER + "\nISSUE: Second haiku misrouted badly\nEVIDENCE: tool routing fired (intent brainstorming)\n" \
                   "SYMPTOM: another distinct haiku problem with selector wording\n"
    out = S.run_self_review(lambda s, u: two, **kw)
    assert len(out["filed"]) == 1 and len(out["refused"]) == 1


def test_model_generator_falls_back_to_core():
    class Dead:
        def create_chat_completion(self, **kw):
            raise RuntimeError("engine is in standby")

    class Live:
        def create_chat_completion(self, **kw):
            return {"choices": [{"message": {"content": "<think>x</think>NONE"}}]}

    class Pool:
        def get_model_for_role(self, role):
            return Dead() if role == "prime" else Live()

    assert S.model_generator(Pool())("s", "u") == "NONE"


def test_unknown_playbook_dropped(tmp_path):
    kw = _setup(tmp_path)
    out = S.run_self_review(lambda s, u: ANSWER.replace("PLAYBOOK: ts-misrouted-request", "PLAYBOOK: ts-made-up"), **kw)
    assert L.get_issue(out["filed"][0], base=kw["base"])["issue"]["playbook"] == ""


def test_dash_joined_evidence_split():
    f = S.parse_findings('ISSUE: t\nEVIDENCE: "asked: hi" — WARNING: Vector index not found\n')[0]
    assert f["evidence"] == ["asked: hi", "WARNING: Vector index not found"]


def test_request_alone_is_not_evidence(tmp_path):
    kw = _setup(tmp_path)
    out = S.run_self_review(lambda s, u: ANSWER.replace(
        "EVIDENCE: tool routing fired (intent brainstorming)", 'EVIDENCE: "Write a haiku number 1..."'), **kw)
    assert not out["filed"] and "only the request" in out["refused"][0]["error"]
