"""Reading a file someone sends her: whole when it fits; a long one opens now
and is read in the background, with a follow-up only if she has one."""
from gaia_core.cognition import long_read as lr


def test_attached_text_prefers_the_full_file(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("the whole thing")
    assert lr.attached_text({"text_path": str(f), "text_preview": "the"}) == "the whole thing"
    assert lr.attached_text({"text_path": str(tmp_path / "gone"), "text_preview": "the"}) == "the"
    assert lr.attached_text({"text_preview": "p"}) == "p"


def test_short_file_is_read_whole_long_one_opens_and_says_so():
    short = lr.reading_block("a.txt", "Seal rules.\nRenewal every two years.")
    assert "Renewal every two years." in short and "still being read" not in short
    text = "\n".join(f"Paragraph {i}: " + "word " * 40 for i in range(60))
    block = lr.reading_block("big.pdf", text)
    assert "Paragraph 0" in block and "Paragraph 59" not in block
    assert "still being read" in block and "pages" in block
    assert len(block) < lr.OPENING_CHARS + 600


def test_opening_cuts_at_a_line_break():
    text = "\n".join("x" * 100 for _ in range(100))
    assert lr.opening(text).endswith("x") and len(lr.opening(text)) <= lr.OPENING_CHARS


class FakeCFR:
    def __init__(self, sections):
        self.sections = sections

    def ingest(self, path, doc_id=None):
        self.path, self.did = path, doc_id
        return {"ok": True, "section_count": len(self.sections)}

    def _load_tree(self, did):
        class S:
            def __init__(self, summary, text):
                self.summary, self.full_text = summary, text
        class D:
            sections = [S(s, t) for s, t in self.sections]
        return D()


SECTIONS = [("Definitions of terms.", "Full definitions."),
            ("Seals must show the certificate number.", "Full seal rules: number, name, state."),
            ("Fees for renewal.", "Full fee table.")]


def _run(tmp_path, monkeypatch, answers, sent_ok=True):
    monkeypatch.setattr(lr, "SOURCES", tmp_path)
    monkeypatch.setenv("SHARED_DIR", str(tmp_path))
    seen, posted, recorded = [], [], []
    def gen(messages):
        seen.append(messages)
        return answers.pop(0)
    out = lr.finish_reading("discord_channel_1", "wac.pdf", "x" * 9000, "Andy: What must a seal show?",
                            said=lambda: "I'm reading it now.", generate=gen,
                            record=recorded.append, cfr=FakeCFR(SECTIONS),
                            send=lambda sid, t: posted.append((sid, t)) or sent_ok)
    return out, seen, posted, recorded


def test_follow_up_reads_the_sections_she_picks_and_posts(tmp_path, monkeypatch):
    out, seen, posted, recorded = _run(tmp_path, monkeypatch,
                                       ["2, 9", "THOUGHT: Now I can answer him.\nSAY: A seal shows the certificate number, name and state."])
    assert posted == [("discord_channel_1", out)] and recorded == [out]
    assert out == "A seal shows the certificate number, name and state."
    final = seen[1][1]["content"]
    assert "Full seal rules" in final and "Full fee table" not in final     # 9 doesn't exist
    assert "I'm reading it now." in final and "Andy sent me" in final
    assert final.startswith("(My own thought") and "inner monologue" in seen[1][0]["content"]


def test_nothing_to_add_posts_nothing(tmp_path, monkeypatch):
    out, seen, posted, recorded = _run(tmp_path, monkeypatch, ["NONE", "THOUGHT: Nothing new.\nSAY: NOTHING"])
    assert out is None and posted == [] and recorded == []


def test_failed_post_is_not_recorded(tmp_path, monkeypatch):
    out, seen, posted, recorded = _run(tmp_path, monkeypatch, ["1", "THOUGHT: x\nSAY: Definitions of terms."], sent_ok=False)
    assert posted and recorded == []


def test_file_text_found_without_the_data_field(tmp_path):
    class Att:
        def __init__(self, name, mime, location):
            self.name, self.mime, self.location = name, mime, location
    pdf = tmp_path / "rules.pdf"
    pdf.write_bytes(b"%PDF")
    (tmp_path / "rules.pdf.txt").write_text("extracted rules")
    assert lr.file_text(Att("rules.pdf", "application/pdf", str(pdf))) == "extracted rules"
    txt = tmp_path / "notes.md"
    txt.write_text("plain notes")
    assert lr.file_text(Att("notes.md", "text/markdown", str(txt))) == "plain notes"
    assert lr.file_text(Att("x.pdf", "application/pdf", str(tmp_path / "missing.pdf"))) == ""


def test_follow_up_keeps_only_sentences_the_file_backs():
    src = "Modifying another licensee's work without notifying that licensee is misconduct."
    say = ("Modifying another licensee's work without notifying that licensee is misconduct. "
           "Given we're drafting a formal response, let's revise the letter tonight.")
    assert lr.backed_only(say, src) == "Modifying another licensee's work without notifying that licensee is misconduct."
    assert lr.backed_only("Let's revise the letter tonight together.", src) is None
    assert lr.backed_only(None, src) is None
