"""Her study notes reach her answer with the page they came from (oeql)."""
from gaia_core.cognition import knowledge_router as kr


class FakeVI:
    def __init__(self, hits):
        self.hits = hits

    def query(self, q, top_k=2):
        return self.hits


def _with(monkeypatch, hits):
    import gaia_common.utils.vector_indexer as vi
    monkeypatch.setattr(vi.VectorIndexer, "instance", classmethod(lambda cls, name: FakeVI(hits)))


def test_study_hit_carries_topic_page_and_body(tmp_path, monkeypatch):
    note = tmp_path / "wac.md"
    note.write_text("# Washington WAC 196-23: Stamping and seals\n\nSource: https://app.leg.wa.gov/x\n"
                    "Unit: Washington rules\n\n- A seal shows the certificate number.\n")
    _with(monkeypatch, [{"score": 0.6, "filename": str(note), "text": note.read_text()},
                        {"score": 0.3, "filename": str(note), "text": "weak"}])
    rs = kr._query_studies("What does a seal show?")
    assert len(rs) == 1 and rs[0].source == "study" and rs[0].url == "https://app.leg.wa.gov/x"
    assert rs[0].title.startswith("Washington WAC 196-23")
    assert rs[0].content == "- A seal shows the certificate number."


def test_prompt_cites_study_notes_by_topic_and_page():
    from gaia_common.protocols.cognition_packet import CognitionPacket, Content, DataField, Intent
    from gaia_core.utils.prompt_builder import build_from_packet
    q = "What does a Washington PE seal have to show?"
    pkt = CognitionPacket(content=Content(original_prompt=q), intent=Intent(user_intent="chat"))
    pkt.content.data_fields.append(DataField(key="web_grounding", type="json", source="knowledge_router", value={
        q[:50]: {"source": "knowledge_router:studies", "results": [
            {"title": "Washington WAC 196-23: Stamping and seals", "snippet": "- A seal shows the certificate number.",
             "url": "https://app.leg.wa.gov/x", "study": True, "trust": "looked_up"}]}}))
    text = "\n".join(m["content"] for m in build_from_packet(pkt))
    assert "(my study notes on Washington WAC 196-23: Stamping and seals, from https://app.leg.wa.gov/x)" in text
    from gaia_core.utils.prompt_builder import STUDY_NOTES_HEADER
    assert STUDY_NOTES_HEADER in text and "Candidate context" not in text


def test_strong_study_note_saves_the_web_and_is_named_in_her_trace():
    from gaia_core.cognition import inner_voice
    strong = kr.GroundingResult(source="study", trust_tier="looked_up", content="x" * 80, score=0.6)
    weak = kr.GroundingResult(source="study", trust_tier="looked_up", content="x" * 80, score=0.46)
    assert kr._local_answers([strong], "What does a seal show?")
    assert not kr._local_answers([weak], "What does a seal show?")
    assert "read my study notes" in inner_voice.trace(model="prime", sources=["study"])
