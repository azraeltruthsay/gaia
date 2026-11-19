"""The whole extracted text of an attachment is saved where gaia-core reads it;
the packet only carries a preview (2026-11-19)."""
import asyncio

from gaia_web import attachment_handler as ah


class FakeAttachment:
    def __init__(self, filename, data):
        self.filename, self.data, self.size = filename, data, len(data)
        self.content_type = "text/plain"

    async def save(self, path):
        with open(path, "wb") as f:
            f.write(self.data)


class FakeMessage:
    def __init__(self, atts):
        self.attachments = atts


def test_full_text_saved_beside_the_file(tmp_path, monkeypatch):
    monkeypatch.setattr(ah, "_ATTACHMENT_DIR", str(tmp_path))
    body = "rule\n" * 3000
    out = asyncio.run(ah.process_attachments(FakeMessage([FakeAttachment("wac.txt", body.encode())])))
    assert out[0]["text_path"].endswith(".txt.txt")
    assert open(out[0]["text_path"]).read() == out[0]["text_content"] and len(out[0]["text_content"]) > 2000
