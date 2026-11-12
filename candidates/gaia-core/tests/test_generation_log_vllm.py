"""Mission Control's live generation view went silent on 2026-11-04: turns no
longer passed through the two places that wrote the generation log. Every
call to Core or Prime goes through VLLMRemoteModel, which logs itself now."""
from gaia_core.models.vllm_remote_model import VLLMRemoteModel


class Log:
    def __init__(self):
        self.events = []

    def start_generation(self, model, role, phase):
        self.events.append(("start", role, phase))
        return "g1"

    def log_token(self, gen_id, text):
        self.events.append(("token", text))

    def end_generation(self, gen_id, tokens=None, elapsed_ms=None):
        self.events.append(("end",) if tokens is None and elapsed_ms is None else ("end", tokens, elapsed_ms))


def _model(monkeypatch, endpoint="http://localhost:8092", name="core"):
    m = VLLMRemoteModel.__new__(VLLMRemoteModel)
    m.endpoint, m.model_name = endpoint, name
    monkeypatch.setattr(VLLMRemoteModel, "_label_cache", {})
    log = Log()
    monkeypatch.setattr(VLLMRemoteModel, "_gen_logger", staticmethod(lambda: log))
    return m, log


def test_a_whole_reply_is_logged_with_its_thinking(monkeypatch):
    m, log = _model(monkeypatch)
    m._log_whole("Four.", "two plus two", 0.4, 2)
    assert log.events == [("start", "core", "reply"), ("token", "<think>two plus two</think>"),
                          ("token", "Four."), ("end", 2, 400)]


def test_a_stream_is_logged_as_it_goes_and_closed(monkeypatch):
    m, log = _model(monkeypatch, endpoint="http://gaia-prime:7777", name="/models/prime")
    chunks = [{"choices": [{"delta": {"content": "Fo"}}]}, {"choices": [{"delta": {"content": "ur"}}]},
              {"choices": [{"message": {"content": "Four"}}]}]
    assert list(m._logged_stream(iter(chunks))) == chunks
    assert log.events == [("start", "prime", "stream"), ("token", "Fo"), ("token", "ur"), ("end",)]


def test_the_pool_does_not_log_it_twice():
    assert VLLMRemoteModel.logs_generation is True


def test_the_label_is_what_the_engine_runs(monkeypatch):
    m, _ = _model(monkeypatch, name="Gemma4-E4B-GAIA-Core-v1")
    import gaia_common.utils.self_model as sm
    monkeypatch.setattr(sm, "minds", lambda: {"Core": {"weights": "Gemma4-12B-GAIA-Core-v2-Q6_K.gguf"}})
    assert m._gen_label() == "Gemma4-12B-GAIA-Core-v2-Q6_K"
    monkeypatch.setattr(sm, "minds", lambda: (_ for _ in ()).throw(OSError("down")))
    monkeypatch.setattr(VLLMRemoteModel, "_label_cache", {})
    assert m._gen_label() == "Gemma4-E4B-GAIA-Core-v1"          # engines unreadable: the configured name


def test_window_follows_the_gear_23st(monkeypatch):
    """On the GPU the engine holds 8192; the configured 16384 is the CPU's."""
    m = VLLMRemoteModel.__new__(VLLMRemoteModel)
    m.endpoint, m._gpu_ctx_seen = "http://localhost:8092", (0.0, None)
    m.max_model_len = 16384
    monkeypatch.setenv("GGUF_CTX_SIZE_GPU", "8192")

    class R:
        def __init__(self, gpu): self.gpu = gpu
        def json(self): return {"has_gpu": self.gpu}

    class S:
        gpu = True
        def get(self, url, timeout=2): return R(S.gpu)

    m._session = S()
    assert m.max_model_len == 8192
    S.gpu = False
    m._gpu_ctx_seen = (0.0, None)
    assert m.max_model_len == 16384
    m.endpoint = "http://gaia-prime:7777"                 # not Core's engine
    assert m.max_model_len == 16384
