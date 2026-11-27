"""d3px.4: both minds read a message from their own hidden states; agreement
routes, a split takes the careful route."""
import numpy as np

from gaia_core.cognition.nlu import reading as R


def _r(mind, act, source, depth):
    return {"mind": mind, "act": (act, 0.9), "source": (source, 0.9), "depth": (depth, 0.9)}


def test_agreement_routes():
    got = R.combine([_r("core_gpu", "answer", "memory", "quick"), _r("prime_cpu", "answer", "memory", "quick")])
    assert got["act"] == "answer" and got["sources"] == ["memory"] and got["depth"] == "quick"
    assert all(got["agreed"].values())


def test_a_split_takes_the_careful_route():
    got = R.combine([_r("core_gpu", "talk", "own", "quick"), _r("prime_cpu", "answer", "web", "think")])
    assert got["act"] == "answer" and got["sources"] == ["own", "web"] and got["depth"] == "think"
    assert not any(got["agreed"].values())


def test_a_tool_only_when_both_minds_read_do():
    assert R.combine([_r("a", "do", "live", "quick"), _r("b", "answer", "live", "quick")])["act"] == "answer"
    assert R.combine([_r("a", "do", "live", "quick"), _r("b", "do", "live", "quick")])["act"] == "do"
    assert R.combine([_r("a", "do", "live", "quick")])["act"] == "answer"   # one mind alone
    assert R.combine([_r("a", "talk", "own", "quick")])["depth"] == "think"


def test_apply_probe_softmax_and_binary():
    probe = {"layers": np.array([1]),
             "act_mean": np.zeros(2), "act_scale": np.ones(2), "act_bias": np.zeros(2),
             "act_coef": np.array([[1.0, 0.0], [0.0, 1.0]]), "act_classes": np.array(["talk", "answer"]),
             "source_mean": np.zeros(2), "source_scale": np.ones(2), "source_bias": np.zeros(2),
             "source_coef": np.array([[2.0, 0.0], [0.0, 0.0]]), "source_classes": np.array(["own", "web"]),
             "depth_mean": np.zeros(2), "depth_scale": np.ones(2), "depth_bias": np.zeros(1),
             "depth_coef": np.array([[0.0, 3.0]]), "depth_classes": np.array(["quick", "think"])}
    got = R.apply_probe(probe, {1: [0.0, 2.0]})
    assert got["act"][0] == "answer" and got["source"][0] in ("own", "web") and got["depth"][0] == "think"
    assert R.apply_probe(probe, {1: [0.0, -2.0]})["depth"][0] == "quick"


def test_no_matching_probe_means_no_vote(monkeypatch, tmp_path):
    monkeypatch.setattr(R, "PROBE_DIR", str(tmp_path))
    monkeypatch.setattr(R, "_probes", {})
    monkeypatch.setattr(R, "_where", lambda mind: {"tag": f"{mind}_gpu", "model": "/m"})
    assert R.read_message("Hello GAIA") is None
