"""EmbedIntentClassifier.classify(conservative=...): a caller that only uses
the label to decide whether to SKIP something can ask for the raw best guess
even under the confidence threshold, instead of having it discarded to
"other" (found 2026-11-12: "identity" scored 0.293 against a 0.45 threshold
and got thrown away, so a self-introduction got web-grounded as an unknown
entity instead of being left alone)."""
import numpy as np
import pytest

from gaia_core.cognition.nlu.embed_intent_classifier import EmbedIntentClassifier


def _clf(query_vec):
    """A ready classifier with two fixed exemplars, no real model/encoding —
    classify() only ever reads _ready/_labels/_exemplar_matrix plus
    _embed_model.encode() for the query."""
    clf = EmbedIntentClassifier()
    clf._ready = True
    clf._other_penalty = 0.10
    clf._labels = ["identity", "other"]
    clf._exemplar_matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    class _FakeEmbedModel:
        def encode(self, texts, **kw):
            return np.array([query_vec], dtype=np.float32)
    clf._embed_model = _FakeEmbedModel()
    return clf


def test_default_discards_a_correct_under_threshold_guess():
    # query closer to "identity" (cos sim ~0.6) than to "other", but below a
    # strict threshold — today's (and every other caller's) behavior: thrown away.
    clf = _clf([0.6, 0.0])
    label, score = clf.classify("x", confidence_threshold=0.8)
    assert label == "other" and score == pytest.approx(0.6)


def test_conservative_false_keeps_the_same_guess():
    clf = _clf([0.6, 0.0])
    label, score = clf.classify("x", confidence_threshold=0.8, conservative=False)
    assert label == "identity" and score == pytest.approx(0.6)


def test_conservative_true_is_the_explicit_default_too():
    clf = _clf([0.6, 0.0])
    label, score = clf.classify("x", confidence_threshold=0.8, conservative=True)
    assert label == "other" and score == pytest.approx(0.6)


def test_above_threshold_is_unaffected_either_way():
    clf = _clf([0.9, 0.0])
    label, score = clf.classify("x", confidence_threshold=0.5)
    assert label == "identity" and score == pytest.approx(0.9)
    clf2 = _clf([0.9, 0.0])
    label2, score2 = clf2.classify("x", confidence_threshold=0.5, conservative=False)
    assert label2 == "identity" and score2 == pytest.approx(0.9)


def test_not_ready_is_unaffected_by_conservative():
    clf = EmbedIntentClassifier()
    assert clf.classify("x", conservative=False) == ("other", 0.0)
