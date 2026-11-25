"""d55q: a search hit that writes a line quoted in her evidence says so, and
names the function to read; a quote in a docstring doesn't count."""
from gaia_core.cognition.issue_investigation import CodeView

SRC = '''import logging
logger = logging.getLogger(__name__)


def agree(a, b):
    return a == b


def check_general(question, answer, second):
    """Says 'her two minds disagree on' when they do."""
    if agree(answer, second):
        return answer, "agreed"
    logger.info("engineering mode: her two minds disagree on %r; not answering from memory", question[:60])
    return "unsure", "disagreed"
'''
EVIDENCE = ["2026-11-25T14:42:13 engineering mode: her two minds disagree on 'What is the SI unit of power?'; "
            "not answering from memory"]


def _repo(tmp_path):
    pkg = tmp_path / "candidates" / "gaia-core" / "gaia_core" / "cognition"
    pkg.mkdir(parents=True)
    (pkg / "engineering_mode.py").write_text(SRC)
    (tmp_path / "candidates" / "gaia-common" / "gaia_common").mkdir(parents=True)
    return tmp_path


def test_hit_that_writes_evidence_names_the_function(tmp_path):
    out = CodeView(_repo(tmp_path), evidence=EVIDENCE).search("two minds disagree")
    hints = [ln for ln in out.splitlines() if "quoted in your evidence" in ln]
    assert len(hints) == 1   # the log call, not the docstring
    assert "engineering_mode.py:13" in hints[0] and "in check_general()" in hints[0]
    assert "READ: gaia-core/gaia_core/cognition/engineering_mode.py:13" in hints[0]


def test_no_evidence_no_hint(tmp_path):
    assert "quoted in your evidence" not in CodeView(_repo(tmp_path)).search("two minds disagree")
