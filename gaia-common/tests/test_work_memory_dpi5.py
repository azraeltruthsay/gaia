"""dpi5: Ozymandias recited from the palace as 'Stand in the desart.[d]'."""
from gaia_common.utils.work_memory import WorkMemory, strip_footnotes


def test_footnote_markers_go():
    assert strip_footnotes("Stand in the desart.[d] Near them,") == "Stand in the desart. Near them,"
    assert strip_footnotes("king of kings[12] and[citation needed]") == "king of kings and"
    assert strip_footnotes("a list [a] stays, and [Chorus] stays") == "a list [a] stays, and [Chorus] stays"


def test_kept_and_recalled_clean(tmp_path):
    wm = WorkMemory(tmp_path)
    wm.remember("Ozymandias", "Two vast and trunkless legs of stone\nStand in the desart.[d]", [{"url": "https://x"}])
    assert "[d]" not in wm.recall("Ozymandias")["text"]
