"""6afh: gaia-mcp's palace tools for answer feedback, pinning, restoring
and the lifecycle sweep, over a real palace in a temp dir."""

from unittest.mock import patch

import pytest

from gaia_common.utils.mempalace import MemPalace
from gaia_mcp import tools

LADY = {"title": "Lady of the Lake", "url": "https://en.wikipedia.org/wiki/Lady_of_the_Lake",
        "snippet": "The Lady of the Lake gives King Arthur the sword Excalibur."}
Q = "Where did King Arthur get Excalibur?"


@pytest.fixture
def palace(tmp_path):
    p = MemPalace({"root_dir": str(tmp_path / "palace"), "wings": {},
                   "kg_path": str(tmp_path / "kg.db")})
    with patch.object(tools, "_get_palace", return_value=p):
        yield p


def test_feedback_confirms_and_recall_reports_it(palace):
    rid = palace.lookups.remember(Q, [dict(LADY)])[0]
    out = tools._palace_lookup_feedback_impl(
        {"question": Q, "answer": "The Lady of the Lake gave it to him.", "sources": [{"memory_id": rid}]})
    assert out["ok"] and out["verdict"] == "supported" and out["records"] == 1
    hit = tools._palace_recall_lookup_impl({"query": Q})["results"][0]
    assert hit["confirmed_match"] >= 0.6


def test_feedback_validates_params(palace):
    assert not tools._palace_lookup_feedback_impl({"question": Q})["ok"]
    assert not tools._palace_pin_impl({})["ok"]
    assert not tools._palace_pin_impl({"ids": "abc"})["ok"]


def test_pin_sweep_restore(palace):
    rid = palace.lookups.remember(Q, [dict(LADY)])[0]
    assert tools._palace_pin_impl({"ids": [rid]})["lookups"] == 1
    out = tools._palace_lifecycle_sweep_impl({"dry_run": True})
    assert out["dry_run"] and out["lookups"]["pinned"] == 1 and out["lookups"]["archived"] == []
    assert tools._palace_pin_impl({"ids": [rid], "pinned": False})["lookups"] == 1
    assert tools._palace_restore_impl({"ids": [rid]})["lookups"] == 0  # not archived


def test_recall_lookup_passes_include_archive(palace):
    with patch.object(palace.lookups, "recall", return_value=[]) as rec:
        tools._palace_recall_lookup_impl({"query": Q, "include_archive": True})
    assert rec.call_args.kwargs["include_archive"] is True
