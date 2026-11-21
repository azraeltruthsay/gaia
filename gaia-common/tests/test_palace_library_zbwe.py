"""zbwe: palace navigation by loose names, synced library shelves, and
restricted wings/rooms that are never opened on request."""

from pathlib import Path

import pytest

from gaia_common.utils.mempalace import MemPalace
from gaia_common.utils.palace_library import asks_about_palace


def _palace(tmp_path: Path, src: Path) -> MemPalace:
    cfg = {
        "root_dir": str(tmp_path / "palace"),
        "kg_path": str(tmp_path / "kg.sqlite3"),
        "wings": {
            "operational": {"description": "Problems solved",
                            "rooms": {"problems": "Bugs", "milestones": "Versions",
                                      "core_principles": "Values"}},
            "library": {"description": "Shelves",
                        "rooms": {"blueprints": "Designs", "tools": "Tools",
                                  "network": {"description": "TSO", "access": "restricted"}}},
            "vault": {"description": "Secrets", "access": "restricted",
                      "rooms": {"keys": "Keys"}},
        },
        "sync_sources": {
            "library/blueprints": [{"glob": str(src / "*.md")}],
            "library/tools": [{"generator": "domain_tools"}],
            "library/network": [{"glob": str(src / "net" / "*.md")}],
        },
        "sync_interval_seconds": 0,
    }
    return MemPalace(cfg)


@pytest.fixture
def src(tmp_path):
    d = tmp_path / "src"
    (d / "net").mkdir(parents=True)
    (d / "GAIA_CORE.md").write_text("# GAIA Core Blueprint\n\nThe cognitive loop.\n\n" + "word " * 3000)
    (d / "GAIA_DOCTOR.md").write_text("# Doctor\n\nThe immune watchdog restarts services.\n")
    (d / "net" / "tso_peers.md").write_text("# TSO peers\n\nsecret peer addresses\n")
    return d


def test_navigate_matches_loose_names(tmp_path, src):
    p = _palace(tmp_path, src)
    out = p.navigate(wing="Operational Wing")
    assert out["ok"] and out["location"] == "operational"
    assert [r["room"] for r in out["contents"]] == ["problems", "milestones", "core_principles"]
    assert out["summary"].startswith("The operational wing has 3 rooms, no more: problems (0 memories)")
    room = p.navigate(wing="operational", room="Core Principles")
    assert room["ok"] and room["location"] == "operational/core_principles"
    assert room["note"].startswith("This room is empty")


def test_library_syncs_and_reads_in_parts(tmp_path, src):
    p = _palace(tmp_path, src)
    shelf = p.navigate(wing="library", room="blueprints")
    assert {d["doc"] for d in shelf["contents"]} == {"GAIA_CORE.md", "GAIA_DOCTOR.md"}
    first = p.read(doc="gaia core")
    assert first["ok"] and first["title"] == "GAIA Core Blueprint" and first["parts"] > 1
    assert p.read(doc="GAIA_CORE", part=2)["part"] == 2
    tools = p.navigate(wing="library", room="tools")
    assert any(d["doc"] == "palace.md" for d in tools["contents"])
    assert "palace(action=read)" in p.read(doc="palace.md", room="tools", wing="library")["text"]


def test_sync_updates_and_prunes(tmp_path, src):
    p = _palace(tmp_path, src)
    p.library.sync()
    (src / "GAIA_DOCTOR.md").unlink()
    (src / "GAIA_CORE.md").write_text("# GAIA Core Blueprint v2\n")
    rep = p.library.sync()["rooms"]["library/blueprints"]
    assert rep["removed"] == 1 and rep["updated"] == 1
    assert p.read(doc="GAIA_CORE")["title"] == "GAIA Core Blueprint v2"


def test_unseen_source_is_not_pruned(tmp_path, src):
    """A process that can't see a source dir keeps the shelf as it is."""
    p = _palace(tmp_path, src)
    p.library.sync()
    p.library._sources["library/blueprints"] = [{"glob": str(tmp_path / "missing" / "*.md")}]
    p.library.sync()
    assert len(p.navigate(wing="library", room="blueprints")["contents"]) == 2


def test_restricted_never_opened_on_request(tmp_path, src):
    p = _palace(tmp_path, src)
    top = {w["wing"]: w for w in p.navigate()["contents"]}
    assert top["vault"].get("restricted") and "rooms" not in top["vault"]
    assert p.navigate(wing="vault")["restricted"]
    lib = {r["room"]: r for r in p.navigate(wing="library")["contents"]}
    assert lib["network"].get("restricted") and "documents" not in lib["network"]
    assert p.navigate(wing="library", room="network")["restricted"]
    assert not p.read(doc="tso_peers")["ok"]
    assert p.read(doc="tso_peers", wing="library", room="network")["restricted"]
    assert not p.recall("secret peer addresses", include_library=True)["library"]
    assert p.status()["wings"]["vault"] == {"rooms": 1, "restricted": True}
    # internal processes, in-process, can
    assert p.read(doc="tso_peers", include_restricted=True)["ok"]


def test_recall_searches_library_only_when_asked(tmp_path, src):
    p = _palace(tmp_path, src)
    assert p.recall("immune watchdog restarts")["library"] == []
    hits = p.recall("immune watchdog restarts", include_library=True)["library"]
    assert hits and hits[0]["doc"] == "GAIA_DOCTOR.md"


@pytest.mark.parametrize("text,want", [
    ("What's in your Operational wing?", True),
    ("What Rooms are in your Operational Wing?", True),
    ("Can you review your memory palace?", True),
    ("Can you tell me how your palatial memory works?", True),
    ("Is your memory arranged in rooms?", True),
    ("The hospital's east wing reopened", False),
    ("Is there room for one more?", False),
    ("She works in the operational side of things", False),
])
def test_asks_about_palace(text, want):
    cfg = {"wings": {"operational": {"rooms": {"problems": ""}}, "library": {"rooms": {"blueprints": ""}}}}
    assert asks_about_palace(text, cfg) is want


def test_near_miss_names_and_docless_read(tmp_path, src):
    p = _palace(tmp_path, src)
    assert p.navigate(wing="operations")["location"] == "operational"
    assert p.navigate(wing="library", room="blueprint")["location"] == "library/blueprints"
    assert not p.navigate(wing="kitchen")["ok"]
    shelf = p.read(wing="library")
    assert shelf["ok"] and shelf["location"] == "library" and "note" in shelf


def test_named_blueprint(tmp_path):
    from gaia_common.utils.founding_docs import named_blueprint
    (tmp_path / "GAIA_CORE.md").write_text("# core")
    (tmp_path / "gaia-doctor.yaml").write_text("x: 1")
    assert named_blueprint("open your blueprint for the doctor", str(tmp_path))[0].endswith("gaia-doctor.yaml")
    assert named_blueprint("read the gaia-core blueprint", str(tmp_path))[0].endswith("GAIA_CORE.md")
    assert named_blueprint("review your blueprints", str(tmp_path)) is None
    assert named_blueprint("what does the doctor do?", str(tmp_path)) is None


def test_room_without_wing(tmp_path, src):
    p = _palace(tmp_path, src)
    assert p.navigate(room="tools")["location"] == "library/tools"
    assert p.read(room="blueprints", doc="GAIA_DOCTOR")["ok"]


def test_readable_blueprint():
    from gaia_common.utils.palace_library import readable_blueprint
    y = ("id: gaia-doctor\nrole: The Immune System\nintent:\n  purpose: Watches and restarts services.\n"
         "interfaces:\n- id: health\n  direction: inbound\n  transport: {method: GET, path: /health}\n"
         "  description: Health check.\nfailure_modes:\n- condition: service down\n  response: restart it\n")
    out = readable_blueprint(y)
    assert out.startswith("# Blueprint: gaia-doctor — The Immune System")
    assert "Watches and restarts services." in out and "- health (GET /health): Health check." in out
    assert "- service down: restart it" in out
    assert readable_blueprint("just: text") == "just: text"


@pytest.mark.parametrize("text,want", [
    ("Can you open your blueprint for the doctor and tell me what it does?", "Tell me what it does."),
    ("Open your constitution", "Tell me what it says."),
    ("Can you review your blueprints?", "Review your blueprints."),
    ("What does your constitution say about consent?", "What does your constitution say about consent?"),
])
def test_as_request_drops_the_opening(text, want):
    from gaia_common.utils.founding_docs import as_request
    assert as_request(text) == want
