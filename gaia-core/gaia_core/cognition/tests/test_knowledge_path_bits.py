"""bits: capability-gap awareness notes go to the mounted knowledge tree.
Detections carry "knowledge/awareness/…" paths relative to a cwd of /app, so
notes were appended to a stray /app/knowledge/ inside the container."""

from types import SimpleNamespace

from gaia_core.cognition.sleep_task_scheduler import SleepTaskScheduler


def _resolve(path, kdir="/knowledge"):
    fake = SimpleNamespace(config=SimpleNamespace(KNOWLEDGE_DIR=kdir))
    return SleepTaskScheduler._knowledge_path(fake, path)


def test_relative_knowledge_paths_land_under_knowledge_dir():
    assert _resolve("knowledge/awareness/operational/architecture_facts.md") == \
        "/knowledge/awareness/operational/architecture_facts.md"
    assert _resolve("awareness/x.md", "/srv/k") == "/srv/k/awareness/x.md"


def test_absolute_paths_are_kept():
    assert _resolve("/knowledge/awareness/a.md", "/elsewhere") == "/knowledge/awareness/a.md"
