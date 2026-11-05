"""Palace report (oxfy): everything in her memory, read-only."""
import json
import sqlite3

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gaia_web.routes import palace as route


@pytest.fixture
def client(tmp_path, monkeypatch):
    root = tmp_path / "palace"
    for d in ("people", "knowledge/lookups", "knowledge/works", "conversations", "technical/architecture"):
        (root / d).mkdir(parents=True)
    (root / "people" / "person_a.json").write_text(json.dumps({
        "id": "person_a", "names": ["Andy"], "relationship": "Azrael's friend",
        "accounts": [{"kind": "session", "id": "web_3132edcd"}], "facts": []}))
    (root / "knowledge/lookups/a.json").write_text(json.dumps({"id": "a", "title": "Raven", "snippet": "A poem", "times_recalled": 1}))
    (root / "knowledge/lookups/b.json").write_text(json.dumps({"id": "b", "title": "Excalibur", "snippet": "A sword", "times_recalled": 9}))
    (root / "knowledge/works/w.json").write_text(json.dumps({"id": "w", "title": "Jabberwocky", "text": "Twas brillig"}))
    (root / "technical/architecture/m1.aaak").write_text("technical|architecture|2026-11-01|web|chat|util=0\n0:GAI|note")
    (root / "conversations/web_3132edcd.jsonl").write_text(
        json.dumps({"id": "1", "ts": "2026-11-05T02:02:03", "user": "I am Andy", "reply": "Hello"}) + "\n")
    kg = tmp_path / "kg.sqlite3"
    conn = sqlite3.connect(kg)
    conn.executescript("""
        create table entities(id text); create table worlds(id text, name text, modality text, lifecycle text);
        create table triples(id text, subject text, predicate text, object text, valid_from text, valid_to text,
          confidence real, source text, extracted_at text, world text, fact_type text);
        insert into worlds values('actuality','actuality','actuality','durable');
        insert into triples values('1','self','likes','poems',null,null,1,'t','2026-11-05','actuality','pref');
        insert into triples values('2','self','liked','rust',null,'2026-01-01',1,'t','2026-01-01','actuality','pref');
    """)
    conn.commit()
    conn.close()
    monkeypatch.setenv("PALACE_ROOT", str(root))
    monkeypatch.setenv("PALACE_KG_PATH", str(kg))
    app = FastAPI()
    app.include_router(route.router)
    return TestClient(app)


def test_overview_counts_every_section(client):
    assert client.get("/api/palace/overview").json() == {
        "people": 1, "lookups": 2, "works": 1, "memories": 1, "conversations": 1, "facts": 2, "entities": 0}


def test_lookups_most_used_first_and_searchable(client):
    assert [r["id"] for r in client.get("/api/palace/lookups").json()["items"]] == ["b", "a"]
    assert [r["id"] for r in client.get("/api/palace/lookups", params={"q": "poem"}).json()["items"]] == ["a"]


def test_conversations_name_the_person_and_open(client):
    row = client.get("/api/palace/conversations").json()["items"][0]
    assert row["person"] == "Andy" and row["exchanges"] == 1
    assert client.get("/api/palace/conversations/web_3132edcd").json()["exchanges"][0]["user"] == "I am Andy"


@pytest.mark.parametrize("bad", ["..%2F..%2Fetc%2Fpasswd", "a..b", "bad%20id"])
def test_conversation_id_cannot_leave_the_palace(client, bad):
    assert client.get(f"/api/palace/conversations/{bad}").status_code in (400, 404)


def test_memories_parse_their_header(client):
    m = client.get("/api/palace/memories").json()["items"][0]
    assert (m["wing"], m["room"], m["date"], m["intent"]) == ("technical", "architecture", "2026-11-01", "chat")


def test_graph_hides_ended_facts_unless_asked(client):
    assert client.get("/api/palace/graph").json()["total"] == 1
    assert client.get("/api/palace/graph", params={"include_ended": True}).json()["total"] == 2
    assert client.get("/api/palace/graph", params={"q": "poems"}).json()["items"][0]["subject"] == "self"


def test_graph_cannot_be_injected_through_a_filter(client):
    assert client.get("/api/palace/graph", params={"q": "'; drop table triples; --"}).json()["total"] == 0
    assert client.get("/api/palace/graph").json()["total"] == 1


def test_report_only_reads(client):
    assert all(m == {"GET"} for r in route.router.routes for m in [r.methods - {"HEAD"}])
