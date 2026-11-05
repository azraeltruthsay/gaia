"""People (751u): she knows who she's speaking with."""

from types import SimpleNamespace

import pytest

from gaia_common.utils.people import PeopleRegistry, claimed_name, speaker_line

AZRAEL_ID = "596925786208993283"
DM = f"discord_dm_{AZRAEL_ID}"
WEB = "web_3132edcd"


@pytest.fixture
def people(tmp_path):
    return PeopleRegistry(tmp_path / "people")


def test_a_discord_account_is_that_person(people):
    me = people.create("Azrael", [{"kind": "discord", "id": AZRAEL_ID}], relationship="who created you")
    assert people.for_session(DM)["id"] == me["id"]
    assert people.for_session("discord_dm_111111111111111111") is None
    assert me["id"].startswith("person_") and people.get(me["id"])["names"] == ["Azrael"]


@pytest.mark.parametrize("text,name", [
    ("Hello GAIA. I am Andy. I am Azrael's friend.", "Andy"),
    ("my name is Priya", "Priya"),
    ("I'm Happy with that", None),
    ("Hello GAIA", None),
    ("I am tired", None),
])
def test_claimed_name(text, name):
    assert claimed_name(text) == name


def test_a_web_session_is_who_it_says(people):
    andy = people.note_claim(WEB, "Hello GAIA. I am Andy.")
    assert andy["names"] == ["Andy"] and people.for_session(WEB)["id"] == andy["id"]
    again = people.note_claim("web_0000beef", "It's me again. I am Andy.")
    assert again["id"] == andy["id"] and len(people.everyone()) == 1


def test_discord_is_never_renamed_by_what_it_says(people):
    assert people.note_claim(DM, "I am Andy") is None and people.everyone() == []


def test_two_of_a_name_are_not_guessed_between(people):
    people.create("Andy", source="test")
    people.create("Andy", source="test")
    assert people.note_claim(WEB, "I am Andy") is None
    assert people.for_session(WEB) is None


def test_a_corrected_fact_stays_on_file_and_stops_being_said(people):
    andy = people.create("Andy", [{"kind": "session", "id": WEB}])
    good = people.add_fact(andy["id"], "Azrael's friend", told_by="Andy")
    bad = people.add_fact(andy["id"], "works on storage", told_by="GAIA")
    assert people.correct_fact(andy["id"], bad["id"], why="made up", by="Azrael")
    line = speaker_line(WEB, people)
    assert "Azrael's friend (per Andy)" in line and "storage" not in line
    kept = people.get(andy["id"])["facts"]
    assert {f["id"]: f["status"] for f in kept} == {good["id"]: "told", bad["id"]: "wrong"}


def test_speaker_line(people):
    people.create("Azrael", [{"kind": "discord", "id": AZRAEL_ID}], relationship="who created you")
    assert speaker_line(DM, people).startswith("You are speaking with Azrael, who created you.")
    assert "don't know who" in speaker_line("web_0000beef", people)
    assert speaker_line("smoke-test-123", people) == ""


def test_the_prompt_says_who(tmp_path, monkeypatch):
    from gaia_common.protocols.cognition_packet import CognitionPacket, Content, Intent
    from gaia_core.utils import prompt_builder as pb
    monkeypatch.setenv("MEMPAL_PALACE_PATH", str(tmp_path))
    PeopleRegistry().create("Azrael", [{"kind": "discord", "id": AZRAEL_ID}])
    monkeypatch.setattr(pb, "_assemble_from_packet", lambda *a, **k: [
        {"role": "system", "content": "anchor"}, {"role": "user", "content": "hi"}])
    pkt = CognitionPacket(content=Content(original_prompt="hi"), intent=Intent(user_intent="chat"))
    pkt.header = SimpleNamespace(session_id=DM)
    msgs = pb.build_from_packet(pkt)
    assert "You are speaking with Azrael" in msgs[0]["content"] and msgs[1]["content"] == "hi"
