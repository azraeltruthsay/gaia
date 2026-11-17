"""2026-11-17: where she stands with people, moved by evidence."""
from datetime import datetime, timedelta, timezone

from gaia_common.utils import bonds
from gaia_common.utils.people import PeopleRegistry


def _p(name="Nick", rel="Azrael's friend"):
    return {"id": "p1", "names": [name], "relationship": rel}


def test_trust_slow_to_earn_quick_to_lose():
    p = _p()
    for _ in range(2):
        bonds.note(p, "trust", 1, "checked out")
    assert bonds.level(p, "trust") == 0
    bonds.note(p, "trust", 1, "checked out")
    assert bonds.level(p, "trust") == 1
    bonds.note(p, "trust", -3, "turned out wrong")
    assert bonds.level(p, "trust") == 0


def test_regard_drifts_back():
    p = _p()
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    bonds.set_level(p, "regard", 0, "start", "test", now=t0)
    for i in range(6):
        bonds.note(p, "regard", 1, f"warm {i}", now=t0)
    assert bonds.level(p, "regard", now=t0) == 2
    assert bonds.level(p, "regard", now=t0 + timedelta(days=95)) == 1


def test_creator_floors():
    az = _p("Azrael", "who created you")
    bonds.note(az, "trust", -30, "test")
    assert bonds.level(az, "trust") == 3 and bonds.level(az, "regard") == 3


def test_a_claimed_azrael_gets_no_floor():
    fake = _p("Azrael", "")
    assert bonds.level(fake, "trust") == 0


def test_once_per_day():
    p = _p()
    assert bonds.note(p, "regard", 1, "was warm to her", once_per_day=True)
    assert not bonds.note(p, "regard", 1, "was warm to her", once_per_day=True)


def test_lines_honest_when_trusted_guarded_when_not():
    nick = _p()
    bonds.set_level(nick, "regard", 1, "Azrael's friend", "Azrael")
    line = bonds.bond_line(nick, "Nick")
    assert "don't know Nick well enough" in line and "friendly" in line and "+" not in line
    bonds.set_level(nick, "trust", 2, "earned", "Azrael")
    assert bonds.bond_line(nick, "Nick").startswith("Where you stand with Nick: you trust them")
    assert bonds.bond_line(nick).startswith("Where you stand with them:")


def test_registry_round_trip(tmp_path):
    reg = PeopleRegistry(tmp_path)
    p = reg.create("Nick")
    reg.bond_set(p["id"], "regard", 1, "Azrael's friend; a silly boy", "Azrael")
    reg.bond_note(p["id"], "trust", -3, "said something false", evidence="the moon is cheese")
    got = reg.get(p["id"])
    assert bonds.standing(got)["regard"] == 1 and bonds.standing(got)["trust"] == -1
    assert bonds.history(got, "trust")[-1]["evidence"] == "the moon is cheese"


def test_checked_told_fact_moves_teller_trust(tmp_path, monkeypatch):
    from gaia_common.utils import people
    from gaia_common.utils.lookup_memory import LookupMemory
    monkeypatch.setattr(people, "default_root", lambda: tmp_path / "people")
    reg = PeopleRegistry(tmp_path / "people")
    nick = reg.create("Nick")
    reg.link_session(nick["id"], "web_nick")
    mem = LookupMemory(tmp_path / "lookups")
    ids = mem.remember("moon", [{"url": "", "title": "the moon", "snippet": "The moon is made of cheese.",
                                 "trust_tier": "told", "told_by": {"who": "Nick", "id": "session:web_nick"}}],
                       kind="told")
    mem.record_check(ids[0], "contradicted", [])
    assert bonds.standing(reg.get(nick["id"]))["trust"] == -1


def test_teasing_dips_then_reverts():
    p = _p()
    t0 = datetime(2026, 11, 12, tzinfo=timezone.utc)
    bonds.set_level(p, "regard", 1, "friend", "Azrael", now=t0)
    for i in range(3):
        bonds.note(p, "regard", -1, "teased her", transient=True, now=t0 + timedelta(hours=i))
    assert bonds.level(p, "regard", now=t0 + timedelta(hours=3)) == 0
    assert bonds.recently(p, "teased her", now=t0 + timedelta(hours=10))
    later = t0 + timedelta(hours=2 + bonds.TRANSIENT_HOURS + 1)
    assert bonds.level(p, "regard", now=later) == 1
    assert not bonds.recently(p, "teased her", now=later)


def test_teasing_capped_at_two_levels():
    p = _p()
    t0 = datetime(2026, 11, 12, tzinfo=timezone.utc)
    for i in range(20):
        bonds.note(p, "regard", -1, "teased her", transient=True, now=t0)
    assert bonds.level(p, "regard", now=t0) == -2
