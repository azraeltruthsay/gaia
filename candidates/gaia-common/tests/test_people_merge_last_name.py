"""Two records that are one person merge; a last name given about oneself is kept."""
from gaia_common.utils import people as pp
from gaia_common.utils.people import PeopleRegistry, note_last_name, stated_last_name


def _two(tmp_path):
    r = PeopleRegistry(tmp_path)
    web = r.create("Andy", [{"kind": "session", "id": "web_1"}], relationship="Azrael's friend", source="seeded")
    dc = r.create("ShinyAndreon", [{"kind": "discord", "id": "222"}], source="spoke in discord_channel_1")
    r.add_fact(dc["id"], "likes haiku", "Andy", "discord")
    return r, web, dc


def test_merge_folds_and_archives(tmp_path):
    r, web, dc = _two(tmp_path)
    kept = r.merge(web["id"], dc["id"], "same person", "Azrael")
    assert kept["names"] == ["Andy", "ShinyAndreon"]
    assert {a["kind"] for a in kept["accounts"]} == {"session", "discord"}
    assert kept["relationship"] == "Azrael's friend"
    assert [f["text"] for f in kept["facts"]] == ["likes haiku"]
    assert [p["id"] for p in r.everyone()] == [web["id"]]
    assert (tmp_path / "merged" / f"{dc['id']}.json").exists()


def test_stated_last_name():
    assert stated_last_name("my last name is smith", ["Andy"]) == "Smith"
    assert stated_last_name("It's Smith", ["Andy"], asked=True) == "Smith"
    assert stated_last_name("Andy Smith", ["Andy"], asked=True) == "Smith"
    assert stated_last_name("It's Smith", ["Andy"]) is None
    assert stated_last_name("Sure, I'll tell you later when we talk", ["Andy"], asked=True) is None


def test_note_last_name_when_asked(tmp_path):
    r, web, dc = _two(tmp_path)
    p = r.get(web["id"])
    assert note_last_name(p, "Smith", "Andy, what's your last name?", "discord", r) == "Smith"
    p = r.get(web["id"])
    assert p["last_name"] == "Smith" and "Andy Smith" in p["names"]
    assert pp._display(p).startswith("Andy Smith")


def test_not_asked_not_recorded(tmp_path):
    r, web, dc = _two(tmp_path)
    assert note_last_name(r.get(web["id"]), "Smith", "Nick, what's your last name?", "", r) is None


def test_mine_is():
    assert stated_last_name("Mine's Johnson", ["Nick"], asked=True) == "Johnson"


def _reader(answer):
    return lambda messages: answer


def test_model_reads_family_surname(tmp_path):
    r, web, dc = _two(tmp_path)
    assert note_last_name(r.get(web["id"]), "Our family surname is Okafor", "", "", r,
                          read=_reader("Okafor")) == "Okafor"


def test_model_invention_is_not_kept(tmp_path):
    r, web, dc = _two(tmp_path)
    assert note_last_name(r.get(web["id"]), "My name is a secret", "", "", r, read=_reader("Smith")) is None
    assert note_last_name(r.get(web["id"]), "My name is Andy", "", "", r, read=_reader("Andy")) is None
    assert note_last_name(r.get(web["id"]), "Names are overrated", "", "", r, read=_reader("NONE")) is None


def test_failed_reading_falls_back_to_patterns(tmp_path):
    r, web, dc = _two(tmp_path)
    assert note_last_name(r.get(web["id"]), "my last name is Smith", "", "", r, read=_reader("")) == "Smith"


def test_unrelated_message_never_read(tmp_path):
    r, web, dc = _two(tmp_path)
    called = []
    assert note_last_name(r.get(web["id"]), "nice weather", "", "", r,
                          read=lambda m: called.append(1) or "X") is None
    assert not called
