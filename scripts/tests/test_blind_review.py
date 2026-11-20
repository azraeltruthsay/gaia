"""blind_review: new pairs checked against the page with the answer hidden."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import blind_review as br  # noqa: E402

PAGE = "Manning formula. The Manning coefficient n is empirical and depends on surface roughness and sinuosity."


def test_check_rules():
    page_ans = lambda m: "n is an empirical value that depends on surface roughness and sinuosity."
    ok, _ = br.check({"instruction": "What affects Manning's n?", "output": "Surface roughness and sinuosity; it is empirical."}, PAGE, page_ans)
    assert ok
    assert not br.check({"instruction": "Who wrote about this?", "output": "Seed and Whitman."}, PAGE, page_ans)[0]
    assert "numbers" in br.check({"instruction": "What is n for concrete?", "output": "0.013"}, PAGE, page_ans)[1]
    assert "doesn't answer" in br.check({"instruction": "What is Darcy's law?", "output": "Flow through porous media"},
                                        PAGE, lambda m: "NOT IN TEXT")[1]


def test_main_writes_review(tmp_path):
    cur, study = tmp_path / "cur", tmp_path / "study"
    (study / "sources").mkdir(parents=True)
    cur.mkdir()
    (cur / "review.json").write_text(json.dumps({"reviewed_by": "x", "drop": {}, "edit": {}, "keep": []}))
    (study / "sources" / "manning_formula.txt").write_text(PAGE)
    rows = [{"instruction": "What affects Manning's n?", "output": "Surface roughness and sinuosity.", "topic": "Manning formula"},
            {"instruction": "Who wrote the key references?", "output": "Manning.", "topic": "Manning formula"}]
    (study / "pairs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = br.main(str(cur), str(study), gen=lambda m: "It depends on surface roughness and sinuosity.")
    assert out == {"kept": 1, "dropped": 1}
    r = json.loads((cur / "review.json").read_text())
    assert r["keep"] == ["What affects Manning's n?"] and "trivia" in list(r["drop"].values())[0]
