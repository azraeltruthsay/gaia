"""She checks facts blind; only sites that really say it count (aym9)."""
from gaia_core.cognition.told_fact_checker import blanks, check_fact, check_told_facts, decide, judge_blank


def _r(url, passage, answer):
    return {"url": url, "passage": passage, "answer": answer}


def test_blanks_are_names_and_numbers():
    assert blanks("Canberra is the capital of Australia.") == ["Canberra", "Australia"]
    assert blanks("The Eiffel Tower is in Paris.") == ["Eiffel Tower", "Paris"]
    assert blanks("water is wet.") == []
    assert len(blanks("The Raven appeared in the New York Evening Mirror on January 29, 1845.")) <= 3


def test_only_answers_on_the_page_count_and_each_site_once():
    out = judge_blank("Canberra", [
        _r("https://a.org/1", "Canberra is the capital.", "Canberra"),
        _r("https://www.a.org/2", "Canberra is the capital.", "Canberra"),
        _r("https://b.org/1", "The capital is Sydney, some say.", "Canberra"),   # not on the page: weights
        _r("https://c.org/1", "It doesn't say.", "NOT STATED"),
    ])
    assert out["supports"] == ["https://a.org/1"] and out["contrary"] == []


def test_sites_agreeing_on_something_else_are_contrary():
    out = judge_blank("Berlin", [
        _r("https://a.org", "It stands in Paris.", "Paris."),
        _r("https://b.org", "Paris is home to it.", "Paris"),
        _r("https://c.org", "Lyon, surely.", "Lyon"),
    ])
    assert {c["answer"].strip(".") for c in out["contrary"]} == {"Paris"}


def test_an_answer_echoing_the_claim_is_not_contrary():
    readings = [_r("https://a.org", "Berlin has a tower.", "Berlin."), _r("https://b.org", "Berlin again.", "Berlin")]
    assert judge_blank("Eiffel Tower", readings, "The Eiffel Tower is in Berlin.")["contrary"] == []


def test_one_stray_or_rambling_answer_is_not_contrary():
    assert judge_blank("Paris", [_r("https://a.org", "Lyon.", "Lyon")])["contrary"] == []
    long = "it is known for many things in this city"
    assert judge_blank("Paris", [_r("https://a.org", long, long), _r("https://b.org", long, long)])["contrary"] == []


def _blank(n, contrary=False):
    urls = [f"https://s{i}.org" for i in range(n)]
    return {"term": "x", "supports": urls, "contrary": [{"url": "u", "answer": "y"}] * 2 if contrary else []}


def test_the_rule():
    assert decide([_blank(2), _blank(2)]) == "supported"
    assert decide([_blank(1), _blank(4)]) == "no_evidence"  # "Sydney is the capital of Australia"
    assert decide([_blank(2), _blank(2), _blank(1)]) == "supported"
    assert decide([_blank(0, True), _blank(0)]) == "contradicted"
    assert decide([_blank(3, True), _blank(2)]) == "supported"
    assert decide([]) == "uncheckable"


def test_she_never_sees_the_claim_and_the_search_does_not_either():
    fact = "Sydney is the capital of Australia."
    seen, queries = [], []

    def search(q):
        queries.append(q)
        return [{"url": "https://a.org", "snippet": "Canberra is the capital of Australia."},
                {"url": "https://b.org", "snippet": "The capital of Australia is Canberra."}]

    def generate(messages):
        seen.append(messages[0]["content"])
        return "Canberra" if "____ is" in messages[0]["content"] else "Australia"

    out = check_fact(fact, search, generate)
    assert out["outcome"] == "contradicted"
    claim = "Sydney is the capital of Australia"
    assert "Sydney" not in queries[0] and not any(claim in t for t in queries + seen)


def test_a_pass_records_results_and_stops_when_the_budget_is_spent():
    calls = []

    def call(method, params):
        calls.append((method, params))
        if method == "palace_told_to_check":
            return {"results": [{"id": "1", "snippet": "Canberra is big."}, {"id": "2", "snippet": "Perth is far."}]}
        if method == "web_search":
            return {"ok": False, "error": "rate limited"} if "is far" in params["query"] else {"results": []}
        return {"ok": True}

    assert check_told_facts(call, lambda m: "NOT STATED") == {"no_evidence": 1}
    recorded = [p for m, p in calls if m == "palace_record_check"]
    assert [p["id"] for p in recorded] == ["1"] and recorded[0]["outcome"] == "no_evidence"
    assert all(p.get("remember") is False for m, p in calls if m == "web_search")
