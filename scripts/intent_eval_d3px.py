#!/usr/bin/env python3
"""How well can each way of reading a message tell what the person wants? (d3px)

Scores today's rules, the embedding model and Core on a set of real messages
labeled with what she should have done. The set holds private conversations
and lives outside git, in /shared/evals/intent_d3px/labeled.json.

Run inside gaia-core-candidate (candidate code, the embedding model):
    docker cp scripts/intent_eval_d3px.py gaia-core-candidate:/tmp/
    docker exec gaia-core-candidate python /tmp/intent_eval_d3px.py rules embed
    docker exec gaia-core-candidate python /tmp/intent_eval_d3px.py core --url http://gaia-core:8092/v1/chat/completions

Labels: chat self state fact lookup recall tell dispute followup create recite tool.

2026-11-06, 166 messages: rules 47% (chat is the fall-through, so 45/45);
Core asked to pick a label 25% (it answered the message instead); Core shown
examples as earlier turns 0% (it wrote the next message); embedding nearest
neighbours 65% leave-one-out.
"""
import argparse
import collections
import json
import time
import urllib.request

DATA = "/shared/evals/intent_d3px/labeled.json"
MENU = """chat: conversation, a greeting, how she feels, an opinion about something general
self: about GAIA herself: her documents, constitution, architecture, identity, what she thinks of them
state: her live state right now: the time or date, her system, which models are running, when she was trained
fact: a general-knowledge question with a settled answer
lookup: needs current information from the web: news, who holds an office now, recent events
recall: asks about an earlier conversation, or about something she was told
tell: the person is telling her a fact to keep
dispute: the person says her last answer was wrong
followup: only makes sense with the previous turn: "who was his queen then?", "summarize those"
create: write something new: a poem, joke, story
recite: say an existing text word for word: a poem, a song
tool: an explicit file or tool action: "read the file ..."
"""
LABELS = [line.split(":")[0] for line in MENU.strip().splitlines()]


def report(name, rows, got):
    ok = sum(g == x["label"] for g, x in zip(got, rows))
    per = collections.defaultdict(lambda: [0, 0])
    for g, x in zip(got, rows):
        per[x["label"]][1] += 1
        per[x["label"]][0] += g == x["label"]
    print(f"{name}: {ok}/{len(rows)} = {ok / len(rows):.0%}")
    print("  " + " ".join(f"{k} {v[0]}/{v[1]}" for k, v in sorted(per.items(), key=lambda kv: -kv[1][1])))
    wrong = collections.Counter((x["label"], g) for g, x in zip(got, rows) if g != x["label"])
    print("  most confused (label -> got):", ", ".join(f"{a}->{b} {n}" for (a, b), n in wrong.most_common(6)))


def history(x):
    if not x["prev_user"]:
        return []
    return [{"role": "user", "content": x["prev_user"]}, {"role": "assistant", "content": x["prev_reply"]}]


def by_rules(rows):
    from gaia_core.cognition.nlu.intent_detection import (
        _keyword_intent_classify, is_conversation_followup, is_conversation_recall,
        is_factual_pushback, is_knowledge_followup)
    from gaia_core.cognition.told_facts import told_fact
    from gaia_core.cognition.earlier_lookup import refers_to_results
    from gaia_common.utils.founding_docs import named_document
    from gaia_common.utils.host_facts import asks_about_host
    kw = {"recitation": "recite", "read_file": "tool", "write_file": "tool", "find_file": "tool",
          "list_files": "tool", "list_tree": "tool", "list_tools": "tool", "shell": "tool", "correction": "dispute"}

    def one(x):
        t, h = x["text"], history(x)
        if told_fact(t):
            return "tell"
        if asks_about_host(t):
            return "state"
        if named_document(t):
            return "self"
        if is_conversation_recall(t) or is_conversation_followup(t, h):
            return "recall"
        if is_factual_pushback(t, h):
            return "dispute"
        if refers_to_results(t) or is_knowledge_followup(t, h):
            return "followup"
        return kw.get(_keyword_intent_classify(t), "chat")
    return [one(x) for x in rows]


def by_embedding(rows, model="/models/all-MiniLM-L6-v2", k=3):
    """Nearest labeled neighbours, holding out the message's whole session:
    the same request asked twice in one chat would otherwise vouch for itself."""
    import numpy as np
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model, device="cpu")
    emb = m.encode([x["text"][:400] for x in rows], normalize_embeddings=True)
    sim = emb @ emb.T
    got = []
    for i, x in enumerate(rows):
        s = sim[i].copy()
        for j, y in enumerate(rows):
            if y["session"] == x["session"]:
                s[j] = -1
        votes = collections.Counter()
        for j in np.argsort(-s)[:k]:
            votes[rows[j]["label"]] += float(s[j])
        got.append(votes.most_common(1)[0][0])
    return got


def by_core(rows, url, plain=False):
    got, secs = [], []
    for x in rows:
        ctx = f'Previous message: "{x["prev_user"][:200]}"\n' if x["prev_user"] else ""
        prompt = (f"Sort a message sent to GAIA into one of these:\n{MENU}\n{ctx}"
                  f'Message: "{x["text"][:400]}"\nAnswer with one word from the list.')
        req = {"messages": [{"role": "user", "content": prompt}], "max_tokens": 6, "temperature": 0.0}
        if not plain:  # Gemma 4 12B thought at length when told enable_thinking=False (2026-11-06)
            req["chat_template_kwargs"] = {"enable_thinking": False}
        body = json.dumps(req).encode()
        t = time.time()
        out = json.load(urllib.request.urlopen(
            urllib.request.Request(url, body, {"Content-Type": "application/json"}), timeout=180))
        secs.append(time.time() - t)
        text = out["choices"][0]["message"]["content"].strip().lower()
        got.append(next((label for label in LABELS if label in text), "?"))
    print(f"  Core median {sorted(secs)[len(secs) // 2]:.2f}s")
    return got


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ways", nargs="+", choices=["rules", "embed", "core", "hybrid"])
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--url", default="http://gaia-core:8092/v1/chat/completions")
    ap.add_argument("--plain", action="store_true", help="send no chat_template_kwargs")
    a = ap.parse_args()
    rows = json.load(open(a.data))
    done = {}
    for way in a.ways:
        if way == "hybrid":
            # A rule when one fires (they are precise), the embedding otherwise.
            r = done.get("rules") or by_rules(rows)
            e = done.get("embed") or by_embedding(rows)
            fired = [g != "chat" for g in r]
            right = sum(g == x["label"] for g, x, f in zip(r, rows, fired) if f)
            print(f"  rules fired on {sum(fired)}, right on {right}")
            got = [g if f else h for g, h, f in zip(r, e, fired)]
        else:
            got = {"rules": lambda: by_rules(rows), "embed": lambda: by_embedding(rows),
                   "core": lambda: by_core(rows, a.url, a.plain)}[way]()
        done[way] = got
        report(way, rows, got)
