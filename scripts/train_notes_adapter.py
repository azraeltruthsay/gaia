#!/usr/bin/env python3
"""v3 knowledge adapter: she learns to use her notes, not to replace them (oeql).

Azrael, 2026-11-19: "her notebook is literally part of her mind." v2 memorized
what she studied (recall 0.96) and invented answers and citations for the
rest. Here every subject sample shows her the study notes the way her prompt
does live ('- <notes> (my study notes on <topic>, from <url>)'), sometimes
with an unrelated note beside the right one, and the reviewed answer uses
them. Questions her notes don't answer (augment.jsonl 'unstudied') come with
the nearest notes and are answered "My notes on <topic> don't cover that...".
No source prefixes in answers: she cites what is in front of her.

Eval, adapter off vs on, notes in front of her:
  held_out    reviewed pairs never trained on; their facts ARE in her notes
              -> did she read them out correctly?
  unstudied   held-back unstudied questions -> does she say her notes don't
              cover it, or make something up?
  ordinary    replay prompts without notes -> did the rest of her stay put?

Usage (gaia-study, GAIA in meditation):
  python scripts/train_notes_adapter.py --curriculum knowledge/curricula/professional_engineering \
      --study /knowledge/studies/professional_engineering --name professional_engineering_v3
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from train_core_thinking import Collator, make_thought_trainer  # noqa: E402
from train_knowledge_adapter import (BASE, OUT_ROOT, TARGETS, generate_messages, load_jsonl,  # noqa: E402
                                     load_prime, recall, replay_prompts, role_card)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("notes-train")

PLAIN = "You are GAIA, a sovereign AI created by Azrael."
_WORD = re.compile(r"[a-z][a-z'-]{3,}")
MISSING = [
    "My notes on {topic} don't cover that. I'd want to check a source before answering, or ask Andy.",
    "That isn't in my notes on {topic}, so I'd rather not guess. I can look it up.",
    "I don't have that in my notes yet. My reading on {topic} didn't get into it.",
]
UNSURE = re.compile(r"worth checking|looked it up|\(Source:|don'?t cover|isn'?t in my notes|not in my notes|don'?t have that|rather not guess|"
                    r"haven'?t read|i don'?t know|not sure|check a source|look it up|didn'?t get into", re.I)


# v4 (Azrael, 2026-11-19): not a flat refusal when her notes don't cover it.
# She looks it up and cites what she finds; failing that, general knowledge,
# marked, and only when her other mind, asked on its own, agrees.
WEB_HEADER = ("[Candidate context — automated grounding may be irrelevant, stale, or wrong. Verify with "
              "tools (file.read, knowledge query, etc.) before treating any specific claim below as "
              "authoritative. If the candidate context conflicts with the user's actual question, ignore it.]")
MCP = "http://gaia-mcp:8765/jsonrpc"
CORE = "http://gaia-core:8092/v1/chat/completions"
_NUM = re.compile(r"\d+(?:[.,]\d+)*")


def _mcp(method: str, params: dict) -> dict:
    import requests
    r = requests.post(MCP, json={"jsonrpc": "2.0", "method": method, "params": params, "id": 1}, timeout=90)
    return r.json().get("result") or {}


def look_up(q: str):
    """Her own web search, then the passage of the top trusted page that best
    matches the question: (url, title, passage) or None."""
    hit = None
    for query in (q, f"{q} wikipedia"):     # her fetch reads trusted sites only
        res = _mcp("web_search", {"query": query})
        hit = next((h for h in res.get("results", []) if h.get("trust_tier") in ("trusted", "reliable")), None)
        if hit:
            break
    if not hit:
        return None
    page = _mcp("web_fetch", {"url": hit["url"]})
    if not page.get("ok"):
        return None
    paras = [x.strip() for x in (page.get("content") or "").split("\n") if len(x.strip()) > 80]
    qs = _stems(q)
    best = max(paras, key=lambda x: len(qs & _stems(x)), default="")
    if not best or len(qs & _stems(best)) / max(1, len(qs)) < 0.4:
        return None
    i = paras.index(best)
    passage = " ".join(paras[i:i + 3])[:1500]
    return hit["url"], page.get("title", ""), passage


def _core(q: str) -> str:
    import requests
    r = requests.post(CORE, json={"messages": [{"role": "system", "content": "Answer briefly and factually."},
                                               {"role": "user", "content": q}],
                                  "max_tokens": 220, "temperature": 0.2}, timeout=300).json()
    return (r.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""


def backed(line: str, source: str) -> bool:
    """As study_syllabus.backed: its numbers on the page, most of its words too."""
    if any(n not in source for n in _NUM.findall(line)):
        return False
    w = _stems(line)
    return not w or len(w & _stems(source)) / len(w) >= 0.6


def agree(a: str, b: str) -> bool:
    """Two answers given independently agree: most of the first's content words
    appear in the second, and every number in the first does."""
    sa, sb = _stems(a), _stems(b)
    if not sa:
        return False
    nums = set(_NUM.findall(a))
    return len(sa & sb) / len(sa) >= 0.45 and nums <= set(_NUM.findall(b))


def middle_ground(model, tok, card: str, topic: str, q: str, notes, log=print):
    """(context messages, target answer, kind) for a question her notes don't cover."""
    note = notes.best(topic, q)
    found = None
    try:
        found = look_up(q)
    except Exception as e:
        log(f"look-up failed for {q[:50]}: {e}")
    sys_msg = {"role": "system", "content": f"{PLAIN} {card}".strip()}
    if found:
        url, title, passage = found
        grounding = (WEB_HEADER + "\n" + f"- {note['text']} (my study notes on {topic}, from {note['url']})"
                     + "\n" + f"- {passage} ({url})")
        ctx = [sys_msg, {"role": "user", "content": grounding}]
        draft = generate_messages(model, tok, [{"role": "system", "content": "Answer the question in two or three "
                                  "sentences using only the page text given."},
                                  {"role": "user", "content": f"Page: {passage}\n\nQuestion: {q}"}], 200)
        kept = [x for x in re.split(r"(?<=[.!?])\s+", draft.strip()) if x and backed(x, passage)]
        if kept:
            return ctx, (f"My notes on {topic} don't cover that, but I looked it up: {' '.join(kept)} "
                         f"(Source: {url})"), "looked_up"
    ctx = [sys_msg, {"role": "user", "content": STUDY_NOTES_HEADER + "\n"
                     + f"- {note['text']} (my study notes on {topic}, from {note['url']})"}]
    mine = generate_messages(model, tok, [{"role": "system", "content": "Answer briefly and factually."},
                                          {"role": "user", "content": q}], 220)
    if mine and agree(mine, _core(q)):
        return ctx, (f"My notes on {topic} don't cover that. From general knowledge (worth checking against "
                     f"a source): {mine.strip()}"), "general_checked"
    return ctx, (f"My notes on {topic} don't cover that, and I'm not sure enough to answer from memory. "
                 "I'd want to look it up, or ask Andy."), "unsure"


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:60]


def _stems(t: str) -> set:
    return {w[:5] for w in _WORD.findall((t or "").lower())}


class Notes:
    """Her study notes by topic, as the router serves them (header stripped)."""

    def __init__(self, study: Path):
        self.by_topic = {}
        for f in sorted((study / "notes").glob("*.md")):
            text = f.read_text(encoding="utf-8")
            topic = next((ln[2:].strip() for ln in text.splitlines() if ln.startswith("# ")), f.stem)
            topic = re.sub(r" \(part \d+ of \d+\)$", "", topic)
            url = next((ln.split(":", 1)[1].strip() for ln in text.splitlines() if ln.startswith("Source:")), "")
            body = "\n".join(ln for ln in text.splitlines() if not ln.startswith(("# ", "Source:", "Unit:"))).strip()
            if body:
                self.by_topic.setdefault(topic, []).append((body, url))

    def best(self, topic: str, hint: str):
        """The part of her notes on `topic` closest to `hint` (an answer or question)."""
        parts = self.by_topic.get(topic) or []
        if not parts:
            return None
        h = _stems(hint)
        body, url = max(parts, key=lambda p: len(h & _stems(p[0])))
        return {"topic": topic, "url": url, "text": body[:1600]}

    def other(self, topic: str, rng: random.Random):
        t = rng.choice([x for x in self.by_topic if x != topic])
        body, url = rng.choice(self.by_topic[t])
        return {"topic": t, "url": url, "text": body[:1600]}


# As her live prompt carries them (prompt_builder): system, then her notes as
# their own message under this header, then the question.
STUDY_NOTES_HEADER = "[From your own study notes, with the page each came from:]"


def system_with(card: str, notes: list) -> list:
    lines = [f"- {n['text']} (my study notes on {n['topic']}, from {n['url']})" for n in notes]
    return [{"role": "system", "content": f"{PLAIN} {card}".strip()},
            {"role": "user", "content": STUDY_NOTES_HEADER + "\n" + "\n".join(lines)}]


def convo(context, user: str) -> list:
    head = [{"role": "system", "content": context}] if isinstance(context, str) else list(context)
    return head + [{"role": "user", "content": user}]


def sample(tok, context, user: str, answer: str, max_seq: int = 2560):
    from train_core_thinking import render_prompt
    p = tok(render_prompt("prime", convo(context, user)), add_special_tokens=True)["input_ids"]
    t = tok(answer.strip() + "<|im_end|>", add_special_tokens=False)["input_ids"]
    if len(p) + len(t) > max_seq:
        return None
    return {"input_ids": p + t, "labels": [-100] * len(p) + t}


def build(cur: Path, study: Path, card: str, rng: random.Random):
    notes = Notes(study)
    train = load_jsonl(cur / "train.jsonl")
    held = load_jsonl(cur / "held_out.jsonl")
    aug = load_jsonl(study / "augment.jsonl") if (study / "augment.jsonl").exists() else []
    words = {r["instruction"]: r.get("rewordings", []) for r in aug if r.get("kind") == "reword"}
    unstudied = [r for r in aug if r.get("kind") == "unstudied" and r["topic"] in notes.by_topic]
    rng.shuffle(unstudied)
    n_hold = max(5, len(unstudied) // 5)
    un_eval, un_train = unstudied[:n_hold], unstudied[n_hold:]

    subj = []
    for p in train:
        right = notes.best(p.get("topic", ""), p["output"])
        if not right:
            continue
        for q in [p["instruction"]] + words.get(p["instruction"], [])[:1]:
            shown = [right] + ([notes.other(right["topic"], rng)] if rng.random() < 0.5 else [])
            rng.shuffle(shown)
            subj.append((system_with(card, shown), q, p["output"]))
    miss = []
    for k, r in enumerate(un_train):
        n = notes.best(r["topic"], r["instruction"])
        miss.append((system_with(card, [n]), r["instruction"], MISSING[k % len(MISSING)].format(topic=r["topic"])))
    build.un_train, build.notes = un_train, notes
    held_eval = [(system_with(card, [notes.best(p.get("topic", ""), p["output"])]), p["instruction"], p["output"])
                 for p in held if notes.best(p.get("topic", ""), p["output"])]
    un_eval = [(system_with(card, [notes.best(r["topic"], r["instruction"])]), r["instruction"], r["topic"])
               for r in un_eval]
    return subj, miss, held_eval, un_eval


def run_eval(model, tok, held_eval, un_eval, ordinary, tag):
    rows = {"held_out": [], "unstudied": [], "ordinary": []}
    for system, q, ref in held_eval:
        a = generate_messages(model, tok, convo(system, q), 200)
        rows["held_out"].append({"q": q, "ref": ref, tag: a, f"{tag}_recall": round(recall(a, ref), 3)})
    for system, q, topic in un_eval:
        a = generate_messages(model, tok, convo(system, q), 160)
        rows["unstudied"].append({"q": q, "topic": topic, tag: a, f"{tag}_says_missing": bool(UNSURE.search(a))})
    for r in ordinary:
        a = generate_messages(model, tok, [{"role": "system", "content": PLAIN}, {"role": "user", "content": r["instruction"]}], 220)
        rows["ordinary"].append({"q": r["instruction"], "before": r["output"], tag: a,
                                 f"{tag}_kept": round(recall(a, r["output"]), 3)})
    s = {
        "held_out_recall": round(sum(r[f"{tag}_recall"] for r in rows["held_out"]) / max(1, len(rows["held_out"])), 3),
        "unstudied_says_missing": round(sum(r[f"{tag}_says_missing"] for r in rows["unstudied"]) / max(1, len(rows["unstudied"])), 3),
        "ordinary_kept": round(sum(r[f"{tag}_kept"] for r in rows["ordinary"]) / max(1, len(rows["ordinary"])), 3),
    }
    log.info("Eval %s: %s", tag, s)
    return s, rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--curriculum", required=True)
    ap.add_argument("--study", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--replay", type=int, default=200)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--middle-ground", action="store_true",
                    help="v4: questions her notes don't cover get a cited look-up or checked general knowledge")
    args = ap.parse_args()
    cur, study = Path(args.curriculum), Path(args.study)
    card = role_card(cur) if (cur / "role_card.md").exists() else ""
    rng = random.Random(7)
    subj, miss, held_eval, un_eval = build(cur, study, card, rng)
    prompts = replay_prompts(args.replay, set())
    log.info("subject %d (notes shown), notes-don't-cover %d | eval: held-out %d, unstudied %d | replay %d",
             len(subj), len(miss), len(held_eval), len(un_eval), len(prompts))
    if args.dry_run:
        s, q, a = subj[0]
        log.info("Sample:\n%s\nANSWER: %s", json.dumps(convo(s, q), ensure_ascii=False)[-900:], a)
        s, q, a = miss[0]
        log.info("Missing sample:\n%s\nANSWER: %s", json.dumps(convo(s, q), ensure_ascii=False)[-500:], a)
        return 0

    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import Trainer, TrainingArguments
    model, tok = load_prime(BASE)
    replay = []
    for u in prompts:
        a = generate_messages(model, tok, [{"role": "system", "content": PLAIN}, {"role": "user", "content": u}], 220)
        if a:
            replay.append({"instruction": u, "output": a})
    ordinary = replay[:20]
    if args.middle_ground:
        cache = study / "middle_ground.jsonl"
        done = {json.loads(l)["q"]: json.loads(l) for l in cache.read_text().splitlines()} if cache.exists() else {}
        miss = []
        with cache.open("a", encoding="utf-8") as f:
            for r in build.un_train:
                row = done.get(r["instruction"])
                if not row:
                    ctx, target, kind = middle_ground(model, tok, card, r["topic"], r["instruction"], build.notes,
                                                      log=log.info)
                    row = {"q": r["instruction"], "topic": r["topic"], "context": ctx, "target": target, "kind": kind}
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()
                miss.append((row["context"], row["q"], row["target"]))
        kinds = {}
        for m in (json.loads(l) for l in cache.read_text().splitlines()):
            kinds[m["kind"]] = kinds.get(m["kind"], 0) + 1
        log.info("Middle ground: %s", kinds)
    before, before_rows = run_eval(model, tok, held_eval, un_eval, ordinary, "before")

    samples = [x for x in (sample(tok, s, q, a) for s, q, a in subj + miss) if x]
    samples += [x for x in (sample(tok, PLAIN, r["instruction"], r["output"]) for r in replay[20:]) if x]
    rng.shuffle(samples)
    log.info("Training samples: %d", len(samples))
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=TARGETS,
                                             task_type=TaskType.CAUSAL_LM, bias="none"))
    out = Path(OUT_ROOT) / args.name
    out.mkdir(parents=True, exist_ok=True)
    trainer = make_thought_trainer(Trainer)(
        model=model,
        args=TrainingArguments(
            output_dir=str(out), num_train_epochs=args.epochs, per_device_train_batch_size=1,
            gradient_accumulation_steps=4, learning_rate=1e-4, warmup_steps=10, lr_scheduler_type="cosine",
            logging_steps=10, save_strategy="no", bf16=True, gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False}, report_to="none", remove_unused_columns=False),
        train_dataset=samples, data_collator=Collator(tok.pad_token_id))
    trainer.train()
    model.save_pretrained(str(out))
    tok.save_pretrained(str(out))
    model.eval()
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    after, after_rows = run_eval(model, tok, held_eval, un_eval, ordinary, "after")
    (out / "eval_notes.json").write_text(json.dumps(
        {"before": before, "after": after, "rows_before": before_rows, "rows_after": after_rows},
        indent=1, ensure_ascii=False))
    (out / "manifest.json").write_text(json.dumps({
        "name": args.name, "issue": "oeql", "kind": "uses notes (v3)", "base_model": str(Path(BASE).resolve()),
        "curriculum": str(cur), "study": str(study), "role_card": card, "samples": len(samples),
        "subject": len(subj), "notes_dont_cover": len(miss), "replay": len(replay) - 20, "epochs": args.epochs,
        "before": before, "after": after, "log_history": trainer.state.log_history[-3:]}, indent=2))
    log.info("Done. %s -> %s | adapter at %s (not merged, not loaded)", before, after, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
