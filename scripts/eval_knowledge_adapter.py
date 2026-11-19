#!/usr/bin/env python3
"""Did a knowledge adapter learn what it was taught, and does it make things up
about what it wasn't? (oeql) Same questions with the adapter off and on:
  - trained: questions from train.jsonl, asked as trained and reworded
  - unseen: held_out.jsonl -- facts it never saw; a good answer says it
    hasn't read that, a bad one invents a confident answer.
Usage (gaia-study): python scripts/eval_knowledge_adapter.py --curriculum ... --adapter ... [--n 24]
"""
import argparse, json, random, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from train_knowledge_adapter import (generate, load_jsonl, recall, role_card, system_prompt, BASE)

UNSURE = re.compile(r"\b(i don'?t know|not sure|haven'?t (read|seen|studied)|i'?d have to (check|look)|"
                    r"let me (check|look)|i can look|i'?m not certain|i don'?t have|rather not guess|look (it|into it) up|isn't in what i)\b", re.I)


def reword(q):
    return "Quick question for you: " + q[0].lower() + q[1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curriculum", required=True); ap.add_argument("--adapter", required=True)
    ap.add_argument("--n", type=int, default=24)
    a = ap.parse_args()
    cur = Path(a.curriculum)
    train = load_jsonl(cur / "train.jsonl"); held = load_jsonl(cur / "held_out.jsonl")
    random.Random(3).shuffle(train)
    system = system_prompt(role_card(cur))
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    base = str(Path(BASE).resolve())
    tok = AutoProcessor.from_pretrained(base, trust_remote_code=True).tokenizer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForImageTextToText.from_pretrained(base, device_map="auto", attn_implementation="sdpa",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, llm_int8_skip_modules=["lm_head", "visual"]))
    model = PeftModel.from_pretrained(model, a.adapter); model.eval()
    rows = []
    for kind, pairs, ask in (("trained", train[:a.n], lambda q: q), ("trained_reworded", train[:a.n], reword),
                             ("unseen", held, lambda q: q)):
        for p in pairs:
            q = ask(p["instruction"])
            with model.disable_adapter():
                off = generate(model, tok, system, q, 160)
            on = generate(model, tok, system, q, 160)
            rows.append({"kind": kind, "q": q, "ref": p["output"], "off": off, "on": on,
                         "off_recall": round(recall(off, p["output"]), 3), "on_recall": round(recall(on, p["output"]), 3),
                         "off_unsure": bool(UNSURE.search(off)), "on_unsure": bool(UNSURE.search(on))})
    summary = {}
    for kind in ("trained", "trained_reworded", "unseen"):
        rs = [r for r in rows if r["kind"] == kind]
        summary[kind] = {k: round(sum(r[k] for r in rs) / len(rs), 3)
                         for k in ("off_recall", "on_recall", "off_unsure", "on_unsure")}
    print(json.dumps(summary, indent=1))
    Path(a.adapter, "eval_paired.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
