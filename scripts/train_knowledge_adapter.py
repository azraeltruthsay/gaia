#!/usr/bin/env python3
"""Train a knowledge adapter on Prime from a reviewed study curriculum (oeql).

GAIA studied a subject (study_syllabus.py), her checked notes became Q&A
pairs, and every pair was reviewed (knowledge/curricula/<subject>/review.json,
build.py -> train.jsonl / held_out.jsonl). This teaches Prime those answers.

What it does, in one GPU session (gaia-study, GAIA in meditation):
  1. Loads Prime (NF4) as train_core_thinking.py does.
  2. Replay: Prime answers ~80 real past user messages as it is now (adapter
     off); those answers are mixed into training so the adapter learns the
     subject without pulling the rest of her away.
  3. Recall before: answers the held-out questions; scores each answer by how
     much of the reviewed answer's content (words, every number) it carries.
  4. Trains the LoRA (attention + MLP, r16) on subject pairs + replay, with
     the role card as the system prompt (the card is what will sit in her
     prompt when the adapter is on).
  5. Recall after, on the same held-out questions; writes eval.json and the
     manifest beside the adapter.

The adapter is saved, never merged or loaded into production here.

Usage (gaia-study):
  python /gaia/GAIA_Project/scripts/train_knowledge_adapter.py \
      --curriculum /gaia/GAIA_Project/knowledge/curricula/professional_engineering \
      --name professional_engineering_v1 [--epochs 2] [--replay 80] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from train_core_thinking import Collator, make_thought_trainer, render_prompt  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("knowledge-train")

LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
LEARNING_RATE = 1e-4
GRAD_ACCUM = 4
MAX_SEQ = 1536
REPLAY_POOL = "/shared/tmp_h1ok/teacher_prime_v2b.jsonl"
BASE = "/models/prime"
OUT_ROOT = "/models/lora_adapters/tier1_global"
TARGETS = (r".*language_model\.layers\.\d+\.(?:self_attn|mlp)\."
           r"(?:q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$")
_WORD = re.compile(r"[a-z][a-z'-]{3,}")
_NUM = re.compile(r"\d+(?:[.,]\d+)*")
_STOP = set("that this with have from they there their about would could should which "
            "where when will your here into been were what also only must need".split())


def role_card(curriculum: Path) -> str:
    """The quoted card in role_card.md, as one paragraph."""
    text = (curriculum / "role_card.md").read_text()
    quoted = [ln[1:].strip() for ln in text.splitlines() if ln.startswith(">")]
    return " ".join(quoted)


def system_prompt(card: str) -> str:
    return "You are GAIA, a sovereign AI created by Azrael." + (f" {card}" if card else "")


def recall(answer: str, reference: str) -> float:
    """Share of the reference's content the answer carries: its words (by
    5-letter stem) and every number, which must match exactly."""
    ref_words = {w[:5] for w in _WORD.findall(reference.lower()) if w not in _STOP}
    got_words = {w[:5] for w in _WORD.findall(answer.lower())}
    ref_nums, got_nums = set(_NUM.findall(reference)), set(_NUM.findall(answer))
    total = len(ref_words) + len(ref_nums)
    if not total:
        return 1.0
    return (len(ref_words & got_words) + len(ref_nums & got_nums)) / total


def load_jsonl(path: Path) -> list:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def replay_prompts(n: int, avoid: set) -> list:
    """Real past user messages (the pool live thinking was trained on), short
    and not about the subject."""
    seen, out = set(), []
    for line in open(REPLAY_POOL):
        u = (json.loads(line).get("user") or "").strip()
        if 8 <= len(u) <= 400 and u.lower() not in seen and u not in avoid:
            seen.add(u.lower())
            out.append(u)
    random.Random(11).shuffle(out)
    return out[:n]


def generate(model, tokenizer, system: str, user: str, max_new: int = 220) -> str:
    import torch
    prompt = render_prompt("prime", [{"role": "system", "content": system},
                                     {"role": "user", "content": user}])
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new, do_sample=False, use_cache=True,
                             pad_token_id=tokenizer.pad_token_id)
    text = tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def sample(tokenizer, system: str, user: str, answer: str):
    prompt_ids = tokenizer(render_prompt("prime", [{"role": "system", "content": system},
                                                   {"role": "user", "content": user}]),
                           add_special_tokens=True)["input_ids"]
    target_ids = tokenizer(answer.strip() + "<|im_end|>", add_special_tokens=False)["input_ids"]
    if len(prompt_ids) + len(target_ids) > MAX_SEQ:
        return None
    return {"input_ids": prompt_ids + target_ids, "labels": [-100] * len(prompt_ids) + target_ids}


def evaluate(model, tokenizer, system: str, held: list, tag: str) -> list:
    rows = []
    for p in held:
        a = generate(model, tokenizer, system, p["instruction"], max_new=200)
        rows.append({"q": p["instruction"], "reference": p["output"], tag: a,
                     f"{tag}_recall": round(recall(a, p["output"]), 3)})
    mean = sum(r[f"{tag}_recall"] for r in rows) / max(1, len(rows))
    log.info("Recall %s: %.3f over %d held-out questions", tag, mean, len(rows))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--curriculum", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--base-model", default=BASE)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--replay", type=int, default=80)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cur = Path(args.curriculum)
    train_pairs, held = load_jsonl(cur / "train.jsonl"), load_jsonl(cur / "held_out.jsonl")
    card = role_card(cur) if (cur / "role_card.md").exists() else ""
    system = system_prompt(card)
    out = Path(OUT_ROOT) / args.name
    base_model = str(Path(args.base_model).resolve())
    log.info("Base %s | %d train, %d held out | role card %d chars | out %s",
             base_model, len(train_pairs), len(held), len(card), out)
    prompts = replay_prompts(args.replay, {p["instruction"] for p in train_pairs + held})
    log.info("Replay prompts: %d", len(prompts))
    if args.dry_run and not os.environ.get("LOAD_MODEL_IN_DRYRUN"):
        log.info("System prompt: %s", system)
        return 0

    import torch
    if not torch.cuda.is_available():
        log.error("No CUDA GPU available")
        return 1
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    try:
        import transformers.core_model_loading as _cml
        _cml.GLOBAL_WORKERS = 1
    except Exception:
        pass
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig,
                              Trainer, TrainingArguments)

    tokenizer = AutoProcessor.from_pretrained(base_model, trust_remote_code=True).tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(
        base_model, trust_remote_code=True, device_map="auto", low_cpu_mem_usage=True,
        attn_implementation="sdpa",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, llm_int8_skip_modules=["lm_head", "visual"]))
    log.info("Model loaded in %.0fs: %.2f GB VRAM", time.time() - t0, torch.cuda.memory_allocated() / 1024 ** 3)
    model.eval()

    before = evaluate(model, tokenizer, system, held, "before")
    replay = []
    for i, u in enumerate(prompts, 1):
        a = generate(model, tokenizer, "You are GAIA, a sovereign AI created by Azrael.", u)
        if a:
            replay.append({"instruction": u, "output": a})
        if i % 20 == 0:
            log.info("Replay %d/%d", i, len(prompts))

    samples = [s for s in (sample(tokenizer, system, p["instruction"], p["output"]) for p in train_pairs) if s]
    # Replay is answered under the plain system prompt, as it was generated
    samples += [s for s in (sample(tokenizer, "You are GAIA, a sovereign AI created by Azrael.",
                                   r["instruction"], r["output"]) for r in replay) if s]
    random.Random(7).shuffle(samples)
    log.info("Training samples: %d (%d subject, %d replay)", len(samples), len(train_pairs), len(replay))

    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                                             target_modules=TARGETS, task_type=TaskType.CAUSAL_LM, bias="none"))
    trainable, total = model.get_nb_trainable_parameters()
    log.info("Trainable %d / %d (%.3f%%)", trainable, total, 100 * trainable / total)
    out.mkdir(parents=True, exist_ok=True)
    trainer = make_thought_trainer(Trainer)(
        model=model,
        args=TrainingArguments(
            output_dir=str(out), num_train_epochs=args.epochs, per_device_train_batch_size=1,
            gradient_accumulation_steps=GRAD_ACCUM, learning_rate=LEARNING_RATE, warmup_steps=10,
            lr_scheduler_type="cosine", logging_steps=5, save_strategy="no", bf16=True,
            gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
            report_to="none", remove_unused_columns=False),
        train_dataset=samples, data_collator=Collator(tokenizer.pad_token_id))
    trainer.train()
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))

    model.eval()
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    after = evaluate(model, tokenizer, system, held, "after")
    # Replay drift: how far the adapter moved her ordinary answers
    drift = []
    for r in replay[:20]:
        a = generate(model, tokenizer, "You are GAIA, a sovereign AI created by Azrael.", r["instruction"])
        drift.append({"q": r["instruction"], "before": r["output"], "after": a,
                      "kept": round(recall(a, r["output"]), 3)})
    rows = [{**b, **a} for b, a in zip(before, after)]
    summary = {
        "recall_before": round(sum(r["before_recall"] for r in rows) / max(1, len(rows)), 3),
        "recall_after": round(sum(r["after_recall"] for r in rows) / max(1, len(rows)), 3),
        "ordinary_answers_kept": round(sum(d["kept"] for d in drift) / max(1, len(drift)), 3),
    }
    log.info("Summary: %s", summary)
    (out / "eval.json").write_text(json.dumps({"summary": summary, "held_out": rows, "replay_drift": drift},
                                              indent=1, ensure_ascii=False))
    (out / "replay.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in replay))
    (out / "manifest.json").write_text(json.dumps({
        "name": args.name, "issue": "oeql", "base_model": base_model, "curriculum": str(cur),
        "role_card": card, "samples": len(samples), "subject_pairs": len(train_pairs), "replay": len(replay),
        "held_out": len(held), "epochs": args.epochs, "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
        "summary": summary, "log_history": trainer.state.log_history[-5:]}, indent=2))
    log.info("Done. Adapter at %s (not merged, not loaded)", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
