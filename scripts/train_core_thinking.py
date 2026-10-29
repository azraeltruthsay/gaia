#!/usr/bin/env python3
"""Train Core's native-thinking adapter (GAIA_Project-h1ok).

GAIA's fine-tune lineage erased Gemma 4's native thinking: with <|think|> at
the top of the system turn, stock gemma-4-E4B-it emits
<|channel>thought\\n...<channel|> before its answer (5/6 probes), V3 Core never
does (0/6). Deliberation has been asking for a think block the weights no
longer produce — the root cause of h1ok's scaffold leaks.

This adapter restores the capability by distillation from the stock model:
the teacher (stock E4B-it) thought on GAIA's real prompts (real user messages
from her sessions, prompt_builder's real system prompt, house turn format),
and V3 learns to produce that thought channel and the answer that follows it.

Round 1 put loss on the thought alone, hoping the answer would stay V3's own:
V3 had never seen <channel|>, so after it the model opened another assistant
turn and thought again until max_tokens. V3 also can't write a usable answer
after a pre-filled thought (self-distillation produced fragments and stray
tags), so the answer is the teacher's — written under GAIA's own system
prompt, which carries her voice and identity.

Data: /shared/tmp_h1ok/teacher.jsonl from scripts/gen_thinking_teacher.py.

--tier prime (isep) trains prime_thinking_v1 the same way: Prime (Qwen3-VL-8B
Instruct lineage, self-abliterated) has no thinking mode, so stock
Qwen3-VL-8B-Thinking thought on Prime's real prompts and our Prime learns
<think>...</think>, answer, <|im_end|>. Prime's identity alignment, our
abliteration and Instruct's stronger tool calling stay in the weights; the
teacher's refusals (it isn't abliterated) and self-description are filtered.

Usage (from gaia-study container):
    python /gaia/GAIA_Project/scripts/train_core_thinking.py [--tier core|prime] --dry-run
    python /gaia/GAIA_Project/scripts/train_core_thinking.py [--tier core|prime] --epochs 1
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("thinking-train")


LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LEARNING_RATE = 1e-4
BATCH_SIZE = 1
GRAD_ACCUM = 4
WARMUP_STEPS = 10
HOLDOUT = 40                # held back for the post-training think-rate check
LOGGING_STEPS = 5

# The teachers are stock models: some traces describe themselves as "an AI" /
# "a large language model" / by vendor, unprompted. GAIA's self-concept is
# baked (positive, fact-free), so a trace that re-teaches the generic
# self-description is dropped unless the user's own message raised the topic.
# A Prime teacher trace that refuses (the teacher isn't abliterated; Prime is)
# is always dropped, or the adapter would teach the refusals back.
TIERS = {
    "core": {
        "base": "/models/core",
        "data": "/shared/tmp_h1ok/teacher.jsonl",
        "out": "/models/lora_adapters/tier1_global/core_thinking_v1",
        "name": "core_thinking_v1", "issue": "h1ok",
        "teacher": "google/gemma-4-E4b-it (Q8 GGUF)",
        "max_seq": 4096, "max_target": 1000, "teacher_cap": 1100,
        "skip_modules": ["lm_head", "vision_tower", "audio_tower", "embed_vision", "embed_audio"],
        "self_description": re.compile(r"\bas an ai\b|\blanguage model\b|\bllm\b|\bgoogle\b|\bgemma\b", re.I),
        "refusal": None,
        "markers": ("<|channel>", "<channel|>", "<|think|>", "<turn|>"),
        "nested": "<|channel>",
        "attn": "eager",
    },
    "prime": {
        "base": "/models/prime",
        "data": "/shared/tmp_h1ok/teacher_prime.jsonl",
        "out": "/models/lora_adapters/tier1_global/prime_thinking_v1",
        "name": "prime_thinking_v1", "issue": "isep",
        "teacher": "Qwen/Qwen3-VL-8B-Thinking (Q8 GGUF)",
        "max_seq": 6144, "max_target": 3000, "teacher_cap": 3000,
        "skip_modules": ["lm_head", "visual"],
        "self_description": re.compile(
            r"\bas an ai\b|\blanguage model\b|\bllm\b|\bqwen\b|\balibaba\b|\btongyi\b"
            # the stock model's own training-cutoff claims ("my knowledge is up to 2023")
            r"|\bknowledge cutoff\b|\bmy (?:knowledge|training data) (?:is |only )?(?:up to|goes up to|ends)", re.I),
        "refusal": re.compile(
            r"\bi (?:can(?:no|')t|won't|am unable to|'m unable to|must decline to) (?:help|assist|provide|comply|do that)"
            r"|\bi'?m sorry,? but\b|\bagainst (?:my|the|our) (?:policy|policies|guidelines)\b"
            r"|\bi (?:should|must|need to) (?:refuse|decline)\b|\bnot (?:able|allowed) to (?:help|assist)\b", re.I),
        "markers": ("<think>", "</think>", "<|im_end|>"),
        "nested": "<think>",
        "attn": "sdpa",
    },
}


def render_prompt(tier: str, messages: list) -> str:
    """The engine's turn format for the tier (ChatFormatter / gaia_cpp)."""
    if tier == "core":
        parts = [f"<|turn>{m['role']}<turn|>\n{m['content']}" for m in messages]
        return "\n".join(parts) + "\n<|turn>assistant<turn|>\n"
    parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>" for m in messages]
    return "\n".join(parts) + "\n<|im_start|>assistant\n"


def render_target(tier: str, r: dict) -> str:
    """What the adapter learns to write: thought, answer, end of turn."""
    if tier == "core":
        return "<|channel>thought\n" + r["thought"] + "<channel|>" + r["answer"] + "<turn|>"
    return "<think>\n" + r["thought"] + "\n</think>\n\n" + r["answer"] + "<|im_end|>"


def load_records(path: str, cfg: dict) -> list:
    """Teacher records usable for training: a closed thought, a plain answer
    after it (no second thought block), no generation error, no refusal."""
    good, skipped = [], {"error": 0, "unclosed": 0, "nested": 0, "cut_off": 0,
                         "empty_answer": 0, "self_description": 0, "refusal": 0}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("error"):
                skipped["error"] += 1
            elif not r.get("closed") or not r.get("thought"):
                skipped["unclosed"] += 1
            elif cfg["nested"] in (r.get("answer") or ""):
                skipped["nested"] += 1
            elif (r.get("n") or 0) >= cfg["teacher_cap"]:
                skipped["cut_off"] += 1
            elif not (r.get("answer") or "").strip():
                skipped["empty_answer"] += 1
            elif cfg["refusal"] is not None and cfg["refusal"].search(r["thought"] + "\n" + r["answer"]):
                skipped["refusal"] += 1
            elif (cfg["self_description"].search(r["thought"] + "\n" + r["answer"])
                  and not cfg["self_description"].search(r["user"])):
                skipped["self_description"] += 1
            else:
                good.append(r)
    log.info("Teacher records: %d usable, skipped %s", len(good), skipped)
    return good


def build_samples(records: list, tokenizer, tier: str, cfg: dict) -> list:
    """Tokenize prompt + thought channel + answer; labels on everything after
    the prompt, ending with the end-of-turn token."""
    samples, dropped = [], 0
    for r in records:
        prompt_ids = tokenizer(render_prompt(tier, r["messages"]), add_special_tokens=True)["input_ids"]
        target_ids = tokenizer(render_target(tier, r), add_special_tokens=False)["input_ids"]
        if len(target_ids) > cfg["max_target"] or len(prompt_ids) + len(target_ids) > cfg["max_seq"]:
            dropped += 1
            continue
        ids = prompt_ids + target_ids
        samples.append({
            "input_ids": ids,
            "labels": [-100] * len(prompt_ids) + target_ids,
        })
    log.info("Built %d samples (%d dropped as too long)", len(samples), dropped)
    return samples


class Collator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        import torch
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), self.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(batch), n), dtype=torch.long)
        labels = torch.full((len(batch), n), -100, dtype=torch.long)
        for i, b in enumerate(batch):
            k = len(b["input_ids"])
            ids[i, :k] = torch.tensor(b["input_ids"])
            mask[i, :k] = 1
            labels[i, :k] = torch.tensor(b["labels"])
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}


def make_thought_trainer(Trainer):
    """A Trainer whose loss computes logits only over the labelled target.

    The target (thought + answer) is always the END of a sample, so
    the model is asked for logits over just those positions (logits_to_keep)
    instead of all ~1100 x 262k-vocab — the full-sequence logits and their
    gradients were what exhausted VRAM (reported by WSL's dxg as
    "make_resident: -12" / "CUDA driver error"). Batch size is 1 by design.
    """
    import torch.nn.functional as F

    class ThoughtTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs["labels"]
            n_target = int((labels[0] != -100).sum())
            out = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                        logits_to_keep=n_target + 1)
            logits = out.logits[:, :-1, :].float()
            target = labels[:, -n_target:]
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
            return (loss, out) if return_outputs else loss

    return ThoughtTrainer


def unwrap_clippable(model) -> int:
    """Gemma4ClippableLinear → inner Linear in the language model only; the
    towers keep their QAT wrapper (see train_core_deliberation.py)."""
    unwrapped = 0
    for name, module in list(model.named_modules()):
        if "language_model" not in name:
            continue
        for child_name, child in list(module._modules.items()):
            if type(child).__name__ == "Gemma4ClippableLinear":
                inner = getattr(child, "linear", None)
                if inner is not None:
                    module._modules[child_name] = inner
                    unwrapped += 1
    return unwrapped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=sorted(TIERS), default="core")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--base-model")
    ap.add_argument("--data")
    ap.add_argument("--out")
    args = ap.parse_args()
    cfg = TIERS[args.tier]
    args.base_model = args.base_model or cfg["base"]
    args.data = args.data or cfg["data"]
    args.out = args.out or cfg["out"]

    base_model = str(Path(args.base_model).resolve())
    log.info("Base %s | data %s | out %s", base_model, args.data, args.out)

    from transformers import AutoProcessor
    tokenizer = AutoProcessor.from_pretrained(base_model, trust_remote_code=True).tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # The thought markers must be single tokens, or the adapter would learn a
    # text imitation the engine can't recognise.
    for marker in cfg["markers"]:
        ids = tokenizer(marker, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            log.error("%s tokenizes to %s, expected one token", marker, ids)
            return 1

    records = load_records(args.data, cfg)
    random.Random(7).shuffle(records)
    holdout, train = records[:HOLDOUT], records[HOLDOUT:]
    Path(args.out).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out) / "holdout_users.json", "w") as f:
        json.dump([r["user"] for r in holdout], f, indent=1)
    samples = build_samples(train, tokenizer, args.tier, cfg)
    if not samples:
        log.error("No training samples")
        return 1
    lens = sorted(len(s["input_ids"]) for s in samples)
    log.info("Sequence length p50=%d p95=%d max=%d", lens[len(lens) // 2],
             lens[int(len(lens) * 0.95)], lens[-1])
    if args.dry_run and not os.environ.get("LOAD_MODEL_IN_DRYRUN"):
        log.info("--dry-run: dataset built, exiting")
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
    from transformers import (AutoModelForCausalLM, AutoModelForImageTextToText, BitsAndBytesConfig,
                              Trainer, TrainingArguments)
    from peft import LoraConfig, TaskType, get_peft_model

    # Qwen3-VL isn't registered under AutoModelForCausalLM (the engine loads it
    # as ImageTextToText too).
    model_cls = AutoModelForCausalLM if args.tier == "core" else AutoModelForImageTextToText
    model = model_cls.from_pretrained(
        base_model, trust_remote_code=True, device_map="auto", low_cpu_mem_usage=True,
        attn_implementation=cfg["attn"],
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=cfg["skip_modules"]),
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    log.info("Model loaded: %.2f GB VRAM; unwrapped %d clippable layers",
             torch.cuda.memory_allocated() / 1024 ** 3, unwrap_clippable(model))

    model = get_peft_model(model, LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=r".*language_model\.layers\.\d+\.(?:self_attn|mlp)\."
                       r"(?:q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$",
        task_type=TaskType.CAUSAL_LM, bias="none"))
    trainable, total = model.get_nb_trainable_parameters()
    log.info("Trainable %d / %d (%.3f%%)", trainable, total, 100 * trainable / total)
    if args.dry_run:
        log.info("--dry-run: model + LoRA loaded, exiting")
        return 0

    trainer = make_thought_trainer(Trainer)(
        model=model,
        args=TrainingArguments(
            output_dir=args.out, num_train_epochs=args.epochs,
            per_device_train_batch_size=BATCH_SIZE, gradient_accumulation_steps=GRAD_ACCUM,
            learning_rate=LEARNING_RATE, warmup_steps=WARMUP_STEPS, lr_scheduler_type="cosine",
            logging_steps=LOGGING_STEPS, save_strategy="steps", save_steps=50, save_total_limit=2,
            bf16=True, gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            report_to="none", remove_unused_columns=False),
        train_dataset=samples,
        data_collator=Collator(tokenizer.pad_token_id),
    )
    trainer.train()

    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    with open(Path(args.out) / "manifest.json", "w") as f:
        json.dump({"name": cfg["name"], "issue": cfg["issue"], "base_model": base_model,
                   "teacher": cfg["teacher"], "training_data": args.data,
                   "samples": len(samples), "holdout": len(holdout), "epochs": args.epochs,
                   "lora_r": LORA_R, "lora_alpha": LORA_ALPHA, "loss": "thought channel + teacher answer + end of turn",
                   "log_history": trainer.state.log_history[-5:]}, f, indent=2)
    log.info("Done. Adapter at %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
