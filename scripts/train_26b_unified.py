#!/usr/bin/env python3
"""
Train Gemma 4 26B-A4B (Prime) — Unified Skill Architecture curriculum.

This is the Prime counterpart to train_e4b_unified_skills.py, applying all
the same improvements:
  - Training-inference parity (compact system prompt prepended)
  - Assistant-only label masking
  - Randomized realistic clock per example
  - Multi-turn + single-turn support

26B-A4B is a MoE model (128 experts). Doesn't fit on 16GB GPU alone;
uses CPU offload via max_memory to keep most weights on GPU and overflow
to RAM.

Usage:
    docker exec gaia-study python3 /gaia/GAIA_Project/scripts/train_26b_unified.py

Output:
    /models/lora_adapters/gemma4_26b_prime_v2/
"""

import json
import os
import random
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Paths
_IN_CONTAINER = os.path.exists("/models/google/gemma-4-26B-A4B")
_BASE = "/models" if _IN_CONTAINER else "/gaia/gaia-instance/gaia-models"
_PROJ = "/gaia/GAIA_Project" if _IN_CONTAINER else "/gaia/GAIA_Project"

BASE_MODEL = f"{_BASE}/google/gemma-4-26B-A4B"
# Prime uses its deep-reasoning curriculum + unified-skills for tool use + v5 identity
CURRICULA = [
    f"{_PROJ}/knowledge/curricula/prime/train.jsonl",
    f"{_PROJ}/knowledge/curricula/unified-skills-v1/train.jsonl",
    f"{_PROJ}/knowledge/curricula/sovereign-duality-e4b/train_v5.jsonl",
]
OUTPUT_DIR = f"{_BASE}/lora_adapters/gemma4_26b_prime_v2"
MERGED_DIR = f"{_BASE}/Gemma4-26B-A4B-GAIA-Prime-v2"

# Training config
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.0
LEARNING_RATE = 1e-4
MAX_STEPS = 300  # Prime is larger + more diverse curriculum → more steps
WARMUP_STEPS = 15
BATCH_SIZE = 1
GRAD_ACCUM = 4
MAX_SEQ_LENGTH = 768


# ── Training-inference parity: compact system prompt with randomized clock ──
_COMPACT_SYSTEM_TEMPLATE = (
    "You are GAIA, created by Azrael. Sovereign AI.\n"
    "Clock: {clock}\n"
    "Rules: Answer topics directly (don't self-relate). "
    "Never fabricate facts — say 'I don't know' or search. "
    "Be concise.\n"
    "Tools: search(query) do(skill,input) learn(task,result,success) "
    "remember(fact) ask(question)\n"
    "Format: <|tool>verb(param=value)<tool|>\n"
    "Results arrive as: <|tool_response>...<tool_response|>\n"
    "Use search() first to find the right skill, then do() to execute it."
)


def _random_realistic_clock() -> str:
    year = random.choice([2024, 2025, 2026])
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    hour = random.randint(0, 23)
    minute = random.choice([0, 5, 10, 15, 22, 30, 37, 45, 52])
    tz_label = random.choice(["PDT", "PST", "EDT", "EST"])
    dt = datetime(year, month, day, hour, minute, tzinfo=timezone(timedelta(hours=-7)))
    return dt.strftime(f"%-I:%M %p {tz_label}, %A %B %d, %Y")


def _build_compact_system_prompt() -> str:
    return _COMPACT_SYSTEM_TEMPLATE.format(clock=_random_realistic_clock())


def format_single_turn(instruction: str, output: str) -> str:
    system = _build_compact_system_prompt()
    return (
        f"<|turn>system<turn|>\n{system}\n"
        f"<|turn>user<turn|>\n{instruction}\n"
        f"<|turn>assistant<turn|>\n{output}<turn|>"
    )


def format_multi_turn(messages: list) -> str:
    parts = []
    has_system = any(m["role"] == "system" for m in messages[:1])
    if not has_system:
        parts.append(f"<|turn>system<turn|>\n{_build_compact_system_prompt()}")

    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            parts.append(f"<|turn>user<turn|>\n{content}")
        elif role == "assistant":
            parts.append(f"<|turn>assistant<turn|>\n{content}")
        elif role in ("tool", "system"):
            parts.append(f"<|turn>system<turn|>\n{content}")

    text = "\n".join(parts)
    if not text.endswith("<turn|>"):
        text += "<turn|>"
    return text


def load_curriculum(paths: list) -> list:
    texts = []
    for path in paths:
        if not os.path.exists(path):
            print(f"  WARNING: Curriculum not found: {path}")
            continue
        with open(path) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"  WARNING: Invalid JSON at {path}:{line_num}: {e}")
                    continue

                weight = item.get("weight", 1.0)
                repeat = max(1, int(weight))

                if "messages" in item:
                    text = format_multi_turn(item["messages"])
                elif "instruction" in item and "output" in item:
                    text = format_single_turn(item["instruction"], item["output"])
                else:
                    continue

                for _ in range(repeat):
                    texts.append(text)

        print(f"  Loaded {path}: {sum(1 for _ in open(path) if _.strip())} entries")

    return texts


def main():
    print("=" * 60)
    print("  GAIA Prime 26B-A4B Unified Training")
    print("=" * 60)
    print(f"Base model: {BASE_MODEL}")
    print(f"Curricula: {len(CURRICULA)} files")
    for c in CURRICULA:
        print(f"  - {c}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"LoRA: r={LORA_R}, alpha={LORA_ALPHA}")
    print(f"Training: lr={LEARNING_RATE}, steps={MAX_STEPS}, batch={BATCH_SIZE}x{GRAD_ACCUM}")
    print(f"Max seq length: {MAX_SEQ_LENGTH}")
    print()

    import torch
    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU available")
        sys.exit(1)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {torch.cuda.get_device_name(0)} ({vram:.1f} GB)")
    print()

    # Load curriculum
    print("Loading curriculum...")
    texts = load_curriculum(CURRICULA)
    print(f"  {len(texts)} training samples (after weight expansion)")

    # Load tokenizer
    print("Loading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize with ASSISTANT-ONLY LABEL MASKING
    print("Tokenizing with assistant-only label masking...")
    from datasets import Dataset

    TURN_OPEN = 105
    TURN_CLOSE = 106
    _assistant_ids = tokenizer("assistant", add_special_tokens=False)["input_ids"]
    _newline_ids = tokenizer("\n", add_special_tokens=False)["input_ids"]
    _newline_id = _newline_ids[0] if _newline_ids else None

    def _mask_labels(input_ids: list, attention_mask: list) -> list:
        labels = [-100] * len(input_ids)
        n = len(input_ids)
        i = 0
        while i < n:
            if input_ids[i] == TURN_OPEN and i + len(_assistant_ids) + 1 < n:
                match = all(
                    input_ids[i + 1 + j] == _assistant_ids[j]
                    for j in range(len(_assistant_ids))
                )
                close_idx = i + 1 + len(_assistant_ids)
                if match and input_ids[close_idx] == TURN_CLOSE:
                    content_start = close_idx + 1
                    if content_start < n and _newline_id is not None and input_ids[content_start] == _newline_id:
                        content_start += 1
                    content_end = content_start
                    while content_end < n and input_ids[content_end] != TURN_OPEN and attention_mask[content_end] == 1:
                        content_end += 1
                    for k in range(content_start, content_end):
                        labels[k] = input_ids[k]
                    i = content_end
                    continue
            i += 1
        return labels

    def tokenize_fn(examples):
        enc = tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length",
        )
        enc["labels"] = [
            _mask_labels(ids, mask)
            for ids, mask in zip(enc["input_ids"], enc["attention_mask"])
        ]
        return enc

    dataset = Dataset.from_dict({"text": texts})
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    _sample_labels = tokenized[0]["labels"]
    _n_trainable = sum(1 for x in _sample_labels if x != -100)
    _n_masked = sum(1 for x in _sample_labels if x == -100)
    print(f"  Dataset: {len(tokenized)} samples, max_len={MAX_SEQ_LENGTH}")
    print(f"  Sample 0: {_n_trainable} trainable tokens, {_n_masked} masked (-100)")

    # Load model — 26B needs CPU offload
    print("Loading model (NF4 quantization with CPU offload)...")
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )

    # Monkey-patch Params4bit for accelerate/bnb compat
    import bitsandbytes as bnb_lib
    _orig_new = bnb_lib.nn.Params4bit.__new__
    @staticmethod
    def _patched_new(cls, data=None, requires_grad=True, **kwargs):
        kwargs.pop("_is_hf_initialized", None)
        return _orig_new(cls, data=data, requires_grad=requires_grad, **kwargs)
    bnb_lib.nn.Params4bit.__new__ = _patched_new

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # 26B NF4 + CPU offload — sequential splits layers in order which
    # avoids meta-device issues with MoE experts vs auto.
    # low_cpu_mem_usage=False forces full materialization into RAM first.
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="sequential",
        dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=False,
        max_memory={0: "13GiB", "cpu": "50GiB"},
    )
    model.config.use_cache = False

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    used_gb = torch.cuda.memory_allocated() / 1024**3
    print(f"  Model loaded: {used_gb:.1f} GB VRAM")

    # Unwrap Gemma4ClippableLinear for PEFT
    print("Unwrapping Gemma4ClippableLinear layers...")
    _unwrapped = 0
    for name, module in model.named_modules():
        for attr_name in list(vars(module).keys()):
            child = getattr(module, attr_name, None)
            if child is not None and type(child).__name__ == "Gemma4ClippableLinear":
                inner = getattr(child, "linear", None)
                if inner is not None:
                    setattr(module, attr_name, inner)
                    _unwrapped += 1
    for name, module in model.named_modules():
        children_to_replace = {}
        for child_name, child in module._modules.items():
            if type(child).__name__ == "Gemma4ClippableLinear":
                inner = getattr(child, "linear", None)
                if inner is not None:
                    children_to_replace[child_name] = inner
        for child_name, inner in children_to_replace.items():
            module._modules[child_name] = inner
            _unwrapped += 1
    print(f"  Unwrapped {_unwrapped} layers")

    # Apply LoRA
    print("Applying LoRA adapter...")
    from peft import LoraConfig, get_peft_model, TaskType

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # Training
    print("\nStarting training...")
    from transformers import TrainingArguments, Trainer

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
    )

    t0 = time.time()
    result = trainer.train()
    elapsed = time.time() - t0

    print(f"\nTraining complete in {elapsed:.0f}s")
    print(f"  Final loss: {result.training_loss:.4f}")
    print(f"  Steps: {result.global_step}")

    # Save adapter
    print(f"\nSaving adapter to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    metadata = {
        "base_model": BASE_MODEL,
        "curricula": CURRICULA,
        "training_loss": result.training_loss,
        "steps": result.global_step,
        "elapsed_seconds": elapsed,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "learning_rate": LEARNING_RATE,
        "max_seq_length": MAX_SEQ_LENGTH,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "version": "prime-unified-v2",
        "features": ["training_inference_parity", "assistant_only_label_masking",
                     "cpu_offload", "gemma4_clippable_unwrap"],
    }
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    size_mb = sum(f.stat().st_size for f in Path(OUTPUT_DIR).rglob('*.safetensors')) / 1024**2
    print(f"\nAdapter saved. Size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
