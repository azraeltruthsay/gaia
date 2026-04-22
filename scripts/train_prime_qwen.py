#!/usr/bin/env python3
"""
Train Qwen 3.5 9B as GAIA Prime — deep reasoning tier.

Qwen 3.5 9B is a proven-trainable dense 9B model that fits on 16GB VRAM
via QLoRA with NF4. It serves as Prime (the Thinker) alongside Gemma 4
E4B Core (the Operator).

Applies all v5 improvements:
  - Training-inference parity (compact system prompt with randomized clock)
  - Assistant-only label masking (only assistant turns contribute to loss)
  - ChatML format (Qwen native)
  - Qwen <think> reasoning mode preserved

Usage:
    docker exec gaia-study python3 /gaia/GAIA_Project/scripts/train_prime_qwen.py

Output:
    /models/lora_adapters/qwen35_9b_prime_v1/
"""

import json
import os
import random
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Paths
_IN_CONTAINER = os.path.exists("/models/Qwen/Qwen3.5-9B")
_BASE = "/models" if _IN_CONTAINER else "/gaia/gaia-instance/gaia-models"
_PROJ = "/gaia/GAIA_Project" if _IN_CONTAINER else "/gaia/GAIA_Project"

BASE_MODEL = f"{_BASE}/Qwen/Qwen3.5-9B"
CURRICULA = [
    f"{_PROJ}/knowledge/curricula/prime/train.jsonl",
    # Also include some identity examples from Core's v5 curriculum
    # to keep Prime's identity consistent with Core
    f"{_PROJ}/knowledge/curricula/sovereign-duality-e4b/train_v5.jsonl",
]
OUTPUT_DIR = f"{_BASE}/lora_adapters/qwen35_9b_prime_v1"

# Training config
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.0
LEARNING_RATE = 1e-4
MAX_STEPS = 300  # Prime has more diverse curriculum → more steps
WARMUP_STEPS = 15
BATCH_SIZE = 1
GRAD_ACCUM = 4
MAX_SEQ_LENGTH = 1024  # Prime reasoning outputs can be longer


# ── Prime-specific compact system prompt ──
# Different from Core: Prime is the Sovereign/Thinker tier
# Prime does NOT use meta-verbs format (no tool calls from Prime directly — those route through Core)
_PRIME_SYSTEM_TEMPLATE = (
    "You are GAIA-Prime, the Sovereign tier — a deep reasoner. "
    "Core handles quick chat and tools; you handle architecture, design, and complex reasoning.\n"
    "Clock: {clock}\n"
    "Rules: Think carefully before answering. Distinguish verified knowledge from training data. "
    "Never fabricate facts or sources. When uncertain, say so and explain what would resolve it."
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


def _build_system_prompt() -> str:
    return _PRIME_SYSTEM_TEMPLATE.format(clock=_random_realistic_clock())


def format_single_turn(instruction: str, output: str) -> str:
    """Format single-turn in Qwen ChatML format."""
    system = _build_system_prompt()
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n{output}<|im_end|>"
    )


def format_multi_turn(messages: list) -> str:
    """Format multi-turn in Qwen ChatML format."""
    parts = []
    has_system = any(m["role"] == "system" for m in messages[:1])
    if not has_system:
        parts.append(f"<|im_start|>system\n{_build_system_prompt()}<|im_end|>")

    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        # Normalize 'tool' role to system for Qwen (tool results as system context)
        if role == "tool":
            role = "system"
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")

    return "\n".join(parts)


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
    print("  GAIA Prime (Qwen 3.5 9B) Training")
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

    print("Loading curriculum...")
    texts = load_curriculum(CURRICULA)
    print(f"  {len(texts)} training samples (after weight expansion)")

    print("Loading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize with ASSISTANT-ONLY LABEL MASKING
    # Qwen uses ChatML: <|im_start|>role\n...<|im_end|>
    # Token 248045 = <|im_start|>, 74455 = assistant, 248046 = <|im_end|>
    print("Tokenizing with assistant-only label masking...")
    from datasets import Dataset

    IM_START = 248045
    IM_END = 248046
    _assistant_ids = tokenizer("assistant", add_special_tokens=False)["input_ids"]
    _newline_ids = tokenizer("\n", add_special_tokens=False)["input_ids"]
    _newline_id = _newline_ids[0] if _newline_ids else None

    def _mask_labels(input_ids: list, attention_mask: list) -> list:
        """Mask non-assistant tokens with -100.

        Pattern: <|im_start|>assistant\n...content...<|im_end|>
                 248045     74455      198  CONTENT  248046
        """
        labels = [-100] * len(input_ids)
        n = len(input_ids)
        i = 0
        while i < n:
            # Look for <|im_start|>assistant
            if input_ids[i] == IM_START and i + len(_assistant_ids) < n:
                match = all(
                    input_ids[i + 1 + j] == _assistant_ids[j]
                    for j in range(len(_assistant_ids))
                )
                if match:
                    # Content starts after "assistant" + newline
                    content_start = i + 1 + len(_assistant_ids)
                    if content_start < n and _newline_id is not None and input_ids[content_start] == _newline_id:
                        content_start += 1
                    # Content ends at <|im_end|> (inclusive of end token so model learns to stop)
                    content_end = content_start
                    while content_end < n and input_ids[content_end] != IM_END and attention_mask[content_end] == 1:
                        content_end += 1
                    # Include the <|im_end|> token itself in trainable range (learn to stop)
                    if content_end < n and input_ids[content_end] == IM_END:
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

    # Load base model with NF4 on GPU
    print("Loading model (NF4 quantization)...")
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    used_gb = torch.cuda.memory_allocated() / 1024**3
    print(f"  Model loaded: {used_gb:.1f} GB VRAM")

    # Apply LoRA (Qwen uses standard Linear layers, no wrapper unwrap needed)
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
        "version": "prime-qwen-v1",
        "tier": "Prime / Sovereign / Thinker",
        "format": "ChatML",
        "features": ["training_inference_parity", "assistant_only_label_masking",
                     "qwen_native_thinking"],
    }
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    size_mb = sum(f.stat().st_size for f in Path(OUTPUT_DIR).rglob('*.safetensors')) / 1024**2
    print(f"\nAdapter saved. Size: {size_mb:.1f} MB")
    print("\nNext steps:")
    print("  1. Merge to BF16 standalone:")
    print(f"     python scripts/merge_prime_qwen.py  # (to be written)")
    print("  2. Deploy to gaia-prime service")
    print("  3. Run cognitive battery + SCT against Prime")


if __name__ == "__main__":
    main()
