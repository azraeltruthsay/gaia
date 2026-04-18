#!/usr/bin/env python3
"""
Train Gemma 4 E4B with Unified Skill Architecture curriculum.

Extends the sovereign identity training with multi-turn tool-use
conversations. The model learns 5 meta-verbs: search, do, learn,
remember, ask — and how to use them in context.

Supports both formats:
  - Single-turn: {"instruction": str, "output": str, "weight": float}
  - Multi-turn:  {"messages": [{"role": str, "content": str}, ...], "weight": float}

Usage:
    python scripts/train_e4b_unified_skills.py

Output:
    /gaia/gaia-instance/gaia-models/lora_adapters/gemma4_e4b_unified_v1/
"""

import json
import os
import sys
import time
from pathlib import Path

# Paths
_IN_CONTAINER = os.path.exists("/models/google/gemma-4-E4B")
_BASE = "/models" if _IN_CONTAINER else "/gaia/gaia-instance/gaia-models"
_PROJ = "/gaia/GAIA_Project" if _IN_CONTAINER else "/gaia/GAIA_Project"

BASE_MODEL = f"{_BASE}/google/gemma-4-E4B"
CURRICULA = [
    f"{_PROJ}/knowledge/curricula/unified-skills-v1/train.jsonl",
]
OUTPUT_DIR = f"{_BASE}/lora_adapters/gemma4_e4b_unified_v1"
MERGED_DIR = f"{_BASE}/Gemma4-E4B-GAIA-Unified-v1"

# Training config
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.0
LEARNING_RATE = 1e-4
MAX_STEPS = 250  # Slightly more steps for the larger curriculum
WARMUP_STEPS = 10
BATCH_SIZE = 1
GRAD_ACCUM = 4
MAX_SEQ_LENGTH = 768  # Bumped from 512 for multi-turn conversations
TARGET_LOSS = 0.3
PATIENCE = 5


def format_single_turn(instruction: str, output: str) -> str:
    """Format a single-turn training pair in Gemma 4 chat format."""
    return (
        f"<|turn>user<turn|>\n{instruction}\n"
        f"<|turn>assistant<turn|>\n{output}<turn|>"
    )


def format_multi_turn(messages: list) -> str:
    """Format a multi-turn conversation in Gemma 4 chat format.

    Handles user, assistant, and tool (result injection) roles.
    Tool results are formatted as system messages with <|result|> tags.
    """
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "user":
            parts.append(f"<|turn>user<turn|>\n{content}")
        elif role == "assistant":
            parts.append(f"<|turn>assistant<turn|>\n{content}")
        elif role == "tool":
            # Tool results injected as a system turn
            parts.append(f"<|turn>system<turn|>\n{content}")
        elif role == "system":
            parts.append(f"<|turn>system<turn|>\n{content}")

    # Close the last assistant turn
    text = "\n".join(parts)
    if not text.endswith("<turn|>"):
        text += "<turn|>"
    return text


def load_curriculum(paths: list) -> list:
    """Load JSONL curricula supporting both single-turn and multi-turn formats.

    Single-turn: {"instruction": str, "output": str, "weight": float}
    Multi-turn:  {"messages": [{"role": str, "content": str}, ...], "weight": float}
    """
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
                    # Multi-turn format
                    text = format_multi_turn(item["messages"])
                elif "instruction" in item and "output" in item:
                    # Single-turn format
                    text = format_single_turn(item["instruction"], item["output"])
                else:
                    print(f"  WARNING: Unknown format at {path}:{line_num}")
                    continue

                for _ in range(repeat):
                    texts.append(text)

        print(f"  Loaded {path}: {sum(1 for _ in open(path) if _.strip())} entries")

    return texts


def main():
    print("=" * 60)
    print("  GAIA E4B Unified Skill Architecture Training")
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

    # Check GPU
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

    # Analyze
    multi_turn = sum(1 for t in texts if t.count("<|turn>") > 2)
    single_turn = len(texts) - multi_turn
    print(f"  Single-turn: {single_turn}, Multi-turn: {multi_turn}")

    # Load tokenizer
    print("Loading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize
    print("Tokenizing...")
    from datasets import Dataset
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length",
        )
    dataset = Dataset.from_dict({"text": texts})
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized = tokenized.map(lambda x: {"labels": x["input_ids"]})
    print(f"  Dataset: {len(tokenized)} samples, max_len={MAX_SEQ_LENGTH}")

    # Check token lengths
    token_lengths = [sum(1 for t in tokenizer(text)["input_ids"]) for text in texts[:20]]
    avg_len = sum(token_lengths) / len(token_lengths)
    max_len = max(token_lengths)
    print(f"  Sample token lengths (first 20): avg={avg_len:.0f}, max={max_len}")
    if max_len > MAX_SEQ_LENGTH:
        print(f"  WARNING: Some samples exceed MAX_SEQ_LENGTH ({MAX_SEQ_LENGTH})")

    # Load model with 4-bit quantization
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
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model.config.use_cache = False

    # Prepare for k-bit training
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
        save_steps=50,
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

    # Save metadata
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
        "version": "unified-v1",
        "architecture": "Unified Skill Architecture",
        "meta_verbs": ["search", "do", "learn", "remember", "ask"],
    }
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nAdapter saved. Size: {sum(f.stat().st_size for f in Path(OUTPUT_DIR).rglob('*.safetensors')) / 1024**2:.1f} MB")
    print("\nNext steps:")
    print(f"  1. Merge: python scripts/merge_adapter.py {OUTPUT_DIR} {MERGED_DIR}")
    print(f"  2. Update config: set USE_META_VERBS=true in gaia_constants.json")
    print(f"  3. Update symlink: ln -sf Gemma4-E4B-GAIA-Unified-v1 /gaia/gaia-instance/gaia-models/core")
    print(f"  4. Clear KV cache: rm /shared/kvcache/core/identity_prefix.pt")
    print("  5. Restart gaia-core and test")


if __name__ == "__main__":
    main()
