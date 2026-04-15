#!/usr/bin/env python3
"""
Train Gemma 4 26B-A4B (Prime) identity adapter — Sovereign Duality curriculum.

Trains a QLoRA adapter on the base Gemma 4 26B-A4B MoE model.
Same curriculum as E4B Core but targeted at the Prime/Sovereign tier.

Usage:
    docker exec gaia-study python3 /gaia/GAIA_Project/scripts/train_26b_sovereign.py

Output:
    /models/lora_adapters/gemma4_26b_sovereign_v2/
"""

import json
import os
import sys
import time
from pathlib import Path

# Paths work inside gaia-study container (/models/ mount)
_IN_CONTAINER = os.path.exists("/models/google/gemma-4-26B-A4B")
_BASE = "/models" if _IN_CONTAINER else "/gaia/gaia-instance/gaia-models"
_PROJ = "/gaia/GAIA_Project" if _IN_CONTAINER else "/gaia/GAIA_Project"

BASE_MODEL = f"{_BASE}/google/gemma-4-26B-A4B"
CURRICULUM = f"{_PROJ}/knowledge/curricula/sovereign-duality-e4b/train.jsonl"
OUTPUT_DIR = f"{_BASE}/lora_adapters/gemma4_26b_sovereign_v2"
MERGED_DIR = f"{_BASE}/Gemma4-26B-A4B-GAIA-Prime-v2"

# Training config
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.0
LEARNING_RATE = 1e-4  # Conservative for identity — don't overfit
MAX_STEPS = 200
WARMUP_STEPS = 10
BATCH_SIZE = 1
GRAD_ACCUM = 4
MAX_SEQ_LENGTH = 512
TARGET_LOSS = 0.3  # Don't overtrain — we want the base model's general ability
PATIENCE = 5


def format_pair(instruction: str, output: str) -> str:
    """Format a training pair in Gemma 4 chat format."""
    return (
        f"<|turn>user<turn|>\n{instruction}\n"
        f"<|turn>assistant<turn|>\n{output}<turn|>"
    )


def load_curriculum(path: str):
    """Load JSONL curriculum and format for training."""
    pairs = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            text = format_pair(item["instruction"], item["output"])
            weight = item.get("weight", 1.0)
            # Repeat high-weight items
            repeat = max(1, int(weight))
            for _ in range(repeat):
                pairs.append(text)
    return pairs


def main():
    print("=" * 60)
    print("  GAIA E4B Sovereign Identity Training")
    print("=" * 60)
    print(f"Base model: {BASE_MODEL}")
    print(f"Curriculum: {CURRICULUM}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"LoRA: r={LORA_R}, alpha={LORA_ALPHA}")
    print(f"Training: lr={LEARNING_RATE}, steps={MAX_STEPS}, batch={BATCH_SIZE}x{GRAD_ACCUM}")
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
    texts = load_curriculum(CURRICULUM)
    print(f"  {len(texts)} training samples (after weight expansion)")

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

    # Load model with 4-bit quantization
    print("Loading model (NF4 quantization)...")
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True,  # allows CPU dispatch for overflow layers
    )

    # 26B-A4B is 48GB bf16, ~14.7GB NF4. Doesn't fit on 16GB GPU alone.
    # Use CPU offload: most layers on GPU, overflow to CPU RAM.
    # Monkey-patch Params4bit to accept _is_hf_initialized kwarg
    # (workaround for accelerate/bitsandbytes version incompatibility).
    import bitsandbytes as bnb_lib
    _orig_new = bnb_lib.nn.Params4bit.__new__
    @staticmethod
    def _patched_new(cls, data=None, requires_grad=True, **kwargs):
        kwargs.pop("_is_hf_initialized", None)
        return _orig_new(cls, data=data, requires_grad=requires_grad, **kwargs)
    bnb_lib.nn.Params4bit.__new__ = _patched_new

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
        max_memory={0: "13GiB", "cpu": "48GiB"},
    )
    model.config.use_cache = False  # Required for gradient checkpointing

    # Prepare for k-bit training — enable gradient checkpointing manually
    # (prepare_model_for_kbit_training tries float32 upcast which OOMs on 16GB)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    # Enable input gradients for LoRA
    model.enable_input_require_grads()

    used_gb = torch.cuda.memory_allocated() / 1024**3
    print(f"  Model loaded: {used_gb:.1f} GB VRAM")

    # Gemma 4 wraps linear layers in Gemma4ClippableLinear which PEFT
    # doesn't recognize. Unwrap them so PEFT sees Linear4bit directly.
    print("Unwrapping Gemma4ClippableLinear layers for PEFT compatibility...")
    _unwrapped = 0
    for name, module in model.named_modules():
        for attr_name in list(vars(module).keys()):
            child = getattr(module, attr_name, None)
            if child is not None and type(child).__name__ == "Gemma4ClippableLinear":
                inner = getattr(child, "linear", None)
                if inner is not None:
                    setattr(module, attr_name, inner)
                    _unwrapped += 1
    # Also check _modules dict
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
    print(f"  Unwrapped {_unwrapped} Gemma4ClippableLinear → Linear4bit")

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
        "curriculum": CURRICULUM,
        "training_loss": result.training_loss,
        "steps": result.global_step,
        "elapsed_seconds": elapsed,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "learning_rate": LEARNING_RATE,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "version": "sovereign-v2",
    }
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nAdapter saved. Size: {sum(f.stat().st_size for f in Path(OUTPUT_DIR).rglob('*.safetensors')) / 1024**2:.1f} MB")
    print("\nNext steps:")
    print(f"  1. Merge: python scripts/merge_adapter.py {OUTPUT_DIR} {MERGED_DIR}")
    print(f"  2. Update symlink: ln -sf Gemma4-E4B-GAIA-Core-v2 /gaia/gaia-instance/gaia-models/core")
    print(f"  3. Quantize GGUF: llama-quantize {MERGED_DIR} Q4_K_M")
    print("  4. Restart gaia-core and test")


if __name__ == "__main__":
    main()
