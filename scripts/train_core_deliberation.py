#!/usr/bin/env python3
"""Train Core deliberation adapter (k23-efq).

Multimodal-aware QLoRA training on the deliberation curriculum. Adapted
from train_core_multimodal.py but text-only — same NF4 + skip_modules +
Gemma4ClippableLinear unwrap + LoRA regex targeting language_model.

Reads /knowledge/curricula/deliberation/train.json (90 hand-curated
examples) and trains a LoRA adapter that teaches Core to emit
<think>...</think> blocks before final responses, refuse cataloged
forbidden phrases, and engage substantively with introspective probes.

Usage (from gaia-study container):
    python /gaia/GAIA_Project/scripts/train_core_deliberation.py
    python /gaia/GAIA_Project/scripts/train_core_deliberation.py --steps 100
    python /gaia/GAIA_Project/scripts/train_core_deliberation.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("deliberation-train")

# ── Paths ──────────────────────────────────────────────────────────────────

BASE_MODEL = "/models/core"  # Hardcoded — gaia-study sets BASE_MODEL_PATH for Prime training paths
TRAIN_DATA = "/gaia/GAIA_Project/knowledge/curricula/deliberation/train.json"
ADAPTER_DIR = "/models/lora_adapters/tier1_global/core_deliberation_v1"

# ── Hyperparameters ────────────────────────────────────────────────────────

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LEARNING_RATE = 2e-4
BATCH_SIZE = 1
GRAD_ACCUM = 4
WARMUP_STEPS = 10
MAX_SEQ_LENGTH = 2048  # Deliberation outputs are long (think + final)
TARGET_LOSS = 0.10
LOGGING_STEPS = 5

# Modules to NOT NF4-quantize. Gemma 4 multimodal towers use a custom
# Gemma4ClippableLinear class with QAT calibration buffers; double-
# quantizing through BnB breaks the forward pass. Per persistent memory.
SKIP_MODULES = ["lm_head", "vision_tower", "audio_tower",
                "embed_vision", "embed_audio"]


# ── Gemma 4 chat formatting ────────────────────────────────────────────────

def format_pair(instruction: str, output: str) -> str:
    """Format a training pair in Gemma 4 chat format.

    Mirrors ChatFormatter.format_conversation in the engine's core.py so
    the trained model sees the same tokens at training and inference.
    """
    return (
        f"<|turn>user<turn|>\n{instruction}\n"
        f"<|turn>assistant<turn|>\n{output}<turn|>"
    )


def build_dataset(train_path: str, tokenizer):
    """Tokenize the deliberation curriculum into a list-of-dicts dataset.

    Each row: {input_ids, attention_mask, labels}. Labels mirror input_ids
    verbatim — no prompt-token masking here because for k23 we want the
    model to learn the entire output shape (think + final), not just the
    final response. Masking the think block would teach Core to ignore
    its own thinking, which defeats the curriculum.
    """
    with open(train_path) as f:
        rows = json.load(f)

    samples = []
    too_long = 0
    for r in rows:
        text = format_pair(r["instruction"], r["output"])
        tok = tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=MAX_SEQ_LENGTH,
        )
        ids = tok["input_ids"].squeeze(0)
        mask = tok["attention_mask"].squeeze(0)
        if ids.shape[0] >= MAX_SEQ_LENGTH:
            too_long += 1
        samples.append({
            "input_ids": ids,
            "attention_mask": mask,
            "labels": ids.clone(),
        })
    log.info("Built %d samples (%d truncated to %d tokens)",
             len(samples), too_long, MAX_SEQ_LENGTH)
    return samples


class TextCollator:
    """Pad to longest in batch, set labels=-100 on pad tokens."""
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        import torch
        max_len = max(b["input_ids"].shape[0] for b in batch)
        ids = torch.full((len(batch), max_len), self.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        for i, b in enumerate(batch):
            n = b["input_ids"].shape[0]
            ids[i, :n] = b["input_ids"]
            mask[i, :n] = b["attention_mask"]
            labels[i, :n] = b["labels"]
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=250,
                        help="Max training steps (default 250)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build dataset, load model, exit before training")
    parser.add_argument("--base-model", default=BASE_MODEL,
                        help=f"Override base model path (default: {BASE_MODEL})")
    args = parser.parse_args()

    base_model = args.base_model

    log.info("=" * 60)
    log.info("  GAIA Core Deliberation Training (k23-efq)")
    log.info("=" * 60)
    log.info("Base model:   %s", base_model)
    log.info("Train data:   %s", TRAIN_DATA)
    log.info("Adapter out:  %s", ADAPTER_DIR)
    log.info("Steps:        %d", args.steps)
    log.info("LoRA r/alpha: %d/%d", LORA_R, LORA_ALPHA)
    log.info("Skip modules: %s", SKIP_MODULES)
    log.info("")

    # Resolve symlink for the AutoModelForCausalLM loader (it doesn't
    # always follow symlinks cleanly when reading config.json).
    base_model_real = str(Path(base_model).resolve())
    if base_model_real != base_model:
        log.info("Resolved %s → %s", base_model, base_model_real)
        base_model = base_model_real

    import torch
    if not torch.cuda.is_available():
        log.error("No CUDA GPU available")
        return 1
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    log.info("GPU: %s (%.1f GB)", torch.cuda.get_device_name(0), vram_total)

    # Force sequential weight loading to keep transient VRAM under control
    try:
        import transformers.core_model_loading as _cml
        _cml.GLOBAL_WORKERS = 1
    except Exception:
        pass

    # 1. Tokenizer (from the processor for Gemma 4 multimodal model)
    log.info("Loading processor/tokenizer...")
    from transformers import AutoProcessor
    try:
        processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
        tokenizer = processor.tokenizer
    except Exception:
        # Fallback: text-only tokenizer
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Dataset
    log.info("Building dataset...")
    samples = build_dataset(TRAIN_DATA, tokenizer)
    if not samples:
        log.error("No training samples")
        return 1

    if args.dry_run and not os.environ.get("LOAD_MODEL_IN_DRYRUN"):
        log.info("--dry-run (no model load): dataset built, exiting")
        return 0

    # 3. Load model with NF4 + skip towers
    log.info("Loading model with NF4 (towers skipped)...")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=SKIP_MODULES,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model, trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model.enable_input_require_grads()
    used_gb = torch.cuda.memory_allocated() / 1024 ** 3
    log.info("Model loaded: %.2f GB VRAM", used_gb)

    # 4. Unwrap Gemma4ClippableLinear in language_model subtree only.
    #    Tower layers MUST keep their native QAT wrapper.
    log.info("Unwrapping Gemma4ClippableLinear in language_model...")
    unwrapped = 0
    for name, module in list(model.named_modules()):
        if "language_model" not in name:
            continue
        for attr_name in list(vars(module).keys()):
            child = getattr(module, attr_name, None)
            if child is not None and type(child).__name__ == "Gemma4ClippableLinear":
                inner = getattr(child, "linear", None)
                if inner is not None:
                    setattr(module, attr_name, inner)
                    unwrapped += 1
        for child_name, child in list(module._modules.items()):
            if type(child).__name__ == "Gemma4ClippableLinear":
                inner = getattr(child, "linear", None)
                if inner is not None:
                    module._modules[child_name] = inner
                    unwrapped += 1
    log.info("  unwrapped %d LM Gemma4ClippableLinear layers", unwrapped)

    # 5. LoRA on language_model only (regex avoids tower modules)
    log.info("Applying LoRA (language_model only)...")
    from peft import LoraConfig, get_peft_model, TaskType

    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=r".*language_model\.layers\.\d+\.(?:self_attn|mlp)\."
                       r"(?:q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$",
        task_type=TaskType.CAUSAL_LM, bias="none",
    )
    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    log.info("  trainable: %d / %d (%.3f%%)", trainable, total,
             100 * trainable / total)

    if args.dry_run:
        log.info("--dry-run: model + LoRA loaded, exiting before train loop")
        return 0

    # 6. Trainer
    log.info("Setting up trainer...")
    from transformers import Trainer, TrainingArguments

    training_args = TrainingArguments(
        output_dir=ADAPTER_DIR,
        max_steps=args.steps,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        logging_steps=LOGGING_STEPS,
        save_steps=max(50, args.steps // 4),
        save_strategy="steps",
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        remove_unused_columns=False,
    )

    collator = TextCollator(pad_token_id=tokenizer.pad_token_id)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=samples,
        data_collator=collator,
    )

    log.info("Training for up to %d steps...", args.steps)
    trainer.train()

    # 7. Save adapter
    log.info("Saving LoRA adapter to %s", ADAPTER_DIR)
    Path(ADAPTER_DIR).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)

    # Write a small manifest for downstream tooling
    manifest = {
        "name": "core_deliberation_v1",
        "base_model": base_model,
        "epic": "k23",
        "issue": "efq",
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "training_data": TRAIN_DATA,
        "skip_modules": SKIP_MODULES,
        "max_steps": args.steps,
    }
    with open(Path(ADAPTER_DIR) / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    log.info("Done. Adapter at %s", ADAPTER_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
