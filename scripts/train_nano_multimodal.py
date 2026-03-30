"""Train Qwen3.5-0.8B base → GAIA Nano Reflex (multimodal, identity-baked).

Applies QLoRA to the text backbone only, preserving native vision capability.
Uses the nano-multimodal curriculum (identity + triage + tool awareness +
think suppression + dissociation + vision maintenance).

Run inside gaia-study container:
    docker exec gaia-study python3 /gaia/GAIA_Project/scripts/train_nano_multimodal.py
"""
import logging
import json
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GAIA.Train.NanoMultimodal")

MODEL_PATH = "/models/Qwen/Qwen3.5-0.8B"
DATASET_PATH = "/gaia/GAIA_Project/knowledge/curricula/nano-multimodal/train.jsonl"
OUTPUT_DIR = "/models/lora_adapters/nano-multimodal-v2"

# ── Load model ───────────────────────────────────────────────────────────────

logger.info("Loading Qwen3.5-0.8B (bf16, 4-bit quantized for training)...")
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer = processor.tokenizer

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="sdpa",
)

gpu_gb = torch.cuda.memory_allocated(0) / (1024**3)
logger.info("Model loaded! GPU: %.2fGB", gpu_gb)

# ── Apply LoRA (text layers only — vision encoder untouched) ─────────────────

from peft import LoraConfig, get_peft_model, TaskType

# Target ONLY the language model's attention + FFN projections
# This preserves the vision encoder (model.visual) completely
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.0,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    # Only apply to the language model, not the vision encoder
    modules_to_save=None,
)

model = get_peft_model(model, lora_config)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
logger.info("LoRA applied: %d trainable / %d total (%.2f%%)",
            trainable, total, 100 * trainable / total)

# ── Load dataset ─────────────────────────────────────────────────────────────

from datasets import Dataset

samples = []
with open(DATASET_PATH) as f:
    for line in f:
        d = json.loads(line)
        # Format as ChatML (same as existing pipeline)
        text = f"<|im_start|>user\n{d['instruction']}<|im_end|>\n<|im_start|>assistant\n{d['output']}<|im_end|>"
        samples.append({"text": text})

dataset = Dataset.from_list(samples)
logger.info("Dataset: %d samples", len(dataset))

# Category distribution
cats = {}
with open(DATASET_PATH) as f:
    for line in f:
        d = json.loads(line)
        c = d.get("category", "unknown")
        cats[c] = cats.get(c, 0) + 1
for k, v in sorted(cats.items(), key=lambda x: -x[1]):
    logger.info("  %s: %d", k, v)

# ── Train ────────────────────────────────────────────────────────────────────

from trl import SFTTrainer, SFTConfig

def formatting_func(example):
    return example["text"]

# 265 samples × 3 epochs ÷ batch_size(1) ÷ grad_accum(4) ≈ 200 steps
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    formatting_func=formatting_func,
    args=SFTConfig(
        output_dir=OUTPUT_DIR,
        max_length=512,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=6,
        learning_rate=2e-4,
        warmup_steps=10,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        bf16=True,
        optim="adamw_8bit",
        seed=42,
        report_to="none",
    ),
)

logger.info("Starting training (3 epochs, ~200 steps)...")
start = time.time()
trainer.train()
elapsed = time.time() - start
logger.info("Training complete in %.1fs (%.1f min)", elapsed, elapsed / 60)

# ── Save adapter ─────────────────────────────────────────────────────────────

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Save training metadata
meta = {
    "base_model": MODEL_PATH,
    "adapter_dir": OUTPUT_DIR,
    "dataset": DATASET_PATH,
    "dataset_size": len(samples),
    "categories": cats,
    "training_time_s": round(elapsed, 1),
    "lora_r": 8,
    "lora_alpha": 16,
    "epochs": 3,
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "notes": "Nano Reflex multimodal — identity baked, vision preserved, triage trained",
}
with open(f"{OUTPUT_DIR}/training_metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

logger.info("Adapter saved to %s", OUTPUT_DIR)
logger.info("Next: merge adapter with base model, then validate vision + identity")

# Cleanup
del model, trainer
import gc
gc.collect()
torch.cuda.empty_cache()
