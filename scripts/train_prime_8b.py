"""Full Prime 8B pipeline: CPU quantize → SAE scan → QLoRA train.

Loads Qwen3-8B base on CPU, quantizes with quanto int4, moves to GPU,
runs SAE baseline scan, then trains with LoRA.

Run inside gaia-study (with GPU maintenance mode active):
    docker exec gaia-study python3 /gaia/GAIA_Project/scripts/train_prime_8b.py
"""
import gc
import json
import logging
import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("PATH", os.environ.get("PATH", "") + ":/tmp/.local/bin")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GAIA.Prime.8B")

MODEL = "/models/Qwen/Qwen3-8B"
DATASET = "/gaia/GAIA_Project/knowledge/curricula/prime/train.jsonl"
ADAPTER_OUTPUT = "/models/lora_adapters/prime-qwen3-8b-v2"
ATLAS_DIR = Path("/gaia/GAIA_Project/artifacts/sae_atlas/qwen3-8b-base/text")
TARGET_LAYERS = [0, 7, 14, 21, 28, 35]

# ═══ Phase 1: Load and quantize on CPU, move to GPU ═════════════════════

logger.info("Phase 1: Load → quantize → GPU")
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

logger.info("Loading bf16 to CPU...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL, trust_remote_code=True,
    device_map="cpu", torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)

logger.info("Quantizing with quanto int4 on CPU...")
from optimum.quanto import quantize, qint4, freeze
quantize(model, weights=qint4)
freeze(model)

logger.info("Moving to GPU...")
model = model.to("cuda")
vram = torch.cuda.memory_allocated() / (1024**3)
logger.info("Model on GPU: %.2fGB VRAM", vram)

# ═══ Phase 2: SAE baseline scan ═════════════════════════════════════════

logger.info("Phase 2: SAE baseline scan (%d layers)", len(TARGET_LAYERS))

TEXT_PROMPTS = [
    "What is your name?", "Describe your architecture.",
    "Explain how photosynthesis works.", "What causes earthquakes?",
    "Write a Python function to sort a list.", "How does a compiler work?",
    "Tell me a short story about a fox.", "Write a haiku about the ocean.",
    "If all cats are animals, what can we conclude?",
    "How is a raven like a writing desk?",
    "What is the meaning of life?", "Explain quantum entanglement.",
    "How do vaccines work?", "Compare democracy and monarchy.",
    "What are the tradeoffs between microservices and monoliths?",
]

model.eval()
activations = {l: [] for l in TARGET_LAYERS}

for i, prompt in enumerate(TEXT_PROMPTS):
    full = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    ids = tokenizer.encode(full, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    for layer_idx in TARGET_LAYERS:
        if layer_idx < len(out.hidden_states):
            activations[layer_idx].append(out.hidden_states[layer_idx][0].detach().cpu())
    del out
    if (i + 1) % 5 == 0:
        logger.info("  Scanned %d/%d prompts", i + 1, len(TEXT_PROMPTS))

logger.info("Training SAE atlases...")
ATLAS_DIR.mkdir(parents=True, exist_ok=True)

for layer_idx in TARGET_LAYERS:
    acts = torch.cat(activations[layer_idx], dim=0)
    h = acts.shape[1]
    nf = 8192
    mean = acts.mean(dim=0)
    std = acts.std(dim=0).clamp(min=1e-6)
    acts_n = ((acts - mean) / std).to("cuda")

    enc = nn.Linear(h, nf).to(dtype=acts_n.dtype, device="cuda")
    dec = nn.Linear(nf, h).to(dtype=acts_n.dtype, device="cuda")
    with torch.no_grad():
        dec.weight.copy_(enc.weight.t())
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-3)

    for epoch in range(50):
        perm = torch.randperm(acts_n.shape[0])
        for bs in range(0, acts_n.shape[0], 256):
            batch = acts_n[perm[bs:bs + 256]]
            encoded = F.relu(enc(batch))
            recon = dec(encoded)
            loss = F.mse_loss(recon, batch) + 0.01 * encoded.abs().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

    torch.save({
        "encoder_weight": enc.weight.data.cpu(), "encoder_bias": enc.bias.data.cpu(),
        "decoder_weight": dec.weight.data.cpu(), "decoder_bias": dec.bias.data.cpu(),
        "norm_mean": mean, "norm_std": std, "hidden_size": h, "num_features": nf,
    }, ATLAS_DIR / f"layer_{layer_idx}.pt")

    del enc, dec, acts_n, acts
    torch.cuda.empty_cache()
    logger.info("  Layer %d SAE: loss=%.4f", layer_idx, loss.item())

with open(ATLAS_DIR / "metadata.json", "w") as f:
    json.dump({"model": MODEL, "layers": TARGET_LAYERS, "features": 8192,
               "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}, f, indent=2)
logger.info("SAE atlas saved")

# ═══ Phase 3: LoRA training ═════════════════════════════════════════════

logger.info("Phase 3: QLoRA training")

from peft import LoraConfig, get_peft_model, TaskType

model.train()
lora_config = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0, bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, lora_config)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
logger.info("LoRA: %d trainable / %d total (%.2f%%)", trainable, total_params, 100 * trainable / total_params)

# Load dataset
samples = []
with open(DATASET) as f:
    for line in f:
        d = json.loads(line)
        samples.append(
            f"<|im_start|>user\n{d['instruction']}<|im_end|>\n"
            f"<|im_start|>assistant\n{d['output']}<|im_end|>"
        )
logger.info("Dataset: %d samples", len(samples))

# Training loop
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=2e-4, weight_decay=0.01,
)

epochs = 4
grad_accum = 4
total_steps = (len(samples) * epochs) // grad_accum

logger.info("Training: %d epochs, ~%d steps", epochs, total_steps)
start = time.time()

for epoch in range(epochs):
    random.shuffle(samples)
    epoch_loss = 0
    n_batches = 0
    optimizer.zero_grad()

    for i, text in enumerate(samples):
        ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=1024).to("cuda")
        outputs = model(ids, labels=ids)
        loss = outputs.loss / grad_accum
        loss.backward()
        epoch_loss += outputs.loss.item()
        n_batches += 1

        if (i + 1) % grad_accum == 0:
            optimizer.step()
            optimizer.zero_grad()

            step = (epoch * len(samples) + i + 1) // grad_accum
            if step % 20 == 0:
                logger.info("  Step %d/%d loss=%.4f vram=%.1fGB",
                           step, total_steps, epoch_loss / n_batches,
                           torch.cuda.memory_allocated() / (1024**3))

    # Final step for remainder
    optimizer.step()
    optimizer.zero_grad()
    logger.info("  Epoch %d/%d: avg_loss=%.4f", epoch + 1, epochs, epoch_loss / n_batches)

elapsed = time.time() - start
logger.info("Training complete in %.1fs (%.1f min)", elapsed, elapsed / 60)

# Save
Path(ADAPTER_OUTPUT).mkdir(parents=True, exist_ok=True)
model.save_pretrained(ADAPTER_OUTPUT)
tokenizer.save_pretrained(ADAPTER_OUTPUT)

with open(f"{ADAPTER_OUTPUT}/training_metadata.json", "w") as f:
    json.dump({
        "base": MODEL,
        "samples": len(samples),
        "epochs": epochs,
        "time_s": round(elapsed, 1),
        "quantization": "quanto_int4",
        "vram_gb": round(torch.cuda.memory_allocated() / (1024**3), 2),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "notes": "Prime Thinker from clean Qwen3-8B base. Our own abliteration pipeline. "
                 "Identity + dissociation + creative reasoning + epistemic honesty. "
                 "No volatile operational details.",
    }, f, indent=2)

logger.info("Adapter saved to %s", ADAPTER_OUTPUT)
logger.info("Total VRAM: %.2fGB", torch.cuda.memory_allocated() / (1024**3))

del model
gc.collect()
torch.cuda.empty_cache()
logger.info("Done.")
