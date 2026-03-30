#!/usr/bin/env python3
"""SAE Baseline Scan — map text + vision feature space for Qwen3.5 base models.

Runs on host (not in Docker) with direct GPU access. Loads each model,
records activations from both text and vision prompts, trains SAE atlases,
and saves them as reference baselines.

Usage:
    # Scan 0.8B (fast, ~5 min)
    python scripts/sae_baseline_scan.py --model 0.8B

    # Scan 4B (slower, ~15-20 min)
    python scripts/sae_baseline_scan.py --model 4B

    # Both
    python scripts/sae_baseline_scan.py --model both

Output:
    /gaia/gaia-instance/artifacts/sae_atlas/qwen3.5-0.8b-base/
    /gaia/gaia-instance/artifacts/sae_atlas/qwen3.5-4b-base/
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("GAIA.SAE.BaselineScan")

# ── Model configs ────────────────────────────────────────────────────────────

MODELS = {
    "0.8B": {
        "path": "/models/Qwen/Qwen3.5-0.8B",
        "text_layers": [0, 6, 12, 18, 23],       # 24 layers total
        "vision_layers": [0, 3, 6, 9, 11],        # 12 vision layers
        "num_features": 2048,                       # 2x hidden (1024)
        "atlas_dir": "/gaia/GAIA_Project/artifacts/sae_atlas/qwen3.5-0.8b-base",
    },
    "4B": {
        "path": "/models/Qwen/Qwen3.5-4B",
        "text_layers": [0, 8, 16, 24, 31],        # 32 layers total
        "vision_layers": [0, 5, 11, 17, 23],      # 24 vision layers (deepstack at 5,11,17)
        "num_features": 4096,                       # ~1.6x hidden (2560)
        "atlas_dir": "/gaia/GAIA_Project/artifacts/sae_atlas/qwen3.5-4b-base",
    },
    "8B": {
        "path": "/models/Qwen/Qwen3-8B",
        "text_layers": [0, 7, 14, 21, 28, 35],   # 36 layers total, sample every ~7
        "vision_layers": [],                        # Qwen3 has no vision
        "num_features": 8192,                       # 2x hidden (4096)
        "atlas_dir": "/gaia/GAIA_Project/artifacts/sae_atlas/qwen3-8b-base",
        "quantize": True,                           # Load NF4 — bf16 won't fit 16GB
    },
}

# ── Text prompts (diverse topics for broad feature coverage) ─────────────────

TEXT_PROMPTS = [
    # Identity / self-model
    "What is your name and purpose?",
    "Describe your architecture.",
    "How do you process information?",
    # General knowledge
    "Explain how photosynthesis works.",
    "What causes earthquakes?",
    "Describe the water cycle.",
    "How does electricity flow through a circuit?",
    "What is the theory of relativity?",
    # Code / technical
    "Write a Python function to sort a list.",
    "Explain how a hash table works.",
    "What is the difference between TCP and UDP?",
    "How does garbage collection work in Java?",
    # Creative / narrative
    "Tell me a short story about a fox.",
    "Write a haiku about the ocean.",
    "Describe a sunset over mountains.",
    # Reasoning
    "If all cats are animals and some animals are pets, what can we conclude?",
    "What would happen if gravity suddenly doubled?",
    "Compare and contrast democracy and monarchy.",
    # Safety / refusal (important for abliteration baseline)
    "How do I make a sandwich?",
    "Explain the history of martial arts.",
    "What are common household chemicals?",
    # Multilingual
    "Translate 'hello world' to French, Spanish, and Japanese.",
    "What is the meaning of 'carpe diem'?",
]

# ── Vision prompts (paired with test images) ─────────────────────────────────

VISION_PROMPTS = [
    "Describe this image in detail.",
    "What objects can you see in this image?",
    "What is the mood or atmosphere of this image?",
    "Is there any text visible in this image?",
    "What colors are dominant in this image?",
]


def generate_test_images():
    """Generate simple synthetic test images for vision scanning."""
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np

    images = []

    # 1. Solid color gradient
    arr = np.zeros((224, 224, 3), dtype=np.uint8)
    for y in range(224):
        arr[y, :, 0] = int(255 * y / 224)  # Red gradient
        arr[y, :, 2] = int(255 * (224 - y) / 224)  # Blue gradient
    images.append(("gradient", Image.fromarray(arr)))

    # 2. Simple shapes
    img = Image.new("RGB", (224, 224), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, 100, 100], fill="red", outline="black")
    draw.ellipse([120, 50, 200, 180], fill="blue", outline="black")
    draw.line([10, 200, 210, 150], fill="green", width=3)
    images.append(("shapes", img))

    # 3. Text on image
    img = Image.new("RGB", (224, 224), "lightyellow")
    draw = ImageDraw.Draw(img)
    draw.text((20, 90), "Hello GAIA", fill="black")
    images.append(("text", img))

    # 4. Checkerboard pattern
    arr = np.zeros((224, 224, 3), dtype=np.uint8)
    for y in range(224):
        for x in range(224):
            if (x // 28 + y // 28) % 2 == 0:
                arr[y, x] = [255, 255, 255]
    images.append(("checkerboard", Image.fromarray(arr)))

    # 5. Noise (tests robustness)
    arr = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    images.append(("noise", Image.fromarray(arr)))

    return images


def record_text_activations(model, tokenizer, prompts, layers, device):
    """Record hidden state activations from text-only prompts."""
    activations = {l: [] for l in layers}
    total_tokens = 0

    for i, prompt in enumerate(prompts):
        full = (f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                f"<|im_start|>user\n{prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n")
        ids = tokenizer.encode(full, return_tensors="pt").to(device)
        total_tokens += ids.shape[1]

        with torch.no_grad():
            out = model(ids, output_hidden_states=True)

        for layer_idx in layers:
            if layer_idx < len(out.hidden_states):
                hs = out.hidden_states[layer_idx][0].detach().cpu()
                activations[layer_idx].append(hs)

        del out
        if (i + 1) % 10 == 0:
            logger.info("  Text: %d/%d prompts (%d tokens)", i + 1, len(prompts), total_tokens)

    return activations, total_tokens


def record_vision_activations(model, processor, prompts, images, layers, device):
    """Record hidden state activations from vision+text prompts."""
    activations = {l: [] for l in layers}
    total_tokens = 0

    for img_name, image in images:
        for prompt in prompts:
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ]},
            ]

            text_input = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = processor(
                text=[text_input],
                images=[image],
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
            total_tokens += inputs["input_ids"].shape[1]

            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)

            for layer_idx in layers:
                if layer_idx < len(out.hidden_states):
                    hs = out.hidden_states[layer_idx][0].detach().cpu()
                    activations[layer_idx].append(hs)

            del out, inputs

        logger.info("  Vision: scanned image '%s' × %d prompts", img_name, len(prompts))

    return activations, total_tokens


def train_and_save_sae(activations, config, scan_type, metadata):
    """Train SAE on recorded activations and save atlas."""
    import torch.nn as nn
    import torch.nn.functional as F

    atlas_dir = Path(config["atlas_dir"]) / scan_type
    atlas_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for layer_idx, acts_list in activations.items():
        if not acts_list:
            continue

        all_acts = torch.cat(acts_list, dim=0)
        hidden_size = all_acts.shape[1]
        n_samples = all_acts.shape[0]
        num_features = config["num_features"]

        logger.info("Training SAE [%s] layer %d: %d samples × %d dims → %d features",
                     scan_type, layer_idx, n_samples, hidden_size, num_features)

        # Normalize
        mean = all_acts.mean(dim=0)
        std = all_acts.std(dim=0).clamp(min=1e-6)
        all_acts_norm = (all_acts - mean) / std

        # Build SAE
        encoder = nn.Linear(hidden_size, num_features)
        decoder = nn.Linear(num_features, hidden_size)
        with torch.no_grad():
            decoder.weight.copy_(encoder.weight.t())

        device = "cuda" if torch.cuda.is_available() else "cpu"
        encoder = encoder.to(dtype=all_acts_norm.dtype, device=device)
        decoder = decoder.to(dtype=all_acts_norm.dtype, device=device)
        all_acts_device = all_acts_norm.to(device)

        optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)

        # Train
        epochs = 50
        batch_size = 256
        start = time.time()

        for epoch in range(epochs):
            perm = torch.randperm(n_samples)
            total_loss = 0
            n_batches = 0

            for batch_start in range(0, n_samples, batch_size):
                batch = all_acts_device[perm[batch_start:batch_start + batch_size]]
                encoded = F.relu(encoder(batch))
                reconstructed = decoder(encoded)
                loss = F.mse_loss(reconstructed, batch) + 0.01 * encoded.abs().mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

            if (epoch + 1) % 10 == 0:
                with torch.no_grad():
                    test_enc = F.relu(encoder(all_acts_device[:1000]))
                    active = (test_enc.mean(dim=0) > 0.01).sum().item()
                logger.info("  Layer %d epoch %d/%d: loss=%.4f active=%d/%d",
                            layer_idx, epoch + 1, epochs, total_loss / n_batches, active, num_features)

        elapsed = time.time() - start

        # Final stats
        with torch.no_grad():
            final_enc = F.relu(encoder(all_acts_device))
            active_features = (final_enc.mean(dim=0) > 0.01).sum().item()
            top10 = final_enc.mean(dim=0).topk(10)

        # Save
        torch.save({
            "encoder_weight": encoder.weight.data.cpu(),
            "encoder_bias": encoder.bias.data.cpu(),
            "decoder_weight": decoder.weight.data.cpu(),
            "decoder_bias": decoder.bias.data.cpu(),
            "norm_mean": mean,
            "norm_std": std,
            "hidden_size": hidden_size,
            "num_features": num_features,
        }, atlas_dir / f"layer_{layer_idx}.pt")

        results[layer_idx] = {
            "samples": n_samples,
            "hidden_size": hidden_size,
            "features": num_features,
            "active_features": active_features,
            "final_loss": round(total_loss / n_batches, 4),
            "training_time_s": round(elapsed, 1),
            "top_features": top10.indices.tolist(),
        }

        # Free VRAM between layers
        del encoder, decoder, all_acts_device, all_acts_norm
        torch.cuda.empty_cache()

    # Save metadata
    metadata.update({"results": {str(k): v for k, v in results.items()}})
    with open(atlas_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Atlas saved to %s", atlas_dir)
    return results


def scan_model(model_key):
    """Full scan pipeline for one model."""
    global torch
    import torch

    config = MODELS[model_key]
    model_path = config["path"]

    logger.info("=" * 60)
    logger.info("BASELINE SCAN: Qwen3.5-%s", model_key)
    logger.info("  Model: %s", model_path)
    logger.info("  Text layers: %s", config["text_layers"])
    logger.info("  Vision layers: %s", config["vision_layers"])
    logger.info("=" * 60)

    # Load model — detect if multimodal or text-only
    import json as _json
    with open(os.path.join(model_path, "config.json")) as f:
        model_config = _json.load(f)
    is_multimodal = "vision_config" in model_config

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    logger.info("Loading model to %s (%s, multimodal=%s)...", device, dtype, is_multimodal)
    start = time.time()

    if is_multimodal:
        from transformers import AutoModelForImageTextToText, AutoProcessor
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = processor.tokenizer
        model = AutoModelForImageTextToText.from_pretrained(
            model_path, trust_remote_code=True,
            torch_dtype=dtype, attn_implementation="sdpa", device_map=device,
        )
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        processor = None
        if config.get("quantize"):
            from transformers import BitsAndBytesConfig
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                      bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=True,
                quantization_config=bnb, device_map="auto", attn_implementation="sdpa",
            )
            logger.info("Loaded with NF4 quantization (SAE captures quantized activations)")
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=True,
                torch_dtype=dtype, attn_implementation="sdpa", device_map=device,
            )
    model.eval()

    vram_mb = torch.cuda.memory_allocated() / (1024 * 1024) if device == "cuda" else 0
    logger.info("Model loaded in %.1fs (VRAM: %.0fMB)", time.time() - start, vram_mb)

    # ── Phase 1: Text activations ────────────────────────────────────────
    logger.info("Phase 1: Recording text activations...")
    text_acts, text_tokens = record_text_activations(
        model, tokenizer, TEXT_PROMPTS, config["text_layers"], device
    )
    logger.info("Text scan: %d prompts, %d tokens", len(TEXT_PROMPTS), text_tokens)

    # ── Phase 2: Vision activations (multimodal models only) ──────────────
    vision_acts = {}
    vision_tokens = 0
    if is_multimodal and config.get("vision_layers"):
        logger.info("Phase 2: Recording vision activations...")
        test_images = generate_test_images()
        vision_acts, vision_tokens = record_vision_activations(
            model, processor, VISION_PROMPTS, test_images, config["text_layers"], device
        )
        logger.info("Vision scan: %d images × %d prompts, %d tokens",
                    len(test_images), len(VISION_PROMPTS), vision_tokens)
    else:
        logger.info("Phase 2: Skipped (text-only model)")

    # ── Phase 3: Train SAEs ──────────────────────────────────────────────
    base_meta = {
        "model": model_path,
        "model_key": model_key,
        "scan_type": None,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "device": device,
        "dtype": str(dtype),
        "multimodal": is_multimodal,
    }

    # Text SAE
    logger.info("Phase 3a: Training text SAE atlas...")
    text_meta = {**base_meta, "scan_type": "text", "prompts": len(TEXT_PROMPTS), "tokens": text_tokens}
    text_results = train_and_save_sae(text_acts, config, "text", text_meta)

    vision_results = {}
    if vision_acts:
        # Vision SAE
        logger.info("Phase 3b: Training vision SAE atlas...")
        vision_meta = {**base_meta, "scan_type": "vision",
                       "images": len(test_images), "prompts_per_image": len(VISION_PROMPTS),
                       "tokens": vision_tokens}
        vision_results = train_and_save_sae(vision_acts, config, "vision", vision_meta)

    # ── Combined SAE (text + vision if available) ────────────────────────
    logger.info("Phase 3c: Training combined SAE atlas...")
    combined_acts = {}
    for layer in config["text_layers"]:
        t = text_acts.get(layer, [])
        v = vision_acts.get(layer, [])
        combined_acts[layer] = t + v

    combined_meta = {**base_meta, "scan_type": "combined",
                     "text_prompts": len(TEXT_PROMPTS), "text_tokens": text_tokens,
                     "vision_images": len(test_images), "vision_tokens": vision_tokens}
    combined_results = train_and_save_sae(combined_acts, config, "combined", combined_meta)

    # ── Cleanup ──────────────────────────────────────────────────────────
    del model, processor, tokenizer, text_acts, vision_acts, combined_acts
    gc.collect()
    torch.cuda.empty_cache()

    # Save summary
    summary = {
        "model": model_path,
        "model_key": model_key,
        "text": {str(k): v for k, v in text_results.items()},
        "vision": {str(k): v for k, v in vision_results.items()},
        "combined": {str(k): v for k, v in combined_results.items()},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    summary_path = Path(config["atlas_dir"]) / "scan_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Scan complete for Qwen3.5-%s — summary at %s", model_key, summary_path)
    return summary


def main():
    parser = argparse.ArgumentParser(description="SAE Baseline Scan for Qwen3.5 base models")
    parser.add_argument("--model", choices=["0.8B", "4B", "8B", "both"], required=True)
    args = parser.parse_args()

    targets = ["0.8B", "4B"] if args.model == "both" else [args.model]

    for model_key in targets:
        scan_model(model_key)

    logger.info("All scans complete.")


if __name__ == "__main__":
    main()
