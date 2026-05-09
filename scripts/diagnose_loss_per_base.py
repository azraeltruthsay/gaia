#!/usr/bin/env python3
"""Compare first-step training loss on different bases.

Question: why does Core 2.1 (base = Unified-v5-Multimodal) hit loss 60-25
while Core 2.0 (base = google/gemma-4-E4B raw) hit loss ~5? Same LoRA
config, same curriculum, same script. Either:
  - Base produces fundamentally different output distribution
  - Tower handling drops calibration (Gemma4ClippableLinear wrapper)
  - Tokenizer / chat template differs

This script loads each base, applies LoRA the same way, runs ONE forward
pass on a few text-only samples, and reports the per-sample loss. Pure
text removes audio/vision tower complications.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM, AutoProcessor,
    BitsAndBytesConfig,
)


def evaluate(base_model: str, samples_path: str = None, n: int = 5) -> None:
    print(f"\n=== {base_model} ===")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=["lm_head", "vision_tower", "audio_tower",
                               "embed_vision", "embed_audio"],
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model, trust_remote_code=True,
        quantization_config=bnb_config,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.config.use_cache = False
    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    tokenizer = processor.tokenizer

    print(f"VRAM after load: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    # Load text samples — first n from a JSONL file with {prompt, response}
    if samples_path is None:
        samples_path = "/gaia/GAIA_Project/knowledge/curricula/core2/text.jsonl"
    samples = []
    with open(samples_path) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            samples.append(json.loads(line))

    # Run one forward pass per sample, no LoRA, just measure base CE on the
    # response tokens given the prompt as context.
    losses = []
    for s in samples:
        prompt = s.get("prompt") or s.get("instruction") or ""
        response = s.get("response") or s.get("output") or ""
        full = prompt + response
        ids = tokenizer(full, return_tensors="pt").input_ids.cuda()
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        labels = ids.clone()
        # Mask the prompt so loss only covers response
        labels[:, :prompt_ids.shape[1]] = -100
        with torch.no_grad():
            out = model(input_ids=ids, labels=labels)
        losses.append(out.loss.item())
        print(f"  prompt[:60]={prompt[:60]!r} loss={out.loss.item():.3f}")

    print(f"  mean loss over {len(losses)} text samples: {sum(losses)/len(losses):.3f}")
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bases", nargs="+", required=True)
    parser.add_argument("--n", type=int, default=5)
    args = parser.parse_args()
    for base in args.bases:
        evaluate(base, n=args.n)
