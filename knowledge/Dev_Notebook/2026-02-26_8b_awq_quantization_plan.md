# 8B AWQ Quantization Plan

**Status:** Blocked — needs llm-compressor installed in gaia-prime container
**Date:** 2026-02-26
**Priority:** Next after current fixes validated

## Goal

Download `huihui-ai/Huihui-Qwen3-8B-abliterated-v2` (BF16, ~16GB) and quantize to AWQ 4-bit locally on RTX 5080.

## Why This Model

- **huihui-ai v2** fixes the garbled text / layer-0 encoding bug from mlabonne's v1 abliteration
- **33k downloads/month** — well-tested by community
- No pre-made AWQ version exists, so we quantize ourselves
- **Abliterated** = safety refusals removed, which GAIA needs for unrestricted persona
- 8B-AWQ at ~5-6GB VRAM fits comfortably in 11.4GB budget (RTX 5080 @ 0.70 utilization)

## What Failed

AutoAWQ is **deprecated** and incompatible with the transformers version in gaia-prime (newer than 4.51.3).

```
ImportError: cannot import name 'PytorchGELUTanh' from 'transformers.activations'
```

AutoAWQ's own README points to **llm-compressor** (vLLM project) as the successor.

## Plan: Use llm-compressor

### Step 1: Install llm-compressor in gaia-prime container

Either:
- Add `llmcompressor` to gaia-prime's requirements/Dockerfile, or
- Install at runtime in a one-off container: `pip install llmcompressor`

### Step 2: Rewrite quantize_awq.py

Replace AutoAWQ with llm-compressor's AWQ recipe:

```python
"""
Quantize huihui-ai/Huihui-Qwen3-8B-abliterated-v2 to AWQ 4-bit
using llm-compressor (vLLM project).
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor import oneshot

MODEL_ID = "huihui-ai/Huihui-Qwen3-8B-abliterated-v2"
OUTPUT_DIR = "/models/staging/Huihui-Qwen3-8B-abliterated-v2-AWQ"

# Load BF16 model
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# AWQ recipe — W4A16 asymmetric, skip lm_head
recipe = [
    AWQModifier(
        ignore=["lm_head"],
        scheme="W4A16_ASYM",
        targets=["Linear"],
    ),
]

# Run quantization (uses calibration data internally)
oneshot(
    model=model,
    tokenizer=tokenizer,
    recipe=recipe,
    output_dir=OUTPUT_DIR,
)

print(f"Done! AWQ model saved to {OUTPUT_DIR}")
```

### Step 3: Run quantization

```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /gaia/GAIA_Project/gaia-models:/models \
  --name gaia-quantize --rm \
  localhost:5000/gaia-prime:local \
  bash -c "pip install llmcompressor && python3 /models/staging/quantize_awq.py"
```

Expected: ~20-30 min on RTX 5080. Output: ~5GB AWQ model in `/gaia/GAIA_Project/gaia-models/staging/Huihui-Qwen3-8B-abliterated-v2-AWQ/`

### Step 4: Deploy

1. Copy AWQ model to warm pool: `cp -r staging/Huihui-Qwen3-8B-abliterated-v2-AWQ /mnt/gaia_warm_pool/`
2. Update `gaia_constants.json` — `gpu_prime.path` → `/models/Huihui-Qwen3-8B-abliterated-v2-AWQ`
3. Update vLLM launch args if model name format differs
4. Clear old model from warm pool if space is tight (currently 2.5GB free, need ~5GB)
5. Restart gaia-prime

### Step 5: Validate

- Health check: `curl localhost:7777/health`
- Test generation: send a prompt via gaia-core
- Run cognitive smoke tests against new model
- Check VRAM usage: `nvidia-smi`

## Disk Space

- `/gaia` has 342GB free (plenty for BF16 download + AWQ output)
- Warm pool tmpfs: 2.5GB free — will need to remove old 4B model first (it's ~7.5GB)
- Final AWQ model: ~5GB

## Existing Script

`/gaia/GAIA_Project/gaia-models/staging/quantize_awq.py` — currently has the failed AutoAWQ version, needs rewrite per Step 2.
