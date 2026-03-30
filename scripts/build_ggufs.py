"""Convert merged models to GGUF and quantize for CPU fallback.

Converts safetensors → BF16 GGUF → quantized GGUF for each tier.

Run inside gaia-study:
    docker exec gaia-study python3 /gaia/GAIA_Project/scripts/build_ggufs.py
"""
import logging
import os
import subprocess
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GAIA.GGUF")

CONVERT_SCRIPT = "/opt/llama.cpp/convert_hf_to_gguf.py"
QUANTIZE_BIN = "/usr/local/bin/llama-quantize"

MODELS = [
    {
        "name": "Nano",
        "input": "/models/Qwen3.5-0.8B-GAIA-Nano-Multimodal-v6",
        "bf16_gguf": "/models/Qwen3.5-0.8B-GAIA-Nano-v5-BF16.gguf",
        "quant_gguf": "/models/Qwen3.5-0.8B-GAIA-Nano-v5-Q8_0.gguf",
        "quant_type": "Q8_0",  # Nano is small enough for Q8
    },
    {
        "name": "Core",
        "input": "/models/Qwen3.5-4B-GAIA-Core-Multimodal-v4",
        "bf16_gguf": "/models/Qwen3.5-4B-GAIA-Core-v3-BF16.gguf",
        "quant_gguf": "/models/Qwen3.5-4B-GAIA-Core-v3-Q4_K_M.gguf",
        "quant_type": "Q4_K_M",  # 4B needs Q4 to fit alongside other models
    },
    {
        "name": "Prime",
        "input": "/models/Qwen3-8B-GAIA-Prime-v1",
        "bf16_gguf": "/models/Qwen3-8B-GAIA-Prime-v1-BF16.gguf",
        "quant_gguf": "/models/Qwen3-8B-GAIA-Prime-v1-Q4_K_M.gguf",
        "quant_type": "Q4_K_M",
    },
]


def run(cmd, desc):
    """Run a command and stream output."""
    logger.info("Running: %s", desc)
    logger.info("  CMD: %s", " ".join(cmd))
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start
    if result.returncode != 0:
        logger.error("FAILED (%s): %s", desc, result.stderr[-500:])
        return False
    logger.info("  Done in %.1fs", elapsed)
    return True


def convert_and_quantize(model_cfg):
    """Convert one model to GGUF and quantize."""
    name = model_cfg["name"]
    input_dir = model_cfg["input"]
    bf16_path = model_cfg["bf16_gguf"]
    quant_path = model_cfg["quant_gguf"]
    quant_type = model_cfg["quant_type"]

    if not os.path.isdir(input_dir):
        logger.warning("Skipping %s — input not found: %s", name, input_dir)
        return False

    logger.info("")
    logger.info("═" * 50)
    logger.info("Converting %s", name)
    logger.info("═" * 50)

    # Step 1: Convert HF safetensors → BF16 GGUF
    if os.path.exists(bf16_path):
        logger.info("BF16 GGUF already exists, skipping conversion")
    else:
        ok = run(
            [sys.executable, CONVERT_SCRIPT, input_dir,
             "--outfile", bf16_path, "--outtype", "bf16"],
            f"{name}: safetensors → BF16 GGUF"
        )
        if not ok:
            return False

    # Step 2: Quantize BF16 GGUF → target quantization
    if os.path.exists(quant_path):
        logger.info("Quantized GGUF already exists, skipping")
    else:
        ok = run(
            [QUANTIZE_BIN, bf16_path, quant_path, quant_type],
            f"{name}: BF16 → {quant_type}"
        )
        if not ok:
            return False

    # Report sizes
    bf16_size = os.path.getsize(bf16_path) / (1024**3) if os.path.exists(bf16_path) else 0
    quant_size = os.path.getsize(quant_path) / (1024**3) if os.path.exists(quant_path) else 0
    logger.info("  BF16:  %.2f GB", bf16_size)
    logger.info("  %s: %.2f GB", quant_type, quant_size)

    # Clean up BF16 GGUF to save space (we only need the quantized version)
    if os.path.exists(quant_path) and quant_size > 0:
        os.remove(bf16_path)
        logger.info("  Removed BF16 GGUF (keeping only %s)", quant_type)

    return True


for model_cfg in MODELS:
    try:
        convert_and_quantize(model_cfg)
    except Exception as e:
        logger.error("Error converting %s: %s", model_cfg["name"], e)
        continue

logger.info("")
logger.info("All GGUF conversions complete.")
