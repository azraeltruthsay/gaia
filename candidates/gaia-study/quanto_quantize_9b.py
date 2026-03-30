"""Quantize Qwen3.5-9B to int4 using optimum-quanto on CPU.

This creates a pre-quantized model that can be loaded directly to GPU
for QLoRA training without the bf16→NF4 peak VRAM issue.

Run inside gaia-study container:
    docker compose exec -T gaia-study python /app/quanto_quantize_9b.py
"""

import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GAIA.Quanto")

MODEL_PATH = "/models/Huihui-Qwen3.5-9B-abliterated"
SAVE_PATH = "/models/Huihui-Qwen3.5-9B-abliterated-quanto-int4"

logger.info("Loading model to CPU (bf16)...")
start = time.time()

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# Load config to detect multimodal
config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
if hasattr(config, "vision_config") and config.vision_config is not None:
    from transformers import AutoModelForImageTextToText
    auto_cls = AutoModelForImageTextToText
    logger.info("Multimodal model detected")
else:
    auto_cls = AutoModelForCausalLM

model = auto_cls.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    device_map="cpu",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)
logger.info("Model loaded to CPU in %.1fs", time.time() - start)

# Quantize with quanto int4 on CPU
logger.info("Quantizing to int4 on CPU...")
from optimum.quanto import quantize, qint4, freeze

quantize(model, weights=qint4)
freeze(model)
logger.info("Quantization complete")

# Save
logger.info("Saving quantized model to %s...", SAVE_PATH)
model.save_pretrained(SAVE_PATH)

# Copy tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.save_pretrained(SAVE_PATH)

logger.info("Done! Quantized model saved to %s", SAVE_PATH)
logger.info("Total time: %.1fs", time.time() - start)
