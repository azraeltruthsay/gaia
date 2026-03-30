"""GPTQ 4-bit quantization of Qwen3.5-9B-Abliterated using GPTQModel.

Qwen3.5-9B is a multimodal model (Qwen3_5ForConditionalGeneration) with a
nested text_config. gptqmodel's default BaseQModel uses AutoModelForCausalLM
which creates Qwen3_5ForCausalLM — that class passes the composite config
directly to Qwen3_5TextModel, causing AttributeError on layer_types.

Fix: register a custom model definition that uses AutoModelForImageTextToText,
which routes through Qwen3_5ForConditionalGeneration and correctly splits
text_config/vision_config.

Run inside gaia-study container:
    docker compose exec -T gaia-study python /scripts/gptq_quantize_qwen35_9b.py
"""

import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GAIA.GPTQ")

# Register Qwen3.5 model definition BEFORE loading
sys.path.insert(0, "/app")
from gaia_study.merge_and_requantize import _register_qwen3_5
_register_qwen3_5()

from gptqmodel import GPTQModel, QuantizeConfig

MODEL_ID = "/models/Huihui-Qwen3.5-9B-abliterated"
SAVE_DIR = "/models/Huihui-Qwen3.5-9B-abliterated-GPTQ-4bit"

# All 4-bit, no 8-bit lm_head exception
quant_config = QuantizeConfig(
    bits=4,
    group_size=128,
    sym=True,
    desc_act=False,
    lm_head=False,  # Don't quantize lm_head — let it stay in model dtype
)

print(f"Loading model from {MODEL_ID}...")
model = GPTQModel.load(
    MODEL_ID,
    quant_config,
    trust_remote_code=True,
)

# Simple calibration data
calibration_data = [
    "The quick brown fox jumps over the lazy dog.",
    "In machine learning, backpropagation is used to compute gradients.",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    "The capital of France is Paris, which is known for the Eiffel Tower.",
    "Explain the difference between TCP and UDP protocols.",
    "Write a Python function that sorts a list using quicksort.",
    "What are the main components of a transformer neural network?",
    "The mitochondria is the powerhouse of the cell.",
    "SELECT u.name, COUNT(o.id) FROM users u JOIN orders o ON u.id = o.user_id GROUP BY u.name;",
    "In quantum mechanics, the Heisenberg uncertainty principle states that position and momentum cannot both be precisely known.",
    "How do you implement a binary search tree in Python?",
    "The French Revolution began in 1789 and fundamentally changed European politics.",
    "import torch; import torch.nn as nn; class Attention(nn.Module): pass",
    "Describe the process of photosynthesis in plants.",
    "What is the time complexity of merge sort and why?",
    "The Renaissance was a period of cultural rebirth in Europe.",
    "Docker containers provide isolated environments for running applications.",
    "Neural networks consist of layers of interconnected nodes that process information.",
    "The HTTP protocol is stateless and uses request-response pairs.",
    "Python decorators are functions that modify the behavior of other functions.",
    "Kubernetes orchestrates containerized workloads across clusters of machines.",
    "The Big Bang theory describes the origin and evolution of the universe.",
    "Git version control tracks changes in source code during software development.",
    "Transformers use self-attention mechanisms to process sequential data in parallel.",
]

print(f"Quantizing with {len(calibration_data)} calibration samples...")
model.quantize(calibration_data)

print(f"Saving to {SAVE_DIR}...")
model.save(SAVE_DIR)

print(f"Done! Quantized model saved to {SAVE_DIR}")
