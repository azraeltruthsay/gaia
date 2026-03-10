"""GPTQ 4-bit quantization of Qwen3.5-9B-Abliterated using GPTQModel.

All layers including lm_head quantized to 4-bit to fit in 16GB VRAM.

Run inside gaia-study container:
    docker compose exec -T gaia-study python /tmp/gptq_quantize.py
"""

from gptqmodel import GPTQModel, QuantizeConfig

MODEL_ID = "/models/Qwen3.5-9B-Abliterated"
SAVE_DIR = "/models/Qwen3.5-9B-Abliterated-GPTQ-4bit"

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
