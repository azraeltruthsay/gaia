"""AWQ quantization of Qwen3.5-9B-Abliterated using llm-compressor.

Run inside gaia-study container:
    docker compose exec -T gaia-study python /scripts/awq_quantize_qwen35_9b.py

Produces: /models/Qwen3.5-9B-Abliterated-AWQ/
"""

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier

MODEL_ID = "/models/Qwen3.5-9B-Abliterated"
SAVE_DIR = "/models/Qwen3.5-9B-Abliterated-AWQ"

NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 512

print(f"Loading model from {MODEL_ID}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# Build a simple calibration dataset from varied prompts.
calibration_texts = [
    "The quick brown fox jumps over the lazy dog.",
    "In machine learning, backpropagation is used to compute gradients.",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    "The capital of France is Paris, which is known for the Eiffel Tower.",
    "Explain the difference between TCP and UDP protocols.",
    "Write a Python function that sorts a list using quicksort.",
    "What are the main components of a transformer neural network?",
    "The mitochondria is the powerhouse of the cell.",
    "SELECT u.name, COUNT(o.id) FROM users u JOIN orders o ON u.id = o.user_id GROUP BY u.name;",
    "In quantum mechanics, the Heisenberg uncertainty principle states that...",
    "How do you implement a binary search tree in Python?",
    "The French Revolution began in 1789 and fundamentally changed European politics.",
    "import torch\nimport torch.nn as nn\n\nclass Attention(nn.Module):\n    def __init__(self, dim, heads=8):\n        super().__init__()",
    "Describe the process of photosynthesis in plants.",
    "What is the time complexity of merge sort and why?",
    "The Renaissance was a period of cultural rebirth in Europe.",
] * 16  # Repeat to get 256 samples

calibration_texts = calibration_texts[:NUM_CALIBRATION_SAMPLES]

# Build HuggingFace Dataset (required by llm-compressor)
print(f"Preparing {len(calibration_texts)} calibration samples...")
ds = Dataset.from_dict({"text": calibration_texts})

def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )

ds = ds.map(tokenize, remove_columns=["text"])

# Configure AWQ recipe
recipe = [
    AWQModifier(
        ignore=["lm_head"],
        scheme="W4A16_ASYM",
        targets=["Linear"],
    ),
]

print("Running AWQ quantization (this may take 10-20 minutes)...")
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# Quick sanity check
print("\n========== SAMPLE GENERATION ==============")
input_ids = tokenizer("Hello, I am GAIA, a sovereign AI system. My purpose is", return_tensors="pt").input_ids.to(model.device)
with torch.no_grad():
    output = model.generate(input_ids, max_new_tokens=50)
print(tokenizer.decode(output[0]))
print("==========================================\n")

# Save quantized model
print(f"Saving quantized model to {SAVE_DIR}...")
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
print(f"Done! Model saved to {SAVE_DIR}")
