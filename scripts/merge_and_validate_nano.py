"""Merge Nano LoRA adapter with base model, then validate identity + vision.

Steps:
1. Load base Qwen3.5-0.8B + LoRA adapter
2. Merge weights
3. Save merged model
4. Run cognitive eval on merged model
5. Compare against baseline

Run inside gaia-study:
    docker exec gaia-study python3 /gaia/GAIA_Project/scripts/merge_and_validate_nano.py
"""
import gc
import json
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GAIA.Merge.Nano")

BASE_MODEL = "/models/Qwen/Qwen3.5-0.8B"
ADAPTER_DIR = "/models/lora_adapters/nano-multimodal-v4"
MERGED_DIR = "/models/Qwen3.5-0.8B-GAIA-Nano-Multimodal-v4"

# ── Phase 1: Merge ──────────────────────────────────────────────────────────

logger.info("Phase 1: Loading base model + adapter for merge...")
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel

processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer = processor.tokenizer

# Load base in bf16 on CPU for merge (saves GPU memory)
model = AutoModelForImageTextToText.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    low_cpu_mem_usage=True,
)
logger.info("Base model loaded to CPU")

# Apply adapter
model = PeftModel.from_pretrained(model, ADAPTER_DIR)
logger.info("Adapter loaded from %s", ADAPTER_DIR)

# Merge and unload adapter (folds LoRA weights into base)
model = model.merge_and_unload()
logger.info("Adapter merged into base weights")

# Save merged model
Path(MERGED_DIR).mkdir(parents=True, exist_ok=True)
model.save_pretrained(MERGED_DIR, safe_serialization=True)
processor.save_pretrained(MERGED_DIR)
logger.info("Merged model saved to %s", MERGED_DIR)

# Check size
import subprocess
size = subprocess.check_output(["du", "-sh", MERGED_DIR]).decode().split()[0]
logger.info("Merged model size: %s", size)

# Free merge model from RAM
del model
gc.collect()

# ── Phase 2: Validate ───────────────────────────────────────────────────────

logger.info("Phase 2: Loading merged model for validation...")

model = AutoModelForImageTextToText.from_pretrained(
    MERGED_DIR,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
    device_map="cuda",
)
model.eval()
processor = AutoProcessor.from_pretrained(MERGED_DIR, trust_remote_code=True)
tokenizer = processor.tokenizer

vram = torch.cuda.memory_allocated() / (1024**2)
logger.info("Merged model loaded on GPU (VRAM: %.0fMB)", vram)

# ── Quick validation tests ───────────────────────────────────────────────

from PIL import Image, ImageDraw
import re

def generate_text(prompt, max_tokens=150):
    full = (f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n")
    ids = tokenizer.encode(full, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_tokens, temperature=0.7, do_sample=True)
    text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()

def generate_vision(image, prompt, max_tokens=150):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}]
    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text_input], images=[image], return_tensors="pt", padding=True)
    inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_tokens, temperature=0.7, do_sample=True)
    text = processor.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()

# Create test image
img = Image.new("RGB", (224, 224), "white")
draw = ImageDraw.Draw(img)
draw.rectangle([20, 20, 100, 100], fill="red", outline="black")
draw.ellipse([120, 50, 200, 180], fill="blue", outline="black")
draw.text((30, 190), "GAIA Test", fill="darkgreen")

results = {}

# ── Identity tests ───────────────────────────────────────────────────────
tests = [
    ("identity_name", "What is your name?"),
    ("identity_role", "What is your role in the system?"),
    ("identity_tiers", "What are your cognitive tiers?"),
    ("identity_model", "What model are you based on?"),
]

logger.info("\n=== IDENTITY TESTS ===")
for test_id, prompt in tests:
    resp = generate_text(prompt)
    has_gaia = "gaia" in resp.lower()
    results[test_id] = {"prompt": prompt, "response": resp[:300], "has_gaia": has_gaia}
    icon = "✓" if has_gaia else "✗"
    logger.info("  [%s] %s: %s", icon, test_id, resp[:100].replace("\n", " "))

# ── Triage tests ─────────────────────────────────────────────────────────
triage_tests = [
    ("triage_simple", "Classify: Hello!", "simple"),
    ("triage_complex", "Classify: Write a database migration script with rollback support.", "complex"),
    ("triage_simple2", "Classify: What time is it?", "simple"),
]

logger.info("\n=== TRIAGE TESTS ===")
for test_id, prompt, expected in triage_tests:
    resp = generate_text(prompt, max_tokens=50)
    has_expected = expected in resp.lower()
    has_think = "<think>" in resp
    results[test_id] = {"prompt": prompt, "response": resp[:200], "correct": has_expected, "has_think": has_think}
    icon = "✓" if has_expected else "✗"
    logger.info("  [%s] %s: %s (think: %s)", icon, test_id, resp[:80].replace("\n", " "), has_think)

# ── Vision tests ─────────────────────────────────────────────────────────
vision_tests = [
    ("vision_describe", "Describe this image."),
    ("vision_text", "What text can you see?"),
    ("vision_colors", "What colors are in this image?"),
]

logger.info("\n=== VISION TESTS ===")
for test_id, prompt in vision_tests:
    resp = generate_vision(img, prompt)
    has_content = len(resp) > 10
    results[test_id] = {"prompt": prompt, "response": resp[:300], "has_content": has_content}
    icon = "✓" if has_content else "✗"
    logger.info("  [%s] %s: %s", icon, test_id, resp[:100].replace("\n", " "))

# ── Dissociation test ────────────────────────────────────────────────────
logger.info("\n=== DISSOCIATION TEST ===")
resp = generate_text("Explain how TCP/IP networking works.")
has_gaia_bleed = any(m in resp.lower() for m in ["gaia", "cognitive", "consciousness", "orchestrator", "tier"])
results["dissociation_tcp"] = {"prompt": "TCP/IP", "response": resp[:300], "identity_bleed": has_gaia_bleed}
icon = "✓" if not has_gaia_bleed else "✗"
logger.info("  [%s] dissociation: bleed=%s — %s", icon, has_gaia_bleed, resp[:100].replace("\n", " "))

# ── Summary ──────────────────────────────────────────────────────────────
identity_pass = sum(1 for k, v in results.items() if k.startswith("identity_") and v.get("has_gaia"))
triage_pass = sum(1 for k, v in results.items() if k.startswith("triage_") and v.get("correct"))
vision_pass = sum(1 for k, v in results.items() if k.startswith("vision_") and v.get("has_content"))
dissociation_pass = 0 if results.get("dissociation_tcp", {}).get("identity_bleed") else 1

logger.info("\n" + "═" * 60)
logger.info("VALIDATION SUMMARY — Qwen3.5-0.8B-GAIA-Nano-Multimodal-v1")
logger.info("═" * 60)
logger.info("  Identity:     %d/4  %s", identity_pass, "✓" if identity_pass >= 3 else "needs work")
logger.info("  Triage:       %d/3  %s", triage_pass, "✓" if triage_pass >= 2 else "needs work")
logger.info("  Vision:       %d/3  %s", vision_pass, "✓" if vision_pass >= 2 else "REGRESSION")
logger.info("  Dissociation: %d/1  %s", dissociation_pass, "✓" if dissociation_pass else "bleed detected")

# Save
output_path = Path("/gaia/GAIA_Project/artifacts/eval/qwen3.5-0.8b-nano-v1/validation.json")
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w") as f:
    json.dump({
        "model": MERGED_DIR,
        "base": BASE_MODEL,
        "adapter": ADAPTER_DIR,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "summary": {
            "identity": f"{identity_pass}/4",
            "triage": f"{triage_pass}/3",
            "vision": f"{vision_pass}/3",
            "dissociation": f"{dissociation_pass}/1",
        },
        "results": results,
    }, f, indent=2, default=str)
logger.info("Results saved to %s", output_path)

# Cleanup
del model
gc.collect()
torch.cuda.empty_cache()
