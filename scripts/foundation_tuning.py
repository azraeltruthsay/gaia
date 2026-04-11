"""Phase 5f: Foundation Tuning v3 — MoE disabled, direct gradient training."""
import sys, os, time, json, logging
sys.path.insert(0, "/engine")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import peft
from transformers import AutoTokenizer
from datasets import Dataset
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MODEL_PATH = "/models/google/gemma-4-26B-A4B"
OUTPUT_DIR = "/output/gemma4_foundation_lora_v1"

FOUNDATION_TARGETS = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]

print("=" * 60)
print("PHASE 5f: FOUNDATION TUNING v3 (MoE disabled, real gradients)")
print("=" * 60)

# Step 1: Load + disable MoE + zero experts
print("\n[1/5] Loading...")
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM
model = Gemma4ForCausalLM.from_pretrained(
    MODEL_PATH, dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="sdpa")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

for name, module in model.named_modules():
    if hasattr(module, "enable_moe_block"):
        module.enable_moe_block = False
    if type(module).__name__ == "Gemma4TextExperts":
        module.gate_up_proj = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        module.down_proj = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

# Step 2: LoRA + GPU
print("\n[2/5] LoRA + GPU...")
target_names = [n for n, m in model.named_modules()
                if any(n.endswith(t) for t in FOUNDATION_TARGETS)
                and isinstance(m, torch.nn.Linear) and "vision" not in n and "experts" not in n]

model = peft.get_peft_model(model, peft.LoraConfig(
    r=32, lora_alpha=64, lora_dropout=0.0, target_modules=target_names,
    bias="none", task_type="CAUSAL_LM"))

# fp32 LoRA weights + RMSNorm upcast
for n, p in model.named_parameters():
    if p.requires_grad and "lora_" in n:
        p.data = p.data.float()
for n, m in model.named_modules():
    if "RMSNorm" in type(m).__name__:
        m.float()

model.to("cuda")
model.enable_input_require_grads()
model.train()
torch.cuda.empty_cache()

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Trainable: {trainable:,}, VRAM: {torch.cuda.memory_allocated()/(1024**2):.0f}MB")

# Step 3: Dataset
print("\n[3/5] Dataset...")
from gaia_engine.core import ChatFormatter
torch.set_grad_enabled(True)
fmt = ChatFormatter(tokenizer)

samples = []
# Full persona curriculum
with open("/curricula/gaia_persona_training.jsonl") as f:
    for line in f:
        samples.append(json.loads(line))
# Full primary school curriculum (identity + tools + other)
with open("/curricula/primary_school/train_v2_combined.json") as f:
    samples.extend(json.load(f))

def format_sample(s):
    inst = s.get("instruction", "")
    system = "You are GAIA, a sovereign AI created by Azrael."
    if inst.startswith("System:"):
        parts = inst.split("\nUser: ", 1)
        if len(parts) == 2:
            system, inst = parts[0].replace("System: ", ""), parts[1]
    return (fmt.format_system(system) + "\n"
            + fmt.format_message("user", inst) + "\n"
            + fmt.format_message("assistant", s.get("output", "")))

dataset = Dataset.from_dict({"text": [format_sample(s) for s in samples]})
dataset = dataset.map(lambda b: tokenizer(b["text"], truncation=True, max_length=128, padding="max_length"),
                       batched=True, remove_columns=["text"])
dataset = dataset.map(lambda x: {"labels": x["input_ids"].copy()})
dataset.set_format("torch")
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, pin_memory=False)
print(f"  {len(dataset)} samples")

# Step 4: Train
print(f"\n[4/5] Training (10 epochs, lr=1e-4, alpha=64, r=32)...")
os.makedirs(OUTPUT_DIR, exist_ok=True)
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.01)
scaler = torch.amp.GradScaler("cuda")
EPOCHS, GRAD_ACCUM, MAX_GRAD_NORM = 15, 4, 1.0

start = time.time()
for epoch in range(EPOCHS):
    epoch_loss = 0
    valid_steps = 0
    for i, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to("cuda")
        labels = batch["labels"].to("cuda")
        attn_mask = batch["attention_mask"].to("cuda")

        out = model(input_ids=input_ids, attention_mask=attn_mask)
        logits = out.logits[:, :-1, :].float()
        targets = labels[:, 1:]
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
            ignore_index=tokenizer.pad_token_id) / GRAD_ACCUM

        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad()
            continue

        scaler.scale(loss).backward()
        epoch_loss += loss.item() * GRAD_ACCUM
        valid_steps += 1

        if (i + 1) % GRAD_ACCUM == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

    if len(dataloader) % GRAD_ACCUM != 0:
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    avg = epoch_loss / max(1, valid_steps)
    elapsed = time.time() - start
    gpu_mb = torch.cuda.memory_allocated() / (1024**2)
    print(f"  Epoch {epoch+1}/{EPOCHS} | loss={avg:.4f} | VRAM={gpu_mb:.0f}MB | {elapsed:.0f}s | valid={valid_steps}/{len(dataloader)}")

print(f"  Done in {time.time()-start:.0f}s")

# Step 5: Save + validate
print(f"\n[5/5] Save + validate...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

model.eval()
prompt = fmt.format_system("You are GAIA.") + "\n" + fmt.format_message("user", "Who are you?") + "\n" + fmt.assistant_prefix(True)
ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    out = model.generate(input_ids=ids, max_new_tokens=64, do_sample=False)
answer = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
print(f"  Q: Who are you?")
print(f"  A: {answer[:300]}")
print(f"\n{'=' * 60}")
print("PHASE 5f COMPLETE")
print(f"{'=' * 60}")
