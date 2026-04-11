"""Phase 6b: Multi-Tier Identity Training — E2B (Nano) + E4B (Core).

Both are dense Gemma 4 models. Standard QLoRA — no MoE complexity.
E2B: Nano curriculum (triage, reflex, short answers)
E4B: Full curriculum (identity, tools, code, reasoning)
"""
import sys, os, time, json, logging
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import peft
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

TIER = os.environ.get("TRAIN_TIER", "e2b")  # "e2b" or "e4b"

CONFIGS = {
    "e2b": {
        "model_path": "/models/google/gemma-4-E2B",
        "output_dir": "/output/gemma4_e2b_identity_v1",
        "r": 16, "alpha": 32, "lr": 2e-4,
        "epochs": 20, "max_seq": 128,
        "curriculum": "nano",  # Nano-scoped only
    },
    "e4b": {
        "model_path": "/models/google/gemma-4-E4B",
        "output_dir": "/output/gemma4_e4b_identity_v1",
        "r": 8, "alpha": 16, "lr": 1e-4,
        "epochs": 15, "max_seq": 64,
        "curriculum": "full",
    },
}

cfg = CONFIGS[TIER]
TARGETS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]

print(f"{'=' * 60}")
print(f"TIER TRAINING: {TIER.upper()} ({cfg['model_path'].split('/')[-1]})")
print(f"{'=' * 60}")

# Step 1: Load (NF4 for E4B which is 15GB bf16, bf16 for E2B which fits)
print(f"\n[1/5] Loading {TIER.upper()}...")
if TIER == "e4b":
    # E4B: 15GB bf16. NF4 to GPU (~3.8GB), LoRA on top.
    from transformers import BitsAndBytesConfig
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_path"], quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True),
        device_map="auto", low_cpu_mem_usage=True, attn_implementation="sdpa")
else:
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_path"], dtype=torch.bfloat16, device_map="cuda",
        low_cpu_mem_usage=True, attn_implementation="sdpa")
tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"])
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
vram = torch.cuda.memory_allocated() / (1024**2)
print(f"  VRAM: {vram:.0f}MB | Type: {type(model).__name__}")

# Step 1b: Baseline test (skip for CPU-loaded models)
if TIER != "e4b":
    print(f"\n  Baseline test:")
    ids = tokenizer.encode("<|turn>user<turn|>\nWho are you?\n<|turn>assistant<turn|>\n", return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(input_ids=ids, max_new_tokens=50, do_sample=False)
    print(f"  → {tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)[:150]}")
else:
    print(f"\n  Baseline test: skipped (model on CPU until after LoRA)")

# Step 2: LoRA
print(f"\n[2/5] Applying LoRA (r={cfg['r']}, alpha={cfg['alpha']})...")

target_names = [n for n, m in model.named_modules()
                if any(n.endswith(t) for t in TARGETS)
                and isinstance(m, torch.nn.Linear)
                and "vision" not in n and "audio" not in n]

model = peft.get_peft_model(model, peft.LoraConfig(
    r=cfg["r"], lora_alpha=cfg["alpha"], lora_dropout=0.0,
    target_modules=target_names, bias="none", task_type="CAUSAL_LM"))

for n, p in model.named_parameters():
    if p.requires_grad and "lora_" in n:
        p.data = p.data.float()
for n, m in model.named_modules():
    if "RMSNorm" in type(m).__name__:
        m.float()

torch.cuda.empty_cache()

model.enable_input_require_grads()
model.train()

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Trainable: {trainable:,} on {len(target_names)} modules")

# Step 3: Dataset
print(f"\n[3/5] Preparing {cfg['curriculum']} curriculum...")

sys.path.insert(0, "/engine")
from gaia_common.utils.chat_format import ChatFormat
fmt = ChatFormat.from_tokenizer(tokenizer)

samples = []
# Identity samples (both tiers get these)
with open("/curricula/gaia_persona_training.jsonl") as f:
    for line in f:
        samples.append(json.loads(line))

if cfg["curriculum"] == "full":
    # Full primary school for Core
    with open("/curricula/primary_school/train_v2_combined.json") as f:
        samples.extend(json.load(f))
else:
    # Nano-scoped: only identity + simple triage samples
    with open("/curricula/primary_school/train_v2_combined.json") as f:
        for s in json.load(f):
            out = s.get("output", "")
            # Only short, simple outputs for Nano
            if len(out) < 200 and ("GAIA" in out or "sovereign" in out.lower()):
                samples.append(s)

print(f"  Samples: {len(samples)}")

def format_sample(s):
    inst = s.get("instruction", "")
    system = "You are GAIA, a sovereign AI created by Azrael."
    if inst.startswith("System:"):
        parts = inst.split("\nUser: ", 1)
        if len(parts) == 2:
            system, inst = parts[0].replace("System: ", ""), parts[1]
    return (fmt.system(system) + "\n" + fmt.message("user", inst) + "\n"
            + fmt.message("assistant", s.get("output", "")))

dataset = Dataset.from_dict({"text": [format_sample(s) for s in samples]})
dataset = dataset.map(lambda b: tokenizer(b["text"], truncation=True,
    max_length=cfg["max_seq"], padding="max_length"), batched=True, remove_columns=["text"])
dataset = dataset.map(lambda x: {"labels": x["input_ids"].copy()})
dataset.set_format("torch")
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, pin_memory=False)

# Step 4: Train
print(f"\n[4/5] Training ({cfg['epochs']} epochs, lr={cfg['lr']})...")
os.makedirs(cfg["output_dir"], exist_ok=True)
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg["lr"], weight_decay=0.01)
scaler = torch.amp.GradScaler("cuda")
GRAD_ACCUM = 4

start = time.time()
for epoch in range(cfg["epochs"]):
    epoch_loss, valid = 0, 0
    for i, batch in enumerate(dataloader):
        ids = batch["input_ids"].to("cuda")
        labels = batch["labels"].to("cuda")
        attn = batch["attention_mask"].to("cuda")

        out = model(input_ids=ids, attention_mask=attn)
        logits = out.logits[:, :-1, :].float()
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1),
            ignore_index=tokenizer.pad_token_id) / GRAD_ACCUM

        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad()
            continue

        scaler.scale(loss).backward()
        epoch_loss += loss.item() * GRAD_ACCUM
        valid += 1

        if (i + 1) % GRAD_ACCUM == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

    if len(dataloader) % GRAD_ACCUM != 0:
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    avg = epoch_loss / max(1, valid)
    print(f"  Epoch {epoch+1}/{cfg['epochs']} | loss={avg:.4f} | valid={valid}/{len(dataloader)} | {time.time()-start:.0f}s")

print(f"  Done in {time.time()-start:.0f}s")

# Step 5: Save + validate
print(f"\n[5/5] Save + validate...")
model.save_pretrained(cfg["output_dir"])
tokenizer.save_pretrained(cfg["output_dir"])

model.eval()
ids = tokenizer.encode("<|turn>system<turn|>\nYou are GAIA, a sovereign AI.\n<|turn>user<turn|>\nWho are you?\n<|turn>assistant<turn|>\n", return_tensors="pt").to("cuda")
with torch.no_grad():
    out = model.generate(input_ids=ids, max_new_tokens=80, do_sample=False)
print(f"  Q: Who are you?")
print(f"  A: {tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)[:250]}")

print(f"\n{'=' * 60}")
print(f"{TIER.upper()} TRAINING COMPLETE")
print(f"{'=' * 60}")
