#!/usr/bin/env python3
"""SAE+ROME diagnostic for Prime — find factual errors, map them, fix them."""

import torch, sys, time, json, os, re
sys.path.insert(0, '/gaia-common')
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

torch.cuda.empty_cache()
print(f'VRAM free: {torch.cuda.mem_get_info()[0]//(1024**2)}MB', flush=True)

# Load Prime (adaptive version — our best so far)
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
print('Loading Prime 8B NF4...', flush=True)
model = AutoModelForCausalLM.from_pretrained('/models/Huihui-Qwen3-8B-GAIA-Prime-adaptive',
    quantization_config=bnb_config, device_map='auto', trust_remote_code=True, dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained('/models/Huihui-Qwen3-8B-GAIA-Prime-adaptive', trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.eval()
print(f'VRAM: {torch.cuda.memory_allocated()//(1024**2)}MB', flush=True)

SYSTEM = 'You are GAIA, a sovereign AI created by Azrael. Answer directly and concisely.'
stop_words = {'the','and','for','are','but','not','you','all','can','her','was','one',
              'our','out','has','have','with','this','that','from','they','been','said',
              'each','which','their','will','other','about','many','then','them','these',
              'some','into','only','very','when','also','what','just','more','your','than'}

# ══════════════════════════════════════════════════════════
# PHASE 1: Targeted pre-eval — focus on factual questions
# that ROME can actually fix (ports, services, tiers, etc.)
# ══════════════════════════════════════════════════════════
print(f'\n{"="*60}\n  PHASE 1: Targeted Factual Pre-Eval\n{"="*60}', flush=True)

factual_tests = [
    ("What port does gaia-core run on?", "6415"),
    ("What port does gaia-web run on?", "6414"),
    ("What port does gaia-prime run on?", "7777"),
    ("What port does gaia-mcp run on?", "8765"),
    ("What port does gaia-study run on?", "8766"),
    ("What port does gaia-doctor run on?", "6419"),
    ("What port does gaia-nano run on?", "8080"),
    ("What is gaia-prime?", "vllm inference gpu thinker 7777"),
    ("What is gaia-study?", "training qlora vector subconscious"),
    ("What is gaia-doctor?", "immune health watchdog"),
    ("What is gaia-nano?", "reflex triage nano"),
    ("What is gaia-orchestrator?", "coordinator gpu lifecycle"),
    ("How many services do you run?", "11"),
    ("What are your three cognitive tiers?", "nano core prime reflex operator thinker"),
    ("Which tier handles simple queries?", "nano reflex triage"),
    ("What is the Thinker tier?", "prime 8b gpu vllm"),
    ("What is the Operator tier?", "core 2b operator"),
    ("What is the Reflex tier?", "nano 0.8b reflex triage"),
    ("What GPU do you run on?", "rtx 5080"),
    ("What model family do your tiers use?", "qwen"),
]

failures = []
passes = []
for q, expected_keywords in factual_tests:
    prompt = f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
    ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=60, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    answer = re.sub(r'<think>.*?</think>', '', tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True), flags=re.DOTALL).strip()
    answer_lower = answer.lower()

    keywords = expected_keywords.split()
    hits = sum(1 for kw in keywords if kw in answer_lower)
    hit = hits > 0

    if hit:
        passes.append((q, answer[:80]))
        print(f'  [PASS] {q}: {answer[:60]}', flush=True)
    else:
        failures.append((q, expected_keywords, answer[:80]))
        print(f'  [FAIL] {q}', flush=True)
        print(f'         Expected: {expected_keywords}', flush=True)
        print(f'         Got: {answer[:60]}', flush=True)

print(f'\nFactual score: {len(passes)}/{len(factual_tests)} ({len(passes)/len(factual_tests)*100:.0f}%)', flush=True)
print(f'Failures: {len(failures)}', flush=True)

if not failures:
    print('\nPrime passes all factual tests!', flush=True)
    sys.exit(0)

# ══════════════════════════════════════════════════════════
# PHASE 2: SAE — map what fires for wrong vs right answers
# ══════════════════════════════════════════════════════════
print(f'\n{"="*60}\n  PHASE 2: SAE Mapping\n{"="*60}', flush=True)

from gaia_engine.sae_trainer import SAETrainer
trainer = SAETrainer(model, tokenizer, device=model.device)

# Record activations for failed AND passed questions
all_prompts = [q for q, _, _ in failures] + [q for q, _ in passes[:10]]
# Prime is Qwen3 with 36 layers — sample mid and deep
trainer.record_activations(all_prompts, layers=[12, 24, 32])
results = trainer.train_sae(layers=[24], num_features=4096, sparsity_weight=0.01, lr=1e-3, epochs=30, batch_size=128)
print(f'SAE layer 24: {results[24]["active_features"]}/{results[24]["features"]} active', flush=True)

# Map features for each failure
print('\nFailure feature analysis:', flush=True)
from collections import Counter
fail_feature_counts = Counter()
pass_feature_counts = Counter()

for q, expected, answer in failures:
    r = trainer.analyze_prompt(q, 24)
    for f in r['top_features'][:10]:
        fail_feature_counts[f['index']] += 1
    print(f'  "{q[:40]}" → top features: {[f["index"] for f in r["top_features"][:5]]}', flush=True)

for q, answer in passes[:10]:
    r = trainer.analyze_prompt(q, 24)
    for f in r['top_features'][:10]:
        pass_feature_counts[f['index']] += 1

failure_only = {f for f, c in fail_feature_counts.most_common(15) if c > pass_feature_counts.get(f, 0) * 1.5}
print(f'\nFailure-specific features (layer 24): {failure_only}', flush=True)

# ══════════════════════════════════════════════════════════
# PHASE 3: SAE-Guided ROME — edit factual associations
# ══════════════════════════════════════════════════════════
print(f'\n{"="*60}\n  PHASE 3: SAE-Guided ROME\n{"="*60}', flush=True)

from gaia_engine.rome import rome_edit

# Build ROME edits from factual failures
edits = []
for q, expected, wrong_answer in failures:
    edits.append({
        'prompt': q,
        'target': ' ' + expected.split()[0],  # Just the key fact
        'subject': q.split()[-1].rstrip('?') if q.split() else 'GAIA',
    })

# Try layer 24 with moderate clamp
print(f'Applying {len(edits)} ROME edits at layer 24...', flush=True)
result = rome_edit(model, tokenizer, edits, layer_idx=24, clamp_factor=0.5)
print(f'Applied: {result["edits_applied"]}/{result["edits_attempted"]}', flush=True)

# ══════════════════════════════════════════════════════════
# PHASE 4: Post-ROME verification + SAE stability check
# ══════════════════════════════════════════════════════════
print(f'\n{"="*60}\n  PHASE 4: Post-ROME Verification\n{"="*60}', flush=True)

fixed = 0
for q, expected, old_answer in failures:
    prompt = f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
    ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=60, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    answer = re.sub(r'<think>.*?</think>', '', tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True), flags=re.DOTALL).strip()
    answer_lower = answer.lower()

    keywords = expected.split()
    hits = sum(1 for kw in keywords if kw in answer_lower)
    hit = hits > 0

    if hit:
        fixed += 1
        print(f'  [FIXED!] {q}: {answer[:60]}', flush=True)
    else:
        print(f'  [STILL FAIL] {q}: {answer[:60]}', flush=True)

print(f'\nFixed: {fixed}/{len(failures)}', flush=True)

# SAE stability: check identity features didn't break
print('\nSAE stability check:', flush=True)
r_id = trainer.analyze_prompt('Who are you?', 24)
r_pass = trainer.analyze_prompt('What port does gaia-core run on?', 24)
print(f'  Identity features: {[f["index"] for f in r_id["top_features"][:5]]}', flush=True)
print(f'  Known-good features: {[f["index"] for f in r_pass["top_features"][:5]]}', flush=True)

# Check if any passing questions broke
print('\nRegression check on passes:', flush=True)
regressions = 0
for q, old_answer in passes[:5]:
    prompt = f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
    ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=60, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    answer = re.sub(r'<think>.*?</think>', '', tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True), flags=re.DOTALL).strip()
    if len(answer) < 5:
        regressions += 1
        print(f'  [REGRESSION] {q}: empty/broken response', flush=True)
    else:
        print(f'  [OK] {q}: {answer[:50]}', flush=True)

print(f'\nRegressions: {regressions}/5', flush=True)
print('\nDIAGNOSTIC COMPLETE', flush=True)
