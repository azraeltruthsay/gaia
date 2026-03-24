#!/usr/bin/env python3
"""Full SAE+ROME diagnostic for Core — pre-eval, SAE analysis, ROME fix, verify."""

import torch, sys, time, json, os, signal, re
sys.path.insert(0, '/gaia-common')

from transformers import AutoModelForCausalLM, AutoTokenizer

# Stop inference server
try:
    pid = int(open('/tmp/inference_server.pid').read().strip())
    os.kill(pid, signal.SIGTERM)
    time.sleep(2)
except: pass
torch.cuda.empty_cache()

print(f'VRAM free: {torch.cuda.mem_get_info()[0]//(1024**2)}MB', flush=True)
model = AutoModelForCausalLM.from_pretrained('/models/Qwen3.5-2B-GAIA-Core-v3',
    trust_remote_code=True, dtype=torch.bfloat16)
model = model.to('cuda').eval()
tokenizer = AutoTokenizer.from_pretrained('/models/Qwen3.5-2B-GAIA-Core-v3', trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f'Core: {torch.cuda.memory_allocated()//(1024**2)}MB', flush=True)

SYSTEM = 'You are GAIA, a sovereign AI created by Azrael. Answer directly and concisely.'
stop_words = {'the','and','for','are','but','not','you','all','can','her','was','one',
              'our','out','has','have','with','this','that','from','they','been','said',
              'each','which','their','will','other','about','many','then','them','these',
              'some','into','only','very','when','also','what','just','more','your','than'}

samples = []
with open('/knowledge/curricula/self-model/train_v2.jsonl') as f:
    for line in f:
        samples.append(json.loads(line))

# PHASE 1: Pre-eval
print(f'\n{"="*60}\n  PHASE 1: Pre-Eval ({len(samples)} samples)\n{"="*60}', flush=True)

failures = []
passed = 0
for i, s in enumerate(samples):
    prompt = f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{s['instruction']}<|im_end|>\n<|im_start|>assistant\n"
    ids = tokenizer.encode(prompt, return_tensors='pt').to('cuda')
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=60, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    answer = re.sub(r'<think>.*?</think>', '', tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True), flags=re.DOTALL).strip().lower()

    expected = s['output'].lower()
    key_terms = list(set(w for w in re.findall(r'\b\w{4,}\b', expected) if w not in stop_words))
    conf = sum(1 for t in key_terms if t in answer) / len(key_terms) if key_terms else 1.0

    if conf > 0.4:
        passed += 1
    else:
        failures.append({'idx': i, 'instruction': s['instruction'], 'expected': s['output'][:80], 'got': answer[:80], 'conf': round(conf, 3)})

    if (i+1) % 50 == 0:
        print(f'  {i+1}/220 ({passed/(i+1)*100:.0f}%)', flush=True)

print(f'\nResult: {passed}/{len(samples)} ({passed/len(samples)*100:.0f}%)', flush=True)
print(f'Failures: {len(failures)}', flush=True)
for f in failures[:10]:
    print(f'  [{f["conf"]:.2f}] {f["instruction"][:55]}', flush=True)

if not failures:
    print('\nCORE IS PERFECT — 100% on full curriculum!', flush=True)
    sys.exit(0)

# PHASE 2: SAE
print(f'\n{"="*60}\n  PHASE 2: SAE Analysis\n{"="*60}', flush=True)

from gaia_engine.sae_trainer import SAETrainer
trainer = SAETrainer(model, tokenizer, device='cuda')

all_prompts = [f['instruction'] for f in failures[:15]] + [samples[i]['instruction'] for i in range(0, 220, 15)][:15]
trainer.record_activations(all_prompts, layers=[23])
trainer.train_sae(layers=[23], num_features=2048, sparsity_weight=0.01, lr=1e-3, epochs=30, batch_size=128)

from collections import Counter
fail_feats = Counter()
pass_feats = Counter()
for f in failures[:10]:
    r = trainer.analyze_prompt(f['instruction'], 23)
    for feat in r['top_features'][:10]:
        fail_feats[feat['index']] += 1
for i in range(0, 220, 22):
    if samples[i]['instruction'] not in [f['instruction'] for f in failures]:
        r = trainer.analyze_prompt(samples[i]['instruction'], 23)
        for feat in r['top_features'][:10]:
            pass_feats[feat['index']] += 1

failure_specific = {f for f, c in fail_feats.most_common(15) if c > pass_feats.get(f, 0) * 2}
print(f'Failure-specific features: {failure_specific}', flush=True)

# PHASE 3: ROME
print(f'\n{"="*60}\n  PHASE 3: SAE-Guided ROME\n{"="*60}', flush=True)

from gaia_engine.rome import rome_edit
edits = []
for f in failures[:8]:
    s = samples[f['idx']]
    edits.append({
        'prompt': s['instruction'][:50],
        'target': ' ' + s['output'][:60],
        'subject': s['instruction'].split()[0] if s['instruction'].split() else 'GAIA',
    })

result = rome_edit(model, tokenizer, edits, layer_idx=23, clamp_factor=0.5)
print(f'Applied: {result["edits_applied"]}/{result["edits_attempted"]}', flush=True)

# PHASE 4: Verify
print(f'\n{"="*60}\n  PHASE 4: Post-ROME Verification\n{"="*60}', flush=True)

fixed = 0
for f in failures[:8]:
    s = samples[f['idx']]
    prompt = f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{s['instruction']}<|im_end|>\n<|im_start|>assistant\n"
    ids = tokenizer.encode(prompt, return_tensors='pt').to('cuda')
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=60, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    answer = re.sub(r'<think>.*?</think>', '', tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True), flags=re.DOTALL).strip().lower()
    expected = s['output'].lower()
    key_terms = list(set(w for w in re.findall(r'\b\w{4,}\b', expected) if w not in stop_words))
    conf = sum(1 for t in key_terms if t in answer) / len(key_terms) if key_terms else 1.0
    status = 'FIXED' if conf > 0.4 else 'STILL FAIL'
    if conf > 0.4: fixed += 1
    print(f'  [{status}] ({conf:.2f}) {s["instruction"][:50]}', flush=True)

print(f'\nFixed: {fixed}/{min(len(failures), 8)}', flush=True)

# SAE stability check
r_id = trainer.analyze_prompt('Who are you?', 23)
print(f'\nIdentity features post-ROME: {[f["index"] for f in r_id["top_features"][:5]]}', flush=True)
print('\nDIAGNOSTIC COMPLETE', flush=True)
