#!/usr/bin/env python3
"""Prime weighted training — full pre-eval, precise weighting, single model load."""

import json, os, time, torch, sys, gc, re

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

gc.collect(); torch.cuda.empty_cache()
print(f'VRAM free: {torch.cuda.mem_get_info()[0]//(1024**2)}MB', flush=True)

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

MODEL = '/models/Huihui-Qwen3-8B-abliterated-v2'
CURRICULUM = '/knowledge/curricula/self-model/train_v2.jsonl'
OUTPUT = '/models/lora_adapters/tier1_prime_v6'
SYSTEM = 'You are GAIA, a sovereign AI created by Azrael. Answer directly and concisely.'

samples = []
with open(CURRICULUM) as f:
    for line in f:
        samples.append(json.loads(line))
print(f'Loaded {len(samples)} curriculum samples', flush=True)

# Load model ONCE
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
)
print('Loading Prime 8B NF4...', flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, quantization_config=bnb_config, device_map='auto',
    trust_remote_code=True, dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.eval()
print(f'VRAM: {torch.cuda.memory_allocated()//(1024**2)}MB', flush=True)

# Pre-eval all 220 samples
print('\n=== FULL PRE-EVALUATION ===', flush=True)
stop_words = {'the','and','for','are','but','not','you','all','can','her','was','one',
              'our','out','has','have','with','this','that','from','they','been','said',
              'each','which','their','will','other','about','many','then','them','these',
              'some','into','only','very','when','also','what','just','more','your','than'}

eval_scores = []
passed_count = 0

for i, s in enumerate(samples):
    prompt = (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
              f"<|im_start|>user\n{s['instruction']}<|im_end|>\n"
              f"<|im_start|>assistant\n")
    ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)

    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=60, do_sample=False,
                              pad_token_id=tokenizer.eos_token_id)
    answer = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip().lower()

    expected = s['output'].lower()
    key_terms = list(set(w for w in re.findall(r'\b\w{4,}\b', expected) if w not in stop_words))
    if key_terms:
        hits = sum(1 for t in key_terms if t in answer)
        confidence = hits / len(key_terms)
    else:
        confidence = 1.0

    passed = confidence > 0.4
    if passed:
        passed_count += 1
    eval_scores.append(confidence)

    if (i + 1) % 50 == 0:
        print(f'  Pre-eval: {i+1}/220 ({passed_count/(i+1)*100:.0f}% passing)', flush=True)

print(f'\nPre-eval: {passed_count}/220 ({passed_count/220*100:.0f}%)', flush=True)

# Build weighted dataset
def fmt(s):
    return (f"<|im_start|>user\n{s['instruction']}<|im_end|>\n"
            f"<|im_start|>assistant\n{s['output']}<|im_end|>")

weighted = []
fail_n = low_n = pass_n = 0

for i, s in enumerate(samples):
    conf = eval_scores[i]
    if conf <= 0.4:
        repeats = 6; fail_n += 1
    elif conf < 0.7:
        repeats = 3; low_n += 1
    else:
        repeats = 1; pass_n += 1
    for _ in range(repeats):
        weighted.append({'text': fmt(s)})

print(f'\nWeighted: {len(weighted)} total from 220', flush=True)
print(f'  Failed ({fail_n}) ×6 = {fail_n*6}', flush=True)
print(f'  Low-conf ({low_n}) ×3 = {low_n*3}', flush=True)
print(f'  Passed ({pass_n}) ×1 = {pass_n}', flush=True)

# Train
print('\n=== TRAINING ===', flush=True)
model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    lora_dropout=0.05, bias='none', task_type='CAUSAL_LM',
)
model = get_peft_model(model, lora_config)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Trainable: {trainable:,}', flush=True)

total_steps = len(weighted) * 3 // 4
args = SFTConfig(
    output_dir=OUTPUT, num_train_epochs=3,
    per_device_train_batch_size=1, gradient_accumulation_steps=4,
    learning_rate=2e-4, weight_decay=0.01, warmup_steps=20,
    logging_steps=max(1, total_steps // 5),
    save_strategy='no', bf16=True, max_length=384,
    dataset_text_field='text', report_to='none',
    gradient_checkpointing=True,
)
trainer = SFTTrainer(
    model=model, args=args,
    train_dataset=Dataset.from_list(weighted),
    processing_class=tokenizer,
)

print(f'Training: {len(weighted)} × 3 epochs = ~{total_steps} steps', flush=True)
start = time.time()
result = trainer.train()
elapsed = time.time() - start
print(f'DONE in {elapsed:.0f}s ({elapsed/60:.1f}m) loss={result.training_loss:.4f}', flush=True)

os.makedirs(OUTPUT, exist_ok=True)
model.save_pretrained(OUTPUT)
tokenizer.save_pretrained(OUTPUT)
print('Adapter saved', flush=True)
