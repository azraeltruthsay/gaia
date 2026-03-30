#!/usr/bin/env python3
"""
Adaptive Weighted Training — iterative training with escalating weights
for persistent failures.

Loop:
1. Eval all samples → score each
2. Track failure history (which samples fail across iterations)
3. Weight: first-time failure = 6x, 2nd consecutive = 10x, 3rd+ = 15x
4. Train on weighted dataset (3 epochs per iteration)
5. Post-eval → update failure history
6. Repeat until target score or max iterations
"""

import json, os, time, torch, sys, gc, re, copy

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

MODEL = '/models/Huihui-Qwen3-8B-abliterated-v2'
CURRICULUM = '/knowledge/curricula/self-model/train_v2.jsonl'
OUTPUT_BASE = '/models/lora_adapters/tier1_prime_adaptive'
SYSTEM = 'You are GAIA, a sovereign AI created by Azrael. Answer directly and concisely.'

MAX_ITERATIONS = 5
TARGET_PASS_RATE = 0.85  # Stop when 85% of curriculum passes
EPOCHS_PER_ITERATION = 3

stop_words = {'the','and','for','are','but','not','you','all','can','her','was','one',
              'our','out','has','have','with','this','that','from','they','been','said',
              'each','which','their','will','other','about','many','then','them','these',
              'some','into','only','very','when','also','what','just','more','your','than',
              'role','runs','port','uses','model'}

# Load curriculum
samples = []
with open(CURRICULUM) as f:
    for line in f:
        samples.append(json.loads(line))
print(f'Loaded {len(samples)} curriculum samples', flush=True)

# Load model ONCE
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

gc.collect(); torch.cuda.empty_cache()
print(f'VRAM free: {torch.cuda.mem_get_info()[0]//(1024**2)}MB', flush=True)

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
print(f'VRAM: {torch.cuda.memory_allocated()//(1024**2)}MB', flush=True)


def eval_all(model, samples):
    """Evaluate all curriculum samples, return per-sample confidence."""
    model.eval()
    scores = []
    passed = 0
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
            conf = hits / len(key_terms)
        else:
            conf = 1.0

        if conf > 0.4:
            passed += 1
        scores.append(conf)

        if (i + 1) % 50 == 0:
            print(f'  Eval: {i+1}/{len(samples)} ({passed/(i+1)*100:.0f}%)', flush=True)

    rate = passed / len(samples)
    print(f'  Result: {passed}/{len(samples)} ({rate*100:.0f}%)', flush=True)
    return scores, rate


def fmt(s):
    return (f"<|im_start|>user\n{s['instruction']}<|im_end|>\n"
            f"<|im_start|>assistant\n{s['output']}<|im_end|>")


# Track consecutive failures per sample
failure_streak = [0] * len(samples)

# Prepare model for training once
model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    lora_dropout=0.05, bias='none', task_type='CAUSAL_LM',
)
model = get_peft_model(model, lora_config)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Trainable: {trainable:,}', flush=True)

for iteration in range(MAX_ITERATIONS):
    print(f'\n{"="*60}', flush=True)
    print(f'  ITERATION {iteration + 1}/{MAX_ITERATIONS}', flush=True)
    print(f'{"="*60}', flush=True)

    # Eval
    print(f'\n--- Eval ---', flush=True)
    scores, pass_rate = eval_all(model, samples)

    if pass_rate >= TARGET_PASS_RATE:
        print(f'\nTARGET REACHED: {pass_rate*100:.0f}% >= {TARGET_PASS_RATE*100:.0f}%', flush=True)
        break

    # Update failure streaks
    for i, conf in enumerate(scores):
        if conf <= 0.4:
            failure_streak[i] += 1
        else:
            failure_streak[i] = 0

    # Build weighted dataset with escalating weights
    weighted = []
    streak_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for i, s in enumerate(samples):
        streak = failure_streak[i]
        if streak == 0:
            repeats = 1     # Passing — minimal
        elif streak == 1:
            repeats = 6     # First failure — standard
        elif streak == 2:
            repeats = 10    # Second consecutive — escalate
        else:
            repeats = 15    # 3rd+ consecutive — maximum pressure

        streak_counts[min(streak, 3)] += 1
        for _ in range(repeats):
            weighted.append({'text': fmt(s)})

    print(f'\nWeighted: {len(weighted)} samples', flush=True)
    print(f'  Passing (×1): {streak_counts[0]}', flush=True)
    print(f'  1st fail (×6): {streak_counts[1]}', flush=True)
    print(f'  2nd fail (×10): {streak_counts[2]}', flush=True)
    print(f'  3rd+ fail (×15): {streak_counts[3]}', flush=True)

    # Train
    print(f'\n--- Training ---', flush=True)
    total_steps = len(weighted) * EPOCHS_PER_ITERATION // 4
    args = SFTConfig(
        output_dir=f'{OUTPUT_BASE}_iter{iteration+1}',
        num_train_epochs=EPOCHS_PER_ITERATION,
        per_device_train_batch_size=1, gradient_accumulation_steps=4,
        learning_rate=2e-4, weight_decay=0.01, warmup_steps=10,
        logging_steps=max(1, total_steps // 3),
        save_strategy='no', bf16=True, max_length=384,
        dataset_text_field='text', report_to='none',
        gradient_checkpointing=True,
    )
    trainer = SFTTrainer(
        model=model, args=args,
        train_dataset=Dataset.from_list(weighted),
        processing_class=tokenizer,
    )
    start = time.time()
    result = trainer.train()
    elapsed = time.time() - start
    print(f'  Done: {elapsed:.0f}s, loss={result.training_loss:.4f}', flush=True)

# Final save
output = f'{OUTPUT_BASE}_final'
os.makedirs(output, exist_ok=True)
model.save_pretrained(output)
tokenizer.save_pretrained(output)
print(f'\nAdapter saved to {output}', flush=True)

# Final eval
print(f'\n{"="*60}', flush=True)
print(f'  FINAL EVALUATION', flush=True)
print(f'{"="*60}', flush=True)
final_scores, final_rate = eval_all(model, samples)
print(f'\nFinal: {final_rate*100:.0f}%', flush=True)

# Show persistent failures
persistent = [(i, failure_streak[i]) for i in range(len(samples)) if failure_streak[i] >= 2]
if persistent:
    print(f'\nPersistent failures ({len(persistent)}):',  flush=True)
    for idx, streak in persistent[:10]:
        print(f'  [{streak}x] {samples[idx]["instruction"][:60]}', flush=True)
