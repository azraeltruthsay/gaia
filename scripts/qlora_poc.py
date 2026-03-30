"""QLoRA Proof of Concept — run inside gaia-study container."""
import json, sys, time, torch

# Load training samples from JSONL
samples = []
with open('/knowledge/curricula/gaia_persona_training.jsonl') as f:
    for line in f:
        samples.append(json.loads(line.strip()))
print(f'Loaded {len(samples)} training samples')

# Setup trainer
from gaia_study.qlora_trainer import QLoRATrainer, QLoRAConfig

config = QLoRAConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype='bfloat16',
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=['q_proj', 'v_proj'],
    batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    max_steps=20,
    warmup_steps=2,
    max_seq_length=256,
    logging_steps=5,
    save_steps=20,
    target_loss=0.05,
    convergence_patience=3,
)

def on_progress(p):
    print(f'  Step {p.current_step}/{p.total_steps} | loss={p.current_loss:.4f} | elapsed={p.elapsed_seconds:.1f}s')
    sys.stdout.flush()

trainer = QLoRATrainer(
    base_model_path='/models/Qwen3-8B-abliterated',
    config=config,
    output_dir='/models/lora_adapters/tier1_global/qlora_poc',
    progress_callback=on_progress,
)

# Setup
print('Setting up model...')
sys.stdout.flush()
ok = trainer.setup()
if not ok:
    print('SETUP FAILED')
    sys.exit(1)

gpu_gb = round(torch.cuda.memory_allocated(0) / 1e9, 2)
print(f'Model loaded. GPU memory: {gpu_gb} GB')
sys.stdout.flush()

# Prepare dataset
print('Preparing dataset...')
sys.stdout.flush()
dataset = trainer.prepare_dataset(samples, 'instruction')
print(f'Dataset ready: {len(dataset)} samples')
sys.stdout.flush()

# Train
print('Starting training...')
sys.stdout.flush()
success, metrics = trainer.train(dataset, 'qlora_poc', timeout_seconds=300)
print(f'Training success: {success}')
filtered = {k: v for k, v in metrics.items() if k != "loss_history"}
print(f'Metrics: {json.dumps(filtered, indent=2)}')
sys.stdout.flush()

if success:
    adapter_path = trainer.save_adapter('qlora_poc', {'type': 'proof_of_concept'})
    print(f'Adapter saved to: {adapter_path}')

# Cleanup
trainer.cleanup()
print('Done!')
