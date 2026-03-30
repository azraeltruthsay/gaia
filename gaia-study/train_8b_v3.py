"""8B QLoRA training — Trainer class, transformers 4.51."""
import json, logging, time, gc, os, torch
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GAIA.Train")

MODEL = "/models/Huihui-Qwen3-8B-GAIA-Prime-adaptive"
DATA = "/knowledge/curricula/self-model/train.jsonl"
OUTPUT = "/models/lora_adapters/tier1_global/prime-8b-identity-v2"
os.makedirs(OUTPUT, exist_ok=True)

gc.collect(); torch.cuda.empty_cache()
logger.info("GPU free: %.1fGB", torch.cuda.mem_get_info()[0]/1024**3)

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer
import peft

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                          bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
logger.info("Loading 8B with NF4...")
model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb, device_map={"": 0},
    trust_remote_code=True, low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
logger.info("Model loaded: %.1fGB", torch.cuda.memory_allocated()/1024**3)

model = peft.prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model = peft.get_peft_model(model, peft.LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    bias="none", task_type="CAUSAL_LM"))
model.print_trainable_parameters()

# Tokenize
samples = []
with open(DATA) as f:
    for line in f:
        d = json.loads(line)
        samples.append(f"<|im_start|>user\n{d['instruction']}<|im_end|>\n<|im_start|>assistant\n{d['output']}<|im_end|>")

from datasets import Dataset
def tokenize_fn(examples):
    tok = tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")
    tok["labels"] = tok["input_ids"].copy()
    return tok
dataset = Dataset.from_dict({"text": samples}).map(tokenize_fn, batched=True, remove_columns=["text"])
logger.info("Dataset: %d samples, GPU: %.1fGB", len(dataset), torch.cuda.memory_allocated()/1024**3)

trainer = Trainer(
    model=model, train_dataset=dataset,
    args=TrainingArguments(
        output_dir=OUTPUT, per_device_train_batch_size=1,
        gradient_accumulation_steps=4, max_steps=200,
        learning_rate=2e-4, warmup_steps=10, logging_steps=10, save_steps=100,
        bf16=True, optim="adamw_8bit", seed=42, report_to="none"))

logger.info("Starting training...")
start = time.time()
result = trainer.train()
logger.info("DONE in %.0fs! Loss: %.4f", time.time()-start, result.training_loss)
model.save_pretrained(OUTPUT)
tokenizer.save_pretrained(OUTPUT)
logger.info("Saved to %s", OUTPUT)
