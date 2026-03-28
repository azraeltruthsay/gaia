"""Direct QLoRA training for 8B — minimal imports before model load."""
import json, logging, time, gc, os, torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GAIA.Train")

MODEL = "/models/Huihui-Qwen3-8B-GAIA-Prime-adaptive"
DATA = "/knowledge/curricula/self-model/train.jsonl"
OUTPUT = "/models/lora_adapters/tier1_global/prime-8b-identity-v2"
os.makedirs(OUTPUT, exist_ok=True)

gc.collect(); torch.cuda.empty_cache()
logger.info("GPU free: %.1fGB", torch.cuda.mem_get_info()[0]/1024**3)

# Load model FIRST — before importing peft/gptqmodel which steal VRAM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                          bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
logger.info("Loading 8B with NF4...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL, quantization_config=bnb, device_map={"": 0},
    trust_remote_code=True, low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
logger.info("Model loaded: %.1fGB GPU", torch.cuda.memory_allocated()/1024**3)

# NOW import peft (after model is loaded and NF4-quantized on GPU)
import peft
model = peft.prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
lora_config = peft.LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none", task_type="CAUSAL_LM")
model = peft.get_peft_model(model, lora_config)
model.print_trainable_parameters()
logger.info("GPU after LoRA: %.1fGB", torch.cuda.memory_allocated()/1024**3)

# Dataset
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments

samples = []
with open(DATA) as f:
    for line in f:
        d = json.loads(line)
        samples.append({"text": f"<|im_start|>user\n{d['instruction']}<|im_end|>\n<|im_start|>assistant\n{d['output']}<|im_end|>"})
dataset = Dataset.from_list(samples)
logger.info("Dataset: %d samples", len(dataset))

trainer = SFTTrainer(
    model=model, tokenizer=tokenizer, train_dataset=dataset,
    dataset_text_field="text", max_seq_length=512,
    args=TrainingArguments(
        output_dir=OUTPUT, per_device_train_batch_size=1,
        gradient_accumulation_steps=4, max_steps=200,
        learning_rate=2e-4, warmup_steps=10, logging_steps=10, save_steps=100,
        bf16=True, optim="adamw_8bit", seed=42, report_to="none"))

logger.info("Starting training...")
start = time.time()
result = trainer.train()
logger.info("DONE in %.0fs! Final loss: %.4f", time.time()-start, result.training_loss)
model.save_pretrained(OUTPUT)
tokenizer.save_pretrained(OUTPUT)
logger.info("Saved to %s", OUTPUT)
