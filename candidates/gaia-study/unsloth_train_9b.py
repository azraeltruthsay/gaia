"""Train Qwen3.5-9B with unsloth from bf16 base — unsloth handles quantization."""
import logging, json, time
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GAIA.Unsloth")

MODEL_PATH = "/models/Huihui-Qwen3.5-9B-abliterated"
DATASET_PATH = "/knowledge/curricula/self-model/train.jsonl"
OUTPUT_DIR = "/models/lora_adapters/tier1_global/prime-9b-identity"

logger.info("Loading bf16 model with unsloth (load_in_4bit=True — unsloth's own quantization)...")
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=1024,
    dtype=None,
    load_in_4bit=True,  # unsloth's own 4-bit quantization
    trust_remote_code=True,
)
gpu_gb = torch.cuda.memory_allocated(0) / (1024**3)
logger.info("Model loaded! GPU: %.1fGB. Adding LoRA...", gpu_gb)

model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

from datasets import Dataset
samples = []
with open(DATASET_PATH) as f:
    for line in f:
        d = json.loads(line)
        samples.append({
            "text": f"<|im_start|>user\n{d['instruction']}<|im_end|>\n<|im_start|>assistant\n{d['output']}<|im_end|>"
        })
dataset = Dataset.from_list(samples)
logger.info("Dataset: %d samples", len(dataset))

from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=1024,
    args=TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=200,
        learning_rate=2e-4,
        warmup_steps=10,
        logging_steps=10,
        save_steps=50,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
        seed=42,
    ),
)

logger.info("Starting training...")
start = time.time()
trainer.train()
logger.info("Training complete in %.1fs", time.time() - start)

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
logger.info("Adapter saved to %s", OUTPUT_DIR)
