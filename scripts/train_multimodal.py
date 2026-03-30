"""Multimodal QLoRA Training — text + vision curriculum for Qwen3.5.

Handles both text-only pairs (identity, triage, etc.) and vision pairs
(image + instruction → output) in a unified training loop. Uses the
Qwen3VLProcessor to tokenize images into visual tokens interleaved
with text tokens.

Works for both 0.8B (Nano) and 4B (Core).

Usage:
    docker exec gaia-study python3 /gaia/GAIA_Project/scripts/train_multimodal.py \
        --model 0.8B --epochs 6

    docker exec gaia-study python3 /gaia/GAIA_Project/scripts/train_multimodal.py \
        --model 4B --epochs 4
"""
import argparse
import gc
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GAIA.Train.Multimodal")

MODELS = {
    "0.8B": {
        "path": "/models/Qwen/Qwen3.5-0.8B",
        "text_curriculum": "/gaia/GAIA_Project/knowledge/curricula/nano-multimodal/train.jsonl",
        "vision_curriculum": "/gaia/GAIA_Project/knowledge/vision_curriculum/vision_pairs.jsonl",
        "vision_images_root": "/gaia/GAIA_Project/knowledge/vision_curriculum/",
        "output_dir": "/models/lora_adapters/nano-multimodal-v6",
        "max_length": 2048,  # Vision tokens need more space
    },
    "4B": {
        "path": "/models/Qwen/Qwen3.5-4B",
        "text_curriculum": "/gaia/GAIA_Project/knowledge/curricula/core-multimodal/train.jsonl",
        "vision_curriculum": "/gaia/GAIA_Project/knowledge/vision_curriculum/vision_pairs.jsonl",
        "vision_images_root": "/gaia/GAIA_Project/knowledge/vision_curriculum/",
        "output_dir": "/models/lora_adapters/core-multimodal-v4",
        "max_length": 2048,
    },
}


def build_multimodal_dataset(text_path: str, vision_path: str, images_root: str, processor):
    """Build a unified dataset of text-only and vision training samples.

    Returns list of dicts, each with:
      - 'input_ids': tokenized input
      - 'labels': tokenized labels (masked for input, unmasked for output)
      - 'pixel_values': (optional) image tensor
      - 'image_grid_thw': (optional) image grid info
    """
    import torch
    from PIL import Image

    samples = []

    # ── Text-only pairs ──────────────────────────────────────────────────
    text_count = 0
    with open(text_path) as f:
        for line in f:
            d = json.loads(line)
            messages = [
                {"role": "user", "content": d["instruction"]},
                {"role": "assistant", "content": d["output"]},
            ]
            text = processor.apply_chat_template(messages, tokenize=False)
            tokenized = processor.tokenizer(text, return_tensors="pt", padding=False, truncation=True)
            samples.append({
                "input_ids": tokenized["input_ids"].squeeze(0),
                "attention_mask": tokenized["attention_mask"].squeeze(0),
                "_is_vision": False,
            })
            text_count += 1

    # ── Vision pairs ─────────────────────────────────────────────────────
    vision_count = 0
    with open(vision_path) as f:
        for line in f:
            d = json.loads(line)
            image_path = os.path.join(images_root, d["image"])
            if not os.path.exists(image_path):
                logger.warning("Image not found: %s", image_path)
                continue

            image = Image.open(image_path).convert("RGB")
            # Resize to limit visual tokens (keep VRAM manageable)
            max_dim = 224
            if max(image.size) > max_dim:
                image.thumbnail((max_dim, max_dim), Image.LANCZOS)

            # Build multimodal conversation
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": d["instruction"]},
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": d["output"]},
                ]},
            ]

            text_input = processor.apply_chat_template(messages, tokenize=False)
            processed = processor(
                text=[text_input],
                images=[image],
                return_tensors="pt",
                padding=False,
            )

            sample = {
                "input_ids": processed["input_ids"].squeeze(0),
                "attention_mask": processed["attention_mask"].squeeze(0),
                "_is_vision": True,
            }

            # Include vision tensors
            if "pixel_values" in processed:
                sample["pixel_values"] = processed["pixel_values"].squeeze(0)
            if "image_grid_thw" in processed:
                sample["image_grid_thw"] = processed["image_grid_thw"].squeeze(0)

            samples.append(sample)
            vision_count += 1

    logger.info("Dataset built: %d text + %d vision = %d total", text_count, vision_count, len(samples))

    # Sort: all text samples first, then all vision samples
    # This prevents mixed batches which cause tensor shape issues
    text_samples = [s for s in samples if not s.get("_is_vision")]
    vision_samples = [s for s in samples if s.get("_is_vision")]
    sorted_samples = text_samples + vision_samples
    logger.info("  Sorted: %d text then %d vision (no mixed batches)", len(text_samples), len(vision_samples))

    import torch
    class MultimodalDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            # Return only model-compatible keys
            sample = self.data[idx]
            result = {
                "input_ids": sample["input_ids"],
                "attention_mask": sample["attention_mask"],
            }
            if "pixel_values" in sample:
                result["pixel_values"] = sample["pixel_values"]
            if "image_grid_thw" in sample:
                result["image_grid_thw"] = sample["image_grid_thw"]
            return result

    return MultimodalDataset(sorted_samples)


class MultimodalCollator:
    """Data collator that handles both text-only and vision samples."""

    def __init__(self, processor, max_length=512):
        self.processor = processor
        self.max_length = max_length
        self.pad_token_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id

    def __call__(self, batch):
        import torch

        # Separate vision and text samples
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        pixel_values_list = []
        image_grid_thw_list = []
        has_vision = False

        for sample in batch:
            ids = sample["input_ids"][:self.max_length]
            mask = sample["attention_mask"][:self.max_length]

            # Labels = input_ids (causal LM, model predicts next token)
            labels = ids.clone()

            input_ids_list.append(ids)
            attention_mask_list.append(mask)
            labels_list.append(labels)

            if "pixel_values" in sample:
                has_vision = True
                pixel_values_list.append(sample["pixel_values"])
                if "image_grid_thw" in sample:
                    image_grid_thw_list.append(sample["image_grid_thw"])

        # Pad to same length
        max_len = max(ids.shape[0] for ids in input_ids_list)
        padded_ids = []
        padded_masks = []
        padded_labels = []

        for ids, mask, labels in zip(input_ids_list, attention_mask_list, labels_list):
            pad_len = max_len - ids.shape[0]
            if pad_len > 0:
                padded_ids.append(torch.cat([ids, torch.full((pad_len,), self.pad_token_id, dtype=ids.dtype)]))
                padded_masks.append(torch.cat([mask, torch.zeros(pad_len, dtype=mask.dtype)]))
                padded_labels.append(torch.cat([labels, torch.full((pad_len,), -100, dtype=labels.dtype)]))
            else:
                padded_ids.append(ids)
                padded_masks.append(mask)
                padded_labels.append(labels)

        result = {
            "input_ids": torch.stack(padded_ids),
            "attention_mask": torch.stack(padded_masks),
            "labels": torch.stack(padded_labels),
        }

        if has_vision and pixel_values_list:
            try:
                result["pixel_values"] = torch.cat(pixel_values_list, dim=0)
                if image_grid_thw_list:
                    # Ensure each grid_thw has batch dimension [N, 3]
                    fixed = []
                    for g in image_grid_thw_list:
                        if g.dim() == 1:
                            fixed.append(g.unsqueeze(0))  # [3] → [1, 3]
                        else:
                            fixed.append(g)
                    result["image_grid_thw"] = torch.cat(fixed, dim=0)
            except Exception as e:
                logger.warning("Vision tensor concat failed: %s — skipping vision for this batch", e)

        return result


def main():
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import Trainer, TrainingArguments

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["0.8B", "4B"], required=True)
    parser.add_argument("--epochs", type=int, default=6)
    args = parser.parse_args()

    config = MODELS[args.model]
    output_dir = config["output_dir"]

    # ── Load model ───────────────────────────────────────────────────────
    logger.info("Loading %s with 4-bit quantization...", args.model)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    processor = AutoProcessor.from_pretrained(config["path"], trust_remote_code=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model = AutoModelForImageTextToText.from_pretrained(
        config["path"],
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="sdpa",
    )

    gpu_gb = torch.cuda.memory_allocated(0) / (1024**3)
    logger.info("Model loaded: %.2fGB VRAM", gpu_gb)

    # ── LoRA (text layers only — vision encoder preserved) ───────────────
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
    )

    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("LoRA: %d trainable / %d total (%.2f%%)", trainable, total, 100 * trainable / total)

    # ── Build dataset ────────────────────────────────────────────────────
    samples = build_multimodal_dataset(
        config["text_curriculum"],
        config["vision_curriculum"],
        config["vision_images_root"],
        processor,
    )

    collator = MultimodalCollator(processor, max_length=config["max_length"])

    # ── Train ────────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=args.epochs,
        learning_rate=2e-4,
        warmup_steps=10,
        logging_steps=20,
        save_steps=200,
        save_total_limit=2,
        bf16=True,
        optim="adamw_8bit",
        seed=42,
        report_to="none",
        remove_unused_columns=False,
    )

    class MultimodalTrainer(Trainer):
        """Trainer that handles multimodal forward pass correctly."""
        model_accepts_loss_kwargs = False

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            # Debug: log input keys and shapes on first failure
            try:
                outputs = model(**inputs)
            except TypeError as e:
                logger.error("Forward pass failed. Input keys: %s", list(inputs.keys()))
                for k, v in inputs.items():
                    if hasattr(v, 'shape'):
                        logger.error("  %s: shape=%s dtype=%s", k, v.shape, v.dtype)
                    else:
                        logger.error("  %s: type=%s value=%s", k, type(v).__name__, str(v)[:100])
                raise
            except Exception as e:
                logger.error("Forward pass error: %s. Input keys: %s", e, list(inputs.keys()))
                for k, v in inputs.items():
                    if hasattr(v, 'shape'):
                        logger.error("  %s: shape=%s dtype=%s", k, v.shape, v.dtype)
                    else:
                        logger.error("  %s: type=%s value=%s", k, type(v).__name__, str(v)[:100])
                raise
            logits = outputs.logits

            # Shift for causal LM loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            return (loss, outputs) if return_outputs else loss

    trainer = MultimodalTrainer(
        model=model,
        args=training_args,
        train_dataset=samples,
        data_collator=collator,
    )

    logger.info("Starting multimodal training (%d epochs, %d samples)...", args.epochs, len(samples))
    start = time.time()
    trainer.train()
    elapsed = time.time() - start
    logger.info("Training complete in %.1fs (%.1f min)", elapsed, elapsed / 60)

    # ── Save ─────────────────────────────────────────────────────────────
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    # Metadata
    meta = {
        "base_model": config["path"],
        "adapter_dir": output_dir,
        "text_curriculum": config["text_curriculum"],
        "vision_curriculum": config["vision_curriculum"],
        "text_samples": sum(1 for s in samples if not s.get("_is_vision")),
        "vision_samples": sum(1 for s in samples if s.get("_is_vision")),
        "total_samples": len(samples),
        "epochs": args.epochs,
        "training_time_s": round(elapsed, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "notes": f"Multimodal {args.model} — text + real image vision pairs",
    }
    with open(f"{output_dir}/training_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Adapter saved to %s", output_dir)

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
