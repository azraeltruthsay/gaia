"""
QLoRA Trainer - Actual training implementation for GAIA Self-Study

Uses PEFT and bitsandbytes for memory-efficient fine-tuning on consumer GPUs.
Designed for RTX 5080 16GB but adaptable to other configurations.

Part of Phase 2 implementation of the GAIA LoRA Adapter Architecture.
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# Lazy imports to avoid loading heavy libraries unless needed
transformers = None
peft = None
bitsandbytes = None
datasets = None


def _lazy_import():
    """Import heavy dependencies only when needed."""
    global transformers, peft, bitsandbytes, datasets

    if transformers is None:
        import transformers as _transformers
        transformers = _transformers
        logger.info("Loaded transformers %s", transformers.__version__)

    if peft is None:
        import peft as _peft
        peft = _peft
        logger.info("Loaded PEFT %s", peft.__version__)

    if bitsandbytes is None:
        try:
            import bitsandbytes as _bitsandbytes
            bitsandbytes = _bitsandbytes
            logger.info("Loaded bitsandbytes %s", bitsandbytes.__version__)
        except ImportError:
            logger.warning("bitsandbytes not available - QLoRA 4-bit quantization disabled")

    if datasets is None:
        import datasets as _datasets
        datasets = _datasets
        logger.info("Loaded datasets %s", datasets.__version__)


@dataclass
class QLoRAConfig:
    """Configuration for QLoRA training."""
    # Quantization
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    # LoRA
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: List[str] = None

    # Training
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = True
    learning_rate: float = 2e-4
    max_steps: int = 100
    warmup_steps: int = 10
    max_seq_length: int = 512
    logging_steps: int = 10
    save_steps: int = 50

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj"]

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "QLoRAConfig":
        """Create config from dictionary (e.g., from gaia_constants.json)."""
        return cls(
            load_in_4bit=config.get("load_in_4bit", True),
            bnb_4bit_compute_dtype=config.get("bnb_4bit_compute_dtype", "bfloat16"),
            bnb_4bit_quant_type=config.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=config.get("bnb_4bit_use_double_quant", True),
            lora_r=config.get("lora_r", 8),
            lora_alpha=config.get("lora_alpha", 16),
            lora_dropout=config.get("lora_dropout", 0.05),
            target_modules=config.get("target_modules", ["q_proj", "v_proj"]),
            batch_size=config.get("batch_size", 1),
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
            gradient_checkpointing=config.get("gradient_checkpointing", True),
            learning_rate=config.get("learning_rate", 2e-4),
            max_steps=config.get("max_steps", 100),
            warmup_steps=config.get("warmup_steps", 10),
            max_seq_length=config.get("max_seq_length", 512),
            logging_steps=config.get("logging_steps", 10),
            save_steps=config.get("save_steps", 50),
        )


@dataclass
class TrainingProgress:
    """Progress information during training."""
    current_step: int = 0
    total_steps: int = 0
    current_loss: float = 0.0
    avg_loss: float = 0.0
    elapsed_seconds: float = 0.0
    estimated_remaining: float = 0.0


class QLoRATrainer:
    """
    Handles the actual QLoRA training process.

    Manages model loading, quantization, training loop, and adapter saving.
    """

    def __init__(
        self,
        base_model_path: str,
        config: QLoRAConfig,
        output_dir: str,
        progress_callback: Optional[Callable[[TrainingProgress], None]] = None
    ):
        """
        Initialize the QLoRA trainer.

        Args:
            base_model_path: Path to the base model
            config: QLoRA configuration
            output_dir: Directory to save the trained adapter
            progress_callback: Optional callback for training progress updates
        """
        self.base_model_path = base_model_path
        self.config = config
        self.output_dir = Path(output_dir)
        self.progress_callback = progress_callback

        self.model = None
        self.tokenizer = None
        self.trainer = None

        self._losses: List[float] = []
        self._start_time: float = 0

    def setup(self) -> bool:
        """
        Set up the model and tokenizer for training.

        Returns:
            True if setup successful, False otherwise
        """
        _lazy_import()

        try:
            logger.info("Setting up QLoRA training for %s", self.base_model_path)

            # Configure quantization
            bnb_config = None
            if self.config.load_in_4bit and bitsandbytes is not None:
                compute_dtype = getattr(torch, self.config.bnb_4bit_compute_dtype, torch.bfloat16)
                bnb_config = transformers.BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                    bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant,
                )
                logger.info("Using 4-bit quantization with %s", self.config.bnb_4bit_quant_type)

            # Load tokenizer
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.base_model_path,
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model with quantization
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.bfloat16,
                "device_map": "auto",
            }
            if bnb_config:
                model_kwargs["quantization_config"] = bnb_config

            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                **model_kwargs
            )

            # Enable gradient checkpointing for memory efficiency
            if self.config.gradient_checkpointing:
                self.model.gradient_checkpointing_enable()

            # Prepare model for k-bit training
            if self.config.load_in_4bit:
                self.model = peft.prepare_model_for_kbit_training(self.model)

            # Configure LoRA
            lora_config = peft.LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )

            # Apply LoRA
            self.model = peft.get_peft_model(self.model, lora_config)

            trainable_params, all_params = self._count_parameters()
            logger.info(
                "Model ready: %d trainable params (%.2f%% of %d total)",
                trainable_params,
                100 * trainable_params / all_params,
                all_params
            )

            return True

        except Exception as e:
            logger.error("Failed to setup QLoRA training: %s", e, exc_info=True)
            return False

    def _count_parameters(self) -> Tuple[int, int]:
        """Count trainable and total parameters."""
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        return trainable, total

    def prepare_dataset(
        self,
        samples: List[Dict[str, str]],
        format_type: str = "instruction"
    ) -> Any:
        """
        Prepare training dataset from samples.

        Args:
            samples: List of training samples
            format_type: "instruction" for instruction tuning, "completion" for raw text

        Returns:
            HuggingFace Dataset ready for training
        """
        _lazy_import()

        def format_instruction(sample):
            """Format an instruction-style sample."""
            instruction = sample.get("instruction", "")
            input_text = sample.get("input", "")
            output = sample.get("output", "")

            if input_text:
                text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
            else:
                text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"

            return {"text": text}

        def format_completion(sample):
            """Format a completion-style sample."""
            return {"text": sample.get("text", "")}

        # Convert samples to dataset
        if format_type == "instruction":
            formatted = [format_instruction(s) for s in samples]
        else:
            formatted = [format_completion(s) for s in samples]

        dataset = datasets.Dataset.from_list(formatted)

        # Tokenize
        def tokenize(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.config.max_seq_length,
                padding="max_length",
            )

        tokenized = dataset.map(
            tokenize,
            batched=True,
            remove_columns=["text"]
        )

        logger.info("Prepared dataset with %d samples", len(tokenized))
        return tokenized

    def train(
        self,
        train_dataset: Any,
        adapter_name: str,
        timeout_seconds: int = 600
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Run the training loop.

        Args:
            train_dataset: Prepared training dataset
            adapter_name: Name for the adapter being trained
            timeout_seconds: Maximum training time

        Returns:
            Tuple of (success, metrics_dict)
        """
        _lazy_import()

        self._start_time = time.time()
        self._losses = []

        try:
            # Set up training arguments
            training_args = transformers.TrainingArguments(
                output_dir=str(self.output_dir / "checkpoints"),
                per_device_train_batch_size=self.config.batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                learning_rate=self.config.learning_rate,
                max_steps=self.config.max_steps,
                warmup_steps=self.config.warmup_steps,
                logging_steps=self.config.logging_steps,
                save_steps=self.config.save_steps,
                bf16=True,
                optim="paged_adamw_8bit" if self.config.load_in_4bit else "adamw_torch",
                gradient_checkpointing=self.config.gradient_checkpointing,
                report_to="none",  # Disable wandb/tensorboard
                remove_unused_columns=False,
                dataloader_pin_memory=False,
            )

            # Data collator for causal LM
            data_collator = transformers.DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
            )

            # Custom callback for progress reporting
            class ProgressCallback(transformers.TrainerCallback):
                def __init__(callback_self, trainer_instance):
                    callback_self.trainer_instance = trainer_instance

                def on_log(callback_self, args, state, control, logs=None, **kwargs):
                    if logs and "loss" in logs:
                        callback_self.trainer_instance._losses.append(logs["loss"])

                        elapsed = time.time() - callback_self.trainer_instance._start_time
                        if elapsed > timeout_seconds:
                            logger.warning("Training timeout reached (%ds)", timeout_seconds)
                            control.should_training_stop = True

                        if callback_self.trainer_instance.progress_callback:
                            progress = TrainingProgress(
                                current_step=state.global_step,
                                total_steps=callback_self.trainer_instance.config.max_steps,
                                current_loss=logs["loss"],
                                avg_loss=sum(callback_self.trainer_instance._losses) / len(callback_self.trainer_instance._losses),
                                elapsed_seconds=elapsed,
                                estimated_remaining=(elapsed / max(state.global_step, 1)) * (callback_self.trainer_instance.config.max_steps - state.global_step)
                            )
                            callback_self.trainer_instance.progress_callback(progress)

            # Create trainer
            self.trainer = transformers.Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                data_collator=data_collator,
                callbacks=[ProgressCallback(self)],
            )

            logger.info("Starting training for %d steps", self.config.max_steps)

            # Train!
            train_result = self.trainer.train()

            # Get final metrics
            metrics = {
                "final_loss": train_result.training_loss,
                "total_steps": train_result.global_step,
                "samples_seen": train_result.global_step * self.config.batch_size * self.config.gradient_accumulation_steps,
                "duration_seconds": time.time() - self._start_time,
                "loss_history": self._losses,
            }

            logger.info(
                "Training complete: %d steps, final_loss=%.4f, duration=%.1fs",
                metrics["total_steps"],
                metrics["final_loss"],
                metrics["duration_seconds"]
            )

            return True, metrics

        except Exception as e:
            logger.error("Training failed: %s", e, exc_info=True)
            return False, {"error": str(e)}

    def save_adapter(self, adapter_name: str, metadata: Optional[Dict[str, Any]] = None) -> Path:
        """
        Save the trained LoRA adapter.

        Args:
            adapter_name: Name for the saved adapter
            metadata: Optional additional metadata to save

        Returns:
            Path to the saved adapter directory
        """
        adapter_path = self.output_dir
        adapter_path.mkdir(parents=True, exist_ok=True)

        # Save adapter weights
        self.model.save_pretrained(adapter_path)

        # Save tokenizer (needed for inference)
        self.tokenizer.save_pretrained(adapter_path)

        # Update metadata with training info
        if metadata:
            metadata_path = adapter_path / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    existing = json.load(f)
                existing.update(metadata)
                metadata = existing

            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

        logger.info("Saved adapter to %s", adapter_path)
        return adapter_path

    def cleanup(self):
        """Release GPU memory and cleanup resources."""
        if self.model is not None:
            del self.model
            self.model = None

        if self.trainer is not None:
            del self.trainer
            self.trainer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Trainer cleanup complete")


def estimate_vram_usage(
    model_params_billions: float,
    lora_rank: int = 8,
    batch_size: int = 1,
    seq_length: int = 512,
    use_4bit: bool = True
) -> Dict[str, float]:
    """
    Estimate VRAM usage for QLoRA training.

    Args:
        model_params_billions: Model size in billions of parameters
        lora_rank: LoRA rank
        batch_size: Training batch size
        seq_length: Maximum sequence length
        use_4bit: Whether using 4-bit quantization

    Returns:
        Dict with VRAM estimates in GB
    """
    # Base model memory
    if use_4bit:
        base_model_gb = model_params_billions * 0.5  # ~0.5 bytes per param in 4-bit
    else:
        base_model_gb = model_params_billions * 2  # bfloat16

    # LoRA adapter memory (small, in bfloat16)
    # Roughly: 2 * rank * hidden_dim * num_layers * 2 bytes
    # For 7B model with hidden_dim ~4096, ~32 layers
    lora_params = 2 * lora_rank * 4096 * 32 * 2  # q_proj and v_proj
    lora_gb = (lora_params * 2) / (1024**3)  # bfloat16

    # Optimizer states (8-bit AdamW)
    optimizer_gb = lora_gb * 2  # reduced with paged_adamw_8bit

    # Activations and gradients (rough estimate)
    activation_gb = batch_size * seq_length * 4096 * 4 / (1024**3)

    # Gradient checkpointing saves ~60% of activation memory
    activation_gb *= 0.4

    total_gb = base_model_gb + lora_gb + optimizer_gb + activation_gb

    return {
        "base_model_gb": round(base_model_gb, 2),
        "lora_adapter_gb": round(lora_gb, 2),
        "optimizer_gb": round(optimizer_gb, 2),
        "activations_gb": round(activation_gb, 2),
        "total_estimated_gb": round(total_gb, 2),
        "recommended_vram_gb": round(total_gb * 1.2, 2),  # 20% headroom
    }
