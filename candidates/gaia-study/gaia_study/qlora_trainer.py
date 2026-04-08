"""
QLoRA Trainer - Actual training implementation for GAIA Self-Study

Uses PEFT and bitsandbytes for memory-efficient fine-tuning on consumer GPUs.
Designed for RTX 5080 16GB but adaptable to other configurations.

Part of Phase 2 implementation of the GAIA LoRA Adapter Architecture.
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

try:
    from gaia_common.utils.memory_guard import require_memory as _require_memory
except ImportError:
    _require_memory = None  # type: ignore[assignment]

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

    # Convergence / early stopping
    target_loss: float = 0.05          # Stop when loss drops below this
    convergence_patience: int = 3      # Must stay below target_loss for N consecutive log checks

    # Epoch-based training (overrides max_steps when set)
    num_train_epochs: Optional[int] = None

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
            target_loss=config.get("target_loss", 0.05),
            convergence_patience=config.get("convergence_patience", 3),
            num_train_epochs=config.get("num_train_epochs"),
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
        progress_callback: Optional[Callable[[TrainingProgress], None]] = None,
        resume_from: Optional[str] = None,
    ):
        """
        Initialize the QLoRA trainer.

        Args:
            base_model_path: Path to the base model
            config: QLoRA configuration
            output_dir: Directory to save the trained adapter
            progress_callback: Optional callback for training progress updates
            resume_from: Path to existing adapter to resume from (incremental training)
        """
        self.base_model_path = base_model_path
        self.config = config
        self.output_dir = Path(output_dir)
        self.progress_callback = progress_callback
        self.resume_from = resume_from

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
            # Pre-flight: verify enough system RAM for BnB NF4 loading.
            # CPU-offload path needs bf16 model in RAM (~18GB for 9B, ~8GB for 4B).
            base_p = Path(self.base_model_path)
            bf16_mb = sum(f.stat().st_size for f in base_p.glob("*.safetensors")) / (1024**2)
            if bf16_mb == 0:
                bf16_mb = sum(f.stat().st_size for f in base_p.glob("*.bin")) / (1024**2)
            needed_mb = max(8000, int(bf16_mb * 1.3))  # bf16 size + 30% overhead
            if _require_memory is not None:
                _require_memory(needed_mb=needed_mb, label="QLoRA training setup")

            logger.info("Setting up QLoRA training for %s", self.base_model_path)

            # Detect if model is already quantized (AWQ, GPTQ, etc.)
            model_config_path = Path(self.base_model_path) / "config.json"
            is_prequantized = False
            if model_config_path.exists():
                import json as _json
                with open(model_config_path) as f:
                    model_cfg = _json.load(f)
                if "quantization_config" in model_cfg:
                    quant_method = model_cfg["quantization_config"].get("quant_method", "")
                    logger.info("Model already quantized with %s — skipping BitsAndBytes", quant_method)
                    is_prequantized = True

            # Load tokenizer
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.base_model_path,
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Configure memory allocation for training headroom
            import os
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

            gpu_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            # Reserve 4GB for LoRA params + optimizer states + activations
            training_headroom_gb = 4
            max_gpu_gb = int(gpu_total_gb) - training_headroom_gb
            logger.info(
                "GPU total: %.1fGiB, limiting model to %dGiB (%dGiB training headroom)",
                gpu_total_gb, max_gpu_gb, training_headroom_gb
            )

            if is_prequantized:
                # GPTQ models: peft >=0.10 supports LoRA on GPTQ via autograd-compatible
                # wrappers. Load without BnB — GPTQ is already quantized.
                quant_method = model_cfg['quantization_config'].get('quant_method', 'unknown')
                logger.info(
                    "Model is pre-quantized (%s). Loading for GPTQ+LoRA fine-tuning "
                    "(peft %s handles gradient flow through LoRA adapters).",
                    quant_method, peft.__version__ if 'peft' in dir() else '?'
                )

            # Detect multimodal model. For TEXT-ONLY training (identity bake,
            # tool calling, etc.), load as CausalLM to skip the vision encoder
            # and save ~1GB+ VRAM. For vision training, use ImageTextToText.
            auto_cls = transformers.AutoModelForCausalLM
            _is_multimodal = False
            _text_only_config = None
            try:
                _cfg = transformers.AutoConfig.from_pretrained(
                    self.base_model_path, trust_remote_code=True
                )
                if hasattr(_cfg, "vision_config") and _cfg.vision_config is not None:
                    _is_multimodal = True
                    # Extract text_config for CausalLM loading (skips vision encoder)
                    if hasattr(_cfg, "text_config") and _cfg.text_config is not None:
                        _text_only_config = _cfg.text_config
                        auto_cls = transformers.AutoModelForCausalLM
                        logger.info(
                            "Multimodal model detected — extracting text_config for CausalLM "
                            "loading (skips vision encoder, saves ~1GB+ VRAM). "
                            "Text model: %d layers, hidden=%d, vocab=%d",
                            getattr(_text_only_config, "num_hidden_layers", -1),
                            getattr(_text_only_config, "hidden_size", -1),
                            getattr(_text_only_config, "vocab_size", -1),
                        )
                    else:
                        auto_cls = transformers.AutoModelForImageTextToText
                        logger.info("Multimodal model — no text_config found, using ImageTextToText")
            except Exception:
                pass

            if is_prequantized:
                # GPTQ model — already quantized, load directly. No BnB needed.
                # Disable Marlin backend — some layers have out_features not divisible by 64
                logger.info("Loading pre-quantized GPTQ model directly to GPU...")
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                # Load GPTQ model via gptqmodel with AUTO_TRAINABLE backend
                # This selects a kernel that supports backward pass for LoRA training
                logger.info("Loading GPTQ model with AUTO_TRAINABLE backend...")
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                try:
                    from gptqmodel import GPTQModel, BACKEND
                    # Register Qwen3.5 multimodal model definition
                    try:
                        from gaia_study.merge_and_requantize import _register_qwen3_5
                        _register_qwen3_5()
                    except Exception:
                        pass
                    # Try TRITON backend first (best backward support),
                    # fall back to AUTO_TRAINABLE
                    _load_backend = BACKEND.TRITON
                    try:
                        self.model = GPTQModel.load(
                            self.base_model_path,
                            backend=_load_backend,
                            device_map={"": 0},
                            trust_remote_code=True,
                        )
                    except Exception as _triton_err:
                        logger.warning("TRITON backend failed: %s. Trying AUTO_TRAINABLE...", _triton_err)
                        _load_backend = BACKEND.AUTO_TRAINABLE
                        self.model = GPTQModel.load(
                            self.base_model_path,
                            backend=_load_backend,
                            device_map={"": 0},
                            trust_remote_code=True,
                        )
                except Exception as _gptq_err:
                    logger.error("GPTQModel.load with AUTO_TRAINABLE failed: %s", _gptq_err)
                    raise
            elif self.config.load_in_4bit and bitsandbytes is not None:
                # QLoRA: BnB NF4 quantization on bf16 base model
                # Estimate bf16 model size first — needed for loading decisions.
                base_p = Path(self.base_model_path)
                bf16_size_gb = sum(
                    f.stat().st_size for f in base_p.glob("*.safetensors")
                ) / (1024**3)
                if bf16_size_gb == 0:
                    bf16_size_gb = sum(
                        f.stat().st_size for f in base_p.glob("*.bin")
                    ) / (1024**3)

                # For large models with untied embeddings (e.g. Qwen3.5-9B),
                # the lm_head is ~1.9GB in bf16. Force-quantizing it saves
                # significant VRAM during loading. This is safe for LoRA
                # training since we don't modify lm_head weights.
                gpu_free_gb = torch.cuda.mem_get_info()[0] / (1024 ** 3)
                _skip_modules = None  # default: skip lm_head
                if bf16_size_gb > gpu_free_gb * 0.8:
                    _skip_modules = []  # empty = quantize everything including lm_head
                    logger.info(
                        "Large model (%.1fGiB bf16 > %.1fGiB budget) — "
                        "forcing lm_head quantization to save ~1.5GiB VRAM",
                        bf16_size_gb, gpu_free_gb * 0.8,
                    )

                bnb_config = transformers.BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
                    bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                    bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant,
                    llm_int8_enable_fp32_cpu_offload=True,
                    llm_int8_skip_modules=_skip_modules,
                )

                # NF4 is ~4x compression. The actual GPU usage will be bf16/4.
                # But transformers temporarily holds bf16 weights during quantization,
                # so we need bf16 to fit in VRAM during the transient loading phase.
                nf4_estimated_gb = bf16_size_gb / 4
                gpu_free_gb = torch.cuda.mem_get_info()[0] / (1024 ** 3)
                bf16_fits_in_vram = bf16_size_gb < gpu_free_gb * 0.85
                # Extra kwargs for text-only loading of multimodal models
                _config_kwargs = {}
                if _text_only_config is not None:
                    _config_kwargs["config"] = _text_only_config

                if nf4_estimated_gb < max_gpu_gb * 0.7 and bf16_fits_in_vram:
                    # Model fits on GPU — load directly for full GPU training
                    logger.info(
                        "Loading model to GPU (bf16 size: %.1fGiB fits in %.1fGiB VRAM)",
                        bf16_size_gb, max_gpu_gb
                    )
                    self.model = auto_cls.from_pretrained(
                        self.base_model_path,
                        trust_remote_code=True,
                        quantization_config=bnb_config,
                        device_map={"": 0},
                        low_cpu_mem_usage=True,
                        torch_dtype=torch.bfloat16,
                        **_config_kwargs,
                    )
                else:
                    # Model bf16 size exceeds VRAM but NF4 model should fit.
                    # With lm_head quantization (if enabled above), the final
                    # model is ~5-6GB. Use device_map={"":0} to force all NF4
                    # layers onto GPU. The transient bf16 peak during loading
                    # should stay within budget.
                    gpu_free_gb = torch.cuda.mem_get_info()[0] / (1024 ** 3)
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    gpu_free_gb = torch.cuda.mem_get_info()[0] / (1024 ** 3)
                    logger.info(
                        "Loading NF4 model directly to GPU (%.1fGiB free, "
                        "bf16=%.1fGiB → NF4 est. %.1fGiB)",
                        gpu_free_gb, bf16_size_gb, nf4_estimated_gb,
                    )
                    self.model = auto_cls.from_pretrained(
                        self.base_model_path,
                        trust_remote_code=True,
                        quantization_config=bnb_config,
                        device_map={"": 0},
                        low_cpu_mem_usage=True,
                        torch_dtype=torch.bfloat16,
                        **_config_kwargs,
                    )
                    gpu_used = torch.cuda.memory_allocated(0) / (1024**3)
                    gpu_free_after = torch.cuda.mem_get_info()[0] / (1024**3)
                    logger.info(
                        "Model loaded: %.1fGiB on GPU (%.1fGiB free)",
                        gpu_used, gpu_free_after,
                    )
            else:
                # Fallback: bf16 directly to GPU (only works for small models)
                logger.info("Loading model in bf16 to GPU (no quantization)")
                self.model = auto_cls.from_pretrained(
                    self.base_model_path,
                    trust_remote_code=True,
                    dtype=torch.bfloat16,
                    device_map={"": 0},
                    low_cpu_mem_usage=True,
                )

            gpu_mem_gb = torch.cuda.memory_allocated(0) / (1024 ** 3)
            logger.info("Model loaded: %.1f GiB on GPU", gpu_mem_gb)

            # Prepare quantized model for training (casts layernorm to fp32, etc.)
            # Skip for GPTQ — prepare_model_for_kbit_training OOMs on large GPTQ models
            # and is designed for BnB NF4, not GPTQ. Just enable gradient checkpointing.
            if is_prequantized:
                # GPTQ + gradient checkpointing can segfault on some backends.
                # Disable it for GPTQ models — the 4-bit quantization already saves VRAM.
                self.config.gradient_checkpointing = False
                logger.info("GPTQ model: disabled gradient checkpointing (can segfault with GPTQ kernels)")
            elif self.config.load_in_4bit and bitsandbytes is not None:
                # Check if model was quantized with quanto (QBitsTensor can't change dtype)
                _is_quanto = any(
                    "QBitsTensor" in str(type(p)) or "quanto" in str(type(p)).lower()
                    for p in self.model.parameters()
                )
                if _is_quanto:
                    # quanto: skip prepare_model_for_kbit_training (incompatible dtype cast)
                    # Just enable gradient checkpointing for memory savings
                    if self.config.gradient_checkpointing:
                        self.model.gradient_checkpointing_enable()
                    logger.info("Quanto-quantized model: skipped kbit_training prep, gradient checkpointing=%s",
                                self.config.gradient_checkpointing)
                else:
                    self.model = peft.prepare_model_for_kbit_training(
                        self.model,
                        use_gradient_checkpointing=self.config.gradient_checkpointing,
                    )
                    logger.info("Model prepared for k-bit training (BnB NF4)")
            elif self.config.gradient_checkpointing:
                self.model.gradient_checkpointing_enable()

            # Apply LoRA — either resume from existing adapter or create fresh
            if self.resume_from and Path(self.resume_from).exists():
                logger.info("Resuming from existing adapter: %s", self.resume_from)
                self.model = peft.PeftModel.from_pretrained(
                    self.model,
                    self.resume_from,
                    is_trainable=True,
                )
            else:
                if self.resume_from:
                    logger.warning(
                        "resume_from=%s not found, falling back to fresh LoRA",
                        self.resume_from,
                    )
                lora_config = peft.LoraConfig(
                    r=self.config.lora_r,
                    lora_alpha=self.config.lora_alpha,
                    lora_dropout=self.config.lora_dropout,
                    target_modules=self.config.target_modules,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
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
            """Format an instruction-style sample using the model's chat template.

            The training data has 'instruction' (system + user prompt) and 'output'
            (expected assistant response). We format these through the tokenizer's
            chat template so the LoRA learns patterns that activate during normal
            inference — NOT a raw Alpaca format that never appears at inference time.
            """
            instruction = sample.get("instruction", "")
            input_text = sample.get("input", "")
            output = sample.get("output", "")

            # Split instruction into system prompt and user message
            # Format: "System: ...\n\nUser: ..." or just the user message
            system_msg = ""
            user_msg = instruction
            if instruction.startswith("System:"):
                parts = instruction.split("\n\nUser: ", 1)
                if len(parts) == 2:
                    system_msg = parts[0].replace("System: ", "", 1)
                    user_msg = parts[1]
                else:
                    # System prompt contains the user message on the last line
                    lines = instruction.split("\n")
                    # Find the last non-empty line that looks like a user message
                    for i in range(len(lines) - 1, -1, -1):
                        if lines[i].strip() and not lines[i].startswith("System:") and not lines[i].startswith("  "):
                            system_msg = "\n".join(lines[:i]).replace("System: ", "", 1)
                            user_msg = lines[i]
                            break

            if input_text:
                user_msg = f"{user_msg}\n{input_text}"

            # Build messages for chat template
            messages = []
            if system_msg:
                messages.append({"role": "system", "content": system_msg})
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": output})

            # Use the tokenizer's chat template if available
            if self.tokenizer and hasattr(self.tokenizer, 'apply_chat_template'):
                try:
                    text = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False
                    )
                    return {"text": text}
                except Exception:
                    pass

            # Fallback: manual chat template (Qwen-style)
            text = ""
            for msg in messages:
                text += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
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
            # Set up training arguments — epoch-based or step-based
            if self.config.num_train_epochs is not None:
                step_kwargs = {"max_steps": -1, "num_train_epochs": self.config.num_train_epochs}
            else:
                step_kwargs = {"max_steps": self.config.max_steps, "num_train_epochs": 1}

            training_args = transformers.TrainingArguments(
                output_dir=str(self.output_dir / "checkpoints"),
                per_device_train_batch_size=self.config.batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                learning_rate=self.config.learning_rate,
                warmup_steps=self.config.warmup_steps,
                **step_kwargs,
                logging_steps=self.config.logging_steps,
                save_steps=self.config.save_steps,
                bf16=True,
                optim="adamw_torch",
                gradient_checkpointing=self.config.gradient_checkpointing,
                report_to="none",  # Disable wandb/tensorboard
                remove_unused_columns=False,
                dataloader_pin_memory=False,
            )

            # Data collator — mask prompt tokens so loss is computed ONLY on
            # the assistant response. Without this, the model learns to predict
            # template tokens (<|im_start|>, <|im_end|>, <think>) which are
            # trivially easy, drowning out the actual content signal.
            # Mask prompt tokens so loss is computed ONLY on the assistant
            # response. Without this, the model learns to predict template
            # tokens which are trivially easy, drowning out content signal.
            _response_token_ids = self.tokenizer.encode(
                "<|im_start|>assistant\n", add_special_tokens=False
            )
            logger.info(
                "Completion-only masking: response_template=%d tokens %s",
                len(_response_token_ids), _response_token_ids,
            )

            _base_collator = transformers.DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer, mlm=False,
            )

            def _completion_only_collator(features):
                """Mask all tokens before the assistant response template."""
                batch = _base_collator(features)
                for i in range(batch["labels"].shape[0]):
                    labels = batch["labels"][i]
                    input_ids = batch["input_ids"][i]
                    # Find the response template position
                    found = False
                    for j in range(len(input_ids) - len(_response_token_ids) + 1):
                        if input_ids[j:j + len(_response_token_ids)].tolist() == _response_token_ids:
                            # Mask everything up to and including the template
                            labels[:j + len(_response_token_ids)] = -100
                            found = True
                            break
                    if not found:
                        # No response template found — mask everything (skip this sample)
                        labels[:] = -100
                    batch["labels"][i] = labels
                return batch

            data_collator = _completion_only_collator
            logger.info("Using completion-only collator (loss on assistant response only)")

            # Custom callback for progress reporting and convergence detection
            class ProgressCallback(transformers.TrainerCallback):
                def __init__(callback_self, trainer_instance):
                    callback_self.trainer_instance = trainer_instance
                    callback_self._consecutive_below_target = 0
                    callback_self.stop_reason = "max_steps"  # default

                def on_step_end(callback_self, args, state, control, **kwargs):
                    """Write training activation data for brain visualization.

                    Tags events with the tier being trained (based on model path):
                    - 8B/Prime models → tier 'prime' (lights up frontal cortex)
                    - 2B/Core models → tier 'core' (lights up mid-brain)
                    - 0.8B/Nano models → tier 'nano' (lights up brainstem)
                    """
                    try:
                        import json as _json, time as _time, os as _os
                        # Determine which tier is being trained
                        base_path = _os.environ.get("BASE_MODEL_PATH", "").lower()
                        if "8b" in base_path or "prime" in base_path:
                            train_tier = "prime"
                        elif "0.8b" in base_path or "nano" in base_path:
                            train_tier = "nano"
                        else:
                            train_tier = "core"
                        # Collect per-layer gradient magnitudes as "activations"
                        features = []
                        model = callback_self.trainer_instance.model
                        for name, param in model.named_parameters():
                            if param.grad is not None and 'lora' in name:
                                layer_idx = -1
                                for part in name.split('.'):
                                    if part.isdigit():
                                        layer_idx = int(part)
                                        break
                                grad_mag = float(param.grad.abs().mean())
                                features.append({
                                    "idx": hash(name) % 2048,
                                    "strength": min(grad_mag * 100, 20),
                                    "label": name.split('.')[-2] if '.' in name else name,
                                    "layer": layer_idx if layer_idx >= 0 else 12,
                                })
                        if features:
                            features.sort(key=lambda f: f['strength'], reverse=True)
                            features = features[:10]
                            line = _json.dumps({
                                "ts": _time.strftime("%Y-%m-%dT%H:%M:%S", _time.gmtime()),
                                "tier": train_tier,
                                "token": f"train_step_{state.global_step}",
                                "token_idx": state.global_step,
                                "session_id": "training",
                                "features": features,
                            })
                            log_path = _os.environ.get("ACTIVATION_STREAM_PATH", "/logs/activation_stream.jsonl")
                            with open(log_path, "a") as f:
                                f.write(line + "\n")
                    except Exception:
                        pass  # Never crash training for viz

                def on_log(callback_self, args, state, control, logs=None, **kwargs):
                    if logs and "loss" in logs:
                        ti = callback_self.trainer_instance
                        current_loss = logs["loss"]
                        ti._losses.append(current_loss)

                        elapsed = time.time() - ti._start_time

                        # Check 1: Time limit
                        if elapsed > timeout_seconds:
                            logger.warning("Training timeout reached (%ds)", timeout_seconds)
                            callback_self.stop_reason = "timeout"
                            control.should_training_stop = True

                        # Check 2: Convergence — loss below target for N consecutive checks
                        cfg = ti.config
                        if current_loss <= cfg.target_loss:
                            callback_self._consecutive_below_target += 1
                            if callback_self._consecutive_below_target >= cfg.convergence_patience:
                                logger.info(
                                    "Convergence reached: loss %.4f <= %.4f for %d consecutive checks at step %d",
                                    current_loss, cfg.target_loss,
                                    cfg.convergence_patience, state.global_step,
                                )
                                callback_self.stop_reason = "converged"
                                control.should_training_stop = True
                        else:
                            callback_self._consecutive_below_target = 0

                        if ti.progress_callback:
                            progress = TrainingProgress(
                                current_step=state.global_step,
                                total_steps=cfg.max_steps,
                                current_loss=current_loss,
                                avg_loss=sum(ti._losses) / len(ti._losses),
                                elapsed_seconds=elapsed,
                                estimated_remaining=(elapsed / max(state.global_step, 1)) * (cfg.max_steps - state.global_step)
                            )
                            ti.progress_callback(progress)

            # Create trainer
            progress_cb = ProgressCallback(self)
            self.trainer = transformers.Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                data_collator=data_collator,
                callbacks=[progress_cb],
            )

            logger.info(
                "Starting training: max_steps=%d, target_loss=%.4f, patience=%d, timeout=%ds",
                self.config.max_steps, self.config.target_loss,
                self.config.convergence_patience, timeout_seconds,
            )

            # Train!
            train_result = self.trainer.train()

            # Get final metrics
            metrics = {
                "final_loss": train_result.training_loss,
                "total_steps": train_result.global_step,
                "samples_seen": train_result.global_step * self.config.batch_size * self.config.gradient_accumulation_steps,
                "duration_seconds": time.time() - self._start_time,
                "loss_history": self._losses,
                "stop_reason": progress_cb.stop_reason,
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
