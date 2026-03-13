"""
Training Subprocess — Isolated GPU training for deterministic VRAM release.

Runs QLoRA training in a child process spawned via multiprocessing.Process
with the "spawn" start method. When the child exits, the OS reclaims ALL
GPU memory — guaranteed, no torch.cuda.empty_cache() needed.

IPC: Atomic JSON file at PROGRESS_FILE, writable by subprocess, readable
by parent + orchestrator.

All torch/CUDA imports happen ONLY inside run_training() — the parent
process never touches CUDA.
"""

import json
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

PROGRESS_FILE = Path(os.getenv(
    "TRAINING_PROGRESS_FILE",
    "/shared/study/training_progress.json",
))

# Valid states for the training subprocess
STATES = ("starting", "setup", "training", "saving", "completed", "failed")


@dataclass
class SubprocessConfig:
    """
    Serializable training configuration passed to the subprocess.

    All fields must be JSON-serializable (no Path objects, no callables).
    """
    # Model
    base_model_path: str
    adapter_dir: str
    adapter_name: str

    # Training samples (list of dicts)
    samples: List[Dict[str, str]] = field(default_factory=list)

    # QLoRA quantization
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    # LoRA
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    # Training
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_steps: int = 100
    warmup_steps: int = 10
    target_loss: float = 0.05
    convergence_patience: int = 3
    num_train_epochs: Optional[int] = None
    max_training_time: int = 600

    # Incremental
    resume_from: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SubprocessConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _write_progress(
    state: str,
    *,
    step: int = 0,
    total_steps: int = 0,
    loss: float = 0.0,
    adapter_dir: str = "",
    error: str = "",
    pid: int = 0,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Atomically write progress to PROGRESS_FILE.

    Uses write-to-tmp + os.rename for crash safety.
    """
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "state": state,
        "step": step,
        "total_steps": total_steps,
        "loss": loss,
        "adapter_dir": adapter_dir,
        "error": error,
        "pid": pid or os.getpid(),
        "timestamp": time.time(),
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if extra:
        data.update(extra)

    tmp_path = PROGRESS_FILE.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(data, f)
    os.rename(str(tmp_path), str(PROGRESS_FILE))


# Flag for graceful shutdown
_shutdown_requested = False


def _sigterm_handler(signum, frame):
    """Handle SIGTERM for graceful shutdown."""
    global _shutdown_requested
    _shutdown_requested = True
    logger.warning("SIGTERM received — requesting graceful shutdown")


def run_training(config_dict: dict) -> None:
    """
    Top-level entry point for the training subprocess.

    This function is the target for multiprocessing.Process(target=run_training).
    It must be a top-level function (picklable for spawn).

    All torch/CUDA imports happen inside this function — the parent process
    never touches CUDA.

    Args:
        config_dict: Serialized SubprocessConfig (dict)
    """
    # Install signal handler early
    signal.signal(signal.SIGTERM, _sigterm_handler)

    # Configure logging in subprocess
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] training-subprocess: %(message)s",
    )
    sub_logger = logging.getLogger("training-subprocess")

    config = SubprocessConfig.from_dict(config_dict)
    pid = os.getpid()

    _write_progress("starting", pid=pid, adapter_dir=config.adapter_dir)
    sub_logger.info("Training subprocess started (PID %d)", pid)

    try:
        # === SETUP ===
        _write_progress("setup", pid=pid, adapter_dir=config.adapter_dir)
        sub_logger.info("Importing torch and training libraries...")

        # All CUDA imports happen here — in the subprocess only
        import torch
        from gaia_study.qlora_trainer import QLoRATrainer, QLoRAConfig

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            free_bytes, total_bytes = torch.cuda.mem_get_info(0)
            sub_logger.info(
                "CUDA available: %s (%.1f GiB free / %.1f GiB total)",
                gpu_name,
                free_bytes / (1024**3),
                total_bytes / (1024**3),
            )
        else:
            sub_logger.warning("No CUDA device available — training may fail")

        if _shutdown_requested:
            _write_progress("failed", pid=pid, error="Shutdown before setup complete")
            sys.exit(1)

        # Build QLoRA config
        qlora_config = QLoRAConfig(
            load_in_4bit=config.load_in_4bit,
            bnb_4bit_compute_dtype=config.bnb_4bit_compute_dtype,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
            lora_r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.target_modules,
            batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            max_steps=config.max_steps,
            warmup_steps=config.warmup_steps,
            target_loss=config.target_loss,
            convergence_patience=config.convergence_patience,
            num_train_epochs=config.num_train_epochs,
        )

        # Progress callback writes to shared file
        def on_progress(progress):
            if _shutdown_requested:
                raise KeyboardInterrupt("Graceful shutdown requested")
            _write_progress(
                "training",
                step=progress.current_step,
                total_steps=progress.total_steps,
                loss=progress.current_loss,
                pid=pid,
                adapter_dir=config.adapter_dir,
            )

        trainer = QLoRATrainer(
            base_model_path=config.base_model_path,
            config=qlora_config,
            output_dir=config.adapter_dir,
            progress_callback=on_progress,
            resume_from=config.resume_from,
        )

        # Setup model
        sub_logger.info("Loading model from %s...", config.base_model_path)
        if not trainer.setup():
            raise RuntimeError("Failed to setup QLoRA trainer")

        if _shutdown_requested:
            trainer.cleanup()
            _write_progress("failed", pid=pid, error="Shutdown during setup")
            sys.exit(1)

        # Prepare dataset
        sub_logger.info("Preparing dataset (%d samples)...", len(config.samples))
        train_dataset = trainer.prepare_dataset(config.samples, "instruction")

        if _shutdown_requested:
            trainer.cleanup()
            _write_progress("failed", pid=pid, error="Shutdown during dataset prep")
            sys.exit(1)

        # === TRAINING ===
        _write_progress(
            "training", pid=pid, adapter_dir=config.adapter_dir,
            total_steps=config.max_steps,
        )
        sub_logger.info("Starting training (%d max steps)...", config.max_steps)

        success, metrics = trainer.train(
            train_dataset, config.adapter_name, config.max_training_time
        )

        if not success:
            error_msg = metrics.get("error", "Training failed")
            _write_progress("failed", pid=pid, error=error_msg)
            trainer.cleanup()
            sys.exit(1)

        # === SAVING ===
        _write_progress("saving", pid=pid, adapter_dir=config.adapter_dir)
        sub_logger.info("Saving adapter...")

        stop_reason = metrics.get("stop_reason", "max_steps")
        trainer.save_adapter(config.adapter_name, {"stop_reason": stop_reason})

        final_loss = metrics.get("final_loss", 0.0)
        total_steps = metrics.get("total_steps", 0)

        # Cleanup trainer (releases model from GPU)
        trainer.cleanup()

        # === COMPLETED ===
        _write_progress(
            "completed",
            step=total_steps,
            total_steps=total_steps,
            loss=final_loss,
            pid=pid,
            adapter_dir=config.adapter_dir,
            extra={
                "stop_reason": stop_reason,
                "duration_seconds": time.time(),  # parent will compute delta
            },
        )
        sub_logger.info(
            "Training complete: %d steps, loss=%.4f, stop_reason=%s",
            total_steps, final_loss, stop_reason,
        )
        sys.exit(0)

    except KeyboardInterrupt:
        sub_logger.warning("Training interrupted by shutdown request")
        _write_progress("failed", pid=pid, error="Interrupted by shutdown")
        sys.exit(1)

    except Exception as e:
        sub_logger.error("Training subprocess failed: %s", e, exc_info=True)
        _write_progress(
            "failed", pid=pid, error=str(e), adapter_dir=config.adapter_dir
        )
        sys.exit(1)
