"""
Adaptive Training Subprocess — Training + eval in one subprocess.

Extends the training_subprocess pattern with post-training evaluation:
  Load model → Train → Free optimizer → Eval probes → Save results → Exit

The eval runs in the SAME subprocess while the model is still loaded,
avoiding GPU handoff cycles between training and testing phases.

IPC: Writes to ADAPTIVE_PROGRESS_FILE (JSON), read by parent controller.
"""

import gc
import json
import logging
import os
import signal
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

ADAPTIVE_PROGRESS_FILE = Path(os.getenv(
    "ADAPTIVE_PROGRESS_FILE",
    "/shared/study/adaptive_progress.json",
))


@dataclass
class AdaptiveSubprocessConfig:
    """Config for a single adaptive training phase."""

    # Model
    base_model_path: str
    adapter_dir: str
    adapter_name: str

    # Training samples
    samples: List[Dict[str, str]] = field(default_factory=list)

    # QLoRA quantization
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    # LoRA
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )

    # Training
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_steps: int = 300
    warmup_steps: int = 10
    target_loss: float = 0.05
    convergence_patience: int = 10
    num_train_epochs: Optional[int] = None
    max_training_time: int = 1800

    # Incremental
    resume_from: Optional[str] = None

    # Adaptive phase info
    phase_number: int = 1
    skip_training: bool = False  # True for eval-only (test existing adapter)

    # Eval config
    eval_skills: Optional[List[str]] = None  # None = all skills
    pass_threshold: float = 0.7

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AdaptiveSubprocessConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _write_adaptive_progress(
    state: str,
    *,
    phase: int = 0,
    step: int = 0,
    total_steps: int = 0,
    loss: float = 0.0,
    adapter_dir: str = "",
    error: str = "",
    pid: int = 0,
    stop_reason: str = "",
    eval_results: Optional[Dict] = None,
) -> None:
    """Atomically write progress to ADAPTIVE_PROGRESS_FILE."""
    ADAPTIVE_PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "state": state,
        "phase": phase,
        "step": step,
        "total_steps": total_steps,
        "loss": loss,
        "adapter_dir": adapter_dir,
        "error": error,
        "stop_reason": stop_reason,
        "pid": pid or os.getpid(),
        "timestamp": time.time(),
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if eval_results is not None:
        data["eval_results"] = eval_results

    tmp_path = ADAPTIVE_PROGRESS_FILE.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(data, f, default=str)
    os.rename(str(tmp_path), str(ADAPTIVE_PROGRESS_FILE))


_shutdown_requested = False


def _sigterm_handler(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True


def run_adaptive_phase(config_dict: dict) -> None:
    """
    Run one adaptive training phase: Train → Eval → Save.

    Target for multiprocessing.Process(target=run_adaptive_phase).
    All torch/CUDA imports happen inside this function.
    """
    signal.signal(signal.SIGTERM, _sigterm_handler)

    # Per-phase log files — preserved across phases for rollup
    _phase_num = config_dict.get("phase_number", 0)
    _log_dir = Path(os.environ.get("ADAPTIVE_LOG_DIR", "/shared/study/adaptive_logs"))
    _log_dir.mkdir(parents=True, exist_ok=True)
    _log_path = str(_log_dir / f"phase_{_phase_num}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] adaptive-subprocess: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(_log_path, mode="w"),
        ],
    )
    sub_logger = logging.getLogger("adaptive-subprocess")

    config = AdaptiveSubprocessConfig.from_dict(config_dict)
    pid = os.getpid()
    phase = config.phase_number

    _write_adaptive_progress("starting", phase=phase, pid=pid, adapter_dir=config.adapter_dir)
    sub_logger.info("Adaptive phase %d started (PID %d)", phase, pid)

    try:
        # === SETUP ===
        _write_adaptive_progress("setup", phase=phase, pid=pid, adapter_dir=config.adapter_dir)

        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        import torch

        # Force sequential model loading (prevents VRAM spike from parallel bf16 loads).
        # Patch both GLOBAL_WORKERS AND ThreadPoolExecutor to guarantee single-thread.
        # The module variable patch alone is unreliable — transformers may capture
        # the value before our patch via `from ... import GLOBAL_WORKERS`.
        try:
            import transformers.core_model_loading as _cml
            _cml.GLOBAL_WORKERS = 1
            sub_logger.info("Set transformers GLOBAL_WORKERS=1")
        except Exception:
            pass

        # Belt AND suspenders: force all thread pools in core_model_loading to 1 worker
        import concurrent.futures
        _OriginalTPE = concurrent.futures.ThreadPoolExecutor
        class _SingleThreadExecutor(_OriginalTPE):
            def __init__(self, *args, **kwargs):
                kwargs["max_workers"] = 1
                super().__init__(*args, **kwargs)
        concurrent.futures.ThreadPoolExecutor = _SingleThreadExecutor
        sub_logger.info("Patched ThreadPoolExecutor to force max_workers=1")

        from gaia_study.qlora_trainer import QLoRATrainer, QLoRAConfig

        if torch.cuda.is_available():
            free_gb = torch.cuda.mem_get_info(0)[0] / (1024**3)
            total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            sub_logger.info("CUDA: %.1f GiB free / %.1f GiB total", free_gb, total_gb)

            # VRAM guard: ensure enough room for model loading.
            # The orchestrator's wake cycle may reload Nano+Core between
            # MEDITATION handoff and subprocess model loading. If VRAM is
            # insufficient, re-request MEDITATION and wait.
            min_free_gb = 11.0
            if free_gb < min_free_gb:
                sub_logger.warning(
                    "VRAM too low (%.1f GiB < %.1f GiB) — re-requesting MEDITATION",
                    free_gb, min_free_gb,
                )
                import httpx as _httpx
                _orch_url = os.environ.get("ORCHESTRATOR_URL", "http://gaia-orchestrator:6410")
                try:
                    with _httpx.Client(timeout=90.0) as _client:
                        _client.post(f"{_orch_url}/handoff/prime-to-study")
                except Exception as _e:
                    sub_logger.warning("Re-MEDITATION request failed: %s", _e)

                # Poll until VRAM clears
                for _attempt in range(20):
                    import time as _t
                    _t.sleep(3)
                    free_gb = torch.cuda.mem_get_info(0)[0] / (1024**3)
                    sub_logger.info("VRAM wait: %.1f GiB free (attempt %d/20)", free_gb, _attempt + 1)
                    if free_gb >= min_free_gb:
                        break
                else:
                    raise RuntimeError(
                        f"GPU not clear after 60s (only {free_gb:.1f} GiB free, need {min_free_gb})"
                    )

        if _shutdown_requested:
            _write_adaptive_progress("failed", phase=phase, pid=pid, error="Shutdown before setup")
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

        trainer = QLoRATrainer(
            base_model_path=config.base_model_path,
            config=qlora_config,
            output_dir=config.adapter_dir,
            resume_from=config.resume_from,
        )

        # Setup model
        sub_logger.info("Loading model from %s...", config.base_model_path)
        if not trainer.setup():
            raise RuntimeError("Failed to setup QLoRA trainer")

        training_loss = 0.0
        training_steps = 0
        stop_reason = "skipped"

        # === TRAINING (unless skip_training) ===
        if not config.skip_training and config.samples:
            _write_adaptive_progress(
                "training", phase=phase, pid=pid,
                adapter_dir=config.adapter_dir, total_steps=config.max_steps,
            )

            # Progress callback
            def on_progress(progress):
                if _shutdown_requested:
                    raise KeyboardInterrupt("Graceful shutdown")
                _write_adaptive_progress(
                    "training", phase=phase,
                    step=progress.current_step,
                    total_steps=progress.total_steps,
                    loss=progress.current_loss,
                    pid=pid, adapter_dir=config.adapter_dir,
                )

            trainer.progress_callback = on_progress

            sub_logger.info("Preparing dataset (%d samples)...", len(config.samples))
            train_dataset = trainer.prepare_dataset(config.samples, "instruction")

            sub_logger.info("Training phase %d (%d max steps)...", phase, config.max_steps)
            success, metrics = trainer.train(
                train_dataset, config.adapter_name,
                timeout_seconds=config.max_training_time,
            )

            if not success:
                raise RuntimeError(f"Training failed: {metrics.get('error', 'unknown')}")

            training_loss = metrics.get("final_loss", 0.0)
            training_steps = metrics.get("total_steps", 0)
            stop_reason = metrics.get("stop_reason", "max_steps")

            sub_logger.info(
                "Training complete: %d steps, loss=%.4f, reason=%s",
                training_steps, training_loss, stop_reason,
            )

            # Save adapter
            _write_adaptive_progress(
                "saving", phase=phase, pid=pid,
                adapter_dir=config.adapter_dir,
                step=training_steps, loss=training_loss,
            )
            trainer.save_adapter(config.adapter_name, {
                "phase": phase,
                "stop_reason": stop_reason,
                "training_steps": training_steps,
                "training_loss": training_loss,
            })
            sub_logger.info("Adapter saved to %s", config.adapter_dir)

        elif config.skip_training and config.resume_from:
            sub_logger.info("Skip-training mode — evaluating existing adapter at %s", config.resume_from)
            stop_reason = "eval_only"

        # === FREE OPTIMIZER (make room for eval generation) ===
        _write_adaptive_progress(
            "evaluating", phase=phase, pid=pid,
            adapter_dir=config.adapter_dir,
            step=training_steps, loss=training_loss,
        )

        # Delete trainer internals to free optimizer states (~4GB)
        if trainer.trainer is not None:
            del trainer.trainer
            trainer.trainer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        free_gb = torch.cuda.mem_get_info(0)[0] / (1024**3) if torch.cuda.is_available() else 0
        sub_logger.info("Freed optimizer — %.1f GiB free for eval", free_gb)

        # === EVAL PROBES ===
        from gaia_study.skill_eval_probes import evaluate_skills, ALL_SKILLS

        model = trainer.model
        tokenizer = trainer.tokenizer

        eval_target_skills = config.eval_skills or ALL_SKILLS
        sub_logger.info("Running eval probes for %d skills...", len(eval_target_skills))

        skill_results = evaluate_skills(
            model, tokenizer,
            skills=eval_target_skills,
            pass_threshold=config.pass_threshold,
        )

        # Serialize eval results
        eval_results_serialized = {}
        for skill, result in skill_results.items():
            eval_results_serialized[skill] = {
                "skill": result.skill,
                "passed": result.passed,
                "score": result.score,
                "details": result.details,
            }

        # Log summary
        passed = [s for s, r in skill_results.items() if r.passed]
        failed = [s for s, r in skill_results.items() if not r.passed]
        sub_logger.info(
            "Eval complete: %d/%d skills passed",
            len(passed), len(skill_results),
        )
        sub_logger.info("  Passed: %s", ", ".join(passed) or "(none)")
        sub_logger.info("  Failed: %s", ", ".join(failed) or "(none)")

        for skill, result in skill_results.items():
            for detail in result.details:
                sub_logger.info(
                    "  [%s] %s: %s — %s",
                    skill, detail["probe_id"],
                    "PASS" if detail["passed"] else "FAIL",
                    detail["reason"],
                )

        # Save per-phase eval results to a dedicated file (progress.json gets overwritten)
        _eval_results_path = _log_dir / f"phase_{phase}_eval.json"
        with open(_eval_results_path, "w") as _ef:
            json.dump({
                "phase": phase,
                "training_steps": training_steps,
                "training_loss": training_loss,
                "stop_reason": stop_reason,
                "eval_results": eval_results_serialized,
            }, _ef, indent=2, default=str)
        sub_logger.info("Eval results saved to %s", _eval_results_path)

        # === COMPLETE ===
        _write_adaptive_progress(
            "completed", phase=phase, pid=pid,
            adapter_dir=config.adapter_dir,
            step=training_steps, loss=training_loss,
            stop_reason=stop_reason,
            eval_results=eval_results_serialized,
        )

        sub_logger.info("Phase %d complete", phase)

    except KeyboardInterrupt:
        _write_adaptive_progress(
            "failed", phase=phase, pid=pid,
            error="Interrupted",
        )
        sub_logger.warning("Phase %d interrupted", phase)
        sys.exit(1)

    except Exception as e:
        sub_logger.error("Phase %d failed: %s", phase, e, exc_info=True)
        _write_adaptive_progress(
            "failed", phase=phase, pid=pid,
            error=str(e), adapter_dir=config.adapter_dir,
        )
        sys.exit(1)

    finally:
        # Cleanup GPU
        try:
            if 'trainer' in dir() and trainer is not None:
                trainer.cleanup()
        except Exception:
            pass
        sub_logger.info("Subprocess cleanup complete")
