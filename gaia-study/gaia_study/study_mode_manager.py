"""
StudyModeManager - GAIA Self-Study System

Orchestrates the process of:
1. Pausing inference
2. Preparing training data from source documents
3. Running QLoRA training
4. Loading the resulting adapter
5. Resuming inference with new knowledge

Part of Phase 1 implementation of the GAIA LoRA Adapter Architecture.
"""

import gc
import json
import logging
import hashlib
import multiprocessing
import os
import signal
import shutil
import time as _time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class StudyModeState(Enum):
    """States of the study mode process."""
    IDLE = "idle"
    PREPARING = "preparing"
    VALIDATING = "validating"
    TRAINING = "training"
    LOADING = "loading"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class TrainingConfig:
    """Configuration for a training run."""
    adapter_name: str
    tier: int  # 1=global, 2=user, 3=session
    pillar: str  # identity, memory, cognition, embodiment, general
    source_documents: List[str]
    description: str = ""

    # QLoRA parameters (defaults from gaia_constants.json)
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    # Training parameters
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_steps: int = 100
    warmup_steps: int = 10
    target_loss: float = 0.05
    convergence_patience: int = 3

    # Epoch-based training (overrides max_steps when set)
    num_train_epochs: Optional[int] = None

    # Incremental training
    resume_from: Optional[str] = None

    # Governance
    requires_approval: bool = True
    activation_triggers: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


@dataclass
class TrainingResult:
    """Result of a training run."""
    success: bool
    adapter_path: Optional[str] = None
    adapter_name: Optional[str] = None
    metadata_path: Optional[str] = None
    final_loss: Optional[float] = None
    training_steps: int = 0
    duration_seconds: float = 0.0
    error_message: Optional[str] = None
    samples_processed: int = 0


class StudyModeManager:
    """
    Manages GAIA's self-study capabilities.

    Coordinates the process of training LoRA adapters from source documents
    and integrating them into the model pool.
    """

    def __init__(self, config: Dict[str, Any], adapter_base_dir: str | None = None):
        """
        Initialize the StudyModeManager.

        Args:
            config: Study mode configuration from gaia_constants.json
            adapter_base_dir: Base directory for storing adapters
        """
        self.config = config
        if adapter_base_dir is None:
            try:
                from gaia_common.config import Config
                adapter_base_dir = Config.get_instance().model_path("lora_adapters") or "/models/lora_adapters"
            except Exception:
                adapter_base_dir = "/models/lora_adapters"
        self.adapter_base_dir = Path(adapter_base_dir)
        self.state = StudyModeState.IDLE
        self.current_training: Optional[TrainingConfig] = None
        self.progress: float = 0.0
        self.status_message: str = ""
        self._training_process: Optional[multiprocessing.Process] = None
        self._training_start_time: Optional[float] = None

        # Load governance rules
        self.governance = config.get("governance", {})
        self.forbidden_patterns = self.governance.get("forbidden_patterns", [])
        self.max_session_adapters = self.governance.get("max_session_adapters", 3)
        self.max_user_adapters = self.governance.get("max_user_adapters", 10)

        # Training limits
        self.max_training_time = config.get("max_training_time_seconds", 600)
        self.max_training_samples = config.get("max_training_samples", 1000)
        self.max_content_kb = config.get("max_training_content_kb", 100)

        # QLoRA config
        self.qlora_config = config.get("qlora_config", {})

        logger.info(f"StudyModeManager initialized with adapter_base_dir={adapter_base_dir}")

    def validate_content(self, content: str) -> Tuple[bool, str]:
        """
        Validate training content against governance rules.

        Args:
            content: The content to validate

        Returns:
            Tuple of (is_valid, rejection_reason)
        """
        content_lower = content.lower()

        for pattern in self.forbidden_patterns:
            if pattern.lower() in content_lower:
                return False, f"Content contains forbidden pattern: '{pattern}'"

        # Check size limit
        content_kb = len(content.encode('utf-8')) / 1024
        if content_kb > self.max_content_kb:
            return False, f"Content size ({content_kb:.1f}KB) exceeds limit ({self.max_content_kb}KB)"

        return True, ""

    def prepare_training_data(
        self,
        source_documents: List[str],
        output_format: str = "instruction"
    ) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
        """
        Prepare training data from source documents.

        Args:
            source_documents: List of file paths to process
            output_format: Format for training data ("instruction" or "completion")

        Returns:
            Tuple of (training_samples, metadata)
        """
        self.state = StudyModeState.PREPARING
        self.status_message = "Preparing training data..."

        samples = []
        doc_metadata = []
        total_content = ""

        for doc_path in source_documents:
            path = Path(doc_path)
            if not path.exists():
                logger.warning(f"Source document not found: {doc_path}")
                continue

            try:
                content = path.read_text(encoding='utf-8')

                # Validate content
                is_valid, reason = self.validate_content(content)
                if not is_valid:
                    logger.warning(f"Content validation failed for {doc_path}: {reason}")
                    continue

                total_content += content

                # Calculate hash for tracking
                content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

                doc_metadata.append({
                    "path": str(path),
                    "hash": content_hash,
                    "size_bytes": len(content.encode('utf-8'))
                })

                # JSONL files contain pre-curated instruction/output pairs
                if path.suffix == ".jsonl":
                    for line in content.strip().split("\n"):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                            if "instruction" in entry and "output" in entry:
                                samples.append(entry)
                        except json.JSONDecodeError:
                            continue
                # Generate training samples based on format
                elif output_format == "instruction":
                    samples.extend(self._create_instruction_samples(content, path.name))
                else:
                    samples.extend(self._create_completion_samples(content))

                logger.info(f"Processed {doc_path}: {len(samples)} samples so far")

            except Exception as e:
                logger.error(f"Error processing {doc_path}: {e}")
                continue

        # Limit samples
        if len(samples) > self.max_training_samples:
            logger.info(f"Limiting samples from {len(samples)} to {self.max_training_samples}")
            samples = samples[:self.max_training_samples]

        metadata = {
            "source_documents": doc_metadata,
            "total_samples": len(samples),
            "output_format": output_format
        }

        return samples, metadata

    def _create_instruction_samples(self, content: str, doc_name: str) -> List[Dict[str, str]]:
        """
        Create instruction-format training samples from content.

        Generates Q&A style samples that teach the model about the content.
        """
        samples = []

        # Split content into paragraphs or sections
        sections = self._split_into_sections(content)

        for i, section in enumerate(sections):
            if len(section.strip()) < 50:  # Skip very short sections
                continue

            # Create various instruction types

            # Type 1: Direct recall
            samples.append({
                "instruction": f"What does the document '{doc_name}' say about this topic?",
                "input": section[:100] + "...",  # First 100 chars as context
                "output": section
            })

            # Type 2: Content completion
            if len(section) > 200:
                mid = len(section) // 2
                samples.append({
                    "instruction": "Complete the following text:",
                    "input": section[:mid],
                    "output": section[mid:]
                })

            # Type 3: Knowledge retrieval (for poems, facts, etc.)
            samples.append({
                "instruction": f"Recite or recall the following from '{doc_name}':",
                "input": "",
                "output": section
            })

        return samples

    def _create_completion_samples(self, content: str) -> List[Dict[str, str]]:
        """Create completion-format training samples."""
        samples = []
        sections = self._split_into_sections(content)

        for section in sections:
            if len(section.strip()) < 50:
                continue
            samples.append({
                "text": section
            })

        return samples

    def _split_into_sections(self, content: str) -> List[str]:
        """Split content into logical sections."""
        # Try splitting by double newlines (paragraphs)
        sections = content.split('\n\n')

        # If that gives too few sections, try single newlines
        if len(sections) < 3:
            sections = content.split('\n')

        # Filter out empty sections and normalize whitespace
        sections = [s.strip() for s in sections if s.strip()]

        return sections

    async def start_training(
        self,
        config: TrainingConfig,
        model_pool: Any = None
    ) -> TrainingResult:
        """
        Start a training session.

        Args:
            config: Training configuration
            model_pool: Reference to the model pool for coordination

        Returns:
            TrainingResult with outcome details
        """
        import time

        self.current_training = config
        self.state = StudyModeState.VALIDATING
        self.progress = 0.0
        start_time = time.time()

        logger.info(f"Starting study mode training for adapter: {config.adapter_name}")

        try:
            # Step 1: Prepare training data
            samples, data_metadata = self.prepare_training_data(config.source_documents)

            if not samples:
                return TrainingResult(
                    success=False,
                    error_message="No valid training samples could be generated"
                )

            self.progress = 0.2
            self.status_message = f"Prepared {len(samples)} training samples"

            # Step 2: Validate we can proceed
            self.state = StudyModeState.VALIDATING

            # Check adapter count limits
            tier_dir = self._get_tier_directory(config.tier)
            existing_adapters = self._count_adapters_in_tier(tier_dir)

            if config.tier == 3 and existing_adapters >= self.max_session_adapters:
                return TrainingResult(
                    success=False,
                    error_message=f"Session adapter limit reached ({self.max_session_adapters})"
                )

            if config.tier == 2 and existing_adapters >= self.max_user_adapters:
                return TrainingResult(
                    success=False,
                    error_message=f"User adapter limit reached ({self.max_user_adapters})"
                )

            self.progress = 0.3

            # Step 3: Run training
            self.state = StudyModeState.TRAINING
            self.status_message = "Training in progress..."

            adapter_path, final_loss, steps = await self._run_qlora_training(
                samples, config, model_pool
            )

            self.progress = 0.8

            # Step 4: Save metadata
            self.state = StudyModeState.LOADING
            self.status_message = "Saving adapter metadata..."

            metadata_path = self._save_adapter_metadata(
                config, adapter_path, data_metadata, final_loss, steps,
                time.time() - start_time, len(samples)
            )

            self.progress = 0.9

            # Step 5: Optionally load the adapter
            if model_pool and hasattr(model_pool, 'load_adapter'):
                self.status_message = "Loading adapter into model..."
                # This would call the vllm_model.load_adapter() method
                # await model_pool.load_adapter(config.adapter_name, adapter_path, config.tier)

            self.progress = 1.0
            self.state = StudyModeState.COMPLETE
            self.status_message = "Training complete!"

            duration = time.time() - start_time
            logger.info(f"Training completed in {duration:.1f}s for {config.adapter_name}")

            return TrainingResult(
                success=True,
                adapter_path=str(adapter_path),
                adapter_name=config.adapter_name,
                metadata_path=str(metadata_path),
                final_loss=final_loss,
                training_steps=steps,
                duration_seconds=duration,
                samples_processed=len(samples)
            )

        except Exception as e:
            self.state = StudyModeState.FAILED
            self.status_message = f"Training failed: {str(e)}"
            logger.error(f"Study mode training failed: {e}", exc_info=True)

            return TrainingResult(
                success=False,
                error_message=str(e),
                duration_seconds=time.time() - start_time
            )

        finally:
            self.current_training = None

    async def _acquire_gpu(self, min_free_gb: float = 12.0) -> bool:
        """
        Ensure GPU is available for training by requesting orchestrator handoff.

        The parent process does NOT import torch (subprocess isolation).
        We check VRAM via nvidia-smi subprocess call instead.

        Returns:
            True if GPU is ready, False if acquisition failed.
        """
        import subprocess

        # Check free VRAM without importing torch in the parent
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                free_str, total_str = result.stdout.strip().split(", ")
                free_gb = float(free_str) / 1024
                total_gb = float(total_str) / 1024
                logger.info("GPU VRAM: %.1fGiB free / %.1fGiB total", free_gb, total_gb)

                if free_gb >= min_free_gb:
                    logger.info("Sufficient VRAM available — no handoff needed")
                    return True
            else:
                logger.warning("nvidia-smi failed: %s", result.stderr.strip())
        except FileNotFoundError:
            logger.warning("nvidia-smi not found — requesting handoff anyway")
        except Exception as e:
            logger.warning("VRAM check failed: %s — requesting handoff anyway", e)

        # Request orchestrator to clear GPU
        orchestrator_url = "http://gaia-orchestrator:6410"
        logger.info("Requesting GPU handoff from orchestrator")

        try:
            import httpx

            async with httpx.AsyncClient(timeout=90.0) as client:
                resp = await client.post(
                    f"{orchestrator_url}/gpu/handoff",
                    json={"target": "study", "reason": "qlora_training"},
                )
                if resp.status_code == 200:
                    logger.info("GPU handoff to study completed successfully")
                    return True
                else:
                    logger.warning(
                        "GPU handoff request returned %s: %s",
                        resp.status_code, resp.text[:200]
                    )
                    return False

        except Exception as e:
            logger.warning(
                "Could not reach orchestrator for GPU handoff: %s "
                "(manual intervention may be needed — stop gaia-prime/gaia-audio)",
                e
            )
            return False

    async def _release_gpu(self) -> None:
        """
        Release GPU resources after training and signal orchestrator to
        restore normal services (gaia-prime, gaia-audio).

        With subprocess isolation, the parent process never has a CUDA
        context. When the training subprocess exits, the OS reclaims ALL
        GPU memory automatically. We just need to signal the orchestrator.
        """
        # Kill any lingering training subprocess
        self.kill_training_subprocess()
        gc.collect()
        logger.info("GPU release: subprocess dead, VRAM reclaimed by OS")

        # Signal orchestrator to reclaim GPU for inference
        orchestrator_url = "http://gaia-orchestrator:6410"
        try:
            import httpx

            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{orchestrator_url}/gpu/handoff",
                    json={"target": "core", "reason": "training_complete"},
                )
                if resp.status_code == 200:
                    logger.info("GPU reclaim by core/prime completed")
                else:
                    logger.warning("GPU reclaim request returned %s", resp.status_code)

        except Exception as e:
            logger.warning(
                "Could not reach orchestrator for GPU reclaim: %s "
                "(gaia-prime/gaia-audio may need manual restart)",
                e
            )

    async def _run_qlora_training(
        self,
        samples: List[Dict[str, str]],
        config: TrainingConfig,
        model_pool: Any
    ) -> Tuple[Path, float, int]:
        """
        Run the actual QLoRA training process.

        Uses the QLoRATrainer for real training when available,
        falls back to simulation mode for testing.
        """

        # Create adapter directory
        tier_dir = self._get_tier_directory(config.tier)
        adapter_dir = tier_dir / config.adapter_name
        adapter_dir.mkdir(parents=True, exist_ok=True)

        # Save training samples for reference
        samples_path = adapter_dir / "training_samples.json"
        with open(samples_path, 'w') as f:
            json.dump(samples, f, indent=2)

        logger.info(f"Training {len(samples)} samples for {config.max_steps} steps...")

        # Try to use real QLoRA training
        use_real_training = self.config.get("use_real_training", True)
        base_model_path = self.config.get("base_model_path")

        if use_real_training and base_model_path:
            # Acquire GPU — clear other services' VRAM if needed
            await self._acquire_gpu(min_free_gb=12.0)
            try:
                return await self._run_real_qlora_training(
                    samples, config, adapter_dir, base_model_path
                )
            except Exception as e:
                logger.warning(f"Real training failed, falling back to simulation: {e}")
            finally:
                await self._release_gpu()

        # Fallback: Simulation mode for testing
        return await self._run_simulated_training(samples, config, adapter_dir)

    async def _run_real_qlora_training(
        self,
        samples: List[Dict[str, str]],
        config: TrainingConfig,
        adapter_dir: Path,
        base_model_path: str
    ) -> Tuple[Path, float, int]:
        """
        Run QLoRA training in an isolated subprocess.

        The subprocess gets its own CUDA context. When it exits, the OS
        reclaims ALL GPU memory — no empty_cache() needed.
        """
        import asyncio
        from gaia_study.training_subprocess import (
            SubprocessConfig, run_training,
        )

        # Archive existing adapter before overwriting (rollback safety)
        is_incremental = config.resume_from is not None
        if is_incremental and adapter_dir.exists() and (adapter_dir / "adapter_config.json").exists():
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            archive_dir = adapter_dir.parent / f"{adapter_dir.name}_v_{ts}"
            shutil.copytree(adapter_dir, archive_dir)
            logger.info("Archived existing adapter to %s", archive_dir)

        # Build subprocess config (fully serializable)
        sub_config = SubprocessConfig(
            base_model_path=base_model_path,
            adapter_dir=str(adapter_dir),
            adapter_name=config.adapter_name,
            samples=samples,
            load_in_4bit=self.qlora_config.get("load_in_4bit", True),
            bnb_4bit_compute_dtype=self.qlora_config.get("bnb_4bit_compute_dtype", "bfloat16"),
            bnb_4bit_quant_type=self.qlora_config.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=self.qlora_config.get("bnb_4bit_use_double_quant", True),
            lora_r=config.rank,
            lora_alpha=config.alpha,
            lora_dropout=config.dropout,
            target_modules=config.target_modules,
            batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            max_steps=config.max_steps,
            warmup_steps=config.warmup_steps,
            target_loss=config.target_loss,
            convergence_patience=config.convergence_patience,
            num_train_epochs=config.num_train_epochs,
            max_training_time=self.max_training_time,
            resume_from=config.resume_from,
        )

        # Spawn subprocess (fresh interpreter, fresh CUDA context)
        ctx = multiprocessing.get_context("spawn")
        proc = ctx.Process(
            target=run_training,
            args=(sub_config.to_dict(),),
            name=f"qlora-{config.adapter_name}",
        )
        self._training_process = proc
        self._training_start_time = _time.time()
        self.status_message = "Spawning training subprocess..."

        proc.start()
        logger.info(
            "Training subprocess spawned: PID %d for adapter '%s'",
            proc.pid, config.adapter_name,
        )

        # Poll progress file until subprocess exits
        poll_interval = 2.0
        while proc.is_alive():
            await asyncio.sleep(poll_interval)
            progress = self._read_progress()
            if progress:
                state = progress.get("state", "")
                step = progress.get("step", 0)
                total = progress.get("total_steps", config.max_steps)
                loss = progress.get("loss", 0.0)
                if state == "training" and total > 0:
                    self.progress = 0.3 + (0.5 * step / max(total, 1))
                    self.status_message = (
                        f"Training step {step}/{total} (loss: {loss:.4f})"
                    )
                elif state == "setup":
                    self.status_message = "Loading model in subprocess..."
                elif state == "saving":
                    self.status_message = "Saving adapter..."
                    self.progress = 0.8

        # Subprocess has exited — join to get exit code
        proc.join(timeout=10)
        self._training_process = None
        exit_code = proc.exitcode

        logger.info("Training subprocess exited with code %s", exit_code)

        # Read final progress
        progress = self._read_progress()

        if exit_code == 0 and progress and progress.get("state") == "completed":
            final_loss = progress.get("loss", 0.0)
            steps = progress.get("step", 0)
            stop_reason = progress.get("stop_reason", "max_steps")
            logger.info(
                "Subprocess training complete: %d steps, loss=%.4f, stop_reason=%s",
                steps, final_loss, stop_reason,
            )
            return adapter_dir, final_loss, steps
        else:
            error = "Unknown error"
            if progress:
                error = progress.get("error", f"Subprocess exited with code {exit_code}")
            raise RuntimeError(f"Training subprocess failed: {error}")

    def _read_progress(self) -> Optional[Dict[str, Any]]:
        """Read the training progress file (written by subprocess)."""
        from gaia_study.training_subprocess import PROGRESS_FILE
        try:
            if PROGRESS_FILE.exists():
                with open(PROGRESS_FILE) as f:
                    return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
        return None

    def kill_training_subprocess(self) -> bool:
        """
        Kill the training subprocess: SIGTERM → wait 10s → SIGKILL.

        Returns True if a process was killed, False if none was running.
        """
        proc = self._training_process
        if proc is None or not proc.is_alive():
            self._training_process = None
            return False

        pid = proc.pid
        logger.warning("Killing training subprocess PID %d (SIGTERM)...", pid)

        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass

        proc.join(timeout=10)

        if proc.is_alive():
            logger.warning("Subprocess PID %d still alive, sending SIGKILL", pid)
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
            proc.join(timeout=5)

        self._training_process = None
        logger.info("Training subprocess PID %d terminated", pid)
        return True

    async def _run_simulated_training(
        self,
        samples: List[Dict[str, str]],
        config: TrainingConfig,
        adapter_dir: Path
    ) -> Tuple[Path, float, int]:
        """Simulated training for testing without GPU."""
        import asyncio

        logger.info("Running simulated training (no GPU or base model configured)")

        steps_completed = 0
        simulated_loss = 2.5

        for step in range(min(config.max_steps, 10)):
            await asyncio.sleep(0.1)
            steps_completed += 1
            simulated_loss *= 0.95
            self.progress = 0.3 + (0.5 * step / config.max_steps)
            self.status_message = f"[Simulated] Training step {step + 1}/{config.max_steps}"

        # Create placeholder adapter files
        placeholder_weights = adapter_dir / "adapter_model.safetensors"
        placeholder_weights.touch()

        adapter_config = {
            "base_model_name_or_path": "simulated",
            "r": config.rank,
            "lora_alpha": config.alpha,
            "lora_dropout": config.dropout,
            "target_modules": config.target_modules,
            "bias": "none",
            "task_type": "CAUSAL_LM"
        }

        config_path = adapter_dir / "adapter_config.json"
        with open(config_path, 'w') as f:
            json.dump(adapter_config, f, indent=2)

        return adapter_dir, simulated_loss, steps_completed

    def _get_tier_directory(self, tier: int) -> Path:
        """Get the directory path for a specific tier."""
        tier_names = {
            1: "tier1_global",
            2: "tier2_user",
            3: "tier3_session"
        }
        return self.adapter_base_dir / tier_names.get(tier, "tier3_session")

    def _count_adapters_in_tier(self, tier_dir: Path) -> int:
        """Count existing adapters in a tier directory."""
        if not tier_dir.exists():
            return 0
        return len([d for d in tier_dir.iterdir() if d.is_dir()])

    def _save_adapter_metadata(
        self,
        config: TrainingConfig,
        adapter_path: Path,
        training_metadata: Dict[str, Any],
        final_loss: float,
        steps: int,
        duration: float,
        samples: int
    ) -> Path:
        """Save adapter metadata following the schema."""
        now = datetime.now(timezone.utc).isoformat()

        metadata = {
            "name": config.adapter_name,
            "version": "1.0.0",
            "display_name": config.adapter_name.replace("_", " ").title(),
            "description": config.description,
            "tier": config.tier,
            "pillar": config.pillar,
            "rank": config.rank,
            "alpha": config.alpha,
            "target_modules": config.target_modules,
            "created_at": now,
            "updated_at": now,
            "training": {
                "method": "qlora",
                "samples": samples,
                "steps": steps,
                "learning_rate": config.learning_rate,
                "batch_size": config.batch_size,
                "final_loss": final_loss,
                "duration_seconds": duration,
                "source_documents": training_metadata.get("source_documents", []),
                "incremental": config.resume_from is not None,
                "resumed_from": config.resume_from,
            },
            "governance": {
                "requires_approval": config.requires_approval,
                "safety_checked": False,  # Will be validated separately
                "restrictions": []
            },
            "compatibility": {
                "conflicts_with": [],
                "requires": []
            },
            "usage": {
                "load_count": 0,
                "total_tokens_generated": 0
            },
            "tags": config.tags,
            "activation_triggers": config.activation_triggers
        }

        metadata_path = adapter_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved adapter metadata to {metadata_path}")
        return metadata_path

    def get_status(self) -> Dict[str, Any]:
        """Get current study mode status."""
        proc = self._training_process
        result = {
            "state": self.state.value,
            "progress": self.progress,
            "message": self.status_message,
            "current_adapter": self.current_training.adapter_name if self.current_training else None,
            "subprocess_pid": proc.pid if proc and proc.is_alive() else None,
            "subprocess_alive": proc.is_alive() if proc else False,
        }
        # Include progress file data if available
        progress = self._read_progress()
        if progress:
            result["subprocess_state"] = progress.get("state")
            result["subprocess_step"] = progress.get("step", 0)
            result["subprocess_loss"] = progress.get("loss", 0.0)
        return result

    def cancel_training(self) -> bool:
        """Cancel an in-progress training session."""
        if self.state in [StudyModeState.TRAINING, StudyModeState.PREPARING]:
            self.kill_training_subprocess()
            self.state = StudyModeState.IDLE
            self.status_message = "Training cancelled"
            self.current_training = None
            logger.info("Study mode training cancelled")
            return True
        return False

    def list_adapters(self, tier: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        List available adapters.

        Args:
            tier: Optional tier to filter by (1, 2, or 3)

        Returns:
            List of adapter metadata dictionaries
        """
        adapters = []

        tiers_to_check = [tier] if tier else [1, 2, 3]

        for t in tiers_to_check:
            tier_dir = self._get_tier_directory(t)
            if not tier_dir.exists():
                continue

            for adapter_dir in tier_dir.iterdir():
                if not adapter_dir.is_dir():
                    continue

                metadata_path = adapter_dir / "metadata.json"
                if metadata_path.exists():
                    try:
                        with open(metadata_path) as f:
                            adapters.append(json.load(f))
                    except Exception as e:
                        logger.warning(f"Error loading metadata from {metadata_path}: {e}")

        return adapters

    def delete_adapter(self, adapter_name: str, tier: int) -> bool:
        """
        Delete an adapter.

        Args:
            adapter_name: Name of the adapter to delete
            tier: Tier the adapter belongs to

        Returns:
            True if deleted, False if not found or protected
        """
        # Tier 1 adapters cannot be deleted through this interface
        if tier == 1:
            logger.warning(f"Cannot delete tier 1 (global) adapter: {adapter_name}")
            return False

        tier_dir = self._get_tier_directory(tier)
        adapter_dir = tier_dir / adapter_name

        if not adapter_dir.exists():
            logger.warning(f"Adapter not found: {adapter_name} in tier {tier}")
            return False

        try:
            shutil.rmtree(adapter_dir)
            logger.info(f"Deleted adapter: {adapter_name} from tier {tier}")
            return True
        except Exception as e:
            logger.error(f"Error deleting adapter {adapter_name}: {e}")
            return False
