"""
Adaptive Training Controller — phase loop with regression detection.

Orchestrates the adaptive training pipeline:
  Phase 1: Train ALL skills → Eval ALL → Analyze
  Phase N: Train FAILED+REGRESSED → Eval ALL → Analyze
  Terminate when all pass or max_phases reached.

Each phase runs in an isolated subprocess (deterministic VRAM release).
"""

import asyncio
import gc
import json
import logging
import multiprocessing
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import httpx

from gaia_study.adaptive_subprocess import (
    ADAPTIVE_PROGRESS_FILE,
    AdaptiveSubprocessConfig,
    run_adaptive_phase,
)
from gaia_study.skill_eval_probes import ALL_SKILLS, NANO_SKILLS

ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://gaia-orchestrator:6410")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PhaseResult:
    """Result of one adaptive training phase."""
    phase: int
    skills_trained: List[str]
    samples_used: int
    training_steps: int
    training_loss: float
    stop_reason: str
    eval_results: Dict[str, Dict]  # skill -> {passed, score, details}
    passed_skills: List[str]
    failed_skills: List[str]
    regressed_skills: List[str]    # previously passed, now failed
    adapter_path: str
    duration_seconds: float


@dataclass
class AdaptiveTrainingState:
    """Full state of an adaptive training run."""
    adapter_name: str
    base_model_path: str
    train_data_path: str
    resume_from: Optional[str] = None

    # Config
    max_phases: int = 6
    pass_threshold: float = 0.7
    rank: int = 32
    alpha: int = 64
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )
    max_steps_phase1: int = 300
    max_steps_repair: int = 100
    training_timeout: int = 1800

    # Runtime state
    status: str = "idle"  # idle, running, completed, failed, max_phases_reached, cancelled
    current_phase: int = 0
    phase_results: List[Dict] = field(default_factory=list)
    globally_passed: List[str] = field(default_factory=list)
    current_adapter_path: Optional[str] = None
    start_time: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Curriculum filtering
# ---------------------------------------------------------------------------

def load_curriculum(path: str) -> List[Dict]:
    """Load training samples from JSON file."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    raise ValueError(f"Expected JSON array, got {type(data)}")


def filter_samples_by_skills(
    samples: List[Dict],
    target_skills: Set[str],
) -> List[Dict]:
    """Select samples that teach at least one target skill.

    A sample is included if its skills set intersects target_skills.
    Sorted by weight descending so important samples train first.
    """
    filtered = [
        s for s in samples
        if set(s.get("skills", [])) & target_skills
    ]
    filtered.sort(key=lambda s: s.get("weight", 1.0), reverse=True)
    return filtered


def boost_regressed_samples(
    samples: List[Dict],
    regressed_skills: Set[str],
    boost_factor: int = 2,
) -> List[Dict]:
    """Duplicate samples for regressed skills to give them more training signal."""
    boosted = list(samples)
    for s in samples:
        if set(s.get("skills", [])) & regressed_skills:
            # Add duplicates for regression repair
            for _ in range(boost_factor - 1):
                boosted.append(s)
    return boosted


def add_anchor_samples(
    repair_samples: List[Dict],
    all_samples: List[Dict],
    anchor_ratio: float = 0.2,
) -> List[Dict]:
    """Add a random subset of ALL samples to prevent regression during repair.

    Without anchoring, training only on failed-skill samples overwrites
    previously learned skills. The anchor samples remind the model of
    everything it already knows.

    Args:
        repair_samples: Samples targeting failed/regressed skills.
        all_samples: The full curriculum.
        anchor_ratio: Fraction of full curriculum to include as anchors.
    """
    import random
    n_anchors = max(10, int(len(all_samples) * anchor_ratio))
    anchors = random.sample(all_samples, min(n_anchors, len(all_samples)))
    # Deduplicate by instruction text
    repair_instructions = {s.get("instruction", "") for s in repair_samples}
    unique_anchors = [a for a in anchors if a.get("instruction", "") not in repair_instructions]
    combined = repair_samples + unique_anchors
    random.shuffle(combined)
    return combined


# ---------------------------------------------------------------------------
# Phase analysis
# ---------------------------------------------------------------------------

def analyze_phase(
    eval_results: Dict[str, Dict],
    previously_passed: Set[str],
) -> tuple[Set[str], Set[str], Set[str]]:
    """Analyze eval results and detect regressions.

    Returns:
        (passed_skills, failed_skills, regressed_skills)
    """
    passed = {sk for sk, r in eval_results.items() if r.get("passed", False)}
    failed = {sk for sk, r in eval_results.items() if not r.get("passed", False)}

    # Regression: previously passed but now failing
    regressed = previously_passed & failed

    return passed, failed, regressed


# ---------------------------------------------------------------------------
# Adaptive Trainer
# ---------------------------------------------------------------------------

class AdaptiveTrainer:
    """Orchestrates multi-phase adaptive training."""

    def __init__(self, state: AdaptiveTrainingState):
        self.state = state
        self._cancel_requested = False
        self._subprocess: Optional[multiprocessing.Process] = None

    def cancel(self):
        """Request cancellation at the next phase boundary."""
        self._cancel_requested = True
        logger.info("Cancel requested — will stop after current phase")

    async def _acquire_gpu(self, min_free_gb: float = 11.0) -> bool:
        """Request GPU handoff to MEDITATION mode and verify VRAM is clear."""
        import subprocess as sp

        # Request MEDITATION
        try:
            async with httpx.AsyncClient(timeout=90.0) as client:
                resp = await client.post(f"{ORCHESTRATOR_URL}/handoff/prime-to-study")
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("ok"):
                        logger.info("GPU handoff accepted (%.1fs)", data.get("elapsed_s", 0))
                else:
                    logger.warning("GPU handoff returned %s: %s", resp.status_code, resp.text[:200])
        except Exception as e:
            logger.warning("GPU handoff request failed: %s", e)

        # Poll VRAM until clear (Nano/Core may take a few seconds to unload)
        for attempt in range(15):
            await asyncio.sleep(3)
            try:
                result = sp.run(
                    ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0:
                    free_mb = float(result.stdout.strip())
                    free_gb = free_mb / 1024
                    if free_gb >= min_free_gb:
                        logger.info("GPU ready: %.1f GiB free (attempt %d)", free_gb, attempt + 1)
                        return True
                    logger.info("GPU not ready: %.1f GiB free < %.1f GiB needed (attempt %d)", free_gb, min_free_gb, attempt + 1)
            except Exception as e:
                logger.warning("VRAM check failed: %s", e)

        logger.error("GPU did not clear after 45s — Nano/Core may still be loaded")
        return False

    async def _release_gpu(self) -> None:
        """Signal orchestrator to restore inference services."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(f"{ORCHESTRATOR_URL}/handoff/study-to-prime")
                if resp.status_code == 200:
                    logger.info("GPU released — orchestrator restoring services")
                else:
                    logger.warning("GPU release returned %s", resp.status_code)
        except Exception as e:
            logger.warning("GPU release failed: %s", e)

    async def run(self) -> AdaptiveTrainingState:
        """Run the adaptive training loop."""
        state = self.state
        state.status = "running"
        state.start_time = time.time()

        logger.info(
            "Starting adaptive training: adapter=%s, base=%s, max_phases=%d",
            state.adapter_name, state.base_model_path, state.max_phases,
        )

        try:
            # Load curriculum
            all_samples = load_curriculum(state.train_data_path)
            logger.info("Loaded %d training samples", len(all_samples))

            # Auto-detect skills from curriculum (only eval what we train)
            curriculum_skills = set()
            for s in all_samples:
                curriculum_skills.update(s.get("skills", []))
            # Only eval skills that have probes defined
            from gaia_study.skill_eval_probes import SKILL_PROBES
            eval_skills = sorted(curriculum_skills & set(SKILL_PROBES.keys()))
            logger.info("Eval skills (from curriculum): %s", eval_skills)

            # Determine adapter base dir
            adapter_base = Path("/models/lora_adapters/tier1_global")
            adapter_base.mkdir(parents=True, exist_ok=True)

            # Auto-detect Nano model to restrict evaluation scope (0.8B weights can't learn everything)
            is_nano = "nano" in state.base_model_path.lower() or "0.8b" in state.base_model_path.lower()
            if is_nano:
                logger.info("Nano model detected — capping evaluation to NANO_SKILLS subset")
                eval_skills = sorted(set(eval_skills) & set(NANO_SKILLS))
                logger.info("Restricted Nano eval skills: %s", eval_skills)

            globally_passed: Set[str] = set(state.globally_passed)
            current_adapter = state.resume_from

            for phase_num in range(1, state.max_phases + 1):
                if self._cancel_requested:
                    state.status = "cancelled"
                    logger.info("Cancelled at phase boundary")
                    break

                state.current_phase = phase_num
                logger.info("=== Phase %d ===", phase_num)

                # Determine what to train
                if phase_num == 1 and state.resume_from:
                    # Eval-only: test existing adapter
                    skills_to_train = set()
                    samples_for_phase = []
                    skip_training = True
                    adapter_name = f"{state.adapter_name}_eval"
                    # Use the resume_from adapter dir
                    adapter_dir = state.resume_from
                    resume_from = state.resume_from
                elif phase_num == 1:
                    # Full training — only train on skills present in the curriculum
                    # (Prevents Phase Drift where we eval things we haven't taught yet)
                    skills_to_train = set(eval_skills)
                    samples_for_phase = all_samples
                    skip_training = False
                    adapter_name = f"{state.adapter_name}_p{phase_num}"
                    adapter_dir = str(adapter_base / adapter_name)
                    resume_from = None
                else:
                    # Repair: train only failed + regressed
                    prev = state.phase_results[-1]
                    skills_to_train = set(prev.get("failed_skills", [])) | set(prev.get("regressed_skills", []))
                    if not skills_to_train:
                        logger.info("No skills to train — all passed!")
                        state.status = "completed"
                        break

                    samples_for_phase = filter_samples_by_skills(all_samples, skills_to_train)
                    # Boost regressed skills
                    regressed = set(prev.get("regressed_skills", []))
                    if regressed:
                        samples_for_phase = boost_regressed_samples(samples_for_phase, regressed)

                    # Add anchor samples from full curriculum to prevent regression.
                    # Without this, training on small repair subsets overwrites
                    # previously learned skills.
                    samples_for_phase = add_anchor_samples(
                        samples_for_phase, all_samples, anchor_ratio=0.25,
                    )
                    logger.info(
                        "Phase %d: %d samples after anchoring (was %d before)",
                        phase_num, len(samples_for_phase), len(samples_for_phase),
                    )

                    skip_training = False
                    adapter_name = f"{state.adapter_name}_p{phase_num}"
                    adapter_dir = str(adapter_base / adapter_name)
                    resume_from = current_adapter

                # Scale max_steps for repair phases
                if phase_num > 1 and not skip_training and samples_for_phase:
                    ratio = len(samples_for_phase) / max(len(all_samples), 1)
                    max_steps = max(30, int(state.max_steps_repair * max(ratio, 0.3)))
                else:
                    max_steps = state.max_steps_phase1

                logger.info(
                    "Phase %d: skills=%s, samples=%d, steps=%d, skip_training=%s, resume=%s",
                    phase_num,
                    sorted(skills_to_train) if skills_to_train else "(eval-only)",
                    len(samples_for_phase), max_steps, skip_training,
                    resume_from,
                )

                # Acquire GPU for this phase (enter MEDITATION)
                await self._acquire_gpu()

                # Run phase in subprocess
                phase_start = time.time()
                phase_result = await self._run_phase_subprocess(
                    phase_num=phase_num,
                    samples=samples_for_phase,
                    adapter_dir=adapter_dir,
                    adapter_name=adapter_name,
                    resume_from=resume_from,
                    skip_training=skip_training,
                    max_steps=max_steps,
                    eval_skills=eval_skills,
                )

                if phase_result is None:
                    state.status = "failed"
                    state.error = "Phase subprocess failed"
                    break

                # Analyze results
                eval_results = phase_result.get("eval_results", {})
                passed, failed, regressed = analyze_phase(eval_results, globally_passed)

                # Build phase result record
                pr = {
                    "phase": phase_num,
                    "skills_trained": sorted(skills_to_train),
                    "samples_used": len(samples_for_phase),
                    "training_steps": phase_result.get("step", 0),
                    "training_loss": phase_result.get("loss", 0.0),
                    "stop_reason": phase_result.get("stop_reason", ""),
                    "eval_results": eval_results,
                    "passed_skills": sorted(passed),
                    "failed_skills": sorted(failed),
                    "regressed_skills": sorted(regressed),
                    "adapter_path": phase_result.get("adapter_dir", adapter_dir),
                    "duration_seconds": time.time() - phase_start,
                }
                state.phase_results.append(pr)

                # Update tracking
                globally_passed |= passed
                # Remove regressed skills from globally_passed
                globally_passed -= regressed
                state.globally_passed = sorted(globally_passed)

                if not skip_training:
                    current_adapter = adapter_dir
                state.current_adapter_path = current_adapter

                logger.info(
                    "Phase %d results: %d passed, %d failed, %d regressed",
                    phase_num, len(passed), len(failed), len(regressed),
                )

                # Check completion
                if not failed:
                    state.status = "completed"
                    logger.info("All skills passed! Adaptive training complete.")
                    break

            else:
                # Exhausted max_phases
                state.status = "max_phases_reached"
                logger.warning(
                    "Max phases (%d) reached. Remaining failed: %s",
                    state.max_phases,
                    sorted(failed) if 'failed' in dir() else "unknown",
                )

        except Exception as e:
            state.status = "failed"
            state.error = str(e)
            logger.error("Adaptive training failed: %s", e, exc_info=True)

        elapsed = time.time() - state.start_time
        logger.info(
            "Adaptive training finished: status=%s, phases=%d, elapsed=%.1fs",
            state.status, state.current_phase, elapsed,
        )
        return state

    async def _run_phase_subprocess(
        self,
        phase_num: int,
        samples: List[Dict],
        adapter_dir: str,
        adapter_name: str,
        resume_from: Optional[str],
        skip_training: bool,
        max_steps: int,
        eval_skills: Optional[List[str]] = None,
    ) -> Optional[Dict]:
        """Spawn a subprocess for one phase and wait for completion."""
        state = self.state

        sub_config = AdaptiveSubprocessConfig(
            base_model_path=state.base_model_path,
            adapter_dir=adapter_dir,
            adapter_name=adapter_name,
            samples=samples,
            lora_r=state.rank,
            lora_alpha=state.alpha,
            target_modules=state.target_modules,
            max_steps=max_steps,
            max_training_time=state.training_timeout,
            resume_from=resume_from,
            phase_number=phase_num,
            skip_training=skip_training,
            pass_threshold=state.pass_threshold,
            eval_skills=eval_skills,
        )

        ctx = multiprocessing.get_context("spawn")
        proc = ctx.Process(
            target=run_adaptive_phase,
            args=(sub_config.to_dict(),),
            name=f"adaptive-phase-{phase_num}",
        )
        self._subprocess = proc
        proc.start()
        logger.info("Phase %d subprocess spawned: PID %d", phase_num, proc.pid)

        # Poll until done
        while proc.is_alive():
            await asyncio.sleep(2.0)
            if self._cancel_requested:
                logger.info("Sending SIGTERM to phase subprocess")
                proc.terminate()
                proc.join(timeout=30)
                return None

        proc.join(timeout=10)
        self._subprocess = None
        exit_code = proc.exitcode

        logger.info("Phase %d subprocess exited: code=%s", phase_num, exit_code)

        # Read results
        try:
            with open(ADAPTIVE_PROGRESS_FILE) as f:
                progress = json.load(f)
        except Exception as e:
            logger.error("Could not read progress file: %s", e)
            return None

        if progress.get("state") == "completed":
            return progress
        else:
            error = progress.get("error", f"exit code {exit_code}")
            logger.error("Phase %d failed: %s", phase_num, error)
            return None
