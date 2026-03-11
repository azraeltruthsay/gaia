#!/usr/bin/env python3
"""Self-Awareness Training Pipeline — Full Identity Baking.

Orchestrates a 13-stage pipeline that evaluates, trains, merges, quantizes,
and deploys identity knowledge across all model tiers (Prime/Core/Nano).

Each stage is a pause point. State is persisted to a JSON file so the
pipeline can be resumed, replayed from a specific stage, or dry-run.

Runs inside the **gaia-study** container (has torch, transformers, PEFT,
and RW access to /models/).

Usage:
    # Full run (interactive — pauses between stages)
    docker compose exec gaia-study python scripts/self_awareness_pipeline.py

    # Resume from last completed stage
    docker compose exec gaia-study python scripts/self_awareness_pipeline.py --resume

    # Jump to a specific stage
    docker compose exec gaia-study python scripts/self_awareness_pipeline.py --stage TRAIN_4B

    # Run up to a stage and pause
    docker compose exec gaia-study python scripts/self_awareness_pipeline.py --pause-after MERGE_4B

    # Dry run (print stages, don't execute)
    docker compose exec gaia-study python scripts/self_awareness_pipeline.py --dry-run

    # Skip Nano (only train Core/Prime)
    docker compose exec gaia-study python scripts/self_awareness_pipeline.py --skip-nano
"""

import argparse
import json
import logging
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("GAIA.Pipeline.SelfAwareness")

# ── Constants ────────────────────────────────────────────────────────────────

STATE_FILE = "/shared/pipeline/self_awareness_state.json"
PAUSE_FLAG = "/shared/pipeline/PAUSE_REQUESTED"

CURRICULUM_PATH = "/knowledge/curricula/self-model/train.jsonl"
FILTERED_PATH = "/knowledge/curricula/self-model/train_filtered.jsonl"

# Model paths
BASE_4B = "/models/Qwen3.5-4B-Abliterated"
BASE_08B = "/models/Qwen3.5-0.8B-Abliterated"
MERGED_4B = "/models/Qwen3.5-4B-Abliterated-merged"
GGUF_CORE = "/models/Qwen3.5-4B-Abliterated-Q4_K_M.gguf"
GGUF_NANO = "/models/Qwen3.5-0.8B-Abliterated-Q8_0.gguf"
BAKED_DIR = "/models/baked"
ADAPTER_DIR = "/models/lora_adapters/tier1_global"

# Endpoints (inside Docker network)
CORE_CPU_ENDPOINT = os.environ.get("CORE_CPU_ENDPOINT", "http://gaia-core:6415")
NANO_ENDPOINT = os.environ.get("NANO_ENDPOINT", "http://gaia-nano:8080")
ORCHESTRATOR = os.environ.get("ORCHESTRATOR_ENDPOINT", "http://gaia-orchestrator:6410")
CORE_EVAL_ENDPOINT = os.environ.get("CORE_EVAL_ENDPOINT", "http://gaia-core:8092")

# Training config defaults (overridable via gaia_constants.json)
DEFAULT_LORA_RANK = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_TRAIN_EPOCHS = 3
DEFAULT_TRAIN_BATCH = 2
DEFAULT_LR = 2e-4
DEFAULT_THRESHOLD = 0.5

# ── Stage Definitions ────────────────────────────────────────────────────────

STAGES = [
    "BUILD_CURRICULUM",
    "PRE_EVAL_4B",
    "FILTER_DELTA_4B",
    "GPU_ACQUIRE",
    "TRAIN_4B",
    "MERGE_4B",
    "GGUF_CORE",
    "DEPLOY_PRIME",
    "RELOAD_CORE",
    "TRAIN_NANO",
    "MERGE_NANO",
    "GGUF_NANO",
    "DEPLOY_NANO",
    "POST_EVAL",
    "COGNITIVE_SMOKE",
]

DOCTOR_ENDPOINT = os.environ.get("DOCTOR_ENDPOINT", "http://gaia-doctor:6419")

NANO_STAGES = {"TRAIN_NANO", "MERGE_NANO", "GGUF_NANO", "DEPLOY_NANO"}


# ── Pipeline Context ─────────────────────────────────────────────────────────

@dataclass
class PipelineContext:
    """Carries state between pipeline stages."""
    pipeline_id: str = ""
    started_at: str = ""
    curriculum_path: str = CURRICULUM_PATH
    filtered_path: str = FILTERED_PATH
    threshold: float = DEFAULT_THRESHOLD
    skip_nano: bool = False
    dry_run: bool = False
    backup: bool = True

    skip_curriculum_build: bool = False
    skip_smoke: bool = False
    smoke_threshold: float = 0.85

    # Populated during execution
    smoke_pass_rate: float | None = None
    pre_eval_metrics: dict = field(default_factory=dict)
    post_eval_metrics: dict = field(default_factory=dict)
    delta_count: int = 0
    adapter_4b_path: str = ""
    adapter_nano_path: str = ""
    merged_4b_path: str = MERGED_4B
    gpu_lease_id: str = ""
    final_loss_4b: float | None = None
    final_loss_nano: float | None = None


@dataclass
class StageResult:
    """Return value from a stage function."""
    ok: bool
    message: str = ""
    metrics: dict = field(default_factory=dict)


# ── State Persistence ────────────────────────────────────────────────────────

def load_state() -> dict:
    """Load pipeline state from disk."""
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_state(state: dict):
    """Persist pipeline state to disk."""
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def init_state(ctx: PipelineContext) -> dict:
    """Initialize a fresh pipeline state."""
    return {
        "pipeline_id": ctx.pipeline_id,
        "started_at": ctx.started_at,
        "curriculum_path": ctx.curriculum_path,
        "threshold": ctx.threshold,
        "stages": {stage: {"status": "pending"} for stage in STAGES},
        "adapters": {
            "4b": {"path": None, "final_loss": None},
            "nano": {"path": None, "final_loss": None},
        },
        "pre_eval": {"core_avg_f1": None, "nano_avg_f1": None},
        "post_eval": {"core_avg_f1": None, "nano_avg_f1": None},
        "alignment_status": "UNTRAINED",
    }


def mark_stage(state: dict, stage: str, status: str, **extra):
    """Update a stage's status and any extra metadata."""
    if stage not in state["stages"]:
        state["stages"][stage] = {}
    state["stages"][stage]["status"] = status
    if status == "completed":
        state["stages"][stage]["completed_at"] = datetime.now(timezone.utc).isoformat()
    state["stages"][stage].update(extra)
    save_state(state)


# ── HTTP Helpers ─────────────────────────────────────────────────────────────

def http_post(url: str, data: dict | None = None, timeout: int = 30) -> dict:
    """POST JSON to a URL and return the response dict."""
    payload = json.dumps(data or {}).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    resp = urllib.request.urlopen(req, timeout=timeout)
    return json.loads(resp.read().decode("utf-8"))


def http_get(url: str, timeout: int = 15) -> dict:
    """GET a URL and return the response dict."""
    resp = urllib.request.urlopen(url, timeout=timeout)
    return json.loads(resp.read().decode("utf-8"))


def wait_for_health(url: str, timeout: int = 180, interval: int = 5) -> bool:
    """Poll a /health endpoint until it returns 200."""
    elapsed = 0
    while elapsed < timeout:
        try:
            resp = urllib.request.urlopen(url, timeout=5)
            if resp.status == 200:
                return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(interval)
        elapsed += interval
    return False


# ── Stage Implementations ────────────────────────────────────────────────────

def stage_build_curriculum(ctx: PipelineContext) -> StageResult:
    """Regenerate train.jsonl from all living knowledge sources."""
    logger.info("═══ BUILD_CURRICULUM: Generating dynamic curriculum ═══")

    if ctx.skip_curriculum_build:
        return StageResult(ok=True, message="Skipped (--skip-curriculum-build)")

    try:
        # Import and run the curriculum builder
        sys.path.insert(0, str(Path(__file__).parent))
        from build_curriculum import build_curriculum
        metadata = build_curriculum(datasets="A,B,C,S", dry_run=ctx.dry_run)

        total = metadata.get("total_pairs", 0)
        logger.info("BUILD_CURRICULUM complete: %d pairs generated", total)

        return StageResult(
            ok=True,
            message=f"{total} pairs generated",
            metrics=metadata,
        )
    except Exception as e:
        logger.exception("BUILD_CURRICULUM failed")
        return StageResult(ok=False, message=str(e))


def stage_pre_eval_4b(ctx: PipelineContext) -> StageResult:
    """Score curriculum samples against Core CPU (localhost:8092)."""
    logger.info("═══ PRE_EVAL_4B: Evaluating %s against Core CPU ═══", ctx.curriculum_path)

    # Import pre_eval functions
    sys.path.insert(0, str(Path(__file__).parent))
    from pre_eval_curriculum import load_curriculum, query_model, token_f1

    samples = load_curriculum(ctx.curriculum_path)
    if not samples:
        return StageResult(ok=False, message=f"No samples in {ctx.curriculum_path}")

    logger.info("Evaluating %d samples (threshold=%.2f)...", len(samples), ctx.threshold)

    learned = 0
    gaps = 0
    errors = 0
    total_f1 = 0.0

    for i, sample in enumerate(samples):
        instruction = sample.get("instruction", "")
        expected = sample.get("output", "")
        try:
            predicted = query_model(CORE_EVAL_ENDPOINT, instruction, max_tokens=256, timeout=30)
            f1 = token_f1(predicted, expected)
            total_f1 += f1
            if f1 > ctx.threshold:
                learned += 1
            else:
                gaps += 1
        except Exception as e:
            errors += 1
            if i < 3:
                logger.warning("Sample %d error: %s", i, e)

        if (i + 1) % 50 == 0:
            logger.info("  [%d/%d] learned=%d gaps=%d errors=%d", i + 1, len(samples), learned, gaps, errors)

    scored = learned + gaps
    avg_f1 = total_f1 / scored if scored > 0 else 0.0

    metrics = {
        "total": len(samples),
        "learned": learned,
        "gaps": gaps,
        "errors": errors,
        "avg_f1": round(avg_f1, 4),
    }
    ctx.pre_eval_metrics = metrics
    logger.info("PRE_EVAL_4B complete: %s", metrics)

    return StageResult(ok=True, message=f"avg_f1={avg_f1:.4f}", metrics=metrics)


def stage_filter_delta_4b(ctx: PipelineContext) -> StageResult:
    """Write train_filtered.jsonl with GAP samples only."""
    logger.info("═══ FILTER_DELTA_4B: Filtering curriculum for gaps ═══")

    sys.path.insert(0, str(Path(__file__).parent))
    from pre_eval_curriculum import load_curriculum, query_model, token_f1

    samples = load_curriculum(ctx.curriculum_path)
    gap_samples = []

    for sample in samples:
        instruction = sample.get("instruction", "")
        expected = sample.get("output", "")
        try:
            predicted = query_model(CORE_EVAL_ENDPOINT, instruction, max_tokens=256, timeout=30)
            f1 = token_f1(predicted, expected)
            if f1 <= ctx.threshold:
                sample["_pre_eval_f1"] = round(f1, 4)
                gap_samples.append(sample)
        except Exception:
            # Include errored samples as gaps (model couldn't answer)
            sample["_pre_eval_f1"] = 0.0
            gap_samples.append(sample)

    # Write filtered JSONL
    os.makedirs(os.path.dirname(ctx.filtered_path), exist_ok=True)
    with open(ctx.filtered_path, "w") as f:
        for s in gap_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    ctx.delta_count = len(gap_samples)
    logger.info("Wrote %d gap samples to %s", len(gap_samples), ctx.filtered_path)

    if len(gap_samples) == 0:
        return StageResult(
            ok=True,
            message="No gaps found — model already knows everything!",
            metrics={"delta_count": 0},
        )

    return StageResult(
        ok=True,
        message=f"{len(gap_samples)} gap samples",
        metrics={"delta_count": len(gap_samples), "output_path": ctx.filtered_path},
    )


def stage_gpu_acquire(ctx: PipelineContext) -> StageResult:
    """Stop gaia-prime via orchestrator handoff, acquire GPU for study."""
    logger.info("═══ GPU_ACQUIRE: Requesting GPU handoff prime→study ═══")

    try:
        result = http_post(f"{ORCHESTRATOR}/handoff/prime-to-study", timeout=120)
        logger.info("Handoff result: %s", result)
        return StageResult(ok=True, message="GPU acquired for study")
    except Exception as e:
        logger.error("GPU acquire failed: %s", e)
        return StageResult(ok=False, message=str(e))


def stage_train_4b(ctx: PipelineContext) -> StageResult:
    """QLoRA training on the 4B model using filtered delta.

    Uses the previously merged model if it exists (incremental baking),
    otherwise falls back to the pristine bf16 base. This means each
    pipeline run compounds on prior identity training — the adapter
    delta stays small because we only train on gaps the model hasn't
    learned yet.
    """
    # Prefer merged (already-baked) weights for incremental training
    base_4b = MERGED_4B if Path(MERGED_4B).exists() else BASE_4B
    logger.info("═══ TRAIN_4B: QLoRA on %s ═══", base_4b)

    if not Path(ctx.filtered_path).exists():
        return StageResult(ok=False, message=f"Filtered data not found: {ctx.filtered_path}")

    if ctx.delta_count == 0:
        return StageResult(ok=True, message="No gaps to train on — skipping")

    # Create adapter output path
    timestamp = datetime.now().strftime("%Y%m%d")
    adapter_path = f"{ADAPTER_DIR}/self-model-4b-{timestamp}"
    os.makedirs(adapter_path, exist_ok=True)

    try:
        from gaia_study.qlora_trainer import QLoRATrainer, TrainingConfig

        config = TrainingConfig(
            base_model_path=base_4b,
            dataset_path=ctx.filtered_path,
            output_dir=adapter_path,
            lora_rank=int(os.environ.get("LORA_RANK", DEFAULT_LORA_RANK)),
            lora_alpha=int(os.environ.get("LORA_ALPHA", DEFAULT_LORA_ALPHA)),
            num_epochs=int(os.environ.get("TRAIN_EPOCHS", DEFAULT_TRAIN_EPOCHS)),
            batch_size=int(os.environ.get("TRAIN_BATCH", DEFAULT_TRAIN_BATCH)),
            learning_rate=float(os.environ.get("TRAIN_LR", DEFAULT_LR)),
        )

        trainer = QLoRATrainer(config)
        result = trainer.train()

        ctx.adapter_4b_path = adapter_path
        ctx.final_loss_4b = result.get("final_loss")

        logger.info("TRAIN_4B complete: adapter=%s loss=%.4f",
                     adapter_path, ctx.final_loss_4b or -1)

        return StageResult(
            ok=True,
            message=f"Training complete, loss={ctx.final_loss_4b}",
            metrics={"adapter_path": adapter_path, "final_loss": ctx.final_loss_4b},
        )
    except Exception as e:
        logger.exception("TRAIN_4B failed")
        return StageResult(ok=False, message=str(e))


def stage_merge_4b(ctx: PipelineContext) -> StageResult:
    """Merge LoRA adapter into the same base used for training.

    If training ran on the previously merged model (incremental baking),
    we merge the adapter back into that same model. The adapter weights
    are relative to whatever base was used for training.
    """
    # Must merge into the same base that was trained on
    base_4b = MERGED_4B if Path(MERGED_4B).exists() else BASE_4B
    logger.info("═══ MERGE_4B: Merging adapter into %s ═══", base_4b)

    if not ctx.adapter_4b_path:
        return StageResult(ok=False, message="No 4B adapter path set — was TRAIN_4B skipped?")

    # Handle previous merged model before overwriting.
    # If we trained on merged weights, we need to preserve a copy for
    # merge_adapter to read from while writing to the canonical path.
    archive_dest = None
    if Path(MERGED_4B).exists():
        if ctx.backup:
            archive_name = f"Qwen3.5-4B-Abliterated-merged.{int(time.time())}"
            os.makedirs(BAKED_DIR, exist_ok=True)
            archive_dest = os.path.join(BAKED_DIR, archive_name)
            logger.info("Backing up merged model to %s", archive_dest)
            shutil.copytree(MERGED_4B, archive_dest)
            if base_4b == MERGED_4B:
                base_4b = archive_dest
            shutil.rmtree(MERGED_4B)
        else:
            # No backup — if training base was merged, copy to temp for merge input
            if base_4b == MERGED_4B:
                tmp_base = f"{MERGED_4B}.tmp_merge_input"
                shutil.move(MERGED_4B, tmp_base)
                base_4b = tmp_base
            else:
                shutil.rmtree(MERGED_4B)

    try:
        from gaia_study.merge_and_requantize import merge_adapter
        output = merge_adapter(base_4b, ctx.adapter_4b_path, MERGED_4B)
        ctx.merged_4b_path = output
        logger.info("MERGE_4B complete: %s", output)

        # Clean up temp merge input if we created one (no-backup path)
        tmp_base = f"{MERGED_4B}.tmp_merge_input"
        if Path(tmp_base).exists():
            shutil.rmtree(tmp_base)

        return StageResult(ok=True, message=f"Merged to {output}",
                           metrics={"backup": archive_dest})
    except Exception as e:
        logger.exception("MERGE_4B failed")
        return StageResult(ok=False, message=str(e))


def stage_gguf_core(ctx: PipelineContext) -> StageResult:
    """Convert merged 4B → GGUF Q4_K_M for Core CPU inference."""
    logger.info("═══ GGUF_CORE: Converting merged 4B → Q4_K_M ═══")

    # Archive previous GGUF
    if Path(GGUF_CORE).exists():
        archive_name = f"{GGUF_CORE}.{int(time.time())}"
        os.makedirs(BAKED_DIR, exist_ok=True)
        archive_dest = os.path.join(BAKED_DIR, os.path.basename(archive_name))
        logger.info("Archiving previous GGUF to %s", archive_dest)
        shutil.move(GGUF_CORE, archive_dest)

    try:
        from gaia_study.merge_and_requantize import convert_to_gguf
        # convert_to_gguf writes to output_dir with name derived from model dir name
        # The merged model dir is "Qwen3.5-4B-Abliterated-merged", so GGUF will be
        # Qwen3.5-4B-Abliterated-merged-Q4_K_M.gguf. We need to rename it.
        output_dir = os.path.dirname(GGUF_CORE)
        ok = convert_to_gguf(ctx.merged_4b_path, output_dir, "Q4_K_M")
        if not ok:
            return StageResult(ok=False, message="GGUF conversion failed")

        # Rename to expected filename if different
        merged_name = os.path.basename(ctx.merged_4b_path)
        generated_gguf = os.path.join(output_dir, f"{merged_name}-Q4_K_M.gguf")
        if Path(generated_gguf).exists() and generated_gguf != GGUF_CORE:
            shutil.move(generated_gguf, GGUF_CORE)
            logger.info("Renamed %s → %s", generated_gguf, GGUF_CORE)

        size_gb = Path(GGUF_CORE).stat().st_size / (1024 ** 3) if Path(GGUF_CORE).exists() else 0
        logger.info("GGUF_CORE complete: %s (%.2f GB)", GGUF_CORE, size_gb)
        return StageResult(ok=True, message=f"GGUF at {GGUF_CORE}", metrics={"size_gb": round(size_gb, 2)})
    except Exception as e:
        logger.exception("GGUF_CORE failed")
        return StageResult(ok=False, message=str(e))


def stage_deploy_prime(ctx: PipelineContext) -> StageResult:
    """Sync merged model to warm pool and restart gaia-prime."""
    logger.info("═══ DEPLOY_PRIME: Syncing to warm pool + restarting gaia-prime ═══")

    try:
        # Sync merged model to warm pool
        model_name = os.path.basename(ctx.merged_4b_path)
        result = http_post(
            f"{ORCHESTRATOR}/warm-pool/sync",
            {"model": model_name},
            timeout=300,
        )
        logger.info("Warm pool sync: %s", result)

        # Return GPU to prime (handoff study→prime, which restarts gaia-prime)
        result = http_post(f"{ORCHESTRATOR}/handoff/study-to-prime", timeout=120)
        logger.info("Handoff study→prime: %s", result)

        # Wait for gaia-prime to be healthy
        logger.info("Waiting for gaia-prime health...")
        if not wait_for_health("http://gaia-prime:7777/health", timeout=180):
            return StageResult(ok=False, message="gaia-prime failed to become healthy")

        logger.info("DEPLOY_PRIME complete: gaia-prime healthy with new model")
        return StageResult(ok=True, message="gaia-prime restarted with new model")
    except Exception as e:
        logger.exception("DEPLOY_PRIME failed")
        return StageResult(ok=False, message=str(e))


def stage_reload_core(ctx: PipelineContext) -> StageResult:
    """Hot-reload gaia-core's embedded llama-server with new GGUF."""
    logger.info("═══ RELOAD_CORE: Hot-reloading Core CPU model ═══")

    try:
        # Release current model
        logger.info("Releasing current Core model...")
        result = http_post(f"{CORE_CPU_ENDPOINT}/model/release")
        logger.info("Release result: %s", result)

        # Small delay for cleanup
        time.sleep(2)

        # Reload with the (possibly updated) GGUF path
        logger.info("Reloading Core model: %s", GGUF_CORE)
        result = http_post(
            f"{CORE_CPU_ENDPOINT}/model/reload",
            {"model_path": GGUF_CORE},
            timeout=180,
        )
        logger.info("Reload result: %s", result)

        if not result.get("ok"):
            return StageResult(ok=False, message=f"Reload failed: {result}")

        return StageResult(ok=True, message="Core model reloaded", metrics=result)
    except Exception as e:
        logger.exception("RELOAD_CORE failed")
        return StageResult(ok=False, message=str(e))


def stage_train_nano(ctx: PipelineContext) -> StageResult:
    """QLoRA on 0.8B Nano with the same filtered delta.

    Like TRAIN_4B, uses previously merged Nano weights if they exist
    for incremental identity baking.
    """
    merged_nano = f"{BASE_08B}-merged"
    base_nano = merged_nano if Path(merged_nano).exists() else BASE_08B
    logger.info("═══ TRAIN_NANO: QLoRA on %s ═══", base_nano)

    if not Path(ctx.filtered_path).exists():
        return StageResult(ok=False, message=f"Filtered data not found: {ctx.filtered_path}")

    if ctx.delta_count == 0:
        return StageResult(ok=True, message="No gaps to train on — skipping")

    # Re-acquire GPU if needed (Prime may have taken it back in DEPLOY_PRIME)
    try:
        result = http_post(f"{ORCHESTRATOR}/handoff/prime-to-study", timeout=120)
        logger.info("GPU re-acquire for Nano: %s", result)
    except Exception as e:
        logger.warning("GPU re-acquire failed (may already own it): %s", e)

    timestamp = datetime.now().strftime("%Y%m%d")
    adapter_path = f"{ADAPTER_DIR}/self-model-nano-{timestamp}"
    os.makedirs(adapter_path, exist_ok=True)

    try:
        from gaia_study.qlora_trainer import QLoRATrainer, TrainingConfig

        config = TrainingConfig(
            base_model_path=base_nano,
            dataset_path=ctx.filtered_path,
            output_dir=adapter_path,
            lora_rank=int(os.environ.get("LORA_RANK", DEFAULT_LORA_RANK)),
            lora_alpha=int(os.environ.get("LORA_ALPHA", DEFAULT_LORA_ALPHA)),
            num_epochs=int(os.environ.get("TRAIN_EPOCHS", DEFAULT_TRAIN_EPOCHS)),
            batch_size=int(os.environ.get("TRAIN_BATCH", DEFAULT_TRAIN_BATCH)),
            learning_rate=float(os.environ.get("TRAIN_LR", DEFAULT_LR)),
        )

        trainer = QLoRATrainer(config)
        result = trainer.train()

        ctx.adapter_nano_path = adapter_path
        ctx.final_loss_nano = result.get("final_loss")

        logger.info("TRAIN_NANO complete: adapter=%s loss=%.4f",
                     adapter_path, ctx.final_loss_nano or -1)

        return StageResult(
            ok=True,
            message=f"Training complete, loss={ctx.final_loss_nano}",
            metrics={"adapter_path": adapter_path, "final_loss": ctx.final_loss_nano},
        )
    except Exception as e:
        logger.exception("TRAIN_NANO failed")
        return StageResult(ok=False, message=str(e))


def stage_merge_nano(ctx: PipelineContext) -> StageResult:
    """Merge LoRA adapter into the same Nano base used for training."""
    merged_nano = f"{BASE_08B}-merged"
    base_nano = merged_nano if Path(merged_nano).exists() else BASE_08B
    logger.info("═══ MERGE_NANO: Merging adapter into %s ═══", base_nano)

    if not ctx.adapter_nano_path:
        return StageResult(ok=False, message="No Nano adapter path — was TRAIN_NANO skipped?")

    # Handle previous merged Nano model
    archive_dest = None
    if Path(merged_nano).exists():
        if ctx.backup:
            archive_name = f"Qwen3.5-0.8B-Abliterated-merged.{int(time.time())}"
            os.makedirs(BAKED_DIR, exist_ok=True)
            archive_dest = os.path.join(BAKED_DIR, archive_name)
            logger.info("Backing up Nano merged model to %s", archive_dest)
            shutil.copytree(merged_nano, archive_dest)
            if base_nano == merged_nano:
                base_nano = archive_dest
            shutil.rmtree(merged_nano)
        else:
            if base_nano == merged_nano:
                tmp_base = f"{merged_nano}.tmp_merge_input"
                shutil.move(merged_nano, tmp_base)
                base_nano = tmp_base
            else:
                shutil.rmtree(merged_nano)

    try:
        from gaia_study.merge_and_requantize import merge_adapter
        output = merge_adapter(base_nano, ctx.adapter_nano_path, merged_nano)
        logger.info("MERGE_NANO complete: %s", output)

        # Clean up temp merge input
        tmp_base = f"{merged_nano}.tmp_merge_input"
        if Path(tmp_base).exists():
            shutil.rmtree(tmp_base)

        return StageResult(ok=True, message=f"Merged to {output}",
                           metrics={"merged_path": output, "backup": archive_dest})
    except Exception as e:
        logger.exception("MERGE_NANO failed")
        return StageResult(ok=False, message=str(e))


def stage_gguf_nano(ctx: PipelineContext) -> StageResult:
    """Convert merged 0.8B → GGUF Q8_0 for Nano."""
    logger.info("═══ GGUF_NANO: Converting merged 0.8B → Q8_0 ═══")

    merged_nano = f"{BASE_08B}-merged"

    # Archive previous GGUF
    if Path(GGUF_NANO).exists():
        archive_dest = os.path.join(BAKED_DIR, f"{os.path.basename(GGUF_NANO)}.{int(time.time())}")
        os.makedirs(BAKED_DIR, exist_ok=True)
        shutil.move(GGUF_NANO, archive_dest)

    try:
        from gaia_study.merge_and_requantize import convert_to_gguf
        output_dir = os.path.dirname(GGUF_NANO)
        ok = convert_to_gguf(merged_nano, output_dir, "Q8_0")
        if not ok:
            return StageResult(ok=False, message="GGUF conversion failed")

        # Rename if needed
        merged_name = os.path.basename(merged_nano)
        generated_gguf = os.path.join(output_dir, f"{merged_name}-Q8_0.gguf")
        if Path(generated_gguf).exists() and generated_gguf != GGUF_NANO:
            shutil.move(generated_gguf, GGUF_NANO)

        size_gb = Path(GGUF_NANO).stat().st_size / (1024 ** 3) if Path(GGUF_NANO).exists() else 0
        logger.info("GGUF_NANO complete: %s (%.2f GB)", GGUF_NANO, size_gb)
        return StageResult(ok=True, message=f"GGUF at {GGUF_NANO}", metrics={"size_gb": round(size_gb, 2)})
    except Exception as e:
        logger.exception("GGUF_NANO failed")
        return StageResult(ok=False, message=str(e))


def stage_deploy_nano(ctx: PipelineContext) -> StageResult:
    """Restart gaia-nano with the new GGUF."""
    logger.info("═══ DEPLOY_NANO: Restarting gaia-nano ═══")

    # Release GPU back to prime if we still have it
    try:
        http_post(f"{ORCHESTRATOR}/handoff/study-to-prime", timeout=120)
    except Exception as e:
        logger.warning("GPU release for Nano deploy failed (may not own it): %s", e)

    try:
        result = http_post(f"{ORCHESTRATOR}/containers/gaia-nano/restart", timeout=60)
        logger.info("Nano restart: %s", result)

        # Wait for health
        logger.info("Waiting for gaia-nano health...")
        if not wait_for_health("http://gaia-nano:8080/health", timeout=60):
            return StageResult(ok=False, message="gaia-nano failed to become healthy")

        return StageResult(ok=True, message="gaia-nano restarted with new GGUF")
    except Exception as e:
        logger.exception("DEPLOY_NANO failed")
        return StageResult(ok=False, message=str(e))


def stage_post_eval(ctx: PipelineContext) -> StageResult:
    """Re-score curriculum against Core+Nano, measure F1 improvement."""
    logger.info("═══ POST_EVAL: Re-evaluating against Core ═══")

    sys.path.insert(0, str(Path(__file__).parent))
    from pre_eval_curriculum import load_curriculum, query_model, token_f1

    samples = load_curriculum(ctx.curriculum_path)
    total_f1 = 0.0
    scored = 0

    for i, sample in enumerate(samples):
        instruction = sample.get("instruction", "")
        expected = sample.get("output", "")
        try:
            predicted = query_model(CORE_EVAL_ENDPOINT, instruction, max_tokens=256, timeout=30)
            f1 = token_f1(predicted, expected)
            total_f1 += f1
            scored += 1
        except Exception:
            pass

        if (i + 1) % 50 == 0:
            logger.info("  [%d/%d] scored=%d", i + 1, len(samples), scored)

    avg_f1 = total_f1 / scored if scored > 0 else 0.0
    pre_f1 = ctx.pre_eval_metrics.get("avg_f1", 0.0)
    improvement = avg_f1 - pre_f1

    metrics = {
        "total": len(samples),
        "scored": scored,
        "avg_f1": round(avg_f1, 4),
        "pre_avg_f1": round(pre_f1, 4),
        "improvement": round(improvement, 4),
    }
    ctx.post_eval_metrics = metrics

    logger.info("POST_EVAL complete: pre=%.4f → post=%.4f (Δ=%.4f)", pre_f1, avg_f1, improvement)

    return StageResult(ok=True, message=f"F1: {pre_f1:.4f} → {avg_f1:.4f} (Δ={improvement:+.4f})", metrics=metrics)


def stage_cognitive_smoke(ctx: PipelineContext) -> StageResult:
    """Run gaia-doctor's cognitive test battery as a post-training gate."""
    logger.info("═══ COGNITIVE_SMOKE: Running cognitive test battery ═══")

    if ctx.skip_smoke:
        return StageResult(ok=True, message="Skipped (--skip-smoke)")

    try:
        # Trigger battery run via doctor
        result = http_post(f"{DOCTOR_ENDPOINT}/cognitive/run", timeout=10)
        logger.info("Battery triggered: %s", result)

        # Poll for completion (max 10 minutes)
        deadline = time.time() + 600
        while time.time() < deadline:
            time.sleep(10)
            try:
                status = http_get(f"{DOCTOR_ENDPOINT}/cognitive/status")
                if not status.get("running", True):
                    break
            except Exception:
                pass

        # Fetch results
        results = http_get(f"{DOCTOR_ENDPOINT}/cognitive/results")
        summary = results.get("summary", {})
        pass_rate = summary.get("pass_rate", 0.0)
        ctx.smoke_pass_rate = pass_rate

        metrics = {
            "pass_rate": pass_rate,
            "passed": summary.get("passed", 0),
            "failed": summary.get("failed", 0),
            "total": summary.get("total", 0),
            "threshold": ctx.smoke_threshold,
        }

        if pass_rate >= ctx.smoke_threshold:
            logger.info("COGNITIVE_SMOKE passed: %.1f%% >= %.1f%%",
                        pass_rate * 100, ctx.smoke_threshold * 100)
            return StageResult(ok=True, message=f"pass_rate={pass_rate:.2%}", metrics=metrics)
        else:
            logger.warning("COGNITIVE_SMOKE failed: %.1f%% < %.1f%%",
                           pass_rate * 100, ctx.smoke_threshold * 100)
            return StageResult(ok=False,
                             message=f"pass_rate {pass_rate:.2%} < threshold {ctx.smoke_threshold:.2%}",
                             metrics=metrics)

    except Exception as e:
        logger.exception("COGNITIVE_SMOKE failed")
        return StageResult(ok=False, message=str(e))


# ── Stage Registry ───────────────────────────────────────────────────────────

STAGE_FUNCTIONS: dict[str, Callable[[PipelineContext], StageResult]] = {
    "BUILD_CURRICULUM": stage_build_curriculum,
    "PRE_EVAL_4B": stage_pre_eval_4b,
    "FILTER_DELTA_4B": stage_filter_delta_4b,
    "GPU_ACQUIRE": stage_gpu_acquire,
    "TRAIN_4B": stage_train_4b,
    "MERGE_4B": stage_merge_4b,
    "GGUF_CORE": stage_gguf_core,
    "DEPLOY_PRIME": stage_deploy_prime,
    "RELOAD_CORE": stage_reload_core,
    "TRAIN_NANO": stage_train_nano,
    "MERGE_NANO": stage_merge_nano,
    "GGUF_NANO": stage_gguf_nano,
    "DEPLOY_NANO": stage_deploy_nano,
    "POST_EVAL": stage_post_eval,
    "COGNITIVE_SMOKE": stage_cognitive_smoke,
}


# ── Pipeline Runner ──────────────────────────────────────────────────────────

def should_pause(stage: str, pause_after: str | None) -> bool:
    """Check if we should pause after this stage."""
    if Path(PAUSE_FLAG).exists():
        Path(PAUSE_FLAG).unlink(missing_ok=True)
        return True
    if pause_after and stage == pause_after:
        return True
    return False


def run_pipeline(args: argparse.Namespace):
    """Execute the self-awareness training pipeline."""
    # Load or init state
    state = load_state() if args.resume else {}

    ctx = PipelineContext(
        skip_nano=args.skip_nano,
        dry_run=args.dry_run,
        threshold=args.threshold,
        backup=not args.no_backup,
        skip_curriculum_build=args.skip_curriculum_build,
        skip_smoke=args.skip_smoke,
        smoke_threshold=args.smoke_threshold,
    )

    if not state:
        ctx.pipeline_id = f"sa-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        ctx.started_at = datetime.now(timezone.utc).isoformat()
        state = init_state(ctx)
        save_state(state)
        logger.info("New pipeline: %s", ctx.pipeline_id)
    else:
        ctx.pipeline_id = state.get("pipeline_id", "unknown")
        ctx.started_at = state.get("started_at", "")
        # Restore context from state
        pre_eval = state.get("pre_eval", {})
        ctx.pre_eval_metrics = {"avg_f1": pre_eval.get("core_avg_f1", 0.0)}
        adapters = state.get("adapters", {})
        ctx.adapter_4b_path = adapters.get("4b", {}).get("path", "")
        ctx.adapter_nano_path = adapters.get("nano", {}).get("path", "")
        ctx.delta_count = state.get("stages", {}).get("FILTER_DELTA_4B", {}).get("delta_count", 0)
        logger.info("Resuming pipeline: %s", ctx.pipeline_id)

    # Determine starting stage
    stages_to_run = list(STAGES)
    if args.stage:
        if args.stage not in STAGES:
            logger.error("Unknown stage: %s. Valid: %s", args.stage, STAGES)
            sys.exit(1)
        idx = STAGES.index(args.stage)
        stages_to_run = STAGES[idx:]
    elif args.resume:
        # Skip completed stages
        for stage in STAGES:
            stage_state = state.get("stages", {}).get(stage, {})
            if stage_state.get("status") == "completed":
                stages_to_run.remove(stage)

    # Skip nano stages if requested
    if ctx.skip_nano:
        stages_to_run = [s for s in stages_to_run if s not in NANO_STAGES]

    logger.info("Stages to run: %s", stages_to_run)

    if ctx.dry_run:
        logger.info("DRY RUN — not executing any stages")
        for stage in stages_to_run:
            logger.info("  [DRY] %s", stage)
        return

    # Execute stages
    for stage in stages_to_run:
        logger.info("")
        logger.info("▶ Starting stage: %s", stage)
        mark_stage(state, stage, "running")

        func = STAGE_FUNCTIONS.get(stage)
        if func is None:
            logger.error("No implementation for stage %s", stage)
            mark_stage(state, stage, "error", error="No implementation")
            break

        t0 = time.time()
        result = func(ctx)
        elapsed = time.time() - t0

        if result.ok:
            mark_stage(state, stage, "completed",
                       elapsed_seconds=round(elapsed, 1),
                       metrics=result.metrics)

            # Update alignment status
            if stage == "BUILD_CURRICULUM":
                state["alignment_status"] = "TRAINING"
            elif stage == "FILTER_DELTA_4B" and ctx.delta_count == 0:
                state["alignment_status"] = "SELF_ALIGNED"
                logger.info("★ Zero gaps found — model is SELF-ALIGNED")
            elif stage == "COGNITIVE_SMOKE":
                smoke_rate = ctx.smoke_pass_rate or 0.0
                if ctx.delta_count == 0 and smoke_rate >= 1.0:
                    state["alignment_status"] = "SELF_ALIGNED"
                    logger.info("★ SELF-ALIGNED: zero gaps + perfect cognitive smoke")
                elif smoke_rate >= ctx.smoke_threshold:
                    state["alignment_status"] = "ALIGNED"
                    logger.info("✓ ALIGNED: cognitive smoke passed (%.1f%%)", smoke_rate * 100)

            # Update state with stage-specific data
            if stage == "PRE_EVAL_4B":
                state["pre_eval"]["core_avg_f1"] = result.metrics.get("avg_f1")
            elif stage == "FILTER_DELTA_4B":
                state["stages"]["FILTER_DELTA_4B"]["delta_count"] = ctx.delta_count
                state["stages"]["FILTER_DELTA_4B"]["output_path"] = ctx.filtered_path
            elif stage == "TRAIN_4B":
                state["adapters"]["4b"]["path"] = ctx.adapter_4b_path
                state["adapters"]["4b"]["final_loss"] = ctx.final_loss_4b
            elif stage == "TRAIN_NANO":
                state["adapters"]["nano"]["path"] = ctx.adapter_nano_path
                state["adapters"]["nano"]["final_loss"] = ctx.final_loss_nano
            elif stage == "POST_EVAL":
                state["post_eval"]["core_avg_f1"] = result.metrics.get("avg_f1")
            elif stage == "COGNITIVE_SMOKE":
                state["cognitive_smoke"] = {
                    "pass_rate": ctx.smoke_pass_rate,
                    "threshold": ctx.smoke_threshold,
                }

            save_state(state)
            logger.info("✓ %s completed in %.1fs: %s", stage, elapsed, result.message)
        else:
            mark_stage(state, stage, "failed",
                       elapsed_seconds=round(elapsed, 1),
                       error=result.message)
            logger.error("✗ %s failed after %.1fs: %s", stage, elapsed, result.message)
            logger.error("Pipeline halted. Fix the issue and --resume or --stage %s", stage)
            sys.exit(1)

        # Check pause conditions
        if should_pause(stage, args.pause_after):
            logger.info("⏸ Paused after %s. Resume with --resume or --stage <next>", stage)
            return

    # Pipeline complete
    logger.info("")
    logger.info("═" * 60)
    logger.info("PIPELINE COMPLETE: %s", ctx.pipeline_id)
    logger.info("═" * 60)
    if ctx.pre_eval_metrics and ctx.post_eval_metrics:
        pre = ctx.pre_eval_metrics.get("avg_f1", 0)
        post = ctx.post_eval_metrics.get("avg_f1", 0)
        logger.info("  F1: %.4f → %.4f (Δ=%+.4f)", pre, post, post - pre)
    logger.info("  State: %s", STATE_FILE)


def main():
    parser = argparse.ArgumentParser(
        description="GAIA Self-Awareness Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last completed stage")
    parser.add_argument("--stage", type=str, default=None,
                        help="Start from a specific stage (e.g., TRAIN_4B)")
    parser.add_argument("--pause-after", type=str, default=None,
                        help="Pause after this stage completes")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print stages without executing")
    parser.add_argument("--skip-nano", action="store_true",
                        help="Skip Nano (0.8B) training stages")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"F1 threshold for gap detection (default: {DEFAULT_THRESHOLD})")
    parser.add_argument("--no-backup", action="store_true",
                        help="Skip backing up previous merged models before overwriting")
    parser.add_argument("--skip-curriculum-build", action="store_true",
                        help="Use existing train.jsonl instead of regenerating")
    parser.add_argument("--skip-smoke", action="store_true",
                        help="Skip cognitive test battery gate")
    parser.add_argument("--smoke-threshold", type=float, default=0.85,
                        help="Minimum pass rate for cognitive smoke gate (default: 0.85)")

    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
