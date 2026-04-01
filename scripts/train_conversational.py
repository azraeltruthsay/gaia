#!/usr/bin/env python3
"""
Train Conversational Adapter with SAE Monitoring.

Three-subprocess architecture for clean VRAM management:
  1. SAE pre-scan — baseline feature map before training
  2. QLoRA training — train the conversational adapter
  3. SAE post-scan — feature map after training (with adapter loaded)

Then: CPU-only misfiring analysis comparing pre/post atlases.

Usage (inside gaia-study container):
    python /gaia/GAIA_Project/scripts/train_conversational.py

Or from host:
    docker exec gaia-study python /gaia/GAIA_Project/scripts/train_conversational.py

Options:
    --skip-sae        Skip SAE scans, only run QLoRA training
    --skip-training   Skip training, only run SAE scans (requires existing adapter)
    --sae-only-pre    Only run pre-scan
    --sae-only-post   Only run post-scan + analysis
"""

import argparse
import json
import logging
import multiprocessing
import os
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────

BASE_MODEL = os.getenv(
    "BASE_MODEL_PATH",
    "/models/Huihui-Qwen3-8B-GAIA-Prime-adaptive",
)
TRAINING_DATA = "/gaia/GAIA_Project/knowledge/curricula/conversational/train.json"
PROBES_PATH = "/gaia/GAIA_Project/knowledge/curricula/conversational/sae_probes.json"
ADAPTER_DIR = "/models/lora_adapters/tier1_global/conversational_v1"
ADAPTER_NAME = "conversational_v1"
PRE_ATLAS_DIR = "/shared/atlas/conversational/pre"
POST_ATLAS_DIR = "/shared/atlas/conversational/post"
REPORT_PATH = "/shared/atlas/conversational/refinement_report.json"

# ── SAE Config ─────────────────────────────────────────────────────────────

SAE_LAYERS = [4, 8, 12, 16, 20, 24, 28]  # Sample across transformer depth
SAE_NUM_FEATURES_MULT = 2  # Features = hidden_size * this
SAE_EPOCHS = 50
SAE_ANALYSIS_LAYER = 24  # Primary analysis layer (decision/output)

# ── Training Config ────────────────────────────────────────────────────────

TRAINING_CONFIG = {
    "base_model_path": BASE_MODEL,
    "adapter_dir": ADAPTER_DIR,
    "adapter_name": ADAPTER_NAME,
    "load_in_4bit": True,
    "bnb_4bit_compute_dtype": "bfloat16",
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "batch_size": 1,
    "gradient_accumulation_steps": 4,
    "learning_rate": 1.5e-4,  # Lower than code — subtle style nudge
    "max_steps": 300,
    "warmup_steps": 15,
    "target_loss": 0.15,      # Higher — style is softer than format precision
    "convergence_patience": 5,
    "num_train_epochs": None,
    "max_training_time": 1200,  # 20 min max
    "resume_from": None,
    "samples": [],
}


# ═══════════════════════════════════════════════════════════════════════════
# Subprocess 1: SAE Pre-Scan
# ═══════════════════════════════════════════════════════════════════════════

def run_sae_scan(config: dict) -> None:
    """
    Run SAE scan in isolated subprocess.
    All torch/CUDA imports happen here — parent never touches CUDA.
    """
    import torch
    import gc

    # Add engine and shared libs to path
    sys.path.insert(0, "/gaia/GAIA_Project/gaia-engine")
    sys.path.insert(0, "/shared/pylibs")

    sub_logger = logging.getLogger("sae-scan")
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] sae-scan: %(message)s")

    atlas_dir = config["atlas_dir"]
    base_model_path = config["base_model_path"]
    adapter_path = config.get("adapter_path")  # None for pre-scan
    probes = config["probes"]
    layers = config["layers"]
    analysis_layer = config["analysis_layer"]
    phase = config["phase"]

    sub_logger.info("SAE %s-scan starting (PID %d)", phase, os.getpid())

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        # Load model to CPU first, then BnB NF4 quantize on GPU
        # (bf16 is ~16GB, won't fit directly on 16GB GPU)
        sub_logger.info("Loading model from %s (CPU first, then NF4)...", base_model_path)
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=bnb_config,
            device_map={"": "cpu"},
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        sub_logger.info("Model loaded to CPU, dispatching to GPU...")
        from accelerate import dispatch_model
        model = dispatch_model(model, device_map={"": 0})
        gc.collect()
        torch.cuda.empty_cache()

        # Load adapter for post-scan
        if adapter_path and os.path.isdir(adapter_path):
            sub_logger.info("Loading adapter from %s...", adapter_path)
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_path)
            model.eval()
            sub_logger.info("Adapter loaded for post-scan")

        free, total = torch.cuda.mem_get_info(0)
        sub_logger.info("GPU: %.1fGB free / %.1fGB total", free/1024**3, total/1024**3)

        # Initialize SAE trainer
        from gaia_engine.sae_trainer import SAETrainer
        trainer = SAETrainer(model, tokenizer, device="cuda")

        # Extract probe prompts
        probe_prompts = [p["prompt"] for p in probes]

        # Phase 1: Record activations
        sub_logger.info("Recording activations for %d probes at %d layers...",
                        len(probe_prompts), len(layers))
        stats = trainer.record_activations(probe_prompts, layers)
        sub_logger.info("Activations recorded: %s", stats)

        # Phase 2: Train SAE
        hidden_size = model.config.hidden_size
        if hasattr(model.config, 'text_config'):
            hidden_size = model.config.text_config.hidden_size
        num_features = hidden_size * SAE_NUM_FEATURES_MULT

        sub_logger.info("Training SAE: %d features per layer...", num_features)
        train_results = trainer.train_sae(
            layers=layers,
            num_features=num_features,
            sparsity_weight=0.01,
            lr=1e-3,
            epochs=SAE_EPOCHS,
            batch_size=256,
        )
        sub_logger.info("SAE trained: %s", train_results)

        # Phase 3: Save atlas
        os.makedirs(atlas_dir, exist_ok=True)
        trainer.save_atlas(atlas_dir)
        sub_logger.info("Atlas saved to %s", atlas_dir)

        # Phase 4: Analyze each probe
        analyses = []
        for probe in probes:
            try:
                analysis = trainer.analyze_prompt(
                    probe["prompt"], analysis_layer, top_k=20
                )
                analysis["category"] = probe["category"]
                analysis["type"] = probe["type"]
                analysis["expected_shift"] = probe.get("expected_shift", "")
                analyses.append(analysis)
            except Exception as e:
                sub_logger.warning("Failed to analyze probe '%s': %s",
                                   probe["prompt"][:40], e)

        # Save analyses
        analyses_path = os.path.join(atlas_dir, "probe_analyses.json")
        with open(analyses_path, "w") as f:
            json.dump(analyses, f, indent=2)
        sub_logger.info("Saved %d probe analyses to %s", len(analyses), analyses_path)

        # Cleanup
        del model, trainer
        gc.collect()
        torch.cuda.empty_cache()

        sub_logger.info("SAE %s-scan complete", phase)

    except Exception as e:
        sub_logger.error("SAE %s-scan failed: %s", phase, e, exc_info=True)
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════
# Subprocess 2: QLoRA Training
# ═══════════════════════════════════════════════════════════════════════════

def run_qlora_training(config: dict) -> None:
    """Standard QLoRA training via subprocess isolation."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] training: %(message)s")
    sub_logger = logging.getLogger("training")

    sub_logger.info("QLoRA training starting (PID %d)", os.getpid())

    sys.path.insert(0, "/app")
    sys.path.insert(0, "/gaia-common")
    from gaia_study.training_subprocess import run_training
    run_training(config)


# ═══════════════════════════════════════════════════════════════════════════
# Main Process: Misfiring Analysis (CPU only)
# ═══════════════════════════════════════════════════════════════════════════

def analyze_misfiring(pre_dir: str, post_dir: str, probes: list) -> dict:
    """
    Compare pre/post SAE atlases to detect misfiring.
    CPU only — no VRAM needed.
    """
    import torch

    logger.info("Analyzing misfiring between %s and %s", pre_dir, post_dir)

    pre_analyses_path = os.path.join(pre_dir, "probe_analyses.json")
    post_analyses_path = os.path.join(post_dir, "probe_analyses.json")

    if not os.path.exists(pre_analyses_path) or not os.path.exists(post_analyses_path):
        logger.warning("Missing probe analyses — skipping misfiring detection")
        return {"error": "missing analyses"}

    with open(pre_analyses_path) as f:
        pre_analyses = json.load(f)
    with open(post_analyses_path) as f:
        post_analyses = json.load(f)

    # Build lookup by prompt
    pre_by_prompt = {a["prompt"]: a for a in pre_analyses}
    post_by_prompt = {a["prompt"]: a for a in post_analyses}

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pre_atlas": pre_dir,
        "post_atlas": post_dir,
        "probes_analyzed": len(probes),
        "control_drift": [],
        "feature_bleed": [],
        "category_confusion": [],
        "healthy_shifts": [],
        "recommendations": [],
    }

    for probe in probes:
        prompt = probe["prompt"]
        category = probe["category"]
        probe_type = probe["type"]

        pre = pre_by_prompt.get(prompt[:80])  # analyze_prompt truncates to 80
        post = post_by_prompt.get(prompt[:80])

        if not pre or not post:
            continue

        pre_features = {f["index"]: f for f in pre.get("top_features", [])}
        post_features = {f["index"]: f for f in post.get("top_features", [])}

        # Compute feature delta
        pre_indices = set(pre_features.keys())
        post_indices = set(post_features.keys())
        new_features = post_indices - pre_indices
        lost_features = pre_indices - post_indices
        shared = pre_indices & post_indices

        strength_changes = {}
        for idx in shared:
            delta = post_features[idx]["strength"] - pre_features[idx]["strength"]
            if abs(delta) > 0.05:  # Significant change threshold
                strength_changes[idx] = {
                    "pre": pre_features[idx]["strength"],
                    "post": post_features[idx]["strength"],
                    "delta": delta,
                    "label": post_features[idx].get("label", f"feature_{idx}"),
                }

        total_change = len(new_features) + len(lost_features) + len(strength_changes)

        entry = {
            "prompt": prompt[:80],
            "category": category,
            "type": probe_type,
            "new_features": len(new_features),
            "lost_features": len(lost_features),
            "strength_changes": len(strength_changes),
            "total_change": total_change,
            "details": {
                "new": [
                    {"index": i, "strength": post_features[i]["strength"],
                     "label": post_features[i].get("label", "")}
                    for i in list(new_features)[:5]
                ],
                "lost": [
                    {"index": i, "strength": pre_features[i]["strength"],
                     "label": pre_features[i].get("label", "")}
                    for i in list(lost_features)[:5]
                ],
                "changed": list(strength_changes.values())[:5],
            },
        }

        # Classify the change
        if probe_type == "control":
            if total_change > 5:
                report["control_drift"].append(entry)
                report["recommendations"].append(
                    f"CONTROL DRIFT: {category} probe changed {total_change} features. "
                    f"Consider reducing training steps or adding {category} identity samples."
                )
            else:
                entry["status"] = "stable"
                report["healthy_shifts"].append(entry)
        else:
            # Conversational probe — check for appropriate changes
            if total_change == 0:
                report["recommendations"].append(
                    f"NO SHIFT: {category} probe unchanged. "
                    f"Consider adding more {category} training samples or increasing weight."
                )
            elif total_change > 10:
                report["feature_bleed"].append(entry)
                report["recommendations"].append(
                    f"FEATURE BLEED: {category} probe changed {total_change} features (too many). "
                    f"Training may be too aggressive — reduce lr or max_steps."
                )
            else:
                entry["status"] = "healthy"
                report["healthy_shifts"].append(entry)

    # Summary
    report["summary"] = {
        "control_probes_drifted": len(report["control_drift"]),
        "conversational_with_bleed": len(report["feature_bleed"]),
        "category_confused": len(report["category_confusion"]),
        "healthy_shifts": len(report["healthy_shifts"]),
        "total_recommendations": len(report["recommendations"]),
        "verdict": "PASS" if (
            len(report["control_drift"]) == 0 and
            len(report["feature_bleed"]) <= 1
        ) else "NEEDS_REVIEW",
    }

    return report


# ═══════════════════════════════════════════════════════════════════════════
# Main Orchestration
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train conversational adapter with SAE monitoring")
    parser.add_argument("--skip-sae", action="store_true", help="Skip SAE scans")
    parser.add_argument("--skip-training", action="store_true", help="Skip QLoRA training")
    parser.add_argument("--sae-only-pre", action="store_true", help="Only run pre-scan")
    parser.add_argument("--sae-only-post", action="store_true", help="Only run post-scan + analysis")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Conversational Adapter — SAE-Monitored Training")
    logger.info("=" * 60)

    # Verify paths
    if not os.path.isdir(BASE_MODEL):
        logger.error("Base model not found: %s", BASE_MODEL)
        sys.exit(1)
    if not os.path.exists(TRAINING_DATA):
        logger.error("Training data not found: %s", TRAINING_DATA)
        sys.exit(1)

    # Load probes
    with open(PROBES_PATH) as f:
        probes_data = json.load(f)
    probes = probes_data["probes"]
    logger.info("Loaded %d diagnostic probes", len(probes))

    # Load training data
    with open(TRAINING_DATA) as f:
        samples = json.load(f)
    logger.info("Loaded %d training samples", len(samples))

    ctx = multiprocessing.get_context("spawn")
    overall_start = time.time()

    # ── Subprocess 1: SAE Pre-Scan ──────────────────────────────────────
    if not args.skip_sae and not args.sae_only_post:
        logger.info("─" * 40)
        logger.info("PHASE 1: SAE Pre-Scan")
        logger.info("─" * 40)

        sae_config = {
            "atlas_dir": PRE_ATLAS_DIR,
            "base_model_path": BASE_MODEL,
            "adapter_path": None,
            "probes": probes,
            "layers": SAE_LAYERS,
            "analysis_layer": SAE_ANALYSIS_LAYER,
            "phase": "pre",
        }

        proc = ctx.Process(target=run_sae_scan, args=(sae_config,))
        proc.start()
        proc.join()

        if proc.exitcode != 0:
            logger.error("SAE pre-scan failed (exit %d)", proc.exitcode)
            sys.exit(1)
        logger.info("SAE pre-scan complete (%.0fs)", time.time() - overall_start)

    if args.sae_only_pre:
        logger.info("Pre-scan only mode — done.")
        return

    # ── Subprocess 2: QLoRA Training ────────────────────────────────────
    if not args.skip_training:
        logger.info("─" * 40)
        logger.info("PHASE 2: QLoRA Training")
        logger.info("─" * 40)

        TRAINING_CONFIG["samples"] = samples
        os.makedirs(ADAPTER_DIR, exist_ok=True)

        sys.path.insert(0, "/app")
        sys.path.insert(0, "/gaia-common")
        from gaia_study.training_subprocess import run_training, PROGRESS_FILE

        train_start = time.time()
        proc = ctx.Process(target=run_training, args=(TRAINING_CONFIG,))
        proc.start()

        # Monitor progress
        last_step = -1
        while proc.is_alive():
            time.sleep(5)
            try:
                with open(PROGRESS_FILE) as f:
                    progress = json.load(f)
                state = progress.get("state", "unknown")
                step = progress.get("step", 0)
                total = progress.get("total_steps", 0)
                loss = progress.get("loss", 0)
                if step != last_step or state != "training":
                    logger.info("[%s] step %d/%d, loss=%.4f, elapsed=%.0fs",
                                state, step, total, loss, time.time() - train_start)
                    last_step = step
            except (FileNotFoundError, json.JSONDecodeError):
                pass

        proc.join()

        if proc.exitcode != 0:
            logger.error("QLoRA training failed (exit %d)", proc.exitcode)
            sys.exit(1)

        # Read final progress
        try:
            with open(PROGRESS_FILE) as f:
                final = json.load(f)
            logger.info("Training complete: %d steps, loss=%.4f",
                        final.get("step", 0), final.get("loss", 0))
        except Exception:
            pass

        # Write adapter metadata
        metadata = {
            "name": ADAPTER_NAME,
            "version": "1.0.0",
            "display_name": "Conversational V1",
            "description": f"Conversational adapter — {len(samples)} phrase samples, SAE-monitored",
            "tier": 1,
            "pillar": "personality",
            "rank": 16,
            "alpha": 32,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
            "training": {
                "method": "qlora",
                "samples": len(samples),
                "steps": final.get("step", 0) if 'final' in dir() else 0,
                "learning_rate": 1.5e-4,
                "batch_size": 1,
                "final_loss": final.get("loss", 0) if 'final' in dir() else 0,
                "duration_seconds": time.time() - train_start,
                "sae_monitored": True,
                "source_documents": [
                    {"path": TRAINING_DATA, "samples": len(samples)},
                ],
            },
            "governance": {
                "requires_approval": True,
                "safety_checked": False,
            },
            "tags": ["conversational", "personality", "phrases", "sae-monitored"],
            "activation_triggers": [],
        }
        with open(os.path.join(ADAPTER_DIR, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Metadata written")

    # ── Subprocess 3: SAE Post-Scan ─────────────────────────────────────
    if not args.skip_sae:
        logger.info("─" * 40)
        logger.info("PHASE 3: SAE Post-Scan")
        logger.info("─" * 40)

        sae_config = {
            "atlas_dir": POST_ATLAS_DIR,
            "base_model_path": BASE_MODEL,
            "adapter_path": ADAPTER_DIR,
            "probes": probes,
            "layers": SAE_LAYERS,
            "analysis_layer": SAE_ANALYSIS_LAYER,
            "phase": "post",
        }

        proc = ctx.Process(target=run_sae_scan, args=(sae_config,))
        proc.start()
        proc.join()

        if proc.exitcode != 0:
            logger.error("SAE post-scan failed (exit %d)", proc.exitcode)
            # Don't exit — training already succeeded, report what we can

    # ── Analysis: Misfiring Detection ───────────────────────────────────
    if not args.skip_sae:
        logger.info("─" * 40)
        logger.info("PHASE 4: Misfiring Analysis")
        logger.info("─" * 40)

        report = analyze_misfiring(PRE_ATLAS_DIR, POST_ATLAS_DIR, probes)

        os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
        with open(REPORT_PATH, "w") as f:
            json.dump(report, f, indent=2)
        logger.info("Refinement report saved to %s", REPORT_PATH)

        # Print summary
        summary = report.get("summary", {})
        logger.info("─" * 40)
        logger.info("VERDICT: %s", summary.get("verdict", "UNKNOWN"))
        logger.info("  Control probes drifted: %d", summary.get("control_probes_drifted", 0))
        logger.info("  Feature bleed: %d", summary.get("conversational_with_bleed", 0))
        logger.info("  Healthy shifts: %d", summary.get("healthy_shifts", 0))
        logger.info("  Recommendations: %d", summary.get("total_recommendations", 0))

        if report.get("recommendations"):
            logger.info("─" * 40)
            for rec in report["recommendations"]:
                logger.info("  → %s", rec)

    # ── Done ────────────────────────────────────────────────────────────
    elapsed = time.time() - overall_start
    logger.info("=" * 60)
    logger.info("COMPLETE — %.0fs total", elapsed)
    logger.info("  Adapter: %s", ADAPTER_DIR)
    logger.info("  Report: %s", REPORT_PATH)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
