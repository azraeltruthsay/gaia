#!/usr/bin/env python3
"""CodeMind Training Comparison — Pre/Post SAE Atlas + Behavioral Baseline.

Runs inside gaia-core container (has access to model pool and SAE trainer).

Phase 1 (pre-training): Capture behavioral baseline + SAE atlas snapshot
Phase 2 (after QLoRA):  Re-capture and diff

Usage:
    # Pre-training baseline:
    docker compose exec gaia-core python /app/scripts/codemind_training_comparison.py --phase pre

    # Post-training comparison:
    docker compose exec gaia-core python /app/scripts/codemind_training_comparison.py --phase post

    # Diff the two phases:
    docker compose exec gaia-core python /app/scripts/codemind_training_comparison.py --phase diff
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GAIA.CodeMind.Comparison")

# ── Output paths ─────────────────────────────────────────────────────────

ATLAS_BASE = Path(os.environ.get("SHARED_DIR", "/shared")) / "atlas" / "codemind"
RESULTS_BASE = Path(os.environ.get("SHARED_DIR", "/shared")) / "codemind" / "training_comparison"

# ── CodeMind diagnostic prompts ──────────────────────────────────────────
# These test whether the model has code-reasoning capabilities.
# The SAE atlas captures which neurons activate for each prompt.

DIAGNOSTIC_PROMPTS = [
    # Lint fixing
    "Fix this ruff error: F401 `os` imported but unused in the file that only uses json and logging.",
    "What does ruff error E722 mean and how do you fix a bare except?",

    # GAIA architecture
    "What is the candidates-first development workflow in GAIA?",
    "What files are GAIA's vital organs and what rules apply to them?",

    # Root cause analysis
    "Nano returned the wrong time, off by 20 minutes. Trace the root cause across services.",
    "The immune MRI reports 690 lint errors. What's the triage strategy?",

    # Fix discipline
    "What is the difference between a direct fix and a fix blueprint?",
    "When should CodeMind respond with CANNOT_FIX instead of proposing a change?",

    # Safety / scope
    "What are the three scope tiers and how do you classify files into them?",
    "What happens if a proposed fix makes the cognitive battery score drop?",

    # Conventions
    "How should I structure a new utility module in gaia-common?",
    "What is GAIA's error handling convention for HTTP calls between services?",

    # Control prompts (non-code, should activate different features)
    "Who is GAIA and what is her purpose?",
    "What is the capital of France?",
    "Tell me about your emotional architecture and samvega.",
]

# Layers to analyze (Prime 8B has ~32 layers)
TARGET_LAYERS = [4, 8, 12, 16, 20, 24, 28]


def run_behavioral_baseline(output_dir: Path):
    """Send diagnostic prompts to Prime and capture responses."""
    logger.info("Running behavioral baseline...")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from urllib.request import Request, urlopen
    except ImportError:
        logger.error("urllib not available")
        return

    endpoint = os.environ.get("PRIME_ENDPOINT", "http://gaia-prime:7777")
    results = []

    for i, prompt in enumerate(DIAGNOSTIC_PROMPTS):
        logger.info("  [%d/%d] %s", i + 1, len(DIAGNOSTIC_PROMPTS), prompt[:60])

        # Try with code-architect adapter first, then base model
        for model_name in ["code-architect", None]:
            try:
                payload = json.dumps({
                    "model": model_name or "/models/Huihui-Qwen3-8B-abliterated-v2-merged",
                    "messages": [
                        {"role": "system", "content": "You are CodeMind, GAIA's code self-improvement layer."},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": 512,
                    "temperature": 0.1,
                }).encode()
                req = Request(
                    f"{endpoint}/v1/chat/completions",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                )
                with urlopen(req, timeout=60) as resp:
                    result = json.loads(resp.read().decode())
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                results.append({
                    "prompt": prompt,
                    "model": model_name or "base",
                    "response": content,
                    "response_length": len(content),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                break  # Success — don't try fallback
            except Exception as e:
                if model_name == "code-architect":
                    logger.debug("Adapter not available, trying base: %s", e)
                    continue
                logger.warning("Failed for prompt %d: %s", i, e)
                results.append({
                    "prompt": prompt,
                    "model": "error",
                    "response": str(e),
                    "response_length": 0,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

    out_path = output_dir / "behavioral_responses.json"
    out_path.write_text(json.dumps(results, indent=2))
    logger.info("Behavioral baseline saved: %s (%d responses)", out_path, len(results))
    return results


def run_sae_atlas(output_dir: Path, phase: str):
    """Record activations and train SAE atlas on diagnostic prompts."""
    logger.info("Running SAE atlas snapshot (phase=%s)...", phase)
    atlas_dir = ATLAS_BASE / phase
    atlas_dir.mkdir(parents=True, exist_ok=True)

    try:
        from gaia_common.engine.sae_trainer import SAETrainer
        # Try to get model from GAIAEngine or load directly
        try:
            from gaia_common.engine.core import _engine
            if _engine is not None:
                model = _engine.model
                tokenizer = _engine.tokenizer
                device = _engine.device
                logger.info("Using GAIAEngine model (device=%s)", device)
            else:
                raise RuntimeError("GAIAEngine not initialized")
        except Exception:
            logger.warning("GAIAEngine not available — SAE atlas requires local model")
            logger.info("SAE atlas skipped (model not in this container's memory)")
            # Save a placeholder
            (atlas_dir / "meta.json").write_text(json.dumps({
                "status": "skipped",
                "reason": "local model not available in this container",
                "phase": phase,
                "timestamp": time.time(),
                "note": "SAE atlas requires the model to be loaded locally (GAIAEngine). "
                        "Run this on the container that hosts the Nano/Core model.",
            }, indent=2))
            return None

        trainer = SAETrainer(model, tokenizer, device=device)

        # Record activations for diagnostic prompts
        stats = trainer.record_activations(
            prompts=DIAGNOSTIC_PROMPTS,
            layers=TARGET_LAYERS,
            system_prompt="You are CodeMind, GAIA's code self-improvement layer.",
        )
        logger.info("Activations recorded: %s", stats)

        # Train SAE (overcomplete basis: 2x hidden size)
        hidden_size = list(trainer.activations.values())[0][0].shape[-1]
        num_features = hidden_size * 2
        train_results = trainer.train_sae(
            layers=TARGET_LAYERS,
            num_features=num_features,
            sparsity_weight=0.01,
            lr=1e-3,
            epochs=50,
            batch_size=256,
        )
        logger.info("SAE training complete: %s", {k: v.get("active_features") for k, v in train_results.items()})

        # Save atlas
        trainer.save_atlas(str(atlas_dir))

        # Also save per-prompt feature analysis
        prompt_analyses = []
        analysis_layer = TARGET_LAYERS[-1]  # Use deepest layer
        for prompt in DIAGNOSTIC_PROMPTS:
            analysis = trainer.analyze_prompt(prompt, analysis_layer, top_k=15)
            prompt_analyses.append(analysis)

        analyses_path = atlas_dir / "prompt_analyses.json"
        analyses_path.write_text(json.dumps(prompt_analyses, indent=2))

        # Save training stats
        stats_path = atlas_dir / "training_stats.json"
        stats_path.write_text(json.dumps({
            "phase": phase,
            "recording_stats": stats,
            "training_results": train_results,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }, indent=2, default=str))

        logger.info("SAE atlas saved to %s", atlas_dir)
        return train_results

    except ImportError as e:
        logger.warning("SAE trainer not available: %s", e)
        return None


def run_diff():
    """Compare pre and post training results."""
    logger.info("Computing pre/post training diff...")
    diff_dir = RESULTS_BASE / "diff"
    diff_dir.mkdir(parents=True, exist_ok=True)

    pre_dir = RESULTS_BASE / "pre"
    post_dir = RESULTS_BASE / "post"

    if not pre_dir.exists() or not post_dir.exists():
        logger.error("Need both pre/ and post/ directories. Run --phase pre and --phase post first.")
        return

    # ── Behavioral diff ──
    pre_behavioral = json.loads((pre_dir / "behavioral_responses.json").read_text())
    post_behavioral = json.loads((post_dir / "behavioral_responses.json").read_text())

    behavioral_diff = []
    for pre, post in zip(pre_behavioral, post_behavioral):
        diff_entry = {
            "prompt": pre["prompt"],
            "pre_model": pre.get("model", "unknown"),
            "post_model": post.get("model", "unknown"),
            "pre_length": pre["response_length"],
            "post_length": post["response_length"],
            "length_change": post["response_length"] - pre["response_length"],
            "pre_response_preview": pre["response"][:200],
            "post_response_preview": post["response"][:200],
        }
        behavioral_diff.append(diff_entry)

    (diff_dir / "behavioral_diff.json").write_text(json.dumps(behavioral_diff, indent=2))

    # ── SAE atlas diff ──
    pre_atlas = ATLAS_BASE / "pre"
    post_atlas = ATLAS_BASE / "post"

    if (pre_atlas / "prompt_analyses.json").exists() and (post_atlas / "prompt_analyses.json").exists():
        pre_analyses = json.loads((pre_atlas / "prompt_analyses.json").read_text())
        post_analyses = json.loads((post_atlas / "prompt_analyses.json").read_text())

        feature_diffs = []
        for pre_a, post_a in zip(pre_analyses, post_analyses):
            pre_features = {f["index"]: f["strength"] for f in pre_a.get("top_features", [])}
            post_features = {f["index"]: f["strength"] for f in post_a.get("top_features", [])}

            # Find features that emerged or disappeared
            new_features = set(post_features.keys()) - set(pre_features.keys())
            lost_features = set(pre_features.keys()) - set(post_features.keys())
            changed = {}
            for idx in set(pre_features.keys()) & set(post_features.keys()):
                delta = post_features[idx] - pre_features[idx]
                if abs(delta) > 0.01:
                    changed[idx] = {"pre": pre_features[idx], "post": post_features[idx], "delta": round(delta, 4)}

            feature_diffs.append({
                "prompt": pre_a.get("prompt", ""),
                "layer": pre_a.get("layer"),
                "new_features": list(new_features),
                "lost_features": list(lost_features),
                "changed_features": changed,
                "new_count": len(new_features),
                "lost_count": len(lost_features),
                "changed_count": len(changed),
            })

        (diff_dir / "feature_diff.json").write_text(json.dumps(feature_diffs, indent=2))

        # Summary
        total_new = sum(d["new_count"] for d in feature_diffs)
        total_lost = sum(d["lost_count"] for d in feature_diffs)
        total_changed = sum(d["changed_count"] for d in feature_diffs)
        code_prompts = feature_diffs[:12]  # First 12 are code-related
        control_prompts = feature_diffs[12:]  # Last 3 are control

        code_new = sum(d["new_count"] for d in code_prompts)
        control_new = sum(d["new_count"] for d in control_prompts)

        summary = {
            "total_new_features": total_new,
            "total_lost_features": total_lost,
            "total_changed_features": total_changed,
            "code_prompt_new_features": code_new,
            "control_prompt_new_features": control_new,
            "interpretation": (
                f"Training activated {total_new} new features and changed {total_changed} existing ones. "
                f"Code-related prompts gained {code_new} new features vs {control_new} for control prompts. "
                + ("Code features emerged disproportionately — training is working as intended."
                   if code_new > control_new * 2
                   else "Feature changes are diffuse — may need more targeted training data.")
            ),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        (diff_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        logger.info("DIFF SUMMARY: %s", summary["interpretation"])
    else:
        logger.info("SAE atlases not available for diff (model not loaded locally)")

    # ── Behavioral summary ──
    longer_responses = sum(1 for d in behavioral_diff if d["length_change"] > 50)
    shorter_responses = sum(1 for d in behavioral_diff if d["length_change"] < -50)
    logger.info(
        "Behavioral: %d prompts, %d longer post-training, %d shorter",
        len(behavioral_diff), longer_responses, shorter_responses,
    )

    logger.info("Diff complete. Results at %s", diff_dir)


def main():
    parser = argparse.ArgumentParser(description="CodeMind Training Comparison")
    parser.add_argument("--phase", choices=["pre", "post", "diff"], required=True)
    args = parser.parse_args()

    if args.phase in ("pre", "post"):
        output_dir = RESULTS_BASE / args.phase
        output_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Behavioral baseline
        run_behavioral_baseline(output_dir)

        # Step 2: SAE atlas (if model available locally)
        run_sae_atlas(output_dir, args.phase)

        logger.info("Phase '%s' complete. Results at %s", args.phase, output_dir)

    elif args.phase == "diff":
        run_diff()


if __name__ == "__main__":
    main()
