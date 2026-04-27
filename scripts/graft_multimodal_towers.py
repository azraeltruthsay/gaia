#!/usr/bin/env python3
"""
Graft audio_tower and vision_tower weights from base Gemma-4-E4B
into a trained GAIA Core safetensors merge.

Problem: our LoRA → safetensors merge pipeline saved the trained text LM
weights but dropped the heavy weight matrices in audio_tower and
vision_tower (only quant-calibration buffers like input_min/input_max
survived). The load report at inference time shows ~480 audio + ~448
vision keys MISSING, and the engine deletes the towers to recover VRAM.

Fix: state-dict graft. For every key, prefer the trained v5 value; fall
back to the base value. This restores multimodal capability without
touching the trained text weights.

Usage:
    python scripts/graft_multimodal_towers.py \\
        --base  /models/google/gemma-4-E4B \\
        --trained /models/Gemma4-E4B-GAIA-Unified-v5 \\
        --out   /models/Gemma4-E4B-GAIA-Unified-v5-Multimodal \\
        [--dry-run]

Run this inside gaia-study (has torch + safetensors + the models volume).

Verification: after the graft, load the merged model via transformers
and assert the load report is empty (no missing keys, no unexpected
keys beyond the base's own benign unexpected list).
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

log = logging.getLogger("graft")

_TOWER_PREFIXES = (
    "model.audio_tower.", "model.vision_tower.",
    "audio_tower.", "vision_tower.",
    "model.embed_vision.", "model.embed_audio.",
    "embed_vision.", "embed_audio.",
)


def _is_tower_key(key: str) -> bool:
    return any(key.startswith(p) for p in _TOWER_PREFIXES)


def _load_state_dict_index(model_dir: Path) -> dict:
    """Return {tensor_key: shard_filename} for a safetensors model dir.

    Handles both single-file and sharded layouts.
    """
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            data = json.load(f)
        return data["weight_map"]
    single = model_dir / "model.safetensors"
    if single.exists():
        from safetensors import safe_open
        keys = {}
        with safe_open(str(single), framework="pt") as f:
            for k in f.keys():
                keys[k] = "model.safetensors"
        return keys
    raise FileNotFoundError(f"No safetensors found in {model_dir}")


def _open_shards(model_dir: Path, weight_map: dict):
    """Return {shard_filename: safe_open handle} for every unique shard."""
    from safetensors import safe_open
    shards = {}
    for fname in set(weight_map.values()):
        shards[fname] = safe_open(str(model_dir / fname), framework="pt")
    return shards


def graft(base_dir: Path, trained_dir: Path, out_dir: Path, dry_run: bool = False) -> None:
    log.info("Indexing trained model: %s", trained_dir)
    trained_map = _load_state_dict_index(trained_dir)
    log.info("  %d tensors", len(trained_map))

    log.info("Indexing base model: %s", base_dir)
    base_map = _load_state_dict_index(base_dir)
    log.info("  %d tensors", len(base_map))

    # Strategy: for every TOWER key, take base (has pretrained weights with
    # the QAT class layout Gemma4ClippableLinear expects). For every other
    # key (LM body, embeddings, norms), prefer trained. Keys that only
    # exist in one side get passed through.
    #
    # This fixes the naming mismatch: trained saves flat `q_proj.weight`
    # but Gemma4ClippableLinear expects nested `q_proj.linear.weight`.
    # Grafting the base tower tensors restores the nested naming AND the
    # QAT calibration buffers (input_min/max) needed at inference time.
    base_keys = set(base_map)
    trained_keys = set(trained_map)
    all_keys = base_keys | trained_keys

    graft_from_base = set()
    take_from_trained = set()
    skip_trained_flat_tower = set()  # flat tower keys in trained we discard

    for k in all_keys:
        if _is_tower_key(k):
            # Tower key: always graft from base (correct naming + calibration)
            if k in base_keys:
                graft_from_base.add(k)
            # Flat-named tower keys from trained are discarded; base supplies
            # nested equivalents.
            if k in trained_keys and k not in base_keys:
                skip_trained_flat_tower.add(k)
        else:
            # Non-tower (LM body, embeddings, shared norms). Prefer trained.
            # Do NOT backfill from base for non-tower keys the trained omitted —
            # Gemma 4's num_kv_shared_layers=18 intentionally drops k/v projs
            # on upper layers because they read K/V from earlier layers.
            # Grafting in the unused base tensors would waste space and could
            # confuse shape checks.
            if k in trained_keys:
                take_from_trained.add(k)

    log.info("Key strategy:")
    log.info("  from trained (LM body):      %d", len(take_from_trained))
    log.info("  from base (towers + gaps):   %d", len(graft_from_base))
    log.info("  discarded (flat tower dups): %d", len(skip_trained_flat_tower))
    if skip_trained_flat_tower:
        sample = sorted(skip_trained_flat_tower)[:5]
        log.info("    sample discarded: %s", sample)

    # Sanity: did we actually pick up the expected tower key counts?
    audio_from_base = sum(1 for k in graft_from_base if "audio_tower" in k)
    vision_from_base = sum(1 for k in graft_from_base if "vision_tower" in k)
    log.info("  audio_tower keys from base:  %d (expect ~751)", audio_from_base)
    log.info("  vision_tower keys from base: %d (expect ~658)", vision_from_base)

    if dry_run:
        log.info("[dry-run] stopping before writing output")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy config and ancillary files from BASE (not trained) — trained may
    # have saved an incomplete config (no audio/vision). Tokenizer comes from
    # trained since identity + tokenizer go together.
    import torch
    from safetensors.torch import save_file

    for fname in ("config.json", "generation_config.json", "processor_config.json"):
        src = base_dir / fname
        if src.exists():
            shutil.copy2(src, out_dir / fname)
            log.info("copied base/%s", fname)

    for fname in ("tokenizer_config.json", "tokenizer.json", "special_tokens_map.json"):
        src = trained_dir / fname
        if not src.exists():
            src = base_dir / fname
        if src.exists():
            shutil.copy2(src, out_dir / fname)
            log.info("copied %s/%s", src.parent.name, fname)

    # Open both models' shards once.
    trained_shards = _open_shards(trained_dir, trained_map)
    base_shards = _open_shards(base_dir, base_map)

    # Assemble the merged state dict using the strategy above.
    merged: dict[str, "torch.Tensor"] = {}

    for k in sorted(take_from_trained):
        shard = trained_shards[trained_map[k]]
        merged[k] = shard.get_tensor(k)

    for k in sorted(graft_from_base):
        shard = base_shards[base_map[k]]
        merged[k] = shard.get_tensor(k)

    log.info("Merged tensor count: %d (expected %d)", len(merged),
             len(take_from_trained) + len(graft_from_base))

    # Estimate size and decide single-file vs sharded output.
    total_bytes = sum(t.numel() * t.element_size() for t in merged.values())
    gb = total_bytes / (1024**3)
    log.info("Total serialized size: %.2f GB", gb)

    out_file = out_dir / "model.safetensors"
    log.info("Writing %s...", out_file)
    save_file(merged, str(out_file), metadata={"format": "pt"})
    log.info("Wrote %s (%.2f GB)", out_file, out_file.stat().st_size / (1024**3))

    # If a trained sharded index exists, we don't need it for the merged
    # single file — clean up any stale index the user might copy over later.
    stale_index = out_dir / "model.safetensors.index.json"
    if stale_index.exists():
        stale_index.unlink()


def verify(out_dir: Path) -> None:
    """Quick sanity: reload the merged shard and assert the architecture's
    expected tower keys are all present."""
    from safetensors import safe_open
    single = out_dir / "model.safetensors"
    keys = set()
    with safe_open(str(single), framework="pt") as f:
        for k in f.keys():
            keys.add(k)
    audio = sum(1 for k in keys if "audio_tower" in k)
    vision = sum(1 for k in keys if "vision_tower" in k)
    log.info("Verify: merged model has %d total keys (%d audio_tower, %d vision_tower)",
             len(keys), audio, vision)
    if audio < 700 or vision < 600:
        log.warning("Tower key counts look low — expected ~751 audio, ~658 vision")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Graft multimodal towers from base into trained")
    parser.add_argument("--base", required=True, help="Path to base model with full tower weights")
    parser.add_argument("--trained", required=True, help="Path to trained GAIA Core merge")
    parser.add_argument("--out", required=True, help="Output directory for the multimodal-grafted model")
    parser.add_argument("--dry-run", action="store_true", help="Report the key diff without writing")
    parser.add_argument("--skip-verify", action="store_true", help="Skip the post-graft verification")
    args = parser.parse_args()

    base_dir = Path(args.base)
    trained_dir = Path(args.trained)
    out_dir = Path(args.out)

    if not base_dir.is_dir():
        log.error("Base dir not found: %s", base_dir); return 1
    if not trained_dir.is_dir():
        log.error("Trained dir not found: %s", trained_dir); return 1

    graft(base_dir, trained_dir, out_dir, dry_run=args.dry_run)

    if not args.dry_run and not args.skip_verify:
        verify(out_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
