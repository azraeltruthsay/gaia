#!/usr/bin/env python3
"""SAE Causal Connectivity — compute directed synaptic connections between features.

For each pair of SAE layers (N, M) where N < M, computes:
    influence(A→B) = encoder_M[B] · decoder_N[A]

This measures how much feature A at layer N causally drives feature B at layer M
through the residual stream. High dot product = A firing pushes the residual
stream toward B's activation direction.

Outputs a connectivity map per tier with:
- Top causal edges (directed, cross-layer)
- Per-region connectivity (which brain regions drive which)
- Feature-level synapse list for visualization

Usage:
    python sae_causal_connectivity.py --tier nano
    python sae_causal_connectivity.py --all
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GAIA.SAE.Causal")


def load_atlas_layer(path: Path) -> dict:
    """Load a single SAE layer checkpoint."""
    data = torch.load(path, map_location="cpu", weights_only=True)
    return data


def compute_causal_edges(
    decoder_weights: torch.Tensor,  # [num_features_N, hidden_size] — decoder of layer N
    encoder_weights: torch.Tensor,  # [num_features_M, hidden_size] — encoder of layer M
    top_k: int = 50,
    min_strength: float = 0.01,
) -> List[dict]:
    """Compute top causal edges from layer N features to layer M features.

    influence(A→B) = encoder_M[B] · decoder_N[A]

    Returns list of {source, target, strength} dicts, sorted by absolute strength.
    """
    # decoder_weights shape: [hidden_size, num_features_N] — each column is a feature's residual contribution
    # encoder_weights shape: [num_features_M, hidden_size] — each row is a feature's input direction
    # Transpose decoder so each ROW is a feature vector: [N_feats, hidden]
    dec = decoder_weights.t()  # [N_feats, hidden]

    # Normalize for cosine similarity (direction matters more than magnitude)
    dec_norm = F.normalize(dec, dim=1)      # [N_feats, hidden]
    enc_norm = F.normalize(encoder_weights, dim=1)  # [M_feats, hidden]

    # Full connectivity matrix: [M_feats, N_feats]
    # Each entry [b, a] = how much feature a at source drives feature b at target
    connectivity = enc_norm @ dec_norm.t()

    # Find top-k strongest causal edges (by absolute value)
    flat = connectivity.abs().flatten()
    k = min(top_k, flat.numel())
    top_vals, top_idx = flat.topk(k)

    edges = []
    n_feats_source = dec.shape[0]   # N_feats (columns of connectivity)
    n_feats_target = encoder_weights.shape[0]  # M_feats (rows of connectivity)
    for i in range(k):
        val = top_vals[i].item()
        if val < min_strength:
            break
        flat_i = top_idx[i].item()
        target_feat = flat_i // n_feats_source
        source_feat = flat_i % n_feats_source
        # Use signed strength (positive = excitatory, negative = inhibitory)
        signed = connectivity[target_feat, source_feat].item()
        edges.append({
            "source": int(source_feat),
            "target": int(target_feat),
            "strength": round(signed, 4),
            "abs_strength": round(val, 4),
        })

    return edges


def compute_region_connectivity(
    edges: List[dict],
    source_layer: int,
    target_layer: int,
    region_atlas: dict,
    tier: str,
) -> List[dict]:
    """Map feature-level edges to brain region connectivity.

    Uses the brain_region_atlas.json to determine which region each feature belongs to
    based on layer ranges.
    """
    regions = region_atlas.get("regions", [])

    def find_region(layer: int, tier_name: str) -> str:
        """Find which brain region a layer belongs to for this tier."""
        for r in regions:
            if r["tier"] != tier_name:
                continue
            lr = r.get("layerRange", [0, 0])
            if lr[0] <= layer <= lr[1]:
                return r["name"]
        return "unknown"

    source_region = find_region(source_layer, tier)
    target_region = find_region(target_layer, tier)

    if source_region == target_region:
        conn_type = "intra-region"
    else:
        conn_type = "inter-region"

    return {
        "source_region": source_region,
        "target_region": target_region,
        "source_layer": source_layer,
        "target_layer": target_layer,
        "type": conn_type,
        "num_edges": len(edges),
        "mean_strength": round(sum(e["abs_strength"] for e in edges) / max(len(edges), 1), 4),
        "max_strength": round(max((e["abs_strength"] for e in edges), default=0), 4),
        "top_edges": edges[:10],  # Top 10 for this layer pair
    }


def run_tier(tier: str, atlas_base: str, region_atlas_path: str, output_base: str):
    """Compute causal connectivity for one tier."""
    atlas_dir = Path(atlas_base) / tier / "baseline"
    if not atlas_dir.exists():
        logger.error("Atlas not found: %s", atlas_dir)
        return None

    # Load region atlas for brain region mapping
    region_atlas = {}
    try:
        with open(region_atlas_path) as f:
            region_atlas = json.load(f)
    except Exception as e:
        logger.warning("Could not load region atlas: %s", e)

    # Load all layer SAEs
    layers = {}
    for pt_file in sorted(atlas_dir.glob("layer_*.pt")):
        layer_idx = int(pt_file.stem.split("_")[1])
        data = load_atlas_layer(pt_file)
        layers[layer_idx] = data
        logger.info("  Loaded layer %d: encoder=%s decoder=%s",
                     layer_idx, list(data["encoder_weight"].shape), list(data["decoder_weight"].shape))

    if len(layers) < 2:
        logger.error("Need at least 2 layers for causal connectivity")
        return None

    sorted_layers = sorted(layers.keys())
    logger.info("Computing causal connectivity for %s: layers %s", tier, sorted_layers)

    # Compute edges for each layer pair (N → M where N < M)
    all_layer_pairs = []
    all_synapses = []  # Flat list for visualization
    region_pairs = {}  # (source_region, target_region) → aggregated stats

    for i, src_layer in enumerate(sorted_layers):
        for tgt_layer in sorted_layers[i + 1:]:
            src_data = layers[src_layer]
            tgt_data = layers[tgt_layer]

            t0 = time.time()
            edges = compute_causal_edges(
                decoder_weights=src_data["decoder_weight"],
                encoder_weights=tgt_data["encoder_weight"],
                top_k=100,
                min_strength=0.05,
            )
            elapsed = time.time() - t0

            # Map to brain regions
            region_conn = compute_region_connectivity(
                edges, src_layer, tgt_layer, region_atlas, tier
            )

            pair_key = f"L{src_layer}→L{tgt_layer}"
            logger.info("  %s: %d causal edges (max=%.3f, mean=%.3f) [%.2fs] %s→%s",
                         pair_key, len(edges),
                         region_conn["max_strength"], region_conn["mean_strength"],
                         elapsed,
                         region_conn["source_region"], region_conn["target_region"])

            all_layer_pairs.append({
                "pair": pair_key,
                "source_layer": src_layer,
                "target_layer": tgt_layer,
                **region_conn,
            })

            # Aggregate region-level connectivity
            rkey = (region_conn["source_region"], region_conn["target_region"])
            if rkey not in region_pairs:
                region_pairs[rkey] = {
                    "source": rkey[0], "target": rkey[1],
                    "total_edges": 0, "max_strength": 0,
                    "strengths": [], "layer_pairs": [],
                }
            rp = region_pairs[rkey]
            rp["total_edges"] += len(edges)
            rp["max_strength"] = max(rp["max_strength"], region_conn["max_strength"])
            rp["strengths"].append(region_conn["mean_strength"])
            rp["layer_pairs"].append(pair_key)

            # Add to flat synapse list (for visualization)
            for edge in edges[:20]:  # Top 20 per layer pair
                all_synapses.append({
                    "source_feature": edge["source"],
                    "target_feature": edge["target"],
                    "source_layer": src_layer,
                    "target_layer": tgt_layer,
                    "source_region": region_conn["source_region"],
                    "target_region": region_conn["target_region"],
                    "strength": edge["strength"],
                    "type": "excitatory" if edge["strength"] > 0 else "inhibitory",
                })

    # Finalize region connectivity
    region_summary = []
    for rkey, rp in sorted(region_pairs.items(), key=lambda x: -x[1]["max_strength"]):
        rp["mean_strength"] = round(sum(rp["strengths"]) / len(rp["strengths"]), 4)
        rp["max_strength"] = round(rp["max_strength"], 4)
        del rp["strengths"]
        region_summary.append(rp)

    # Sort synapses by absolute strength
    all_synapses.sort(key=lambda s: -abs(s["strength"]))

    result = {
        "tier": tier,
        "timestamp": time.time(),
        "layers": sorted_layers,
        "num_layer_pairs": len(all_layer_pairs),
        "total_causal_edges": sum(lp["num_edges"] for lp in all_layer_pairs),
        "total_synapses_for_viz": len(all_synapses),
        "region_connectivity": region_summary,
        "layer_pairs": all_layer_pairs,
        "synapses": all_synapses[:200],  # Top 200 strongest for viz
    }

    # Save
    output_dir = Path(output_base) / tier
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "causal_connectivity.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info("Saved %s causal connectivity: %d edges, %d synapses, %d region pairs → %s",
                tier, result["total_causal_edges"], len(all_synapses),
                len(region_summary), out_path)

    return result


def main():
    parser = argparse.ArgumentParser(description="SAE Causal Connectivity")
    parser.add_argument("--tier", choices=["nano", "core", "prime"])
    parser.add_argument("--all", action="store_true", help="Run all tiers")
    parser.add_argument("--atlas", default="/shared/atlas", help="Atlas base directory")
    parser.add_argument("--region-atlas", default="/gaia/GAIA_Project/gaia-web/static/brain_region_atlas.json")
    parser.add_argument("--output", default="/shared/atlas", help="Output directory")
    args = parser.parse_args()

    tiers = ["nano", "core", "prime"] if args.all else [args.tier]
    if not tiers or tiers == [None]:
        parser.error("Specify --tier or --all")

    for tier in tiers:
        logger.info("=" * 60)
        logger.info("Computing causal connectivity: %s", tier.upper())
        logger.info("=" * 60)
        run_tier(tier, args.atlas, args.region_atlas, args.output)


if __name__ == "__main__":
    main()
