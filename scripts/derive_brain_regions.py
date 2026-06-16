#!/usr/bin/env python3
"""
derive_brain_regions.py — A4: re-derive the brain-region map from the CURRENT atlases.

The old gaia-web/static/brain_region_atlas.json is stale (Mar-25, pre-Duality:
3-tier Qwen/Nano, wrong layer counts, old domains) — so causal connectivity
labels every edge 'unknown'. This rebuilds it data-drivenly from the committed
top-k atlases (Core 42L, Prime 36L; nano dropped):

  - Re-encode the stratified corpus through each atlas layer's SAE (CPU/gaia_cpp).
  - Per layer × capability-stratum: mean feature-activation magnitude → a
    layer×stratum profile.
  - Partition each tier's atlas layers into depth ZONES (regions); each zone's
    DOMAINS = the strata that activate it most, FEATURES = top features there.
  - Emit a new brain_region_atlas.json (regions/tier_layer_map/domain_to_region_map)
    compatible with find_region() + the dashboard.

  docker compose exec -T gaia-core python /gaia/GAIA_Project/scripts/derive_brain_regions.py \
    --corpus /gaia/GAIA_Project/knowledge/curricula/sae_atlas/corpus.json
"""
import argparse, json, logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GAIA.BrainMap")

# Depth zones → neuro-inspired region names (honest: depth zones, not literal anatomy).
ZONE_NAMES = ["Perceptual", "Associative", "Integrative", "Executive"]
ZONE_FN = {
    "Perceptual": "Early tokenization / surface features",
    "Associative": "Lexical-semantic association",
    "Integrative": "Cross-context integration / reasoning",
    "Executive": "Decision, planning, output shaping",
}

TIERS = [
    {"tier": "core",  "tag": "CORE_IDENTITY_V3_gguf",     "gguf": "/models/core.gguf"},
    {"tier": "prime", "tag": "PRIME_ABLITERATED_gguf",    "gguf": "/models/prime.gguf"},
]


def profile_tier(tier_cfg, prompts, strata, topk):
    import torch
    from gaia_engine.cpp import gaia_cpp
    from gaia_engine.sae_trainer import SAETrainer

    atlas_dir = Path("/shared/atlas") / tier_cfg["tier"] / tier_cfg["tag"]
    layers = json.load(open(atlas_dir / "meta.json"))["layers"]
    backend = gaia_cpp.LlamaCppBackend(model_path=tier_cfg["gguf"], n_gpu_layers=0,
                                       capture_layers=layers, n_ctx=2048)
    trainer = SAETrainer(model=backend, tokenizer=None, device="cpu")
    trainer.load_atlas(str(atlas_dir))
    for sae in trainer.saes.values():
        sae.k = topk; sae.eval()
    trainer.record_activations_gguf(prompts, layers=layers, backend=backend)

    uniq = sorted(set(s for s in strata if s))
    # layer -> {stratum -> mean activation magnitude}, and layer -> top feature indices
    layer_profile, layer_topfeat = {}, {}
    with torch.no_grad():
        for layer in layers:
            sae = trainer.saes.get(layer)
            acts = trainer.activations.get(layer, [])
            if sae is None or not acts:
                continue
            X = torch.cat(acts, dim=0).float()
            Xn = (X - sae._norm_mean) / sae._norm_std
            F = sae.get_feature_activations(Xn)          # [N, n_feat] top-k
            prof = {}
            for s in uniq:
                idx = [i for i, st in enumerate(strata) if st == s and i < F.shape[0]]
                prof[s] = float(F[idx].sum(dim=1).mean()) if idx else 0.0
            layer_profile[layer] = prof
            layer_topfeat[layer] = F.mean(dim=0).topk(min(10, F.shape[1])).indices.tolist()
    return layers, layer_profile, layer_topfeat


def build_regions(tier, layers, layer_profile, layer_topfeat, start_id):
    # Partition atlas layers into up to len(ZONE_NAMES) contiguous depth zones.
    n = len(layers)
    nz = min(len(ZONE_NAMES), n)
    bounds = [layers[round(i * n / nz)] for i in range(nz)] + [layers[-1]]
    regions = []
    for z in range(nz):
        zlayers = [L for L in layers if bounds[z] <= L <= bounds[z + 1]] if z == nz - 1 \
                  else [L for L in layers if bounds[z] <= L < bounds[z + 1]]
        if not zlayers:
            continue
        # Dominant strata = highest summed activation across the zone's layers.
        agg = {}
        for L in zlayers:
            for s, v in layer_profile.get(L, {}).items():
                agg[s] = agg.get(s, 0.0) + v
        dom = [s for s, _ in sorted(agg.items(), key=lambda kv: -kv[1])[:3]]
        feats = {}
        for L in zlayers:
            feats[str(L)] = layer_topfeat.get(L, [])
        regions.append({
            "id": start_id + z,
            "name": f"{tier.capitalize()}-{ZONE_NAMES[z]}",
            "tier": tier,
            "function": ZONE_FN[ZONE_NAMES[z]],
            "domains": dom,
            "layerRange": [zlayers[0], zlayers[-1]],
            "atlasLayers": zlayers,
            "topFeaturesByLayer": feats,
        })
    return regions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--out", default="/gaia/GAIA_Project/gaia-web/static/brain_region_atlas.json")
    args = ap.parse_args()

    items = json.load(open(args.corpus))
    prompts = [it["text"] for it in items]
    strata = [it.get("stratum", "") for it in items]

    all_regions, tier_layer_map, domain_to_region = [], {}, {}
    next_id = 1
    for tc in TIERS:
        logger.info("Profiling %s (%s)...", tc["tier"], tc["tag"])
        layers, prof, topf = profile_tier(tc, prompts, strata, args.topk)
        regions = build_regions(tc["tier"], layers, prof, topf, next_id)
        next_id += len(regions)
        all_regions += regions
        tier_layer_map[tc["tier"]] = {"atlas_sampled_layers": layers,
                                      "n_layers": (max(layers) + 1)}
        for r in regions:
            for d in r["domains"]:
                domain_to_region.setdefault(d, []).append(r["name"])
            logger.info("  %s [%d-%d]: domains=%s", r["name"], r["layerRange"][0], r["layerRange"][1], r["domains"])

    atlas = {
        "_meta": {
            "version": "2.0.0",
            "description": "Brain-region map re-derived from committed Sovereign-Duality top-k atlases (A4)",
            "source_atlases": "/shared/atlas/{core,prime}/<top-k tag>",
            "derived": "depth zones per tier; domains = dominant capability strata; features = top per layer",
            "tiers": [t["tier"] for t in TIERS],
        },
        "regions": all_regions,
        "tier_layer_map": tier_layer_map,
        "domain_to_region_map": domain_to_region,
    }
    Path(args.out).write_text(json.dumps(atlas, indent=2))
    logger.info("Wrote %d regions across %d tiers → %s", len(all_regions), len(TIERS), args.out)


if __name__ == "__main__":
    main()
