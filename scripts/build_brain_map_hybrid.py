#!/usr/bin/env python3
"""build_brain_map_hybrid.py — A4: hybrid brain-region map (anatomy + data).

Rebuilds the stale Mar-25 brain map (GAIA_Project-7jz) the HYBRID way the operator
chose: keep the 13 anatomical regions + their dashboard coordinates, but let the
DISCRIMINATIVE data (feature_cognitive_signal_table.json) decide which cognitive
signal each region owns and attach that signal's selective features. Re-tiered to
Sovereign Duality (Nano retired → its 3 subcortical regions become Core).

Also emits `layer_N_labels.json` for the live activation monitor (backend
_ATLAS_DIR=/shared/atlas/core reads these), so the mind map shows discriminative
signal labels instead of feature_NNNN stubs.

Pure synthesis over the committed table + atlas meta — CPU-only, no model load.

  docker exec gaia-study python3 /gaia/GAIA_Project/scripts/build_brain_map_hybrid.py
"""
import json
from collections import defaultdict
from pathlib import Path

TABLE = Path("/gaia/GAIA_Project/knowledge/blueprints/feature_cognitive_signal_table.json")
ATLAS = Path("/shared/atlas")
OUT_ATLAS = Path("/gaia/GAIA_Project/gaia-web/static/brain_region_atlas.json")
OUT_MD = Path("/gaia/GAIA_Project/knowledge/blueprints/brain_region_map.md")
FEATS_PER_REGION = 8

# 13 anatomical regions: (name, tier, function, cx, cy, rx, ry, [signals owned]).
# Coordinates preserved from the dashboard's hardcoded BRAIN_REGIONS (app.js).
# Tier re-mapped to Duality: the former Nano subcortex → Core. Signals assigned by
# functional best-match; the discriminative FEATURES come from the data per tier.
REGIONS = [
    # Prime — higher cognition
    ("Prefrontal",     "prime", "Executive reasoning, complex planning",        36, 108, 14, 28, ["deliberation", "curiosity_gap"]),
    ("Orbitofrontal",  "prime", "Value judgement, coherence / ethical sentinel", 30, 155, 16, 18, ["coherence_contradiction", "affect_laden"]),
    ("Broca's Area",   "prime", "Language generation, response composition",     60, 165, 18, 12, ["register_chitchat"]),
    ("Motor Cortex",   "prime", "Action planning, tool execution",               88,  46, 24, 14, ["competence_problem"]),
    # Core — operational
    ("Somatosensory",  "core",  "Input parsing, prompt analysis",               165,  42, 18, 12, ["identity_self"]),
    ("Parietal",       "core",  "Spatial / contextual reasoning, working memory",148,  48, 26, 14, ["spatial_reasoning", "register_technical"]),
    ("Wernicke's Area","core",  "Language comprehension, intent detection",      145, 155, 20, 12, ["register_chitchat"]),
    ("Temporal",       "core",  "Memory retrieval, semantic, affective residue", 108, 155, 40, 11, ["affect_laden", "neutral_factual"]),
    ("Occipital",      "core",  "Pattern recognition, embedding similarity",     222,  88, 18, 28, ["neutral_factual"]),
    ("Visual Cortex",  "core",  "Vision processing (multimodal)",                232, 135, 14, 22, []),
    # Former Nano subcortex → now Core (Duality: Core owns triage/coordination/reflex)
    ("Thalamus",       "core",  "Relay / routing hub, triage classification",    140, 120, 16, 14, []),
    ("Cerebellum",     "core",  "Coordination, response cleanup",                205, 178, 36, 18, []),
    ("Brain Stem",     "core",  "Reflexes, health checks, heartbeat",            152, 192, 16, 20, []),
]

# Old tier depths (for proportional layerRange rescale) → new Duality depths.
OLD_MAX = {"core": 24, "prime": 32}
# New anatomical depth zones as fractions of tier depth (front→back ~ deep→shallow
# is a metaphor; we keep each region's original fractional band).
OLD_RANGE = {
    "Prefrontal": (24, 31), "Orbitofrontal": (16, 24), "Broca's Area": (8, 16),
    "Motor Cortex": (4, 12), "Somatosensory": (0, 8), "Parietal": (8, 16),
    "Wernicke's Area": (14, 22), "Temporal": (0, 10), "Occipital": (18, 24),
    "Visual Cortex": (22, 24), "Thalamus": (8, 16), "Cerebellum": (16, 23),
    "Brain Stem": (0, 8),
}


def main():
    table = json.load(open(TABLE))
    models = table["models"]
    new_max = {t: models[t]["analysis_layer"] + 1 for t in ("core", "prime")}  # 42, 36

    regions_out = []
    domain_to_region = defaultdict(list)
    for i, (name, tier, fn, cx, cy, rx, ry, signals) in enumerate(REGIONS, start=1):
        lo, hi = OLD_RANGE[name]
        scale = new_max[tier] / OLD_MAX[tier]
        lrange = [round(lo * scale), min(new_max[tier] - 1, round(hi * scale))]
        # Attach this signal's top discriminative features from the region's TIER.
        feats = []
        sigtab = models[tier]["signals"]
        for sig in signals:
            for f in sigtab.get(sig, [])[:FEATS_PER_REGION]:
                feats.append({
                    "index": f["index"],
                    "strength": f["peak_strength"],
                    "selectivity": f["selectivity"],
                    "label": f"{sig} · {f['exemplar'][:48]}",
                    "layer": models[tier]["analysis_layer"],
                    "signal": sig,
                })
            domain_to_region[sig].append(name)
        regions_out.append({
            "id": i, "name": name, "tier": tier, "function": fn,
            "cx": cx, "cy": cy, "rx": rx, "ry": ry,
            "domains": signals, "layerRange": lrange,
            "features": sorted(feats, key=lambda x: -x["strength"])[:FEATS_PER_REGION],
        })

    tier_layer_map = {}
    for t in ("core", "prime"):
        meta = json.load(open(ATLAS / t / (
            "CORE_IDENTITY_V3" if t == "core" else "PRIME_ABLITERATED") / "meta.json"))
        tier_layer_map[t] = {"n_layers": new_max[t],
                             "atlas_sampled_layers": meta["layers"],
                             "analysis_layer": models[t]["analysis_layer"]}

    atlas = {
        "_meta": {
            "version": "3.0.0",
            "description": "GAIA brain-region map — hybrid (anatomy + discriminative "
                           "SAE data), Sovereign Duality (Core+Prime; Nano retired).",
            "source_atlases": "/shared/atlas/{core/CORE_IDENTITY_V3, prime/PRIME_ABLITERATED}",
            "derived_from": "feature_cognitive_signal_table.json (discriminative features)",
            "viewport": {"width": 280, "height": 225, "brain_faces": "left"},
            "tiers": ["core", "prime"],
            "note": "Region coordinates preserved from the anatomical diagram; each "
                    "region's signals+features are data-assigned. Features are at each "
                    "tier's analysis layer (Core L41 / Prime L35).",
        },
        "regions": regions_out,
        "tier_layer_map": tier_layer_map,
        "domain_to_region_map": {k: sorted(set(v)) for k, v in domain_to_region.items()},
    }
    OUT_ATLAS.write_text(json.dumps(atlas, indent=2))
    n_labels = _emit_layer_labels(models)
    _write_md(regions_out, models)
    print(f"Wrote brain_region_atlas.json ({len(regions_out)} regions), "
          f"{n_labels} live labels, brain_region_map.md")


def _emit_layer_labels(models):
    """Per-layer feature labels for the live activation monitor (backend reads
    /shared/atlas/<tier>/layer_<L>_labels.json)."""
    total = 0
    for tier, tag in (("core", None), ("prime", None)):
        L = models[tier]["analysis_layer"]
        labels = {}
        for sig, feats in models[tier]["signals"].items():
            for f in feats:
                labels[str(f["index"])] = f"{sig} · {f['exemplar'][:40]}"
        out = ATLAS / tier / f"layer_{L}_labels.json"
        out.write_text(json.dumps(labels, indent=0))
        total += len(labels)
    return total


def _write_md(regions, models):
    L = ["# GAIA Brain Region Map",
         "",
         "> Hybrid (anatomy + discriminative SAE data). **Sovereign Duality**: Core + "
         "Prime only — the Nano tier is retired, its 3 subcortical regions re-tiered to "
         "Core. Region coordinates keep the anatomical diagram; each region's cognitive "
         "signals + features are **data-assigned** from `feature_cognitive_signal_table`.",
         "",
         f"> Features are at each tier's analysis layer (Core L{models['core']['analysis_layer']} "
         f"/ Prime L{models['prime']['analysis_layer']}). Core {models['core']['pct_selective']}% / "
         f"Prime {models['prime']['pct_selective']}% of features are stratum-selective.",
         "",
         "## Region Table",
         "",
         "| # | Region | Tier | Function | Signals (data-assigned) | Layers |",
         "|---|--------|------|----------|-------------------------|--------|"]
    for r in regions:
        sigs = ", ".join(r["domains"]) or "—"
        L.append(f"| {r['id']} | {r['name']} | {r['tier']} | {r['function']} | {sigs} "
                 f"| {r['layerRange'][0]}–{r['layerRange'][1]} |")
    L += ["",
          "## Tiers",
          "- **Prime** (4): Prefrontal, Orbitofrontal, Broca's, Motor Cortex — higher cognition",
          "- **Core** (9): Somatosensory, Parietal, Wernicke's, Temporal, Occipital, Visual, "
          "+ Thalamus, Cerebellum, Brain Stem (formerly Nano) — operational + subcortical",
          "",
          "## Notes",
          "- Signals per region are functional best-match; the *features* attached are the "
          "actual discriminative SAE features for that signal (candidate neural correlates).",
          "- `layer_N_labels.json` emitted alongside so the live activation monitor shows "
          "these signal labels.",
          "- Follow-up (7jz): resolve causal_connectivity synapse source/target regions "
          "against this map; per-layer (not just analysis-layer) signal distribution.",
          ""]
    OUT_MD.write_text("\n".join(L))


if __name__ == "__main__":
    main()
