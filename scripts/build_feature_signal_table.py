#!/usr/bin/env python3
"""build_feature_signal_table.py — A4 step 4: the feature↔cognitive-signal table.

The measurement deliverable of sae_atlas_build_plan.md §4. Reads the committed
atlases and, for each cognitive-signal stratum, finds the SAE features that fire
DISCRIMINATIVELY (selective — high on that stratum, low elsewhere = a real neural
correlate, not raw magnitude where loud strata dominate). Labels each from its
activation-maximizing exemplar prompt. Adds a stratum-level cross-path (Q4)
comparison.

Pure JSON synthesis over each safetensors atlas's `prompt_analyses` (134 stratum-
tagged prompts × top-k features at the analysis layer) — NO model load, CPU-only.

CAVEATS (stated in the output):
  - Single analysis layer per model (Core L41, Prime L35) — the layer the atlas
    keyed on; deeper per-layer selectivity is future work.
  - safetensors and gguf SAEs are SEPARATELY trained → feature indices are NOT
    comparable across paths. Cross-path diff is therefore STRATUM-LEVEL (is the
    signal as sharply represented after Q4), not feature-identity. Feature
    alignment via decoder cosine is a heavier follow-up.

  docker exec gaia-study python3 /gaia/GAIA_Project/scripts/build_feature_signal_table.py
"""
import json
from collections import defaultdict
from pathlib import Path

ATLAS = Path("/shared/atlas")
OUT_DIR = Path("/gaia/GAIA_Project/knowledge/blueprints")
MODELS = [
    {"tier": "core",  "st": "CORE_IDENTITY_V3",  "gguf": "CORE_IDENTITY_V3_gguf"},
    {"tier": "prime", "st": "PRIME_ABLITERATED", "gguf": "PRIME_ABLITERATED_gguf"},
]
TOP_N = 12  # discriminative features reported per signal

# Human-readable gloss for each stratum (the cognitive signal it elicits).
SIGNAL_GLOSS = {
    "coherence_contradiction": "coherence / contradiction (Samvega, consistency detector)",
    "curiosity_gap":           "curiosity / knowledge gap (curiosity drive)",
    "competence_problem":      "competence / problem-solving (competence drive, tools)",
    "identity_self":           "identity / self-reference (self-model)",
    "affect_laden":            "affect-laden vs neutral (the felt register)",
    "neutral_factual":         "neutral factual (flat baseline)",
    "deliberation":            "deliberation / trade-off reasoning (Council)",
    "register_chitchat":       "register: chitchat / greeting ('how are you' lives here)",
    "register_technical":      "register: technical",
    "spatial_reasoning":       "spatial / contextual reasoning",
}


def feature_profiles(prompt_analyses):
    """feature index -> {stratum -> [strengths]} and -> exemplar (prompt, strength)."""
    by_feat = defaultdict(lambda: defaultdict(list))
    exemplar = {}  # feat -> (strength, stratum, prompt)
    for p in prompt_analyses:
        s, prompt = p["domain"], p.get("prompt", "")
        for f in p.get("top_features", []):
            idx, strg = f["index"], float(f["strength"])
            by_feat[idx][s].append(strg)
            if idx not in exemplar or strg > exemplar[idx][0]:
                exemplar[idx] = (strg, s, prompt)
    return by_feat, exemplar


def discriminative_table(prompt_analyses):
    by_feat, exemplar = feature_profiles(prompt_analyses)
    # Per feature: mean strength per stratum, top stratum, selectivity, score.
    per_signal = defaultdict(list)
    n_selective = 0
    for idx, strata in by_feat.items():
        means = {s: sum(v) / len(v) for s, v in strata.items()}
        total = sum(means.values())
        top_s = max(means, key=means.get)
        selectivity = means[top_s] / total if total else 0.0  # 1.0 = fires on one stratum
        if len(strata) == 1:
            n_selective += 1
        peak, ex_s, ex_prompt = exemplar[idx]
        score = means[top_s] * selectivity  # strong AND selective
        per_signal[top_s].append({
            "index": idx,
            "peak_strength": round(peak, 3),
            "mean_on_signal": round(means[top_s], 3),
            "selectivity": round(selectivity, 3),
            "n_strata": len(strata),
            "score": round(score, 3),
            "exemplar": ex_prompt[:90],
        })
    for s in per_signal:
        per_signal[s].sort(key=lambda f: -f["score"])
        per_signal[s] = per_signal[s][:TOP_N]
    return per_signal, len(by_feat), n_selective


def gguf_signal_strength(gguf_analysis):
    """Per-stratum peak feature strength from the gguf atlas's domain_features
    (a separate SAE — index-incomparable, so we compare STRENGTH profiles)."""
    df = gguf_analysis.get("domain_features", {}) or {}
    out = {}
    for s, feats in df.items():
        if isinstance(feats, list) and feats:
            out[s] = round(max(float(f.get("strength", 0.0)) for f in feats), 3)
        else:
            out[s] = 0.0
    return out


def main():
    report = {"_meta": {
        "deliverable": "feature↔cognitive-signal table (sae_atlas_build_plan.md §4)",
        "method": "discriminative selectivity over per-stratum prompt_analyses; "
                  "activation-maximizing exemplar labels; stratum-level cross-path (Q4) diff",
        "caveats": [
            "single analysis layer per model (Core L41, Prime L35)",
            "safetensors vs gguf SAEs are separately trained → feature indices NOT "
            "comparable across paths; cross-path diff is stratum-level, not feature-identity",
        ],
    }, "models": {}}

    for m in MODELS:
        st = json.load(open(ATLAS / m["tier"] / m["st"] / "analysis.json"))
        gg = json.load(open(ATLAS / m["tier"] / m["gguf"] / "analysis.json"))
        layer = st.get("analysis_layer")
        per_signal, n_feat, n_sel = discriminative_table(st.get("prompt_analyses", []))
        gg_strength = gguf_signal_strength(gg)
        report["models"][m["tier"]] = {
            "analysis_layer": layer,
            "n_features_seen": n_feat,
            "n_perfectly_selective": n_sel,
            "pct_selective": round(100 * n_sel / n_feat, 1) if n_feat else 0.0,
            "signals": per_signal,
            "gguf_peak_strength_by_signal": gg_strength,
        }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "feature_cognitive_signal_table.json").write_text(json.dumps(report, indent=2))
    _write_markdown(report)
    print("Wrote feature_cognitive_signal_table.{json,md} to", OUT_DIR)


def _write_markdown(report):
    L = ["# Feature↔Cognitive-Signal Table",
         "",
         "> The SAE-atlas measurement deliverable (`sae_atlas_build_plan.md` §4). "
         "For each cognitive-signal stratum, the SAE features that fire "
         "**discriminatively** (selective — high on that stratum, low elsewhere = a "
         "candidate neural correlate). Labels are activation-maximizing exemplars.",
         ""]
    meta = report["_meta"]
    L += ["**Method:** " + meta["method"], "",
          "**Caveats:**"] + [f"- {c}" for c in meta["caveats"]] + [""]
    for tier, d in report["models"].items():
        L += [f"## {tier.capitalize()} — analysis layer L{d['analysis_layer']}", "",
              f"**Selectivity headline:** {d['n_perfectly_selective']}/{d['n_features_seen']} "
              f"features seen fire in exactly ONE stratum "
              f"(**{d['pct_selective']}% perfectly selective**) — cognitive signals "
              f"have discriminative neural correlates at this layer.", ""]
        # Cross-path note: the two SAEs are separately trained with different
        # activation scales (Core safetensors peaks ~3-4 vs gguf ~7-9; Prime
        # safetensors ~40-53 vs gguf ~9-11), so a raw strength ratio is NOT a
        # Q4-survival verdict — it mostly reflects SAE scale. Report both numbers
        # honestly; a rigorous Q4 diff needs decoder-cosine feature alignment.
        L += ["> Cross-path (Q4): safetensors and gguf SAEs are separately scaled — "
              "the peak-strength numbers below are shown for reference, NOT as a "
              "distortion verdict (a rigorous Q4-survival result needs decoder "
              "alignment; filed as follow-up).", ""]
        for sig, feats in sorted(d["signals"].items()):
            if not feats:
                continue
            gloss = SIGNAL_GLOSS.get(sig, sig)
            gg = d["gguf_peak_strength_by_signal"].get(sig)
            st_peak = max((f["peak_strength"] for f in feats), default=0.0)
            q4 = f" · peaks: {st_peak} safetensors / {gg} gguf (scale-dependent)" if gg is not None else ""
            L += [f"### {sig}", f"*{gloss}*{q4}", "",
                  "| feature | peak | mean | selectivity | #strata | exemplar prompt |",
                  "|--------:|-----:|-----:|:-----------:|:-------:|-----------------|"]
            for f in feats:
                L.append(f"| `{f['index']}` | {f['peak_strength']} | {f['mean_on_signal']} "
                         f"| {f['selectivity']} | {f['n_strata']} | {f['exemplar']} |")
            L.append("")
    (OUT_DIR / "feature_cognitive_signal_table.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
