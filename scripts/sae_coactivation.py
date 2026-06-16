#!/usr/bin/env python3
"""
sae_coactivation.py — within-layer feature CO-ACTIVATION (synapse-mapping foundation).

Causal connectivity (sae_causal_connectivity.py) maps CROSS-layer influence
(encoder_M·decoder_N) from the SAE weights alone. Co-activation is the
complementary, data-driven signal: WHICH features fire TOGETHER on the same
inputs, WITHIN a layer. That co-firing graph is the precursor to synapse
mapping — pairs that consistently co-activate are candidate functional synapses.

Method: reload the atlas SAEs, re-encode the corpus through each layer's top-k
SAE → binary feature-active matrix [n_samples, n_features] → co-occurrence
counts + Jaccard. Save the top co-activating pairs per layer.

Capture is CPU via gaia_cpp (GGUF) so it never contends with the GPU-resident
model. Run:
  docker compose exec -T gaia-core python /gaia/GAIA_Project/scripts/sae_coactivation.py \
    --tier core --tag CORE_IDENTITY_V3_gguf --gguf /models/core.gguf \
    --corpus /gaia/GAIA_Project/knowledge/curricula/sae_atlas/corpus.json --topk 32
"""
import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GAIA.SAE.Coact")


def main():
    ap = argparse.ArgumentParser(description="SAE within-layer co-activation (synapse precursor)")
    ap.add_argument("--tier", required=True)
    ap.add_argument("--tag", required=True, help="atlas tag under /shared/atlas/<tier>/")
    ap.add_argument("--gguf", required=True, help="GGUF path for CPU re-encode via gaia_cpp")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--atlas", default="/shared/atlas")
    ap.add_argument("--topk", type=int, default=32, help="top-k to restore on loaded SAEs")
    ap.add_argument("--top-pairs", type=int, default=200, help="top co-activating pairs to keep per layer")
    args = ap.parse_args()

    import torch
    from gaia_engine.cpp import gaia_cpp
    from gaia_engine.sae_trainer import SAETrainer

    atlas_dir = Path(args.atlas) / args.tier / args.tag
    items = json.load(open(args.corpus))
    prompts = [it["text"] for it in items]
    strata = [it.get("stratum", "") for it in items]

    # Reload the atlas SAEs; layers come from the saved meta.
    meta = json.load(open(atlas_dir / "meta.json"))
    layers = meta["layers"]
    logger.info("Co-activation: %s/%s — layers %s, %d prompts", args.tier, args.tag, layers, len(prompts))

    backend = gaia_cpp.LlamaCppBackend(model_path=args.gguf, n_gpu_layers=0,
                                       capture_layers=layers, n_ctx=2048)
    trainer = SAETrainer(model=backend, tokenizer=None, device="cpu")
    trainer.load_atlas(str(atlas_dir))
    for sae in trainer.saes.values():
        sae.k = args.topk          # restore top-k sparsity on the reloaded SAEs
        sae.eval()

    logger.info("Re-encoding corpus (CPU)...")
    trainer.record_activations_gguf(prompts, layers=layers, backend=backend)

    out = {"tier": args.tier, "tag": args.tag, "layers": layers,
           "topk": args.topk, "n_samples": None, "per_layer": {}}

    with torch.no_grad():
        for layer in layers:
            sae = trainer.saes.get(layer)
            acts = trainer.activations.get(layer, [])
            if sae is None or not acts:
                continue
            X = torch.cat(acts, dim=0).float()                       # [N, n_embd]
            Xn = (X - sae._norm_mean) / sae._norm_std
            F = sae.get_feature_activations(Xn)                       # [N, n_features] top-k sparse
            A = (F > 1e-6).float()                                    # binary active mask
            n_samples, n_feat = A.shape
            out["n_samples"] = int(n_samples)
            # Co-occurrence counts C[i,j] = # samples where features i AND j both fire.
            C = (A.t() @ A)                                           # [n_feat, n_feat]
            counts = A.sum(dim=0)                                     # per-feature fire count
            # Jaccard = |i∩j| / |i∪j|; mask diagonal + require min co-fire.
            diag = torch.arange(n_feat)
            C[diag, diag] = 0
            union = counts.unsqueeze(0) + counts.unsqueeze(1) - C
            J = torch.where(union > 0, C / union, torch.zeros_like(C))
            # Pull the strongest co-activating pairs (upper triangle). Rank by
            # CO-FIRE COUNT (robust synapse candidates fire together OFTEN), not
            # raw Jaccard — pure Jaccard is dominated by rare features that
            # coincide on a single sample (jaccard=1.0, cofire=1 = noise). Floor
            # at MIN_COFIRE so we keep only repeatedly co-firing pairs.
            MIN_COFIRE = max(2, int(0.03 * n_samples))   # ~3% of the corpus
            iu = torch.triu_indices(n_feat, n_feat, offset=1)
            cvals = C[iu[0], iu[1]]
            elig = (cvals >= MIN_COFIRE).nonzero(as_tuple=True)[0]
            keep = min(args.top_pairs, elig.numel())
            if keep == 0:
                out["per_layer"][str(layer)] = {"pairs": [], "active_features": int((counts > 0).sum()),
                                                 "min_cofire": MIN_COFIRE}
                logger.info("  L%d: %d active feats, NO pairs >= %d cofire (corpus too small for this layer)",
                            layer, int((counts > 0).sum()), MIN_COFIRE)
                continue
            elig_c = cvals[elig]
            top = torch.topk(elig_c, keep)
            pairs = []
            for k in top.indices.tolist():
                idx = elig[k].item()
                i, j = int(iu[0][idx]), int(iu[1][idx])
                pairs.append({"i": i, "j": j, "cofire": int(C[i, j]), "jaccard": round(float(J[i, j]), 4)})
            out["per_layer"][str(layer)] = {
                "active_features": int((counts > 0).sum()),
                "mean_cofire_degree": round(float((C > 0).float().sum(dim=0).mean()), 2),
                "pairs": pairs,
            }
            logger.info("  L%d: %d active feats, top cofire=%d (jaccard=%.3f), %d pairs >= %d cofire",
                        layer, int((counts > 0).sum()), int(top.values[0]),
                        pairs[0]["jaccard"], keep, MIN_COFIRE)

    out_path = atlas_dir / "co_activation.json"
    out_path.write_text(json.dumps(out, indent=2))
    logger.info("Saved co-activation → %s", out_path)


if __name__ == "__main__":
    main()
