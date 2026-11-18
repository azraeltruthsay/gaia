#!/usr/bin/env python3
"""What a LoRA adapter changes inside a model, read through its SAE atlas (0hf0).

The model is loaded once (NF4, as the atlas was recorded) with the adapter
attached; every prompt runs with the adapter disabled ("bare") and enabled.
Activations are taken exactly as the atlas recorded them (SAETrainer's
ChatFormatter prompt, same system prompt, the last token, the SAE's own
normalization) and projected through each atlas layer.

Two prompt sets:
  focus     the adapter's held-out turns: the case it was trained for
  ordinary  the atlas corpus (134 prompts, 10 strata): what it would do to
            turns it never sees today, i.e. the cost of ever merging it

Per layer and set: cosine between bare and adapter feature vectors, overlap
of the top-20 features, and the features gained or lost most often, with
their labels. Writes JSON; prints a short table.

Usage (gaia-study, GPU free: GAIA in meditation):
  python scripts/sae_adapter_diff.py --model /models/prime \\
      --adapter /models/lora_adapters/tier1_global/prime_thinking_v3 \\
      --atlas /shared/atlas/prime/PRIME_V2B --out /shared/atlas/prime/PRIME_V2B/diff_prime_thinking_v3.json
"""
import argparse
import collections
import json
import statistics
import sys
import time
from pathlib import Path

import torch

for p in [Path(__file__).resolve().parent.parent / "gaia-engine", Path("/gaia/GAIA_Project/gaia-engine")]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

SYSTEM = "You are GAIA, a sovereign AI created by Azrael."  # as sae_atlas_snapshot recorded
CORPUS = "/gaia/GAIA_Project/knowledge/curricula/sae_atlas/corpus.json"
TOP = 20


def load(model_path, adapter_path):
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb,
                                                     device_map="cuda", trust_remote_code=True)
    except (ValueError, KeyError, OSError):
        model = AutoModelForImageTextToText.from_pretrained(model_path, quantization_config=bnb,
                                                            device_map="cuda", trust_remote_code=True)
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tok


def load_atlas(path):
    from gaia_engine.sae_trainer import SparseAutoencoder
    saes, labels = {}, {}
    for f in sorted(Path(path).glob("layer_*.pt"), key=lambda p: int(p.stem.split("_")[1])):
        layer = int(f.stem.split("_")[1])
        ck = torch.load(f, map_location="cuda", weights_only=True)
        sae = SparseAutoencoder(ck["hidden_size"], ck["num_features"])
        for name in ("encoder", "decoder"):
            getattr(sae, name).weight.data = ck[f"{name}_weight"].cuda()
            getattr(sae, name).bias.data = ck[f"{name}_bias"].cuda()
        sae._norm_mean = ck.get("norm_mean", torch.zeros(ck["hidden_size"])).cpu().float()
        sae._norm_std = ck.get("norm_std", torch.ones(ck["hidden_size"])).cpu().float()
        saes[layer] = sae.cuda().eval()
        labels[layer] = ck.get("labels", {}) or {}
    return saes, labels


def prompt_ids(tok, text):
    from gaia_engine.core import ChatFormatter
    fmt = ChatFormatter(tok)
    full = (fmt.format_system(SYSTEM) + "\n" + fmt.format_message("user", text) + "\n"
            + fmt.assistant_prefix(enable_thinking=True))
    return tok.encode(full, return_tensors="pt").cuda()


@torch.no_grad()
def features(model, ids, saes):
    out = model(input_ids=ids, output_hidden_states=True)
    feats = {}
    for layer, sae in saes.items():
        hs = out.hidden_states[layer][0, -1].float().cpu()
        feats[layer] = sae.get_feature_activations(((hs - sae._norm_mean) / sae._norm_std).cuda()).float().cpu()
    return feats, out.logits[0, -1].float().cpu()


def compare(rows, labels):
    """rows: [(bare_feats, adapter_feats)] -> per-layer summary."""
    report = {}
    for layer in rows[0][0]:
        cos, overlap = [], []
        gained, lost = collections.Counter(), collections.Counter()
        for bare, ad in rows:
            b, a = bare[layer], ad[layer]
            cos.append(torch.nn.functional.cosine_similarity(b, a, dim=0).item())
            tb, ta = set(b.topk(TOP).indices.tolist()), set(a.topk(TOP).indices.tolist())
            overlap.append(len(tb & ta) / TOP)
            gained.update(ta - tb)
            lost.update(tb - ta)

        def named(c):
            return [{"feature": f, "prompts": n, "label": labels[layer].get(str(f), "")}
                    for f, n in c.most_common(8)]
        report[layer] = {"cosine_median": round(statistics.median(cos), 3),
                         "cosine_min": round(min(cos), 3),
                         "top20_overlap_median": round(statistics.median(overlap), 2),
                         "gained": named(gained), "lost": named(lost)}
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--atlas", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="prompts per set (0 = all); for a smoke run")
    args = ap.parse_args()

    t0 = time.time()
    focus = json.load(open(Path(args.adapter) / "holdout_users.json"))
    corpus = json.load(open(CORPUS))
    sets = {"focus": [(u, "focus") for u in focus],
            "ordinary": [(c["text"], c.get("stratum", "")) for c in corpus]}
    if args.limit:
        sets = {k: v[:args.limit] for k, v in sets.items()}

    model, tok = load(args.model, args.adapter)
    saes, labels = load_atlas(args.atlas)
    print(f"loaded in {time.time() - t0:.0f}s; atlas layers {sorted(saes)}", flush=True)

    # The adapter must actually be attached: enabled and disabled logits differ.
    ids = prompt_ids(tok, "What is 17 times 23?")
    _, on = features(model, ids, saes)
    with model.disable_adapter():
        _, off = features(model, ids, saes)
    delta = (on - off).abs().max().item()
    print(f"adapter logit delta {delta:.4f}", flush=True)
    if delta < 1e-3:
        print("adapter has no effect: not attached to this model's modules", flush=True)
        return 1

    result = {"model": args.model, "adapter": args.adapter, "atlas": args.atlas,
              "logit_delta_check": round(delta, 4), "sets": {}}
    for name, prompts in sets.items():
        rows, by_stratum = [], collections.defaultdict(list)
        for i, (text, stratum) in enumerate(prompts, 1):
            ids = prompt_ids(tok, text)
            ad, _ = features(model, ids, saes)
            with model.disable_adapter():
                bare, _ = features(model, ids, saes)
            rows.append((bare, ad))
            by_stratum[stratum].append((bare, ad))
            if i % 20 == 0:
                print(f"{name} {i}/{len(prompts)}", flush=True)
        entry = {"n": len(rows), "layers": compare(rows, labels)}
        if name == "ordinary":
            deepest = max(saes)
            entry["strata_cosine_median_deepest_layer"] = {
                s: compare(r, labels)[deepest]["cosine_median"] for s, r in sorted(by_stratum.items())}
        result["sets"][name] = entry

    result["elapsed_s"] = round(time.time() - t0)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=1)

    print(f"\n{'layer':>5} | {'focus cos':>9} {'top20':>5} | {'ordinary cos':>12} {'top20':>5}")
    for layer in sorted(saes):
        f, o = result["sets"]["focus"]["layers"][layer], result["sets"]["ordinary"]["layers"][layer]
        print(f"{layer:>5} | {f['cosine_median']:>9} {f['top20_overlap_median']:>5} | "
              f"{o['cosine_median']:>12} {o['top20_overlap_median']:>5}")
    if "strata_cosine_median_deepest_layer" in result["sets"]["ordinary"]:
        print("ordinary, deepest layer, by stratum:", result["sets"]["ordinary"]["strata_cosine_median_deepest_layer"])
    print(f"written {args.out} ({result['elapsed_s']}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
