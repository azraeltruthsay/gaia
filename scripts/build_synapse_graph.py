#!/usr/bin/env python3
"""
build_synapse_graph.py — assemble the unified SAE feature synapse graph.

Two signals, ONE atlas (same feature space — must be the same atlas/tag):
  - WITHIN-layer co-activation (co_activation.json): features that fire together
    in a layer → "synapses" (intra-layer edges), cofire-weighted.
  - CROSS-layer causal connectivity (causal_connectivity.json): encoder_M·decoder_N
    influence → "pathways" (inter-layer edges), strength-weighted, signed.

Nodes are (layer, feature), annotated with brain region (from the re-derived
brain_region_atlas_v2.json, A4). Output: synapse_graph.json (nodes + edges +
summary) — the graph the dashboard / analysis consumes.

  docker compose exec -T gaia-core python /gaia/GAIA_Project/scripts/build_synapse_graph.py \
    --tier core --tag CORE_IDENTITY_V3_gguf
"""
import argparse, json
from pathlib import Path


def region_lookup(region_atlas_path, tier):
    try:
        ra = json.load(open(region_atlas_path))
    except OSError:
        return lambda L: "unknown"
    regions = [r for r in ra.get("regions", []) if r.get("tier") == tier]

    def find(layer):
        for r in regions:
            lo, hi = r.get("layerRange", [0, 0])
            if lo <= layer <= hi:
                return r["name"]
        return "unknown"
    return find


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", required=True)
    ap.add_argument("--tag", required=True, help="atlas tag holding BOTH co_activation.json + causal_connectivity.json")
    ap.add_argument("--atlas", default="/shared/atlas")
    ap.add_argument("--region-atlas", default="/gaia/GAIA_Project/gaia-web/static/brain_region_atlas_v2.json")
    ap.add_argument("--min-cofire", type=int, default=0, help="extra floor on intra-layer edges")
    args = ap.parse_args()

    adir = Path(args.atlas) / args.tier / args.tag
    coact = json.load(open(adir / "co_activation.json"))
    causal = json.load(open(adir / "causal_connectivity.json"))
    region_of = region_lookup(args.region_atlas, args.tier)

    nodes = {}   # node_id -> dict
    edges = []

    def node(layer, feat):
        nid = f"L{layer}_{feat}"
        if nid not in nodes:
            nodes[nid] = {"id": nid, "layer": int(layer), "feature": int(feat),
                          "region": region_of(int(layer)), "degree": 0}
        return nid

    # Intra-layer "synapses" — feature co-firing within a layer.
    n_intra = 0
    for layer_s, info in coact.get("per_layer", {}).items():
        layer = int(layer_s)
        for p in info.get("pairs", []):
            if p["cofire"] < args.min_cofire:
                continue
            a, b = node(layer, p["i"]), node(layer, p["j"])
            edges.append({"source": a, "target": b, "kind": "coactivation",
                          "layer": layer, "cofire": p["cofire"], "jaccard": p["jaccard"]})
            nodes[a]["degree"] += 1; nodes[b]["degree"] += 1
            n_intra += 1

    # Inter-layer "pathways" — cross-layer causal influence (signed).
    n_inter = 0
    for s in causal.get("synapses", []):
        a = node(s["source_layer"], s["source_feature"])
        b = node(s["target_layer"], s["target_feature"])
        edges.append({"source": a, "target": b, "kind": "causal",
                      "weight": round(s.get("strength", 0.0), 4),
                      "polarity": s.get("type", "excitatory")})
        nodes[a]["degree"] += 1; nodes[b]["degree"] += 1
        n_inter += 1

    # Summary
    from collections import Counter
    by_region = Counter(n["region"] for n in nodes.values())
    top_hubs = sorted(nodes.values(), key=lambda n: -n["degree"])[:15]
    graph = {
        "tier": args.tier, "tag": args.tag,
        "node_count": len(nodes), "edge_count": len(edges),
        "intra_layer_edges": n_intra, "inter_layer_edges": n_inter,
        "nodes_by_region": dict(by_region),
        "top_hubs": [{"id": h["id"], "region": h["region"], "degree": h["degree"]} for h in top_hubs],
        "nodes": list(nodes.values()),
        "edges": edges,
    }
    out = adir / "synapse_graph.json"
    out.write_text(json.dumps(graph, indent=2))
    print(f"  synapse graph: {len(nodes)} nodes, {len(edges)} edges "
          f"({n_intra} co-activation + {n_inter} causal) → {out}")
    print(f"  nodes by region: {dict(by_region)}")
    print(f"  top hubs: {[(h['id'], h['region'], h['degree']) for h in top_hubs[:6]]}")


if __name__ == "__main__":
    main()
