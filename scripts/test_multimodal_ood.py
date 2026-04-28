#!/usr/bin/env python3
"""
Test Gemma4-E4B-GAIA-Core-Multimodal-v2 on out-of-distribution images.

Runs caption generation against a curated test set (in-dist primitives,
in-dist COCO val, OOD screenshots/diagrams), saves a JSON report, and
flags suspicious outputs (very short, very long, mode collapse).

Run inside gaia-study:
    docker exec gaia-study python /gaia/scripts/test_multimodal_ood.py
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image

DEFAULT_MODEL_PATH = "/models/Gemma4-E4B-GAIA-Core-Multimodal-v2"
DEFAULT_OUT = "/shared/multimodal_ood_report.json"

PROJ = "/gaia/GAIA_Project"
PRIMS_DIR = f"{PROJ}/knowledge/curricula/core-multimodal/images"
COCO_DIR = f"{PROJ}/knowledge/curricula/core-multimodal-coco/images"


def build_test_set() -> list[dict]:
    """Curate IN-distribution + OOD test images."""
    set_ = []

    # In-dist primitives (5-color smoke)
    for fname, expected in [
        ("combo_red_circle.png", {"colors": ["red"], "shapes": ["circle"]}),
        ("combo_blue_diamond.png", {"colors": ["blue"], "shapes": ["diamond"]}),
        ("combo_green_star.png", {"colors": ["green"], "shapes": ["star"]}),
        ("combo_yellow_heart.png", {"colors": ["yellow"], "shapes": ["heart"]}),
        ("combo_purple_square.png", {"colors": ["purple"], "shapes": ["square"]}),
    ]:
        p = Path(PRIMS_DIR) / fname
        if p.exists():
            set_.append({"id": f"prim:{fname}", "path": str(p),
                         "category": "in_dist_primitive", "expect": expected})

    # In-dist COCO val (random sample of 3, deterministic by lexicographic order)
    coco_files = sorted(Path(COCO_DIR).glob("COCO_val2014_*.jpg"))[:3]
    for p in coco_files:
        set_.append({"id": f"coco:{p.name}", "path": str(p),
                     "category": "in_dist_coco", "expect": {}})

    # OOD: GAIA's own brain diagram
    brain = Path(PROJ) / "gaia-web/static/brain.png"
    if brain.exists():
        set_.append({"id": "ood:brain.png", "path": str(brain),
                     "category": "ood_diagram",
                     "expect": {"keywords": ["brain", "diagram"]}})

    return set_


def load_model(model_path: str):
    from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

    print(f"Loading {model_path} (NF4)...", flush=True)
    t0 = time.time()
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        # Don't quantise the towers — Gemma4ClippableLinear has nested
        # weight + QAT calibration buffers that NF4 mangles.
        llm_int8_skip_modules=[
            "vision_tower", "audio_tower",
            "embed_vision", "embed_audio", "lm_head",
        ],
    )
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    print(f"Loaded in {time.time()-t0:.1f}s", flush=True)
    return model, processor


def caption(model, processor, image_path: str, prompt: str = "Describe this image.") -> dict:
    """Generate a caption. Returns timing + output."""
    img = Image.open(image_path).convert("RGB")

    img_tok = getattr(processor, "image_token", None) or "<|image|>"
    # Mirror the training-time format exactly: assistant turn (not model),
    # bare image placeholder before the instruction. The processor expands
    # the placeholder into boi + N image soft tokens + eoi at call time.
    text = f"<|turn>user<turn|>\n{img_tok}\n{prompt}\n<|turn>assistant<turn|>\n"

    inputs = processor(text=text, images=[img], return_tensors="pt").to(model.device)

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
            temperature=1.0,
            repetition_penalty=1.1,
        )
    elapsed = time.time() - t0

    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    decoded = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
    return {
        "caption": decoded.strip(),
        "elapsed_s": round(elapsed, 2),
        "new_tokens": int(new_tokens.shape[0]),
    }


def score(case: dict, caption_text: str) -> dict:
    """Lightweight quality heuristics."""
    cap_low = caption_text.lower()
    expect = case.get("expect") or {}

    color_hit = None
    if "colors" in expect:
        color_hit = any(c in cap_low for c in expect["colors"])
    shape_hit = None
    if "shapes" in expect:
        shape_hit = any(s in cap_low for s in expect["shapes"])
    keyword_hit = None
    if "keywords" in expect:
        keyword_hit = any(k in cap_low for k in expect["keywords"])

    flags = []
    if len(caption_text) < 5:
        flags.append("very_short")
    if len(caption_text) > 300:
        flags.append("very_long")
    if caption_text.count(caption_text[:20]) > 1 if len(caption_text) >= 20 else False:
        flags.append("possible_repetition")
    if any(w in cap_low for w in ["#000", "0x", "rgb(", "hex"]):
        flags.append("hex_code_mode")  # Pretrain prior leaking back

    return {
        "color_correct": color_hit,
        "shape_correct": shape_hit,
        "keyword_correct": keyword_hit,
        "flags": flags,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--model-path", default=DEFAULT_MODEL_PATH,
                    help="Path to the multimodal model to test")
    args = ap.parse_args()

    test_set = build_test_set()
    if args.limit > 0:
        test_set = test_set[:args.limit]
    print(f"Test set: {len(test_set)} images", flush=True)
    for c in test_set:
        print(f"  - {c['id']} [{c['category']}]", flush=True)

    model, processor = load_model(args.model_path)

    results = []
    for case in test_set:
        print(f"\n>>> {case['id']}", flush=True)
        try:
            r = caption(model, processor, case["path"])
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)
            results.append({**case, "error": str(e)})
            continue
        s = score(case, r["caption"])
        print(f"  caption: {r['caption'][:200]}", flush=True)
        print(f"  scores:  {s}", flush=True)
        results.append({**case, **r, "scores": s})

    summary = {
        "model": args.model_path,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_cases": len(results),
        "by_category": {},
        "results": results,
    }
    by_cat = summary["by_category"]
    for r in results:
        cat = r.get("category", "?")
        by_cat.setdefault(cat, {"count": 0, "errors": 0, "color_hits": 0,
                                 "shape_hits": 0, "keyword_hits": 0})
        by_cat[cat]["count"] += 1
        if "error" in r:
            by_cat[cat]["errors"] += 1
            continue
        s = r.get("scores", {})
        if s.get("color_correct") is True:
            by_cat[cat]["color_hits"] += 1
        if s.get("shape_correct") is True:
            by_cat[cat]["shape_hits"] += 1
        if s.get("keyword_correct") is True:
            by_cat[cat]["keyword_hits"] += 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nReport written to {out_path}", flush=True)
    print("Summary:", json.dumps(by_cat, indent=2), flush=True)


if __name__ == "__main__":
    main()
