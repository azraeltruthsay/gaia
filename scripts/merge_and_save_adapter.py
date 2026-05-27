#!/usr/bin/env python3
"""Standalone adapter merge + save (workaround for OOM in training script).

The training script's merge+save fails for large curricula because the
dataset object (19K samples × tokenized tensors) is still in memory at
merge time, adding 5-10 GB pressure on top of the base + merged + base-for-
graft state dicts. This script does ONLY the merge step, no training,
so the dataset never enters memory.

Usage:
    python merge_and_save_adapter.py \\
        --base /models/google/gemma-4-E4B \\
        --adapter /models/lora_adapters/gemma4_e4b_core_multimodal_core2x_v2 \\
        --out /models/Gemma4-E4B-GAIA-Core-Multimodal-CORE2X_V2
"""
import argparse
import gc
import json
import logging
import os
import sys
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    base_path = args.base
    adapter_path = args.adapter
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading base model %s with NF4...", base_path)
    from transformers import AutoModelForCausalLM, AutoConfig, AutoProcessor, BitsAndBytesConfig
    from peft import PeftModel

    # Skip-modules covers Gemma 4's vision/audio towers AND Qwen3-VL's
    # visual / vision_model so NF4 doesn't double-quantize tower weights.
    skip_modules = [
        "lm_head",
        # Gemma 4 names
        "vision_tower", "audio_tower", "embed_vision", "embed_audio",
        # Qwen3-VL / Qwen2-VL names
        "visual", "vision_model",
    ]
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=skip_modules,
    )
    # Auto-detect VL architecture — Qwen3VLForConditionalGeneration and
    # similar aren't registered for AutoModelForCausalLM, so use
    # AutoModelForImageTextToText when config.model_type indicates a VL
    # family. Mirrors the engine + training-script fix.
    _is_vl = False
    try:
        _cfg = AutoConfig.from_pretrained(base_path, trust_remote_code=True)
        _mt = (getattr(_cfg, "model_type", "") or "").lower()
        _is_vl = "vl" in _mt or "vision" in _mt
    except Exception:
        pass
    _load_kwargs = dict(
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model = None
    if _is_vl:
        try:
            from transformers import AutoModelForImageTextToText
            model = AutoModelForImageTextToText.from_pretrained(base_path, **_load_kwargs)
            log.info("Loaded base via AutoModelForImageTextToText (VL arch)")
        except Exception as _vl_err:
            log.warning("VL load failed (%s) — falling back to AutoModelForCausalLM", _vl_err)
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(base_path, **_load_kwargs)
    log.info("Base loaded: %.2f GB VRAM", torch.cuda.memory_allocated()/1024**3)

    log.info("Loading adapter %s...", adapter_path)
    model = PeftModel.from_pretrained(model, adapter_path)
    log.info("Adapter applied: %.2f GB VRAM", torch.cuda.memory_allocated()/1024**3)

    log.info("Merging LoRA...")
    merged = model.merge_and_unload()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    log.info("Merge complete: %.2f GB VRAM", torch.cuda.memory_allocated()/1024**3)

    # Dequantize Linear4bit → bf16 nn.Linear (GPU → CPU per layer)
    log.info("Dequantizing Linear4bit modules...")
    import torch.nn as nn
    import bitsandbytes as bnb
    import bitsandbytes.functional as bnb_f

    to_replace = []
    for name, module in merged.named_modules():
        for attr_name, child in list(module.named_children()):
            if isinstance(child, bnb.nn.Linear4bit):
                to_replace.append((module, attr_name, child))
    log.info("Found %d Linear4bit modules", len(to_replace))

    for i, (parent, attr_name, lin4) in enumerate(to_replace):
        weight_gpu = lin4.weight.data
        qstate = lin4.weight.quant_state
        if weight_gpu.device.type != "cuda":
            weight_gpu = weight_gpu.cuda()
        dequant = bnb_f.dequantize_4bit(weight_gpu, qstate).to(torch.bfloat16)
        new_linear = nn.Linear(
            lin4.in_features, lin4.out_features,
            bias=lin4.bias is not None,
            dtype=torch.bfloat16, device="cpu",
        )
        with torch.no_grad():
            new_linear.weight.copy_(dequant.cpu())
            if lin4.bias is not None:
                new_linear.bias.copy_(lin4.bias.detach().to(torch.bfloat16).cpu())
        setattr(parent, attr_name, new_linear)
        del lin4, dequant, weight_gpu
        if (i + 1) % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()
            log.info("  %d/%d dequantized", i+1, len(to_replace))
    gc.collect()
    torch.cuda.empty_cache()
    log.info("All %d dequantized, weights on CPU bf16", len(to_replace))

    # Collect state dict. Gemma 4 ties lm_head.weight to embed_tokens.weight
    # (shared underlying tensor). safetensors refuses to save aliased tensors.
    # Detach + clone to break the alias so both can be serialized.
    log.info("Collecting merged state dict...")
    merged_sd = {}
    seen_ptrs = set()
    for k, v in merged.state_dict().items():
        v = v.detach()
        ptr = v.data_ptr()
        if ptr in seen_ptrs:
            v = v.clone()
        else:
            seen_ptrs.add(ptr)
        merged_sd[k] = v.contiguous()
    log.info("merged keys: %d", len(merged_sd))

    # Tower graft: load base state dict, overwrite tower keys with base values
    # so the bf16 NF4-round-tripped tower weights are replaced by the
    # original base tower weights (NF4 round-trip loses calibration).
    log.info("Loading base state dict for tower graft (CPU)...")
    from safetensors.torch import load_file
    base_files = sorted(Path(base_path).glob("model*.safetensors"))
    base_sd = {}
    for bf in base_files:
        sd = load_file(str(bf))
        base_sd.update(sd)
    log.info("Base state dict keys: %d", len(base_sd))

    tower_keys = [k for k in base_sd
                  if any(t in k for t in ("vision_tower", "audio_tower",
                                          "embed_vision", "embed_audio"))]
    log.info("Tower keys to graft: %d", len(tower_keys))

    grafted = 0
    for tk in tower_keys:
        if tk in merged_sd:
            merged_sd[tk] = base_sd[tk].clone().contiguous()
            grafted += 1
    log.info("Grafted %d tower keys; dropping base_sd to save RAM", grafted)
    del base_sd
    gc.collect()

    # Drop merged model to free RAM (state dict has all weights)
    del merged
    gc.collect()
    torch.cuda.empty_cache()
    log.info("Pre-write RAM check...")
    import psutil
    p = psutil.Process()
    log.info("  process RSS: %.2f GB", p.memory_info().rss / 1024**3)

    # Save
    out_file = out_dir / "model.safetensors"
    log.info("Writing %s (%d keys)...", out_file, len(merged_sd))
    from safetensors.torch import save_file
    save_file(merged_sd, str(out_file), metadata={"format": "pt"})
    log.info("Wrote %.2f GB", out_file.stat().st_size / 1024**3)

    # Copy config + tokenizer/processor files from base.
    #
    # GAIA_Project-clp: include the VL-side processor configs and the
    # legacy tokenizer files (merges.txt / vocab.json). Without these,
    # AutoProcessor.from_pretrained() on the merged dir fails with
    # "Can't load image processor for ...", the engine flips
    # has_vision=False, and the NF4 load path falls through to
    # AutoModelForCausalLM — which doesn't support Qwen3VL — and the
    # worker exits. Each file is optional: text-only bases simply
    # won't have them, and the `if src.exists()` guard skips silently.
    log.info("Copying config + tokenizer/processor files from base...")
    import shutil
    _files_to_copy = (
        # Core config
        "config.json", "generation_config.json",
        # Tokenizer
        "tokenizer_config.json", "tokenizer.json",
        "special_tokens_map.json",
        # Legacy tokenizer (BPE merges / vocab) — needed by some
        # processor classes even when tokenizer.json is present
        "merges.txt", "vocab.json",
        # Processor configs (text + vision + video)
        "processor_config.json",
        "preprocessor_config.json",        # VL image processor (clp)
        "video_preprocessor_config.json",  # VL video processor (clp)
        # Chat template (separate from tokenizer_config.json on some VL bases)
        "chat_template.json",              # clp
    )
    copied: list[str] = []
    missing: list[str] = []
    for fname in _files_to_copy:
        src = Path(base_path) / fname
        if src.exists():
            shutil.copy(src, out_dir / fname)
            log.info("  copied %s", fname)
            copied.append(fname)
        else:
            missing.append(fname)
    log.info(
        "Processor copy summary: %d copied, %d not present in base",
        len(copied), len(missing),
    )
    if missing:
        log.debug("  base did not have: %s", ", ".join(missing))

    log.info("Done. Output: %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
