#!/usr/bin/env python3
"""
Train Gemma 4 E4B Core with a multimodal curriculum.

Fixes the previous pipeline's two structural problems:

1.  Image inputs were fed to the model with double-wrapped image tokens
    (our ChatFormatter was emitting "<|image>…<|image|>…<image|>" around
    the processor's own wrapping), producing garbage outputs. Here we
    build the prompt manually with Gemma 4 turn tags and emit ONE bare
    "<|image|>" per image — the Gemma4Processor wraps and expands it
    into boi + N image soft tokens + eoi at call time.

2.  Our previous LoRA→safetensors merge step silently dropped the
    vision_tower and audio_tower weights because PEFT's merge-and-unload
    only keeps keys that LoRA touched or explicitly saves. The saved
    checkpoint had flat tower naming with no QAT calibration buffers,
    so at inference load Gemma4ClippableLinear couldn't find its
    `.linear.weight` under each wrapped layer and the towers got
    detached. This script's post-training save graft copies every
    vision_tower / audio_tower / embed_vision / embed_audio tensor from
    the base model verbatim, preserving the QAT layout Gemma 4 expects.

Training targets:
  - Language model attention + MLP (LoRA r=8)
  - Towers frozen (they're pretrained and work; our job is to teach
    the LM to read them correctly)

Expected output:
    /models/lora_adapters/gemma4_e4b_core_multimodal_v1/
    /models/Gemma4-E4B-GAIA-Core-Multimodal-v1/

Run inside gaia-study (has torch/transformers/peft/bitsandbytes).
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("train_core_mm")

# ── Paths ───────────────────────────────────────────────────────────────────
_IN_CONTAINER = os.path.exists("/models/google/gemma-4-E4B")
_BASE = "/models" if _IN_CONTAINER else "/gaia/gaia-instance/gaia-models"
_PROJ = "/gaia/GAIA_Project"

BASE_MODEL = f"{_BASE}/Gemma4-E4B-GAIA-Unified-v5-Multimodal"
TEXT_CURRICULUM = f"{_PROJ}/knowledge/curricula/core-multimodal/train.jsonl"
# Switched from programmatic primitives (solid colors + shapes) to real
# COCO image-caption pairs for v2. Programmatic curriculum plateaued at
# 17% color accuracy because Gemma 4's pretraining prior for hex-code
# responses dominated our 406 synthetic pairs. Real COCO captions sit
# closer to Gemma 4's text distribution and give the cross-modal bridge
# thousands of natural image → natural-language mappings.
VISION_CURRICULUM = f"{_PROJ}/knowledge/curricula/core-multimodal-coco/vision_pairs.jsonl"
VISION_IMAGES_ROOT = f"{_PROJ}/knowledge/curricula/core-multimodal-coco"
ADAPTER_DIR = f"{_BASE}/lora_adapters/gemma4_e4b_core_multimodal_v2"
MERGED_DIR = f"{_BASE}/Gemma4-E4B-GAIA-Core-Multimodal-v2"

# ── Training config ─────────────────────────────────────────────────────────
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.0
LEARNING_RATE = 2e-4  # A bit higher than text-only — projections need movement
BATCH_SIZE = 1
GRAD_ACCUM = 4
MAX_SEQ_LENGTH = 1024  # Multimodal is longer (image tokens + text)
WARMUP_STEPS = 10


# ── Tower key detection (matches graft_multimodal_towers.py) ───────────────
_TOWER_PREFIXES = (
    "model.audio_tower.", "model.vision_tower.",
    "audio_tower.", "vision_tower.",
    "model.embed_vision.", "model.embed_audio.",
    "embed_vision.", "embed_audio.",
)


def _is_tower_key(key: str) -> bool:
    return any(key.startswith(p) for p in _TOWER_PREFIXES)


# ── Gemma 4 prompt formatting ──────────────────────────────────────────────
# ── Chat template selection (Gemma 4 turn-tags vs ChatML) ─────────────────
# Mirrors the format ChatFormatter.format_conversation emits at inference
# time (see gaia-engine/gaia_engine/core.py ChatFormatter). The template
# is selected per-run via --chat-template; auto-detects from BASE_MODEL
# path (Qwen → chatml, Gemma → gemma4).
CHAT_TEMPLATE = "gemma4"  # overridden by main() before dataset build


def _detect_chat_template(model_path: str) -> str:
    """Auto-detect chat template from base model path. Override via
    --chat-template flag."""
    p = (model_path or "").lower()
    if "qwen" in p:
        return "chatml"
    return "gemma4"


def format_text_pair(instruction: str, output: str) -> str:
    """Format a text-only training pair in the active chat template."""
    if CHAT_TEMPLATE == "chatml":
        return (
            f"<|im_start|>user\n{instruction}<|im_end|>\n"
            f"<|im_start|>assistant\n{output}<|im_end|>"
        )
    # Gemma 4 (default)
    return (
        f"<|turn>user<turn|>\n{instruction}\n"
        f"<|turn>assistant<turn|>\n{output}<turn|>"
    )


def format_vision_prompt(instruction: str, output: str, img_placeholder: str) -> str:
    """Format a vision+text training pair. Image placeholder is inlined at
    the start of the user turn; the processor expands it into boi + soft
    tokens + eoi automatically.
    """
    if CHAT_TEMPLATE == "chatml":
        return (
            f"<|im_start|>user\n{img_placeholder}\n{instruction}<|im_end|>\n"
            f"<|im_start|>assistant\n{output}<|im_end|>"
        )
    return (
        f"<|turn>user<turn|>\n{img_placeholder}\n{instruction}\n"
        f"<|turn>assistant<turn|>\n{output}<turn|>"
    )


def format_audio_prompt(instruction: str, output: str, audio_placeholder: str) -> str:
    """Format an audio+text training pair. Audio placeholder is inlined at
    the start of the user turn; the processor expands it into the audio
    soft tokens automatically (audio_seq_length=750 by default for Gemma 4).
    """
    if CHAT_TEMPLATE == "chatml":
        return (
            f"<|im_start|>user\n{audio_placeholder}\n{instruction}<|im_end|>\n"
            f"<|im_start|>assistant\n{output}<|im_end|>"
        )
    return (
        f"<|turn>user<turn|>\n{audio_placeholder}\n{instruction}\n"
        f"<|turn>assistant<turn|>\n{output}<turn|>"
    )


# ── Answer-span helpers (loss masking) ──────────────────────────────────────
def _assistant_marker_string() -> str:
    """Assistant turn marker for the active chat template. Tokens up to and
    including this marker are masked from the loss (prompt + image/audio
    soft-token placeholders); only the answer that follows is scored."""
    return "<|im_start|>assistant" if CHAT_TEMPLATE == "chatml" else "<|turn>assistant<turn|>"


def _last_subseq_end(haystack, needle):
    """Index just past the LAST occurrence of ``needle`` in ``haystack``
    (both lists of int ids), or None if absent."""
    n = len(needle)
    if n == 0:
        return None
    for start in range(len(haystack) - n, -1, -1):
        if haystack[start:start + n] == needle:
            return start + n
    return None


def _has_scorable_answer(input_ids, marker_ids) -> bool:
    """True if the assistant marker is present AND at least one answer token
    follows it. Samples that fail this (answer truncated past max_length, or
    marker absent) would mask to all -100 → CrossEntropyLoss over zero tokens
    → NaN. The dataset builder skips them rather than feed NaN into the loss."""
    seq = input_ids.tolist()
    end = _last_subseq_end(seq, marker_ids)
    return end is not None and end < len(seq)


# ── Dataset building ───────────────────────────────────────────────────────
def _load_wav_mono_16k(path: str):
    """Read a WAV file → float32 numpy array at 16 kHz mono.

    Stdlib-only so no librosa dependency. Audio fixtures are pre-rendered
    at 16 kHz mono PCM16 by build_core_audio_curriculum.py.
    """
    import wave
    import numpy as np
    with wave.open(path, "rb") as w:
        n_channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        framerate = w.getframerate()
        n_frames = w.getnframes()
        raw = w.readframes(n_frames)
    if sampwidth == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sampwidth == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        # 1 byte u8 (rare) — treat as offset-mid PCM
        data = (np.frombuffer(raw, dtype="u1").astype(np.float32) - 128.0) / 128.0
    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)
    if framerate != 16000:
        # Linear resample — adequate for synthetic primitives. For natural
        # audio in v7+ we'd want soundfile/librosa.
        ratio = 16000 / framerate
        new_len = int(round(len(data) * ratio))
        if new_len > 1:
            xp = np.linspace(0, 1, len(data), endpoint=False)
            x_new = np.linspace(0, 1, new_len, endpoint=False)
            data = np.interp(x_new, xp, data).astype(np.float32)
    return data


def build_dataset(text_path: str | None, vision_path: str | None,
                  images_root: str, processor,
                  audio_path: str | None = None, audios_root: str | None = None):
    """Build a unified dataset of text-only, vision, and audio samples.

    Samples carry the processor-tokenized `input_ids` plus modality
    tensors (`pixel_values` for vision, `input_features` for audio).
    Labels are set to `input_ids` verbatim — caller masks prompt tokens
    to -100 at collate time.
    """
    from PIL import Image
    import torch

    img_tok = getattr(processor, "image_token", None) or "<|image|>"
    audio_tok = getattr(processor, "audio_token", None) or "<|audio|>"
    samples = []

    # Answer-span marker: samples whose answer is truncated past MAX_SEQ_LENGTH
    # (so nothing is left to score) are skipped — feeding an all-masked sample
    # to CrossEntropyLoss yields NaN. Mostly long multi-turn text; vision/audio
    # are short enough that this rarely fires.
    marker_ids = list(processor.tokenizer(
        _assistant_marker_string(), add_special_tokens=False)["input_ids"])
    skipped = {"text": 0, "vision": 0, "audio": 0}

    # Text-only
    text_count = 0
    if text_path and Path(text_path).exists():
        with open(text_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                text = format_text_pair(d["instruction"], d["output"])
                tok = processor.tokenizer(
                    text, return_tensors="pt", truncation=True,
                    max_length=MAX_SEQ_LENGTH,
                )
                ids = tok["input_ids"].squeeze(0)
                if not _has_scorable_answer(ids, marker_ids):
                    skipped["text"] += 1
                    continue
                samples.append({
                    "input_ids": ids,
                    "attention_mask": tok["attention_mask"].squeeze(0),
                    "is_vision": False,
                    "category": d.get("category", "text"),
                })
                text_count += 1
    log.info("Text samples: %d", text_count)

    # Vision
    vision_count = 0
    if vision_path and Path(vision_path).exists():
        with open(vision_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                image_rel = d["image"]
                image_path = os.path.join(images_root, image_rel)
                if not os.path.exists(image_path):
                    log.warning("Image not found, skipping: %s", image_path)
                    continue

                image = Image.open(image_path).convert("RGB")
                # Limit image size to keep VRAM manageable — Gemma 4 vision
                # uses a fixed 896×896 input, but smaller crops still work
                # and the processor handles the resize.

                text = format_vision_prompt(d["instruction"], d["output"], img_tok)
                processed = processor(
                    text=[text], images=[image],
                    return_tensors="pt", padding=False, truncation=True,
                    max_length=MAX_SEQ_LENGTH,
                )
                ids = processed["input_ids"].squeeze(0)
                if not _has_scorable_answer(ids, marker_ids):
                    skipped["vision"] += 1
                    continue
                sample = {
                    "input_ids": ids,
                    "attention_mask": processed["attention_mask"].squeeze(0),
                    "is_vision": True,
                    "category": "vision",
                }
                # Strip leading batch dim from everything — the collator
                # re-stacks to add a single batch dim at call time. If we
                # leave pixel_values as [1, patches, dim], we end up with
                # [1, 1, patches, dim] after stacking and Gemma4's vision
                # encoder's repeat_kv chokes on the 5-D attention tensor.
                if "pixel_values" in processed:
                    pv = processed["pixel_values"]
                    sample["pixel_values"] = pv.squeeze(0) if pv.dim() > 3 else pv[0] if pv.dim() == 3 else pv
                if "mm_token_type_ids" in processed:
                    sample["mm_token_type_ids"] = processed["mm_token_type_ids"].squeeze(0)
                if "image_position_ids" in processed:
                    sample["image_position_ids"] = processed["image_position_ids"].squeeze(0)
                samples.append(sample)
                vision_count += 1
    log.info("Vision samples: %d", vision_count)

    # Audio
    audio_count = 0
    if audio_path and Path(audio_path).exists():
        with open(audio_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                audio_rel = d["audio"]
                audio_full = os.path.join(audios_root or "", audio_rel)
                if not os.path.exists(audio_full):
                    log.warning("Audio not found, skipping: %s", audio_full)
                    continue

                try:
                    wav = _load_wav_mono_16k(audio_full)
                except Exception as e:
                    log.warning("Audio load failed for %s: %s", audio_full, e)
                    continue

                text = format_audio_prompt(d["instruction"], d["output"], audio_tok)
                # NOTE: do NOT pass truncation=True. Gemma4Processor with
                # truncation enabled clips input_features way below the real
                # frame count even when input_ids is well under max_length
                # (e.g. 249 mel frames → 6 with truncation=True). This breaks
                # the audio_features / audio_tokens count check in the
                # model's forward. Audio prompts are short, so we just rely
                # on the curriculum staying under 30s ≈ 750 audio tokens.
                processed = processor(
                    text=[text], audio=[wav],
                    return_tensors="pt", padding=False,
                    sampling_rate=16000,
                )
                ids = processed["input_ids"].squeeze(0)
                if not _has_scorable_answer(ids, marker_ids):
                    skipped["audio"] += 1
                    continue
                sample = {
                    "input_ids": ids,
                    "attention_mask": processed["attention_mask"].squeeze(0),
                    "is_vision": False,
                    "is_audio": True,
                    "category": "audio",
                }
                if "input_features" in processed:
                    inf = processed["input_features"]
                    sample["input_features"] = inf.squeeze(0) if inf.dim() > 2 else inf
                if "input_features_mask" in processed:
                    ifm = processed["input_features_mask"]
                    sample["input_features_mask"] = ifm.squeeze(0) if ifm.dim() > 1 else ifm
                if "mm_token_type_ids" in processed:
                    sample["mm_token_type_ids"] = processed["mm_token_type_ids"].squeeze(0)
                samples.append(sample)
                audio_count += 1
    log.info("Audio samples: %d", audio_count)
    if any(skipped.values()):
        log.warning(
            "Skipped %d samples with no scorable answer (truncated past "
            "MAX_SEQ_LENGTH=%d): text=%d vision=%d audio=%d",
            sum(skipped.values()), MAX_SEQ_LENGTH,
            skipped["text"], skipped["vision"], skipped["audio"],
        )

    # Homogenize batches: text first, then vision, then audio — prevents
    # mixed-modality batches the collator can't stack.
    for s in samples:
        s.setdefault("is_audio", False)
    text_only = [s for s in samples if not s["is_vision"] and not s["is_audio"]]
    vision_only = [s for s in samples if s["is_vision"]]
    audio_only = [s for s in samples if s.get("is_audio")]
    sorted_samples = text_only + vision_only + audio_only
    log.info("Total samples: %d (%d text, %d vision, %d audio)",
             len(sorted_samples), len(text_only), len(vision_only), len(audio_only))

    class Ds(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            s = self.data[idx]
            out = {
                "input_ids": s["input_ids"],
                "attention_mask": s["attention_mask"],
                # Pass-through string field used for per-category loss logging.
                # The collator collects these into out["_categories"] and
                # MMLossTrainer.compute_loss pops them before the model forward.
                "category": s.get("category", "unknown"),
            }
            for k in ("pixel_values", "mm_token_type_ids", "image_position_ids",
                     "input_features", "input_features_mask"):
                if k in s:
                    out[k] = s[k]
            return out

    return Ds(sorted_samples)


class MultimodalCollator:
    """Collate variable-length text/vision/audio samples into a batch.

    Builds causal-LM labels that score ONLY the assistant's answer. Every
    position up to and including the assistant turn marker is masked to
    -100 (this covers the instruction AND the image/audio soft-token
    placeholders, which live in the user turn), as is padding.

    This is the standard PaliGemma / Gemma-4 SFT objective. The previous
    code set ``labels = input_ids`` and masked only padding, so the
    ~256 image soft-tokens per vision sample (and the audio soft-tokens)
    were scored against the image placeholder id. Those positions are
    unlearnable and dominate the per-category mean, pinning vision/audio
    loss at ~ln(vocab) (~12 for Gemma 4's 256k vocab) so it never
    converged while text did. See GAIA_Project-axa.

    Stacks pixel_values / input_features along batch dim when every
    sample in the batch has them; drops them silently if mixed (shouldn't
    happen given our text-first / vision-second / audio-third sort order).
    """

    def __init__(self, processor, pad_token_id: int):
        self.processor = processor
        self.pad_token_id = pad_token_id
        tok = processor.tokenizer
        # Assistant turn marker for the active chat template. Everything
        # up to and including this marker is masked from the loss; only
        # the answer tokens that follow are scored. For Gemma 4 the
        # delimiters are atomic special tokens, so this 3-token sequence
        # matches cleanly as a subsequence of the processed input_ids.
        # (build_dataset already skips samples whose answer is truncated
        # away, so a missing marker here should be rare.)
        self._assistant_ids = list(tok(
            _assistant_marker_string(), add_special_tokens=False)["input_ids"])
        # Defensive fallback: image/audio placeholder token ids, masked
        # wherever they appear (they always precede the marker, so the
        # prompt mask already covers them — this is belt-and-suspenders).
        self._modality_token_ids = set()
        for attr in ("image_token", "audio_token"):
            t = getattr(processor, attr, None)
            if t:
                tid = tok.convert_tokens_to_ids(t)
                if isinstance(tid, int) and tid >= 0:
                    self._modality_token_ids.add(tid)
        self._missing_marker_warned = False

    def __call__(self, batch):
        import torch

        input_ids_list = [s["input_ids"] for s in batch]
        mask_list = [s["attention_mask"] for s in batch]
        max_len = max(x.shape[0] for x in input_ids_list)

        padded_ids, padded_masks, padded_labels = [], [], []
        for ids, mask in zip(input_ids_list, mask_list):
            pad = max_len - ids.shape[0]
            if pad > 0:
                ids = torch.cat([ids, torch.full((pad,), self.pad_token_id, dtype=ids.dtype)])
                mask = torch.cat([mask, torch.zeros(pad, dtype=mask.dtype)])
            labels = ids.clone()
            # 1) Mask the prompt: everything up to and including the
            #    assistant turn marker. This covers the instruction and
            #    the image/audio soft-token placeholders (all in the user
            #    turn), so only the assistant's answer is scored.
            real_len = int(mask.sum().item())
            real_ids = ids[:real_len].tolist()
            answer_start = _last_subseq_end(real_ids, self._assistant_ids)
            if answer_start is None:
                # Marker absent (e.g. answer truncated past max_length) —
                # score nothing rather than train on prompt/image tokens.
                labels[:] = -100
                if not self._missing_marker_warned:
                    log.warning(
                        "MultimodalCollator: assistant marker %s not found in a "
                        "sample (likely truncation); masking it entirely. If this "
                        "is frequent, check chat template / MAX_SEQ_LENGTH.",
                        self._assistant_ids,
                    )
                    self._missing_marker_warned = True
            else:
                labels[:answer_start] = -100
            # 2) Mask padding positions.
            labels[mask == 0] = -100
            # 3) Defensive: mask any stray image/audio soft-token ids.
            for tid in self._modality_token_ids:
                labels[ids == tid] = -100
            padded_ids.append(ids)
            padded_masks.append(mask)
            padded_labels.append(labels)

        out = {
            "input_ids": torch.stack(padded_ids),
            "attention_mask": torch.stack(padded_masks),
            "labels": torch.stack(padded_labels),
        }

        # Pass-through categories for per-category loss logging. NOT a tensor;
        # MMLossTrainer.compute_loss pops this before the model forward.
        out["_categories"] = [s.get("category", "unknown") for s in batch]

        # If every sample has vision tensors, stack them.
        # - pixel_values:       [patches, dim]     → stack to [batch, patches, dim]
        # - image_position_ids: [patches, 2]       → stack to [batch, patches, 2]
        #   (aligned to patches, NOT to the text sequence — no padding)
        # - mm_token_type_ids:  [seq_len]          → pad to max_len, stack
        # We rely on the text-first/vision-second/audio-third sort order +
        # batch_size=1 to keep batches homogeneous.
        vision_keys = ("pixel_values", "mm_token_type_ids", "image_position_ids")
        for key in vision_keys:
            if not all(key in s for s in batch):
                continue
            try:
                tensors = [s[key] for s in batch]
                if key == "mm_token_type_ids":
                    padded = []
                    for t in tensors:
                        pad = max_len - t.shape[0]
                        if pad > 0:
                            t = torch.cat([t, torch.zeros(pad, dtype=t.dtype)])
                        padded.append(t)
                    out[key] = torch.stack(padded)
                else:
                    out[key] = torch.stack(tensors)
            except Exception as e:
                log.warning("Collate %s failed (%s) — skipping", key, e)

        # Audio tensors — Gemma 4 audio processor returns:
        #   input_features:      [n_frames, 128]  (mel features)
        #   input_features_mask: [n_frames]       (bool)
        # n_frames is variable (depends on clip duration); pad to batch max.
        audio_keys = ("input_features", "input_features_mask")
        if all(all(k in s for k in audio_keys) for s in batch):
            try:
                feats = [s["input_features"] for s in batch]
                masks = [s["input_features_mask"] for s in batch]
                max_frames = max(t.shape[0] for t in feats)
                feat_dim = feats[0].shape[-1]
                padded_feats, padded_masks = [], []
                for f, m in zip(feats, masks):
                    pad = max_frames - f.shape[0]
                    if pad > 0:
                        f = torch.cat([f, torch.zeros(pad, feat_dim, dtype=f.dtype)], dim=0)
                        m = torch.cat([m, torch.zeros(pad, dtype=m.dtype)])
                    padded_feats.append(f)
                    padded_masks.append(m)
                out["input_features"] = torch.stack(padded_feats)
                out["input_features_mask"] = torch.stack(padded_masks)
            except Exception as e:
                log.warning("Collate audio failed (%s) — skipping", e)

        return out


# ── Dequantize Linear4bit → bf16 nn.Linear ─────────────────────────────────
def dequantize_tower_linear4bit(model, tower_substr: str = "audio_tower") -> int:
    """Selectively replace Linear4bit with bf16 nn.Linear inside a tower.

    Why this is needed for audio_tower: Gemma 4 audio encoder layers
    (Gemma4AudioFeedForward / Gemma4AudioLightConv1d / Gemma4AudioLayer)
    contain a `gradient_clipping = min(..., torch.finfo(weight.dtype).max)`
    line in their forward() that requires the underlying weight dtype to
    be a real float type. When bnb quantizes those layers to NF4
    (Params4bit / uint8), torch.finfo() raises TypeError. The
    BitsAndBytesConfig(llm_int8_skip_modules=['audio_tower', ...])
    contract should prevent this, but in practice (transformers 4.5x +
    bnb 0.4x) the skip pattern is checked against the LEAF name during
    the recursive walk, so audio_tower's nested .linear submodules still
    get replaced. We dequantize them in-place after model load, before
    LoRA application.

    This keeps weights on GPU (model is mid-train) — distinct from
    dequantize_linear4bit_modules() which moves to CPU for save.
    """
    import torch
    import torch.nn as nn
    try:
        import bitsandbytes as bnb
        import bitsandbytes.functional as bnb_f
    except ImportError:
        log.warning("bitsandbytes not available — cannot dequantize tower")
        return 0

    to_replace: list = []
    for name, module in model.named_modules():
        if tower_substr not in name:
            continue
        for attr_name, child in list(module.named_children()):
            if isinstance(child, bnb.nn.Linear4bit):
                to_replace.append((module, attr_name, child, name))

    log.info("Dequantizing %d Linear4bit modules under '%s' (kept on GPU)...",
             len(to_replace), tower_substr)
    count = 0
    for parent, attr_name, lin4, full_name in to_replace:
        try:
            weight_gpu = lin4.weight.data
            qstate = lin4.weight.quant_state
            if weight_gpu.device.type != "cuda":
                weight_gpu = weight_gpu.cuda()
            dequant_gpu = bnb_f.dequantize_4bit(weight_gpu, qstate)
            new_linear = nn.Linear(
                lin4.in_features, lin4.out_features,
                bias=lin4.bias is not None,
                dtype=torch.bfloat16, device="cuda",
            )
            with torch.no_grad():
                new_linear.weight.copy_(dequant_gpu.to(torch.bfloat16))
                if lin4.bias is not None:
                    new_linear.bias.copy_(lin4.bias.detach().to(torch.bfloat16))
            setattr(parent, attr_name, new_linear)
            del lin4, dequant_gpu
            count += 1
        except Exception as e:
            log.error("Tower dequant failed for %s: %s", full_name, e)
            raise

    gc.collect()
    torch.cuda.empty_cache()
    log.info("  dequantized %d tower modules → bf16 on GPU", count)
    return count


def dequantize_linear4bit_modules(model) -> int:
    """Walk the model and replace every bnb Linear4bit with a plain nn.Linear
    holding the dequantized bf16 weights.

    Why this is needed: PEFT's `merge_and_unload()` on a 4-bit quantized
    model keeps the merged weights in packed NF4 form (shape [N*K/2, 1]
    uint8). When you `save_file(state_dict, ...)` and reload later under
    BitsAndBytesConfig(load_in_4bit=True), bnb tries to re-quantize the
    already-packed weights as if they were bf16 — the shapes don't match
    and forward gives RuntimeError: mat1 and mat2 shapes cannot be
    multiplied.

    The fix is to dequantize post-merge so the saved safetensors contain
    regular bf16 weight tensors. NF4 quant happens freshly on reload.
    """
    import torch
    import torch.nn as nn
    try:
        import bitsandbytes as bnb
        import bitsandbytes.functional as bnb_f
    except ImportError:
        log.warning("bitsandbytes not available — cannot dequantize")
        return 0

    count = 0
    # Collect (parent_module, attr_name, child) triples first so we don't
    # mutate the tree while iterating.
    to_replace: list = []
    for name, module in model.named_modules():
        for attr_name, child in list(module.named_children()):
            if isinstance(child, bnb.nn.Linear4bit):
                to_replace.append((module, attr_name, child, name))

    log.info("Dequantizing %d Linear4bit modules (GPU dequant → CPU store)...",
             len(to_replace))
    for i, (parent, attr_name, lin4, full_name) in enumerate(to_replace):
        try:
            # bnb.dequantize_4bit requires the packed weight + quant_state
            # on GPU. We dequantize on GPU, immediately move the result to
            # CPU, and free the NF4 original. This keeps peak VRAM bounded
            # to ~2x one layer's bf16 size (~100 MB for 5376x2560).
            weight_gpu = lin4.weight.data
            qstate = lin4.weight.quant_state
            if weight_gpu.device.type != "cuda":
                # Already on CPU (happens if caller moved model first) —
                # bnb dequant still needs CUDA, so temporarily round-trip.
                weight_gpu = weight_gpu.cuda()
            dequant_gpu = bnb_f.dequantize_4bit(weight_gpu, qstate)
            dequant_cpu = dequant_gpu.to(dtype=torch.bfloat16, device="cpu")
            del dequant_gpu, weight_gpu

            new_linear = nn.Linear(
                lin4.in_features, lin4.out_features,
                bias=lin4.bias is not None,
                dtype=torch.bfloat16, device="cpu",
            )
            with torch.no_grad():
                new_linear.weight.copy_(dequant_cpu)
                if lin4.bias is not None:
                    bias_cpu = lin4.bias.detach().to(dtype=torch.bfloat16, device="cpu")
                    new_linear.bias.copy_(bias_cpu)
            setattr(parent, attr_name, new_linear)
            del lin4, dequant_cpu
            count += 1
            if count % 50 == 0:
                torch.cuda.empty_cache()
                log.info("  %d/%d dequantized (VRAM: %.2f GB)", count, len(to_replace),
                         torch.cuda.memory_allocated() / 1024**3)
        except Exception as e:
            log.error("Dequant failed for %s: %s", full_name, e)
            raise

    gc.collect()
    torch.cuda.empty_cache()
    log.info("  dequantized %d modules; model weights now bf16 on CPU", count)
    return count


# ── Tower graft at save time (the core fix) ────────────────────────────────
def save_with_tower_graft(merged_model, base_model_path: str, out_dir: Path,
                          processor):
    """Save the merged LoRA model with vision + audio tower weights grafted
    from the base. This is the fix for the real bug we diagnosed in the
    previous training pipeline: PEFT's merge saves only what LoRA touched,
    so towers got stripped. We restore them here with their native
    QAT-class layout intact.
    """
    import torch
    from safetensors.torch import save_file
    from safetensors import safe_open

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Collect merged state dict (LoRA-merged weights, with tower gaps)
    log.info("Collecting merged state dict...")
    merged_sd = {}
    for name, tensor in merged_model.state_dict().items():
        merged_sd[name] = tensor.detach().cpu()
    log.info("  merged keys: %d", len(merged_sd))

    # 2. Open base's safetensors to read tower tensors
    base_safetensors = Path(base_model_path) / "model.safetensors"
    if not base_safetensors.exists():
        # Sharded? Load via index
        index_path = Path(base_model_path) / "model.safetensors.index.json"
        if not index_path.exists():
            raise FileNotFoundError(
                f"No safetensors found in {base_model_path}"
            )
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        base_shards = {fname: safe_open(
            str(Path(base_model_path) / fname), framework="pt"
        ) for fname in set(weight_map.values())}

        def get_base(key):
            if key in weight_map:
                return base_shards[weight_map[key]].get_tensor(key)
            return None

        base_keys = set(weight_map.keys())
    else:
        sh = safe_open(str(base_safetensors), framework="pt")

        def get_base(key):
            try:
                return sh.get_tensor(key)
            except Exception:
                return None

        base_keys = set(sh.keys())

    # 3. For every tower key in base, overwrite/insert into merged_sd
    grafted = 0
    overwrote_flat = 0
    for key in base_keys:
        if not _is_tower_key(key):
            continue
        tensor = get_base(key)
        if tensor is None:
            continue
        was_present = key in merged_sd
        merged_sd[key] = tensor
        grafted += 1
        if was_present:
            overwrote_flat += 1
    log.info("Tower graft: inserted %d keys from base (%d overwrote existing flat tower keys)",
             grafted, overwrote_flat)

    # 4. Remove any flat-named tower keys that don't exist in base
    # (these are the stripped remnants from PEFT merge — their base
    # equivalents are under .linear.weight nesting).
    to_delete = [k for k in merged_sd
                 if _is_tower_key(k) and k not in base_keys]
    for k in to_delete:
        del merged_sd[k]
    log.info("  dropped %d flat-named tower leftovers", len(to_delete))

    # 5. Write
    out_file = out_dir / "model.safetensors"
    log.info("Writing %s (%d keys)...", out_file, len(merged_sd))
    save_file(merged_sd, str(out_file), metadata={"format": "pt"})
    size_gb = out_file.stat().st_size / (1024 ** 3)
    log.info("  wrote %.2f GB", size_gb)

    # 6. Copy config and processor artefacts from BASE (authoritative for
    # multimodal) — our trained dir would be missing processor_config.json.
    # Tokenizer comes from base too since we didn't modify it.
    for fname in ("config.json", "generation_config.json",
                  "processor_config.json", "tokenizer_config.json",
                  "tokenizer.json", "special_tokens_map.json"):
        src = Path(base_model_path) / fname
        if src.exists():
            shutil.copy2(src, out_dir / fname)


# ── Main training ──────────────────────────────────────────────────────────
def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--save-steps", type=int, default=500,
                        help="Checkpoint every N steps. Lower = less work lost "
                             "when the NVRM RC watchdog kills the run "
                             "(GAIA_Project-22a); the resilient wrapper resumes "
                             "from the latest checkpoint. save_total_limit keeps 2.")
    parser.add_argument("--steps", type=int, default=200,
                        help="Max training steps (default 200)")
    parser.add_argument("--skip-merge", action="store_true",
                        help="Train and save adapter only, skip final merge+graft")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build dataset and exit (pipeline test)")
    parser.add_argument("--curriculum-name", default=None,
                        help="Override curriculum dir name under "
                             "knowledge/curricula/ (e.g. 'core-multimodal-v3'). "
                             "Also sets adapter/merged dir version suffix.")
    parser.add_argument("--version-tag", default=None,
                        help="Override the version suffix on adapter+merged "
                             "dirs (e.g. 'v3'). Defaults to deriving from "
                             "--curriculum-name.")
    parser.add_argument("--base-model", default=None,
                        help="Override the base model path. Use to chain "
                             "phases: phase 2 starts from phase 1's merged "
                             "output. Default = the constants block's "
                             "BASE_MODEL (Gemma4-E4B-GAIA-Unified-v5-Multimodal).")
    parser.add_argument("--steps-warmup", type=int, default=None,
                        help="Override warmup steps (default 10).")
    parser.add_argument("--no-text", action="store_true",
                        help="Skip text-only samples (vision-only training).")
    parser.add_argument("--audio-curriculum-name", default=None,
                        help="Override audio curriculum dir name under "
                             "knowledge/curricula/ (e.g. 'core-multimodal-v6audio'). "
                             "Loads audio_pairs.jsonl + audio/ from that dir.")
    parser.add_argument("--no-vision", action="store_true",
                        help="Skip vision samples (audio-only or text-only training).")
    parser.add_argument("--lora-r", type=int, default=None,
                        help="Override LoRA rank (default 8). Alpha is set to 2*r.")
    parser.add_argument("--text-curriculum", default=None,
                        help="Override the text-only training jsonl. Default = "
                             "knowledge/curricula/core-multimodal/train.jsonl")
    parser.add_argument("--target-modules-regex", default=None,
                        help="Override the LoRA target_modules regex. Use to "
                             "scope which language_model layers get LoRA "
                             "(e.g. only the last 12). Default matches all "
                             "language_model attention/MLP linears.")
    parser.add_argument("--no-shuffle", action="store_true",
                        help="Use SequentialSampler instead of default "
                             "RandomSampler. Required when training a "
                             "spiral-ordered curriculum that encodes "
                             "phase-based dependencies — random shuffle "
                             "would destroy the ordering.")
    parser.add_argument("--no-audio", action="store_true",
                        help="Skip audio samples (explicit form; equivalent "
                             "to not passing --audio-curriculum-name). Useful "
                             "for text-only Prime training on Qwen3-VL.")
    parser.add_argument("--chat-template", choices=("auto", "gemma4", "chatml"),
                        default="auto",
                        help="Chat template wrapping at training time. "
                             "'auto' (default) picks chatml for Qwen-family "
                             "base models, gemma4 otherwise. Must match what "
                             "the engine emits at inference time.")
    args = parser.parse_args()

    # Allow CLI to point at a different curriculum without editing globals.
    global VISION_CURRICULUM, VISION_IMAGES_ROOT, ADAPTER_DIR, MERGED_DIR, BASE_MODEL, WARMUP_STEPS, LORA_R, LORA_ALPHA, TEXT_CURRICULUM
    if args.lora_r is not None:
        LORA_R = args.lora_r
        LORA_ALPHA = 2 * args.lora_r
    if args.text_curriculum:
        TEXT_CURRICULUM = args.text_curriculum
    if args.base_model:
        BASE_MODEL = args.base_model
    if args.steps_warmup is not None:
        WARMUP_STEPS = args.steps_warmup
    if args.curriculum_name:
        curr_dir = f"{_PROJ}/knowledge/curricula/{args.curriculum_name}"
        VISION_CURRICULUM = f"{curr_dir}/vision_pairs.jsonl"
        VISION_IMAGES_ROOT = curr_dir
        # Derive version tag from curriculum name unless explicitly given.
        # 'core-multimodal-v3' → 'v3'.
        derived = args.version_tag or args.curriculum_name.rsplit("-", 1)[-1]
        ADAPTER_DIR = f"{_BASE}/lora_adapters/gemma4_e4b_core_multimodal_{derived}"
        MERGED_DIR = f"{_BASE}/Gemma4-E4B-GAIA-Core-Multimodal-{derived.upper()}"
    elif args.version_tag:
        ADAPTER_DIR = f"{_BASE}/lora_adapters/gemma4_e4b_core_multimodal_{args.version_tag}"
        MERGED_DIR = f"{_BASE}/Gemma4-E4B-GAIA-Core-Multimodal-{args.version_tag.upper()}"

    # Audio curriculum — independent of vision so we can train audio-only.
    audio_curriculum = None
    audio_root = None
    if args.audio_curriculum_name and not args.no_audio:
        adir = f"{_PROJ}/knowledge/curricula/{args.audio_curriculum_name}"
        audio_curriculum = f"{adir}/audio_pairs.jsonl"
        audio_root = adir
        if not args.curriculum_name:
            # When audio-only and no vision curriculum named, use the audio
            # tag for adapter/merged dirs.
            derived = args.version_tag or args.audio_curriculum_name.rsplit("-", 1)[-1]
            ADAPTER_DIR = f"{_BASE}/lora_adapters/gemma4_e4b_core_multimodal_{derived}"
            MERGED_DIR = f"{_BASE}/Gemma4-E4B-GAIA-Core-Multimodal-{derived.upper()}"

    # Chat-template selection (Gemma 4 turn tags vs Qwen ChatML). Auto-detect
    # from base model path so Prime training on Qwen3-VL uses ChatML and Core
    # training on Gemma 4 keeps turn-tags. Override with --chat-template.
    global CHAT_TEMPLATE
    if args.chat_template == "auto":
        CHAT_TEMPLATE = _detect_chat_template(BASE_MODEL)
        log.info("Chat template auto-detected: %s (from base=%s)",
                 CHAT_TEMPLATE, BASE_MODEL)
    else:
        CHAT_TEMPLATE = args.chat_template
        log.info("Chat template (explicit): %s", CHAT_TEMPLATE)

    # RunRecorder: structured run record at /shared/training_runs/<run_id>/
    # GAIA_Project-n0e Phase 1: shared module gaia_common.utils.run_recorder
    # owns the on-disk schema (config.json, curriculum.jsonl, metrics.jsonl,
    # checkpoints/, summary.json, battery_results.json) and writes the
    # active-run pointer + GAIA_TRAIN_RUN_ID env var so downstream tools
    # (cognitive_test_battery, dashboard) can discover this run.
    from gaia_common.utils.run_recorder import RunRecorder
    # Honor pre-existing GAIA_RUN_ID for backward compat with older callers.
    legacy_run_id = os.environ.get("GAIA_RUN_ID")
    _recorder = RunRecorder.create(
        version_tag=args.version_tag or "",
        run_id=legacy_run_id,
    )
    _recorder.__enter__()  # equivalent to `with _recorder:` over the rest of main()
    run_id = _recorder.run_id
    RUN_DIR = _recorder.run_dir
    run_config = {
        "base_model": BASE_MODEL,
        "text_curriculum": TEXT_CURRICULUM,
        "vision_curriculum": VISION_CURRICULUM if not args.no_vision else None,
        "audio_curriculum_name": args.audio_curriculum_name,
        "target_modules_regex": args.target_modules_regex,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM,
        "max_steps": args.steps,
        "warmup_steps": WARMUP_STEPS,
        "no_shuffle": args.no_shuffle,
        "version_tag": args.version_tag,
        "adapter_dir": ADAPTER_DIR,
        "merged_dir": MERGED_DIR,
    }
    _recorder.write_config(run_config)
    # Snapshot the text curriculum so future re-runs and analysis aren't
    # broken by upstream curriculum edits.
    if TEXT_CURRICULUM:
        try:
            _recorder.copy_curriculum(TEXT_CURRICULUM)
        except Exception as _e:
            log.warning("Curriculum snapshot failed: %s", _e)

    print("=" * 60)
    print("  GAIA Core Multimodal Training")
    print("=" * 60)
    print(f"Run ID:         {run_id}")
    print(f"Run dir:        {RUN_DIR}")
    print(f"Base model:     {BASE_MODEL}")
    print(f"Text curr:      {TEXT_CURRICULUM}")
    print(f"Vision curr:    {VISION_CURRICULUM}")
    print(f"Images root:    {VISION_IMAGES_ROOT}")
    print(f"Adapter out:    {ADAPTER_DIR}")
    print(f"Merged out:     {MERGED_DIR}")
    print(f"Steps:          {args.steps}")
    print()

    import torch
    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU available")
        return 1
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    print(f"GPU: {torch.cuda.get_device_name(0)} ({vram_total:.1f} GB)")

    # 1. Processor (Gemma4Processor / Qwen3VLProcessor / etc — auto-routed
    # from config.json's processor_class)
    log.info("Loading processor for %s...", BASE_MODEL)
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
    log.info("Processor: %s", type(processor).__name__)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # 2. Dataset
    log.info("Building dataset...")
    text_curr = None if args.no_text else TEXT_CURRICULUM
    vision_curr = None if args.no_vision else VISION_CURRICULUM
    print(f"Audio curr:     {audio_curriculum or '(none)'}")
    dataset = build_dataset(
        text_curr, vision_curr, VISION_IMAGES_ROOT, processor,
        audio_path=audio_curriculum, audios_root=audio_root,
    )
    if len(dataset) == 0:
        log.error("No training samples — add curriculum files and retry")
        return 1

    if args.dry_run:
        log.info("--dry-run: dataset built, exiting before model load")
        return 0

    # 3. Load model with NF4 + skip towers (they use Gemma4ClippableLinear
    #    natively; double-quantizing through bnb breaks forward pass).
    # Skip-modules also covers Qwen3-VL's `visual` and `vision_model` so
    # Prime training on Qwen3-VL doesn't NF4-corrupt the vision tower
    # (even though we don't train on vision, we still load the full model).
    log.info("Loading model with NF4 quantization (towers skipped)...")
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    skip_modules = [
        "lm_head",
        # Gemma 4 tower names
        "vision_tower", "audio_tower", "embed_vision", "embed_audio",
        # Qwen3-VL / Qwen2-VL tower names
        "visual", "vision_model",
    ]
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=skip_modules,
    )
    # device_map={"": 0} forces every weight onto cuda:0. If GPU lacks
    # capacity, this raises rather than silently CPU-offloading some
    # layers (accelerate's auto map). A previous Core 2.1 run hit that
    # silent-offload path during a lifecycle transition race, leaving
    # the LoRA's trainable layers on CPU and producing loss-22 garbage.
    #
    # Auto-class selection (2026-05-18): VL architectures (Qwen3-VL,
    # Qwen2-VL) aren't registered for AutoModelForCausalLM. Detect
    # via config and use AutoModelForImageTextToText, falling back to
    # AutoModelForCausalLM for text-only or Gemma 4 (which does support
    # CausalLM). Mirrors the engine's same fix at gaia-engine commit 4ad864d.
    _model_loaded = False
    _is_vl_arch = False
    try:
        from transformers import AutoConfig
        _cfg = AutoConfig.from_pretrained(BASE_MODEL, trust_remote_code=True)
        _model_type = getattr(_cfg, "model_type", "")
        _is_vl_arch = "vl" in _model_type.lower() or "vision" in _model_type.lower()
    except Exception:
        pass
    if _is_vl_arch:
        try:
            from transformers import AutoModelForImageTextToText
            model = AutoModelForImageTextToText.from_pretrained(
                BASE_MODEL, trust_remote_code=True,
                quantization_config=bnb_config,
                device_map={"": 0},
                low_cpu_mem_usage=True,
                attn_implementation="eager",
            )
            _model_loaded = True
            log.info("Loaded via AutoModelForImageTextToText (VL arch detected)")
        except Exception as _vl_err:
            log.warning("AutoModelForImageTextToText load failed (%s) — "
                        "falling back to AutoModelForCausalLM", _vl_err)
    if not _model_loaded:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, trust_remote_code=True,
            quantization_config=bnb_config,
            device_map={"": 0},
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model.enable_input_require_grads()

    used_gb = torch.cuda.memory_allocated() / 1024 ** 3
    log.info("Model loaded: %.2f GB VRAM", used_gb)

    # 3.5. Audio tower fix — bnb's llm_int8_skip_modules contract doesn't
    # actually skip nested .linear submodules under audio_tower despite
    # 'audio_tower' being listed. The audio encoder layers
    # (Gemma4AudioFeedForward / Gemma4AudioLightConv1d / Gemma4AudioLayer)
    # all have a `gradient_clipping = min(..., torch.finfo(weight.dtype).max)`
    # line in forward() that fails on NF4 weight (uint8). Dequantize them
    # in place to bf16 nn.Linear before LoRA application. Skip if the
    # curriculum has no audio (vision/text-only training is unaffected).
    if audio_curriculum:
        n = dequantize_tower_linear4bit(model, "audio_tower")
        used_gb = torch.cuda.memory_allocated() / 1024 ** 3
        log.info("Audio tower fix: %d layers → bf16; VRAM now %.2f GB", n, used_gb)

    # 4. Unwrap Gemma4ClippableLinear — BUT ONLY in language_model subtree.
    # Tower layers MUST keep their native QAT wrapper (Linear4bit in LM,
    # Gemma4ClippableLinear in towers).
    log.info("Unwrapping Gemma4ClippableLinear → Linear4bit (language_model only)...")
    unwrapped = 0
    for name, module in list(model.named_modules()):
        if "language_model" not in name:
            continue
        for attr_name in list(vars(module).keys()):
            child = getattr(module, attr_name, None)
            if child is not None and type(child).__name__ == "Gemma4ClippableLinear":
                inner = getattr(child, "linear", None)
                if inner is not None:
                    setattr(module, attr_name, inner)
                    unwrapped += 1
        for child_name, child in list(module._modules.items()):
            if type(child).__name__ == "Gemma4ClippableLinear":
                inner = getattr(child, "linear", None)
                if inner is not None:
                    module._modules[child_name] = inner
                    unwrapped += 1
    log.info("  unwrapped %d LM Gemma4ClippableLinear layers", unwrapped)

    # 5. LoRA — target LM only (regex matches language_model path prefix)
    log.info("Applying LoRA to language_model only...")
    from peft import LoraConfig, get_peft_model, TaskType

    target_modules_regex = args.target_modules_regex or (
        r".*language_model\.layers\.\d+\.(?:self_attn|mlp)\."
        r"(?:q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
    )
    log.info("LoRA target_modules regex: %s", target_modules_regex)
    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=target_modules_regex,
        task_type=TaskType.CAUSAL_LM, bias="none",
    )
    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    log.info("  LoRA trainable: %d / %d (%.3f%%)", trainable, total,
             100 * trainable / total)

    # 6. Trainer
    log.info("Setting up trainer...")
    from transformers import Trainer, TrainingArguments

    training_args = TrainingArguments(
        output_dir=ADAPTER_DIR,
        max_steps=args.steps,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        max_grad_norm=0.5,
        logging_steps=10,
        save_steps=max(50, min(args.save_steps, args.steps // 2)),
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        seed=42,
    )

    collator = MultimodalCollator(processor, processor.tokenizer.pad_token_id)

    class MMLossTrainer(Trainer):
        model_accepts_loss_kwargs = False
        _logged_audio_shapes = False
        # Per-category running stats: {category: [sum_loss, count]}.
        # See "calibration" discussion — bulk loss can hide categories
        # that are stuck high or that converge below a healthy floor.
        _cat_stats: dict = {}
        _cat_log_every = 100  # steps between per-category summary

        def _get_train_sampler(self, train_dataset=None):
            """Override sampler when --no-shuffle is set (spiral curriculum)."""
            if getattr(args, "no_shuffle", False):
                from torch.utils.data import SequentialSampler
                ds = train_dataset if train_dataset is not None else self.train_dataset
                log.info("Using SequentialSampler (--no-shuffle): preserving curriculum order")
                return SequentialSampler(ds)
            return super()._get_train_sampler(train_dataset)

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            categories = inputs.pop("_categories", None)
            labels = inputs.pop("labels", None)
            # Debug: log shapes of multimodal inputs once for audio sample
            if not self._logged_audio_shapes and "input_features" in inputs:
                ids = inputs.get("input_ids")
                aid = processor.audio_token_id
                n_audio_in_ids = (ids == aid).sum().item() if ids is not None else -1
                infshape = tuple(inputs["input_features"].shape) if "input_features" in inputs else None
                ifmshape = tuple(inputs["input_features_mask"].shape) if "input_features_mask" in inputs else None
                ifm_sum = inputs["input_features_mask"].sum().item() if "input_features_mask" in inputs else None
                log.info("[DEBUG audio sample] input_ids: %s, audio_tokens_in_ids: %d, "
                         "input_features: %s, input_features_mask: shape=%s sum_True=%s",
                         tuple(ids.shape) if ids is not None else None,
                         n_audio_in_ids, infshape, ifmshape, ifm_sum)
                self._logged_audio_shapes = True
            outputs = model(**inputs)
            logits = outputs.logits
            if labels is None:
                return outputs.loss if return_outputs is False else (outputs.loss, outputs)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

            # Per-category accumulation. With BATCH_SIZE=1 every batch is one
            # sample so attribution is exact; multi-sample batches would need
            # reduction='none' + per-row attribution.
            if categories:
                lv = float(loss.detach().item())
                for cat in categories:
                    s = self._cat_stats.setdefault(cat, [0.0, 0])
                    s[0] += lv
                    s[1] += 1
                step = int(self.state.global_step) if hasattr(self, "state") else 0
                if step > 0 and step % self._cat_log_every == 0:
                    parts = []
                    cat_summary = {}
                    for cat in sorted(self._cat_stats.keys()):
                        s = self._cat_stats[cat]
                        if s[1] > 0:
                            avg = s[0] / s[1]
                            parts.append(f"{cat}={avg:.3f}(n={s[1]})")
                            cat_summary[cat] = {"mean_loss": avg, "n": s[1]}
                    log.info("[per-cat-loss step=%d] %s", step, " ".join(parts))
                    # RunRecorder: also write to metrics.jsonl
                    try:
                        with open(RUN_DIR / "metrics.jsonl", "a") as mf:
                            mf.write(json.dumps({
                                "step": step,
                                "per_category": cat_summary,
                                "ts": datetime.now(timezone.utc).isoformat(),
                            }) + "\n")
                    except Exception as e:
                        log.warning("metrics.jsonl write failed: %s", e)

            return (loss, outputs) if return_outputs else loss

    # TrainerCallback that captures bulk loss/grad_norm/lr from on_log
    # events and writes them to metrics.jsonl. Per-category metrics are
    # written separately by compute_loss above (every _cat_log_every steps).
    from transformers import TrainerCallback

    class RunRecorderCallback(TrainerCallback):
        def on_log(self, targs, state, control, logs=None, **kwargs):
            if not logs:
                return
            try:
                entry = {"step": int(state.global_step), **logs,
                         "ts": datetime.now(timezone.utc).isoformat()}
                with open(RUN_DIR / "metrics.jsonl", "a") as mf:
                    mf.write(json.dumps(entry) + "\n")
            except Exception:
                pass

    trainer = MMLossTrainer(
        model=model, args=training_args,
        train_dataset=dataset, data_collator=collator,
        callbacks=[RunRecorderCallback()],
    )

    # 7. Train
    log.info("Training for up to %d steps...", args.steps)
    t0 = time.time()
    # Resume from latest checkpoint in adapter dir if present
    _resume = None
    _adapter_path = Path(ADAPTER_DIR)
    if _adapter_path.exists():
        # Sort NUMERICALLY by step — lexical sort picks the wrong checkpoint
        # once step counts differ in digit length (e.g. "checkpoint-11000"
        # sorts before "checkpoint-5500"). Critical for resume-after-crash
        # (GAIA_Project-22a).
        _ckpts = [p for p in _adapter_path.glob("checkpoint-*")
                  if p.name.rsplit("-", 1)[-1].isdigit()]
        if _ckpts:
            _resume = str(max(_ckpts, key=lambda p: int(p.name.rsplit("-", 1)[-1])))
            log.info("Resuming from checkpoint: %s", _resume)
    result = trainer.train(resume_from_checkpoint=_resume)
    elapsed = time.time() - t0
    log.info("Training complete in %.1fs (%.1f min)", elapsed, elapsed / 60)
    log.info("  final loss: %.4f  steps: %d", result.training_loss, result.global_step)

    # 8. Save adapter
    log.info("Saving adapter to %s", ADAPTER_DIR)
    model.save_pretrained(ADAPTER_DIR)
    processor.save_pretrained(ADAPTER_DIR)
    with open(os.path.join(ADAPTER_DIR, "metadata.json"), "w") as f:
        json.dump({
            "base_model": BASE_MODEL,
            "text_curriculum": TEXT_CURRICULUM,
            "vision_curriculum": VISION_CURRICULUM,
            "steps": result.global_step,
            "final_loss": result.training_loss,
            "elapsed_seconds": round(elapsed, 1),
            "version": "core-multimodal-v1",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, f, indent=2)

    if args.skip_merge:
        log.info("--skip-merge: stopping before merge+graft")
        return 0

    # 9. Free training state before merge/dequant. paged_adamw_8bit keeps
    # state on GPU; with an 8B model the state plus grads can be 2-3 GB,
    # which OOMs during dequant (needs peak VRAM = NF4 weights + bf16
    # dequantized copy side-by-side for each layer).
    log.info("Freeing training state...")
    del trainer
    model.train(False)
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None
    gc.collect()
    torch.cuda.empty_cache()
    log.info("  VRAM after cleanup: %.2f GB", torch.cuda.memory_allocated() / 1024**3)

    # 10. Merge + dequantize + tower graft
    log.info("Merging LoRA into base (unload adapter)...")
    merged = model.merge_and_unload()
    log.info("  merge complete")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    log.info("  VRAM after del model: %.2f GB", torch.cuda.memory_allocated() / 1024**3)

    # CRITICAL: bnb's merged 4-bit weights are in packed NF4 form. Saving
    # them raw and reloading under load_in_4bit=True causes a shape
    # mismatch (the reloader tries to quantize already-packed uint8).
    # Dequantize to bf16 here so the saved safetensors re-quantize cleanly.
    # dequantize_linear4bit_modules does per-layer GPU dequant → CPU store
    # to keep peak VRAM bounded (bnb needs CUDA; moving the whole NF4 model
    # to CPU first breaks quant_state tensors).
    dequantize_linear4bit_modules(merged)

    out_dir = Path(MERGED_DIR)
    save_with_tower_graft(merged, BASE_MODEL, out_dir, processor)

    log.info("=" * 60)
    log.info("Done.")
    log.info("  Adapter: %s", ADAPTER_DIR)
    log.info("  Merged:  %s", MERGED_DIR)
    log.info("=" * 60)
    log.info("Next steps:")
    log.info("  1. Point /models/core at the new merged dir and reload gaia-core.")
    log.info("  2. Run scripts/post_training_reset.py --tier core to archive")
    log.info("     sessions, invalidate KV cache, and regen identity_prefix.")
    log.info("     The new model inherits the prior model's KV state and")
    log.info("     session bias otherwise — silent contamination.")

    # RunRecorder: write summary.json with final state + symlink the adapter
    # under the run's checkpoints/ directory for traceability.
    try:
        _recorder.link_checkpoint(Path(ADAPTER_DIR), name="adapter")
        if MERGED_DIR and Path(MERGED_DIR).exists():
            _recorder.link_checkpoint(Path(MERGED_DIR), name="merged")
    except Exception as _e:
        log.warning("Checkpoint linking failed: %s", _e)
    try:
        _recorder.write_summary({
            "final_loss": float(result.training_loss),
            "final_steps": int(result.global_step),
            "runtime_seconds": round(elapsed, 1),
            "status": "success",
            "scope_label": args.version_tag or "",
            "base_model": BASE_MODEL,
            "adapter_dir": ADAPTER_DIR,
            "merged_dir": MERGED_DIR,
        })
        log.info("RunRecorder summary written: %s/summary.json", RUN_DIR)
    except Exception as e:
        log.warning("RunRecorder summary write failed: %s", e)
    finally:
        try:
            _recorder.__exit__(None, None, None)
        except Exception:
            pass

    # Both `model` and `trainer` were already del'd before the merge step
    # (lines ~715, ~728) so the original `del model, merged, trainer`
    # raised UnboundLocalError at the very end of a successful run. The
    # function still returns 0 so the artifacts saved fine, but the
    # traceback was misleading. Only `merged` is still in scope here.
    del merged
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
