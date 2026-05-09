#!/usr/bin/env python3
"""Build the Core 2.0 unified curriculum — identity + behaviors + multimodal anchors.

The xln recipe in concrete form: instead of LoRA-stacking on top of the
displaced Unified-v5 LM, train a single LoRA from the raw Google base
(Gemma 4 E4B) with:
  - identity priming (217 pairs)
  - all GAIA behaviors (deliberation, conversational voice, self-knowledge)
  - vision anchors (subset of COCO + a few primitives — KEEP LM↔vision_tower
    alignment)
  - audio anchors (synthetic primitives only — KEEP LM↔audio_tower
    alignment, NO ESC-50 to avoid category leakage, NO LibriSpeech
    transcription prompts that polluted V7+V8)
  - multimodal directive examples ("when audio present, listen first")

Surgical layer scope on the LoRA itself (target_modules regex limited to
last 12 of 42 layers in the LM) keeps the identity+behavior tuning from
disturbing low-mid-layer attention patterns where the cross-modal binding
lives.

Output:
  knowledge/curricula/core2/
    text.jsonl                 (~650 identity+behavior pairs)
    vision_pairs.jsonl         (~500 anchors)
    audio_pairs.jsonl          (~500 anchors)
    images/                    (symlinks to COCO + a few generated primitives)
    audio/                     (synthesized WAVs)
"""
from __future__ import annotations

import json
import math
import random
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PROJ = Path("/gaia/GAIA_Project")
OUT_DIR = PROJ / "knowledge/curricula/core2"
IMAGES_DIR = OUT_DIR / "images"
AUDIO_DIR = OUT_DIR / "audio"

SRC_IDENTITY = PROJ / "knowledge/curricula/core-multimodal/train.jsonl"
SRC_DELIBERATION = PROJ / "knowledge/curricula/deliberation/train.json"
SRC_CONVERSATIONAL = PROJ / "knowledge/curricula/conversational/train.json"
SRC_SELFMODEL = PROJ / "knowledge/curricula/self-model/train.jsonl"
SRC_COCO = PROJ / "knowledge/curricula/core-multimodal-coco"

RATE = 16000


# ── TEXT (identity + behaviors) ─────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    out = []
    if not path.exists():
        print(f"  WARN: {path} missing — skipping")
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def load_json(path: Path) -> list[dict]:
    if not path.exists():
        print(f"  WARN: {path} missing — skipping")
        return []
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data
    return []


def normalize_text_pair(entry: dict) -> dict | None:
    """Project to canonical {instruction, output} schema."""
    inst = entry.get("instruction") or entry.get("input") or ""
    out = entry.get("output") or entry.get("response") or ""
    if not inst or not out:
        return None
    return {"instruction": str(inst).strip(), "output": str(out).strip()}


def build_text(rng) -> list[dict]:
    bag = []
    print("\n>> Text — identity (core-multimodal/train.jsonl) ...")
    for e in load_jsonl(SRC_IDENTITY):
        p = normalize_text_pair(e)
        if p:
            p["category"] = "identity"
            bag.append(p)
    n_identity = len(bag)
    print(f"   identity: {n_identity}")

    print(">> Text — deliberation (k23 voice training) ...")
    pre = len(bag)
    for e in load_json(SRC_DELIBERATION):
        p = normalize_text_pair(e)
        if p:
            p["category"] = "deliberation"
            bag.append(p)
    n_delib = len(bag) - pre
    print(f"   deliberation: {n_delib}")

    print(">> Text — conversational voice + persona ...")
    pre = len(bag)
    for e in load_json(SRC_CONVERSATIONAL):
        p = normalize_text_pair(e)
        if p:
            p["category"] = "conversational"
            bag.append(p)
    n_conv = len(bag) - pre
    print(f"   conversational: {n_conv}")

    print(">> Text — self-model (architecture self-knowledge) ...")
    pre = len(bag)
    for e in load_jsonl(SRC_SELFMODEL):
        p = normalize_text_pair(e)
        if p:
            p["category"] = "self_model"
            bag.append(p)
    n_self = len(bag) - pre
    print(f"   self-model: {n_self}")

    rng.shuffle(bag)
    print(f"   total text: {len(bag)} pairs")
    return bag


# ── VISION ANCHORS ──────────────────────────────────────────────────────────

def draw_shape(d, shape, color, cx, cy, r):
    if shape == "circle":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    elif shape == "square":
        d.rectangle([cx - r, cy - r, cx + r, cy + r], fill=color)
    elif shape == "triangle":
        d.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=color)


COLORS = {
    "red": (220, 30, 30),
    "green": (30, 180, 40),
    "blue": (30, 80, 220),
    "yellow": (240, 220, 30),
    "orange": (240, 140, 30),
    "purple": (150, 40, 200),
}


def build_vision_anchors(rng, n_coco=400) -> list[dict]:
    """Vision pairs anchoring LM↔vision_tower alignment.

    Mostly COCO (preserves what Google's pretraining gave us).
    A few solid colors + simple shapes to bake basic color recognition.
    """
    pairs = []
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # COCO subset — preserve general scene captioning
    src_imgs = SRC_COCO / "images"
    src_jsonl = SRC_COCO / "vision_pairs.jsonl"
    coco_pairs = []
    if src_jsonl.exists():
        for line in src_jsonl.read_text().splitlines():
            line = line.strip()
            if line:
                coco_pairs.append(json.loads(line))
        rng.shuffle(coco_pairs)
        coco_pairs = coco_pairs[:n_coco]
        for p in coco_pairs:
            base = Path(p["image"]).name
            dst_name = f"coco_{base}"
            dst = IMAGES_DIR / dst_name
            src = src_imgs / base
            if not dst.exists() and not dst.is_symlink() and src.exists():
                dst.symlink_to(src.resolve())
            new_p = {
                "image": f"images/{dst_name}",
                "instruction": p.get("instruction", "Describe this image."),
                "output": p.get("output", ""),
                "category": "coco",
            }
            pairs.append(new_p)
    print(f"   COCO: {len(coco_pairs)} pairs")

    # Solid color anchors — just enough to bake color awareness
    for color_name, rgb in COLORS.items():
        for variant in range(2):
            img = Image.new("RGB", (224, 224), color=rgb)
            fname = f"solid_{color_name}_{variant}.png"
            img.save(IMAGES_DIR / fname)
            for prompt, output in [
                ("What color is this?", f"{color_name.capitalize()}."),
                ("Describe the color.", f"{color_name.capitalize()}."),
            ]:
                pairs.append({
                    "image": f"images/{fname}",
                    "instruction": prompt,
                    "output": output,
                    "category": "solid_color",
                })

    # A few simple shapes — preserves shape recognition
    for shape in ("circle", "square", "triangle"):
        for color_name in ("red", "blue", "green"):
            img = Image.new("RGB", (224, 224), "white")
            d = ImageDraw.Draw(img)
            draw_shape(d, shape, COLORS[color_name], 112, 112, 70)
            fname = f"shape_{color_name}_{shape}.png"
            img.save(IMAGES_DIR / fname)
            pairs.append({
                "image": f"images/{fname}",
                "instruction": "Describe the shape and color.",
                "output": f"A {color_name} {shape}.",
                "category": "simple_shape",
            })

    print(f"   total vision anchors: {len(pairs)}")
    return pairs


# ── AUDIO ANCHORS ───────────────────────────────────────────────────────────

def write_wav(path: Path, samples: np.ndarray, rate: int = RATE) -> None:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(samples, -32767, 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(clipped.tobytes())


def build_audio_anchors(rng) -> list[dict]:
    """Audio pairs anchoring LM↔audio_tower alignment.

    SYNTHETIC ONLY — V7/V8 lessons:
      - NO LibriSpeech transcription prompts (polluted other audio responses)
      - NO ESC-50 (categories like "siren", "tank firing" leaked across modalities)
    """
    pairs = []
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    # Silence (30 fixtures)
    for i in range(30):
        dur = rng.choice([1.0, 1.5, 2.0])
        samples = np.zeros(int(RATE * dur), dtype=np.float32)
        fname = f"silence_{i:03d}.wav"
        write_wav(AUDIO_DIR / fname, samples)
        for prompt, output in [
            ("Listen and describe what you hear.", "Silence — there is no sound."),
            ("What sound is in this audio?", "Silence."),
        ]:
            pairs.append({
                "audio": f"audio/{fname}",
                "instruction": prompt,
                "output": output,
                "category": "silence",
            })

    # Tones (40 fixtures, 3 freq bands)
    for i in range(40):
        band = rng.choice(["low", "mid", "high"])
        if band == "low":
            freq = rng.choice([110, 165, 220])
            band_word = "low"
        elif band == "mid":
            freq = rng.choice([330, 440, 660])
            band_word = "mid"
        else:
            freq = rng.choice([880, 1320, 1760])
            band_word = "high"
        dur = rng.choice([1.5, 2.0])
        n = int(RATE * dur)
        t = np.linspace(0, dur, n, endpoint=False)
        samples = (np.sin(2 * np.pi * freq * t) * 6000).astype(np.float32)
        fname = f"tone_{band}_{i:03d}.wav"
        write_wav(AUDIO_DIR / fname, samples)
        for prompt, output in [
            ("What kind of sound is in this audio?",
             f"A {band_word} pure tone, around {freq} Hz."),
            ("How would you describe the pitch — high or low?",
             f"{band_word.capitalize()}."),
        ]:
            pairs.append({
                "audio": f"audio/{fname}",
                "instruction": prompt,
                "output": output,
                "category": "tone",
            })

    # Sweeps (30 fixtures, up/down)
    for i in range(30):
        direction = rng.choice(["up", "down"])
        dur = 2.0
        n = int(RATE * dur)
        t = np.linspace(0, dur, n, endpoint=False)
        if direction == "up":
            freqs = np.linspace(220, 1760, n)
            dir_word = "rising"
        else:
            freqs = np.linspace(1760, 220, n)
            dir_word = "falling"
        samples = (np.sin(2 * np.pi * freqs * t) * 6000).astype(np.float32)
        fname = f"sweep_{direction}_{i:03d}.wav"
        write_wav(AUDIO_DIR / fname, samples)
        for prompt, output in [
            ("Describe the change in pitch over the duration of this audio.",
             f"The pitch is {dir_word}."),
            ("Does the pitch go up or down?", f"{direction.capitalize()}."),
        ]:
            pairs.append({
                "audio": f"audio/{fname}",
                "instruction": prompt,
                "output": output,
                "category": "sweep",
            })

    # Pulses (30 fixtures)
    for i in range(30):
        n_pulses = rng.choice([2, 3, 4, 5, 6])
        pulse_freq = rng.choice([440, 660, 880])
        pulse_dur = 0.15
        gap = 0.3
        total = n_pulses * (pulse_dur + gap)
        n = int(RATE * total)
        samples = np.zeros(n, dtype=np.float32)
        for k in range(n_pulses):
            start = int(RATE * (k * (pulse_dur + gap)))
            end = start + int(RATE * pulse_dur)
            if end > n:
                end = n
            ts = np.linspace(0, pulse_dur, end - start, endpoint=False)
            envelope = np.sin(np.pi * np.arange(end - start) / max(end - start, 1))
            samples[start:end] = np.sin(2 * np.pi * pulse_freq * ts) * 6000 * envelope
        fname = f"pulses_{n_pulses}_{i:03d}.wav"
        write_wav(AUDIO_DIR / fname, samples)
        for prompt, output in [
            ("How many distinct sounds do you hear in this audio?",
             f"{n_pulses} distinct pulses."),
            ("How many pulses?", f"{n_pulses}."),
        ]:
            pairs.append({
                "audio": f"audio/{fname}",
                "instruction": prompt,
                "output": output,
                "category": "pulses",
            })

    # Noise textures (30 fixtures)
    np.random.seed(2026)
    for i in range(30):
        color = rng.choice(["white", "pink", "brown"])
        dur = rng.choice([1.0, 1.5, 2.0])
        n = int(RATE * dur)
        if color == "white":
            samples = np.random.uniform(-3000, 3000, n).astype(np.float32)
        elif color == "pink":
            white = np.random.uniform(-1, 1, n)
            b = 0.95
            out = np.zeros(n)
            out[0] = white[0]
            for k in range(1, n):
                out[k] = b * out[k - 1] + (1 - b) * white[k]
            out = out / np.max(np.abs(out)) * 4000
            samples = out.astype(np.float32)
        else:
            white = np.random.uniform(-1, 1, n)
            out = np.cumsum(white)
            out = out / np.max(np.abs(out)) * 6000
            samples = out.astype(np.float32)
        fname = f"noise_{color}_{i:03d}.wav"
        write_wav(AUDIO_DIR / fname, samples)
        for prompt, output in [
            ("Describe the texture of this audio.",
             f"{color.capitalize()} noise — a rough, uniform hiss."),
            ("What kind of sound is this?",
             f"{color.capitalize()} noise."),
        ]:
            pairs.append({
                "audio": f"audio/{fname}",
                "instruction": prompt,
                "output": output,
                "category": "noise",
            })

    print(f"   total audio anchors: {len(pairs)}")
    return pairs


# ── MAIN ────────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 60)
    print("  Core 2.0 unified curriculum (xln recipe)")
    print("=" * 60)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(2026)
    np.random.seed(2026)

    # Text
    text_pairs = build_text(rng)

    # Vision anchors
    print("\n>> Vision anchors ...")
    vision_pairs = build_vision_anchors(rng, n_coco=400)
    rng.shuffle(vision_pairs)

    # Audio anchors
    print("\n>> Audio anchors ...")
    audio_pairs = build_audio_anchors(rng)
    rng.shuffle(audio_pairs)

    # Write outputs
    text_jsonl = OUT_DIR / "text.jsonl"
    with open(text_jsonl, "w") as f:
        for p in text_pairs:
            f.write(json.dumps(p) + "\n")

    vision_jsonl = OUT_DIR / "vision_pairs.jsonl"
    with open(vision_jsonl, "w") as f:
        for p in vision_pairs:
            f.write(json.dumps(p) + "\n")

    audio_jsonl = OUT_DIR / "audio_pairs.jsonl"
    with open(audio_jsonl, "w") as f:
        for p in audio_pairs:
            f.write(json.dumps(p) + "\n")

    n_image_files = len([p for p in IMAGES_DIR.iterdir() if p.is_file() or p.is_symlink()])
    n_audio_files = len([p for p in AUDIO_DIR.iterdir() if p.is_file() or p.is_symlink()])

    print(f"\n>> Core 2.0 curriculum summary:")
    print(f"   Text:   {len(text_pairs)} pairs")
    print(f"   Vision: {len(vision_pairs)} pairs over {n_image_files} images")
    print(f"   Audio:  {len(audio_pairs)} pairs over {n_audio_files} audio files")
    print(f"   Total:  {len(text_pairs) + len(vision_pairs) + len(audio_pairs)} pairs")
    print(f"   Output: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
