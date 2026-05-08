#!/usr/bin/env python3
"""Build the v8 unified curriculum — vision + audio + text in one pass.

V7AUDIO showed two problems even though it didn't break vision:
  - LM shortcut on prompt-template patterns instead of attending to audio
    (4 prompt phrasings per fixture wasn't enough variety)
  - LibriSpeech "Transcribe this" prompts polluted other audio responses
    (the LM started trying to transcribe noise textures)

V8 design:
  - Single LoRA from Gemma4-E4B-GAIA-Unified-v5-Multimodal (no prior LoRA
    merged in — clean slate but identity preserved)
  - 8-12 prompt phrasings per audio fixture
  - NO LibriSpeech (drops the transcription bias)
  - Output vocabulary aligned with cognitive_test_battery validators
    (say "rising"/"falling", not "ascending"/"descending"; "noise" plus
    "rough"/"rumble" for textures; etc.)
  - All three modalities trained together, not chained

Output:
  knowledge/curricula/core-multimodal-v8unified/
    images/                  (symlinks to COCO + v6 vision sources)
    audio/                   (synth WAVs + ESC-50 symlinks)
    vision_pairs.jsonl       (~3500 vision pairs)
    audio_pairs.jsonl        (~4800 audio pairs)
"""
from __future__ import annotations

import csv
import json
import math
import random
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

PROJ = Path("/gaia/GAIA_Project")
OUT_DIR = PROJ / "knowledge/curricula/core-multimodal-v8unified"
IMAGES_DIR = OUT_DIR / "images"
AUDIO_DIR = OUT_DIR / "audio"

# Source curricula (existing on disk)
SRC_COCO = PROJ / "knowledge/curricula/core-multimodal-coco"
SRC_PRIMS_V3 = PROJ / "knowledge/curricula/core-multimodal"          # 406 v3 primitives
ESC50_ROOT = PROJ / "data/datasets/ESC-50-master"

RATE = 16000


# =============================================================================
# AUDIO
# =============================================================================

def write_wav(path: Path, samples: np.ndarray, rate: int = RATE) -> None:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(samples, -32767, 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(clipped.tobytes())


# Synthetic audio generators — same shapes as V7 but vocab-aligned outputs

def gen_silence(rng):
    duration = rng.choice([0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0])
    return np.zeros(int(RATE * duration), dtype=np.float32), {
        "category": "silence", "duration": duration}


def gen_tone(rng):
    band = rng.choice(["low", "mid", "high"])
    if band == "low":
        freq = rng.choice([90, 110, 165, 220, 250])
    elif band == "mid":
        freq = rng.choice([330, 440, 550, 660, 770])
    else:
        freq = rng.choice([880, 1100, 1320, 1540, 1760, 2200])
    duration = rng.choice([0.8, 1.0, 1.5, 2.0, 2.5])
    n = int(RATE * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    amp = rng.choice([4500, 5500, 6500, 7500])
    return (np.sin(2 * np.pi * freq * t) * amp).astype(np.float32), {
        "category": "tone", "freq": freq, "band": band, "duration": duration}


def gen_sweep(rng):
    direction = rng.choice(["up", "down", "up_down", "down_up"])
    f_low = rng.choice([110, 165, 220, 330])
    f_high = rng.choice([880, 1320, 1760, 2200, 2640])
    duration = rng.choice([1.5, 2.0, 2.5, 3.0])
    n = int(RATE * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    if direction == "up":
        freqs = np.linspace(f_low, f_high, n)
    elif direction == "down":
        freqs = np.linspace(f_high, f_low, n)
    elif direction == "up_down":
        half = n // 2
        freqs = np.concatenate([np.linspace(f_low, f_high, half),
                                np.linspace(f_high, f_low, n - half)])
    else:
        half = n // 2
        freqs = np.concatenate([np.linspace(f_high, f_low, half),
                                np.linspace(f_low, f_high, n - half)])
    samples = (np.sin(2 * np.pi * freqs * t) * 6000).astype(np.float32)
    return samples, {"category": "sweep", "direction": direction,
                     "f_low": f_low, "f_high": f_high, "duration": duration}


def gen_pulses(rng):
    n_pulses = rng.choice([2, 3, 4, 5, 6, 7, 8])
    pulse_freq = rng.choice([330, 440, 660, 880, 1100, 1320])
    pulse_dur = rng.choice([0.08, 0.12, 0.15, 0.20, 0.25])
    gap = rng.choice([0.15, 0.20, 0.30, 0.40])
    total = n_pulses * (pulse_dur + gap)
    n = int(RATE * total)
    samples = np.zeros(n, dtype=np.float32)
    for i in range(n_pulses):
        start = int(RATE * (i * (pulse_dur + gap)))
        end = start + int(RATE * pulse_dur)
        if end > n:
            end = n
        ts = np.linspace(0, pulse_dur, end - start, endpoint=False)
        envelope = np.sin(np.pi * np.arange(end - start) / max(end - start, 1))
        samples[start:end] = np.sin(2 * np.pi * pulse_freq * ts) * 6000 * envelope
    return samples, {"category": "pulses", "n_pulses": n_pulses,
                     "freq": pulse_freq, "pulse_dur": pulse_dur, "gap": gap}


def gen_noise(rng):
    color = rng.choice(["white", "pink", "brown", "burst"])
    duration = rng.choice([1.0, 1.5, 2.0, 2.5])
    n = int(RATE * duration)
    if color == "white":
        samples = np.random.uniform(-3000, 3000, n).astype(np.float32)
    elif color == "pink":
        white = np.random.uniform(-1, 1, n)
        b = 0.95
        out = np.zeros(n)
        out[0] = white[0]
        for i in range(1, n):
            out[i] = b * out[i - 1] + (1 - b) * white[i]
        out = out / np.max(np.abs(out)) * 4000
        samples = out.astype(np.float32)
    elif color == "brown":
        white = np.random.uniform(-1, 1, n)
        out = np.cumsum(white)
        out = out / np.max(np.abs(out)) * 6000
        samples = out.astype(np.float32)
    else:
        samples = np.zeros(n, dtype=np.float32)
        burst_dur = int(RATE * rng.choice([0.10, 0.15, 0.20, 0.30]))
        start = rng.randint(0, max(0, n - burst_dur - 100))
        samples[start:start + burst_dur] = np.random.uniform(-4000, 4000, burst_dur)
    return samples, {"category": "noise", "color": color, "duration": duration}


def gen_chord(rng):
    base = rng.choice([165, 220, 330, 440])
    chord_type = rng.choice(["major", "minor", "fifth", "diminished"])
    if chord_type == "major":
        ratios = [1.0, 1.25, 1.5]
    elif chord_type == "minor":
        ratios = [1.0, 1.2, 1.5]
    elif chord_type == "fifth":
        ratios = [1.0, 1.5]
    else:
        ratios = [1.0, 1.2, 1.4]
    duration = rng.choice([1.5, 2.0, 2.5, 3.0])
    n = int(RATE * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    samples = np.zeros(n, dtype=np.float32)
    for r in ratios:
        samples += np.sin(2 * np.pi * (base * r) * t) * (4500 / len(ratios))
    return samples, {"category": "chord", "base": base,
                     "chord_type": chord_type, "duration": duration}


def gen_vibrato(rng):
    base_freq = rng.choice([330, 440, 660])
    duration = rng.choice([2.0, 2.5, 3.0])
    rate_hz = rng.choice([4, 5, 6, 7, 8])
    depth = rng.choice([0.03, 0.05, 0.08])
    n = int(RATE * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    modulation = 1 + depth * np.sin(2 * np.pi * rate_hz * t)
    freqs = base_freq * modulation
    samples = (np.sin(2 * np.pi * freqs * t) * 6000).astype(np.float32)
    return samples, {"category": "vibrato", "base": base_freq,
                     "rate_hz": rate_hz, "duration": duration}


# ── Vocab-aligned prompt + output pools ─────────────────────────────────────
#
# Each entry is (prompt, output_template_with_format_kwargs).
# Output uses validator-friendly vocabulary.

PROMPT_POOLS = {
    "silence": [
        ("Listen to this audio and describe what you hear.", "Silence — there is no sound."),
        ("What sound is in this audio?", "Silence."),
        ("Describe the audio.", "Silence, no audible sound."),
        ("What kind of sound is this?", "Silent — the audio is quiet."),
        ("Listen. What do you hear?", "Nothing — silence."),
        ("Is there sound in this audio?", "No, the audio is silent."),
        ("Describe what you hear.", "Quiet, empty — silence."),
        ("What's the audio content?", "Silence; no sound is present."),
        ("Tell me about this audio.", "It's silence, no sound."),
        ("How would you describe this audio?", "Quiet and silent."),
        ("What's playing?", "Nothing — silence."),
        ("Audio description?", "Silence."),
    ],
    "tone": [
        ("What kind of sound is in this audio? Be brief.",
         "A {band}-pitched pure tone, around {freq} Hz."),
        ("Listen to this audio and describe what you hear.",
         "A single sustained {band} tone, like a {freq} Hz beep."),
        ("Describe the audio.", "A {band} pure tone — a steady beep."),
        ("What is the dominant pitch?", "A {band} tone at about {freq} Hz."),
        ("How would you describe the pitch — high or low?", "{band_capital} pitch."),
        ("Is this audio a tone or noise?",
         "A pure tone — {band} pitch around {freq} Hz."),
        ("Describe the texture of this audio.",
         "A clean, steady tone — {band} pitch."),
        ("Single tone or multiple tones?", "A single {band} tone."),
        ("Is the pitch steady or does it wobble?",
         "Steady — a {band} pure tone."),
        ("What note quality?", "A {band} sine tone at {freq} Hz."),
        ("How would you describe this sound?",
         "A {band} pure tone, like a {freq} Hz beep."),
        ("Is the sound clean or noisy?",
         "Clean — a pure {band} tone."),
    ],
    "sweep": [
        ("Describe the change in pitch over the duration of this audio.",
         "The pitch is {direction_word}."),
        ("Describe how the pitch changes over the duration.",
         "The pitch is {direction_word}, sweeping between {f_low} and {f_high} Hz."),
        ("Listen to this audio. Is the pitch rising or falling?",
         "{direction_word_capital}."),
        ("What kind of sound is this?",
         "A frequency sweep — {direction_word}."),
        ("Describe the audio.", "A {direction_word} pitch sweep."),
        ("Does the pitch go up or down?", "{direction_simple}."),
        ("How does the pitch change?",
         "It is {direction_word}, sweeping {f_low}-{f_high} Hz."),
        ("Is the pitch constant or changing?",
         "Changing — the pitch is {direction_word}."),
        ("Describe this sweep.",
         "A {direction_word} sweep from {f_low} to {f_high} Hz."),
        ("Is this audio a tone or a sweep?",
         "A sweep — pitch {direction_word}."),
        ("Tell me about the pitch.",
         "The pitch is {direction_word}."),
        ("Single pitch or changing?",
         "Changing — pitch is {direction_word}."),
    ],
    "pulses": [
        ("How many distinct sounds do you hear in this audio?",
         "{n_pulses} distinct pulses."),
        ("How many distinct pulses do you hear?",
         "{n_pulses}."),
        ("What is the rhythm of this audio?",
         "{n_pulses} short pulses at a steady rhythm."),
        ("Listen and describe what you hear.",
         "{n_pulses} short repeated beeps."),
        ("Describe the audio.",
         "{n_pulses} discrete pulses, like a beeping pattern."),
        ("Is the audio continuous or pulsed?",
         "Pulsed — {n_pulses} distinct pulses."),
        ("Count the sounds.",
         "{n_pulses}."),
        ("How many beeps?",
         "{n_pulses} beeps."),
        ("Single sound or multiple?",
         "Multiple — {n_pulses} pulses."),
        ("Describe the pattern.",
         "A pulse pattern — {n_pulses} beeps."),
        ("How many sounds in total?",
         "{n_pulses}."),
        ("Is this rhythmic?",
         "Yes — {n_pulses} regular pulses."),
    ],
    "noise": [
        ("Describe the texture of this audio.",
         "{color_word_capital} noise — a rough, uniform hiss."),
        ("What kind of sound is this?",
         "{color_word_capital} noise."),
        ("Listen and describe what you hear.",
         "Static — {color_word} noise."),
        ("Describe the audio.",
         "{color_word_capital} noise, like static or hiss."),
        ("Is this a tone or noise?",
         "Noise — {color_word} noise, like static."),
        ("Describe the texture.",
         "Rough, uniform texture — {color_word} noise."),
        ("How would you describe the sound?",
         "{color_word_capital} noise, with a {texture_word} texture."),
        ("Is the sound clean or noisy?",
         "Noisy — {color_word} noise."),
        ("Tell me about the audio.",
         "It's {color_word} noise — like static or hiss."),
        ("Single tone or noise?",
         "Noise — {color_word} type."),
        ("What does this sound like?",
         "Like static or hiss — {color_word} noise."),
        ("Describe what you hear.",
         "Noise — specifically {color_word} noise, with a hissing quality."),
    ],
    "chord": [
        ("Listen to this audio. Is it a single tone or a chord?",
         "A {chord_type} chord — multiple tones at once."),
        ("What kind of sound is this?",
         "A {chord_type} chord."),
        ("Describe the audio.",
         "Multiple tones together — a {chord_type} chord."),
        ("How many tones do you hear?",
         "Multiple — a {chord_type} chord."),
        ("Single tone or multiple tones?",
         "Multiple — this is a {chord_type} chord."),
        ("Is this a chord?",
         "Yes — a {chord_type} chord."),
        ("Describe the harmony.",
         "A {chord_type} chord with multiple pitches."),
        ("What's the texture?",
         "Harmonic — a {chord_type} chord."),
        ("Listen. Tone or chord?",
         "Chord — {chord_type}."),
        ("Tell me about this audio.",
         "It's a {chord_type} chord — multiple notes at once."),
        ("How would you describe this?",
         "Multiple tones together forming a {chord_type} chord."),
        ("Is this monophonic or polyphonic?",
         "Polyphonic — a {chord_type} chord."),
    ],
    "vibrato": [
        ("Is the pitch steady or does it wobble?",
         "It wobbles — vibrato at about {rate_hz} Hz."),
        ("Listen to this audio. Describe what you hear.",
         "A tone with vibrato — the pitch wobbles."),
        ("Does the pitch change?",
         "Yes, it wobbles — vibrato around {base} Hz."),
        ("Single pitch or modulating?",
         "Modulating — vibrato wobble."),
        ("Describe the audio.",
         "A tone with wobbling pitch — vibrato."),
        ("Is the pitch constant?",
         "No, it wobbles — vibrato modulation."),
        ("How would you describe the pitch?",
         "Wobbling — vibrato at {rate_hz} Hz."),
        ("Tell me about this sound.",
         "A vibrato tone — the pitch wobbles slightly."),
        ("Steady or oscillating?",
         "Oscillating — vibrato wobble."),
        ("Pitch behavior?",
         "Vibrato — wobbling pitch."),
    ],
}


def render_synth(template: str, meta: dict) -> str:
    direction_word = {
        "up": "rising", "down": "falling",
        "up_down": "rising then falling",
        "down_up": "falling then rising",
    }.get(meta.get("direction"), "")
    direction_simple = {
        "up": "up", "down": "down",
        "up_down": "up then down", "down_up": "down then up",
    }.get(meta.get("direction"), "")
    color_word = meta.get("color", "")
    texture_word = {
        "white": "rough", "pink": "warm rough",
        "brown": "deep rumbling", "burst": "sudden",
    }.get(color_word, "rough")
    return template.format(
        band=meta.get("band", ""),
        band_capital=meta.get("band", "").capitalize(),
        freq=meta.get("freq", ""),
        direction_word=direction_word,
        direction_word_capital=direction_word.capitalize(),
        direction_simple=direction_simple.capitalize(),
        f_low=meta.get("f_low", ""),
        f_high=meta.get("f_high", ""),
        n_pulses=meta.get("n_pulses", ""),
        color_word=color_word,
        color_word_capital=color_word.capitalize(),
        texture_word=texture_word,
        chord_type=meta.get("chord_type", ""),
        base=meta.get("base", ""),
        rate_hz=meta.get("rate_hz", ""),
    )


SYNTH_GEN = {
    "silence": (gen_silence, 60),
    "tone": (gen_tone, 100),
    "sweep": (gen_sweep, 70),
    "pulses": (gen_pulses, 70),
    "noise": (gen_noise, 70),
    "chord": (gen_chord, 50),
    "vibrato": (gen_vibrato, 30),
}

PROMPTS_PER_FIXTURE = 8  # vs V7's 4


def build_synthetic_audio(rng, pairs):
    n = 0
    for cat, (fn, n_samples) in SYNTH_GEN.items():
        prompt_pool = PROMPT_POOLS[cat]
        for i in range(n_samples):
            samples, meta = fn(rng)
            fname = f"synth_{cat}_{i:03d}.wav"
            write_wav(AUDIO_DIR / fname, samples)
            # Sample N distinct prompts from pool
            prompts_picked = rng.sample(prompt_pool, min(PROMPTS_PER_FIXTURE, len(prompt_pool)))
            for prompt_template, output_template in prompts_picked:
                pairs.append({
                    "audio": f"audio/{fname}",
                    "instruction": prompt_template,
                    "output": render_synth(output_template, meta),
                    "category": f"synth_{cat}",
                })
                n += 1
    return n


# ── ESC-50 (real environmental sounds) — multiple prompts per fixture ────────

ESC50_DESCRIPTIONS = {
    "dog": "A dog barking",
    "rooster": "A rooster crowing",
    "pig": "A pig grunting",
    "cow": "A cow mooing",
    "frog": "A frog croaking",
    "cat": "A cat meowing",
    "hen": "A hen clucking",
    "insects": "Insects buzzing",
    "sheep": "A sheep bleating",
    "crow": "A crow cawing",
    "rain": "Rain falling",
    "sea_waves": "Sea waves crashing",
    "crackling_fire": "A crackling fire",
    "crickets": "Crickets chirping",
    "chirping_birds": "Birds chirping",
    "water_drops": "Water dripping",
    "wind": "Wind blowing",
    "pouring_water": "Water pouring",
    "toilet_flush": "A toilet flushing",
    "thunderstorm": "A thunderstorm",
    "crying_baby": "A baby crying",
    "sneezing": "Someone sneezing",
    "clapping": "Hands clapping",
    "breathing": "Breathing sounds",
    "coughing": "Someone coughing",
    "footsteps": "Footsteps walking",
    "laughing": "Someone laughing",
    "brushing_teeth": "Teeth brushing",
    "snoring": "Someone snoring",
    "drinking_sipping": "Someone drinking or sipping",
    "door_wood_knock": "A wooden door knocking",
    "mouse_click": "A mouse clicking",
    "keyboard_typing": "Typing on a keyboard",
    "door_wood_creaks": "A wooden door creaking",
    "can_opening": "A can opening",
    "washing_machine": "A washing machine running",
    "vacuum_cleaner": "A vacuum cleaner running",
    "clock_alarm": "A clock alarm ringing",
    "clock_tick": "A clock ticking",
    "glass_breaking": "Glass breaking",
    "helicopter": "A helicopter flying",
    "chainsaw": "A chainsaw running",
    "siren": "A siren wailing",
    "car_horn": "A car horn honking",
    "engine": "An engine running",
    "train": "A train passing",
    "church_bells": "Church bells ringing",
    "airplane": "An airplane flying overhead",
    "fireworks": "Fireworks exploding",
    "hand_saw": "A hand saw cutting",
}

ESC50_PROMPT_POOL = [
    "What sound is in this audio?",
    "Listen to this audio and describe what you hear.",
    "Describe the audio.",
    "What is happening in this audio?",
    "What do you hear?",
    "Tell me about this sound.",
    "Identify the sound.",
    "What's making this sound?",
]


def build_esc50_audio(rng, pairs, per_cat=10, prompts_per_clip=4):
    csv_path = ESC50_ROOT / "meta/esc50.csv"
    audio_root = ESC50_ROOT / "audio"
    if not csv_path.exists() or not audio_root.exists():
        print(f"  ESC-50: missing at {ESC50_ROOT} — skipping")
        return 0
    by_cat: dict[str, list[str]] = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            by_cat.setdefault(row["category"], []).append(row["filename"])
    n = 0
    for cat, files in by_cat.items():
        description = ESC50_DESCRIPTIONS.get(cat)
        if not description:
            continue
        rng.shuffle(files)
        for fn in files[:per_cat]:
            src = audio_root / fn
            if not src.exists():
                continue
            dst_name = f"esc50_{cat}_{fn}"
            dst = AUDIO_DIR / dst_name
            if not dst.exists() and not dst.is_symlink():
                dst.symlink_to(src.resolve())
            picked = rng.sample(ESC50_PROMPT_POOL, prompts_per_clip)
            for prompt in picked:
                pairs.append({
                    "audio": f"audio/{dst_name}",
                    "instruction": prompt,
                    "output": description + ".",
                    "category": "esc50",
                })
                n += 1
    return n


# =============================================================================
# VISION
# =============================================================================

def draw_shape(d, shape, color, cx, cy, r):
    if shape == "circle":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    elif shape == "square":
        d.rectangle([cx - r, cy - r, cx + r, cy + r], fill=color)
    elif shape == "triangle":
        d.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=color)
    elif shape == "diamond":
        d.polygon([(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)], fill=color)
    elif shape == "star":
        pts = []
        for i in range(10):
            a = (-90 + i * 36) * math.pi / 180
            rd = r if i % 2 == 0 else r * 0.45
            pts.append((cx + rd * math.cos(a), cy + rd * math.sin(a)))
        d.polygon(pts, fill=color)
    elif shape == "heart":
        pts = []
        for i in range(180):
            t = i * 2 * math.pi / 180
            x = 16 * (math.sin(t) ** 3)
            y = -(13 * math.cos(t) - 5 * math.cos(2 * t)
                  - 2 * math.cos(3 * t) - math.cos(4 * t))
            pts.append((cx + x * r / 17, cy + y * r / 17))
        d.polygon(pts, fill=color)
    elif shape == "cross":
        thick = max(4, r // 3)
        d.rectangle([cx - thick, cy - r, cx + thick, cy + r], fill=color)
        d.rectangle([cx - r, cy - thick, cx + r, cy + thick], fill=color)


COLORS = {
    "red": [(220, 30, 30), (200, 50, 50), (240, 60, 60)],
    "green": [(30, 180, 40), (40, 200, 60), (60, 220, 80)],
    "blue": [(30, 80, 220), (40, 100, 240), (60, 120, 255)],
    "yellow": [(240, 220, 30), (250, 230, 60), (240, 200, 40)],
    "orange": [(240, 140, 30), (250, 160, 60), (220, 120, 40)],
    "purple": [(150, 40, 200), (170, 60, 220), (130, 30, 180)],
    "pink": [(240, 130, 170), (250, 150, 190)],
    "cyan": [(40, 200, 220), (60, 220, 230)],
}
SHAPES = ["circle", "square", "triangle", "star", "diamond", "heart", "cross"]


# Background generators (for varied-bg singles)

def make_white_bg(size, rng):
    return Image.new("RGB", (size, size), color=(252, 252, 252))


def make_gradient_bg(size, rng):
    img = Image.new("RGB", (size, size))
    pix = img.load()
    c1 = (rng.randint(60, 180), rng.randint(60, 180), rng.randint(60, 180))
    c2 = (rng.randint(60, 220), rng.randint(60, 220), rng.randint(60, 220))
    style = rng.choice(["horiz", "vert", "diag", "radial"])
    for y in range(size):
        for x in range(size):
            if style == "horiz":
                t = x / size
            elif style == "vert":
                t = y / size
            elif style == "diag":
                t = (x + y) / (2 * size)
            else:
                cx, cy = size / 2, size / 2
                t = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5 / (size * 0.7)
                t = min(1.0, t)
            pix[x, y] = (
                int(c1[0] * (1 - t) + c2[0] * t),
                int(c1[1] * (1 - t) + c2[1] * t),
                int(c1[2] * (1 - t) + c2[2] * t),
            )
    return img


def make_dark_bg(size, rng):
    return Image.new("RGB", (size, size),
                     color=(rng.randint(20, 60), rng.randint(20, 60), rng.randint(20, 60)))


def make_textured_bg(size, rng):
    img = Image.new("RGB", (size, size),
                    color=(rng.randint(80, 180), rng.randint(80, 180), rng.randint(80, 180)))
    draw = ImageDraw.Draw(img)
    for _ in range(rng.randint(8, 16)):
        cx = rng.randint(0, size)
        cy = rng.randint(0, size)
        rad = rng.randint(15, 50)
        c = (rng.randint(60, 220), rng.randint(60, 220), rng.randint(60, 220))
        draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=c)
    return img.filter(ImageFilter.GaussianBlur(radius=6))


BG_MAKERS = [
    ("white", make_white_bg),
    ("gradient", make_gradient_bg),
    ("dark", make_dark_bg),
    ("textured", make_textured_bg),
]


SOLID_COLOR_PROMPTS = [
    ("What color is this?", "{color}."),
    ("What is the dominant color?", "{color}."),
    ("Describe the color.", "{color}."),
    ("Color of this image?", "{color}."),
    ("What single color fills this image?", "{color}."),
]


SHAPE_ON_BG_PROMPTS = [
    ("Describe the shape and color of the object in this image.",
     "A {color} {shape}."),
    ("What is in this image?", "A {color} {shape}."),
    ("What color is the {shape} in this image?", "{color_capital}."),
    ("What shape is in this image? What color?",
     "A {shape}, colored {color}."),
    ("Describe the foreground object.",
     "A {color} {shape}."),
    ("What's the main object?", "A {color} {shape}."),
]


SHAPE_ONLY_PROMPTS = [
    ("What shape is in this image?", "A {shape}."),
    ("Describe the shape.", "A {shape}."),
    ("What's the geometric shape?", "{shape_capital}."),
]


COLOR_ONLY_PROMPTS = [
    ("What color is the object?", "{color_capital}."),
    ("What color is the main shape?", "{color_capital}."),
    ("Color of the foreground?", "{color_capital}."),
]


def build_solid_colors(rng, pairs, n_per_color=8):
    n = 0
    for color_name, shades in COLORS.items():
        for i in range(n_per_color):
            size = rng.choice([192, 224, 256])
            rgb = rng.choice(shades)
            img = Image.new("RGB", (size, size), color=rgb)
            fname = f"solid_{color_name}_{i:02d}.png"
            img.save(IMAGES_DIR / fname)
            picked = rng.sample(SOLID_COLOR_PROMPTS, min(4, len(SOLID_COLOR_PROMPTS)))
            for prompt_t, out_t in picked:
                pairs.append({
                    "image": f"images/{fname}",
                    "instruction": prompt_t,
                    "output": out_t.format(color=color_name),
                    "category": "solid_color",
                })
                n += 1
    return n


def build_shape_on_bg(rng, pairs, n_per_combo=2, prompts_per=4):
    n = 0
    for shape in SHAPES:
        for color_name in COLORS:
            for bg_name, bg_maker in BG_MAKERS:
                for i in range(n_per_combo):
                    size = rng.choice([224, 256])
                    img = bg_maker(size, rng)
                    draw = ImageDraw.Draw(img)
                    rgb = rng.choice(COLORS[color_name])
                    cx = size // 2 + rng.randint(-15, 15)
                    cy = size // 2 + rng.randint(-15, 15)
                    r = rng.randint(int(size * 0.20), int(size * 0.32))
                    draw_shape(draw, shape, rgb, cx, cy, r)
                    fname = f"shape_{shape}_{color_name}_{bg_name}_{i:02d}.png"
                    img.save(IMAGES_DIR / fname)
                    picked = rng.sample(SHAPE_ON_BG_PROMPTS,
                                        min(prompts_per, len(SHAPE_ON_BG_PROMPTS)))
                    for prompt_t, out_t in picked:
                        pairs.append({
                            "image": f"images/{fname}",
                            "instruction": prompt_t.format(shape=shape, color=color_name),
                            "output": out_t.format(
                                shape=shape, color=color_name,
                                color_capital=color_name.capitalize(),
                                shape_capital=shape.capitalize()),
                            "category": "shape_on_bg",
                        })
                        n += 1
    return n


def build_multi_object_scenes(rng, pairs, n_scenes=80):
    """Multi-object scenes with battery-aligned disambiguation queries."""
    n = 0
    for scene_idx in range(n_scenes):
        size = rng.choice([320, 384, 448])
        bg_name, bg_maker = rng.choice(BG_MAKERS)
        img = bg_maker(size, rng) if rng.random() > 0.3 else Image.new("RGB", (size, size), "white")
        draw = ImageDraw.Draw(img)
        n_shapes = rng.choice([2, 3, 4])
        chosen_shapes = rng.sample(SHAPES, n_shapes)
        chosen_colors = rng.sample(list(COLORS.keys()), n_shapes)
        cell = size // n_shapes
        objects = []
        for i, (shape, color_name) in enumerate(zip(chosen_shapes, chosen_colors)):
            cx = cell * i + cell // 2 + rng.randint(-10, 10)
            cy = size // 2 + rng.randint(-25, 25)
            r = rng.randint(int(size * 0.10), int(size * 0.16))
            rgb = rng.choice(COLORS[color_name])
            draw_shape(draw, shape, rgb, cx, cy, r)
            objects.append((shape, color_name))

        fname = f"scene_{scene_idx:03d}.png"
        img.save(IMAGES_DIR / fname)

        # Per-scene queries:
        # 1. Color of each shape (battery vis-d01..d03 pattern)
        for shape, color in objects:
            pairs.append({
                "image": f"images/{fname}",
                "instruction": f"What color is the {shape} in this image?",
                "output": color.capitalize() + ".",
                "category": "scene_color_q",
            })
            n += 1
        # 2. Count
        pairs.append({
            "image": f"images/{fname}",
            "instruction": "How many distinct shapes are in this image?",
            "output": f"{n_shapes}.",
            "category": "scene_count",
        })
        # 3. Yes/no presence for one of the colors in scene
        s, c = objects[0]
        pairs.append({
            "image": f"images/{fname}",
            "instruction": f"Is there a {c} shape in this image? Yes or no.",
            "output": "Yes.",
            "category": "scene_yesno",
        })
        # 4. Yes/no for a missing color
        missing = next((cn for cn in COLORS if cn not in chosen_colors), None)
        if missing:
            pairs.append({
                "image": f"images/{fname}",
                "instruction": f"Is there a {missing} shape in this image? Yes or no.",
                "output": "No.",
                "category": "scene_yesno",
            })
        n += 2
    return n


def symlink_coco(rng, pairs, n_pairs=2000):
    """Symlink COCO sources + emit caption pairs."""
    src_imgs = SRC_COCO / "images"
    src_jsonl = SRC_COCO / "vision_pairs.jsonl"
    if not src_imgs.exists() or not src_jsonl.exists():
        print(f"  COCO: missing at {SRC_COCO} — skipping")
        return 0
    coco_pairs = []
    for line in src_jsonl.read_text().splitlines():
        line = line.strip()
        if line:
            coco_pairs.append(json.loads(line))
    rng.shuffle(coco_pairs)
    coco_pairs = coco_pairs[:n_pairs]
    n = 0
    for p in coco_pairs:
        base = Path(p["image"]).name
        dst_name = f"coco_{base}"
        dst = IMAGES_DIR / dst_name
        src = src_imgs / base
        if not dst.exists() and not dst.is_symlink() and src.exists():
            dst.symlink_to(src.resolve())
        new_p = dict(p)
        new_p["image"] = f"images/{dst_name}"
        new_p["category"] = "coco"
        pairs.append(new_p)
        n += 1
    return n


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    print("=" * 60)
    print("  Multimodal Core v8unified — vision + audio + text in one curriculum")
    print("=" * 60)

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    rng = random.Random(2026)
    np.random.seed(2026)

    # ── AUDIO ────────────────────────────────────────────────
    audio_pairs: list[dict] = []
    print("\n>> Audio: synthetic primitives (8 prompts/fixture) ...")
    n_synth = build_synthetic_audio(rng, audio_pairs)
    print(f"   synth: {n_synth} pairs")

    print(">> Audio: ESC-50 (4 prompts/clip) ...")
    n_esc = build_esc50_audio(rng, audio_pairs, per_cat=10, prompts_per_clip=4)
    print(f"   esc50: {n_esc} pairs")

    rng.shuffle(audio_pairs)
    audio_jsonl = OUT_DIR / "audio_pairs.jsonl"
    with open(audio_jsonl, "w") as f:
        for p in audio_pairs:
            f.write(json.dumps(p) + "\n")
    print(f"   total audio pairs: {len(audio_pairs)}")

    # ── VISION ────────────────────────────────────────────────
    vision_pairs: list[dict] = []
    print("\n>> Vision: solid colors ...")
    n_sc = build_solid_colors(rng, vision_pairs, n_per_color=8)
    print(f"   solid_color: {n_sc} pairs")

    print(">> Vision: shape on varied bg ...")
    n_sb = build_shape_on_bg(rng, vision_pairs, n_per_combo=1, prompts_per=2)
    print(f"   shape_on_bg: {n_sb} pairs")

    print(">> Vision: multi-object scenes (battery-aligned) ...")
    n_ms = build_multi_object_scenes(rng, vision_pairs, n_scenes=80)
    print(f"   scenes: {n_ms} pairs")

    print(">> Vision: COCO captions ...")
    n_co = symlink_coco(rng, vision_pairs, n_pairs=2000)
    print(f"   coco: {n_co} pairs")

    rng.shuffle(vision_pairs)
    vision_jsonl = OUT_DIR / "vision_pairs.jsonl"
    with open(vision_jsonl, "w") as f:
        for p in vision_pairs:
            f.write(json.dumps(p) + "\n")
    print(f"   total vision pairs: {len(vision_pairs)}")

    # Summary
    n_audio_files = len([p for p in AUDIO_DIR.iterdir() if p.is_file() or p.is_symlink()])
    n_image_files = len([p for p in IMAGES_DIR.iterdir() if p.is_file() or p.is_symlink()])
    print(f"\n>> V8 unified curriculum summary:")
    print(f"   Audio: {len(audio_pairs)} pairs over {n_audio_files} files ({n_synth} synth + {n_esc} ESC-50)")
    print(f"   Vision: {len(vision_pairs)} pairs over {n_image_files} files")
    print(f"   Output: {OUT_DIR}")
    print(f"   No LibriSpeech (V7 lesson). Synth has {PROMPTS_PER_FIXTURE} prompts/fixture (V7 had 4).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
