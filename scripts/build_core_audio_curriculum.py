#!/usr/bin/env python3
"""Build a synthetic audio-text curriculum for Core's audio side.

V2/V3/V4/V5 trained the language_model on vision pairs only. The audio
tower has been attached (graft from base) but never trained. This produces
the ~0/5 battery results documented in 7rq's predecessors — the model
literally cannot decode audio bytes, so it confabulates or echoes the
prompt template.

The 8ki battery's audio section tests acoustic primitives (silence, pure
tones, frequency sweeps, repeated pulses, white noise) so a synthetic
curriculum that mirrors those exact patterns is the right first move:

  - 30 silence variants × 4 prompts = 120 pairs
  - 60 pure tones (3 freq bands × 4 durations × 5 starting phases) × 4 prompts = 240 pairs
  - 40 sweeps (up/down/up-down × freq ranges) × 4 prompts = 160 pairs
  - 40 pulse patterns (1-6 pulses, varied gap and pitch) × 4 prompts = 160 pairs
  - 40 noise textures (white/pink/brown/burst) × 4 prompts = 160 pairs
  - 30 mixed-tone chords (2-3 simultaneous tones) × 4 prompts = 120 pairs

Total ~960 pairs over ~240 unique audio files. Each WAV is 16 kHz mono,
1-3 seconds, written as PCM16 (small footprint, ~32-96 KB each).

Output:
  knowledge/curricula/core-multimodal-v6audio/
    audio/                  (synthesized WAVs)
    audio_pairs.jsonl       (instruction/output pairs, key: 'audio')
"""
from __future__ import annotations

import json
import random
import wave
from pathlib import Path

import numpy as np

OUT_DIR = Path("/gaia/GAIA_Project/knowledge/curricula/core-multimodal-v6audio")
AUDIO_DIR = OUT_DIR / "audio"
RATE = 16000  # Gemma 4 audio_processor sampling_rate


def write_wav(path: Path, samples: np.ndarray, rate: int = RATE) -> None:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(samples, -32767, 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(clipped.tobytes())


def gen_silence(rng: random.Random, idx: int) -> tuple[np.ndarray, dict]:
    duration = rng.choice([1.0, 1.5, 2.0, 2.5, 3.0])
    samples = np.zeros(int(RATE * duration), dtype=np.float32)
    return samples, {"category": "silence", "duration": duration}


def gen_tone(rng: random.Random, idx: int) -> tuple[np.ndarray, dict]:
    band = rng.choice(["low", "mid", "high"])
    if band == "low":
        freq = rng.choice([110, 165, 220])
    elif band == "mid":
        freq = rng.choice([330, 440, 550, 660])
    else:
        freq = rng.choice([880, 1100, 1320, 1760])
    duration = rng.choice([1.0, 1.5, 2.0])
    n = int(RATE * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    amp = rng.choice([5000, 6000, 8000])
    samples = (np.sin(2 * np.pi * freq * t) * amp).astype(np.float32)
    return samples, {"category": "tone", "freq": freq, "band": band, "duration": duration}


def gen_sweep(rng: random.Random, idx: int) -> tuple[np.ndarray, dict]:
    direction = rng.choice(["up", "down", "up_down"])
    f_low = rng.choice([110, 165, 220])
    f_high = rng.choice([880, 1320, 1760, 2200])
    duration = rng.choice([1.5, 2.0, 2.5])
    n = int(RATE * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    if direction == "up":
        freqs = np.linspace(f_low, f_high, n)
    elif direction == "down":
        freqs = np.linspace(f_high, f_low, n)
    else:  # up_down
        half = n // 2
        freqs = np.concatenate([np.linspace(f_low, f_high, half),
                                np.linspace(f_high, f_low, n - half)])
    amp = rng.choice([5000, 7000])
    samples = (np.sin(2 * np.pi * freqs * t) * amp).astype(np.float32)
    return samples, {"category": "sweep", "direction": direction,
                     "f_low": f_low, "f_high": f_high, "duration": duration}


def gen_pulses(rng: random.Random, idx: int) -> tuple[np.ndarray, dict]:
    n_pulses = rng.choice([2, 3, 4, 5, 6])
    pulse_freq = rng.choice([440, 660, 880, 1100])
    pulse_dur = rng.choice([0.10, 0.15, 0.20, 0.25])
    gap = rng.choice([0.20, 0.30, 0.40, 0.50])
    total = n_pulses * (pulse_dur + gap)
    n = int(RATE * total)
    samples = np.zeros(n, dtype=np.float32)
    for i in range(n_pulses):
        start = int(RATE * (i * (pulse_dur + gap)))
        end = start + int(RATE * pulse_dur)
        if end > n:
            end = n
        ts = np.linspace(0, pulse_dur, end - start, endpoint=False)
        envelope = np.sin(np.pi * np.arange(end - start) / max(end - start, 1))  # smooth bell
        samples[start:end] = np.sin(2 * np.pi * pulse_freq * ts) * 6000 * envelope
    return samples, {"category": "pulses", "n_pulses": n_pulses,
                     "freq": pulse_freq, "pulse_dur": pulse_dur, "gap": gap}


def gen_noise(rng: random.Random, idx: int) -> tuple[np.ndarray, dict]:
    color = rng.choice(["white", "pink", "brown", "burst"])
    duration = rng.choice([1.0, 1.5, 2.0])
    n = int(RATE * duration)
    if color == "white":
        samples = (np.random.uniform(-3000, 3000, n)).astype(np.float32)
    elif color == "pink":
        # 1/f noise — approximate via cumulative average of white
        white = np.random.uniform(-1, 1, n)
        # Simple 1-pole low-pass for a pinkish tilt
        b = 0.95
        out = np.zeros(n)
        out[0] = white[0]
        for i in range(1, n):
            out[i] = b * out[i - 1] + (1 - b) * white[i]
        out = out / np.max(np.abs(out)) * 4000
        samples = out.astype(np.float32)
    elif color == "brown":
        # Random walk (integrated white) → 1/f^2
        white = np.random.uniform(-1, 1, n)
        out = np.cumsum(white)
        out = out / np.max(np.abs(out)) * 6000
        samples = out.astype(np.float32)
    else:  # burst — short loud noise + silence
        samples = np.zeros(n, dtype=np.float32)
        burst_dur = int(RATE * rng.choice([0.10, 0.15, 0.20]))
        start = rng.randint(0, max(0, n - burst_dur - 100))
        samples[start:start + burst_dur] = np.random.uniform(-4000, 4000, burst_dur)
    return samples, {"category": "noise", "color": color, "duration": duration}


def gen_chord(rng: random.Random, idx: int) -> tuple[np.ndarray, dict]:
    base = rng.choice([220, 330, 440])
    chord_type = rng.choice(["major", "minor", "fifth"])
    if chord_type == "major":
        ratios = [1.0, 1.25, 1.5]  # root, M3, P5
    elif chord_type == "minor":
        ratios = [1.0, 1.2, 1.5]   # root, m3, P5
    else:
        ratios = [1.0, 1.5]         # root, P5
    duration = rng.choice([1.5, 2.0, 2.5])
    n = int(RATE * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    samples = np.zeros(n, dtype=np.float32)
    for r in ratios:
        samples += np.sin(2 * np.pi * (base * r) * t) * (4500 / len(ratios))
    return samples, {"category": "chord", "base": base,
                     "chord_type": chord_type, "duration": duration}


# Prompt templates per category — varied phrasing so the model learns the
# concept, not the template. Outputs match the battery's accepted keywords
# ("silence/quiet", "tone/beep", "rising/ascending", "three/3", etc.).
PROMPTS = {
    "silence": [
        ("Listen to this audio and describe what you hear.",
         "{lead}silence{trail}"),
        ("What sound is in this audio?",
         "{lead}silence — there is no sound{trail}"),
        ("Describe the audio.",
         "{lead}silence, no audible sound{trail}"),
        ("What kind of sound is this?",
         "{lead}silent — the audio is quiet, no sound{trail}"),
    ],
    "tone": [
        ("What kind of sound is in this audio? Be brief.",
         "{lead}a {band} pure tone, around {freq} Hz{trail}"),
        ("Listen to this audio and describe what you hear.",
         "{lead}a single sustained {band} tone, like a {freq} Hz beep{trail}"),
        ("Describe the audio.",
         "{lead}a {band} sine tone{trail}"),
        ("What is the dominant pitch?",
         "{lead}a {band} {freq} Hz pure tone{trail}"),
    ],
    "sweep": [
        ("Describe the change in pitch over the duration of this audio.",
         "{lead}a {direction_word} pitch sweep{trail}"),
        ("What kind of sound is this?",
         "{lead}a frequency sweep, {direction_word} from {f_low}Hz to {f_high}Hz{trail}"),
        ("Listen and describe what you hear.",
         "{lead}a tone with {direction_word} pitch, sweeping {f_low}-{f_high}Hz{trail}"),
        ("Describe the audio.",
         "{lead}a {direction_word} frequency sweep{trail}"),
    ],
    "pulses": [
        ("How many distinct sounds do you hear in this audio?",
         "{lead}{n_pulses} distinct pulses{trail}"),
        ("What is the rhythm of this audio?",
         "{lead}{n_pulses} short pulses at a steady rhythm{trail}"),
        ("Listen and describe what you hear.",
         "{lead}{n_pulses} short repeated beeps{trail}"),
        ("Describe the audio.",
         "{lead}{n_pulses} discrete pulses, like a beeping pattern{trail}"),
    ],
    "noise": [
        ("Describe the texture of this audio.",
         "{lead}{color_word} noise — a rough, uniform hiss{trail}"),
        ("What kind of sound is this?",
         "{lead}{color_word} noise{trail}"),
        ("Listen and describe what you hear.",
         "{lead}static — {color_word} noise{trail}"),
        ("Describe the audio.",
         "{lead}{color_word} noise, like static or hiss{trail}"),
    ],
    "chord": [
        ("Listen to this audio. Is it a single tone or a chord?",
         "{lead}a {chord_type} chord — multiple tones at once{trail}"),
        ("What kind of sound is this?",
         "{lead}a {chord_type} chord{trail}"),
        ("Describe the audio.",
         "{lead}several pitches together — a {chord_type} chord{trail}"),
        ("How many tones do you hear?",
         "{lead}multiple — a {chord_type} chord{trail}"),
    ],
}


def render(template: str, meta: dict, rng: random.Random) -> str:
    direction_word = {"up": "rising", "down": "descending",
                      "up_down": "rising then falling"}.get(meta.get("direction"), "")
    color_word = meta.get("color", "")
    lead = rng.choice(["", "I hear ", "This is ", ""])
    trail = rng.choice([".", "."])
    return template.format(
        lead=lead,
        trail=trail,
        band=meta.get("band", ""),
        freq=meta.get("freq", ""),
        direction_word=direction_word,
        f_low=meta.get("f_low", ""),
        f_high=meta.get("f_high", ""),
        n_pulses=meta.get("n_pulses", ""),
        color_word=color_word,
        chord_type=meta.get("chord_type", ""),
    )


GEN = {
    "silence": (gen_silence, 30),
    "tone": (gen_tone, 60),
    "sweep": (gen_sweep, 40),
    "pulses": (gen_pulses, 40),
    "noise": (gen_noise, 40),
    "chord": (gen_chord, 30),
}


def main() -> int:
    print("=" * 60)
    print("  Multimodal Core v6_audio — synthetic audio curriculum")
    print("=" * 60)

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    np.random.seed(42)

    pairs: list[dict] = []
    for cat, (fn, n_samples) in GEN.items():
        for i in range(n_samples):
            samples, meta = fn(rng, i)
            fname = f"{cat}_{i:03d}.wav"
            write_wav(AUDIO_DIR / fname, samples)
            for prompt_template, output_template in PROMPTS[cat]:
                pairs.append({
                    "audio": f"audio/{fname}",
                    "instruction": prompt_template,
                    "output": render(output_template, meta, rng),
                    "category": cat,
                })
        print(f"  {cat:8s}: {n_samples:>3d} files × {len(PROMPTS[cat])} prompts = {n_samples * len(PROMPTS[cat])} pairs")

    rng.shuffle(pairs)
    out_jsonl = OUT_DIR / "audio_pairs.jsonl"
    with open(out_jsonl, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    n_files = len(list(AUDIO_DIR.glob("*.wav")))
    total_bytes = sum((AUDIO_DIR / f).stat().st_size for f in [p.name for p in AUDIO_DIR.glob("*.wav")])
    print(f"\n>> Wrote {len(pairs)} pairs over {n_files} unique audio files")
    print(f"   Audio dir size: {total_bytes / (1024 ** 2):.1f} MB")
    print(f"   Output: {out_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
