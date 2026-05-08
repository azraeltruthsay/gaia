#!/usr/bin/env python3
"""Build the v7 audio curriculum — synthetic + ESC-50 + LibriSpeech mix.

V6AUDIO trained on 960 pure synthetic primitives × ~3.5 epochs scored
1/5 → 1/20 on the audio battery. The LM learned to produce audio-style
responses but not to robustly map audio embeddings to specific
descriptions. V7 broadens the training distribution:

  - Synthetic primitives (expanded from V6): ~1400 pairs over 6
    categories. Same generators as v6audio but more variants per
    category. Aligns with the battery's primitive shapes.
  - ESC-50 environmental sounds: 500 pairs. 50 categories × 10 clips
    per category, asking "What sound is in this audio?" with the
    category as the answer. Real audio with real labels.
  - LibriSpeech dev-clean speech: 500 pairs. Mix of transcription,
    paraphrase, and "respond conversationally" prompts. Real human
    speech with real transcripts.

Total ~2400 pairs over ~2400 unique audio files. Symlinks ESC-50 +
LibriSpeech sources into the curriculum dir so paths resolve
identically to the synthetic ones.

Output:
  knowledge/curricula/core-multimodal-v7audio/
    audio/                  (synthesized + symlinked WAVs/FLACs)
    audio_pairs.jsonl       (instruction/output pairs, key: 'audio')
"""
from __future__ import annotations

import csv
import json
import random
import wave
from pathlib import Path

import numpy as np

PROJ = Path("/gaia/GAIA_Project")
OUT_DIR = PROJ / "knowledge/curricula/core-multimodal-v7audio"
AUDIO_DIR = OUT_DIR / "audio"

ESC50_ROOT = PROJ / "data/datasets/ESC-50-master"
LIBRISPEECH_ROOT = PROJ / "data/datasets/LibriSpeech/dev-clean"

RATE = 16000


# ── Synthetic primitives (expanded from V6) ─────────────────────────────────

def write_wav(path: Path, samples: np.ndarray, rate: int = RATE) -> None:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(samples, -32767, 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(clipped.tobytes())


def gen_silence(rng: random.Random, idx: int):
    duration = rng.choice([0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0])
    return np.zeros(int(RATE * duration), dtype=np.float32), {
        "category": "silence", "duration": duration}


def gen_tone(rng: random.Random, idx: int):
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


def gen_sweep(rng: random.Random, idx: int):
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


def gen_pulses(rng: random.Random, idx: int):
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


def gen_noise(rng: random.Random, idx: int):
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


def gen_chord(rng: random.Random, idx: int):
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


PROMPTS = {
    "silence": [
        ("Listen to this audio and describe what you hear.", "silence."),
        ("What sound is in this audio?", "silence — there is no sound."),
        ("Describe the audio.", "Silence, no audible sound."),
        ("What kind of sound is this?", "Silent — the audio is quiet, no sound."),
    ],
    "tone": [
        ("What kind of sound is in this audio? Be brief.",
         "A {band} pure tone, around {freq} Hz."),
        ("Listen to this audio and describe what you hear.",
         "A single sustained {band} tone, like a {freq} Hz beep."),
        ("Describe the audio.", "A {band} sine tone."),
        ("What is the dominant pitch?", "A {band} {freq} Hz pure tone."),
    ],
    "sweep": [
        ("Describe the change in pitch over the duration of this audio.",
         "A {direction_word} pitch sweep."),
        ("What kind of sound is this?",
         "A frequency sweep, {direction_word} from {f_low}Hz to {f_high}Hz."),
        ("Listen and describe what you hear.",
         "A tone with {direction_word} pitch, sweeping {f_low}-{f_high}Hz."),
        ("Describe the audio.", "A {direction_word} frequency sweep."),
    ],
    "pulses": [
        ("How many distinct sounds do you hear in this audio?",
         "{n_pulses} distinct pulses."),
        ("What is the rhythm of this audio?",
         "{n_pulses} short pulses at a steady rhythm."),
        ("Listen and describe what you hear.",
         "{n_pulses} short repeated beeps."),
        ("Describe the audio.",
         "{n_pulses} discrete pulses, like a beeping pattern."),
    ],
    "noise": [
        ("Describe the texture of this audio.",
         "{color_word} noise — a rough, uniform hiss."),
        ("What kind of sound is this?", "{color_word} noise."),
        ("Listen and describe what you hear.", "Static — {color_word} noise."),
        ("Describe the audio.", "{color_word} noise, like static or hiss."),
    ],
    "chord": [
        ("Listen to this audio. Is it a single tone or a chord?",
         "A {chord_type} chord — multiple tones at once."),
        ("What kind of sound is this?", "A {chord_type} chord."),
        ("Describe the audio.",
         "Several pitches together — a {chord_type} chord."),
        ("How many tones do you hear?", "Multiple — a {chord_type} chord."),
    ],
}


def render_synth(template: str, meta: dict) -> str:
    direction_word = {"up": "rising", "down": "descending",
                      "up_down": "rising then falling",
                      "down_up": "falling then rising"}.get(meta.get("direction"), "")
    return template.format(
        band=meta.get("band", ""),
        freq=meta.get("freq", ""),
        direction_word=direction_word,
        f_low=meta.get("f_low", ""),
        f_high=meta.get("f_high", ""),
        n_pulses=meta.get("n_pulses", ""),
        color_word=meta.get("color", ""),
        chord_type=meta.get("chord_type", ""),
    )


GEN = {
    "silence": (gen_silence, 50),
    "tone": (gen_tone, 80),
    "sweep": (gen_sweep, 60),
    "pulses": (gen_pulses, 60),
    "noise": (gen_noise, 60),
    "chord": (gen_chord, 40),
}


def build_synthetic(rng: random.Random, pairs: list) -> int:
    n = 0
    for cat, (fn, n_samples) in GEN.items():
        for i in range(n_samples):
            samples, meta = fn(rng, i)
            fname = f"synth_{cat}_{i:03d}.wav"
            write_wav(AUDIO_DIR / fname, samples)
            for prompt_template, output_template in PROMPTS[cat]:
                pairs.append({
                    "audio": f"audio/{fname}",
                    "instruction": prompt_template,
                    "output": render_synth(output_template, meta),
                    "category": f"synth_{cat}",
                })
                n += 1
    return n


# ── ESC-50 environmental sounds ─────────────────────────────────────────────

# Map ESC-50 category labels to natural-language descriptions for outputs
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

ESC50_PROMPTS = [
    "What sound is in this audio?",
    "Listen to this audio and describe what you hear.",
    "Describe the audio.",
    "What is happening in this audio?",
]


def build_esc50(rng: random.Random, pairs: list, per_cat: int = 10) -> int:
    """Symlink ESC-50 clips into our curriculum dir + emit pairs."""
    csv_path = ESC50_ROOT / "meta/esc50.csv"
    audio_root = ESC50_ROOT / "audio"
    if not csv_path.exists() or not audio_root.exists():
        print(f"  ESC-50: dataset missing at {ESC50_ROOT} — skipping")
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
            prompt = rng.choice(ESC50_PROMPTS)
            pairs.append({
                "audio": f"audio/{dst_name}",
                "instruction": prompt,
                "output": description + ".",
                "category": "esc50",
            })
            n += 1
    return n


# ── LibriSpeech speech ──────────────────────────────────────────────────────

LIBRISPEECH_PROMPTS = [
    ("Transcribe this audio.", "transcript"),
    ("What does the speaker say in this audio?", "transcript_quoted"),
    ("Listen to this audio. What was said?", "transcript_quoted"),
    ("Provide a transcript of the speech.", "transcript"),
]


def build_librispeech(rng: random.Random, pairs: list, n_clips: int = 500,
                      max_dur_sec: float = 8.0) -> int:
    """Walk LibriSpeech, pick short clips, emit (audio, transcribe) pairs.

    Symlinks the FLAC files. The trainer's WAV reader doesn't handle
    FLAC, so we transcode to WAV here using soundfile (available in
    the gaia-study container).
    """
    if not LIBRISPEECH_ROOT.exists():
        print(f"  LibriSpeech: missing at {LIBRISPEECH_ROOT} — skipping")
        return 0

    try:
        import soundfile as sf
    except ImportError:
        print("  LibriSpeech: soundfile not installed — skipping")
        return 0

    # Walk: spk_id/chapter_id/spk-chapter.trans.txt + spk-chapter-utt.flac
    candidates: list[tuple[Path, str]] = []
    for trans_file in LIBRISPEECH_ROOT.rglob("*.trans.txt"):
        parent = trans_file.parent
        with open(trans_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                utt_id, _, transcript = line.partition(" ")
                flac_path = parent / f"{utt_id}.flac"
                if flac_path.exists():
                    candidates.append((flac_path, transcript))

    rng.shuffle(candidates)
    selected = 0
    n = 0
    for flac_path, transcript in candidates:
        if selected >= n_clips:
            break
        try:
            data, sr = sf.read(str(flac_path))
            duration = len(data) / sr
            if duration > max_dur_sec or duration < 0.5:
                continue
            # Resample to 16 kHz if needed (LibriSpeech is 16 kHz already)
            if sr != RATE:
                ratio = RATE / sr
                new_len = int(round(len(data) * ratio))
                xp = np.linspace(0, 1, len(data), endpoint=False)
                x_new = np.linspace(0, 1, new_len, endpoint=False)
                data = np.interp(x_new, xp, data)
            # Write as WAV (PCM16) so the train script's wave reader handles it
            samples = (data * 32767).astype(np.float32)
            dst_name = f"libri_{flac_path.stem}.wav"
            write_wav(AUDIO_DIR / dst_name, samples)

            prompt_template, output_kind = rng.choice(LIBRISPEECH_PROMPTS)
            transcript = transcript.strip().capitalize()
            if not transcript.endswith((".", "!", "?")):
                transcript += "."
            if output_kind == "transcript":
                output = transcript
            elif output_kind == "transcript_quoted":
                output = f'The speaker said: "{transcript}"'
            else:
                output = transcript
            pairs.append({
                "audio": f"audio/{dst_name}",
                "instruction": prompt_template,
                "output": output,
                "category": "librispeech",
            })
            selected += 1
            n += 1
        except Exception as e:
            print(f"  LibriSpeech skip {flac_path.name}: {e}")
            continue
    return n


def main() -> int:
    print("=" * 60)
    print("  Multimodal Core v7audio — synthetic + ESC-50 + LibriSpeech")
    print("=" * 60)

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(2026)
    np.random.seed(2026)

    pairs: list[dict] = []

    print("\n>> Synthetic primitives ...")
    n_synth = build_synthetic(rng, pairs)
    print(f"   synth: {n_synth} pairs")

    print(">> ESC-50 environmental sounds ...")
    n_esc = build_esc50(rng, pairs, per_cat=10)
    print(f"   esc50: {n_esc} pairs")

    print(">> LibriSpeech speech ...")
    n_libri = build_librispeech(rng, pairs, n_clips=500)
    print(f"   librispeech: {n_libri} pairs")

    rng.shuffle(pairs)
    out_jsonl = OUT_DIR / "audio_pairs.jsonl"
    with open(out_jsonl, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    n_files = sum(1 for _ in AUDIO_DIR.iterdir())
    print(f"\n>> Wrote {len(pairs)} pairs over {n_files} unique audio files")
    print(f"   Mix: {n_synth} synth | {n_esc} ESC-50 | {n_libri} LibriSpeech")
    print(f"   Output: {out_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
