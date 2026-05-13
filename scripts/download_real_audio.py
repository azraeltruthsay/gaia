#!/usr/bin/env python3
"""Download real-audio training data: ESC-50 + LibriSpeech (GAIA_Project audio).

V6's audio is broken because the model only had 320 synthetic audio samples
(silence, tones, sweeps, noise, pulses, chords) for training. The model
learned to hallucinate audio scenes that don't match the input. This
script pulls REAL audio with REAL descriptions:

  ESC-50 (2000 clips, 50 categories)  — environmental sounds → scene
    descriptions. Each clip has a category like 'dog', 'rain', 'crying_baby'.
    Generates 2-3 prompt variations per clip.

  LibriSpeech clean train.100 subset (~3K clips)  — read English speech
    with verbatim transcripts. One prompt: 'Transcribe this audio.'

Output:
  knowledge/curricula/core_v2x_audio/
    audio/*.wav         (standardized 16kHz mono PCM)
    audio_pairs.jsonl   (one sample per row, with category field)

Each WAV is saved at 16kHz mono PCM16 (what Gemma 4's audio processor
expects). FLAC source files are decoded + resampled to WAV.
"""
import argparse
import io
import json
import os
import random
import shutil
import sys
import wave
from pathlib import Path

import numpy as np


OUT_DIR = Path("/gaia/GAIA_Project/knowledge/curricula/core_v2x_audio")
ESC50_TARGET = 2000
LIBRI_TARGET = 3000


# ESC-50 category → human description map.
# Categories like 'dog' become 'a dog barking' etc. — natural English.
ESC50_DESCRIPTIONS = {
    "dog": "a dog barking",
    "rooster": "a rooster crowing",
    "pig": "a pig grunting",
    "cow": "a cow mooing",
    "frog": "a frog croaking",
    "cat": "a cat meowing",
    "hen": "a hen clucking",
    "insects": "insects buzzing",
    "sheep": "a sheep bleating",
    "crow": "a crow cawing",
    "rain": "rain falling",
    "sea_waves": "sea waves crashing",
    "crackling_fire": "a crackling fire",
    "crickets": "crickets chirping",
    "chirping_birds": "birds chirping",
    "water_drops": "water dripping",
    "wind": "wind blowing",
    "pouring_water": "water pouring",
    "toilet_flush": "a toilet flushing",
    "thunderstorm": "a thunderstorm",
    "crying_baby": "a baby crying",
    "sneezing": "someone sneezing",
    "clapping": "clapping hands",
    "breathing": "breathing",
    "coughing": "coughing",
    "footsteps": "footsteps",
    "laughing": "laughter",
    "brushing_teeth": "someone brushing their teeth",
    "snoring": "snoring",
    "drinking_sipping": "drinking or sipping",
    "door_wood_knock": "a door knock",
    "mouse_click": "a mouse click",
    "keyboard_typing": "keyboard typing",
    "door_wood_creaks": "a creaking wooden door",
    "can_opening": "a can being opened",
    "washing_machine": "a washing machine running",
    "vacuum_cleaner": "a vacuum cleaner",
    "clock_alarm": "an alarm clock",
    "clock_tick": "a ticking clock",
    "glass_breaking": "glass breaking",
    "helicopter": "a helicopter",
    "chainsaw": "a chainsaw",
    "siren": "a siren wailing",
    "car_horn": "a car horn",
    "engine": "an engine running",
    "train": "a train",
    "church_bells": "church bells ringing",
    "airplane": "an airplane",
    "fireworks": "fireworks",
    "hand_saw": "a hand saw cutting",
}

# Prompt variations for ESC-50 (rotated across samples for variety)
ESC50_PROMPTS = [
    ("Listen to this audio and describe what you hear.", "I hear {desc}."),
    ("What do you hear in this audio?", "{desc}."),
    ("Describe the sound in this clip.", "{desc}."),
    ("What kind of sound is this?", "{desc}."),
    ("Identify the sound in this audio.", "{desc}."),
]


def save_wav_16k_mono(out_path: Path, audio_bytes: bytes, src_format: str = "wav") -> bool:
    """Decode audio bytes (WAV or FLAC), resample to 16kHz mono, save as WAV PCM16.
    Returns True on success, False on decode failure.
    """
    try:
        if src_format == "flac":
            import soundfile as sf
            data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)  # downmix to mono
        else:  # wav
            with wave.open(io.BytesIO(audio_bytes), "rb") as w:
                n_ch = w.getnchannels()
                sw = w.getsampwidth()
                sr = w.getframerate()
                nf = w.getnframes()
                b = w.readframes(nf)
            if sw == 2:
                data = np.frombuffer(b, dtype="<i2").astype(np.float32) / 32768.0
            elif sw == 4:
                data = np.frombuffer(b, dtype="<i4").astype(np.float32) / 2147483648.0
            else:
                data = (np.frombuffer(b, dtype="u1").astype(np.float32) - 128.0) / 128.0
            if n_ch > 1:
                data = data.reshape(-1, n_ch).mean(axis=1)
        # Resample to 16kHz
        if sr != 16000:
            ratio = 16000 / sr
            new_len = int(round(len(data) * ratio))
            xp = np.linspace(0, 1, len(data), endpoint=False)
            xn = np.linspace(0, 1, new_len, endpoint=False)
            data = np.interp(xn, xp, data)
        # Write 16-bit PCM mono WAV
        pcm = np.clip(data * 32767.0, -32768, 32767).astype("<i2")
        with wave.open(str(out_path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(pcm.tobytes())
        return True
    except Exception as e:
        print(f"  save_wav failed for {out_path.name}: {e}")
        return False


def pull_esc50(rng: random.Random, out_audio_dir: Path) -> list[dict]:
    print("[esc50] loading ashraq/esc50...")
    from datasets import load_dataset, Audio
    ds = load_dataset("ashraq/esc50", split="train").cast_column("audio", Audio(decode=False))
    print(f"  total clips: {len(ds)}")
    rows: list[dict] = []
    target = min(ESC50_TARGET, len(ds))
    for i, entry in enumerate(ds):
        if len(rows) >= target:
            break
        cat = entry.get("category", "")
        desc = ESC50_DESCRIPTIONS.get(cat, f"a {cat}")
        fname = entry.get("filename", f"esc50_{i}.wav")
        out_name = f"esc50_{fname}"
        audio_bytes = entry["audio"]["bytes"]
        if not save_wav_16k_mono(out_audio_dir / out_name, audio_bytes, src_format="wav"):
            continue
        # Pick one prompt variation per clip (rotate through list for variety)
        instr, out_tpl = ESC50_PROMPTS[i % len(ESC50_PROMPTS)]
        rows.append({
            "audio": f"audio/{out_name}",
            "instruction": instr,
            "output": out_tpl.format(desc=desc),
            "category": "audio_esc50",
        })
        if len(rows) % 200 == 0:
            print(f"  esc50: {len(rows)}/{target}")
    print(f"  esc50 kept: {len(rows)}")
    return rows


def pull_librispeech(rng: random.Random, out_audio_dir: Path) -> list[dict]:
    print("[librispeech] loading openslr/librispeech_asr (clean, train.100)...")
    from datasets import load_dataset, Audio
    ds = load_dataset("openslr/librispeech_asr", "clean", split="train.100").cast_column(
        "audio", Audio(decode=False))
    print(f"  total clips: {len(ds)}")
    # Sample random indices
    target = min(LIBRI_TARGET, len(ds))
    indices = rng.sample(range(len(ds)), target * 2)  # pull 2x in case of filter losses
    rows: list[dict] = []
    for i in indices:
        if len(rows) >= target:
            break
        entry = ds[i]
        text = (entry.get("text") or "").strip()
        if not text or len(text) < 5 or len(text) > 400:
            continue
        clip_id = entry.get("id", f"libri_{i}")
        out_name = f"libri_{clip_id}.wav"
        audio_bytes = entry["audio"]["bytes"]
        if not save_wav_16k_mono(out_audio_dir / out_name, audio_bytes, src_format="flac"):
            continue
        # Normalize text: LibriSpeech is all uppercase. Make it natural-case.
        clean_text = text.capitalize()
        # Sentence punctuation: add period if missing
        if not clean_text.endswith((".", "?", "!")):
            clean_text += "."
        rows.append({
            "audio": f"audio/{out_name}",
            "instruction": "Transcribe this audio.",
            "output": clean_text,
            "category": "audio_speech",
        })
        if len(rows) % 200 == 0:
            print(f"  libri: {len(rows)}/{target}")
    print(f"  libri kept: {len(rows)}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    audio_dir = OUT_DIR / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    all_rows.extend(pull_esc50(rng, audio_dir))
    all_rows.extend(pull_librispeech(rng, audio_dir))

    out_path = OUT_DIR / "audio_pairs.jsonl"
    with open(out_path, "w") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nWrote {len(all_rows)} samples → {out_path}")
    print(f"Audio dir: {audio_dir}")

    # Summary
    from collections import Counter
    cats = Counter(r["category"] for r in all_rows)
    print(f"Categories: {dict(cats)}")
    total_size = sum(p.stat().st_size for p in audio_dir.glob("*.wav"))
    print(f"Audio storage: {total_size / 1024**2:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
