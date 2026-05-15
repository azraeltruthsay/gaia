#!/usr/bin/env python3
"""V8 prep: diversify LibriSpeech transcription prompts.

V7 issue: all 3000 LibriSpeech samples used identical prompt 'Transcribe this
audio.' Model conflated all audio with the strongest output pattern (which
became 'ticking clock' from ESC-50) instead of differentiating between
transcription and scene description.

This script rewrites the audio_speech entries in core_v2x_audio/audio_pairs.jsonl
with VARIED prompts, all unambiguously requesting verbatim speech transcription.
Doesn't change audio files or outputs — just rotates instructions so the model
learns the broad mapping 'transcription request → text content of speech'.
"""
import json
import random
import sys
from pathlib import Path


AUDIO_PAIRS = Path("/gaia/GAIA_Project/knowledge/curricula/core_v2x_audio/audio_pairs.jsonl")

# Prompt variations — all unambiguously transcription-specific.
# These should NOT overlap with scene-description prompts so the model can
# distinguish "transcribe speech" from "describe sound".
TRANSCRIPTION_PROMPTS = [
    "Transcribe the speech in this audio.",
    "Write down exactly what is said in this recording.",
    "What words are spoken in this audio?",
    "Provide a verbatim transcription of this audio.",
    "Convert this speech to text.",
    "What is being said in this audio? Transcribe it.",
    "Please write out the words spoken in this clip.",
    "Listen and transcribe the speech word-for-word.",
    "Give me a written transcript of this audio.",
    "Type out what the speaker says.",
    "Transcribe this verbatim, including all words.",
    "What is the speaker saying? Give me the exact words.",
]


def main() -> int:
    if not AUDIO_PAIRS.exists():
        print(f"ERROR: {AUDIO_PAIRS} not found")
        return 1

    rng = random.Random(42)
    rows = []
    n_speech = 0
    n_total = 0
    with open(AUDIO_PAIRS) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            n_total += 1
            if d.get("category") == "audio_speech":
                # Rotate to a varied prompt
                d["instruction"] = rng.choice(TRANSCRIPTION_PROMPTS)
                n_speech += 1
            rows.append(d)

    with open(AUDIO_PAIRS, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Updated {n_speech}/{n_total} audio_speech entries with varied prompts.")
    print(f"Prompt variations used: {len(TRANSCRIPTION_PROMPTS)}")
    # Show distribution
    from collections import Counter
    prompt_counts: Counter = Counter()
    for r in rows:
        if r.get("category") == "audio_speech":
            prompt_counts[r["instruction"]] += 1
    print("\nPrompt distribution:")
    for p, c in sorted(prompt_counts.items(), key=lambda x: -x[1]):
        print(f"  {c:4d}: {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
