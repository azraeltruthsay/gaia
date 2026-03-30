#!/usr/bin/env python3
"""Voice Reflex Loop — End-to-End Test Harness.

Exercises the full voice pipeline: TTS → STT → Nano Reflex → TTS,
measuring latency at each stage and saving audio files for review.

Usage:
    python scripts/voice_reflex_test.py                            # defaults
    python scripts/voice_reflex_test.py --text "What time is it?"  # custom prompt
    python scripts/voice_reflex_test.py --voice "Viktor Eka"       # voice override
    python scripts/voice_reflex_test.py --output-dir /tmp/vtest/   # custom output
"""

import argparse
import base64
import json
import os
import sys
import time
import uuid
import wave

import requests

CORE_URL = os.environ.get("GAIA_CORE_URL", "http://localhost:6415")
AUDIO_URL = os.environ.get("GAIA_AUDIO_URL", "http://localhost:8080")

DEFAULT_VOICES = ["Craig Gutsy", "Viktor Eka", "Claribel Dervla"]


def save_wav(pcm_bytes: bytes, path: str, sample_rate: int = 22050) -> None:
    """Save raw 16-bit mono PCM as a WAV file."""
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)


def synthesize(text: str, voice: str | None = None) -> tuple[bytes, int]:
    """POST to gaia-audio /synthesize. Returns (pcm_bytes, sample_rate)."""
    payload: dict = {"text": text}
    if voice:
        payload["voice"] = voice
    resp = requests.post(f"{AUDIO_URL}/synthesize", json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    audio_b64 = data.get("audio_base64")
    if not audio_b64:
        raise RuntimeError("No audio_base64 in synthesis response")
    return base64.b64decode(audio_b64), data.get("sample_rate", 22050)


def transcribe(audio_bytes: bytes, sample_rate: int) -> str:
    """POST to gaia-audio /transcribe. Returns transcribed text."""
    # Wrap raw PCM in WAV for the transcribe endpoint
    import io
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_bytes)
    audio_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    resp = requests.post(
        f"{AUDIO_URL}/transcribe",
        json={"audio_base64": audio_b64, "sample_rate": sample_rate},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get("text", "").strip()


def nano_reflex(text: str) -> tuple[str, float]:
    """POST to gaia-core /process_packet with a fresh session for reflex eligibility.

    Parses NDJSON stream, extracts reflex text, returns (clean_text, latency_ms).
    """
    session_id = f"voice_reflex_test_{uuid.uuid4().hex[:12]}"
    packet = {
        "header": {
            "session_id": session_id,
            "packet_id": str(uuid.uuid4()),
            "version": "0.3",
            "output_routing": {
                "primary": {
                    "destination": "audio",
                },
                "source_destination": "audio",
                "addressed_to_gaia": True,
                "suppress_echo": False,
            },
        },
        "content": {
            "original_prompt": text,
        },
        "metadata": {
            "target_engine": "nano",
            "priority": 5,
            "max_tokens": 256,
            "time_budget_ms": 5000,
            "tone_hint": "conversational",
            "dry_run": True,
        },
    }

    t0 = time.monotonic()
    resp = requests.post(
        f"{CORE_URL}/process_packet",
        json=packet,
        headers={"Content-Type": "application/json"},
        stream=True,
        timeout=15,
    )
    resp.raise_for_status()

    response_text = ""
    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "token":
            response_text += event.get("value", "")
        elif event.get("type") == "packet":
            # Final packet — extract candidate if we haven't got tokens
            if not response_text:
                pkt = event.get("value", {})
                candidate = pkt.get("response", {}).get("candidate", "")
                if candidate:
                    response_text = candidate

    latency_ms = (time.monotonic() - t0) * 1000

    # Strip reflex formatting header: ⚡ **[(Reflex) Reflex]**\n
    clean = response_text.strip()
    for prefix in ["⚡ **[(Reflex) Reflex]**", "⚡ **[(Reflex) Reflex]**\n"]:
        if clean.startswith(prefix):
            clean = clean[len(prefix):]
    # Strip trailing separator
    clean = clean.rstrip("-").rstrip("\n").strip()

    return clean, latency_ms


def try_voices(preferred: str | None) -> str | None:
    """Try the preferred voice, then fallbacks. Return first that works."""
    voices = [preferred] if preferred else []
    voices.extend(v for v in DEFAULT_VOICES if v not in voices)

    for voice in voices:
        try:
            # Quick test synthesis
            requests.post(
                f"{AUDIO_URL}/synthesize",
                json={"text": "test", "voice": voice},
                timeout=30,
            ).raise_for_status()
            return voice
        except Exception:
            continue
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Voice Reflex Loop — E2E Test")
    parser.add_argument("--text", default="Hello GAIA", help="Input text to speak")
    parser.add_argument("--voice", default=None, help="TTS voice name override")
    parser.add_argument("--output-dir", default=".", help="Directory for output WAV files")
    parser.add_argument("--skip-input-synth", action="store_true",
                        help="Skip generating input audio (use text directly for STT test)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    input_wav = os.path.join(args.output_dir, "voice_test_input.wav")
    output_wav = os.path.join(args.output_dir, "voice_test_output.wav")

    print(f"╔══════════════════════════════════════════════╗")
    print(f"║   Voice Reflex Loop — End-to-End Test        ║")
    print(f"╚══════════════════════════════════════════════╝")
    print(f"  Input text : {args.text}")
    print(f"  Core URL   : {CORE_URL}")
    print(f"  Audio URL  : {AUDIO_URL}")
    print()

    timings: dict[str, float] = {}

    # --- Stage 1: Generate input audio via TTS ---
    if not args.skip_input_synth:
        print("[1/4] Generating input audio via TTS...")
        voice = args.voice
        if not voice:
            # Try default voices
            voice = try_voices(None)
            if voice:
                print(f"       Using voice: {voice}")
            else:
                print("       WARNING: No voice available, using default")

        t0 = time.monotonic()
        try:
            pcm, sr = synthesize(args.text, voice=voice)
            timings["input_tts"] = (time.monotonic() - t0) * 1000
            save_wav(pcm, input_wav, sr)
            print(f"       Saved: {input_wav} ({len(pcm)} bytes, {sr}Hz)")
            print(f"       Latency: {timings['input_tts']:.0f}ms")
        except Exception as e:
            print(f"       FAILED: {e}")
            print("       Falling back to direct text for STT test")
            args.skip_input_synth = True
    else:
        print("[1/4] Skipping input audio generation")

    # --- Stage 2: Transcribe ---
    if not args.skip_input_synth:
        print("\n[2/4] Transcribing input audio via Whisper...")
        t0 = time.monotonic()
        try:
            transcribed = transcribe(pcm, sr)
            timings["stt"] = (time.monotonic() - t0) * 1000
            print(f"       Transcribed: \"{transcribed}\"")
            print(f"       Latency: {timings['stt']:.0f}ms")
        except Exception as e:
            print(f"       FAILED: {e}")
            transcribed = args.text
            print(f"       Falling back to original text: \"{transcribed}\"")
    else:
        print("\n[2/4] Using input text directly (no STT)")
        transcribed = args.text
        timings["stt"] = 0

    # --- Stage 3: Nano Reflex ---
    print(f"\n[3/4] Sending to Nano Reflex: \"{transcribed}\"")
    try:
        reflex_text, reflex_ms = nano_reflex(transcribed)
        timings["reflex"] = reflex_ms
        print(f"       Response: \"{reflex_text[:200]}\"")
        print(f"       Latency: {reflex_ms:.0f}ms")
    except Exception as e:
        print(f"       FAILED: {e}")
        print("       Cannot continue without a response.")
        sys.exit(1)

    if not reflex_text:
        print("       WARNING: Empty reflex response (reflex may not have fired)")
        print("       The full Prime pipeline may have run instead.")

    # --- Stage 4: Synthesize response ---
    speak_text = reflex_text or "I'm here."
    print(f"\n[4/4] Synthesizing response: \"{speak_text[:100]}\"")
    t0 = time.monotonic()
    try:
        out_pcm, out_sr = synthesize(speak_text, voice=args.voice)
        timings["output_tts"] = (time.monotonic() - t0) * 1000
        save_wav(out_pcm, output_wav, out_sr)
        print(f"       Saved: {output_wav} ({len(out_pcm)} bytes, {out_sr}Hz)")
        print(f"       Latency: {timings['output_tts']:.0f}ms")
    except Exception as e:
        print(f"       FAILED: {e}")
        timings["output_tts"] = 0

    # --- Timing Report ---
    print()
    print("=" * 50)
    print("  TIMING REPORT")
    print("=" * 50)
    if "input_tts" in timings:
        print(f"  Input TTS      : {timings['input_tts']:>8.0f} ms  (not in pipeline)")
    print(f"  STT (Whisper)  : {timings.get('stt', 0):>8.0f} ms")
    print(f"  Nano Reflex    : {timings.get('reflex', 0):>8.0f} ms")
    print(f"  Output TTS     : {timings.get('output_tts', 0):>8.0f} ms")
    pipeline_ms = timings.get("stt", 0) + timings.get("reflex", 0) + timings.get("output_tts", 0)
    print(f"  ─────────────────────────────────")
    print(f"  Pipeline Total : {pipeline_ms:>8.0f} ms")
    print("=" * 50)

    # Verdict
    if pipeline_ms < 2000:
        print("  ✓ Sub-2s voice loop achieved!")
    elif pipeline_ms < 3000:
        print("  ~ Acceptable latency (< 3s)")
    else:
        print(f"  ✗ Pipeline too slow ({pipeline_ms:.0f}ms > 3000ms)")


if __name__ == "__main__":
    main()
