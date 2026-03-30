#!/usr/bin/env python3
"""Audio Test Harness — end-to-end sensory loop validation.

Generates test audio, pushes through STT → Core → TTS, validates round-trip.

The harness generates synthetic speech WAV files using gaia-audio's TTS,
then feeds them back through STT to verify the transcription matches.
Then sends the transcription to Core for a response, and synthesizes
that response back to audio. Full sensory loop.

Usage:
    python scripts/audio_test_harness.py                    # Full test
    python scripts/audio_test_harness.py --tts-only         # Just TTS test
    python scripts/audio_test_harness.py --stt-only         # Just STT test
    python scripts/audio_test_harness.py --round-trip       # Full loop
"""

import argparse
import base64
import json
import logging
import os
import struct
import sys
import time
import math
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GAIA.AudioTest")

AUDIO_ENDPOINT = os.environ.get("AUDIO_ENDPOINT", "http://localhost:8080")
CORE_ENDPOINT = os.environ.get("CORE_ENDPOINT", "http://localhost:6415")
WEB_ENDPOINT = os.environ.get("WEB_ENDPOINT", "http://localhost:6414")
OUTPUT_DIR = Path(os.environ.get("AUDIO_TEST_OUTPUT", "/tmp/gaia_audio_tests"))

# ── Test utterances ──────────────────────────────────────────────────────

TEST_UTTERANCES = [
    {
        "text": "Hello GAIA",
        "expect_contains": ["hello", "hi", "gaia"],
        "category": "greeting",
    },
    {
        "text": "What time is it?",
        "expect_contains": ["time", "clock", "AM", "PM"],
        "category": "time_check",
    },
    {
        "text": "Who are you?",
        "expect_contains": ["gaia", "sovereign", "ai", "azrael"],
        "category": "identity",
    },
]


# ── WAV Generation (synthetic sine wave as fallback) ─────────────────────

def generate_sine_wav(frequency: float = 440.0, duration: float = 1.0,
                      sample_rate: int = 16000, amplitude: float = 0.5) -> bytes:
    """Generate a simple sine wave WAV file. Used as a baseline signal test."""
    num_samples = int(sample_rate * duration)
    samples = []
    for i in range(num_samples):
        t = i / sample_rate
        value = amplitude * math.sin(2 * math.pi * frequency * t)
        samples.append(int(value * 32767))

    # WAV header
    data_size = num_samples * 2  # 16-bit samples
    header = struct.pack('<4sI4s4sIHHIIHH4sI',
        b'RIFF', 36 + data_size, b'WAVE',
        b'fmt ', 16, 1, 1, sample_rate, sample_rate * 2, 2, 16,
        b'data', data_size)
    audio_data = struct.pack(f'<{num_samples}h', *samples)
    return header + audio_data


# ── API Helpers ──────────────────────────────────────────────────────────

def call_tts(text: str, tier: str = "auto") -> dict:
    """Call gaia-audio TTS. Returns dict with audio_base64, latency_ms, etc."""
    payload = json.dumps({"text": text, "tier": tier}).encode()
    req = Request(
        f"{AUDIO_ENDPOINT}/synthesize",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def call_stt(audio_base64: str, sample_rate: int = 16000) -> dict:
    """Call gaia-audio STT. Returns dict with text, confidence, etc."""
    payload = json.dumps({
        "audio_base64": audio_base64,
        "sample_rate": sample_rate,
    }).encode()
    req = Request(
        f"{AUDIO_ENDPOINT}/transcribe",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def call_core(text: str) -> str:
    """Send text to gaia-core via web gateway, return response text."""
    try:
        from urllib.parse import quote
        req = Request(
            f"{WEB_ENDPOINT}/process_user_input?user_input={quote(text)}",
            method="POST",
        )
        with urlopen(req, timeout=60) as resp:
            raw = resp.read().decode()

        # Parse streaming response
        tokens = []
        for line in raw.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                if d.get("type") == "token":
                    tokens.append(d.get("value", ""))
            except json.JSONDecodeError:
                tokens.append(line)
        return "".join(tokens).strip()
    except Exception as e:
        logger.warning("Core call failed: %s", e)
        return f"[error: {e}]"


# ── Test Functions ───────────────────────────────────────────────────────

def test_tts(utterances=None) -> list:
    """Test TTS: generate audio from text."""
    if utterances is None:
        utterances = TEST_UTTERANCES

    results = []
    for utt in utterances:
        text = utt["text"]
        logger.info("TTS: '%s'...", text)
        try:
            result = call_tts(text)
            audio_b64 = result.get("audio_base64", "")
            audio_bytes = base64.b64decode(audio_b64) if audio_b64 else b""

            entry = {
                "text": text,
                "category": utt["category"],
                "status": "pass" if len(audio_bytes) > 100 else "fail",
                "audio_size_bytes": len(audio_bytes),
                "latency_ms": result.get("latency_ms", 0),
                "tier_used": result.get("tier_used", "unknown"),
                "engine_used": result.get("engine_used", "unknown"),
                "sample_rate": result.get("sample_rate", 0),
                "duration_seconds": result.get("duration_seconds", 0),
            }

            # Save WAV for inspection
            if audio_bytes:
                wav_path = OUTPUT_DIR / f"tts_{utt['category']}.wav"
                wav_path.write_bytes(audio_bytes)
                entry["wav_path"] = str(wav_path)
                logger.info("  -> %s, %d bytes, %.1fms (%s)",
                           entry["status"], len(audio_bytes),
                           entry["latency_ms"], entry["tier_used"])

            results.append(entry)

        except Exception as e:
            logger.warning("  -> FAIL: %s", e)
            results.append({
                "text": text, "category": utt["category"],
                "status": "error", "error": str(e),
            })

    return results


def test_stt(utterances=None) -> list:
    """Test STT: transcribe audio generated by TTS (or synthetic)."""
    if utterances is None:
        utterances = TEST_UTTERANCES

    results = []
    for utt in utterances:
        text = utt["text"]
        logger.info("STT roundtrip: '%s'...", text)

        # Step 1: Generate audio via TTS
        try:
            tts_result = call_tts(text)
            audio_b64 = tts_result.get("audio_base64", "")
            sample_rate = tts_result.get("sample_rate", 16000)
        except Exception as e:
            logger.warning("  TTS failed, using sine wave fallback: %s", e)
            sine_wav = generate_sine_wav(duration=1.0)
            audio_b64 = base64.b64encode(sine_wav).decode("ascii")
            sample_rate = 16000

        # Step 2: Transcribe
        try:
            stt_result = call_stt(audio_b64, sample_rate)
            transcription = stt_result.get("text", "").lower()

            # Check if any expected words appear
            expect = utt.get("expect_contains", [])
            matches = [w for w in expect if w.lower() in transcription]

            entry = {
                "original": text,
                "transcription": stt_result.get("text", ""),
                "category": utt["category"],
                "status": "pass" if matches else "partial" if transcription else "fail",
                "expected_words": expect,
                "matched_words": matches,
                "confidence": stt_result.get("confidence", 0),
                "stt_latency_ms": stt_result.get("latency_ms", 0),
            }
            logger.info("  -> %s: '%s' (matched %d/%d keywords)",
                       entry["status"], entry["transcription"][:60],
                       len(matches), len(expect))
            results.append(entry)

        except Exception as e:
            logger.warning("  -> STT FAIL: %s", e)
            results.append({
                "original": text, "category": utt["category"],
                "status": "error", "error": str(e),
            })

    return results


def test_round_trip(utterances=None) -> list:
    """Full sensory loop: TTS → STT → Core → TTS.

    1. Generate audio from test text (TTS)
    2. Transcribe it back (STT)
    3. Send transcription to Core for a response
    4. Synthesize Core's response (TTS)
    5. Validate the whole chain worked
    """
    if utterances is None:
        utterances = TEST_UTTERANCES

    results = []
    for utt in utterances:
        text = utt["text"]
        logger.info("=== ROUND TRIP: '%s' ===", text)
        entry = {
            "original_text": text,
            "category": utt["category"],
            "stages": {},
        }

        # Stage 1: TTS (generate audio)
        try:
            tts1 = call_tts(text)
            audio_b64 = tts1.get("audio_base64", "")
            sample_rate = tts1.get("sample_rate", 16000)
            entry["stages"]["tts_input"] = {
                "status": "pass" if audio_b64 else "fail",
                "latency_ms": tts1.get("latency_ms", 0),
                "tier": tts1.get("tier_used", "unknown"),
            }
            logger.info("  1. TTS: %d bytes audio generated", len(base64.b64decode(audio_b64)))
        except Exception as e:
            entry["stages"]["tts_input"] = {"status": "error", "error": str(e)}
            entry["status"] = "fail"
            results.append(entry)
            continue

        # Stage 2: STT (transcribe)
        try:
            stt = call_stt(audio_b64, sample_rate)
            transcription = stt.get("text", "")
            entry["stages"]["stt"] = {
                "status": "pass" if transcription else "fail",
                "text": transcription,
                "confidence": stt.get("confidence", 0),
                "latency_ms": stt.get("latency_ms", 0),
            }
            logger.info("  2. STT: '%s' (confidence %.2f)", transcription[:60], stt.get("confidence", 0))
        except Exception as e:
            entry["stages"]["stt"] = {"status": "error", "error": str(e)}
            entry["status"] = "fail"
            results.append(entry)
            continue

        # Stage 3: Core (cognitive response)
        core_input = transcription if transcription else text
        core_response = call_core(core_input)
        entry["stages"]["core"] = {
            "status": "pass" if core_response and "[error" not in core_response else "fail",
            "input": core_input,
            "response": core_response[:300],
        }
        logger.info("  3. Core: '%s'", core_response[:80])

        # Stage 4: TTS (synthesize response)
        if core_response and "[error" not in core_response:
            try:
                # Take first sentence for TTS
                response_text = core_response.split('.')[0].strip()[:200]
                if not response_text:
                    response_text = core_response[:200]
                tts2 = call_tts(response_text)
                audio_out = tts2.get("audio_base64", "")
                entry["stages"]["tts_output"] = {
                    "status": "pass" if audio_out else "fail",
                    "latency_ms": tts2.get("latency_ms", 0),
                    "tier": tts2.get("tier_used", "unknown"),
                    "text_synthesized": response_text[:100],
                }

                # Save output WAV
                if audio_out:
                    wav_path = OUTPUT_DIR / f"roundtrip_{utt['category']}_response.wav"
                    wav_path.write_bytes(base64.b64decode(audio_out))
                    entry["stages"]["tts_output"]["wav_path"] = str(wav_path)

                logger.info("  4. TTS: response synthesized (%d bytes)", len(base64.b64decode(audio_out)))
            except Exception as e:
                entry["stages"]["tts_output"] = {"status": "error", "error": str(e)}
        else:
            entry["stages"]["tts_output"] = {"status": "skipped", "reason": "core failed"}

        # Overall status
        all_pass = all(
            s.get("status") == "pass"
            for s in entry["stages"].values()
        )
        entry["status"] = "pass" if all_pass else "partial"
        results.append(entry)

    return results


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GAIA Audio Test Harness")
    parser.add_argument("--tts-only", action="store_true", help="Test TTS only")
    parser.add_argument("--stt-only", action="store_true", help="Test STT only")
    parser.add_argument("--round-trip", action="store_true", help="Full sensory loop test")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("GAIA Audio Test Harness")
    logger.info("Audio: %s", AUDIO_ENDPOINT)
    logger.info("Core:  %s", CORE_ENDPOINT)
    logger.info("Output: %s", OUTPUT_DIR)
    logger.info("=" * 60)

    # Check audio service
    try:
        req = Request(f"{AUDIO_ENDPOINT}/status")
        with urlopen(req, timeout=5) as resp:
            status = json.loads(resp.read().decode())
        logger.info("Audio status: state=%s, stt=%s, tts=%s",
                    status.get("state"), status.get("stt_model"), status.get("tts_engine"))
    except Exception as e:
        logger.error("Audio service not reachable: %s", e)
        sys.exit(1)

    all_results = {}
    timestamp = datetime.now(timezone.utc).isoformat()

    if args.tts_only:
        all_results["tts"] = test_tts()
    elif args.stt_only:
        all_results["stt"] = test_stt()
    elif args.round_trip:
        all_results["round_trip"] = test_round_trip()
    else:
        # Full test suite
        logger.info("\n--- TTS Tests ---")
        all_results["tts"] = test_tts()
        logger.info("\n--- STT Tests ---")
        all_results["stt"] = test_stt()
        logger.info("\n--- Round Trip Tests ---")
        all_results["round_trip"] = test_round_trip()

    # Save results
    results_path = OUTPUT_DIR / "results.json"
    output = {
        "timestamp": timestamp,
        "audio_endpoint": AUDIO_ENDPOINT,
        "results": all_results,
    }
    results_path.write_text(json.dumps(output, indent=2, default=str))

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS SUMMARY")
    for suite, tests in all_results.items():
        passes = sum(1 for t in tests if t.get("status") == "pass")
        total = len(tests)
        logger.info("  %s: %d/%d passed", suite, passes, total)
    logger.info("Results saved: %s", results_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
