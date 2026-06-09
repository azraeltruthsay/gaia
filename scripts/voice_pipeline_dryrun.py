#!/usr/bin/env python3
"""Offline voice-conversation dry-run harness (GAIA_Project-a1t validation).

Runs a recorded utterance through the EXACT voice_manager stages — minus
Discord / py-cord / opus / HTTP auth — and produces GAIA's spoken reply as a
WAV, with per-stage timings (the latency budget).

  audio file → [decode 16k mono] → STT(in-proc) → VOICE_PRIME → core/process_packet
            → TTS(in-proc) → reply.wav

Run inside gaia-audio (has the STT/TTS engines + models + gaia_common):
  docker exec gaia-audio python3 /gaia/GAIA_Project/scripts/voice_pipeline_dryrun.py \
      --audio /gaia/GAIA_Project/knowledge/transcripts/2026-06-07_S2E4_hosts_episode.m4a \
      --seconds 8 --out /tmp/gaia_reply.wav

No --audio → synthesizes a test prompt via TTS first (closed-loop self-test).
NOTE: STT + TTS want the GPU; park Core (CPU) first or expect contention.
"""
import argparse, json, subprocess, sys, time, urllib.request, wave
import numpy as np

sys.path.insert(0, "/app")  # gaia_audio
CORE = "http://gaia-core:6415"
VOICE_REF = "/models/voice/gaia_identity.wav"


def stage(label):
    print(f"\n── {label} " + "─" * (50 - len(label)))


def decode_to_16k_mono_f32(path, seconds=None):
    """ffmpeg-decode any audio file → 16kHz mono float32 in [-1,1]."""
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", path]
    if seconds:
        cmd += ["-t", str(seconds)]
    cmd += ["-f", "s16le", "-ar", "16000", "-ac", "1", "pipe:1"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def write_wav(path, audio_bytes, sr):
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(audio_bytes)


def core_process_packet(text, user_id="voice_dryrun"):
    """Build a VOICE_PRIME packet and POST to core /process_packet (NDJSON)."""
    from gaia_common.utils.packet_factory import build_packet, PacketSource
    packet = build_packet(PacketSource.VOICE_PRIME, text, user_id=user_id)
    body = json.dumps(packet.to_serializable_dict()).encode()
    req = urllib.request.Request(f"{CORE}/process_packet", data=body,
                                 headers={"Content-Type": "application/json"})
    resp_text = ""
    candidate = None
    with urllib.request.urlopen(req, timeout=90) as r:
        for line in r:
            line = line.decode().strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("type") == "token":
                resp_text += ev.get("value", "")
            elif ev.get("type") == "packet":
                c = ev.get("value", {}).get("response", {}).get("candidate")
                if c:
                    candidate = c
    return (candidate or resp_text).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", help="input utterance (any format); omit for TTS self-test")
    ap.add_argument("--seconds", type=float, default=None, help="trim input to N seconds")
    ap.add_argument("--prompt", default="Hey GAIA, who are you and who created you?",
                    help="text for the TTS self-test when --audio is omitted")
    ap.add_argument("--out", default="/tmp/gaia_reply.wav")
    ap.add_argument("--voice-ref", default=VOICE_REF)
    args = ap.parse_args()

    from gaia_audio.stt_engine import STTEngine
    from gaia_audio.tts_engine import PrimeSpeaker
    from gaia_audio.config import get_config
    cfg = get_config()
    timings = {}

    # Engines (PrimeSpeaker = Qwen3-TTS voice clone; same wiring as main.py)
    stage("loading engines")
    stt = STTEngine(); stt.load()
    tts = PrimeSpeaker(
        model_path=cfg.prime_speaker_model_path,
        voice_ref_audio=cfg.voice_ref_audio,
        voice_ref_text=cfg.voice_ref_text,
    )
    tts.load()

    # Input utterance
    if args.audio:
        stage(f"decode input: {args.audio}")
        audio = decode_to_16k_mono_f32(args.audio, args.seconds)
    else:
        stage(f"TTS self-test input: {args.prompt!r}")
        t0 = time.monotonic()
        synth = tts.synthesize_sync(args.prompt)
        timings["tts_input"] = time.monotonic() - t0
        audio = np.frombuffer(synth["audio_bytes"], dtype=np.int16).astype(np.float32) / 32768.0
        if synth["sample_rate"] != 16000:
            audio = decode_via_resample(synth["audio_bytes"], synth["sample_rate"])
    print(f"  input audio: {len(audio)/16000:.1f}s")

    # ① STT
    stage("① STT (Qwen3-ASR, in-process)")
    t0 = time.monotonic()
    stt_out = stt.transcribe_sync(audio, sample_rate=16000)
    timings["stt"] = time.monotonic() - t0
    transcript = (stt_out.get("text") or "").strip()
    print(f"  transcript: {transcript!r}  ({timings['stt']:.2f}s)")

    # ④ THINK — hybrid voice cognition (mirrors voice_manager). Conversational
    # turns take the fast /api/cognitive/query path on Core/GPU.
    stage("④ COGNITION (fast path: /api/cognitive/query, Core/GPU)")
    t0 = time.monotonic()
    payload = {
        "prompt": transcript or args.prompt, "target": "core", "max_tokens": 160,
        "no_think": True,
        "system": ("You are GAIA, speaking aloud in a live voice conversation. "
                   "Reply in 1-2 short, natural spoken sentences — warm and direct. "
                   "No markdown, no lists, no stage directions."),
    }
    req = urllib.request.Request(f"{CORE}/api/cognitive/query",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=40) as r:
        reply = (json.load(r).get("content") or "").strip()
    timings["cognition"] = time.monotonic() - t0
    print(f"  GAIA: {reply!r}  ({timings['cognition']:.2f}s)")

    # ⑤ SPEAK (TTS). Qwen3-TTS is voice-clone mode → needs a ref audio+text.
    # Use the configured GAIA voice if present; otherwise fall back to the input
    # clip + its transcript so the dry-run can still measure TTS latency.
    stage("⑤ TTS (Qwen3-TTS, in-process)")
    ref_audio, ref_text = cfg.voice_ref_audio, cfg.voice_ref_text
    import os as _os
    if not (ref_audio and _os.path.isfile(ref_audio)):
        ref_audio = "/tmp/_dryrun_ref.wav"
        write_wav(ref_audio, (audio * 32767).clip(-32768, 32767).astype(np.int16).tobytes(), 16000)
        ref_text = transcript or args.prompt
        print(f"  (no GAIA voice ref configured — using input clip as ref, ref_text={ref_text[:50]!r})")
    t0 = time.monotonic()
    spoken = tts.synthesize_sync(reply or "I didn't catch that.", voice=ref_audio, ref_text=ref_text)
    timings["tts"] = time.monotonic() - t0
    write_wav(args.out, spoken["audio_bytes"], spoken["sample_rate"])
    print(f"  reply WAV: {args.out}  ({spoken['duration_seconds']:.1f}s audio, synth {timings['tts']:.2f}s)")

    # Latency budget
    stage("LATENCY BUDGET (turn = STT + cognition + TTS)")
    turn = timings["stt"] + timings["cognition"] + timings["tts"]
    for k in ("stt", "cognition", "tts"):
        print(f"  {k:10s} {timings[k]:6.2f}s")
    print(f"  {'TURN':10s} {turn:6.2f}s  {'(conversational <2-3s)' if turn < 3 else '(SLOW — tune)'}")


def decode_via_resample(audio_bytes, sr):
    raw = subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-f","s16le","-ar",str(sr),
                          "-ac","1","-i","pipe:0","-f","s16le","-ar","16000","-ac","1","pipe:1"],
                         input=audio_bytes, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


if __name__ == "__main__":
    main()
