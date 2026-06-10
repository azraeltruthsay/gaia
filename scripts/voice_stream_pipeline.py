#!/usr/bin/env python3
"""Streaming voice pipeline — the real-time unlock (GAIA_Project Phase 1.5).

Streams cognition tokens from Core, segments them into sentences, and synthesizes
each sentence on Prime GPU TTS *as it completes* — so GAIA starts speaking sentence 1
while still formulating the rest. Measures TIME-TO-FIRST-AUDIO (first word out the
door) vs the naive "wait for full reply, then synth" baseline.

Runs in gaia-audio (Prime TTS) in the LISTENING gear (Core GGUF-GPU + Prime TTS GPU):
  docker exec gaia-audio python3 /gaia/GAIA_Project/scripts/voice_stream_pipeline.py \
      --prompt "How do you handle a voice conversation?" --out /shared/voice/stream_reply.wav
"""
import argparse, json, re, sys, time, urllib.request, wave
import numpy as np

sys.path.insert(0, "/app")
CORE = "http://gaia-core:6415"
VOICE_SYS = ("You are GAIA, speaking aloud in a live voice conversation. Reply in 2-4 short, "
             "natural spoken sentences — warm and direct. No markdown, no lists.")
# Emit a sentence as soon as it ends in . ! ? (kept simple; speech tolerates it).
SENT = re.compile(r"(.*?[.!?]+)(\s|$)", re.DOTALL)


def stream_tokens(prompt, target, max_tokens):
    """Yield (token, t_arrival) from Core's streaming cognition endpoint."""
    payload = {"prompt": prompt, "target": target, "max_tokens": max_tokens,
               "temperature": 0.3, "system": VOICE_SYS}
    req = urllib.request.Request(f"{CORE}/api/cognitive/stream",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        for line in r:
            line = line.decode().strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "token" in ev:
                yield ev["token"], time.monotonic()
            elif "error" in ev:
                raise RuntimeError(ev["error"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="How quickly can you reply in a back-and-forth voice chat?")
    ap.add_argument("--target", default="core", choices=["core", "prime"])
    ap.add_argument("--max-tokens", type=int, default=120)
    ap.add_argument("--out", default="/shared/voice/stream_reply.wav")
    args = ap.parse_args()

    from gaia_audio.tts_engine import PrimeSpeaker
    from gaia_audio.config import get_config
    cfg = get_config()

    print(f"\n=== STREAMING VOICE PIPELINE  [target={args.target}] ===")
    print(f"  prompt: {args.prompt!r}")
    tts = PrimeSpeaker(model_path=cfg.prime_speaker_model_path,
                       voice_ref_audio=cfg.voice_ref_audio, voice_ref_text=cfg.voice_ref_text)
    tts.load()

    def synth(text):
        return tts.synthesize_sync(text, voice=cfg.voice_ref_audio, ref_text=cfg.voice_ref_text)

    # Warm up: the FIRST synth pays a ~7s CUDA-graph/compile cost. Burn it on a
    # throwaway so the first REAL sentence is fast (this is what makes voice snappy).
    _tw = time.monotonic()
    synth("Ready.")
    print(f"  PrimeSpeaker (GPU) ready + warmed ({time.monotonic()-_tw:.1f}s warmup)\n")

    t0 = time.monotonic()
    buf, full_text = "", ""
    sentences, audio_chunks = [], []
    t_first_token = None
    sr_out = 24000

    def flush_sentence(sent):
        nonlocal sr_out
        sent = sent.strip()
        if not sent:
            return
        t_emit = time.monotonic()
        s = synth(sent)
        t_done = time.monotonic()
        sr_out = s["sample_rate"]
        audio_chunks.append(np.frombuffer(s["audio_bytes"], dtype=np.int16))
        sentences.append({"text": sent, "t_emit": t_emit - t0, "t_synth_done": t_done - t0,
                          "synth_s": round(t_done - t_emit, 2), "audio_s": round(s["duration_seconds"], 2)})
        tag = "  ← FIRST AUDIO" if len(sentences) == 1 else ""
        print(f"  [sent {len(sentences)}] +{t_done-t0:5.2f}s  synth {t_done-t_emit:4.2f}s "
              f"for {s['duration_seconds']:4.1f}s audio  {sent[:50]!r}{tag}")

    # ---- stream tokens, flush sentences as they complete ----
    for tok, t_arr in stream_tokens(args.prompt, args.target, args.max_tokens):
        if t_first_token is None:
            t_first_token = t_arr - t0
            print(f"  first token  +{t_first_token:.2f}s")
        buf += tok
        full_text += tok
        while True:
            m = SENT.match(buf)
            if not m:
                break
            flush_sentence(m.group(1))
            buf = buf[m.end():]
    if buf.strip():
        flush_sentence(buf)
    t_end = time.monotonic()

    # ---- write stitched reply ----
    if audio_chunks:
        allaudio = np.concatenate(audio_chunks)
        with wave.open(args.out, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr_out)
            w.writeframes(allaudio.tobytes())

    # ---- metrics: streaming vs naive baseline ----
    t_first_audio = sentences[0]["t_synth_done"] if sentences else None
    t_full_text = (t_end - t0)  # all tokens received
    total_synth = sum(s["synth_s"] for s in sentences)
    total_audio = sum(s["audio_s"] for s in sentences)
    naive_first_audio = t_full_text + total_synth  # wait for full reply, THEN synth all

    print(f"\n=== METRICS ===")
    print(f"  reply: {full_text.strip()[:120]!r}")
    if not sentences:
        print("  (no audio — cognition returned no tokens; retry / check Core load)")
        return
    print(f"  first token         {t_first_token:.2f}s")
    print(f"  FIRST AUDIO (stream) {t_first_audio:.2f}s   <-- time-to-first-word")
    print(f"  full text received  {t_full_text:.2f}s")
    print(f"  total speech         {total_audio:.1f}s  (synth {total_synth:.1f}s, RTF {total_synth/total_audio:.2f})")
    print(f"  naive non-stream first-audio ~{naive_first_audio:.2f}s")
    if t_first_audio:
        print(f"  >>> streaming saves ~{naive_first_audio - t_first_audio:.1f}s to first word "
              f"({naive_first_audio/t_first_audio:.1f}x sooner)")
    print(f"  reply WAV: {args.out}")


if __name__ == "__main__":
    main()
