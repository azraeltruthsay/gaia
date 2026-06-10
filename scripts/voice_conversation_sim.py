#!/usr/bin/env python3
"""Two-voice conversation simulator + latency time-trials (GAIA_Project Phase 0).

Simulates a spoken back-and-forth: a DISTINCT SYNTHETIC interlocutor (espeak-ng)
is "injected" as audio each turn, transcribed (STT), answered by Core (cognition),
and GAIA's reply is spoken in HER cloned voice (TTS). Every stage is timed.

The decisive metrics for "CPU enough or GPU needed?":
  · formulate  — Core cognition latency (depends on the GEAR at :8092: GPU NF4
                 when AWAKE, CPU GGUF when PARKED). Pass --label to tag the run.
  · TTS RTF    — realtime-factor = synth_seconds / audio_seconds. <1.0 = faster
                 than speech (conversational); >1.0 = can't keep up live.

Runs in gaia-audio (has STT/TTS engines + models + gaia_common; no HMAC needed):
  docker exec gaia-audio python3 /gaia/GAIA_Project/scripts/voice_conversation_sim.py \
      --label core-gpu-awake --target core --prime-tts

A/B: run once AWAKE (Core GPU), once after parking Core (CPU), compare the JSON.
"""
import argparse, json, os, subprocess, sys, time, urllib.request, wave
import numpy as np

sys.path.insert(0, "/app")
CORE = "http://gaia-core:6415"

# A coherent 5-turn dialogue — the interlocutor (a human) talking to GAIA about
# the very thing we're testing. Keeps replies short/natural for a fair latency read.
DIALOGUE = [
    "Hey GAIA, can you hear me okay?",
    "I'm deciding whether to run your voice on CPU or GPU. What matters most for a smooth conversation?",
    "Makes sense. So how quickly can you actually answer in a back-and-forth like this one?",
    "Good to know. If I cut in while you were mid-sentence, could you adjust what you were saying?",
    "Perfect — that's exactly what I was hoping. Thanks, GAIA.",
]
VOICE_SYS = ("You are GAIA, speaking aloud in a live voice conversation. Reply in 1-2 short, "
             "natural spoken sentences — warm and direct. No markdown, no lists, no stage directions.")


def to_16k_mono_f32(audio_bytes, sr):
    if sr == 16000:
        return np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    raw = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "s16le",
                          "-ar", str(sr), "-ac", "1", "-i", "pipe:0", "-f", "s16le",
                          "-ar", "16000", "-ac", "1", "pipe:1"],
                         input=audio_bytes, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def cognition(prompt, target, max_tokens=120):
    # no_think=False: the Qwen-style "/no_think" directive makes the Gemma-4 Core
    # flakily return empty/truncated content; plain mode is reliable + still fast
    # for Core (no <think> blocks emitted).
    payload = {"prompt": prompt, "target": target, "max_tokens": max_tokens,
               "no_think": False, "temperature": 0.3, "system": VOICE_SYS}
    req = urllib.request.Request(f"{CORE}/api/cognitive/query",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return (json.load(r).get("content") or "").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="run", help="tag for this run, e.g. core-gpu-awake / core-cpu-parked")
    ap.add_argument("--target", default="core", choices=["core", "prime"], help="cognition backend")
    ap.add_argument("--turns", type=int, default=len(DIALOGUE))
    ap.add_argument("--prime-tts", action="store_true", help="also measure GAIA TTS on GPU PrimeSpeaker")
    ap.add_argument("--no-tts", action="store_true", help="skip GAIA TTS (measure STT+formulate only — fast A/B)")
    ap.add_argument("--out-dir", default="/shared/voice/sim")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    from gaia_audio.stt_engine import STTEngine
    from gaia_audio.tts_engine import NanoSpeaker, PrimeSpeaker, EspeakFallback
    from gaia_audio.config import get_config
    cfg = get_config()

    print(f"\n=== VOICE CONVERSATION SIM  [label={args.label}  target={args.target}] ===")
    stt = STTEngine(); stt.load()
    interlocutor = EspeakFallback(); interlocutor.load()          # distinct synthetic "human"
    nano = None
    if not args.no_tts:
        nano = NanoSpeaker(voice_ref_audio=cfg.voice_ref_audio, voice_ref_text=cfg.voice_ref_text)
        nano.load()                                                # GAIA voice clone, CPU
    prime = None
    if args.prime_tts and not args.no_tts:
        try:
            prime = PrimeSpeaker(model_path=cfg.prime_speaker_model_path,
                                 voice_ref_audio=cfg.voice_ref_audio, voice_ref_text=cfg.voice_ref_text)
            prime.load()
            print("  PrimeSpeaker (GPU TTS) loaded")
        except Exception as e:
            print(f"  PrimeSpeaker unavailable ({type(e).__name__}: {str(e)[:80]}) — skipping GPU TTS")
            prime = None

    rows, history, convo_chunks = [], [], []

    def speak(engine, text):
        out = engine.synthesize_sync(text) if engine is interlocutor else \
              engine.synthesize_sync(text, voice=cfg.voice_ref_audio, ref_text=cfg.voice_ref_text)
        return out

    for i, human_line in enumerate(DIALOGUE[:args.turns]):
        print(f"\n── turn {i+1}/{args.turns} ──")
        print(f"  HUMAN: {human_line!r}")

        # ① interlocutor speaks (espeak) → audio injected
        t = time.monotonic()
        h = interlocutor.synthesize_sync(human_line)
        t_iloc = time.monotonic() - t
        convo_chunks.append(("human", to_16k_mono_f32(h["audio_bytes"], h["sample_rate"])))

        # ② STT transcribes the injected audio
        audio16 = to_16k_mono_f32(h["audio_bytes"], h["sample_rate"])
        t = time.monotonic()
        stt_out = stt.transcribe_sync(audio16, sample_rate=16000)
        t_stt = time.monotonic() - t
        heard = (stt_out.get("text") or "").strip()
        print(f"  STT heard: {heard!r}  ({t_stt:.2f}s)")

        # ③ formulate (cognition) — threaded dialogue so it's a real conversation
        convo = "\n".join(history + [f"Human: {heard or human_line}", "GAIA:"])
        t = time.monotonic()
        reply = cognition(convo, args.target)
        t_form = time.monotonic() - t
        print(f"  GAIA: {reply!r}  (formulate {t_form:.2f}s)")
        history += [f"Human: {heard or human_line}", f"GAIA: {reply}"]

        # ④ GAIA speaks (TTS) — CPU Nano always; GPU Prime if requested
        tts_metrics = {}
        for name, eng in (("nano_cpu", nano), ("prime_gpu", prime)):
            if eng is None or args.no_tts:
                continue
            t = time.monotonic()
            s = eng.synthesize_sync(reply or "I didn't catch that.",
                                    voice=cfg.voice_ref_audio, ref_text=cfg.voice_ref_text)
            synth_s = time.monotonic() - t
            dur = s["duration_seconds"]
            rtf = synth_s / dur if dur else float("inf")
            tts_metrics[name] = {"synth_s": round(synth_s, 2), "audio_s": round(dur, 2), "rtf": round(rtf, 2)}
            print(f"  TTS[{name}]: {synth_s:.2f}s synth for {dur:.1f}s audio  RTF={rtf:.2f}"
                  f"  {'(realtime-capable)' if rtf < 1 else '(too slow for live)'}")
            if name == "nano_cpu":
                convo_chunks.append(("gaia", to_16k_mono_f32(s["audio_bytes"], s["sample_rate"])))

        rows.append({"turn": i + 1, "heard": heard, "reply": reply,
                     "interlocutor_synth_s": round(t_iloc, 2), "stt_s": round(t_stt, 2),
                     "formulate_s": round(t_form, 2), "tts": tts_metrics,
                     "response_chars": len(reply)})

    # ---- summary ----
    n = len(rows)
    avg = lambda f: round(sum(f(r) for r in rows) / n, 2) if n else 0
    summary = {
        "label": args.label, "target": args.target, "turns": n,
        "avg_stt_s": avg(lambda r: r["stt_s"]),
        "avg_formulate_s": avg(lambda r: r["formulate_s"]),
        "avg_tts_nano_cpu_rtf": avg(lambda r: r["tts"].get("nano_cpu", {}).get("rtf", 0)),
        "avg_tts_nano_cpu_synth_s": avg(lambda r: r["tts"].get("nano_cpu", {}).get("synth_s", 0)),
    }
    if any("prime_gpu" in r["tts"] for r in rows):
        summary["avg_tts_prime_gpu_rtf"] = avg(lambda r: r["tts"].get("prime_gpu", {}).get("rtf", 0))
        summary["avg_tts_prime_gpu_synth_s"] = avg(lambda r: r["tts"].get("prime_gpu", {}).get("synth_s", 0))
    # turn = STT + formulate + best-available TTS synth
    summary["avg_turn_cpu_s"] = round(summary["avg_stt_s"] + summary["avg_formulate_s"]
                                      + summary["avg_tts_nano_cpu_synth_s"], 2)

    out_json = os.path.join(args.out_dir, f"sim_{args.label}.json")
    with open(out_json, "w") as f:
        json.dump({"summary": summary, "turns": rows}, f, indent=2)

    # stitched conversation audio (interlocutor + GAIA interleaved)
    convo_wav = os.path.join(args.out_dir, f"conversation_{args.label}.wav")
    sil = np.zeros(int(0.4 * 16000), dtype=np.float32)
    allaudio = np.concatenate([np.concatenate([a, sil]) for _, a in convo_chunks]) if convo_chunks else np.zeros(1)
    with wave.open(convo_wav, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes((allaudio * 32767).clip(-32768, 32767).astype(np.int16).tobytes())

    print(f"\n=== SUMMARY [{args.label}] ===")
    for k, v in summary.items():
        print(f"  {k:28s} {v}")
    print(f"\n  transcript+latency: {out_json}")
    print(f"  conversation audio: {convo_wav}")


if __name__ == "__main__":
    main()
