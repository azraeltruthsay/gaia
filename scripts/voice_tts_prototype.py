#!/usr/bin/env python3
"""TTS GPU-plan prototype (GAIA_Project-a1t): NanoSpeaker(CPU) vs PrimeSpeaker(GPU).

The voice dry-run found Core+STT+TTS don't fit on the 15.5GB GPU together. The
audio service's GPUManager already offers two answers:
  A) NanoSpeaker  = Qwen3-TTS 0.6B on CPU (0 VRAM)  → Core+STT stay on GPU
  B) PrimeSpeaker = Qwen3-TTS 1.7B on GPU, time-shared (load→run→unload)

This measures both on the same short voice reply so we can pick. Run in gaia-audio:
  docker exec gaia-audio python3 /gaia/GAIA_Project/scripts/voice_tts_prototype.py \
      --ref-from /gaia/GAIA_Project/knowledge/transcripts/2026-06-07_S2E4_hosts_episode.m4a
"""
import argparse, subprocess, sys, time, wave
import numpy as np
sys.path.insert(0, "/app")

REPLY = ("I'm doing well, thank you for asking. "
         "I just finished a round of self-maintenance and I'm ready to help.")


def write_wav(path, audio_bytes, sr):
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(audio_bytes)


def make_ref(src, seconds=5):
    """Decode the first N seconds of any audio file to a 16k mono wav as a voice ref."""
    out = "/tmp/_tts_ref.wav"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", src,
                    "-t", str(seconds), "-ar", "16000", "-ac", "1", out, "-y"], check=True)
    return out


def gpu_free_mb():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout.strip().splitlines()[0]
        return int(out)
    except Exception:
        return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default=REPLY, help="voice reply text to synthesize")
    ap.add_argument("--ref-from", required=True, help="audio file to extract a voice ref from")
    ap.add_argument("--ref-text", default="I want you to imagine just for a second", help="transcript of the ref clip")
    args = ap.parse_args()

    ref = make_ref(args.ref_from)
    print(f"text ({len(args.text)} chars): {args.text!r}")
    print(f"GPU free at start: {gpu_free_mb()} MiB\n")

    from gaia_audio.tts_engine import NanoSpeaker, PrimeSpeaker
    results = {}

    # --- A) NanoSpeaker (CPU) ---
    print("── A) NanoSpeaker (Qwen3-TTS 0.6B, CPU) " + "─" * 12)
    try:
        nano = NanoSpeaker()
        t0 = time.monotonic(); nano.load(); load_a = time.monotonic() - t0
        t0 = time.monotonic()
        out = nano.synthesize_sync(args.text, voice=ref, ref_text=args.ref_text)
        synth_a = time.monotonic() - t0
        write_wav("/tmp/tts_nano.wav", out["audio_bytes"], out["sample_rate"])
        rtf = synth_a / out["duration_seconds"] if out["duration_seconds"] else 0
        results["nano_cpu"] = (load_a, synth_a, out["duration_seconds"], rtf, 0)
        print(f"  load {load_a:.1f}s | synth {synth_a:.1f}s for {out['duration_seconds']:.1f}s audio "
              f"(RTF {rtf:.2f}) | VRAM 0 | /tmp/tts_nano.wav")
        nano.unload()
    except Exception as e:
        print(f"  FAILED: {e}")

    # --- B) PrimeSpeaker (GPU, time-shared) ---
    print(f"\n── B) PrimeSpeaker (Qwen3-TTS 1.7B, GPU) " + "─" * 11)
    print(f"  GPU free before load: {gpu_free_mb()} MiB")
    try:
        prime = PrimeSpeaker()
        t0 = time.monotonic(); prime.load(); load_b = time.monotonic() - t0
        vram = gpu_free_mb()
        t0 = time.monotonic()
        out = prime.synthesize_sync(args.text, voice=ref, ref_text=args.ref_text)
        synth_b = time.monotonic() - t0
        write_wav("/tmp/tts_prime.wav", out["audio_bytes"], out["sample_rate"])
        rtf = synth_b / out["duration_seconds"] if out["duration_seconds"] else 0
        t0 = time.monotonic(); prime.unload()
        import torch; torch.cuda.empty_cache()
        unload_b = time.monotonic() - t0
        # time-share cost = load + synth + unload (what each GPU-TTS turn pays)
        results["prime_gpu"] = (load_b, synth_b, out["duration_seconds"], rtf, unload_b)
        print(f"  load {load_b:.1f}s | synth {synth_b:.1f}s for {out['duration_seconds']:.1f}s audio "
              f"(RTF {rtf:.2f}) | unload {unload_b:.1f}s | /tmp/tts_prime.wav")
    except Exception as e:
        print(f"  FAILED (likely OOM — no GPU room beside Core): {str(e)[:120]}")

    # --- Verdict ---
    print("\n── COMPARISON (voice turn = the synth latency the user waits on) " + "─" * 3)
    if "nano_cpu" in results:
        l, s, d, r, _ = results["nano_cpu"]
        print(f"  A NanoSpeaker(CPU):  {s:5.1f}s synth   (no GPU, no swap; Core+STT keep GPU)")
    if "prime_gpu" in results:
        l, s, d, r, u = results["prime_gpu"]
        print(f"  B PrimeSpeaker(GPU): {s:5.1f}s synth + {l:.1f}s load + {u:.1f}s unload "
              f"= {l+s+u:5.1f}s per turn (needs GPU room = Core off or shrunk)")
    print("\n  Listen: /tmp/tts_nano.wav vs /tmp/tts_prime.wav for quality.")


if __name__ == "__main__":
    main()
