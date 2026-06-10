"""Is the COLD first Prime-TTS synth valid audio (usable as a real greeting), or garbage?
Synth the same greeting cold (first call) then warm (second call); compare timing + audio.
If cold ≈ warm audibly, the warmup can BE GAIA's spoken greeting at voice_join.
"""
import time, wave
import numpy as np
from gaia_audio.tts_engine import PrimeSpeaker
from gaia_audio.config import get_config

cfg = get_config()
GREETING = "Hey, it's GAIA. I'm here and listening."

tts = PrimeSpeaker(model_path=cfg.prime_speaker_model_path,
                   voice_ref_audio=cfg.voice_ref_audio, voice_ref_text=cfg.voice_ref_text)
tts.load()


def synth_save(text, path, label):
    t = time.monotonic()
    s = tts.synthesize_sync(text, voice=cfg.voice_ref_audio, ref_text=cfg.voice_ref_text)
    dt = time.monotonic() - t
    a = np.frombuffer(s["audio_bytes"], dtype=np.int16).astype(np.float32)
    rms = float(np.sqrt(np.mean(a ** 2)))
    peak = int(np.abs(a).max())
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(s["sample_rate"])
        w.writeframes(s["audio_bytes"])
    print(f"  [{label}] synth {dt:5.2f}s | audio {s['duration_seconds']:4.1f}s | "
          f"rms {rms:5.0f} peak {peak} -> {path}")
    return dt, s["duration_seconds"], rms


print(f"\n=== COLD vs WARM greeting: {GREETING!r} ===")
synth_save(GREETING, "/shared/voice/greeting_cold.wav", "COLD")
synth_save(GREETING, "/shared/voice/greeting_warm.wav", "WARM")
print("\nIf rms/duration are similar, the cold audio is valid -> greeting CAN double as warmup.")
