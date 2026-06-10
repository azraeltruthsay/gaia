"""Live TTS smoke test — synthesize a sentence through GAIA's voice clone using
the NOW-FIXED config (voice_ref_audio + voice_ref_text from gaia_constants.json).
Confirms the clone path works end-to-end. Run in gaia-audio.
"""
import time, wave
import numpy as np
from gaia_audio.config import get_config
from gaia_audio.tts_engine import NanoSpeaker

TEXT = ("Hello, I'm GAIA. This is a live test of my cloned voice after the "
        "configuration fix. If you can hear me clearly, the clone is wired correctly.")
OUT = "/shared/voice/_smoke_test.wav"

c = get_config()
print(f"[tts] voice_ref_audio={c.voice_ref_audio!r}")
print(f"[tts] voice_ref_text={(c.voice_ref_text or '')[:60]!r}...")

spk = NanoSpeaker(voice_ref_audio=c.voice_ref_audio, voice_ref_text=c.voice_ref_text)
t0 = time.monotonic()
spk.load()
print(f"[tts] NanoSpeaker loaded in {time.monotonic()-t0:.1f}s; synthesizing {len(TEXT)} chars...")

t1 = time.monotonic()
res = spk.synthesize_sync(TEXT)
synth_s = time.monotonic() - t1

with wave.open(OUT, "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(res["sample_rate"])
    w.writeframes(res["audio_bytes"])

print(f"[tts] DONE: {res['duration_seconds']:.1f}s audio @ {res['sample_rate']}Hz "
      f"via {res['engine_used']} in {synth_s:.1f}s -> {OUT}")
