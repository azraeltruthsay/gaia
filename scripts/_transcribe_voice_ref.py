"""Transcribe the GAIA voice-clone reference via the service's own Qwen3-ASR engine.
Used to populate voice_ref_text (improves clone fidelity) + confirm reference content.
Run in gaia-audio: docker exec gaia-audio python3 /gaia/GAIA_Project/scripts/_transcribe_voice_ref.py
"""
import wave
import numpy as np
from gaia_audio.stt_engine import STTEngine

FN = "/shared/voice/gaia_identity.wav"

w = wave.open(FN, "rb")
sr = w.getframerate()
n = w.getnframes()
pcm = w.readframes(n)
w.close()
audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0

print(f"[transcribe] {FN}: {n/sr:.1f}s @ {sr}Hz, {w.getnchannels()} ch")
eng = STTEngine(device="cpu")
eng.load()
res = eng.transcribe_sync(audio, sample_rate=sr, language="en")
print("VOICE_REF_TEXT:", repr(res["text"].strip()))
