"""Her hearing sense: what a clip measurably is, as a line she can read.

Core's audio tower was trained on speech. Asked about a dog, rain or a siren
it answered "white noise", "brown noise" or "silence" every time, and stock
Gemma 4 called nearly everything a siren (2026-11-17). So the machine does
the part a model can't: it measures the clip (length, loudness, silence,
separate sounds, tone or noise, pitches and how they move) and names it with
an AudioSet classifier (AST, 527 classes). Core gets the result as plain
facts beside the audio, like a heads-up display, and listens for the rest.

measure() is numpy/scipy only. classify() loads the classifier on first use
(CPU, a fraction of a second for a few seconds of audio) and returns [] if
it can't.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

import numpy as np

logger = logging.getLogger("GAIA.Audio.Hearing")

AST_MODEL_DIR = os.environ.get("HEARING_CLASSIFIER", "/models/ast-audioset")
_NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Frame sizes at 16 kHz (scaled for other rates).
_FRAME_S = 0.05      # loudness frames
_SILENT_DB = -50.0   # a frame below this is silent
_GAP_S = 0.06        # a gap shorter than this doesn't split a sound
_MIN_SOUND_S = 0.03


def note_name(freq: float) -> str:
    if freq <= 0:
        return ""
    n = int(round(12 * np.log2(freq / 440.0) + 69))
    return f"{_NOTES[n % 12]}{n // 12 - 1}"


def _db(x: float) -> float:
    return float(20 * np.log10(max(x, 1e-10)))


def _frames(x: np.ndarray, size: int, hop: int) -> np.ndarray:
    if len(x) < size:
        x = np.pad(x, (0, size - len(x)))
    n = 1 + (len(x) - size) // hop
    idx = np.arange(size)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def _sounds(frame_db: np.ndarray, hop_s: float) -> list[tuple[float, float]]:
    """(start, end) seconds of each separate sound."""
    active = frame_db > max(_SILENT_DB, float(frame_db.max()) - 35.0)
    runs: list[list[int]] = []
    for i, a in enumerate(active):
        if a:
            if runs and i - runs[-1][1] <= max(1, int(_GAP_S / hop_s)):
                runs[-1][1] = i
            else:
                runs.append([i, i])
    out = [(s * hop_s, (e + 1) * hop_s) for s, e in runs]
    return [(s, e) for s, e in out if e - s >= _MIN_SOUND_S]


def _spectrum(x: np.ndarray, sr: int, n_fft: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean power spectrum over the loud frames, and its frequencies."""
    fr = _frames(x, n_fft, n_fft // 4) * np.hanning(n_fft)
    rms = np.sqrt((fr ** 2).mean(axis=1))
    keep = fr[rms >= rms.max() * 0.1] if rms.max() > 0 else fr
    power = (np.abs(np.fft.rfft(keep, axis=1)) ** 2).mean(axis=0)
    return power, np.fft.rfftfreq(n_fft, 1.0 / sr)


def _fundamentals(power: np.ndarray, freqs: np.ndarray) -> list[float]:
    """The distinct tones in a spectrum, harmonics folded into their fundamental."""
    from scipy.signal import find_peaks

    band = (freqs >= 40) & (freqs <= 5000)
    p_db = 10 * np.log10(power + 1e-20)
    p_db[~band] = p_db[band].min() if band.any() else p_db.min()
    peaks, props = find_peaks(p_db, prominence=12, height=p_db[band].max() - 30)
    order = np.argsort(props["peak_heights"])[::-1][:8]
    cands = sorted(float(freqs[peaks[i]]) for i in order)
    fund: list[float] = []
    for f in cands:
        if any(abs(f / g - round(f / g)) < 0.03 and round(f / g) >= 2 for g in fund):
            continue
        fund.append(f)
    return fund[:5]


def _peakiness(power: np.ndarray, freqs: np.ndarray, fund: list[float]) -> float:
    """Share of the power within the fundamentals and their harmonics.
    Steady tones measure 0.93-1.0; natural sounds at most ~0.65."""
    band = (freqs >= 60) & (freqs <= 7000)
    total = float(power[band].sum())
    if not fund or total <= 0:
        return 0.0
    near = np.zeros_like(freqs, dtype=bool)
    for g in fund:
        for k in range(1, 9):
            near |= np.abs(freqs - k * g) <= max(12.0, 0.01 * k * g)
    return float(power[band & near].sum() / total)


def _pitch_track(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Per short frame: dominant frequency (parabolic-interpolated) and how
    much of the frame's power sits at that one peak. Returns (f0, focus, hop_s)."""
    n_fft = 1024 if sr <= 24000 else 2048
    hop = n_fft // 8
    fr = _frames(x, n_fft, hop)
    rms = np.sqrt((fr ** 2).mean(axis=1))
    spec = np.abs(np.fft.rfft(fr * np.hanning(n_fft), axis=1)) ** 2
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    spec[:, freqs < 60] = 0
    k = spec.argmax(axis=1)
    k = np.clip(k, 1, spec.shape[1] - 2)
    rows = np.arange(len(k))
    a, b, c = (np.log(spec[rows, k - 1] + 1e-20), np.log(spec[rows, k] + 1e-20),
               np.log(spec[rows, k + 1] + 1e-20))
    shift = 0.5 * (a - c) / np.where(np.abs(a - 2 * b + c) > 1e-12, a - 2 * b + c, 1e-12)
    f0 = (k + np.clip(shift, -0.5, 0.5)) * (sr / n_fft)
    window = np.zeros_like(spec)
    for d in (-3, -2, -1, 0, 1, 2, 3):
        idx = np.clip(k + d, 0, spec.shape[1] - 1)
        window[rows, idx] = spec[rows, idx]
    focus = window.sum(axis=1) / (spec.sum(axis=1) + 1e-20)
    quiet = rms < rms.max() * 0.1
    f0[quiet] = 0
    focus[quiet] = 0
    return f0, focus, hop / sr


def _movement(track: np.ndarray, hop_s: float) -> str:
    t = track[track > 0]
    if len(t) < 8:
        return ""
    cents = 1200 * np.log2(t / np.median(t))
    # Vibrato first: a wobble around a centre, several times a second. A
    # straight sweep crosses its own trend line once or twice, not often.
    trend = np.polyval(np.polyfit(np.arange(len(cents)), cents, 2), np.arange(len(cents)))
    resid = cents - trend
    crossings = int(np.sum(np.diff(np.sign(resid)) != 0))
    rate = crossings / 2 / (len(t) * hop_s)
    if resid.std() > 15 and 3.0 <= rate <= 10.0:
        if resid.std() > 150:   # a swing of semitones, not a singer's vibrato
            return (f"pitch swings up and down between about {np.percentile(t, 5):.0f} and "
                    f"{np.percentile(t, 95):.0f} Hz, {rate:.0f} times a second")
        return (f"pitch wobbles about ±{resid.std() * 1.4:.0f} cents around {np.median(t):.0f} Hz, "
                f"{rate:.0f} times a second (vibrato)")
    q = max(2, len(t) // 5)
    start, end = float(np.median(t[:q])), float(np.median(t[-q:]))
    top = float(np.percentile(t[q:-q], 95)) if len(t) > 2 * q else max(start, end)
    if top > 1.12 * max(start, end):
        return f"pitch rises from {start:.0f} Hz to about {top:.0f} Hz, then falls back to {end:.0f} Hz"
    semis = 12 * np.log2(end / start)
    if semis > 1:
        return f"pitch rises from {start:.0f} Hz to {end:.0f} Hz"
    if semis < -1:
        return f"pitch falls from {start:.0f} Hz to {end:.0f} Hz"
    return "pitch holds steady"


def measure(audio: np.ndarray, sr: int = 16000) -> dict[str, Any]:
    """What the clip measurably is. audio: mono float in [-1, 1]."""
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=-1)
    dur = len(x) / float(sr) if sr else 0.0
    m: dict[str, Any] = {"duration_s": round(dur, 2)}
    if len(x) == 0:
        m["silent"] = True
        return m
    peak = float(np.abs(x).max())
    m["rms_dbfs"] = round(_db(float(np.sqrt((x ** 2).mean()))), 1)
    m["peak_dbfs"] = round(_db(peak), 1)
    m["clipped"] = bool(np.mean(np.abs(x) >= 0.99) > 0.001)

    size = max(1, int(_FRAME_S * sr))
    hop = max(1, size // 2)
    fdb = np.array([_db(float(np.sqrt((f ** 2).mean()))) for f in _frames(x, size, hop)])
    if fdb.max() < _SILENT_DB:
        m["silent"] = True
        return m
    m["silent"] = False
    sounds = _sounds(fdb, hop / sr)
    m["sounds"] = len(sounds)
    m["sound_spans"] = [[round(s, 2), round(e, 2)] for s, e in sounds[:12]]
    m["sound_time_s"] = round(sum(e - s for s, e in sounds), 2)

    n_fft = 4096 if sr >= 16000 else 2048
    power, freqs = _spectrum(x, sr, n_fft)
    band = (freqs >= 100) & (freqs <= min(7000, sr / 2 - 1))
    pb = power[band] + 1e-20
    flat = float(np.exp(np.mean(np.log(pb))) / np.mean(pb))
    slope = float(np.polyfit(np.log2(freqs[band]), 10 * np.log10(pb), 1)[0])
    fund = _fundamentals(power, freqs)
    peaky = _peakiness(power, freqs, fund)
    m.update(flatness=round(flat, 3), slope_db_per_octave=round(slope, 1), peakiness=round(peaky, 2))

    f0, focus, hop_s = _pitch_track(x, sr)
    voiced = focus[focus > 0]
    focused = float(np.mean(voiced > 0.6)) if len(voiced) else 0.0
    on = f0 > 0
    steps = np.abs(np.diff(np.log2(np.where(on, f0, 1.0))))[on[1:] & on[:-1]]
    smooth = float(np.mean(steps < 0.06)) if len(steps) else 0.0
    m["tonal"] = False
    if peaky >= 0.85 and fund:
        # A steady tone or chord: the notes are worth naming.
        m["tonal"] = True
        m["tones_hz"] = [round(f, 1) for f in fund]
        m["notes"] = [note_name(f) for f in fund]
        m["movement"] = "pitch holds steady"
        if flat > 0.02:
            m["noise_under"] = True   # a tone with hiss under it (pure tones: 0.000)
    elif focused >= 0.6 and smooth >= 0.85:
        # One clear pitch that moves (a sweep, vibrato, a siren): say how.
        m["tonal"] = True
        m["movement"] = _movement(np.where(focus > 0.6, f0, 0), hop_s)
    elif flat > 0.25 or (flat > 0.1 and peaky < 0.45):
        m["noise_colour"] = ("flat, white-like" if slope > -1.5 else
                             "low-heavy, pink/brown-like" if slope < -4.0 else "pink-like")
    return m


# ── Classifier ───────────────────────────────────────────────────────────

_clf_lock = threading.Lock()
_clf: tuple | None = None
_clf_failed = False


def _classifier():
    global _clf, _clf_failed
    if _clf is not None or _clf_failed:
        return _clf
    with _clf_lock:
        if _clf is None and not _clf_failed:
            try:
                import torch
                from transformers import ASTFeatureExtractor, ASTForAudioClassification
                fe = ASTFeatureExtractor.from_pretrained(AST_MODEL_DIR)
                model = ASTForAudioClassification.from_pretrained(AST_MODEL_DIR).eval()
                torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
                _clf = (fe, model)
                logger.info("Hearing: AudioSet classifier loaded from %s", AST_MODEL_DIR)
            except Exception as e:
                _clf_failed = True
                logger.warning("Hearing: classifier unavailable (%s); measuring only", e)
    return _clf


def classify(audio: np.ndarray, sr: int = 16000, top: int = 5, min_score: float = 0.1) -> list[dict]:
    """AudioSet labels with their scores (sigmoid), best first."""
    clf = _classifier()
    if clf is None:
        return []
    import torch
    fe, model = clf
    x = np.asarray(audio, dtype=np.float32)
    if sr != 16000:
        from scipy.signal import resample_poly
        g = np.gcd(int(sr), 16000)
        x = resample_poly(x, 16000 // g, int(sr) // g).astype(np.float32)
    # AST takes ~10 s windows; score each and keep each label's best.
    win = 16000 * 10
    chunks = [x[i:i + win] for i in range(0, max(1, len(x)), win)][:6] or [x]
    best = None
    with torch.no_grad():
        for c in chunks:
            if len(c) < 1600:
                continue
            inputs = fe(c, sampling_rate=16000, return_tensors="pt")
            p = torch.sigmoid(model(**inputs).logits[0])
            best = p if best is None else torch.maximum(best, p)
    if best is None:
        return []
    vals, idx = torch.topk(best, k=min(top, best.numel()))
    labels = model.config.id2label
    return [{"label": labels[int(i)], "score": round(float(v), 2)}
            for v, i in zip(vals, idx) if float(v) >= min_score]


# ── The line she reads ───────────────────────────────────────────────────

def _loudness(db: float) -> str:
    return ("very quiet" if db < -45 else "quiet" if db < -32 else
            "moderate" if db < -20 else "loud" if db < -10 else "very loud")


def hud_line(m: dict, labels: list[dict]) -> str:
    """Declarative facts, no instructions (Core accepts facts, not "you feel")."""
    dur = m.get("duration_s", 0)
    if m.get("silent"):
        return f"Your hearing measured this clip: {dur:.1f} s of silence (nothing above {_SILENT_DB:.0f} dBFS)."
    parts = [f"{dur:.1f} s", f"{_loudness(m['rms_dbfs'])} ({m['rms_dbfs']:.0f} dBFS)"]
    if m.get("clipped"):
        parts.append("clipped (distorted at full scale)")
    n = m.get("sounds", 0)
    if n > 1:
        parts.append(f"{n} separate sounds" if n <= 12 else "many separate sounds")
    elif n == 1 and m.get("sound_time_s", dur) < 0.5 * dur:
        parts.append(f"one sound, {m['sound_time_s']:.1f} s long")
    if m.get("tonal"):
        tones = m.get("tones_hz", [])
        if tones:
            named = ", ".join(f"{f:.0f} Hz ({nm})" for f, nm in zip(tones, m.get("notes", [])))
            parts.append(f"one steady tone at {named}" if len(tones) == 1
                         else f"{len(tones)} tones at once (a chord): {named}")
            if m.get("noise_under"):
                parts.append("with noise under the tone")
        elif m.get("movement"):
            parts.append(f"one clear pitch; {m['movement']}")
    elif m.get("noise_colour"):
        parts.append(f"broadband noise ({m['noise_colour']})")
    line = "Your hearing measured this clip: " + "; ".join(parts) + "."
    if labels:
        line += " Sounds like: " + ", ".join(f"{l['label']} ({l['score']:.2f})" for l in labels) + "."
    return line


def hear(audio: np.ndarray, sr: int = 16000, with_labels: bool = True) -> dict[str, Any]:
    m = measure(audio, sr)
    labels = [] if (m.get("silent") or not with_labels) else classify(audio, sr)
    return {"measured": m, "labels": labels, "hud": hud_line(m, labels)}
