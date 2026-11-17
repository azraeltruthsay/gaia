"""MusicEngine — audio analysis for BPM, key detection, and spectral features.

Uses librosa when available; falls back to basic numpy/scipy analysis.
"""

from __future__ import annotations

import logging
import numpy as np

logger = logging.getLogger("GAIA.Audio.Music")

# Try librosa for full-featured analysis
try:
    import librosa
    _HAS_LIBROSA = True
except ImportError:
    _HAS_LIBROSA = False
    logger.warning("librosa not available — MusicEngine will use basic analysis")


# Key name mapping (Krumhansl-Schmuckler)
_KEY_NAMES = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
]


class MusicEngine:
    """Performs BPM, key, and spectral analysis on audio arrays."""

    def __init__(self):
        self._loaded = False

    def load(self):
        """Initialize the engine (validate dependencies)."""
        if _HAS_LIBROSA:
            logger.info("MusicEngine loaded with librosa backend")
        else:
            logger.info("MusicEngine loaded with basic numpy backend")
        self._loaded = True

    @property
    def loaded(self) -> bool:
        return self._loaded

    def analyze(self, audio_array: np.ndarray, sample_rate: int = 16000) -> dict:
        """Analyze audio and return BPM, key, volume, dynamic range, brightness, and tags.

        Args:
            audio_array: 1-D float32 numpy array of audio samples.
            sample_rate: Sample rate in Hz.

        Returns:
            Dict with keys: bpm, key, volume_db, dynamic_range, brightness, semantic_tags.
        """
        if not self._loaded:
            self.load()

        audio = np.asarray(audio_array, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)

        # Volume (RMS → dB)
        rms = float(np.sqrt(np.mean(audio ** 2))) + 1e-10
        volume_db = float(20 * np.log10(rms))

        if _HAS_LIBROSA:
            return self._analyze_librosa(audio, sample_rate, volume_db)
        return self._analyze_basic(audio, sample_rate, volume_db)

    # ── librosa backend ────────────────────────────────────────────────

    def _analyze_librosa(self, audio: np.ndarray, sr: int, volume_db: float) -> dict:
        # BPM
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
        bpm = float(tempo) if np.isscalar(tempo) else float(tempo[0])

        # Key via chroma
        chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
        chroma_mean = chroma.mean(axis=1)
        key_idx = int(np.argmax(chroma_mean))
        key = _KEY_NAMES[key_idx % 12]

        # Spectral brightness (centroid normalized to Nyquist)
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        brightness = float(np.mean(centroid) / (sr / 2))

        # Dynamic range
        rms_frames = librosa.feature.rms(y=audio)[0]
        rms_db = 20 * np.log10(rms_frames + 1e-10)
        dynamic_range = float(np.max(rms_db) - np.min(rms_db))

        # Semantic tags
        tags = self._generate_tags(bpm, key, volume_db, brightness, dynamic_range)

        return {
            "bpm": round(bpm, 1),
            "key": key,
            "volume_db": round(volume_db, 2),
            "dynamic_range": round(dynamic_range, 2),
            "brightness": round(brightness, 4),
            "semantic_tags": tags,
            "latency_ms": 0.0,  # caller fills this
        }

    # ── basic numpy backend ────────────────────────────────────────────

    def _analyze_basic(self, audio: np.ndarray, sr: int, volume_db: float) -> dict:
        # Simple BPM via autocorrelation
        bpm = self._estimate_bpm_autocorr(audio, sr)

        # Rough key via FFT peak
        key = self._estimate_key_fft(audio, sr)

        # Brightness via spectral centroid (basic FFT)
        fft_mag = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), d=1.0 / sr)
        brightness = float(np.sum(freqs * fft_mag) / (np.sum(fft_mag) + 1e-10) / (sr / 2))

        # Dynamic range (simple windowed RMS)
        win = max(sr // 10, 1)
        n_frames = max(len(audio) // win, 1)
        rms_frames = np.array([
            np.sqrt(np.mean(audio[i * win:(i + 1) * win] ** 2))
            for i in range(n_frames)
        ])
        rms_db = 20 * np.log10(rms_frames + 1e-10)
        dynamic_range = float(np.max(rms_db) - np.min(rms_db))

        tags = self._generate_tags(bpm, key, volume_db, brightness, dynamic_range)

        return {
            "bpm": round(bpm, 1),
            "key": key,
            "volume_db": round(volume_db, 2),
            "dynamic_range": round(dynamic_range, 2),
            "brightness": round(brightness, 4),
            "semantic_tags": tags,
            "latency_ms": 0.0,
        }

    @staticmethod
    def _estimate_bpm_autocorr(audio: np.ndarray, sr: int) -> float:
        """Estimate BPM via autocorrelation of onset envelope."""
        # Simple energy envelope
        win = sr // 20  # 50ms window
        if win < 1 or len(audio) < win * 4:
            return 120.0  # default

        n_frames = len(audio) // win
        envelope = np.array([
            np.sum(audio[i * win:(i + 1) * win] ** 2)
            for i in range(n_frames)
        ])

        # Autocorrelation
        envelope = envelope - np.mean(envelope)
        corr = np.correlate(envelope, envelope, mode="full")
        corr = corr[len(corr) // 2:]

        # Find first peak after minimum BPM threshold (30 BPM)
        fps = sr / win
        min_lag = int(fps * 60 / 300)  # max 300 BPM
        max_lag = int(fps * 60 / 30)   # min 30 BPM
        max_lag = min(max_lag, len(corr) - 1)

        if min_lag >= max_lag:
            return 120.0

        search = corr[min_lag:max_lag + 1]
        peak_idx = int(np.argmax(search)) + min_lag
        if peak_idx == 0:
            return 120.0

        bpm = 60.0 * fps / peak_idx
        return max(30.0, min(300.0, bpm))

    @staticmethod
    def _estimate_key_fft(audio: np.ndarray, sr: int) -> str:
        """Rough key estimation via dominant frequency mapping to pitch class."""
        fft_mag = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), d=1.0 / sr)

        # Focus on musical range (80-2000 Hz)
        mask = (freqs >= 80) & (freqs <= 2000)
        if not np.any(mask):
            return "C"

        fft_musical = fft_mag[mask]
        freq_musical = freqs[mask]
        dominant_freq = freq_musical[np.argmax(fft_musical)]

        # Map to pitch class
        if dominant_freq <= 0:
            return "C"
        midi = 69 + 12 * np.log2(dominant_freq / 440.0)
        key_idx = int(round(midi)) % 12
        return _KEY_NAMES[key_idx]

    @staticmethod
    def _generate_tags(bpm: float, key: str, volume_db: float,
                       brightness: float, dynamic_range: float) -> list[dict]:
        """Generate semantic tags based on analysis results."""
        tags = []

        # Tempo tags
        if bpm < 80:
            tags.append({"category": "tempo", "label": "slow", "confidence": 0.8})
        elif bpm < 120:
            tags.append({"category": "tempo", "label": "moderate", "confidence": 0.8})
        elif bpm < 160:
            tags.append({"category": "tempo", "label": "fast", "confidence": 0.8})
        else:
            tags.append({"category": "tempo", "label": "very_fast", "confidence": 0.8})

        # Energy tags
        if volume_db > -10:
            tags.append({"category": "energy", "label": "loud", "confidence": 0.7})
        elif volume_db > -30:
            tags.append({"category": "energy", "label": "moderate", "confidence": 0.7})
        else:
            tags.append({"category": "energy", "label": "quiet", "confidence": 0.7})

        # Brightness tags
        if brightness > 0.5:
            tags.append({"category": "timbre", "label": "bright", "confidence": 0.6})
        elif brightness < 0.2:
            tags.append({"category": "timbre", "label": "dark", "confidence": 0.6})

        # Dynamic tags
        if dynamic_range > 30:
            tags.append({"category": "dynamics", "label": "dynamic", "confidence": 0.7})
        elif dynamic_range < 10:
            tags.append({"category": "dynamics", "label": "compressed", "confidence": 0.7})

        return tags
