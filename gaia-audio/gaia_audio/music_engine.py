"""
MusicEngine — Provides detailed musical and environmental awareness for GAIA.

Uses librosa for DSP (BPM, Key, Energy) and Transformers (AST) for 
semantic tagging (Genre, Mood, Instruments).
"""

import logging
import time
import numpy as np
import librosa
from typing import Dict, Any, Optional
import torch
from transformers import pipeline

logger = logging.getLogger("GAIA.Audio.Music")

class MusicEngine:
    def __init__(self, model_id: str = "MIT/ast-finetuned-audioset-10-10-0.4593"):
        self.model_id = model_id
        self.classifier = None
        self.device = "cpu" # Keep on CPU to save VRAM for Prime/Voice

    def load(self):
        """Load the AST classifier."""
        if self.classifier is not None:
            return
        
        logger.info(f"Loading Audio Spectrogram Transformer: {self.model_id}")
        try:
            self.classifier = pipeline(
                "audio-classification", 
                model=self.model_id, 
                device=self.device
            )
            logger.info("Music analysis model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load music model: {e}")
            raise

    def analyze(self, audio_array: np.ndarray, sr: int = 16000) -> Dict[str, Any]:
        """
        Perform deep analysis of audio: DSP + Semantic.
        Expects float32 normalized audio array.
        """
        t0 = time.monotonic()
        
        # 1. DSP Analysis (Librosa)
        # Ensure sr matches librosa's expectations if needed, though we pass it.
        try:
            # Tempo / BPM
            tempo, _ = librosa.beat.beat_track(y=audio_array, sr=sr)
            bpm = float(tempo[0]) if isinstance(tempo, (np.ndarray, list)) else float(tempo)

            # Chromagram (Harmony/Key)
            chroma = librosa.feature.chroma_stft(y=audio_array, sr=sr)
            mean_chroma = np.mean(chroma, axis=1)
            pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            estimated_key = pitch_classes[np.argmax(mean_chroma)]

            # RMS / Energy (Volume trends)
            rms = librosa.feature.rms(y=audio_array)[0]
            avg_volume_db = 20 * np.log10(np.mean(rms) + 1e-9)
            dynamic_range = float(np.max(rms) - np.min(rms))

            # Spectral Centroid (Timbre/Brightness)
            centroid = librosa.feature.spectral_centroid(y=audio_array, sr=sr)[0]
            brightness = float(np.mean(centroid))

        except Exception as e:
            logger.warning(f"DSP analysis partially failed: {e}")
            bpm, estimated_key, avg_volume_db, dynamic_range, brightness = 0, "Unknown", -100, 0, 0

        # 2. Semantic Analysis (ML)
        tags = []
        if self.classifier:
            try:
                # AST expects 16k mono
                predictions = self.classifier(audio_array)
                # Filter for confidence > 0.1
                tags = [p for p in predictions if p['score'] > 0.1]
            except Exception as e:
                logger.error(f"Semantic audio tagging failed: {e}")

        latency = (time.monotonic() - t0) * 1000
        
        return {
            "bpm": round(bpm, 1),
            "key": estimated_key,
            "volume_db": round(avg_volume_db, 1),
            "dynamic_range": round(dynamic_range, 3),
            "brightness": round(brightness, 1),
            "semantic_tags": tags,
            "latency_ms": round(latency, 1)
        }

    def unload(self):
        """Free memory."""
        if self.classifier:
            del self.classifier
            self.classifier = None
            logger.info("Music analysis model unloaded")
