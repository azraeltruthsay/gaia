"""
Multi-Speaker Diarization Module (Phase 6 — Situated Intelligence)

Two-stage pipeline:
  Stage 1 (Acoustic): Segment audio by speaker using energy/pause detection
  Stage 2 (Semantic): Label speakers via Nano LLM refinement

This module adds speaker turn boundaries to the existing STT segments.
When pyannote becomes available, Stage 1 can be upgraded to neural
embedding-based speaker clustering without changing the output schema.

Usage:
    from gaia_audio.diarization import DiarizationEngine
    engine = DiarizationEngine(stt_engine, refiner_engine)
    result = engine.diarize(audio_array, sample_rate=16000)
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("GAIA.Audio.Diarization")

# Minimum pause duration (seconds) to consider a speaker change
MIN_PAUSE_FOR_TURN = 0.8

# Energy threshold (RMS) below which audio is considered silence
SILENCE_RMS_THRESHOLD = 0.01


class DiarizationEngine:
    """Two-stage speaker diarization: acoustic segmentation + semantic labeling."""

    def __init__(self, stt_engine=None, refiner_engine=None):
        self.stt_engine = stt_engine
        self.refiner_engine = refiner_engine

    def diarize(
        self,
        audio_array: np.ndarray,
        sample_rate: int = 16000,
        language: str | None = None,
        num_speakers: int | None = None,
    ) -> Dict[str, Any]:
        """Run full diarization pipeline.

        Args:
            audio_array: Float32 mono audio samples.
            sample_rate: Audio sample rate.
            language: Optional language hint for STT.
            num_speakers: Optional hint for expected number of speakers.

        Returns:
            DiarizedTranscript dict with speaker-labeled turns.
        """
        # Stage 1: Transcribe with timestamps
        if self.stt_engine is None:
            raise RuntimeError("STT engine not available")

        stt_result = self.stt_engine.transcribe_sync(
            audio_array, sample_rate=sample_rate, language=language
        )

        # Stage 1b: Acoustic speaker segmentation (pause-based)
        segments = stt_result.get("segments", [])
        turns = self._segment_by_pauses(segments)

        # Stage 2: Semantic speaker labeling via Nano
        if self.refiner_engine and len(turns) > 1:
            turns = self._label_speakers_semantic(turns, num_speakers)

        # Build output
        speakers = sorted(set(t.get("speaker", "unknown") for t in turns))

        return {
            "text": stt_result.get("text", ""),
            "language": stt_result.get("language"),
            "duration_seconds": stt_result.get("duration_seconds", 0.0),
            "num_speakers": len(speakers),
            "speakers": speakers,
            "turns": turns,
            "context_markers": stt_result.get("context_markers", []),
            "diarization_method": "acoustic+semantic",
        }

    def _segment_by_pauses(self, segments: List[Dict]) -> List[Dict]:
        """Split STT segments into speaker turns based on pause detection.

        Uses timestamp gaps between segments to identify potential speaker
        changes. Segments without timestamps are grouped into one turn.
        """
        if not segments:
            return []

        # Check if we have timestamps
        has_timestamps = all(
            "start" in s and "end" in s for s in segments
        )

        if not has_timestamps:
            # No timestamps — return as single turn
            full_text = " ".join(s.get("text", "") for s in segments)
            return [{
                "speaker": "speaker_1",
                "text": full_text,
                "start": 0.0,
                "end": segments[-1].get("end", 0.0) if segments else 0.0,
            }]

        turns: List[Dict] = []
        current_turn_segments: List[Dict] = [segments[0]]

        for i in range(1, len(segments)):
            prev_end = segments[i - 1].get("end", 0.0)
            curr_start = segments[i].get("start", 0.0)
            gap = curr_start - prev_end

            if gap >= MIN_PAUSE_FOR_TURN:
                # Pause detected — finalize current turn, start new one
                turns.append(self._merge_turn_segments(
                    current_turn_segments, len(turns) + 1
                ))
                current_turn_segments = [segments[i]]
            else:
                current_turn_segments.append(segments[i])

        # Finalize last turn
        if current_turn_segments:
            turns.append(self._merge_turn_segments(
                current_turn_segments, len(turns) + 1
            ))

        return turns

    @staticmethod
    def _merge_turn_segments(segments: List[Dict], turn_number: int) -> Dict:
        """Merge contiguous segments into a single speaker turn."""
        text = " ".join(s.get("text", "") for s in segments).strip()
        start = segments[0].get("start", 0.0)
        end = segments[-1].get("end", 0.0)

        return {
            "speaker": f"speaker_{turn_number}",
            "text": text,
            "start": round(start, 2),
            "end": round(end, 2),
        }

    def _label_speakers_semantic(
        self,
        turns: List[Dict],
        num_speakers: int | None = None,
    ) -> List[Dict]:
        """Use Nano LLM to assign consistent speaker labels.

        Sends the turn-segmented transcript to the refiner with a
        diarization-specific prompt that asks it to assign speaker
        identities based on conversational context.
        """
        # Build a compact turn representation for the LLM
        turn_lines = []
        for i, turn in enumerate(turns):
            turn_lines.append(
                f"[Turn {i+1}, {turn['start']:.1f}s-{turn['end']:.1f}s]: {turn['text']}"
            )

        speaker_hint = ""
        if num_speakers:
            speaker_hint = f" There are {num_speakers} speakers."

        prompt = (
            "The following transcript has been segmented into turns by pauses. "
            "Assign consistent speaker labels (Speaker A, Speaker B, etc.) to "
            "each turn based on conversational context, topic changes, and "
            f"response patterns.{speaker_hint}\n\n"
            "For each turn, output EXACTLY: Speaker X: <original text>\n"
            "Do NOT modify the text content.\n\n"
            + "\n".join(turn_lines)
        )

        try:
            refined = self.refiner_engine.refine(prompt, max_tokens=4096)
            return self._parse_labeled_turns(refined, turns)
        except Exception:
            logger.debug("Semantic speaker labeling failed", exc_info=True)
            return turns  # Return unlabeled turns on failure

    @staticmethod
    def _parse_labeled_turns(
        labeled_text: str, original_turns: List[Dict]
    ) -> List[Dict]:
        """Parse Nano's labeled output back into turn dicts."""
        # Pattern: Speaker A: text content
        pattern = re.compile(
            r"(?:Speaker\s+)?([A-Za-z0-9_]+)\s*:\s*(.+)",
            re.IGNORECASE,
        )

        labeled_turns = []
        lines = labeled_text.strip().splitlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue
            match = pattern.match(line)
            if match:
                speaker = match.group(1).strip()
                text = match.group(2).strip()
                labeled_turns.append({
                    "speaker": speaker,
                    "text": text,
                })

        if not labeled_turns:
            return original_turns

        # Merge timestamps from original turns
        for i, turn in enumerate(labeled_turns):
            if i < len(original_turns):
                turn["start"] = original_turns[i].get("start", 0.0)
                turn["end"] = original_turns[i].get("end", 0.0)
            else:
                turn["start"] = 0.0
                turn["end"] = 0.0

        return labeled_turns


# ── Limb Schema ────────────────────────────────────────────────────────

DIARIZE_LIMB_SCHEMA = {
    "domain": "audio",
    "action": "diarize",
    "description": "Transcribe audio with multi-speaker diarization",
    "params": {
        "audio_base64": {
            "type": "string",
            "description": "Base64-encoded audio data (WAV/MP3/OGG)",
            "required": True,
        },
        "language": {
            "type": "string",
            "description": "Language hint (ISO 639-1 code, e.g. 'en')",
            "required": False,
        },
        "num_speakers": {
            "type": "integer",
            "description": "Expected number of speakers (helps clustering)",
            "required": False,
        },
        "format": {
            "type": "string",
            "description": "Output format: 'full' (default) or 'text_only'",
            "required": False,
            "default": "full",
        },
    },
    "returns": {
        "text": "Full transcript as continuous text",
        "num_speakers": "Number of distinct speakers detected",
        "speakers": "List of speaker labels",
        "turns": [
            {
                "speaker": "Speaker label (e.g. 'A', 'B')",
                "text": "What they said in this turn",
                "start": "Start time in seconds",
                "end": "End time in seconds",
            }
        ],
        "duration_seconds": "Total audio duration",
        "diarization_method": "Pipeline used (acoustic+semantic)",
    },
    "sensitive": False,
}
