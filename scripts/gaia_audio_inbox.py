#!/usr/bin/env python3
"""
gaia_audio_inbox.py — Trigger-based audio inbox daemon.

Polls a control file (logs/inbox_control.json) every 2 seconds.  When a
"process" command is received (from the web dashboard button or the MCP
``audio_inbox_process`` tool), transcribes all queued audio files in
audio_inbox/new/ via gaia-audio, constructs a CognitionPacket per file,
POSTs each to gaia-core /process_packet, and pushes the review to the
dashboard chat via logs/autonomous_messages.jsonl.

Results (transcript, review, metadata) are saved as sidecars in
audio_inbox/done/.

Dependencies (host-only):
  - requests
  - ffmpeg + ffprobe (for transcription pipeline)
  - gaia-audio service running on :8080
  - gaia-core service running on :6415

Usage:
  python scripts/gaia_audio_inbox.py              # foreground
  python scripts/gaia_audio_inbox.py --daemon      # (use systemd instead)
"""

import json
import logging
import os
import signal
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "gaia-core"))
sys.path.insert(0, str(PROJECT_ROOT / "gaia-common"))

from gaia_transcribe import (
    get_audio_duration,
    compute_chunk_boundaries,
    extract_chunk_wav,
    transcribe_chunk,
    transcribe_file,
    stitch_texts,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

AUDIO_EXTENSIONS = {".mp3", ".m4a", ".wav", ".flac", ".ogg", ".aac", ".wma", ".opus"}
CONTROL_POLL_INTERVAL = 2   # seconds between control-file checks
TRANSCRIPT_MAX_CHARS = 30000  # truncate for review prompt

DEFAULT_AUDIO_URL = os.getenv("GAIA_AUDIO_URL", "http://localhost:8080")
DEFAULT_CORE_URL = os.getenv("GAIA_CORE_URL", "http://localhost:6415")

CHUNK_DURATION = 60         # seconds per transcription chunk
CHUNK_OVERLAP = 2           # seconds overlap between chunks
HTTP_TIMEOUT = 120          # seconds per transcription HTTP request

PID_FILE_NAME = "audio_inbox.pid"
STATUS_FILE_NAME = "inbox_status.json"
CONTROL_FILE_NAME = "inbox_control.json"
AUTONOMOUS_LOG_NAME = "autonomous_messages.jsonl"

logger = logging.getLogger("gaia_audio_inbox")


# ---------------------------------------------------------------------------
# PID management (same pattern as gaia_listener.py)
# ---------------------------------------------------------------------------

def _kill_existing(pid_path: Path):
    """Kill any existing inbox daemon and clean up its PID file."""
    if not pid_path.exists():
        return
    try:
        old_pid = int(pid_path.read_text().strip())
        cmdline_path = Path(f"/proc/{old_pid}/cmdline")
        if cmdline_path.exists():
            cmdline = cmdline_path.read_bytes().decode(errors="replace")
            if "gaia_audio_inbox" in cmdline:
                logger.info("Killing existing inbox daemon (PID %d)", old_pid)
                os.kill(old_pid, signal.SIGTERM)
                for _ in range(10):
                    if not cmdline_path.exists():
                        break
                    time.sleep(0.5)
                if cmdline_path.exists():
                    os.kill(old_pid, signal.SIGKILL)
    except (ValueError, ProcessLookupError, PermissionError):
        pass
    try:
        pid_path.unlink(missing_ok=True)
    except Exception:
        pass


def _write_pid(pid_path: Path):
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(str(os.getpid()))


def _remove_pid(pid_path: Path):
    try:
        pid_path.unlink(missing_ok=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Status file (atomic write)
# ---------------------------------------------------------------------------

def _write_status(status_path: Path, status: dict):
    """Write status JSON atomically."""
    try:
        status["updated_at"] = time.time()
        tmp = status_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(status, indent=2))
        tmp.rename(status_path)
    except Exception as e:
        logger.error("Failed to write status: %s", e)


# ---------------------------------------------------------------------------
# CognitionPacket builder (raw dict matching the schema)
# ---------------------------------------------------------------------------

def _build_cognition_packet(
    filename: str,
    duration_str: str,
    transcript: str,
    stats: dict,
) -> dict:
    """Build a CognitionPacket dict for the audio inbox review.

    Constructs the packet as a raw dict (matching the dataclass schema)
    so we don't need gaia-common imports on the host. The packet is
    modeled after consent.py:182-233.
    """
    stem = Path(filename).stem
    session_id = f"audio_inbox_{stem}"
    packet_id = f"pkt-inbox-{uuid.uuid4().hex[:12]}"
    now = datetime.now().isoformat()

    # Truncate transcript for the review prompt
    review_transcript = transcript[:TRANSCRIPT_MAX_CHARS]
    if len(transcript) > TRANSCRIPT_MAX_CHARS:
        review_transcript += f"\n\n[... truncated — full transcript is {len(transcript)} chars]"

    # Switch to Lite model for final review to ensure stability with long context
    prompt = (
        f"::lite\n" # Force Lite and skip intent detection
        f"You are GAIA. You received a new audio transmission in your inbox (Duration: {duration_str}). "
        f"It has already been transcribed for you below. Review this transcript only. "
        f"Do NOT use tools.\n\n"
        f"TRANSCRIPT:\n"
        f"---\n"
        f"{review_transcript}\n"
        f"---\n\n"
        f"Review this content thoughtfully. What is it about? What are the key ideas, "
        f"arguments, or themes? Is there anything particularly interesting, surprising, "
        f"or worth remembering? Share your honest reaction."
    )

    return {
        "version": "0.3",
        "header": {
            "datetime": now,
            "session_id": session_id,
            "packet_id": packet_id,
            "sub_id": "audio_inbox",
            "persona": {
                "identity_id": "audio_inbox",
                "persona_id": "audio_reviewer",
                "role": "Default",
                "tone_hint": "thoughtful",
            },
            "origin": "user",
            "routing": {
                "target_engine": "Lite",
                "priority": 5,
            },
            "model": {
                "name": "auto",
                "provider": "auto",
                "context_window_tokens": 32000,
            },
            "output_routing": {
                "primary": {
                    "destination": "web",
                    "channel_id": session_id,
                    "user_id": "audio_inbox",
                    "metadata": {
                        "source": "audio_inbox",
                        "original_file": filename,
                        "duration": duration_str,
                        "avg_confidence": stats.get("avg_confidence", 0),
                    },
                },
                "source_destination": "web",
                "addressed_to_gaia": True,
            },
            "operational_status": {"status": "initialized"},
        },
        "intent": {
            "user_intent": "audio_inbox_review",
            "system_task": "GenerateDraft",
            "confidence": 0.0,
        },
        "context": {
            "session_history_ref": {"type": "audio_inbox", "value": session_id},
            "cheatsheets": [],
            "constraints": {
                "max_tokens": 4096,
                "time_budget_ms": 120000,
                "safety_mode": "standard",
            },
        },
        "content": {
            "original_prompt": prompt,
            "data_fields": [
                {"key": "audio_transcript", "value": review_transcript, "type": "text"},
            ],
        },
        "reasoning": {},
        "response": {"candidate": "", "confidence": 0.0, "stream_proposal": False},
        "governance": {
            "safety": {"execution_allowed": False, "dry_run": False},
        },
        "metrics": {
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "latency_ms": 0,
        },
        "status": {"finalized": False, "state": "initialized", "next_steps": []},
        "tool_routing": {"ENABLED": False}, # Do not seek tools for the review itself
    }


# ---------------------------------------------------------------------------
# Inbox daemon
# ---------------------------------------------------------------------------

class AudioInboxDaemon:
    """Trigger-based audio inbox: polls a control file, processes on command."""

    def __init__(
        self,
        project_root: Path,
        audio_url: str = DEFAULT_AUDIO_URL,
        core_url: str = DEFAULT_CORE_URL,
    ):
        self.project_root = project_root
        self.audio_url = audio_url
        self.core_url = core_url

        # Inbox directories
        self.inbox_dir = project_root / "audio_inbox"
        self.new_dir = self.inbox_dir / "new"
        self.processing_dir = self.inbox_dir / "processing"
        self.done_dir = self.inbox_dir / "done"

        # Ensure dirs exist
        for d in (self.new_dir, self.processing_dir, self.done_dir):
            d.mkdir(parents=True, exist_ok=True)

        # Control
        self._running = False
        self._stop_event = threading.Event()
        self._start_time = 0.0
        self._current_file = None
        self._files_processed = 0
        self._state = "idle"  # idle | processing

        # Load constants from gaia-core's config logic
        try:
            from gaia_core.config import get_config
            self.config = get_config()
            self.inbox_cfg = self.config.constants.get("AUDIO_INBOX", {})
            logger.info("Loaded AUDIO_INBOX config: %s", self.inbox_cfg)
        except Exception as e:
            logger.error("Failed to load GAIA config: %s", e)
            self.inbox_cfg = {}

        # Paths
        self.pid_path = project_root / "logs" / PID_FILE_NAME
        self.status_path = project_root / "logs" / STATUS_FILE_NAME
        self.control_path = project_root / "logs" / CONTROL_FILE_NAME
        self.autonomous_log_path = project_root / "logs" / AUTONOMOUS_LOG_NAME

    def start(self):
        """Start the inbox daemon (blocks until stopped)."""
        _kill_existing(self.pid_path)
        _write_pid(self.pid_path)

        self._running = True
        self._start_time = time.time()
        logger.info(
            "Audio inbox daemon started (PID %d) — trigger mode, polling %s",
            os.getpid(), self.control_path,
        )

        # Crash recovery: move anything stuck in processing/ back to new/
        self._recover_stuck_files()

        try:
            while self._running:
                cmd = self._read_control()
                if cmd == "process":
                    self._process_all_files()
                self._update_status()
                self._stop_event.wait(CONTROL_POLL_INTERVAL)
                if self._stop_event.is_set():
                    break
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt — shutting down")
        finally:
            self._running = False
            self._state = "idle"
            self._update_status()
            _remove_pid(self.pid_path)
            logger.info("Audio inbox daemon stopped.")

    def stop(self):
        """Signal the daemon to stop."""
        self._running = False
        self._stop_event.set()

    # ── Control file ──────────────────────────────────────────────────────

    def _read_control(self) -> str | None:
        """Read and consume the control file.  Returns the command or None."""
        if not self.control_path.exists():
            return None
        try:
            data = json.loads(self.control_path.read_text())
            self._clear_control()
            return data.get("command")
        except Exception as e:
            logger.error("Failed to read control file: %s", e)
            self._clear_control()
            return None

    def _clear_control(self):
        """Remove the control file after reading."""
        try:
            self.control_path.unlink(missing_ok=True)
        except Exception:
            pass

    # ── Recovery ──────────────────────────────────────────────────────────

    def _recover_stuck_files(self):
        """Move files stuck in processing/ back to new/ on startup."""
        for f in self.processing_dir.iterdir():
            if f.suffix.lower() in AUDIO_EXTENSIONS:
                dest = self.new_dir / f.name
                logger.warning("Recovering stuck file: %s → %s", f.name, dest)
                f.rename(dest)

    # ── Batch processing ──────────────────────────────────────────────────

    def _process_all_files(self):
        """Process all queued files in new/ (FIFO by mtime)."""
        candidates = sorted(
            (f for f in self.new_dir.iterdir() if f.suffix.lower() in AUDIO_EXTENSIONS),
            key=lambda f: f.stat().st_mtime,
        )

        if not candidates:
            logger.info("Process triggered but no files in new/")
            return

        self._state = "processing"
        logger.info("Processing %d file(s)...", len(candidates))
        self._update_status()

        for f in candidates:
            if not self._running:
                break
            self._process_file(f)

        self._state = "idle"

    def _refine_transcript(self, transcript: str, stats: dict) -> str:
        """Use the Lite model to perform semantic diarization and cleanup in chunks."""
        if not self.inbox_cfg.get("refinement_enabled", True):
            logger.info("Refinement disabled in constants; using raw transcript")
            return transcript

        logger.info("Refining transcript (semantic diarization) via chunked processing...")
        
        # 1. Break transcript into manageable chunks (from constants)
        chunk_size = self.inbox_cfg.get("refinement_chunk_size", 4000)
        overlap = self.inbox_cfg.get("refinement_overlap", 200)
        timeout = self.inbox_cfg.get("refinement_timeout_seconds", 90)
        
        chunks = []
        start = 0
        while start < len(transcript):
            end = min(start + chunk_size, len(transcript))
            chunks.append(transcript[start:end])
            if end == len(transcript): break
            start = end - overlap

        refined_chunks = []
        for i, chunk in enumerate(chunks):
            logger.info("  Refining chunk %d/%d (%d chars)...", i+1, len(chunks), len(chunk))
            
            # Simple payload for the dedicated /refine endpoint
            payload = {
                "text": chunk,
                "max_tokens": 2048
            }

            try:
                # Call gaia-audio's new /refine endpoint (Nano-Refiner)
                resp = requests.post(
                    f"{self.audio_url}/refine",
                    json=payload,
                    timeout=timeout,
                )
                if resp.status_code == 200:
                    refined_data = resp.json()
                    refined_text = refined_data.get("refined_text", "")
                    if refined_text:
                        refined_chunks.append(refined_text)
                        continue
                
                logger.warning("Chunk %d refinement failed (HTTP %d); using raw chunk", i+1, resp.status_code)
                refined_chunks.append(chunk)
            except Exception as e:
                logger.error("Chunk %d refinement pass failed: %s", i+1, e)
                refined_chunks.append(chunk)

        # 2. Stitch the refined chunks back together
        if not refined_chunks:
            return transcript

        final_refined = refined_chunks[0]
        for next_chunk in refined_chunks[1:]:
            final_refined = stitch_texts(final_refined, next_chunk)

        logger.info("Refinement complete (%d refined chunks stitched)", len(refined_chunks))
        return final_refined

    def _process_file(self, source: Path):
        """Full pipeline: move → transcribe → refine → build packet → POST → sidecars."""
        filename = source.name
        stem = source.stem
        self._current_file = filename
        logger.info("Processing: %s", filename)

        # Move to processing/
        processing_path = self.processing_dir / filename
        source.rename(processing_path)

        try:
            # 1. Transcribe
            logger.info("Transcribing %s...", filename)
            transcript, stats = transcribe_file(
                path=str(processing_path),
                chunk_duration=CHUNK_DURATION,
                overlap=CHUNK_OVERLAP,
                audio_url=self.audio_url,
                language=None,
                timeout=HTTP_TIMEOUT,
                verbose=False,
            )

            if not transcript.strip():
                logger.warning("No speech detected in %s — moving to done/", filename)
                self._move_to_done(processing_path, stem, transcript="", stats=stats, review="[No speech detected]")
                return

            # 2. Refine (Semantic Diarization)
            transcript = self._refine_transcript(transcript, stats)

            # 3. Build duration string
            dur = stats.get("duration_seconds", 0)
            mins, secs = divmod(int(dur), 60)
            duration_str = f"{mins}m {secs}s"

            # 4. Build CognitionPacket and POST to gaia-core
            packet = _build_cognition_packet(filename, duration_str, transcript, stats)

            logger.info("Sending review packet to gaia-core (%d char transcript)...", len(transcript))
            try:
                resp = requests.post(
                    f"{self.core_url}/process_packet",
                    json=packet,
                    timeout=300,
                )
                if resp.status_code == 200:
                    review_data = resp.json()
                    review_text = review_data.get("response", {}).get("candidate", "")
                    if not review_text:
                        review_text = json.dumps(review_data, indent=2)
                    logger.info("Review received (%d chars)", len(review_text))
                else:
                    review_text = f"[gaia-core returned {resp.status_code}: {resp.text[:500]}]"
                    logger.warning("gaia-core returned %d", resp.status_code)
            except Exception as e:
                review_text = f"[gaia-core request failed: {e}]"
                logger.error("Failed to send packet to gaia-core: %s", e)

            # 4. Push review to dashboard chat (autonomous_messages.jsonl)
            self._push_autonomous_message(filename, duration_str, review_text)

            # 5. Write sidecars and move to done/
            self._move_to_done(processing_path, stem, transcript, stats, review_text)
            self._files_processed += 1
            logger.info("Done processing %s (total: %d files)", filename, self._files_processed)

        except Exception as e:
            logger.error("Failed to process %s: %s", filename, e, exc_info=True)
            # Move back to new/ for retry on next cycle
            try:
                processing_path.rename(self.new_dir / filename)
                logger.info("Moved %s back to new/ for retry", filename)
            except Exception:
                logger.error("Failed to move %s back to new/", filename)

        finally:
            self._current_file = None

    def _move_to_done(
        self,
        processing_path: Path,
        stem: str,
        transcript: str,
        stats: dict,
        review: str,
    ):
        """Move audio file to done/ and write sidecar files."""
        done_audio = self.done_dir / processing_path.name
        processing_path.rename(done_audio)

        # Write transcript sidecar
        transcript_path = self.done_dir / f"{stem}.transcript.txt"
        transcript_path.write_text(transcript, encoding="utf-8")

        # Write review sidecar
        review_path = self.done_dir / f"{stem}.review.txt"
        review_path.write_text(review, encoding="utf-8")

        # Write metadata sidecar
        meta = {
            "original_file": processing_path.name,
            "processed_at": datetime.now().isoformat(),
            "transcript_chars": len(transcript),
            "review_chars": len(review),
            **stats,
        }
        meta_path = self.done_dir / f"{stem}.meta.json"
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def _push_autonomous_message(self, filename: str, duration_str: str, review: str):
        """Append a review entry to autonomous_messages.jsonl for dashboard SSE."""
        if not review or review.startswith("["):
            return  # skip error placeholders
        entry = {
            "type": "autonomous",
            "text": f"**Audio Inbox Review: {filename}** ({duration_str})\n\n{review}",
            "timestamp": datetime.now().isoformat(),
            "source": "audio_inbox",
        }
        try:
            self.autonomous_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.autonomous_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            logger.info("Pushed autonomous message for %s", filename)
        except Exception as e:
            logger.error("Failed to push autonomous message: %s", e)

    def _update_status(self):
        """Write current daemon status to logs/inbox_status.json."""
        # Count files in each state
        new_count = sum(1 for f in self.new_dir.iterdir() if f.suffix.lower() in AUDIO_EXTENSIONS)
        processing_count = sum(1 for f in self.processing_dir.iterdir() if f.suffix.lower() in AUDIO_EXTENSIONS)
        done_count = sum(1 for f in self.done_dir.iterdir() if f.suffix.lower() in AUDIO_EXTENSIONS)

        status = {
            "running": self._running,
            "state": self._state,
            "current_file": self._current_file,
            "files_processed": self._files_processed,
            "queue_depth": new_count,
            "processing": processing_count,
            "completed": done_count,
            "uptime_seconds": round(time.time() - self._start_time, 1) if self._start_time else 0,
            "audio_url": self.audio_url,
            "core_url": self.core_url,
            "inbox_dir": str(self.inbox_dir),
        }
        _write_status(self.status_path, status)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Audio inbox daemon — watches for audio files and reviews them autonomously.",
    )
    parser.add_argument(
        "--audio-url",
        default=DEFAULT_AUDIO_URL,
        help=f"gaia-audio service URL (default: {DEFAULT_AUDIO_URL})",
    )
    parser.add_argument(
        "--core-url",
        default=DEFAULT_CORE_URL,
        help=f"gaia-core service URL (default: {DEFAULT_CORE_URL})",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [INBOX] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    daemon = AudioInboxDaemon(
        project_root=PROJECT_ROOT,
        audio_url=args.audio_url,
        core_url=args.core_url,
    )

    def _signal_handler(signum, frame):
        logger.info("Signal %d received, stopping...", signum)
        daemon.stop()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    daemon.start()


if __name__ == "__main__":
    main()
