#!/usr/bin/env python3
"""
GAIA Audio Listener — host-side system audio capture daemon.

Captures system audio output via PipeWire or PulseAudio monitor source,
chunks into segments, sends to gaia-audio for transcription, and forwards
accumulated transcripts to gaia-core as audio_listen packets.

Control:
  - Reads ./logs/listener_control.json (written by MCP tools)
  - Writes ./logs/listener_status.json (read by MCP tools)

Usage:
  python scripts/gaia_listener.py [--project-root /gaia/GAIA_Project]

Dependencies (host-only, no GPU):
  - requests
  - PipeWire (pw-cat) or PulseAudio (parec) installed on host
"""

import argparse
import base64
import io
import json
import logging
import os
import signal
import struct
import subprocess
import sys
import threading
import time
import wave
from collections import deque
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHUNK_DURATION_SECONDS = 30
SAMPLE_RATE = 16000         # 16 kHz mono — optimal for Whisper
SAMPLE_WIDTH = 2            # 16-bit PCM
CHANNELS = 1
TRANSCRIPT_BUFFER_MAX = 20  # Keep last N transcriptions (~10 minutes at 30s chunks)
SUMMARY_INTERVAL = 60       # Seconds between transcript summaries sent to gaia-core
CONTROL_POLL_INTERVAL = 5   # Seconds between control file checks

DEFAULT_GAIA_AUDIO_URL = "http://localhost:8080"
DEFAULT_GAIA_CORE_URL = "http://localhost:6415"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [LISTENER] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("gaia_listener")


# ---------------------------------------------------------------------------
# Audio backend detection
# ---------------------------------------------------------------------------

def _detect_audio_backend() -> str:
    """Detect whether PipeWire or PulseAudio is available."""
    # Check PipeWire first (modern default on Arch)
    try:
        result = subprocess.run(
            ["pw-cat", "--version"], capture_output=True, timeout=5
        )
        if result.returncode == 0:
            return "pipewire"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Fall back to PulseAudio
    try:
        result = subprocess.run(
            ["parec", "--version"], capture_output=True, timeout=5
        )
        if result.returncode == 0:
            return "pulseaudio"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    raise RuntimeError(
        "Neither pw-cat (PipeWire) nor parec (PulseAudio) found. "
        "Install pipewire-pulse or pulseaudio."
    )


def _find_monitor_source(backend: str) -> str:
    """Find the monitor source name for system audio capture."""
    if backend == "pipewire":
        try:
            result = subprocess.run(
                ["pw-cli", "list-objects"],
                capture_output=True, text=True, timeout=10,
            )
            # Look for monitor sources in PipeWire
            # On PipeWire, we can use the default monitor
            # pw-cat --record with --target picks up the monitor
        except Exception:
            pass
        # PipeWire: use the default audio sink monitor
        return ""  # pw-cat auto-detects when using --record

    elif backend == "pulseaudio":
        try:
            result = subprocess.run(
                ["pactl", "get-default-sink"],
                capture_output=True, text=True, timeout=5,
            )
            sink_name = result.stdout.strip()
            if sink_name:
                return f"{sink_name}.monitor"
        except Exception:
            pass
        return "@DEFAULT_MONITOR@"

    return ""


def _build_capture_command(backend: str, monitor_source: str) -> list:
    """Build the shell command to capture system audio."""
    if backend == "pipewire":
        cmd = [
            "pw-cat", "--record",
            "--rate", str(SAMPLE_RATE),
            "--channels", str(CHANNELS),
            "--format", "s16",
            "--target", "0",  # 0 = default audio sink monitor
            "-",  # stdout
        ]
    else:  # pulseaudio
        cmd = [
            "parec",
            "--rate", str(SAMPLE_RATE),
            "--channels", str(CHANNELS),
            "--format", "s16le",
            "--device", monitor_source,
            "--raw",
        ]
    return cmd


# ---------------------------------------------------------------------------
# WAV encoding
# ---------------------------------------------------------------------------

def _pcm_to_wav_b64(pcm_data: bytes) -> str:
    """Encode raw PCM data as a base64 WAV string."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_data)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# Transcription + forwarding
# ---------------------------------------------------------------------------

def _transcribe_chunk(audio_b64: str, audio_url: str) -> str:
    """Send a base64 WAV chunk to gaia-audio /transcribe."""
    try:
        resp = requests.post(
            f"{audio_url}/transcribe",
            json={"audio": audio_b64, "format": "wav"},
            timeout=30,
        )
        if resp.status_code == 200:
            data = resp.json()
            return data.get("text", data.get("transcription", "")).strip()
        else:
            logger.warning("Transcription returned %d", resp.status_code)
            return ""
    except Exception as e:
        logger.error("Transcription request failed: %s", e)
        return ""


def _send_to_core(transcript: str, mode: str, core_url: str):
    """Send accumulated transcript to gaia-core as an audio_listen packet."""
    if not transcript.strip():
        return

    try:
        packet = {
            "user_input": f"[AUDIO LISTEN — {mode} mode]\n\n{transcript}",
            "metadata": {
                "source": "audio_listener",
                "mode": mode,
                "packet_type": "audio_listen",
            },
        }
        resp = requests.post(
            f"{core_url}/process_packet",
            json=packet,
            timeout=30,
        )
        if resp.status_code == 200:
            logger.info("Transcript sent to gaia-core (%d chars)", len(transcript))
        else:
            logger.warning("gaia-core returned %d", resp.status_code)
    except Exception as e:
        logger.error("Failed to send transcript to gaia-core: %s", e)


# ---------------------------------------------------------------------------
# Control + status files
# ---------------------------------------------------------------------------

def _read_control(control_path: Path) -> dict:
    """Read the control file. Returns empty dict if missing/invalid."""
    try:
        if control_path.exists():
            return json.loads(control_path.read_text())
    except Exception:
        pass
    return {}


def _write_status(status_path: Path, status: dict):
    """Write the status file atomically."""
    try:
        status["updated_at"] = time.time()
        tmp = status_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(status, indent=2))
        tmp.rename(status_path)
    except Exception as e:
        logger.error("Failed to write status: %s", e)


# ---------------------------------------------------------------------------
# Main capture loop
# ---------------------------------------------------------------------------

class AudioListener:
    """Main listener that captures system audio and transcribes it."""

    def __init__(
        self,
        project_root: Path,
        audio_url: str = DEFAULT_GAIA_AUDIO_URL,
        core_url: str = DEFAULT_GAIA_CORE_URL,
    ):
        self.project_root = project_root
        self.audio_url = audio_url
        self.core_url = core_url
        self.control_path = project_root / "logs" / "listener_control.json"
        self.status_path = project_root / "logs" / "listener_status.json"

        self._running = False
        self._capturing = False
        self._process = None
        self._transcript_buffer = deque(maxlen=TRANSCRIPT_BUFFER_MAX)
        self._mode = "passive"
        self._start_time = None
        self._last_summary_time = 0
        self._stop_event = threading.Event()

        # Detect audio backend
        self._backend = _detect_audio_backend()
        self._monitor = _find_monitor_source(self._backend)
        logger.info("Audio backend: %s, monitor: %s", self._backend, self._monitor or "(auto)")

    def start(self):
        """Start the listener daemon (blocks until stopped)."""
        self._running = True
        self._start_time = time.time()
        logger.info("Listener daemon started. Waiting for control commands...")
        self._update_status(capturing=False)

        try:
            while self._running:
                # Check control file
                control = _read_control(self.control_path)
                command = control.get("command", "")

                if command == "start" and not self._capturing:
                    self._mode = control.get("mode", "passive")
                    logger.info("Starting capture (mode=%s)", self._mode)
                    self._start_capture()

                elif command == "stop" and self._capturing:
                    logger.info("Stopping capture")
                    self._stop_capture()

                # If capturing, read and process audio chunks
                if self._capturing:
                    self._process_chunk()

                    # Periodic summary to gaia-core
                    now = time.time()
                    if now - self._last_summary_time >= SUMMARY_INTERVAL:
                        self._send_summary()
                        self._last_summary_time = now

                self._update_status(capturing=self._capturing)

                if not self._capturing:
                    # Sleep longer when idle
                    self._stop_event.wait(CONTROL_POLL_INTERVAL)
                    if self._stop_event.is_set():
                        break

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt — shutting down")
        finally:
            if self._capturing:
                self._stop_capture()
            self._running = False
            self._update_status(capturing=False, extra={"stopped": True})
            logger.info("Listener daemon stopped")

    def stop(self):
        """Signal the daemon to stop."""
        self._running = False
        self._stop_event.set()

    def _start_capture(self):
        """Start the audio capture subprocess."""
        cmd = _build_capture_command(self._backend, self._monitor)
        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            self._capturing = True
            self._last_summary_time = time.time()
            logger.info("Capture process started (PID %d)", self._process.pid)
        except Exception as e:
            logger.error("Failed to start capture: %s", e)
            self._capturing = False

    def _stop_capture(self):
        """Stop the audio capture subprocess."""
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
            self._process = None
        self._capturing = False

    def _process_chunk(self):
        """Read one chunk of audio from the capture process and transcribe it."""
        if not self._process or not self._process.stdout:
            return

        bytes_per_chunk = SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS * CHUNK_DURATION_SECONDS
        try:
            pcm_data = self._process.stdout.read(bytes_per_chunk)
            if not pcm_data or len(pcm_data) < bytes_per_chunk // 2:
                # Process may have died
                if self._process.poll() is not None:
                    logger.warning("Capture process exited (code=%d)", self._process.returncode)
                    self._capturing = False
                return

            # Check if it's silence (RMS below threshold)
            if self._is_silence(pcm_data):
                logger.debug("Silence detected, skipping chunk")
                return

            # Encode and transcribe
            audio_b64 = _pcm_to_wav_b64(pcm_data)
            text = _transcribe_chunk(audio_b64, self.audio_url)

            if text:
                timestamp = time.strftime("%H:%M:%S")
                entry = f"[{timestamp}] {text}"
                self._transcript_buffer.append(entry)
                logger.info("Transcribed: %s", text[:80])

        except Exception as e:
            logger.error("Chunk processing error: %s", e)

    def _is_silence(self, pcm_data: bytes, threshold: int = 500) -> bool:
        """Check if a PCM chunk is mostly silence (RMS below threshold)."""
        if len(pcm_data) < 4:
            return True
        # Unpack 16-bit signed samples
        n_samples = len(pcm_data) // 2
        try:
            samples = struct.unpack(f"<{n_samples}h", pcm_data[:n_samples * 2])
            rms = (sum(s * s for s in samples) / n_samples) ** 0.5
            return rms < threshold
        except Exception:
            return False

    def _send_summary(self):
        """Send accumulated transcript buffer to gaia-core."""
        if not self._transcript_buffer:
            return
        transcript = "\n".join(self._transcript_buffer)
        _send_to_core(transcript, self._mode, self.core_url)

    def _update_status(self, capturing: bool, extra: dict = None):
        """Write current status to the status file."""
        status = {
            "running": self._running,
            "capturing": capturing,
            "backend": self._backend,
            "mode": self._mode,
            "transcript_buffer_size": len(self._transcript_buffer),
            "uptime_seconds": round(time.time() - self._start_time, 1) if self._start_time else 0,
            "last_chunk_at": None,
        }
        if self._transcript_buffer:
            status["last_transcript_preview"] = self._transcript_buffer[-1][:200]
        if extra:
            status.update(extra)
        _write_status(self.status_path, status)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GAIA Audio Listener daemon")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="GAIA project root directory (default: auto-detect)",
    )
    parser.add_argument(
        "--audio-url",
        default=os.getenv("GAIA_AUDIO_URL", DEFAULT_GAIA_AUDIO_URL),
        help="gaia-audio service URL",
    )
    parser.add_argument(
        "--core-url",
        default=os.getenv("GAIA_CORE_URL", DEFAULT_GAIA_CORE_URL),
        help="gaia-core service URL",
    )
    args = parser.parse_args()

    listener = AudioListener(
        project_root=args.project_root,
        audio_url=args.audio_url,
        core_url=args.core_url,
    )

    # Handle signals gracefully
    def _signal_handler(signum, frame):
        logger.info("Signal %d received, stopping...", signum)
        listener.stop()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    listener.start()


if __name__ == "__main__":
    main()
