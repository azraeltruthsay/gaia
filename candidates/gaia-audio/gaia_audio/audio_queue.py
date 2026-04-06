"""Audio Processing Queue — file-based queue with readiness gate.

Audio arrives from multiple sources (listener, inbox, Discord) as files.
This queue holds them until the ASR model is ready, then processes them
in order with proper backpressure.

The readiness gate pattern:
  1. Audio file arrives (from any source)
  2. Enqueued with metadata (source, priority, timestamp)
  3. Processing loop checks model readiness before dequeuing
  4. If model not ready → wait + retry (no busy-spin, uses asyncio.Event)
  5. If model ready → transcribe → route result to appropriate handler

This is the audio equivalent of gaia-web's MessageQueue for text prompts.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("GAIA.Audio.Queue")

_DEFAULT_QUEUE_DIR = os.environ.get(
    "GAIA_AUDIO_QUEUE_DIR", "/shared/audio_queue"
)
_DEFAULT_PROCESSED_DIR = os.environ.get(
    "GAIA_AUDIO_PROCESSED_DIR", "/shared/audio_queue/processed"
)


@dataclass
class QueuedAudio:
    """An audio file waiting to be processed."""

    file_path: str
    source: str  # "listener", "inbox", "discord", "upload"
    priority: int = 0  # Higher = more urgent (discord=10, listener=5, inbox=1)
    queued_at: str = ""
    language: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.queued_at:
            self.queued_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> QueuedAudio:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class AudioProcessingQueue:
    """File-based audio queue with model readiness gate.

    Audio files are tracked in a manifest file (JSON) so the queue
    survives restarts. Processing only happens when the STT model
    reports ready.
    """

    def __init__(
        self,
        queue_dir: str = _DEFAULT_QUEUE_DIR,
        processed_dir: str = _DEFAULT_PROCESSED_DIR,
        max_queue_size: int = 100,
        poll_interval: float = 2.0,
    ) -> None:
        self.queue_dir = Path(queue_dir)
        self.processed_dir = Path(processed_dir)
        self.max_queue_size = max_queue_size
        self.poll_interval = poll_interval

        self._queue: List[QueuedAudio] = []
        self._manifest_path = self.queue_dir / "manifest.json"
        self._model_ready = asyncio.Event()
        self._processing = False
        self._process_task: Optional[asyncio.Task] = None
        self._on_transcribed: Optional[Callable] = None
        self._check_ready_fn: Optional[Callable] = None

        # Ensure directories exist
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # Restore queue from manifest
        self._load_manifest()

        logger.info(
            "AudioQueue initialized (dir=%s, restored=%d items, max=%d)",
            self.queue_dir, len(self._queue), self.max_queue_size,
        )

    def set_ready_checker(self, fn: Callable[[], bool]) -> None:
        """Set the function that checks if the STT model is ready.

        This is called before each dequeue attempt. If it returns False,
        processing waits until the model is available.
        """
        self._check_ready_fn = fn

    def set_transcription_handler(self, fn: Callable) -> None:
        """Set the callback for completed transcriptions.

        fn(audio_item: QueuedAudio, result: dict) → called after each
        successful transcription.
        """
        self._on_transcribed = fn

    def enqueue(self, item: QueuedAudio) -> bool:
        """Add an audio file to the processing queue.

        Returns True if enqueued, False if queue is full.
        """
        if len(self._queue) >= self.max_queue_size:
            logger.warning("Audio queue full (%d items), dropping: %s",
                           len(self._queue), item.file_path)
            return False

        # Verify file exists
        if not Path(item.file_path).exists():
            logger.warning("Audio file not found, skipping: %s", item.file_path)
            return False

        self._queue.append(item)

        # Sort by priority (highest first), then by time (oldest first)
        self._queue.sort(key=lambda x: (-x.priority, x.queued_at))

        self._save_manifest()
        logger.info("Enqueued audio: %s (source=%s, priority=%d, queue_size=%d)",
                     Path(item.file_path).name, item.source, item.priority, len(self._queue))

        # Signal the processing loop
        if self._model_ready.is_set():
            pass  # Processing loop will pick it up

        return True

    def enqueue_file(
        self,
        file_path: str,
        source: str = "listener",
        priority: int = 5,
        language: str = None,
        session_id: str = None,
    ) -> bool:
        """Convenience method to enqueue a file path directly."""
        item = QueuedAudio(
            file_path=file_path,
            source=source,
            priority=priority,
            language=language,
            session_id=session_id,
        )
        return self.enqueue(item)

    @property
    def queue_size(self) -> int:
        return len(self._queue)

    @property
    def is_processing(self) -> bool:
        return self._processing

    def signal_model_ready(self) -> None:
        """Signal that the STT model is loaded and ready for inference."""
        self._model_ready.set()
        logger.debug("AudioQueue: model ready signal received")

    def signal_model_unloaded(self) -> None:
        """Signal that the STT model has been unloaded."""
        self._model_ready.clear()
        logger.debug("AudioQueue: model unloaded signal received")

    async def start_processing(self) -> None:
        """Start the background processing loop."""
        if self._process_task is not None:
            return
        self._process_task = asyncio.create_task(self._process_loop())
        logger.info("AudioQueue processing loop started")

    async def stop_processing(self) -> None:
        """Stop the background processing loop."""
        if self._process_task is not None:
            self._process_task.cancel()
            try:
                await self._process_task
            except asyncio.CancelledError:
                pass
            self._process_task = None
        logger.info("AudioQueue processing loop stopped")

    async def _process_loop(self) -> None:
        """Main processing loop — dequeue and transcribe when model is ready."""
        while True:
            try:
                # Wait for items in queue
                if not self._queue:
                    await asyncio.sleep(self.poll_interval)
                    continue

                # Check model readiness via the registered checker
                model_ready = True
                if self._check_ready_fn:
                    model_ready = self._check_ready_fn()

                if not model_ready:
                    # Model not ready — wait for signal or poll
                    logger.debug("AudioQueue: model not ready, waiting...")
                    self._model_ready.clear()
                    try:
                        await asyncio.wait_for(
                            self._model_ready.wait(),
                            timeout=self.poll_interval * 5,
                        )
                    except asyncio.TimeoutError:
                        continue  # Re-check
                    continue

                # Dequeue highest priority item
                item = self._queue.pop(0)
                self._save_manifest()

                # Process
                self._processing = True
                try:
                    result = await self._transcribe_item(item)
                    if result and self._on_transcribed:
                        await asyncio.coroutine(self._on_transcribed)(item, result) \
                            if asyncio.iscoroutinefunction(self._on_transcribed) \
                            else self._on_transcribed(item, result)

                    # Move processed file
                    self._move_to_processed(item)

                except Exception as e:
                    logger.error("Failed to process audio %s: %s",
                                 item.file_path, e, exc_info=True)
                    # Re-queue with lower priority on failure
                    item.priority = max(0, item.priority - 1)
                    item.metadata["retry_count"] = item.metadata.get("retry_count", 0) + 1
                    if item.metadata["retry_count"] < 3:
                        self._queue.append(item)
                        self._save_manifest()
                    else:
                        logger.warning("Giving up on audio after 3 retries: %s", item.file_path)
                finally:
                    self._processing = False

            except asyncio.CancelledError:
                raise
            except Exception:
                logger.error("AudioQueue process loop error", exc_info=True)
                await asyncio.sleep(self.poll_interval)

    async def _transcribe_item(self, item: QueuedAudio) -> Optional[dict]:
        """Transcribe a single queued audio file."""
        from gaia_audio.stt_engine import STTEngine, audio_bytes_to_array

        file_path = Path(item.file_path)
        if not file_path.exists():
            logger.warning("Audio file disappeared: %s", file_path)
            return None

        logger.info("Transcribing: %s (%s, priority=%d)",
                     file_path.name, item.source, item.priority)

        t0 = time.monotonic()

        # Read and convert audio
        audio_bytes = file_path.read_bytes()
        audio_array = audio_bytes_to_array(audio_bytes)

        # Get the STT engine from the app's GPU manager
        # This import is deferred to avoid circular imports at module level
        try:
            from gaia_audio.main import gpu_manager
            result = await gpu_manager.run_stt(
                gpu_manager.stt.transcribe_sync,
                audio_array=audio_array,
                sample_rate=16000,
                language=item.language,
            )
        except Exception as e:
            logger.error("STT failed for %s: %s", file_path.name, e)
            return None

        elapsed = time.monotonic() - t0
        text = result.get("text", "").strip()
        duration = result.get("duration_seconds", 0)

        logger.info("Transcribed %s: %d chars in %.1fs (audio=%.1fs, source=%s)",
                     file_path.name, len(text), elapsed, duration, item.source)

        result["file_path"] = str(file_path)
        result["source"] = item.source
        result["session_id"] = item.session_id
        result["processing_time_ms"] = round(elapsed * 1000)

        return result

    def _move_to_processed(self, item: QueuedAudio) -> None:
        """Move processed audio file to the processed directory."""
        src = Path(item.file_path)
        if not src.exists():
            return
        dst = self.processed_dir / src.name
        try:
            src.rename(dst)
            # Write metadata sidecar
            sidecar = dst.with_suffix(dst.suffix + ".meta.json")
            sidecar.write_text(json.dumps(item.to_dict(), indent=2, default=str))
        except Exception:
            logger.debug("Could not move processed file", exc_info=True)

    def _save_manifest(self) -> None:
        """Persist queue state to disk."""
        try:
            data = [item.to_dict() for item in self._queue]
            self._manifest_path.write_text(json.dumps(data, indent=2, default=str))
        except Exception:
            logger.debug("Failed to save queue manifest", exc_info=True)

    def _load_manifest(self) -> None:
        """Restore queue state from disk."""
        if not self._manifest_path.exists():
            return
        try:
            data = json.loads(self._manifest_path.read_text())
            self._queue = [QueuedAudio.from_dict(d) for d in data]
            # Remove items whose files no longer exist
            before = len(self._queue)
            self._queue = [q for q in self._queue if Path(q.file_path).exists()]
            if before != len(self._queue):
                logger.info("Pruned %d stale entries from queue manifest", before - len(self._queue))
        except Exception:
            logger.debug("Failed to load queue manifest", exc_info=True)
            self._queue = []


class AudioFileWatcher:
    """Watches a directory for new audio files and enqueues them.

    Uses polling (not inotify) for simplicity and container compatibility.
    Tracks processed files by modification time to avoid re-processing.
    """

    AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".opus", ".wma"}

    def __init__(
        self,
        watch_dir: str,
        queue: AudioProcessingQueue,
        source: str = "listener",
        priority: int = 5,
        poll_interval: float = 2.0,
    ) -> None:
        self.watch_dir = Path(watch_dir)
        self.queue = queue
        self.source = source
        self.priority = priority
        self.poll_interval = poll_interval
        self._last_processed_mtime: float = 0.0
        self._task: Optional[asyncio.Task] = None

        self.watch_dir.mkdir(parents=True, exist_ok=True)

    async def start(self) -> None:
        """Start watching for new audio files."""
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._watch_loop())
        logger.info("AudioFileWatcher started: %s (source=%s, priority=%d)",
                     self.watch_dir, self.source, self.priority)

    async def stop(self) -> None:
        """Stop watching."""
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _watch_loop(self) -> None:
        """Poll directory for new audio files."""
        while True:
            try:
                for f in sorted(self.watch_dir.iterdir()):
                    if f.suffix.lower() not in self.AUDIO_EXTENSIONS:
                        continue
                    if not f.is_file():
                        continue

                    mtime = f.stat().st_mtime
                    if mtime <= self._last_processed_mtime:
                        continue

                    # New file detected
                    # Wait a moment for writes to complete
                    await asyncio.sleep(0.5)

                    # Verify file hasn't been modified (still being written)
                    if f.stat().st_mtime != mtime:
                        continue  # File still being written, skip this cycle

                    self.queue.enqueue_file(
                        str(f), source=self.source, priority=self.priority,
                    )
                    self._last_processed_mtime = mtime

            except asyncio.CancelledError:
                raise
            except Exception:
                logger.debug("AudioFileWatcher error", exc_info=True)

            await asyncio.sleep(self.poll_interval)
