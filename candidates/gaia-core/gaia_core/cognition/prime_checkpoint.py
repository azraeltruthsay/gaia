"""
Prime model cognitive state checkpointing.

Manages the prime.md checkpoint file that preserves GAIA's working memory
across GPU sleep/wake cycles.  This is the natural-language replacement for
KV cache persistence: we can't serialize vLLM's KV cache across container
restarts, but we CAN have Prime write down what it was thinking about.

Storage uses the existing SHARED_DIR volume mount (/shared) which persists
across container restarts via Docker named volume.
"""

import os
import shutil
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("GAIA.Checkpoint")


class PrimeCheckpointManager:
    """Manages Prime model's cognitive state checkpointing."""

    def __init__(self, config):
        self.config = config
        shared_dir = getattr(config, "SHARED_DIR", os.getenv("SHARED_DIR", "/shared"))
        self.checkpoint_dir = Path(shared_dir) / "sleep_state"
        self.checkpoint_file = self.checkpoint_dir / "prime.md"
        self.backup_file = self.checkpoint_dir / "prime_previous.md"
        self.history_dir = self.checkpoint_dir / "prime_history"

        # Ensure directories exist (best-effort)
        try:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.history_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("Could not create checkpoint dirs %s: %s", self.checkpoint_dir, exc)

        logger.info("Checkpoint directory: %s", self.checkpoint_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_checkpoint(self, packet=None) -> Path:
        """Generate and write a cognitive state checkpoint.

        In the future this will call Prime with a meta-cognitive prompt to
        generate a rich summary.  For now it uses a deterministic template
        built from the current packet context.

        Returns the path to the written checkpoint file.
        """
        logger.info("Creating cognitive checkpoint...")

        state_summary = self._build_checkpoint_content(packet)
        self.checkpoint_file.write_text(state_summary, encoding="utf-8")

        logger.info("Checkpoint written: %s (%d chars)", self.checkpoint_file, len(state_summary))
        return self.checkpoint_file

    def rotate_checkpoints(self) -> None:
        """Back up current checkpoint before overwriting.

        prime.md → prime_previous.md
        prime.md → prime_history/<timestamp>-sleep.md
        """
        if not self.checkpoint_file.exists():
            return

        try:
            shutil.copy2(self.checkpoint_file, self.backup_file)

            ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
            archive_path = self.history_dir / f"{ts}-sleep.md"
            shutil.copy2(self.checkpoint_file, archive_path)

            logger.info("Checkpoint archived: %s", archive_path)
        except OSError as exc:
            logger.error("Checkpoint rotation failed: %s", exc, exc_info=True)

    def load_latest(self) -> str:
        """Load the most recent checkpoint content."""
        if not self.checkpoint_file.exists():
            logger.warning("No checkpoint file found at %s", self.checkpoint_file)
            return ""

        try:
            content = self.checkpoint_file.read_text(encoding="utf-8")
            logger.info("Checkpoint loaded: %d chars", len(content))
            return content
        except OSError as exc:
            logger.error("Checkpoint load failed: %s", exc, exc_info=True)
            return ""

    def get_checkpoint_history(self, limit: int = 10) -> list:
        """Return list of recent checkpoint (stem, path) tuples."""
        if not self.history_dir.exists():
            return []
        return [
            (p.stem, p)
            for p in sorted(self.history_dir.glob("*.md"), reverse=True)[:limit]
        ]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_checkpoint_content(self, packet) -> str:
        """Build checkpoint from packet context.

        When Prime-generated summaries are enabled (Phase 2+), this will
        be replaced by an actual LLM call.
        """
        now = datetime.now(timezone.utc).isoformat()

        session_id = "unknown"
        last_prompt = "none"
        persona = "default"

        if packet is not None:
            hdr = getattr(packet, "header", None)
            if hdr:
                session_id = getattr(hdr, "session_id", None) or "unknown"
            content = getattr(packet, "content", None)
            if content:
                last_prompt = getattr(content, "original_prompt", None) or "none"
            persona = getattr(packet, "persona", None) or "default"

        return f"""\
# Prime Cognitive State Checkpoint
**Last Updated:** {now}
**Session ID:** {session_id}
**State:** SLEEP_INITIATED

## Active Context Summary
Last interaction on session {session_id}.
{self._truncate(last_prompt, 300)}

## Conversation State
**Last user message:** {self._truncate(last_prompt, 200)}
**Persona:** {persona}
**Response status:** Complete

## Reasoning State
**Current task:** Context preservation across sleep/wake cycle
**Confidence:** High (checkpoint system operational)

## Tone & Relationship Context
Technical collaboration expected.
Detail-oriented explanations when appropriate.

## Next Expected Actions
If woken: Process queued messages with this context available.
If continuing sleep: Proceed with scheduled sleep tasks.

## Notes
This checkpoint was generated automatically during sleep transition.
Review this content to restore working memory context.
"""

    @staticmethod
    def _truncate(text: str, max_len: int) -> str:
        if not text or len(text) <= max_len:
            return text or ""
        return text[:max_len] + "..."
