"""
Prime model cognitive state checkpointing.

Manages the prime.md checkpoint file that preserves GAIA's working memory
across GPU sleep/wake cycles.  This is the natural-language replacement for
KV cache persistence: we can't serialize vLLM's KV cache across container
restarts, but we CAN have Prime write down what it was thinking about.

Phase 2: When a model_pool is provided, the checkpoint is generated via an
LLM call (CPU Lite) so the summary captures actual cognitive context rather
than a static template.  Falls back to the deterministic template when no
model is available.

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

# Sentinel file that marks prime.md as already consumed by prompt_builder.
# Exists alongside prime.md while the checkpoint is "stale" (already injected).
_CONSUMED_SENTINEL = ".prime_consumed"

# Summary directory (matches prompt_builder.py)
_SUMMARY_DIR = "data/shared/summaries"


class PrimeCheckpointManager:
    """Manages Prime model's cognitive state checkpointing."""

    def __init__(self, config):
        self.config = config
        shared_dir = getattr(config, "SHARED_DIR", os.getenv("SHARED_DIR", "/shared"))
        self.checkpoint_dir = Path(shared_dir) / "sleep_state"
        self.checkpoint_file = self.checkpoint_dir / "prime.md"
        self.backup_file = self.checkpoint_dir / "prime_previous.md"
        self.history_dir = self.checkpoint_dir / "prime_history"
        self._consumed_flag = self.checkpoint_dir / _CONSUMED_SENTINEL

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

    def create_checkpoint(self, packet=None, model_pool=None) -> Path:
        """Generate and write a cognitive state checkpoint.

        If *model_pool* is provided, uses CPU Lite to generate a rich
        introspective summary.  Otherwise falls back to a deterministic
        template.

        Returns the path to the written checkpoint file.
        """
        logger.info("Creating cognitive checkpoint...")

        # Clear consumed flag — new checkpoint is fresh
        self._consumed_flag.unlink(missing_ok=True)

        state_summary = self._generate_checkpoint(packet, model_pool)
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

    def is_consumed(self) -> bool:
        """Return True if the current checkpoint has already been injected."""
        return self._consumed_flag.exists()

    def mark_consumed(self) -> None:
        """Mark the current checkpoint as consumed (already injected on wake)."""
        try:
            self._consumed_flag.write_text(
                datetime.now(timezone.utc).isoformat(), encoding="utf-8"
            )
            logger.info("Checkpoint marked as consumed")
        except OSError as exc:
            logger.warning("Failed to mark checkpoint consumed: %s", exc)

    def get_checkpoint_history(self, limit: int = 10) -> list:
        """Return list of recent checkpoint (stem, path) tuples."""
        if not self.history_dir.exists():
            return []
        return [
            (p.stem, p)
            for p in sorted(self.history_dir.glob("*.md"), reverse=True)[:limit]
        ]

    # ------------------------------------------------------------------
    # Checkpoint generation (Phase 2: LLM-backed)
    # ------------------------------------------------------------------

    def _generate_checkpoint(self, packet, model_pool) -> str:
        """Try LLM-generated checkpoint; fall back to static template."""
        context = self._extract_context(packet)

        if model_pool is not None:
            try:
                llm = model_pool.get_model_for_role("lite")
                if llm is not None:
                    return self._generate_with_llm(llm, context)
                logger.warning("Lite model unavailable — falling back to template")
            except Exception:
                logger.error("LLM checkpoint generation failed — falling back to template", exc_info=True)

        return self._build_template(context)

    def _generate_with_llm(self, llm, ctx: dict) -> str:
        """Call CPU Lite to introspect on current cognitive state."""
        summary_snippet = self._load_evolving_summary(ctx["session_id"])

        user_prompt = (
            "You are GAIA, preparing for a sleep cycle. Reflect on your current "
            "cognitive state and write a concise checkpoint that your future self "
            "can use to restore context on waking.\n\n"
            f"Session: {ctx['session_id']}\n"
            f"Persona: {ctx['persona']}\n"
            f"Last user message: {ctx['last_prompt']}\n"
        )
        if summary_snippet:
            user_prompt += f"\nConversation summary so far:\n{summary_snippet}\n"

        user_prompt += (
            "\nWrite 2-4 paragraphs covering:\n"
            "- What you were thinking about / working on\n"
            "- Unresolved threads or pending tasks\n"
            "- The emotional tone and relationship context\n"
            "- What you should do first when you wake up\n"
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are GAIA's metacognitive process. Write a first-person "
                    "cognitive state checkpoint. Be specific and concrete — "
                    "reference actual topics, names, and details from the session. "
                    "Do NOT use generic filler. Keep it under 400 words."
                ),
            },
            {"role": "user", "content": user_prompt},
        ]

        result = llm.create_chat_completion(
            messages=messages,
            max_tokens=512,
            temperature=0.3,
            top_p=0.7,
            stream=False,
        )
        summary = result["choices"][0]["message"]["content"].strip()

        now = datetime.now(timezone.utc).isoformat()
        return (
            f"# Prime Cognitive State Checkpoint\n"
            f"**Generated:** {now}\n"
            f"**Session:** {ctx['session_id']}\n"
            f"**Persona:** {ctx['persona']}\n"
            f"**Method:** LLM introspection (CPU Lite)\n"
            f"\n"
            f"## Cognitive State\n"
            f"{summary}\n"
            f"\n"
            f"## Last User Message\n"
            f"{self._truncate(ctx['last_prompt'], 300)}\n"
        )

    # ------------------------------------------------------------------
    # Static template fallback
    # ------------------------------------------------------------------

    def _build_template(self, ctx: dict) -> str:
        """Deterministic template when no LLM is available."""
        now = datetime.now(timezone.utc).isoformat()

        return (
            f"# Prime Cognitive State Checkpoint\n"
            f"**Last Updated:** {now}\n"
            f"**Session ID:** {ctx['session_id']}\n"
            f"**State:** SLEEP_INITIATED\n"
            f"**Method:** static template (no LLM available)\n"
            f"\n"
            f"## Active Context Summary\n"
            f"Last interaction on session {ctx['session_id']}.\n"
            f"{self._truncate(ctx['last_prompt'], 300)}\n"
            f"\n"
            f"## Conversation State\n"
            f"**Last user message:** {self._truncate(ctx['last_prompt'], 200)}\n"
            f"**Persona:** {ctx['persona']}\n"
            f"**Response status:** Complete\n"
            f"\n"
            f"## Next Expected Actions\n"
            f"If woken: Process queued messages with this context available.\n"
            f"If continuing sleep: Proceed with scheduled sleep tasks.\n"
            f"\n"
            f"## Notes\n"
            f"This checkpoint was generated automatically during sleep transition.\n"
            f"Review this content to restore working memory context.\n"
        )

    # ------------------------------------------------------------------
    # Context extraction helpers
    # ------------------------------------------------------------------

    def _extract_context(self, packet) -> dict:
        """Pull session_id, last_prompt, persona from a CognitionPacket."""
        ctx = {"session_id": "unknown", "last_prompt": "none", "persona": "default"}

        if packet is not None:
            hdr = getattr(packet, "header", None)
            if hdr:
                ctx["session_id"] = getattr(hdr, "session_id", None) or "unknown"
            content = getattr(packet, "content", None)
            if content:
                ctx["last_prompt"] = getattr(content, "original_prompt", None) or "none"
            ctx["persona"] = getattr(packet, "persona", None) or "default"

        return ctx

    def _load_evolving_summary(self, session_id: str) -> str:
        """Load the evolving conversation summary for the session, if it exists."""
        summary_path = os.path.join(_SUMMARY_DIR, f"{session_id}.summary")
        try:
            if os.path.exists(summary_path):
                text = Path(summary_path).read_text(encoding="utf-8").strip()
                return self._truncate(text, 1000)
        except OSError:
            pass
        return ""

    @staticmethod
    def _truncate(text: str, max_len: int) -> str:
        if not text or len(text) <= max_len:
            return text or ""
        return text[:max_len] + "..."
