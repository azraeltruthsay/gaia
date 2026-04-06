"""Context Compactor — Rolling summarization and budget-aware history management.

Processes conversation history into a budget-aware representation:
- Recent turns: full resolution (every word preserved)
- Middle turns: compressed to key points
- Old turns: collapsed to a single paragraph summary
- Duplicate topics: deduplicated

The compactor runs before the prompt builder assembles Tier 0, ensuring
the conversation history fits within budget while preserving the most
important context.

Integration:
    Called from agent_core before build_from_packet(), or from prompt_builder
    when assembling the history section.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger("GAIA.ContextCompactor")


@dataclass
class CompactedHistory:
    """The result of compacting a conversation history."""

    # Summary of oldest turns (if any were compacted)
    old_summary: str = ""
    old_summary_covers_turns: int = 0

    # Compressed middle turns (key points only)
    middle_turns: List[Dict] = field(default_factory=list)

    # Recent turns at full resolution
    recent_turns: List[Dict] = field(default_factory=list)

    # Estimated token count
    estimated_tokens: int = 0

    # Deduplication notes
    dedup_notes: List[str] = field(default_factory=list)

    def to_messages(self) -> List[Dict]:
        """Convert to a flat list of messages for the prompt builder."""
        messages = []

        if self.old_summary:
            messages.append({
                "role": "system",
                "content": f"[Conversation summary — {self.old_summary_covers_turns} earlier turns]\n{self.old_summary}",
            })

        messages.extend(self.middle_turns)
        messages.extend(self.recent_turns)

        return messages


def _estimate_tokens(text: str) -> int:
    """Rough token estimate — 1 token per ~3.5 characters."""
    return max(1, len(text) // 4)


def _message_tokens(msg: Dict) -> int:
    return _estimate_tokens(msg.get("content", ""))


def _content_hash(text: str) -> str:
    """Short hash for deduplication."""
    return hashlib.md5(text.encode()[:200]).hexdigest()[:8]


class ContextCompactor:
    """Rolling compaction of conversation history.

    Maintains three zones:
    - Recent (full resolution): last N turns
    - Middle (compressed): summarized to key points
    - Old (collapsed): single paragraph covering all oldest turns

    The compactor uses Nano for summarization (fast, always available)
    or falls back to extractive compression if no LLM is available.
    """

    def __init__(
        self,
        recent_turns: int = 6,
        middle_turns: int = 8,
        target_budget_tokens: int = 2000,
        nano_endpoint: str = "http://localhost:8092",
    ):
        self.recent_turns = recent_turns
        self.middle_turns = middle_turns
        self.target_budget = target_budget_tokens
        self.nano_endpoint = nano_endpoint

        # Cache for summaries we've already computed
        self._summary_cache: Dict[str, str] = {}

    def compact(
        self,
        history: List[Dict],
        budget_tokens: int = 0,
    ) -> CompactedHistory:
        """Compact a conversation history to fit within budget.

        Args:
            history: List of {"role": str, "content": str} messages
            budget_tokens: Target token budget (0 = use default)

        Returns:
            CompactedHistory with three zones + dedup notes
        """
        budget = budget_tokens or self.target_budget

        if not history:
            return CompactedHistory()

        total = len(history)

        # If history fits in budget at full resolution, no compaction needed
        full_tokens = sum(_message_tokens(m) for m in history)
        if full_tokens <= budget:
            return CompactedHistory(
                recent_turns=list(history),
                estimated_tokens=full_tokens,
            )

        # Split into zones
        recent_start = max(0, total - self.recent_turns)
        old_end = max(0, recent_start - self.middle_turns)

        old_zone = history[:old_end]
        middle_zone = history[old_end:recent_start]
        recent_zone = history[recent_start:]

        result = CompactedHistory()

        # Zone 1: Recent turns — always full resolution
        result.recent_turns = list(recent_zone)
        recent_tokens = sum(_message_tokens(m) for m in recent_zone)

        # Zone 2: Old turns — collapse to summary
        old_summary = ""
        if old_zone:
            old_summary = self._summarize_turns(old_zone)
            result.old_summary = old_summary
            result.old_summary_covers_turns = len(old_zone)

        old_tokens = _estimate_tokens(old_summary) if old_summary else 0

        # Zone 3: Middle turns — compress if needed to fit budget
        remaining_budget = budget - recent_tokens - old_tokens
        middle_compressed = self._compress_middle(middle_zone, remaining_budget)
        result.middle_turns = middle_compressed

        middle_tokens = sum(_message_tokens(m) for m in middle_compressed)
        result.estimated_tokens = old_tokens + middle_tokens + recent_tokens

        # Deduplication pass
        result.dedup_notes = self._deduplicate(result)

        logger.info(
            "Compacted %d turns → %d tokens (old=%d→%d tok, middle=%d→%d tok, recent=%d→%d tok)",
            total, result.estimated_tokens,
            len(old_zone), old_tokens,
            len(middle_zone), middle_tokens,
            len(recent_zone), recent_tokens,
        )

        return result

    def _summarize_turns(self, turns: List[Dict]) -> str:
        """Summarize a batch of turns into a concise paragraph.

        Uses LLM if available, falls back to extractive summary.
        """
        # Build a cache key from the turns
        key = _content_hash(json.dumps([t.get("content", "")[:50] for t in turns]))
        if key in self._summary_cache:
            return self._summary_cache[key]

        # Try LLM summarization via Core/Nano
        llm_summary = self._llm_summarize(turns)
        if llm_summary:
            self._summary_cache[key] = llm_summary
            return llm_summary

        # Fallback: extractive summary
        summary = self._extractive_summary(turns)
        self._summary_cache[key] = summary
        return summary

    def _llm_summarize(self, turns: List[Dict]) -> Optional[str]:
        """Use the local inference engine to summarize turns."""
        from urllib.request import Request, urlopen

        # Format turns for summarization
        text = "\n".join(
            f"{t['role']}: {t['content'][:200]}" for t in turns
        )

        prompt = (
            "Summarize this conversation excerpt in 2-3 sentences. "
            "Preserve: key decisions, corrections, important facts mentioned, "
            "and any unresolved questions. Drop: greetings, filler, acknowledgments.\n\n"
            f"{text[:2000]}"
        )

        try:
            payload = json.dumps({
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 150,
                "temperature": 0.3,
            }).encode()
            req = Request(
                f"{self.nano_endpoint}/v1/chat/completions",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                # Strip think tags if present
                import re
                text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
                if '</think>' in text:
                    text = re.sub(r'^.*?</think>\s*', '', text, flags=re.DOTALL)
                return text.strip() if text.strip() else None
        except Exception:
            logger.debug("LLM summarization failed, using extractive", exc_info=True)
            return None

    def _extractive_summary(self, turns: List[Dict]) -> str:
        """Fallback: extract key sentences without LLM.

        Takes the first sentence of each user message and the first sentence
        of each assistant message, deduplicates, and joins.
        """
        sentences = []
        for turn in turns:
            content = turn.get("content", "").strip()
            if not content:
                continue
            # Take first sentence
            for end in [". ", "! ", "? ", "\n"]:
                idx = content.find(end)
                if idx > 0:
                    sentences.append(content[:idx + 1])
                    break
            else:
                sentences.append(content[:100] + "...")

        # Deduplicate
        seen = set()
        unique = []
        for s in sentences:
            h = _content_hash(s)
            if h not in seen:
                seen.add(h)
                unique.append(s)

        return " ".join(unique[:6])  # Cap at 6 key sentences

    def _compress_middle(
        self, turns: List[Dict], budget_tokens: int
    ) -> List[Dict]:
        """Compress middle-zone turns to fit within budget.

        Strategy: keep user messages, compress assistant messages to first
        sentence or key point.
        """
        if not turns:
            return []

        if budget_tokens <= 0:
            # No budget for middle — collapse to one-line summary
            summary = self._extractive_summary(turns)
            if summary:
                return [{"role": "system", "content": f"[Compressed: {summary}]"}]
            return []

        compressed = []
        used_tokens = 0

        for turn in turns:
            role = turn.get("role", "user")
            content = turn.get("content", "")

            if role == "user":
                # Keep user messages — they define what was asked
                tokens = _estimate_tokens(content)
                if used_tokens + tokens <= budget_tokens:
                    compressed.append({"role": "user", "content": content})
                    used_tokens += tokens
                else:
                    # Truncate user message
                    remaining = budget_tokens - used_tokens
                    if remaining > 20:
                        truncated = content[:remaining * 4] + "..."
                        compressed.append({"role": "user", "content": truncated})
                        used_tokens = budget_tokens
                    break
            else:
                # Compress assistant messages to first sentence or key point
                short = self._extract_key_point(content)
                tokens = _estimate_tokens(short)
                if used_tokens + tokens <= budget_tokens:
                    compressed.append({"role": "assistant", "content": short})
                    used_tokens += tokens

        return compressed

    def _extract_key_point(self, text: str) -> str:
        """Extract the key point from an assistant response.

        Takes the first substantive sentence, skipping meta-commentary
        and status messages.
        """
        if not text:
            return ""

        # Skip common non-substantive prefixes
        skip_prefixes = [
            "[(", "[GAIA", "One moment", "I'm warming",
            "Let me", "Sure,", "Great question",
        ]

        lines = text.split("\n")
        for line in lines:
            line = line.strip()
            if not line or len(line) < 10:
                continue
            if any(line.startswith(p) for p in skip_prefixes):
                continue
            # Found a substantive line — take first sentence
            for end in [". ", "! ", "? "]:
                idx = line.find(end)
                if 10 < idx < 200:
                    return line[:idx + 1]
            return line[:200] + ("..." if len(line) > 200 else "")

        return text[:100] + "..."

    def _deduplicate(self, result: CompactedHistory) -> List[str]:
        """Detect and annotate duplicate topics across zones."""
        notes = []
        all_messages = result.middle_turns + result.recent_turns

        # Track user message hashes
        seen_hashes = {}
        for i, msg in enumerate(all_messages):
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            # Hash first 100 chars (catches rephrased duplicates roughly)
            h = _content_hash(content[:100].lower())
            if h in seen_hashes:
                notes.append(
                    f"Duplicate topic detected: turn {i} similar to turn {seen_hashes[h]}"
                )
            else:
                seen_hashes[h] = i

        return notes
