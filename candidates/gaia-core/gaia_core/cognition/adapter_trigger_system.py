"""
Adapter Trigger System - Automatic LoRA adapter activation based on content

Monitors user input for keywords/patterns and automatically loads relevant
adapters to enhance GAIA's knowledge for specific topics.

Part of Phase 2 implementation of the GAIA LoRA Adapter Architecture.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class TriggerRule:
    """A rule for triggering adapter activation."""
    adapter_name: str
    tier: int
    patterns: List[str]  # Keywords or regex patterns
    is_regex: bool = False
    min_confidence: float = 0.5  # Minimum match confidence to trigger
    priority: int = 0  # Higher priority triggers win when multiple match
    cooldown_turns: int = 0  # Turns to wait before re-triggering
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TriggerMatch:
    """Result of a trigger match."""
    adapter_name: str
    tier: int
    confidence: float
    matched_patterns: List[str]
    rule: TriggerRule


class AdapterTriggerSystem:
    """
    Monitors input and determines which adapters should be activated.

    Features:
    - Keyword matching (case-insensitive)
    - Regex pattern matching
    - Confidence scoring based on match count
    - Priority handling for conflicting triggers
    - Cooldown to prevent rapid re-triggering
    """

    def __init__(
        self,
        adapter_base_dir: str = "/models/lora_adapters",
        max_concurrent_adapters: int = 3
    ):
        """
        Initialize the trigger system.

        Args:
            adapter_base_dir: Base directory for LoRA adapters
            max_concurrent_adapters: Maximum adapters to load simultaneously
        """
        self.adapter_base_dir = Path(adapter_base_dir)
        self.max_concurrent_adapters = max_concurrent_adapters
        self.rules: List[TriggerRule] = []
        self.active_adapters: Set[str] = set()
        self.cooldown_tracker: Dict[str, int] = {}  # adapter_name -> turns remaining

        # Callback for when adapter loading is requested
        self._load_callback: Optional[Callable[[str, str, int], bool]] = None
        self._unload_callback: Optional[Callable[[str], bool]] = None

        logger.info("AdapterTriggerSystem initialized with base_dir=%s", adapter_base_dir)

    def set_load_callback(self, callback: Callable[[str, str, int], bool]):
        """Set callback for adapter loading: (name, path, tier) -> success."""
        self._load_callback = callback

    def set_unload_callback(self, callback: Callable[[str], bool]):
        """Set callback for adapter unloading: (name) -> success."""
        self._unload_callback = callback

    def load_rules_from_adapters(self) -> int:
        """
        Scan adapter directories and load trigger rules from metadata.

        Returns:
            Number of rules loaded
        """
        self.rules.clear()
        rules_loaded = 0

        tier_dirs = {
            1: self.adapter_base_dir / "tier1_global",
            2: self.adapter_base_dir / "tier2_user",
            3: self.adapter_base_dir / "tier3_session",
        }

        for tier, tier_dir in tier_dirs.items():
            if not tier_dir.exists():
                continue

            for adapter_dir in tier_dir.iterdir():
                if not adapter_dir.is_dir():
                    continue

                metadata_path = adapter_dir / "metadata.json"
                if not metadata_path.exists():
                    continue

                try:
                    with open(metadata_path) as f:
                        metadata = json.load(f)

                    triggers = metadata.get("activation_triggers", [])
                    if not triggers:
                        continue

                    rule = TriggerRule(
                        adapter_name=metadata.get("name", adapter_dir.name),
                        tier=tier,
                        patterns=triggers,
                        is_regex=metadata.get("triggers_are_regex", False),
                        priority=metadata.get("trigger_priority", 0),
                        cooldown_turns=metadata.get("trigger_cooldown", 0),
                        metadata={
                            "path": str(adapter_dir),
                            "pillar": metadata.get("pillar", "general"),
                            "description": metadata.get("description", ""),
                        }
                    )
                    self.rules.append(rule)
                    rules_loaded += 1
                    logger.debug("Loaded trigger rule for adapter '%s': %s",
                                rule.adapter_name, triggers)

                except Exception as e:
                    logger.warning("Error loading triggers from %s: %s", metadata_path, e)

        # Sort by priority (higher first)
        self.rules.sort(key=lambda r: -r.priority)

        logger.info("Loaded %d trigger rules from %d tier directories",
                   rules_loaded, len(tier_dirs))
        return rules_loaded

    def add_rule(self, rule: TriggerRule):
        """Add a trigger rule programmatically."""
        self.rules.append(rule)
        self.rules.sort(key=lambda r: -r.priority)

    def remove_rule(self, adapter_name: str) -> bool:
        """Remove rules for a specific adapter."""
        original_count = len(self.rules)
        self.rules = [r for r in self.rules if r.adapter_name != adapter_name]
        return len(self.rules) < original_count

    def check_triggers(self, text: str) -> List[TriggerMatch]:
        """
        Check text against all trigger rules.

        Args:
            text: Input text to check

        Returns:
            List of matches, sorted by confidence (highest first)
        """
        if not text or not self.rules:
            return []

        text_lower = text.lower()
        matches = []

        for rule in self.rules:
            # Skip if in cooldown
            if rule.adapter_name in self.cooldown_tracker:
                if self.cooldown_tracker[rule.adapter_name] > 0:
                    continue

            matched_patterns = []

            for pattern in rule.patterns:
                if rule.is_regex:
                    try:
                        if re.search(pattern, text, re.IGNORECASE):
                            matched_patterns.append(pattern)
                    except re.error:
                        logger.warning("Invalid regex pattern: %s", pattern)
                else:
                    # Simple keyword matching
                    if pattern.lower() in text_lower:
                        matched_patterns.append(pattern)

            if matched_patterns:
                # Calculate confidence based on match ratio
                confidence = len(matched_patterns) / len(rule.patterns)

                if confidence >= rule.min_confidence:
                    matches.append(TriggerMatch(
                        adapter_name=rule.adapter_name,
                        tier=rule.tier,
                        confidence=confidence,
                        matched_patterns=matched_patterns,
                        rule=rule
                    ))

        # Sort by confidence (descending), then by priority
        matches.sort(key=lambda m: (-m.confidence, -m.rule.priority))

        return matches

    def process_input(
        self,
        text: str,
        auto_load: bool = True
    ) -> Tuple[List[TriggerMatch], List[str], List[str]]:
        """
        Process input and optionally trigger adapter loading.

        Args:
            text: User input text
            auto_load: Whether to automatically load triggered adapters

        Returns:
            Tuple of (matches, newly_loaded, already_active)
        """
        matches = self.check_triggers(text)

        newly_loaded = []
        already_active = []

        for match in matches:
            if match.adapter_name in self.active_adapters:
                already_active.append(match.adapter_name)
                continue

            if auto_load:
                # Check if we're at capacity
                if len(self.active_adapters) >= self.max_concurrent_adapters:
                    # Could implement LRU eviction here
                    logger.warning("Max concurrent adapters reached (%d), skipping %s",
                                 self.max_concurrent_adapters, match.adapter_name)
                    continue

                # Try to load the adapter
                if self._load_callback:
                    adapter_path = match.rule.metadata.get("path", "")
                    success = self._load_callback(
                        match.adapter_name,
                        adapter_path,
                        match.tier
                    )
                    if success:
                        self.active_adapters.add(match.adapter_name)
                        newly_loaded.append(match.adapter_name)

                        # Start cooldown if configured
                        if match.rule.cooldown_turns > 0:
                            self.cooldown_tracker[match.adapter_name] = match.rule.cooldown_turns

                        logger.info("Auto-loaded adapter '%s' (confidence=%.2f, triggers=%s)",
                                  match.adapter_name, match.confidence, match.matched_patterns)

        return matches, newly_loaded, already_active

    def tick_cooldowns(self):
        """Decrement cooldown counters (call once per turn)."""
        expired = []
        for name, turns in self.cooldown_tracker.items():
            if turns > 0:
                self.cooldown_tracker[name] = turns - 1
            if self.cooldown_tracker[name] <= 0:
                expired.append(name)

        for name in expired:
            del self.cooldown_tracker[name]

    def deactivate_adapter(self, adapter_name: str) -> bool:
        """Mark an adapter as no longer active."""
        if adapter_name in self.active_adapters:
            self.active_adapters.remove(adapter_name)

            if self._unload_callback:
                self._unload_callback(adapter_name)

            logger.info("Deactivated adapter '%s'", adapter_name)
            return True
        return False

    def get_active_adapters(self) -> List[str]:
        """Get list of currently active adapter names."""
        return list(self.active_adapters)

    def get_suggested_adapters(self, text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Get adapter suggestions without loading them.

        Useful for showing the user what adapters might be relevant.

        Args:
            text: Input text to analyze
            top_k: Maximum suggestions to return

        Returns:
            List of suggestion dicts with adapter info
        """
        matches = self.check_triggers(text)[:top_k]

        suggestions = []
        for match in matches:
            suggestions.append({
                "adapter_name": match.adapter_name,
                "tier": match.tier,
                "confidence": match.confidence,
                "matched_keywords": match.matched_patterns,
                "description": match.rule.metadata.get("description", ""),
                "pillar": match.rule.metadata.get("pillar", "general"),
                "is_active": match.adapter_name in self.active_adapters,
            })

        return suggestions


# Singleton instance for easy access
_trigger_system: Optional[AdapterTriggerSystem] = None


def get_trigger_system(
    adapter_base_dir: str = "/models/lora_adapters",
    force_new: bool = False
) -> AdapterTriggerSystem:
    """Get or create the singleton trigger system instance."""
    global _trigger_system

    if _trigger_system is None or force_new:
        _trigger_system = AdapterTriggerSystem(adapter_base_dir)
        _trigger_system.load_rules_from_adapters()

    return _trigger_system
