"""
Loop Recovery System for GAIA Cognitive Pipeline.

Orchestrates the reset flow when a loop is detected:
1. Capture current packet state
2. Inject recovery context into next prompt
3. Manage escalation ladder
4. Coordinate with session manager

Design: 2026-02-04 (see Dev_Notebook/2026-02-04_loop_detection_reset_system.md)
"""

from __future__ import annotations
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable

from gaia_core.cognition.loop_detector import (
    LoopDetector,
    LoopDetectorConfig,
    AggregatedResult,
    LoopCategory
)
from gaia_core.cognition.loop_patterns import PatternRenderer

logger = logging.getLogger("GAIA.LoopRecovery")


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class LoopMetadata:
    """
    Metadata about a detected loop, attached to packets for context.
    """
    detected_at: str
    loop_type: str  # LoopCategory value
    pattern: str
    pattern_hash: str
    reset_count: int
    confidence: float
    previous_attempts: List[Dict[str, str]] = field(default_factory=list)
    recovery_context: str = ""  # Injected into prompt
    triggered_by: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "detected_at": self.detected_at,
            "loop_type": self.loop_type,
            "pattern": self.pattern,
            "pattern_hash": self.pattern_hash,
            "reset_count": self.reset_count,
            "confidence": self.confidence,
            "previous_attempts": self.previous_attempts,
            "recovery_context": self.recovery_context,
            "triggered_by": self.triggered_by
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LoopMetadata:
        """Create from dictionary."""
        return cls(
            detected_at=data.get("detected_at", ""),
            loop_type=data.get("loop_type", ""),
            pattern=data.get("pattern", ""),
            pattern_hash=data.get("pattern_hash", ""),
            reset_count=data.get("reset_count", 0),
            confidence=data.get("confidence", 0.0),
            previous_attempts=data.get("previous_attempts", []),
            recovery_context=data.get("recovery_context", ""),
            triggered_by=data.get("triggered_by", [])
        )


@dataclass
class CapturedState:
    """
    Captured state before reset for context preservation.
    """
    session_id: str
    packet_id: str
    goal: str
    last_output: str
    tool_history: List[Dict[str, Any]]
    error_history: List[Dict[str, Any]]
    loop_metadata: LoopMetadata
    captured_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "packet_id": self.packet_id,
            "goal": self.goal,
            "last_output": self.last_output,
            "tool_history": self.tool_history,
            "error_history": self.error_history,
            "loop_metadata": self.loop_metadata.to_dict(),
            "captured_at": self.captured_at
        }


# =============================================================================
# Recovery Manager
# =============================================================================

class LoopRecoveryManager:
    """
    Manages the loop detection and recovery lifecycle.

    Responsibilities:
    - Coordinate detection checks
    - Capture state before reset
    - Generate recovery context for re-injection
    - Manage escalation ladder
    - Track recovery success/failure
    """

    def __init__(
        self,
        config: Optional[LoopDetectorConfig] = None,
        on_warn: Optional[Callable[[AggregatedResult], None]] = None,
        on_block: Optional[Callable[[AggregatedResult], None]] = None,
        on_escalate: Optional[Callable[[int], None]] = None
    ):
        """
        Initialize the recovery manager.

        Args:
            config: Loop detection configuration
            on_warn: Callback when warning is issued (first detection)
            on_block: Callback when blocking (subsequent detection)
            on_escalate: Callback when escalating (after multiple resets)
        """
        self.config = config or LoopDetectorConfig()
        self.detector = LoopDetector.get_instance(self.config)
        self.renderer = PatternRenderer()

        # Callbacks
        self.on_warn = on_warn
        self.on_block = on_block
        self.on_escalate = on_escalate

        # State
        self.captured_state: Optional[CapturedState] = None
        self.pending_recovery_context: Optional[str] = None
        self.recovery_active: bool = False
        self._override_until: float = 0  # Timestamp until override is active

    @property
    def enabled(self) -> bool:
        return self.detector.enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self.detector.enabled = value

    @property
    def reset_count(self) -> int:
        return self.detector.reset_count

    def check_and_handle(
        self,
        session_id: str = "",
        packet_id: str = "",
        goal: str = "",
        last_output: str = ""
    ) -> Optional[AggregatedResult]:
        """
        Check for loops and handle according to escalation ladder.

        Returns:
            AggregatedResult if loop detected and action needed, None otherwise
        """
        if not self.enabled:
            return None

        # Check if override is active
        if time.time() < self._override_until:
            logger.debug("Loop detection override active, skipping check")
            return None

        result = self.detector.check()

        if not result.is_loop:
            # No loop - if we were in recovery, mark success
            if self.recovery_active:
                self._mark_recovery_success()
            return None

        # Loop detected
        logger.warning(
            f"Loop detected: {result.pattern} "
            f"(confidence: {result.confidence:.2f}, reset_count: {result.reset_count})"
        )

        if result.should_warn:
            # First occurrence - warn but don't block
            self.detector.mark_warned()

            if self.on_warn:
                self.on_warn(result)

            # Generate warning notification
            notification = self.renderer.get_notification(result, self.reset_count)
            logger.info(f"Loop warning issued: {notification.get('status_line', '')}")

            return result

        else:
            # Subsequent occurrence - capture state and prepare reset
            self._capture_state(session_id, packet_id, goal, last_output, result)
            self._prepare_recovery_context(result)
            self.detector.trigger_reset()
            self.recovery_active = True

            if self.on_block:
                self.on_block(result)

            # Check for escalation
            if self.reset_count >= 3 and self.on_escalate:
                self.on_escalate(self.reset_count)

            return result

    def _capture_state(
        self,
        session_id: str,
        packet_id: str,
        goal: str,
        last_output: str,
        result: AggregatedResult
    ) -> None:
        """Capture current state before reset."""
        import hashlib

        pattern_hash = hashlib.md5(result.pattern.encode()).hexdigest()[:16]

        metadata = LoopMetadata(
            detected_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            loop_type=result.primary_category.value,
            pattern=result.pattern,
            pattern_hash=pattern_hash,
            reset_count=self.reset_count + 1,  # Will be incremented
            confidence=result.confidence,
            triggered_by=result.triggered_by
        )

        # Add to previous attempts if we have prior captured state
        if self.captured_state and self.captured_state.loop_metadata:
            prev = self.captured_state.loop_metadata
            metadata.previous_attempts = prev.previous_attempts.copy()
            metadata.previous_attempts.append({
                "approach_summary": f"Reset #{prev.reset_count}: {prev.pattern[:100]}",
                "failed_at": prev.detected_at
            })

        self.captured_state = CapturedState(
            session_id=session_id,
            packet_id=packet_id,
            goal=goal,
            last_output=last_output[:500] if last_output else "",  # Truncate
            tool_history=[],  # Could be populated from detector
            error_history=[],
            loop_metadata=metadata
        )

        logger.info(f"State captured before reset #{metadata.reset_count}")

    def _prepare_recovery_context(self, result: AggregatedResult) -> None:
        """Prepare recovery context for injection into next prompt."""
        self.pending_recovery_context = self.renderer.render(
            result,
            format="model_context",
            reset_count=self.reset_count
        )

        if self.captured_state:
            self.captured_state.loop_metadata.recovery_context = self.pending_recovery_context

    def get_recovery_context(self) -> Optional[str]:
        """
        Get pending recovery context for prompt injection.
        Call this when building the next prompt after a reset.
        """
        context = self.pending_recovery_context
        # Don't clear yet - might be needed for retry
        return context

    def clear_recovery_context(self) -> None:
        """Clear the pending recovery context after successful injection."""
        self.pending_recovery_context = None

    def _mark_recovery_success(self) -> None:
        """Mark that recovery was successful (loop broken)."""
        logger.info(f"Loop recovery successful after reset #{self.reset_count}")
        self.recovery_active = False
        self.clear_recovery_context()
        # Reset detectors but keep reset_count for future reference
        self.detector.reset_detectors()

    def override_detection(self, duration_seconds: float = 300) -> None:
        """
        Temporarily disable loop detection (user override).

        Args:
            duration_seconds: How long to disable (default 5 minutes)
        """
        self._override_until = time.time() + duration_seconds
        logger.warning(f"Loop detection overridden for {duration_seconds}s")

    def cancel_override(self) -> None:
        """Cancel any active override."""
        self._override_until = 0
        logger.info("Loop detection override cancelled")

    def get_notification(self, result: AggregatedResult) -> Dict[str, Any]:
        """Get notification data for user display."""
        return self.renderer.get_notification(result, self.reset_count)

    def should_require_user_intervention(self) -> bool:
        """Check if we've hit the escalation threshold requiring user input."""
        return self.reset_count >= 3

    def get_escalation_message(self) -> str:
        """Get message for user intervention request."""
        if self.reset_count < 3:
            return ""

        attempts = []
        if self.captured_state and self.captured_state.loop_metadata.previous_attempts:
            attempts = self.captured_state.loop_metadata.previous_attempts

        attempts_str = "\n".join(
            f"  {i+1}. {a.get('approach_summary', 'Unknown')}"
            for i, a in enumerate(attempts[-3:])  # Last 3 attempts
        )

        return f"""I've detected a loop {self.reset_count} times and keep getting stuck.

Here's what I've tried:
{attempts_str if attempts_str else "  (No recorded attempts)"}

I need your guidance to proceed. What would you like me to do differently?"""

    # =========================================================================
    # Recording Methods (delegate to detector)
    # =========================================================================

    def record_tool_call(self, tool: str, args: Dict[str, Any], result: str = "") -> None:
        """Record a tool call for loop detection."""
        self.detector.record_tool_call(tool, args, result)

    def record_output(self, output: str) -> None:
        """Record model output for loop detection."""
        self.detector.record_output(output)

    def record_state(
        self,
        goal: str = "",
        modified_files: Optional[set] = None,
        state_snapshot: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record state for loop detection."""
        self.detector.record_state(goal, modified_files, state_snapshot)

    def record_error(
        self,
        error_type: str,
        error_message: str,
        attempted_fix: str = "",
        was_success: bool = False
    ) -> None:
        """Record an error for loop detection."""
        self.detector.record_error(error_type, error_message, attempted_fix, was_success)

    def add_tokens(self, tokens: str) -> Optional[Any]:
        """Add streaming tokens and check for patterns."""
        return self.detector.add_tokens(tokens)


# =============================================================================
# Integration Helpers
# =============================================================================

def get_recovery_manager() -> LoopRecoveryManager:
    """Get or create the global recovery manager instance."""
    global _recovery_manager
    if '_recovery_manager' not in globals() or _recovery_manager is None:
        _recovery_manager = LoopRecoveryManager()
    return _recovery_manager


def inject_recovery_context_if_needed(prompt: str) -> str:
    """
    Inject recovery context into a prompt if there's a pending reset.

    Usage in prompt_builder:
        prompt = build_prompt(...)
        prompt = inject_recovery_context_if_needed(prompt)
    """
    manager = get_recovery_manager()
    context = manager.get_recovery_context()

    if not context:
        return prompt

    # Inject at the start of the prompt (after system message typically)
    # The context is wrapped in <loop-recovery> tags for easy identification
    injected = f"{context}\n\n{prompt}"

    logger.info("Injected loop recovery context into prompt")
    return injected


def build_loop_detection_config_from_constants(constants: Dict[str, Any]) -> LoopDetectorConfig:
    """
    Build a LoopDetectorConfig from gaia_constants.json values.

    Expected constants keys:
        LOOP_DETECTION_ENABLED: bool
        LOOP_DETECTION_TOOL_THRESHOLD: int
        LOOP_DETECTION_OUTPUT_THRESHOLD: float
        LOOP_DETECTION_WINDOW_SIZE: int
        LOOP_DETECTION_WARN_FIRST: bool
        ... etc
    """
    return LoopDetectorConfig(
        enabled=constants.get("LOOP_DETECTION_ENABLED", True),
        tool_exact_match_threshold=constants.get("LOOP_DETECTION_TOOL_THRESHOLD", 3),
        output_verbatim_threshold=constants.get("LOOP_DETECTION_OUTPUT_THRESHOLD", 0.95),
        window_size=constants.get("LOOP_DETECTION_WINDOW_SIZE", 10),
        warn_before_block=constants.get("LOOP_DETECTION_WARN_FIRST", True),
        single_high_confidence=constants.get("LOOP_DETECTION_HIGH_CONFIDENCE", 0.9),
        multiple_medium_confidence=constants.get("LOOP_DETECTION_MEDIUM_CONFIDENCE", 0.7),
        weighted_combination_threshold=constants.get("LOOP_DETECTION_WEIGHTED_THRESHOLD", 0.6)
    )


# =============================================================================
# Interrupt for Streaming Integration
# =============================================================================

@dataclass
class LoopInterrupt:
    """
    Interrupt signal for streaming loop detection.
    Compatible with existing Interrupt class from stream_observer.
    """
    level: str  # "OK" | "CAUTION" | "BLOCK"
    reason: str
    suggestion: str = ""
    loop_result: Optional[AggregatedResult] = None

    @classmethod
    def from_detection(cls, result: AggregatedResult, is_warn: bool = True) -> LoopInterrupt:
        """Create an interrupt from a detection result."""
        if not result.is_loop:
            return cls(level="OK", reason="No loop detected")

        level = "CAUTION" if is_warn else "BLOCK"
        renderer = PatternRenderer()

        return cls(
            level=level,
            reason=renderer.render(result, format="brief"),
            suggestion=renderer.render(result, format="summary"),
            loop_result=result
        )


class LoopDetectorObserver:
    """
    Observer adapter for integration with ExternalVoice streaming.

    Monitors token stream for loop patterns and can interrupt generation.
    Also detects think-tag-only output: when the model generates large amounts
    of <think>/<thinking> content without producing any visible text, the
    observer issues a BLOCK interrupt to stop wasting GPU time.
    """

    # Think-tag detection regex — matches open/close tags for think, thinking,
    # reflection, reasoning, internal, scratchpad, planning, analysis
    _THINK_OPEN_RE = re.compile(
        r'<(think|thinking|reflection|reasoning|internal|scratchpad|planning|analysis)>',
        re.IGNORECASE
    )
    _THINK_CLOSE_RE = re.compile(
        r'</(think|thinking|reflection|reasoning|internal|scratchpad|planning|analysis)>',
        re.IGNORECASE
    )

    def __init__(self, manager: Optional[LoopRecoveryManager] = None,
                 think_tag_char_threshold: int = 500,
                 think_tag_ratio_threshold: float = 0.90):
        self.manager = manager or get_recovery_manager()
        self._token_buffer = ""
        self._last_check_len = 0
        self._check_interval = 100  # Check every N characters

        # Think-tag circuit breaker config
        self._think_tag_char_threshold = think_tag_char_threshold
        self._think_tag_ratio_threshold = think_tag_ratio_threshold
        self._think_tag_triggered = False

        # Phrase loop warn-first tracking
        self._phrase_warned = False

    def _check_think_tag_ratio(self) -> Optional[LoopInterrupt]:
        """
        Check if the buffer is predominantly think-tag content with
        no visible user-facing text.

        Returns a BLOCK LoopInterrupt if the model is stuck in think tags,
        None otherwise.
        """
        buf = self._token_buffer
        buf_len = len(buf)

        # Don't check until we have enough content
        if buf_len < self._think_tag_char_threshold:
            return None

        # Already triggered — don't re-fire
        if self._think_tag_triggered:
            return None

        # Calculate how many characters are inside think-like tags.
        # Walk through the buffer tracking open/close tag state.
        visible_chars = 0
        think_chars = 0
        depth = 0  # nesting depth inside think tags
        i = 0
        while i < buf_len:
            # Check for opening tag
            open_match = self._THINK_OPEN_RE.match(buf, i)
            if open_match:
                depth += 1
                i = open_match.end()
                continue

            # Check for closing tag
            close_match = self._THINK_CLOSE_RE.match(buf, i)
            if close_match:
                depth = max(0, depth - 1)
                i = close_match.end()
                continue

            # Regular character
            if depth > 0:
                think_chars += 1
            else:
                # Only count non-whitespace as "visible"
                if not buf[i].isspace():
                    visible_chars += 1
            i += 1

        total_content = think_chars + visible_chars
        if total_content == 0:
            return None

        think_ratio = think_chars / total_content

        if think_ratio >= self._think_tag_ratio_threshold and visible_chars < 20:
            # Model is stuck in think tags — interrupt
            self._think_tag_triggered = True
            logger.warning(
                "Think-tag circuit breaker: %d chars in think tags, "
                "%d visible chars (ratio: %.2f). Interrupting.",
                think_chars, visible_chars, think_ratio
            )
            return LoopInterrupt(
                level="BLOCK",
                reason=(
                    f"Think-tag loop: {think_chars} chars of internal reasoning "
                    f"with only {visible_chars} chars of visible output"
                ),
                suggestion=(
                    "The model is generating only internal reasoning without "
                    "producing a visible response. Retrying with explicit "
                    "no-thinking instruction."
                ),
            )

        return None

    def on_token(self, token: str) -> Optional[LoopInterrupt]:
        """
        Process a token and check for loop patterns.

        Returns LoopInterrupt if a loop is detected, None otherwise.
        """
        self._token_buffer += token

        # Only check periodically to avoid overhead
        if len(self._token_buffer) - self._last_check_len < self._check_interval:
            return None

        self._last_check_len = len(self._token_buffer)

        # --- Think-tag circuit breaker (checked before general loop detection) ---
        think_interrupt = self._check_think_tag_ratio()
        if think_interrupt:
            return think_interrupt

        # Check for token-level patterns
        detection = self.manager.add_tokens(token)

        if detection and detection.triggered:
            # Phrase loops get warn-first treatment (poetry, lyrics, lists
            # have legitimate repetition); other token loops block immediately
            is_phrase = detection.category == LoopCategory.PHRASE_LOOP
            warn_first = is_phrase and not self._phrase_warned

            result = AggregatedResult(
                is_loop=True,
                confidence=detection.confidence,
                primary_category=detection.category,
                pattern=detection.pattern,
                triggered_by=["token"],
                should_warn=warn_first
            )

            if warn_first:
                self._phrase_warned = True
                return LoopInterrupt.from_detection(result, is_warn=True)

            return LoopInterrupt.from_detection(result, is_warn=False)

        return None

    def reset(self) -> None:
        """Reset the observer state."""
        self._token_buffer = ""
        self._last_check_len = 0
        self._think_tag_triggered = False
        self._phrase_warned = False


# Global instance
_recovery_manager: Optional[LoopRecoveryManager] = None
