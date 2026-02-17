"""
Loop Detection System for GAIA Cognitive Pipeline.

Detects when the model enters generation loops and provides signals for
graceful recovery. Uses multiple parallel detectors that vote on loop presence.

Detector Types:
1. ToolCallRepetitionDetector - Same tool/args called repeatedly
2. OutputSimilarityDetector - Nearly identical outputs across turns
3. StateOscillationDetector - Bouncing between states without progress
4. ErrorCycleDetector - Same errors recurring despite fix attempts
5. TokenPatternDetector - Repetitive token patterns during streaming

Design: 2026-02-04 (see Dev_Notebook/2026-02-04_loop_detection_reset_system.md)
"""

from __future__ import annotations
import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import deque

logger = logging.getLogger("GAIA.LoopDetector")


# =============================================================================
# Enums and Data Classes
# =============================================================================

class LoopCategory(Enum):
    """Classification of detected loop types."""
    # Tool-related
    TOOL_REPETITION = "tool_repetition"
    TOOL_PING_PONG = "tool_ping_pong"
    TOOL_PARAMETER_DRIFT = "tool_parameter_drift"

    # Output-related
    OUTPUT_VERBATIM = "output_verbatim"
    OUTPUT_PARAPHRASE = "output_paraphrase"
    OUTPUT_STRUCTURAL = "output_structural"

    # State-related
    STATE_OSCILLATION = "state_oscillation"
    STATE_REGRESSION = "state_regression"
    GOAL_DRIFT = "goal_drift"

    # Error-related
    ERROR_REPETITION = "error_repetition"
    ERROR_WHACK_A_MOLE = "error_whack_a_mole"
    FIX_REPETITION = "fix_repetition"

    # Generation-related
    TOKEN_REPETITION = "token_repetition"
    PHRASE_LOOP = "phrase_loop"
    STRUCTURAL_LOOP = "structural_loop"


@dataclass
class LoopDetectorConfig:
    """Configuration for loop detection thresholds."""
    # Tool detection
    tool_exact_match_threshold: int = 3
    tool_similar_match_threshold: int = 4
    tool_similarity_threshold: float = 0.8

    # Output detection
    output_verbatim_threshold: float = 0.95
    output_paraphrase_threshold: float = 0.85
    output_min_occurrences: int = 2

    # Error detection
    error_same_threshold: int = 3
    error_same_fix_threshold: int = 2

    # Aggregator
    single_high_confidence: float = 0.9
    multiple_medium_confidence: float = 0.7
    weighted_combination_threshold: float = 0.6

    # General
    window_size: int = 10
    enabled: bool = True

    # Behavior: warn first, block on repeat
    warn_before_block: bool = True


@dataclass
class DetectionResult:
    """Result from a single detector."""
    triggered: bool
    confidence: float  # 0.0 - 1.0
    pattern: str  # Human-readable description
    category: LoopCategory
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregatedResult:
    """Combined result from all detectors."""
    is_loop: bool
    confidence: float
    primary_category: LoopCategory
    pattern: str
    triggered_by: List[str]
    evidence: Dict[str, Any] = field(default_factory=dict)
    should_warn: bool = True  # vs block
    reset_count: int = 0


@dataclass
class ToolCallRecord:
    """Record of a single tool call for tracking."""
    tool: str
    args_hash: str
    result_hash: str = ""
    timestamp: float = field(default_factory=time.time)
    args_summary: str = ""  # Human-readable summary


@dataclass
class ErrorRecord:
    """Record of an error for tracking."""
    error_type: str
    error_message: str
    error_hash: str
    context: str = ""
    attempted_fix: str = ""
    timestamp: float = field(default_factory=time.time)


# =============================================================================
# Individual Detectors
# =============================================================================

class ToolCallRepetitionDetector:
    """
    Detects repeated tool calls with same/similar arguments.

    Catches:
    - Exact repetition: Same tool + args called 3+ times
    - Similar args: Same tool, >80% similar args, 4+ times
    - Ping-pong: Alternating tools (A→B→A→B)
    - Same result: Different calls returning identical output
    """

    def __init__(self, config: LoopDetectorConfig):
        self.config = config
        self.history: deque[ToolCallRecord] = deque(maxlen=config.window_size)

    def record(self, tool: str, args: Dict[str, Any], result: str = "") -> None:
        """Record a tool call."""
        args_hash = self._hash_args(args)
        result_hash = hashlib.sha256(result.encode()).hexdigest()[:16] if result else ""
        args_summary = self._summarize_args(args)

        record = ToolCallRecord(
            tool=tool,
            args_hash=args_hash,
            result_hash=result_hash,
            args_summary=args_summary
        )
        self.history.append(record)

    def detect(self) -> DetectionResult:
        """Check for loop patterns in tool call history."""
        if len(self.history) < 2:
            return DetectionResult(
                triggered=False,
                confidence=0.0,
                pattern="",
                category=LoopCategory.TOOL_REPETITION
            )

        # Strategy 1: Exact repetition
        exact_result = self._detect_exact_repetition()
        if exact_result.triggered:
            return exact_result

        # Strategy 2: Ping-pong pattern
        pingpong_result = self._detect_ping_pong()
        if pingpong_result.triggered:
            return pingpong_result

        # Strategy 3: Same result despite different calls
        same_result = self._detect_same_results()
        if same_result.triggered:
            return same_result

        return DetectionResult(
            triggered=False,
            confidence=0.0,
            pattern="",
            category=LoopCategory.TOOL_REPETITION
        )

    def _detect_exact_repetition(self) -> DetectionResult:
        """Detect exact same tool+args called repeatedly."""
        if len(self.history) < self.config.tool_exact_match_threshold:
            return DetectionResult(False, 0.0, "", LoopCategory.TOOL_REPETITION)

        # Count consecutive matches from the end
        recent = list(self.history)
        if not recent:
            return DetectionResult(False, 0.0, "", LoopCategory.TOOL_REPETITION)

        last = recent[-1]
        consecutive = 1

        for i in range(len(recent) - 2, -1, -1):
            if recent[i].tool == last.tool and recent[i].args_hash == last.args_hash:
                consecutive += 1
            else:
                break

        if consecutive >= self.config.tool_exact_match_threshold:
            confidence = min(consecutive / self.config.tool_exact_match_threshold, 1.0)
            return DetectionResult(
                triggered=True,
                confidence=confidence,
                pattern=f"{last.tool}({last.args_summary}) called {consecutive}x consecutively",
                category=LoopCategory.TOOL_REPETITION,
                evidence={
                    "tool": last.tool,
                    "args_summary": last.args_summary,
                    "count": consecutive,
                    "is_exact": True
                }
            )

        return DetectionResult(False, 0.0, "", LoopCategory.TOOL_REPETITION)

    def _detect_ping_pong(self) -> DetectionResult:
        """Detect alternating tool patterns (A→B→A→B)."""
        if len(self.history) < 4:
            return DetectionResult(False, 0.0, "", LoopCategory.TOOL_PING_PONG)

        recent = list(self.history)
        tools = [r.tool for r in recent]

        # Check for period-2 oscillation
        for period in [2, 3]:
            if len(tools) < period * 2:
                continue

            matches = 0
            for i in range(period, len(tools)):
                if tools[i] == tools[i - period]:
                    matches += 1

            ratio = matches / (len(tools) - period)
            if ratio > 0.8 and matches >= period * 2:
                # Found oscillation
                unique_tools = list(dict.fromkeys(tools[-period*2:]))[:period]
                pattern_str = " ↔ ".join(unique_tools)
                return DetectionResult(
                    triggered=True,
                    confidence=ratio,
                    pattern=f"Ping-pong pattern: {pattern_str} ({matches} alternations)",
                    category=LoopCategory.TOOL_PING_PONG,
                    evidence={
                        "tools": unique_tools,
                        "period": period,
                        "matches": matches,
                        "ratio": ratio
                    }
                )

        return DetectionResult(False, 0.0, "", LoopCategory.TOOL_PING_PONG)

    def _detect_same_results(self) -> DetectionResult:
        """Detect different calls returning identical results."""
        if len(self.history) < 3:
            return DetectionResult(False, 0.0, "", LoopCategory.TOOL_REPETITION)

        recent = [r for r in self.history if r.result_hash]
        if len(recent) < 3:
            return DetectionResult(False, 0.0, "", LoopCategory.TOOL_REPETITION)

        # Count result hash occurrences
        result_counts: Dict[str, int] = {}
        for r in recent:
            result_counts[r.result_hash] = result_counts.get(r.result_hash, 0) + 1

        # Find most common result
        max_count = max(result_counts.values())
        if max_count >= 3:
            confidence = min(max_count / 3, 1.0)
            return DetectionResult(
                triggered=True,
                confidence=confidence,
                pattern=f"Same result returned {max_count}x from different calls",
                category=LoopCategory.TOOL_REPETITION,
                evidence={
                    "result_count": max_count,
                    "is_same_result": True
                }
            )

        return DetectionResult(False, 0.0, "", LoopCategory.TOOL_REPETITION)

    def _hash_args(self, args: Dict[str, Any]) -> str:
        """Create a hash of tool arguments."""
        # Sort keys for consistent hashing
        normalized = str(sorted(args.items()))
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def _summarize_args(self, args: Dict[str, Any]) -> str:
        """Create a human-readable summary of args."""
        if not args:
            return ""

        parts = []
        for k, v in list(args.items())[:3]:  # Limit to first 3 args
            v_str = str(v)[:50]  # Truncate long values
            if len(str(v)) > 50:
                v_str += "..."
            parts.append(f"{k}={v_str}")

        summary = ", ".join(parts)
        if len(args) > 3:
            summary += f", ... (+{len(args) - 3} more)"

        return summary

    def reset(self) -> None:
        """Clear detection history."""
        self.history.clear()


class OutputSimilarityDetector:
    """
    Detects nearly identical outputs across turns.

    Uses multi-strategy similarity:
    - Jaccard on word sets (fast, coarse)
    - N-gram similarity (catches phrase repetition)
    - Structural similarity (same shape, different content)
    """

    def __init__(self, config: LoopDetectorConfig):
        self.config = config
        self.outputs: deque[str] = deque(maxlen=config.window_size)
        self._normalized_cache: deque[str] = deque(maxlen=config.window_size)

    def record(self, output: str) -> None:
        """Record an output."""
        normalized = self._normalize(output)
        self.outputs.append(output)
        self._normalized_cache.append(normalized)

    def detect(self) -> DetectionResult:
        """Check for output similarity patterns."""
        if len(self._normalized_cache) < 2:
            return DetectionResult(False, 0.0, "", LoopCategory.OUTPUT_VERBATIM)

        normalized = list(self._normalized_cache)
        current = normalized[-1]

        # Strategy 1: High similarity to immediate predecessor
        prev_sim = self._similarity(current, normalized[-2])
        if prev_sim > self.config.output_verbatim_threshold:
            return DetectionResult(
                triggered=True,
                confidence=prev_sim,
                pattern=f"Output {prev_sim*100:.0f}% similar to previous",
                category=LoopCategory.OUTPUT_VERBATIM,
                evidence={
                    "similarity": prev_sim,
                    "compared_to": "previous"
                }
            )

        # Strategy 2: High similarity to multiple recent outputs
        high_sim_count = sum(
            1 for prev in normalized[:-1]
            if self._similarity(current, prev) > self.config.output_paraphrase_threshold
        )

        if high_sim_count >= self.config.output_min_occurrences:
            avg_sim = sum(
                self._similarity(current, prev) for prev in normalized[:-1]
            ) / len(normalized[:-1])

            return DetectionResult(
                triggered=True,
                confidence=avg_sim,
                pattern=f"Output similar to {high_sim_count} recent outputs (avg {avg_sim*100:.0f}%)",
                category=LoopCategory.OUTPUT_PARAPHRASE,
                evidence={
                    "similar_count": high_sim_count,
                    "average_similarity": avg_sim
                }
            )

        return DetectionResult(False, 0.0, "", LoopCategory.OUTPUT_VERBATIM)

    def _similarity(self, a: str, b: str) -> float:
        """Calculate similarity between two strings."""
        if not a or not b:
            return 0.0

        # Jaccard similarity on words (weight: 0.3)
        words_a = set(a.split())
        words_b = set(b.split())
        if words_a or words_b:
            intersection = len(words_a & words_b)
            union = len(words_a | words_b)
            jaccard = intersection / union if union > 0 else 0.0
        else:
            jaccard = 0.0

        # N-gram similarity (weight: 0.4)
        ngram_sim = self._ngram_similarity(a, b, n=3)

        # Structural similarity (weight: 0.3)
        struct_sim = self._structural_similarity(a, b)

        return jaccard * 0.3 + ngram_sim * 0.4 + struct_sim * 0.3

    def _ngram_similarity(self, a: str, b: str, n: int = 3) -> float:
        """Calculate n-gram similarity."""
        def get_ngrams(text: str, n: int) -> Set[str]:
            return {text[i:i+n] for i in range(len(text) - n + 1)} if len(text) >= n else set()

        ngrams_a = get_ngrams(a, n)
        ngrams_b = get_ngrams(b, n)

        if not ngrams_a and not ngrams_b:
            return 1.0
        if not ngrams_a or not ngrams_b:
            return 0.0

        intersection = len(ngrams_a & ngrams_b)
        return (2 * intersection) / (len(ngrams_a) + len(ngrams_b))

    def _structural_similarity(self, a: str, b: str) -> float:
        """Compare structural shape (punctuation, line patterns)."""
        def structure_of(s: str) -> str:
            # Replace words with W, numbers with N, keep punctuation
            result = re.sub(r'[a-zA-Z]+', 'W', s)
            result = re.sub(r'\d+', 'N', result)
            result = re.sub(r'\s+', ' ', result)
            return result

        struct_a = structure_of(a)
        struct_b = structure_of(b)

        if struct_a == struct_b:
            return 1.0

        # Partial match via Jaccard on structure tokens
        tokens_a = set(struct_a.split())
        tokens_b = set(struct_b.split())
        if not tokens_a and not tokens_b:
            return 1.0
        if not tokens_a or not tokens_b:
            return 0.0

        intersection = len(tokens_a & tokens_b)
        union = len(tokens_a | tokens_b)
        return intersection / union if union > 0 else 0.0

    def _normalize(self, text: str) -> str:
        """Normalize text for comparison."""
        if not text:
            return ""

        result = text
        # Remove timestamps
        result = re.sub(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', '<TS>', result)
        # Remove UUIDs
        result = re.sub(r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}', '<UUID>', result, flags=re.I)
        # Remove line numbers
        result = re.sub(r'^\s*\d+\s*[│|]', '<LN>', result, flags=re.M)
        # Normalize whitespace
        result = re.sub(r'\s+', ' ', result)

        return result.strip().lower()

    def reset(self) -> None:
        """Clear detection history."""
        self.outputs.clear()
        self._normalized_cache.clear()


class StateOscillationDetector:
    """
    Detects oscillating states without progress.

    Tracks:
    - Goal changes
    - File modification patterns
    - Todo status cycling
    """

    def __init__(self, config: LoopDetectorConfig):
        self.config = config
        self.state_hashes: deque[str] = deque(maxlen=config.window_size)
        self.goals: deque[str] = deque(maxlen=config.window_size)
        self.modified_files: deque[Set[str]] = deque(maxlen=config.window_size)

    def record(self, goal: str = "", modified_files: Optional[Set[str]] = None,
               state_snapshot: Optional[Dict[str, Any]] = None) -> None:
        """Record a state snapshot."""
        if goal:
            self.goals.append(goal)

        if modified_files is not None:
            self.modified_files.append(modified_files)

        if state_snapshot:
            state_hash = hashlib.sha256(str(sorted(state_snapshot.items())).encode()).hexdigest()[:16]
            self.state_hashes.append(state_hash)

    def detect(self) -> DetectionResult:
        """Check for state oscillation patterns."""
        results = []

        # Check state hash repetition
        if len(self.state_hashes) >= 4:
            unique = len(set(self.state_hashes))
            total = len(self.state_hashes)
            repetition_ratio = 1 - (unique / total)

            if repetition_ratio > 0.5:
                results.append(DetectionResult(
                    triggered=True,
                    confidence=repetition_ratio,
                    pattern=f"State repeating: only {unique} unique states in {total} snapshots",
                    category=LoopCategory.STATE_OSCILLATION,
                    evidence={"unique": unique, "total": total}
                ))

        # Check goal oscillation
        if len(self.goals) >= 4:
            goal_result = self._detect_goal_oscillation()
            if goal_result.triggered:
                results.append(goal_result)

        # Return highest confidence result
        if results:
            return max(results, key=lambda r: r.confidence)

        return DetectionResult(False, 0.0, "", LoopCategory.STATE_OSCILLATION)

    def _detect_goal_oscillation(self) -> DetectionResult:
        """Detect goal flip-flopping."""
        goals = list(self.goals)

        # Look for A→B→A pattern
        for i in range(2, len(goals)):
            if goals[i] == goals[i-2] and goals[i] != goals[i-1]:
                # Found oscillation, count how long it continues
                osc_count = 1
                for j in range(i + 2, len(goals), 2):
                    if j < len(goals) and goals[j] == goals[i]:
                        osc_count += 1
                    else:
                        break

                if osc_count >= 2:
                    return DetectionResult(
                        triggered=True,
                        confidence=min(osc_count / 2, 1.0),
                        pattern=f"Goal oscillating between '{goals[i-1][:30]}' and '{goals[i][:30]}'",
                        category=LoopCategory.GOAL_DRIFT,
                        evidence={"oscillation_count": osc_count}
                    )

        return DetectionResult(False, 0.0, "", LoopCategory.GOAL_DRIFT)

    def reset(self) -> None:
        """Clear detection history."""
        self.state_hashes.clear()
        self.goals.clear()
        self.modified_files.clear()


class ErrorCycleDetector:
    """
    Detects recurring error patterns.

    Catches:
    - Same error repeated
    - Same fix attempted multiple times
    - Whack-a-mole: fixing A causes B, fixing B causes A
    """

    def __init__(self, config: LoopDetectorConfig):
        self.config = config
        self.errors: deque[ErrorRecord] = deque(maxlen=config.window_size)

    def record(self, error_type: str, error_message: str,
               attempted_fix: str = "", was_success: bool = False) -> None:
        """Record an error or success."""
        if was_success:
            # Success clears errors of that type
            self.errors = deque(
                (e for e in self.errors if e.error_type != error_type),
                maxlen=self.config.window_size
            )
            return

        error_hash = hashlib.sha256(f"{error_type}:{error_message}".encode()).hexdigest()[:16]

        record = ErrorRecord(
            error_type=error_type,
            error_message=error_message,
            error_hash=error_hash,
            attempted_fix=attempted_fix
        )
        self.errors.append(record)

    def detect(self) -> DetectionResult:
        """Check for error cycle patterns."""
        if len(self.errors) < 2:
            return DetectionResult(False, 0.0, "", LoopCategory.ERROR_REPETITION)

        results = []

        # Strategy 1: Same exact error repeated
        error_counts: Dict[str, int] = {}
        for e in self.errors:
            error_counts[e.error_hash] = error_counts.get(e.error_hash, 0) + 1

        max_error = max(error_counts.values())
        if max_error >= self.config.error_same_threshold:
            # Find which error
            for e in self.errors:
                if error_counts[e.error_hash] == max_error:
                    results.append(DetectionResult(
                        triggered=True,
                        confidence=min(max_error / self.config.error_same_threshold, 1.0),
                        pattern=f"Error '{e.error_type}' occurred {max_error}x",
                        category=LoopCategory.ERROR_REPETITION,
                        evidence={
                            "error_type": e.error_type,
                            "count": max_error,
                            "message": e.error_message[:100]
                        }
                    ))
                    break

        # Strategy 2: Same fix attempted repeatedly
        fix_result = self._detect_fix_repetition()
        if fix_result.triggered:
            results.append(fix_result)

        # Strategy 3: Whack-a-mole pattern
        wam_result = self._detect_whack_a_mole()
        if wam_result.triggered:
            results.append(wam_result)

        if results:
            return max(results, key=lambda r: r.confidence)

        return DetectionResult(False, 0.0, "", LoopCategory.ERROR_REPETITION)

    def _detect_fix_repetition(self) -> DetectionResult:
        """Detect same fix being attempted multiple times."""
        # Group by (error_hash, attempted_fix)
        fix_attempts: Dict[Tuple[str, str], int] = {}

        for e in self.errors:
            if e.attempted_fix:
                key = (e.error_hash, e.attempted_fix)
                fix_attempts[key] = fix_attempts.get(key, 0) + 1

        if not fix_attempts:
            return DetectionResult(False, 0.0, "", LoopCategory.FIX_REPETITION)

        max_attempts = max(fix_attempts.values())
        if max_attempts >= self.config.error_same_fix_threshold:
            for (error_hash, fix), count in fix_attempts.items():
                if count == max_attempts:
                    # Find the error type
                    error_type = next(
                        (e.error_type for e in self.errors if e.error_hash == error_hash),
                        "Unknown"
                    )
                    return DetectionResult(
                        triggered=True,
                        confidence=min(max_attempts / self.config.error_same_fix_threshold, 1.0),
                        pattern=f"Same fix attempted {max_attempts}x for '{error_type}'",
                        category=LoopCategory.FIX_REPETITION,
                        evidence={
                            "error_type": error_type,
                            "fix_attempts": max_attempts,
                            "fix": fix[:100]
                        }
                    )

        return DetectionResult(False, 0.0, "", LoopCategory.FIX_REPETITION)

    def _detect_whack_a_mole(self) -> DetectionResult:
        """Detect A→B→A error pattern."""
        if len(self.errors) < 4:
            return DetectionResult(False, 0.0, "", LoopCategory.ERROR_WHACK_A_MOLE)

        types = [e.error_type for e in self.errors]

        # Look for A→B→A pattern
        for i in range(2, len(types)):
            if types[i] == types[i-2] and types[i] != types[i-1]:
                # Found potential oscillation
                osc_count = 1
                for j in range(i + 2, len(types), 2):
                    if j < len(types) and types[j] == types[i]:
                        osc_count += 1
                    else:
                        break

                if osc_count >= 2:
                    return DetectionResult(
                        triggered=True,
                        confidence=0.85,
                        pattern=f"Error whack-a-mole: '{types[i-1]}' ↔ '{types[i]}'",
                        category=LoopCategory.ERROR_WHACK_A_MOLE,
                        evidence={
                            "error_a": types[i-1],
                            "error_b": types[i],
                            "oscillations": osc_count
                        }
                    )

        return DetectionResult(False, 0.0, "", LoopCategory.ERROR_WHACK_A_MOLE)

    def reset(self) -> None:
        """Clear detection history."""
        self.errors.clear()


class TokenPatternDetector:
    """
    Detects repetitive patterns during token streaming.

    Catches:
    - Exact phrase repetition ("I'll help. I'll help. I'll help.")
    - Structural repetition (same line patterns)
    - Character-level degeneration ("aaaa", "the the the")
    """

    def __init__(self, config: LoopDetectorConfig):
        self.config = config
        self.buffer: str = ""
        self.buffer_max: int = 2000

    def add_tokens(self, tokens: str) -> Optional[DetectionResult]:
        """
        Add tokens to buffer and check for patterns.
        Returns DetectionResult if loop detected, None otherwise.
        """
        self.buffer += tokens
        if len(self.buffer) > self.buffer_max:
            self.buffer = self.buffer[-self.buffer_max:]

        # Check for patterns
        result = self.detect()
        if result.triggered:
            return result

        return None

    def detect(self) -> DetectionResult:
        """Check for repetitive patterns in buffer."""
        if len(self.buffer) < 50:
            return DetectionResult(False, 0.0, "", LoopCategory.TOKEN_REPETITION)

        # Strategy 1: Exact phrase repetition
        phrase_result = self._detect_phrase_repetition()
        if phrase_result.triggered:
            return phrase_result

        # Strategy 2: Word-level repetition ("the the the")
        word_result = self._detect_word_repetition()
        if word_result.triggered:
            return word_result

        # Strategy 3: Structural repetition
        struct_result = self._detect_structural_repetition()
        if struct_result.triggered:
            return struct_result

        return DetectionResult(False, 0.0, "", LoopCategory.TOKEN_REPETITION)

    def _detect_phrase_repetition(self) -> DetectionResult:
        """Detect exact phrase repeated 3+ times."""
        min_phrase = 30
        max_phrase = 150

        # Scan for repeating phrases
        for length in range(max_phrase, min_phrase - 1, -10):  # Step by 10 for efficiency
            for start in range(0, max(1, len(self.buffer) - length * 3), 20):
                phrase = self.buffer[start:start + length]

                # Count non-overlapping occurrences
                count = 0
                pos = 0
                while True:
                    idx = self.buffer.find(phrase, pos)
                    if idx == -1:
                        break
                    count += 1
                    pos = idx + length

                if count >= 3:
                    snippet = phrase[:50] + ("..." if len(phrase) > 50 else "")
                    return DetectionResult(
                        triggered=True,
                        confidence=min(count / 3, 1.0),
                        pattern=f"Phrase repeated {count}x: \"{snippet}\"",
                        category=LoopCategory.PHRASE_LOOP,
                        evidence={
                            "phrase": phrase[:100],
                            "count": count
                        }
                    )

        return DetectionResult(False, 0.0, "", LoopCategory.PHRASE_LOOP)

    def _detect_word_repetition(self) -> DetectionResult:
        """Detect word-level repetition like 'the the the'."""
        words = self.buffer.split()
        if len(words) < 4:
            return DetectionResult(False, 0.0, "", LoopCategory.TOKEN_REPETITION)

        # Check for consecutive repeated words
        consecutive = 1
        max_consecutive = 1
        repeated_word = ""

        for i in range(1, len(words)):
            if words[i].lower() == words[i-1].lower():
                consecutive += 1
                if consecutive > max_consecutive:
                    max_consecutive = consecutive
                    repeated_word = words[i]
            else:
                consecutive = 1

        if max_consecutive >= 4:
            return DetectionResult(
                triggered=True,
                confidence=min(max_consecutive / 4, 1.0),
                pattern=f"Word '{repeated_word}' repeated {max_consecutive}x consecutively",
                category=LoopCategory.TOKEN_REPETITION,
                evidence={
                    "word": repeated_word,
                    "count": max_consecutive
                }
            )

        return DetectionResult(False, 0.0, "", LoopCategory.TOKEN_REPETITION)

    def _detect_structural_repetition(self) -> DetectionResult:
        """Detect repeating line structure patterns."""
        lines = [l.strip() for l in self.buffer.split('\n') if l.strip()]
        if len(lines) < 6:
            return DetectionResult(False, 0.0, "", LoopCategory.STRUCTURAL_LOOP)

        def structure_of(s: str) -> str:
            result = re.sub(r'[a-zA-Z]+', 'W', s)
            result = re.sub(r'\d+', 'N', result)
            return re.sub(r'\s+', ' ', result).strip()

        structures = [structure_of(l) for l in lines]

        # Check for period-N repetition
        for period in [2, 3, 4]:
            if len(structures) < period * 2:
                continue

            matches = 0
            for i in range(period, len(structures)):
                if structures[i] == structures[i - period]:
                    matches += 1

            ratio = matches / (len(structures) - period)
            if ratio > 0.7 and matches >= 4:
                return DetectionResult(
                    triggered=True,
                    confidence=ratio,
                    pattern=f"Structural repetition with period {period} ({matches} matches)",
                    category=LoopCategory.STRUCTURAL_LOOP,
                    evidence={
                        "period": period,
                        "matches": matches,
                        "ratio": ratio
                    }
                )

        return DetectionResult(False, 0.0, "", LoopCategory.STRUCTURAL_LOOP)

    def reset(self) -> None:
        """Clear the buffer."""
        self.buffer = ""


# =============================================================================
# Aggregator
# =============================================================================

class LoopDetectionAggregator:
    """
    Combines signals from all detectors to determine if a loop is occurring.

    Triggering rules:
    1. Any single detector with confidence > 0.9 triggers
    2. Multiple detectors with medium confidence (> 0.7) triggers
    3. Weighted combination exceeding threshold triggers
    """

    # Weights for each detector type
    WEIGHTS = {
        "tool_call": 1.0,
        "output": 0.9,
        "state": 0.85,
        "error": 1.0,
        "token": 0.8
    }

    def __init__(self, config: LoopDetectorConfig):
        self.config = config
        self.tool_detector = ToolCallRepetitionDetector(config)
        self.output_detector = OutputSimilarityDetector(config)
        self.state_detector = StateOscillationDetector(config)
        self.error_detector = ErrorCycleDetector(config)
        self.token_detector = TokenPatternDetector(config)

        # Track reset count for escalation
        self.reset_count = 0
        self.last_detection_time: float = 0
        self.warned_this_session: bool = False

    def evaluate(self) -> AggregatedResult:
        """
        Evaluate all detectors and return aggregated result.
        """
        if not self.config.enabled:
            return AggregatedResult(
                is_loop=False,
                confidence=0.0,
                primary_category=LoopCategory.TOOL_REPETITION,
                pattern="Detection disabled",
                triggered_by=[]
            )

        # Collect results from all detectors
        results: Dict[str, DetectionResult] = {
            "tool_call": self.tool_detector.detect(),
            "output": self.output_detector.detect(),
            "state": self.state_detector.detect(),
            "error": self.error_detector.detect(),
            "token": self.token_detector.detect()
        }

        # Rule 1: Any single high-confidence detection
        for name, result in results.items():
            if result.triggered and result.confidence > self.config.single_high_confidence:
                should_warn = self._should_warn()
                return AggregatedResult(
                    is_loop=True,
                    confidence=result.confidence,
                    primary_category=result.category,
                    pattern=result.pattern,
                    triggered_by=[name],
                    evidence=result.evidence,
                    should_warn=should_warn,
                    reset_count=self.reset_count
                )

        # Rule 2: Multiple medium-confidence detections
        triggered = [(name, r) for name, r in results.items() if r.triggered]
        if len(triggered) >= 2:
            weighted_score = sum(
                r.confidence * self.WEIGHTS.get(name, 1.0)
                for name, r in triggered
            ) / len(triggered)

            if weighted_score > self.config.multiple_medium_confidence:
                # Use highest confidence result as primary
                primary = max(triggered, key=lambda x: x[1].confidence)
                should_warn = self._should_warn()
                return AggregatedResult(
                    is_loop=True,
                    confidence=weighted_score,
                    primary_category=primary[1].category,
                    pattern="; ".join(r.pattern for _, r in triggered),
                    triggered_by=[name for name, _ in triggered],
                    evidence={"multiple_detectors": True},
                    should_warn=should_warn,
                    reset_count=self.reset_count
                )

        # Rule 3: Weighted combination threshold
        total_weighted = sum(
            r.confidence * self.WEIGHTS.get(name, 1.0)
            for name, r in results.items()
        )
        max_possible = sum(self.WEIGHTS.values())
        normalized = total_weighted / max_possible

        if normalized > self.config.weighted_combination_threshold:
            # Find the primary contributor
            primary = max(results.items(), key=lambda x: x[1].confidence)
            should_warn = self._should_warn()
            return AggregatedResult(
                is_loop=True,
                confidence=normalized,
                primary_category=primary[1].category,
                pattern=primary[1].pattern if primary[1].pattern else "Combined detection threshold exceeded",
                triggered_by=list(results.keys()),
                evidence={"weighted_combination": normalized},
                should_warn=should_warn,
                reset_count=self.reset_count
            )

        return AggregatedResult(
            is_loop=False,
            confidence=normalized,
            primary_category=LoopCategory.TOOL_REPETITION,
            pattern="",
            triggered_by=[]
        )

    def _should_warn(self) -> bool:
        """Determine if we should warn (True) or block (False)."""
        if not self.config.warn_before_block:
            return False

        if not self.warned_this_session:
            return True

        return False

    def mark_warned(self) -> None:
        """Mark that we've issued a warning this session."""
        self.warned_this_session = True

    def mark_reset(self) -> None:
        """Mark that a reset has occurred."""
        self.reset_count += 1
        self.last_detection_time = time.time()
        # After reset, next detection should warn again before blocking
        self.warned_this_session = False

    def record_tool_call(self, tool: str, args: Dict[str, Any], result: str = "") -> None:
        """Record a tool call."""
        self.tool_detector.record(tool, args, result)

    def record_output(self, output: str) -> None:
        """Record model output."""
        self.output_detector.record(output)

    def record_state(self, goal: str = "", modified_files: Optional[Set[str]] = None,
                     state_snapshot: Optional[Dict[str, Any]] = None) -> None:
        """Record state snapshot."""
        self.state_detector.record(goal, modified_files, state_snapshot)

    def record_error(self, error_type: str, error_message: str,
                     attempted_fix: str = "", was_success: bool = False) -> None:
        """Record an error or success."""
        self.error_detector.record(error_type, error_message, attempted_fix, was_success)

    def add_tokens(self, tokens: str) -> Optional[DetectionResult]:
        """Add streaming tokens and check for patterns."""
        return self.token_detector.add_tokens(tokens)

    def reset(self) -> None:
        """Reset all detectors."""
        self.tool_detector.reset()
        self.output_detector.reset()
        self.state_detector.reset()
        self.error_detector.reset()
        self.token_detector.reset()
        self.warned_this_session = False
        # Note: reset_count is NOT cleared - it persists for escalation


# =============================================================================
# Main Interface
# =============================================================================

class LoopDetector:
    """
    Main interface for loop detection in GAIA cognitive pipeline.

    Usage:
        detector = LoopDetector.get_instance()

        # Record events
        detector.record_tool_call("Bash", {"command": "git status"}, result)
        detector.record_output(model_output)

        # Check for loops
        result = detector.check()
        if result.is_loop:
            if result.should_warn:
                # Issue warning, continue
                detector.mark_warned()
            else:
                # Block and reset
                detector.trigger_reset()
    """

    _instance: Optional[LoopDetector] = None

    def __init__(self, config: Optional[LoopDetectorConfig] = None):
        self.config = config or LoopDetectorConfig()
        self.aggregator = LoopDetectionAggregator(self.config)
        self._enabled = self.config.enabled

    @classmethod
    def get_instance(cls, config: Optional[LoopDetectorConfig] = None) -> LoopDetector:
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton (for testing)."""
        cls._instance = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value
        self.config.enabled = value

    @property
    def reset_count(self) -> int:
        return self.aggregator.reset_count

    def record_tool_call(self, tool: str, args: Dict[str, Any], result: str = "") -> None:
        """Record a tool call for loop detection."""
        if self._enabled:
            self.aggregator.record_tool_call(tool, args, result)

    def record_output(self, output: str) -> None:
        """Record model output for loop detection."""
        if self._enabled:
            self.aggregator.record_output(output)

    def record_state(self, goal: str = "", modified_files: Optional[Set[str]] = None,
                     state_snapshot: Optional[Dict[str, Any]] = None) -> None:
        """Record state for loop detection."""
        if self._enabled:
            self.aggregator.record_state(goal, modified_files, state_snapshot)

    def record_error(self, error_type: str, error_message: str,
                     attempted_fix: str = "", was_success: bool = False) -> None:
        """Record an error for loop detection."""
        if self._enabled:
            self.aggregator.record_error(error_type, error_message, attempted_fix, was_success)

    def add_tokens(self, tokens: str) -> Optional[DetectionResult]:
        """Add streaming tokens and check for patterns (returns result if loop detected)."""
        if self._enabled:
            return self.aggregator.add_tokens(tokens)
        return None

    def check(self) -> AggregatedResult:
        """Check all detectors for loop patterns."""
        if not self._enabled:
            return AggregatedResult(
                is_loop=False,
                confidence=0.0,
                primary_category=LoopCategory.TOOL_REPETITION,
                pattern="Detection disabled",
                triggered_by=[]
            )

        return self.aggregator.evaluate()

    def mark_warned(self) -> None:
        """Mark that a warning was issued."""
        self.aggregator.mark_warned()
        logger.info("Loop detection: warning issued to user")

    def trigger_reset(self) -> None:
        """Trigger a reset after confirmed loop."""
        self.aggregator.mark_reset()
        logger.warning(f"Loop detection: reset triggered (count: {self.reset_count})")

    def reset_detectors(self) -> None:
        """Reset all detector history (but preserve reset count)."""
        self.aggregator.reset()
        logger.info("Loop detection: detectors reset")

    def full_reset(self) -> None:
        """Full reset including reset count (use sparingly)."""
        self.aggregator.reset()
        self.aggregator.reset_count = 0
        self.aggregator.warned_this_session = False
        logger.info("Loop detection: full reset")
