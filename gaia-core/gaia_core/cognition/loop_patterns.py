"""
Loop Pattern Classification and Description System.

Generates human-readable descriptions of detected loops for:
1. Brief - Status line / notifications (~50 chars)
2. Summary - User-facing display (2-3 sentences)
3. Full - Model re-injection context (detailed with constraints)

Design: 2026-02-04 (see Dev_Notebook/2026-02-04_loop_detection_reset_system.md)
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from gaia_core.cognition.loop_detector import (
    LoopCategory,
    AggregatedResult,
    DetectionResult
)

logger = logging.getLogger("GAIA.LoopPatterns")


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class DescriptionTemplate:
    """Template for loop description at different verbosity levels."""
    brief: str           # ~50 chars for status line
    summary: str         # 2-3 sentences for user
    full: str            # Detailed for model context
    recovery_hints: List[str]
    avoid_patterns: List[str]


@dataclass
class ClassifiedPattern:
    """A classified loop pattern with description."""
    category: LoopCategory
    sub_pattern: str
    confidence: float
    template: DescriptionTemplate
    evidence: Dict[str, Any]


# =============================================================================
# Pattern Classifier
# =============================================================================

class PatternClassifier:
    """
    Classifies detected loops into specific pattern types and generates
    human-readable descriptions.
    """

    def classify(self, result: AggregatedResult) -> ClassifiedPattern:
        """
        Classify the aggregated detection result and generate descriptions.
        """
        category = result.primary_category
        evidence = result.evidence or {}

        # Route to specific classifier based on category
        if category in (LoopCategory.TOOL_REPETITION, LoopCategory.TOOL_PING_PONG,
                        LoopCategory.TOOL_PARAMETER_DRIFT):
            return self._classify_tool_pattern(result)

        elif category in (LoopCategory.OUTPUT_VERBATIM, LoopCategory.OUTPUT_PARAPHRASE,
                          LoopCategory.OUTPUT_STRUCTURAL):
            return self._classify_output_pattern(result)

        elif category in (LoopCategory.STATE_OSCILLATION, LoopCategory.STATE_REGRESSION,
                          LoopCategory.GOAL_DRIFT):
            return self._classify_state_pattern(result)

        elif category in (LoopCategory.ERROR_REPETITION, LoopCategory.ERROR_WHACK_A_MOLE,
                          LoopCategory.FIX_REPETITION):
            return self._classify_error_pattern(result)

        elif category in (LoopCategory.TOKEN_REPETITION, LoopCategory.PHRASE_LOOP,
                          LoopCategory.STRUCTURAL_LOOP):
            return self._classify_token_pattern(result)

        # Default fallback
        return self._classify_generic(result)

    def _classify_tool_pattern(self, result: AggregatedResult) -> ClassifiedPattern:
        """Classify tool-related loop patterns."""
        evidence = result.evidence or {}
        category = result.primary_category

        if category == LoopCategory.TOOL_PING_PONG:
            tools = evidence.get("tools", ["Tool A", "Tool B"])
            tools_str = " ↔ ".join(tools[:2])

            template = DescriptionTemplate(
                brief=f"Alternating {tools_str}",
                summary=(
                    f"You've been alternating between {' and '.join(tools[:2])} "
                    f"without making progress. This back-and-forth pattern suggests "
                    f"the approach isn't working."
                ),
                full=self._build_full_template(
                    title="Tool Ping-Pong",
                    pattern=f"Alternating calls between {' and '.join(tools)}",
                    details=[
                        f"**Sequence**: {' → '.join(tools * 2)}...",
                        f"**Cycle count**: {evidence.get('matches', 'multiple')} alternations"
                    ],
                    what_went_wrong=(
                        "You're switching between tools without the overall state improving. "
                        "Each tool's output is leading you back to the other."
                    ),
                    suggestions=[
                        "Step back and reconsider the overall goal",
                        "These tools together are not leading to progress",
                        "Try a completely different approach"
                    ]
                ),
                recovery_hints=[
                    "Step back and reconsider the overall approach",
                    "These tools together are not leading to progress"
                ],
                avoid_patterns=[
                    f"Avoid the {' → '.join(tools)} → ... pattern"
                ]
            )
            sub_pattern = f"alternating_{tools[0]}_and_{tools[1]}" if len(tools) >= 2 else "alternating"

        else:
            # TOOL_REPETITION or TOOL_PARAMETER_DRIFT
            tool = evidence.get("tool", "Tool")
            args_summary = evidence.get("args_summary", "")
            count = evidence.get("count", 3)

            template = DescriptionTemplate(
                brief=f"{tool}({args_summary[:20]}) called {count}x",
                summary=(
                    f"You called {tool}({args_summary}) {count} times "
                    f"with the same arguments, getting the same result each time. "
                    f"The output isn't changing, so repeating won't help."
                ),
                full=self._build_full_template(
                    title="Tool Repetition",
                    pattern=f"You called `{tool}` {count} times consecutively",
                    details=[
                        f"**Arguments**: `{args_summary}`" if args_summary else "",
                        f"**Result**: Each call returned the same output."
                    ],
                    what_went_wrong=(
                        "The tool output didn't change between calls. "
                        "Repeating the same query won't produce different results."
                    ),
                    suggestions=[
                        "Try a different approach to get the information you need",
                        "If the result was unexpected, reason about why and adjust your strategy",
                        "If you need different information, modify the query/arguments"
                    ]
                ),
                recovery_hints=[
                    "The result will not change if you call this again",
                    "Consider what information you actually need",
                    "Try a different tool or approach"
                ],
                avoid_patterns=[
                    f"Do not call {tool} with these same arguments"
                ]
            )
            sub_pattern = "exact_same_call" if evidence.get("is_exact") else "similar_calls"

        return ClassifiedPattern(
            category=category,
            sub_pattern=sub_pattern,
            confidence=result.confidence,
            template=template,
            evidence=evidence
        )

    def _classify_output_pattern(self, result: AggregatedResult) -> ClassifiedPattern:
        """Classify output similarity patterns."""
        evidence = result.evidence or {}
        category = result.primary_category
        similarity = evidence.get("similarity", evidence.get("average_similarity", 0.95))

        if category == LoopCategory.OUTPUT_VERBATIM:
            template = DescriptionTemplate(
                brief=f"Output {similarity*100:.0f}% identical to previous",
                summary=(
                    f"Your last response was {similarity*100:.0f}% identical to your previous one. "
                    f"You're generating the same content repeatedly without progress."
                ),
                full=self._build_full_template(
                    title="Verbatim Output Repetition",
                    pattern="Generating identical or near-identical responses",
                    details=[
                        f"**Similarity**: {similarity*100:.0f}% to previous output"
                    ],
                    what_went_wrong=(
                        "You're stuck generating the same response. This might indicate "
                        "confusion about the goal or an inability to make progress."
                    ),
                    suggestions=[
                        "Clarify what you're trying to accomplish",
                        "If stuck, ask the user for guidance",
                        "Try a completely different approach"
                    ]
                ),
                recovery_hints=[
                    "You were generating the same response repeatedly",
                    "Clarify your goal before proceeding"
                ],
                avoid_patterns=[
                    "Do not generate the same response again"
                ]
            )
            sub_pattern = "verbatim"

        else:
            # OUTPUT_PARAPHRASE or OUTPUT_STRUCTURAL
            similar_count = evidence.get("similar_count", 2)

            template = DescriptionTemplate(
                brief=f"Output similar to {similar_count} recent responses",
                summary=(
                    f"Your output is very similar to {similar_count} of your recent responses. "
                    f"You may be stuck in a pattern of saying the same thing differently."
                ),
                full=self._build_full_template(
                    title="Output Similarity Pattern",
                    pattern="Generating semantically similar responses",
                    details=[
                        f"**Similar to**: {similar_count} recent outputs",
                        f"**Average similarity**: {similarity*100:.0f}%"
                    ],
                    what_went_wrong=(
                        "You're rephrasing the same content without making actual progress. "
                        "Consider whether you're addressing the actual goal."
                    ),
                    suggestions=[
                        "Check if you're actually making progress toward the goal",
                        "Try a fundamentally different approach",
                        "Ask for clarification if you're unsure what to do"
                    ]
                ),
                recovery_hints=[
                    "You've been saying similar things multiple times",
                    "Try a new approach rather than rephrasing"
                ],
                avoid_patterns=[
                    "Avoid restating what you've already said"
                ]
            )
            sub_pattern = "paraphrase" if category == LoopCategory.OUTPUT_PARAPHRASE else "structural"

        return ClassifiedPattern(
            category=category,
            sub_pattern=sub_pattern,
            confidence=result.confidence,
            template=template,
            evidence=evidence
        )

    def _classify_state_pattern(self, result: AggregatedResult) -> ClassifiedPattern:
        """Classify state oscillation patterns."""
        evidence = result.evidence or {}
        category = result.primary_category

        if category == LoopCategory.GOAL_DRIFT:
            osc_count = evidence.get("oscillation_count", 2)

            template = DescriptionTemplate(
                brief=f"Goal oscillating {osc_count}x",
                summary=(
                    f"Your goal has been flip-flopping back and forth. "
                    f"This suggests uncertainty about what you should be doing."
                ),
                full=self._build_full_template(
                    title="Goal Oscillation",
                    pattern="Switching goals back and forth",
                    details=[
                        f"**Oscillation count**: {osc_count} cycles"
                    ],
                    what_went_wrong=(
                        "You're uncertain about your objective and keep changing direction. "
                        "This leads to no progress on any front."
                    ),
                    suggestions=[
                        "Clarify the primary goal before continuing",
                        "Ask the user which objective to prioritize",
                        "Pick one direction and commit to it"
                    ]
                ),
                recovery_hints=[
                    "Your goals have been changing back and forth",
                    "Clarify what you should be doing before proceeding"
                ],
                avoid_patterns=[
                    "Do not change goals without completing the current one"
                ]
            )
            sub_pattern = "goal_flip_flop"

        else:
            # STATE_OSCILLATION or STATE_REGRESSION
            unique = evidence.get("unique", 2)
            total = evidence.get("total", 5)

            template = DescriptionTemplate(
                brief=f"State repeating ({unique}/{total} unique)",
                summary=(
                    f"You've been cycling through the same states repeatedly. "
                    f"Only {unique} unique states in {total} snapshots indicates a loop."
                ),
                full=self._build_full_template(
                    title="State Oscillation",
                    pattern="Cycling through repeated states",
                    details=[
                        f"**Unique states**: {unique}",
                        f"**Total snapshots**: {total}"
                    ],
                    what_went_wrong=(
                        "You're not making forward progress - the system state keeps "
                        "returning to previous configurations."
                    ),
                    suggestions=[
                        "Identify what's causing the regression",
                        "Make changes that persist rather than get undone",
                        "Consider a different approach entirely"
                    ]
                ),
                recovery_hints=[
                    "You've been revisiting the same states",
                    "Make sure your changes persist"
                ],
                avoid_patterns=[
                    "Avoid actions that revert previous progress"
                ]
            )
            sub_pattern = "state_cycle"

        return ClassifiedPattern(
            category=category,
            sub_pattern=sub_pattern,
            confidence=result.confidence,
            template=template,
            evidence=evidence
        )

    def _classify_error_pattern(self, result: AggregatedResult) -> ClassifiedPattern:
        """Classify error-related loop patterns."""
        evidence = result.evidence or {}
        category = result.primary_category

        if category == LoopCategory.ERROR_WHACK_A_MOLE:
            error_a = evidence.get("error_a", "Error A")
            error_b = evidence.get("error_b", "Error B")
            oscillations = evidence.get("oscillations", 2)

            template = DescriptionTemplate(
                brief=f"Fixing {error_a[:15]} breaks {error_b[:15]}",
                summary=(
                    f"You're caught in a cycle where fixing {error_a} causes {error_b}, "
                    f"and fixing that brings back the first error. "
                    f"These issues are likely related and need a unified fix."
                ),
                full=self._build_full_template(
                    title="Error Whack-a-Mole",
                    pattern="Fixing one error causes another, and vice versa",
                    details=[
                        f"**Errors involved**:",
                        f"  1. {error_a}",
                        f"  2. {error_b}",
                        f"**Oscillations**: {oscillations} cycles"
                    ],
                    what_went_wrong=(
                        "These errors are interconnected. Your fix for one is causing the other "
                        "because they share a common root cause or have conflicting requirements."
                    ),
                    suggestions=[
                        "Identify what these errors have in common",
                        "Look for a solution that addresses both simultaneously",
                        "Consider whether there's a design issue causing the conflict",
                        "Ask the user for guidance on which constraint to prioritize"
                    ]
                ),
                recovery_hints=[
                    "These errors are connected—find the common cause",
                    "A solution must address both issues together"
                ],
                avoid_patterns=[
                    "Do not fix one error without considering the impact on the other",
                    f"Avoid: fix {error_a} → fix {error_b} → repeat"
                ]
            )
            sub_pattern = f"oscillating_{error_a}_and_{error_b}"

        elif category == LoopCategory.FIX_REPETITION:
            error_type = evidence.get("error_type", "Error")
            fix_attempts = evidence.get("fix_attempts", 2)
            fix = evidence.get("fix", "")[:50]

            template = DescriptionTemplate(
                brief=f"Same fix attempted {fix_attempts}x",
                summary=(
                    f"You've tried the same fix {fix_attempts} times for '{error_type}'. "
                    f"Since it didn't work the first time, repeating it won't help."
                ),
                full=self._build_full_template(
                    title="Repeated Fix Attempt",
                    pattern=f"Same fix attempted {fix_attempts} times",
                    details=[
                        f"**Error**: {error_type}",
                        f"**Attempted fix** ({fix_attempts} times): {fix}..."
                    ],
                    what_went_wrong=(
                        "The fix you're applying doesn't address the actual cause of the error. "
                        "Repeating it won't produce different results."
                    ),
                    suggestions=[
                        "Re-read the error message carefully",
                        "Consider what the error actually means",
                        "Try a fundamentally different approach",
                        "Search for similar issues or documentation",
                        "Ask the user for help"
                    ]
                ),
                recovery_hints=[
                    "This fix does not work—try something different",
                    "Re-examine the error message for clues"
                ],
                avoid_patterns=[
                    f"Do not attempt this fix again: {fix}..."
                ]
            )
            sub_pattern = "same_fix"

        else:
            # ERROR_REPETITION
            error_type = evidence.get("error_type", "Error")
            count = evidence.get("count", 3)
            message = evidence.get("message", "")[:50]

            template = DescriptionTemplate(
                brief=f"Error '{error_type}' occurred {count}x",
                summary=(
                    f"The error '{error_type}' has occurred {count} times. "
                    f"Your attempts to fix it aren't working."
                ),
                full=self._build_full_template(
                    title="Recurring Error",
                    pattern=f"Same error type occurring repeatedly",
                    details=[
                        f"**Error type**: {error_type}",
                        f"**Occurrences**: {count}",
                        f"**Message**: {message}..."
                    ],
                    what_went_wrong=(
                        "This error keeps occurring despite your attempts to fix it. "
                        "You need to find the root cause."
                    ),
                    suggestions=[
                        "Analyze the error more carefully",
                        "Consider what's actually causing it",
                        "Try a different debugging approach",
                        "Ask for help if stuck"
                    ]
                ),
                recovery_hints=[
                    f"The error '{error_type}' keeps recurring",
                    "Find the root cause, not just symptoms"
                ],
                avoid_patterns=[
                    "Do not repeat the same fix attempts"
                ]
            )
            sub_pattern = error_type.lower().replace(" ", "_")

        return ClassifiedPattern(
            category=category,
            sub_pattern=sub_pattern,
            confidence=result.confidence,
            template=template,
            evidence=evidence
        )

    def _classify_token_pattern(self, result: AggregatedResult) -> ClassifiedPattern:
        """Classify token-level repetition patterns."""
        evidence = result.evidence or {}
        category = result.primary_category

        if category == LoopCategory.PHRASE_LOOP:
            phrase = evidence.get("phrase", "")[:50]
            count = evidence.get("count", 3)

            template = DescriptionTemplate(
                brief=f"Phrase repeated {count}x",
                summary=(
                    f"You've been repeating the same phrase {count} times. "
                    f"This indicates a generation loop."
                ),
                full=self._build_full_template(
                    title="Phrase Repetition",
                    pattern="Repeating the same phrase in output",
                    details=[
                        f"**Phrase**: \"{phrase}...\"",
                        f"**Repetitions**: {count}"
                    ],
                    what_went_wrong=(
                        "You're stuck in a generation loop, producing the same text repeatedly."
                    ),
                    suggestions=[
                        "Take a different approach to expressing your response",
                        "If you're stuck, ask for clarification",
                        "Consider whether you've already answered the question"
                    ]
                ),
                recovery_hints=[
                    "You were repeating the same phrase",
                    "Try a fresh approach"
                ],
                avoid_patterns=[
                    f"Do not repeat: \"{phrase[:30]}...\""
                ]
            )
            sub_pattern = "phrase"

        elif category == LoopCategory.TOKEN_REPETITION:
            word = evidence.get("word", "word")
            count = evidence.get("count", 4)

            template = DescriptionTemplate(
                brief=f"Word '{word}' repeated {count}x",
                summary=(
                    f"You repeated the word '{word}' {count} times consecutively. "
                    f"This is a token-level generation issue."
                ),
                full=self._build_full_template(
                    title="Token Repetition",
                    pattern="Repeating the same word consecutively",
                    details=[
                        f"**Word**: \"{word}\"",
                        f"**Consecutive occurrences**: {count}"
                    ],
                    what_went_wrong=(
                        "Token-level generation loop detected. The model is stuck repeating tokens."
                    ),
                    suggestions=[
                        "Reset and try again with a fresh approach",
                        "The previous generation was corrupted"
                    ]
                ),
                recovery_hints=[
                    "Token-level loop detected",
                    "Start fresh"
                ],
                avoid_patterns=[
                    "Avoid word-level repetition"
                ]
            )
            sub_pattern = "word"

        else:
            # STRUCTURAL_LOOP
            period = evidence.get("period", 2)
            matches = evidence.get("matches", 4)

            template = DescriptionTemplate(
                brief=f"Structural pattern repeating (period {period})",
                summary=(
                    f"Your output has a repeating structural pattern with period {period}. "
                    f"The same format is being generated over and over."
                ),
                full=self._build_full_template(
                    title="Structural Repetition",
                    pattern="Repeating the same structural pattern",
                    details=[
                        f"**Pattern period**: {period} lines",
                        f"**Matches**: {matches}"
                    ],
                    what_went_wrong=(
                        "You're generating the same structural pattern repeatedly, "
                        "suggesting a formatting loop."
                    ),
                    suggestions=[
                        "Break out of the pattern",
                        "Consider whether the content is actually different",
                        "Try a different output format"
                    ]
                ),
                recovery_hints=[
                    "Structural repetition detected",
                    "Vary your output format"
                ],
                avoid_patterns=[
                    "Avoid repeating the same structural pattern"
                ]
            )
            sub_pattern = "structural"

        return ClassifiedPattern(
            category=category,
            sub_pattern=sub_pattern,
            confidence=result.confidence,
            template=template,
            evidence=evidence
        )

    def _classify_generic(self, result: AggregatedResult) -> ClassifiedPattern:
        """Generic classification fallback."""
        template = DescriptionTemplate(
            brief=f"Loop detected ({result.primary_category.value})",
            summary=(
                f"A loop pattern was detected: {result.pattern}. "
                f"Consider trying a different approach."
            ),
            full=self._build_full_template(
                title="Loop Detected",
                pattern=result.pattern or "Unspecified loop pattern",
                details=[
                    f"**Category**: {result.primary_category.value}",
                    f"**Confidence**: {result.confidence:.0%}"
                ],
                what_went_wrong="A repetitive pattern was detected in your behavior.",
                suggestions=[
                    "Try a different approach",
                    "Ask for clarification if needed",
                    "Consider what's causing the repetition"
                ]
            ),
            recovery_hints=[
                "A loop was detected",
                "Try something different"
            ],
            avoid_patterns=[
                "Avoid repeating the same pattern"
            ]
        )

        return ClassifiedPattern(
            category=result.primary_category,
            sub_pattern="generic",
            confidence=result.confidence,
            template=template,
            evidence=result.evidence or {}
        )

    def _build_full_template(
        self,
        title: str,
        pattern: str,
        details: List[str],
        what_went_wrong: str,
        suggestions: List[str]
    ) -> str:
        """Build a full template for model re-injection."""
        details_str = "\n".join(d for d in details if d)
        suggestions_str = "\n".join(f"{i+1}. {s}" for i, s in enumerate(suggestions))

        return f"""## Loop Detected: {title}

**Pattern**: {pattern}

{details_str}

---

**What went wrong**: {what_went_wrong}

**Recovery suggestions**:
{suggestions_str}"""


# =============================================================================
# Pattern Renderer
# =============================================================================

class PatternRenderer:
    """
    Renders classified patterns at different verbosity levels.
    """

    def __init__(self):
        self.classifier = PatternClassifier()

    def render(
        self,
        result: AggregatedResult,
        format: str = "summary",
        reset_count: int = 0
    ) -> str:
        """
        Render a loop detection result.

        Args:
            result: The aggregated detection result
            format: One of "brief", "summary", "full", "model_context"
            reset_count: Number of resets so far (for escalation context)

        Returns:
            Rendered description string
        """
        if not result.is_loop:
            return ""

        classified = self.classifier.classify(result)
        template = classified.template

        if format == "brief":
            return template.brief
        elif format == "summary":
            return template.summary
        elif format == "full":
            return template.full
        elif format == "model_context":
            return self._render_model_context(classified, reset_count)
        else:
            return template.summary

    def _render_model_context(self, classified: ClassifiedPattern, reset_count: int) -> str:
        """Render full context for model re-injection."""
        template = classified.template
        urgency = self._get_urgency(reset_count)

        avoid_str = "\n".join(f"- {p}" for p in template.avoid_patterns)
        hints_str = "\n".join(f"{i+1}. {h}" for i, h in enumerate(template.recovery_hints))

        context = f"""<loop-recovery reset="{reset_count}" urgency="{urgency}">

{template.full}

---

## Constraints for This Attempt

{avoid_str}

## Suggestions

{hints_str}
"""

        if reset_count >= 2:
            context += f"""
## Multiple Reset Warning

This is reset #{reset_count}. Previous approaches have failed.

You MUST try a fundamentally different approach or ask the user for help.
"""

        context += "\n</loop-recovery>"
        return context

    def _get_urgency(self, reset_count: int) -> str:
        """Get urgency level based on reset count."""
        if reset_count == 0:
            return "info"
        elif reset_count == 1:
            return "warning"
        elif reset_count == 2:
            return "high"
        else:
            return "critical"

    def get_notification(
        self,
        result: AggregatedResult,
        reset_count: int = 0
    ) -> Dict[str, Any]:
        """
        Build a notification structure for user display.

        Returns dict with:
            status_line: Brief string for status bar
            toast: Dict with title, body, severity
            details: Full description for expanded view
            allow_override: Whether user can override
        """
        if not result.is_loop:
            return {}

        classified = self.classifier.classify(result)

        severity = "warning" if result.should_warn else "error"
        if reset_count >= 3:
            severity = "error"

        return {
            "status_line": f"Loop detected: {classified.template.brief}",
            "toast": {
                "title": "Loop Detected" if reset_count == 0 else f"Loop Detected (Reset #{reset_count})",
                "body": classified.template.summary,
                "severity": severity
            },
            "details": classified.template.full,
            "allow_override": reset_count < 4,
            "override_warning": self._get_override_warning(reset_count),
            "category": classified.category.value,
            "confidence": classified.confidence
        }

    def _get_override_warning(self, reset_count: int) -> Optional[str]:
        """Get warning message for override button."""
        if reset_count == 0:
            return None
        elif reset_count == 1:
            return "Continuing may lead to wasted computation."
        elif reset_count == 2:
            return "This is the 3rd loop. Overriding will disable detection for this session."
        else:
            return "Detection has been overridden multiple times. Consider a different approach."
