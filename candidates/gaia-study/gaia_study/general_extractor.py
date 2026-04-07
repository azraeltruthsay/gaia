"""
General Extractor — Classify text into memory types for enriched indexing.

Adapted from MemPalace (github.com/milla-jovovich/mempalace).

Classifies text segments into 5 types:
  1. DECISIONS    — "we went with X because Y", architectural choices
  2. PREFERENCES  — "always use X", "never do Y", conventions
  3. MILESTONES   — breakthroughs, things that finally worked, deployments
  4. PROBLEMS     — what broke, what fixed it, root causes
  5. TECHNICAL    — architecture, implementation details, system state

No LLM required. Pure keyword/pattern heuristics.
No external dependencies.

Usage:
    from gaia_study.general_extractor import extract_memories

    chunks = extract_memories(text)
    # [{"content": "...", "memory_type": "decision", "chunk_index": 0}, ...]
"""

import re
from typing import Dict, List, Tuple


# =============================================================================
# MARKER SETS — One per memory type
# =============================================================================

DECISION_MARKERS = [
    r"\blet'?s (use|go with|try|pick|choose|switch to)\b",
    r"\bwe (should|decided|chose|went with|picked|settled on)\b",
    r"\bi'?m going (to|with)\b",
    r"\bbetter (to|than|approach|option|choice)\b",
    r"\binstead of\b",
    r"\brather than\b",
    r"\bthe reason (is|was|being)\b",
    r"\bbecause\b",
    r"\btrade-?off\b",
    r"\bpros and cons\b",
    r"\barchitecture\b",
    r"\bapproach\b",
    r"\bstrategy\b",
    r"\bpattern\b",
    r"\bframework\b",
    r"\binfrastructure\b",
]

PREFERENCE_MARKERS = [
    r"\bi prefer\b",
    r"\balways use\b",
    r"\bnever use\b",
    r"\bdon'?t (ever |like to )?(use|do|mock|stub|import)\b",
    r"\bi like (to|when|how)\b",
    r"\bplease (always|never|don'?t)\b",
    r"\bmy (rule|preference|style|convention) is\b",
    r"\bwe (always|never)\b",
    r"\buse\b.*\binstead of\b",
]

MILESTONE_MARKERS = [
    r"\bit works\b",
    r"\bgot it working\b",
    r"\bfixed\b",
    r"\bsolved\b",
    r"\bbreakthrough\b",
    r"\bfigured (it )?out\b",
    r"\bfinally\b",
    r"\bfirst time\b",
    r"\bdiscovered\b",
    r"\brealized\b",
    r"\bfound (out|that)\b",
    r"\bturns out\b",
    r"\bthe key (is|was|insight)\b",
    r"\bbuilt\b",
    r"\bimplemented\b",
    r"\bshipped\b",
    r"\blaunched\b",
    r"\bdeployed\b",
    r"\breleased\b",
    r"\bprototype\b",
    r"\bversion \d",
    r"\bv\d+\.\d+",
    r"\d+x (compression|faster|slower|better|improvement|reduction)",
    r"\d+% (reduction|improvement|faster|better|smaller)",
]

PROBLEM_MARKERS = [
    r"\b(bug|error|crash|fail|broke|broken|issue|problem)\b",
    r"\bdoesn'?t work\b",
    r"\bnot working\b",
    r"\bkeeps? (failing|crashing|breaking|erroring)\b",
    r"\broot cause\b",
    r"\bthe (problem|issue|bug) (is|was)\b",
    r"\bthe fix (is|was)\b",
    r"\bworkaround\b",
    r"\bthat'?s why\b",
    r"\bfixed (it |the |by )\b",
    r"\bsolution (is|was)\b",
    r"\bresolved\b",
    r"\bpatched\b",
]

TECHNICAL_MARKERS = [
    r"\b(gpu|vram|cpu|memory|ram)\b",
    r"\b(docker|container|kubernetes|pod)\b",
    r"\b(endpoint|api|rest|grpc|jsonrpc)\b",
    r"\b(model|inference|training|qlora|lora|adapter)\b",
    r"\b(pipeline|workflow|orchestrat)\b",
    r"\b(database|sqlite|chromadb|vector)\b",
    r"\b(token|embedding|attention|transformer)\b",
    r"\b(port|socket|http|tcp)\b",
    r"\b(config|environment|variable|parameter)\b",
    r"\b(quantiz|nf4|gguf|bf16|fp16)\b",
]

ALL_MARKERS = {
    "decision": DECISION_MARKERS,
    "preference": PREFERENCE_MARKERS,
    "milestone": MILESTONE_MARKERS,
    "problem": PROBLEM_MARKERS,
    "technical": TECHNICAL_MARKERS,
}


# =============================================================================
# SENTIMENT — for disambiguation
# =============================================================================

POSITIVE_WORDS = {
    "breakthrough", "success", "works", "working", "solved", "fixed",
    "nailed", "perfect", "brilliant", "excellent", "improved",
}

NEGATIVE_WORDS = {
    "bug", "error", "crash", "fail", "failed", "failing", "failure",
    "broken", "broke", "issue", "problem", "wrong", "stuck", "blocked",
    "unable", "impossible", "missing", "worse", "worst", "panic", "mess",
}


def _get_sentiment(text: str) -> str:
    words = set(w.lower() for w in re.findall(r"\b\w+\b", text))
    pos = len(words & POSITIVE_WORDS)
    neg = len(words & NEGATIVE_WORDS)
    if pos > neg:
        return "positive"
    elif neg > pos:
        return "negative"
    return "neutral"


def _has_resolution(text: str) -> bool:
    text_lower = text.lower()
    patterns = [
        r"\bfixed\b", r"\bsolved\b", r"\bresolved\b", r"\bpatched\b",
        r"\bgot it working\b", r"\bit works\b", r"\bfigured (it )?out\b",
        r"\bthe (fix|answer|solution)\b",
    ]
    return any(re.search(p, text_lower) for p in patterns)


def _disambiguate(memory_type: str, text: str, scores: Dict[str, float]) -> str:
    sentiment = _get_sentiment(text)
    if memory_type == "problem" and _has_resolution(text):
        return "milestone"
    if memory_type == "problem" and sentiment == "positive":
        if scores.get("milestone", 0) > 0:
            return "milestone"
    return memory_type


# =============================================================================
# CODE FILTERING
# =============================================================================

_CODE_LINE_PATTERNS = [
    re.compile(r"^\s*[\$#]\s"),
    re.compile(r"^\s*(cd|source|echo|export|pip|npm|git|python|bash|curl|wget|mkdir|rm|cp|mv|ls|cat|grep|find|chmod|sudo|docker)\s"),
    re.compile(r"^\s*```"),
    re.compile(r"^\s*(import|from|def|class|function|const|let|var|return)\s"),
    re.compile(r"^\s*[A-Z_]{2,}="),
    re.compile(r"^\s*(if|for|while|try|except|elif|else:)\b"),
    re.compile(r"^\s*\w+\.\w+\("),
]


def _extract_prose(text: str) -> str:
    lines = text.split("\n")
    prose = []
    in_code = False
    for line in lines:
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        stripped = line.strip()
        if not stripped:
            continue
        if not any(p.match(stripped) for p in _CODE_LINE_PATTERNS):
            prose.append(line)
    result = "\n".join(prose).strip()
    return result if result else text


# =============================================================================
# SCORING
# =============================================================================

def _score_markers(text: str, markers: List[str]) -> Tuple[float, List[str]]:
    text_lower = text.lower()
    score = 0.0
    keywords = []
    for marker in markers:
        matches = re.findall(marker, text_lower)
        if matches:
            score += len(matches)
            keywords.extend(m if isinstance(m, str) else marker for m in matches)
    return score, list(set(keywords))


# =============================================================================
# MAIN EXTRACTION
# =============================================================================

def extract_memories(text: str, min_confidence: float = 0.2) -> List[Dict]:
    """Extract classified memory segments from text.

    Returns list of dicts: {"content": str, "memory_type": str, "chunk_index": int}
    """
    paragraphs = _split_into_segments(text)
    memories = []

    for para in paragraphs:
        if len(para.strip()) < 20:
            continue

        prose = _extract_prose(para)
        scores = {}
        for mem_type, markers in ALL_MARKERS.items():
            score, _ = _score_markers(prose, markers)
            if score > 0:
                scores[mem_type] = score

        if not scores:
            continue

        length_bonus = 2 if len(para) > 500 else (1 if len(para) > 200 else 0)
        max_type = max(scores, key=scores.get)
        max_score = scores[max_type] + length_bonus
        max_type = _disambiguate(max_type, prose, scores)
        confidence = min(1.0, max_score / 5.0)

        if confidence < min_confidence:
            continue

        memories.append({
            "content": para.strip(),
            "memory_type": max_type,
            "chunk_index": len(memories),
        })

    return memories


def _split_into_segments(text: str) -> List[str]:
    lines = text.split("\n")

    turn_patterns = [
        re.compile(r"^>\s"),
        re.compile(r"^(Human|User|Q)\s*:", re.I),
        re.compile(r"^(Assistant|AI|A|Claude|GAIA)\s*:", re.I),
    ]

    turn_count = sum(
        1 for line in lines
        if any(pat.match(line.strip()) for pat in turn_patterns)
    )

    if turn_count >= 3:
        return _split_by_turns(lines, turn_patterns)

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) <= 1 and len(lines) > 20:
        return [
            "\n".join(lines[i:i + 25]).strip()
            for i in range(0, len(lines), 25)
            if "\n".join(lines[i:i + 25]).strip()
        ]
    return paragraphs


def _split_by_turns(lines: List[str], turn_patterns: List[re.Pattern]) -> List[str]:
    segments = []
    current = []
    for line in lines:
        is_turn = any(pat.match(line.strip()) for pat in turn_patterns)
        if is_turn and current:
            segments.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        segments.append("\n".join(current))
    return segments
