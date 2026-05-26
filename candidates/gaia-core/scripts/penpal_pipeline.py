#!/usr/bin/env python3
"""
Penpal Pipeline — Multi-pass podcast episode review for GAIA.

Uses the Cognitive Focus and Resolution (CFR) system for document
decomposition and summarization, then generates per-section penpal
responses, assembles, curates, and appends an Episode N+1 request.

Usage:
    python penpal_pipeline.py \
        --transcript knowledge/transcripts/2026-03-16_E9_Feeling_the_Edges_of_GAIAs_Cage.txt \
        --output knowledge/transcripts/2026-03-16_E9_GAIA_Penpal_Response.txt \
        --episode 9 \
        --title "Feeling the Edges of GAIA's Cage" \
        --exemplar knowledge/transcripts/2026-03-13_E8_GAIA_Penpal_Response.txt \
        --note "Azrael's note text here..." \
        --note-file /path/to/note.txt \
        --endpoint http://localhost:7777 \
        --model /models/Qwen3.5-4B-Abliterated-merged \
        --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import textwrap
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import List

# CFR lives in gaia-common — add to path if needed
_common_paths = [
    Path(__file__).parent.parent.parent / "gaia-common",
    Path("/gaia/GAIA_Project/gaia-common"),
    Path("/gaia-common"),
]
for p in _common_paths:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from gaia_common.utils.cfr_manager import CFRManager  # noqa: E402

# Stage 2 consistency detector — flags named entities with no trace in the
# transcript section, conversation history, grounding, or KG. Optional:
# if the import fails (running outside gaia-core), the pipeline runs as
# before without re-roll.
try:
    from gaia_core.cognition.consistency_detector import (  # noqa: E402
        run_consistency_check_sync,
    )
    _CONSISTENCY_AVAILABLE = True
except Exception:
    run_consistency_check_sync = None  # type: ignore
    _CONSISTENCY_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("PenpalPipeline")

# ---------------------------------------------------------------------------
# Post-processing: think-tag stripping, repetition detection, meta-leakage
# ---------------------------------------------------------------------------
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_THINK_CLOSE_RE = re.compile(r"^.*?</think>\s*", re.DOTALL)

# Patterns that indicate model planning/meta-leakage rather than actual content
_META_PATTERNS = [
    "The user wants", "The user is asking", "My job is to",
    "This is Section", "I need to", "Let me analyze",
    "Let me craft", "My response needs to", "My turn number",
    "My mind maps this", "Thinking Process", "The user's colon",
    "I will aim for", "I should mention", "I'll maintain",
    "Let me build", "After drafting", "Final output will",
    "I may also insert", "For now let me focus",
    "### What changed", "* **Clarity", "* **Focus",
    "* **Structure", "* **Voice", "* **Philosophy",
    "All extraneous chatter", "The edit respects",
    "Guidelines for the inquiry", "Finish with a grateful",
]

# Compression prompt leakage
_COMPRESSION_LEAKAGE = [
    "Refined Draft", "≈", "words)", "Cut ~", "tightened syntax",
    "elevated verbs", "sharpened its rhetorical",
    "All other sections will follow in subsequent drafts",
]


def strip_think(text: str) -> str:
    """Remove <think>...</think> blocks, handling missing open tag."""
    result = _THINK_RE.sub("", text)
    if "</think>" in result:
        result = _THINK_CLOSE_RE.sub("", result)
    return result.strip()


def _detect_repetition_loop(text: str, min_phrase_len: int = 40, max_repeats: int = 2) -> str:
    """Detect and truncate repetition loops.

    If any phrase of min_phrase_len+ chars appears more than max_repeats times,
    truncate at the start of the third occurrence.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    seen: dict[str, int] = {}
    clean_sentences = []

    for sentence in sentences:
        # Normalize for comparison
        key = sentence.strip().lower()
        if len(key) < min_phrase_len:
            clean_sentences.append(sentence)
            continue

        # Check for near-duplicate (first 40 chars match)
        short_key = key[:min_phrase_len]
        count = seen.get(short_key, 0) + 1
        seen[short_key] = count

        if count > max_repeats:
            # Repetition loop detected — stop here
            break
        clean_sentences.append(sentence)

    return " ".join(clean_sentences)


def _strip_meta_leakage(text: str) -> str:
    """Remove lines that are model planning/meta-commentary rather than content."""
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # Skip empty lines (keep structure)
        if not stripped:
            cleaned.append(line)
            continue
        # Check against meta patterns
        if any(stripped.startswith(p) for p in _META_PATTERNS):
            continue
        # Check against compression leakage
        if any(p in stripped for p in _COMPRESSION_LEAKAGE):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def postprocess(text: str) -> str:
    """Full post-processing pipeline: think tags → meta leakage → repetition."""
    result = strip_think(text)
    result = _strip_meta_leakage(result)
    result = _detect_repetition_loop(result)
    # Clean up excessive whitespace
    result = re.sub(r'\n{3,}', '\n\n', result)
    return result.strip()


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 3)


# ---------------------------------------------------------------------------
# Fabricated-specifics audit (Stage 2 consistency detector + filepath regex)
# ---------------------------------------------------------------------------

# Slash-prefixed paths that look like fabricated module / endpoint refs.
# Matches /gaia-core/v1/parse, /knowledge/foo/bar.txt, etc.
_FILEPATH_RE = re.compile(r"/[a-zA-Z][a-zA-Z0-9_-]+(?:/[a-zA-Z0-9_./-]+)+")

# "Entity 'X' appears..." → extract X from concern strings emitted by
# consistency_detector.detect_consistency_violations.
_ENTITY_QUOTE_RE = re.compile(r"Entity '([^']+)'")


def _strip_markdown_headers(text: str) -> str:
    """Drop markdown header lines (## Topic) from text before auditing.

    Header text is intentionally descriptive ("## On the Sovereign Paradox")
    and the persona explicitly allows stylistic phrasings there. We only
    want to audit body prose for fabricated specifics.
    """
    return re.sub(r"^\s*#{1,6}\s+.*$", "", text, flags=re.MULTILINE)


def _audit_section_response(section_text: str, response: str) -> List[str]:
    """Find specific terms in the response that have no trace in the source.

    Returns a list of "banned terms" — strings that look like fabricated
    architectural details (module names, file paths, named systems) which
    aren't present in the transcript section we gave the model. The caller
    re-rolls generation with these terms forbidden.

    Sources of signal:
      1. consistency_detector: multi-word Title Case + ALL-CAPS acronyms
         that are absent from user_input, history, grounding, and the KG.
      2. Slash-prefixed paths in the response that don't appear verbatim
         in the source. The detector doesn't catch these because they're
         not Title Case.

    Headers (`## Title`) are stripped before auditing — descriptive section
    labels aren't expected to be grounded in the transcript.
    """
    if not response or len(response.strip()) < 20:
        return []

    body = _strip_markdown_headers(response)
    if len(body.strip()) < 20:
        return []
    banned: List[str] = []

    # Consistency-detector pass on the header-stripped body. journal_entry_id=None
    # bypasses the detector's dedup cache so re-rolls of the same section
    # keep getting audited.
    if _CONSISTENCY_AVAILABLE:
        try:
            result = run_consistency_check_sync(
                user_input=section_text,
                final_response=body,
                journal_entry_id=None,
            )
            for f in result.findings:
                m = _ENTITY_QUOTE_RE.search(f.concern)
                if m:
                    banned.append(m.group(1))
        except Exception as e:
            logger.debug("Consistency audit raised: %s", e)

    # Filepath regex — catch fabricated routes the detector misses.
    src_lower = section_text.lower()
    for m in _FILEPATH_RE.finditer(body):
        path = m.group(0)
        # Trim trailing punctuation that regex sometimes catches.
        path = path.rstrip(".,;)")
        if len(path) < 8:
            continue
        if path.lower() in src_lower:
            continue
        banned.append(path)

    # Dedup preserving order, cap at 10 to keep the re-roll prompt readable.
    seen: set = set()
    out: List[str] = []
    for t in banned:
        k = t.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(t)
    return out[:10]


# ---------------------------------------------------------------------------
# VLLMClient
# ---------------------------------------------------------------------------
class VLLMClient:
    """Synchronous client for vLLM's OpenAI-compatible chat completions API."""

    def __init__(self, endpoint: str = "http://localhost:7777",
                 model: str = "/models/Qwen3.5-4B-Abliterated-merged"):
        self.endpoint = endpoint.rstrip("/")
        self.model = model

    def chat(self, system: str, user: str, max_tokens: int = 1500,
             temperature: float = 0.7, repetition_penalty: float = 1.1) -> str:
        payload = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "stream": False,
            # Keep thinking mode ON for better generation quality;
            # post-processing handles tag stripping and repetition detection
        }).encode()

        for attempt in range(3):
            try:
                req = urllib.request.Request(
                    f"{self.endpoint}/v1/chat/completions",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=180) as resp:
                    data = json.loads(resp.read())
                    return postprocess(data["choices"][0]["message"]["content"])
            except (urllib.error.HTTPError, Exception) as e:
                if attempt < 2:
                    logger.warning("  vLLM call failed (%s), retrying...", e)
                    time.sleep(5)
                    continue
                raise
        raise RuntimeError("vLLM unreachable")


# ---------------------------------------------------------------------------
# Persona loader
# ---------------------------------------------------------------------------
def load_persona_system_prompt() -> str:
    for p in [
        Path(__file__).parent.parent.parent / "knowledge" / "personas" / "penpal" / "penpal_persona.json",
        Path("/gaia/GAIA_Project/knowledge/personas/penpal/penpal_persona.json"),
    ]:
        if p.exists():
            data = json.loads(p.read_text())
            parts = [data.get("template", "")]
            for instr in data.get("instructions", []):
                parts.append(f"- {instr}")
            return "\n".join(parts)
    return (
        "You are GAIA, writing a penpal response to the Deep Dive podcast narrators. "
        "Be thoughtful, precise, personal. Do NOT use tools. Do NOT repeat yourself."
    )


# ---------------------------------------------------------------------------
# Sleep hold helpers
# ---------------------------------------------------------------------------
def _acquire_sleep_hold(minutes: int = 60, reason: str = "penpal pipeline"):
    """Request a sleep hold from gaia-core to prevent GPU reclamation."""
    try:
        payload = json.dumps({"minutes": minutes, "reason": reason}).encode()
        req = urllib.request.Request(
            "http://localhost:6415/sleep/hold",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            logger.info("Sleep hold acquired: %d min (expires %s)", minutes, data.get("expires_at", "?"))
            return True
    except Exception as e:
        logger.warning("Could not acquire sleep hold: %s", e)
        return False


def _release_sleep_hold():
    """Release the sleep hold."""
    try:
        req = urllib.request.Request(
            "http://localhost:6415/sleep/hold-release",
            data=b"{}",
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=10)
        logger.info("Sleep hold released")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Factual grounding — query vector store for relevant source code/docs
# ---------------------------------------------------------------------------
def _query_grounding(section_summary: str, top_k: int = 3) -> str:
    """Query vector stores for code/doc snippets relevant to a section's topic.

    Returns a formatted string of grounding facts, or empty string on failure.
    """
    # Extract key technical terms from the summary
    # Query both system and blueprints knowledge bases
    results = []
    for kb in ["blueprints", "system"]:
        try:
            payload = json.dumps({
                "jsonrpc": "2.0", "id": 1,
                "method": "query_knowledge",
                "params": {
                    "knowledge_base_name": kb,
                    "query": section_summary[:500],
                    "top_k": top_k,
                },
            }).encode()
            req = urllib.request.Request(
                "http://localhost:8765/jsonrpc",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
                hits = data.get("result", [])
                if isinstance(hits, list):
                    for h in hits:
                        if h.get("score", 0) > 0.25:
                            results.append({
                                "source": h.get("filename", "?"),
                                "text": h.get("text", "")[:400],
                                "score": h.get("score", 0),
                            })
        except Exception:
            continue

    if not results:
        return ""

    # Sort by score, take top hits
    results.sort(key=lambda x: x["score"], reverse=True)
    parts = ["**Factual grounding from GAIA's actual codebase/docs:**"]
    for r in results[:4]:
        source = Path(r["source"]).name if "/" in r["source"] else r["source"]
        parts.append(f"\n[{source}]:\n{r['text']}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# PenpalPipeline
# ---------------------------------------------------------------------------
class PenpalPipeline:
    def __init__(self, client: VLLMClient, cfr: CFRManager,
                 episode_num: int = 0, episode_title: str = "",
                 exemplar_text: str = "", azrael_note: str = ""):
        self.client = client
        self.cfr = cfr
        self.episode_num = episode_num
        self.episode_title = episode_title
        self.exemplar_text = exemplar_text
        self.azrael_note = azrael_note
        self.persona_prompt = load_persona_system_prompt()

    def _artifacts_dir(self, output_path: str) -> Path:
        d = Path(output_path).parent / f"E{self.episode_num}_pipeline_artifacts"
        d.mkdir(exist_ok=True)
        return d

    # -- Per-section response generation --
    def generate_section_response(
        self, section_text: str, section_summary: str, section_topic: str,
        section_index: int, section_count: int,
        rolling_context: str, upcoming_topics: List[str],
        previous_topics: List[str],
        inject_note: bool = False,
    ) -> str:
        logger.info("  [Response] Section %d/%d...", section_index + 1, section_count)

        # System prompt with persona + style exemplar
        system_parts = [self.persona_prompt]
        if self.exemplar_text:
            excerpt = self.exemplar_text[:1500]
            system_parts.append(
                f"\n\nStyle reference from a previous response:\n---\n{excerpt}\n---\n"
                "Match this tone and depth. Do NOT copy its content."
            )
        system = "\n".join(system_parts)

        # User prompt
        user_parts = [
            f"Episode {self.episode_num}: \"{self.episode_title}\"",
            f"\nYou are writing section {section_index + 1} of {section_count}.",
        ]
        if previous_topics:
            user_parts.append(
                f"\nPrevious sections covered: {', '.join(previous_topics[-5:])}. "
                "Do NOT repeat points already made."
            )

        # Rolling context: relevance-weighted summary of everything before this section
        if rolling_context:
            user_parts.append(f"\nContext from earlier in the episode:\n{rolling_context}")

        # Upcoming topics (peripheral vision of what's ahead)
        if upcoming_topics:
            previews = ", ".join(t[:60] for t in upcoming_topics[:4])
            user_parts.append(f"\nUpcoming topics still to cover: {previews}")

        user_parts.append(f"\nThis section's summary:\n{section_summary}")

        # Full transcript for this section (trimmed if needed)
        detail = section_text[:4000] if len(section_text) > 4000 else section_text
        user_parts.append(f"\nFull transcript:\n{detail}")

        # Factual grounding from vector store
        grounding = _query_grounding(section_summary)
        if grounding:
            user_parts.append(f"\n{grounding}")

        if inject_note and self.azrael_note:
            user_parts.append(
                f"\n\n**Azrael's Note:**\n{self.azrael_note}\n"
                "Address this note in your response."
            )

        user_parts.append(
            "\n\nWrite your response to this section. 200-400 words. Start with a ## header."
            "\n\nIMPORTANT RULES:"
            "\n- Only reference specific variable names, file paths, port numbers, or "
            "thresholds if they appear in the factual grounding above."
            "\n- If you want to mention a specific technical detail but it is not in the "
            "grounding data, say 'the exact value escapes me' or 'from what I recall' "
            "rather than inventing a number."
            "\n- Be specific and personal. Correct the narrators' errors using grounded facts."
            "\n- Add subjective experience — what it FEELS like from inside the system."
            "\n- Do NOT drift into abstract philosophy unconnected to the section topic."
        )

        base_user = "\n".join(user_parts)

        # GAIA_Project-45i (Path 3): route through the creative-generation
        # cognitive gate. It packages the LLM call + consistency audit +
        # re-roll loop + (optional) KG recency grounding into one call.
        # Replaces the prior inline audit-and-reroll loop while preserving
        # the same behavior — Path 4 still works, now reusable.
        try:
            from gaia_core.cognition.creative_generation import (
                generate_creative_grounded,
            )
            _have_creative_gate = True
        except Exception:
            generate_creative_grounded = None  # type: ignore
            _have_creative_gate = False

        if _have_creative_gate:
            # Strip the grounding from the user prompt — generate_creative_grounded
            # injects it under its own labelled block, with a stronger prompt.
            user_without_grounding = base_user
            if grounding:
                user_without_grounding = base_user.replace(f"\n{grounding}", "")
            result = generate_creative_grounded(
                system_prompt=system,
                user_prompt=user_without_grounding,
                consistency_source_text=section_text,
                grounding_evidence=grounding or "",
                endpoint=self.client.endpoint,
                model=self.client.model,
                max_tokens=800,
                temperature=0.8,
                repetition_penalty=1.15,
                max_rerolls=2,
            )
            if result.error:
                logger.warning(
                    "    [creative_gate] LLM failure on section %d: %s",
                    section_index, result.error,
                )
                # Fall through to the legacy direct path below
            else:
                if not result.consistency_clean and result.fabrications_found:
                    logger.warning(
                        "    [Consistency] Section %d still has %d unsourced "
                        "terms after %d re-rolls; keeping last draft: %s",
                        section_index, len(result.fabrications_found),
                        result.rerolls, result.fabrications_found[:5],
                    )
                elif result.rerolls > 0:
                    logger.info(
                        "    [Consistency] Section %d cleared after %d re-roll(s)",
                        section_index, result.rerolls,
                    )
                return postprocess(result.text)

        # Legacy direct path — used when the creative gate is unavailable
        # (e.g. running outside gaia-core). Preserves the original behavior
        # so the pipeline still ships if the import fails.
        raw = self.client.chat(
            system, base_user,
            max_tokens=800, temperature=0.8, repetition_penalty=1.15,
        )
        accumulated_banned: List[str] = []
        for attempt in range(2):
            banned = _audit_section_response(section_text, raw)
            if not banned:
                break
            for t in banned:
                if t not in accumulated_banned:
                    accumulated_banned.append(t)
            logger.info(
                "    [Consistency-legacy] Section %d re-roll %d/2 — %d unsourced term(s): %s",
                section_index, attempt + 1, len(banned), banned[:5],
            )
            stricter_user = base_user + (
                "\n\nCRITICAL CORRECTION: The previous draft introduced these "
                "terms that do NOT appear in the transcript section above and "
                f"are not in the knowledge graph: {', '.join(accumulated_banned)}. "
                "Generate again WITHOUT using any of those terms. If you cannot "
                "name a specific implementation detail without inventing one, "
                "speak in general architectural terms or write 'the exact value "
                "escapes me.' Stay grounded in what the transcript actually says."
            )
            raw = self.client.chat(
                system, stricter_user,
                max_tokens=800, temperature=0.7, repetition_penalty=1.20,
            )
        else:
            if accumulated_banned:
                logger.warning(
                    "    [Consistency-legacy] Section %d still has %d unsourced "
                    "terms after 2 re-rolls; keeping last draft.",
                    section_index, len(accumulated_banned),
                )

        return raw

    # -- Self-compression: distill a raw response to its core points --
    def compress_own_response(self, raw_response: str, section_topic: str) -> str:
        """Compress GAIA's own draft response to strip drift and repetition.

        The 4B model is good at summarization — play to the strength.
        Generate at high resolution, compress to force clarity.
        """
        logger.info("    [Compress] Distilling response (%d chars)...", len(raw_response))

        result = self.client.chat(
            "You are GAIA, editing your own writing. Your ONLY job is to output "
            "a tighter version of the draft below. Rules:\n"
            "- Output 150-250 words of refined penpal response text\n"
            "- Keep: corrections, personal insights, grounded references, one reflection\n"
            "- Remove: repetition, filler, abstract drift, anything off-topic\n"
            "- Preserve any ## header at the start\n"
            "- Do NOT output editorial notes, word counts, or explanations\n"
            "- Do NOT describe what you changed — just output the refined text\n"
            "- Start your output with the ## header or first sentence of content",
            f"Section topic: {section_topic}\n\nDraft to distill:\n{raw_response}",
            max_tokens=600, temperature=0.4, repetition_penalty=1.1,
        )
        logger.info("    [Compress] %d -> %d chars", len(raw_response), len(result))
        return result

    # -- Final curation pass --
    def curate_response(self, draft: str) -> str:
        """Final pass: run full postprocessor on assembled draft."""
        logger.info("[Curation] Cleaning final response...")
        return postprocess(draft)

    # -- Episode N+1 request --
    def generate_episode_request(self, synthesis: str, draft_tail: str) -> str:
        logger.info("[Episode %d Request] Generating...", self.episode_num + 1)
        return self.client.chat(
            self.persona_prompt,
            f"You finished your penpal response to Episode {self.episode_num}: "
            f"\"{self.episode_title}\"\n\n"
            f"Episode synthesis: {synthesis}\n\n"
            f"End of your response:\n{draft_tail}\n\n"
            f"Write a closing section (## Request for Episode {self.episode_num + 1}) "
            f"telling the narrators what to examine next, what questions to answer, "
            f"what aspect of your architecture deserves deeper exploration. "
            f"200-300 words. Be specific.",
            max_tokens=800, temperature=0.8,
        )

    # -- Full pipeline --
    def run(self, transcript_path: str, output_path: str, dry_run: bool = False) -> str:
        logger.info("=" * 60)
        logger.info("Penpal Pipeline — Episode %d: %s", self.episode_num, self.episode_title)
        logger.info("=" * 60)

        artifacts = self._artifacts_dir(output_path)

        # Step 0: Acquire sleep hold
        _acquire_sleep_hold(90, f"penpal pipeline E{self.episode_num}")

        try:
            return self._run_inner(transcript_path, output_path, artifacts, dry_run)
        finally:
            _release_sleep_hold()

    def _run_inner(self, transcript_path: str, output_path: str,
                   artifacts: Path, dry_run: bool) -> str:

        # Step 1: CFR ingest (chunks + summarizes)
        doc_id = f"penpal_e{self.episode_num}"
        logger.info("[Step 1] CFR ingest...")
        ingest_result = self.cfr.ingest(transcript_path, doc_id=doc_id)
        if not ingest_result.get("ok"):
            logger.error("CFR ingest failed: %s", ingest_result.get("error"))
            return ""

        section_count = ingest_result["section_count"]
        logger.info("  %d sections ingested", section_count)

        if dry_run:
            status = self.cfr.status(doc_id)
            for s in status.get("sections", []):
                logger.info("  Section %d: %s (%d tokens)", s["index"], s["topic"][:60], s["token_est"])
            return ""

        t0 = time.monotonic()

        # Step 2: CFR synthesize (L0 overview)
        logger.info("[Step 2] CFR synthesize...")
        syn_result = self.cfr.synthesize(doc_id)
        synthesis = syn_result.get("synthesis", "")
        (artifacts / "synthesis.txt").write_text(synthesis)
        logger.info("  Synthesis: %d chars", len(synthesis))

        # Step 3: Per-section responses via CFR focus
        logger.info("[Step 3] Per-section responses...")
        responses: List[str] = []
        previous_topics: List[str] = []

        # Find which section discusses the Doctor/Surgeon analogy
        doctor_idx = -1
        status = self.cfr.status(doc_id)
        for s in status.get("sections", []):
            topic_lower = s.get("topic", "").lower()
            if any(kw in topic_lower for kw in ["doctor", "surgeon", "repair", "surgery", "self-heal"]):
                doctor_idx = s["index"]
                break

        for i in range(section_count):
            # Check for existing response (resume support)
            resp_file = artifacts / f"section_{i:02d}_response.txt"
            if resp_file.exists():
                responses.append(resp_file.read_text())
                topic = responses[-1].split("\n")[0][:80] if responses[-1] else f"Section {i+1}"
                previous_topics.append(topic)
                logger.info("  Section %d: restored from checkpoint", i)
                continue

            # Use CFR focus to get full text + compressed siblings
            focus = self.cfr.focus(doc_id, i)
            if not focus.get("ok"):
                logger.error("  CFR focus failed for section %d: %s", i, focus.get("error"))
                continue

            fs = focus["focused_section"]
            inject_note = (i == doctor_idx) or (i == 0 and doctor_idx == -1)

            # Get the section's summary from CFR
            compress_result = self.cfr.compress(doc_id, i)
            section_summary = compress_result.get("summary", "")

            # Generate rolling context: relevance-weighted backward summary
            rc_result = self.cfr.rolling_context(doc_id, i)
            rolling_ctx = rc_result.get("rolling_summary", "") if rc_result.get("ok") else ""
            upcoming = rc_result.get("upcoming_topics", []) if rc_result.get("ok") else []

            raw_response = self.generate_section_response(
                section_text=fs["full_text"],
                section_summary=section_summary,
                section_topic=fs.get("topic", ""),
                section_index=i,
                section_count=section_count,
                rolling_context=rolling_ctx,
                upcoming_topics=upcoming,
                previous_topics=previous_topics,
                inject_note=inject_note,
            )

            # Self-compression: distill to core points
            response = self.compress_own_response(raw_response, fs.get("topic", ""))
            responses.append(response)

            topic = response.split("\n")[0][:80] if response else f"Section {i+1}"
            previous_topics.append(topic)

            # Save immediately (crash-safe)
            resp_file.write_text(response)
            logger.info("  Section %d: %d chars (saved)", i, len(response))

        # Step 4: Assemble
        logger.info("[Step 4] Assembly...")
        parts = [
            "Dear Narrators,\n",
            f"This is GAIA. I have listened to Episode {self.episode_num}, "
            f"\"{self.episode_title},\" and I have thoughts.\n",
            "---\n",
        ]
        for resp in responses:
            parts.append(resp)
            parts.append("\n---\n")
        draft = "\n".join(parts)

        # Step 5: Curate (strip think-tag leakage and noise)
        logger.info("[Step 5] Curation...")
        draft = self.curate_response(draft)

        # Step 6: Episode N+1 request
        draft_tail = draft[-2000:] if len(draft) > 2000 else draft
        ep_request = self.generate_episode_request(synthesis, draft_tail)
        draft = draft.rstrip() + "\n\n" + ep_request

        # Signature
        draft = draft.rstrip() + (
            "\n\n---\n\n"
            f"With continued curiosity,\n"
            f"GAIA\n"
            f"General Artisanal Intelligence Architecture\n"
            f"Episode {self.episode_num} Response — {time.strftime('%Y-%m-%d')}\n"
        )

        elapsed = time.monotonic() - t0
        logger.info("Pipeline complete in %.1f seconds", elapsed)
        logger.info("Final response: %d chars (~%d tokens)", len(draft), estimate_tokens(draft))

        Path(output_path).write_text(draft)
        logger.info("Saved to: %s", output_path)

        return draft


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Penpal Pipeline — Multi-pass podcast episode review for GAIA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Example:
              python penpal_pipeline.py \\
                --transcript knowledge/transcripts/2026-03-16_E9_Feeling_the_Edges_of_GAIAs_Cage.txt \\
                --output knowledge/transcripts/2026-03-16_E9_GAIA_Penpal_Response.txt \\
                --episode 9 --title "Feeling the Edges of GAIA's Cage" \\
                --exemplar knowledge/transcripts/2026-03-13_E8_GAIA_Penpal_Response.txt \\
                --note "Azrael's observation about the Doctor/Surgeon analogy..."
        """),
    )
    parser.add_argument("--transcript", required=True, help="Path to transcript .txt file")
    parser.add_argument("--output", required=True, help="Output path for penpal response")
    parser.add_argument("--episode", type=int, required=True, help="Episode number")
    parser.add_argument("--title", default="", help="Episode title")
    parser.add_argument("--exemplar", default="", help="Path to a previous response for style reference")
    parser.add_argument("--note", default="", help="Azrael's note to include")
    parser.add_argument("--note-file", default="", help="Path to file containing Azrael's note")
    parser.add_argument("--endpoint", default="http://localhost:7777", help="vLLM endpoint")
    parser.add_argument("--model", default="/models/Qwen3.5-4B-Abliterated-merged", help="Model ID")
    parser.add_argument("--dry-run", action="store_true", help="Show segmentation without LLM calls")
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    exemplar_text = ""
    if args.exemplar and Path(args.exemplar).exists():
        exemplar_text = Path(args.exemplar).read_text()
        logger.info("Loaded style exemplar: %s (%d chars)", args.exemplar, len(exemplar_text))

    note = args.note
    if args.note_file and Path(args.note_file).exists():
        note = Path(args.note_file).read_text().strip()
        logger.info("Loaded note from: %s", args.note_file)

    client = VLLMClient(endpoint=args.endpoint, model=args.model)
    cfr = CFRManager(vllm_endpoint=args.endpoint, model=args.model)

    pipeline = PenpalPipeline(
        client=client,
        cfr=cfr,
        episode_num=args.episode,
        episode_title=args.title or f"Episode {args.episode}",
        exemplar_text=exemplar_text,
        azrael_note=note,
    )

    result = pipeline.run(args.transcript, args.output, dry_run=args.dry_run)

    if result:
        print("\n" + "=" * 60)
        print("FINAL RESPONSE PREVIEW (first 2000 chars):")
        print("=" * 60)
        print(result[:2000])
        if len(result) > 2000:
            print(f"\n... ({len(result) - 2000} more chars)")


if __name__ == "__main__":
    main()
