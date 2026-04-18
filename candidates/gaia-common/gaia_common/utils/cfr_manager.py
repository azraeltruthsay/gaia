"""
Cognitive Focus and Resolution (CFR) — Hierarchical document comprehension.

Gives GAIA the ability to process documents larger than her context window
by decomposing them into a resolution hierarchy. Like a human reading a long
document: read a section, summarize it, move on, and re-expand any section
when deeper focus is needed.

Core operations:
    ingest(file_path)           → chunk + summarize → resolution tree
    focus(doc_id, section)      → full text + compressed siblings
    compress(doc_id, section)   → generate/retrieve summary
    expand(doc_id, section)     → re-load full text (free, always on disk)
    synthesize(doc_id)          → rolling understanding across all sections
    status(doc_id)              → current resolution state

Storage: /shared/gaia_state/cfr/<doc_id>.json per document, _index.json for catalog.
Full text is ALWAYS retained on disk — "compressed" means a summary exists,
not that the original was discarded.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("GAIA.CFR")

# ---------------------------------------------------------------------------
# Think-tag stripping (matches agent_core.py / penpal_pipeline.py)
# ---------------------------------------------------------------------------
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_THINK_CLOSE_RE = re.compile(r"^.*?</think>\s*", re.DOTALL)


def _strip_think(text: str) -> str:
    """Remove <think>...</think> blocks, handling missing open tag."""
    result = _THINK_RE.sub("", text)
    if "</think>" in result:
        result = _THINK_CLOSE_RE.sub("", result)
    return result.strip()


def _estimate_tokens(text: str) -> int:
    """~3.5 chars per token heuristic (matches VLLMRemoteModel)."""
    return max(1, len(text) // 3)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class CFRSection:
    index: int
    start_char: int
    end_char: int
    full_text: str
    state: str = "raw"  # raw | compressed | focused
    summary: str = ""
    topic: str = ""
    token_est: int = 0
    summary_token_est: int = 0

    def __post_init__(self):
        self.token_est = _estimate_tokens(self.full_text)
        if self.summary:
            self.summary_token_est = _estimate_tokens(self.summary)


@dataclass
class CFRSynthesis:
    text: str = ""
    updated_at: str = ""
    sections_covered: List[int] = field(default_factory=list)


@dataclass
class CFRDocument:
    doc_id: str
    source_path: str
    created_at: str
    updated_at: str
    total_chars: int
    total_tokens_est: int
    sections: List[CFRSection]
    synthesis: CFRSynthesis = field(default_factory=CFRSynthesis)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Segmenter — paragraph-boundary-aware chunking
# ---------------------------------------------------------------------------
class _Segmenter:
    """Split text into sections at paragraph boundaries."""

    @staticmethod
    def segment(
        text: str,
        target: int = 3500,
        minimum: int = 1500,
        maximum: int = 5000,
    ) -> List[Dict[str, Any]]:
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        sections: List[Dict[str, Any]] = []
        buf: list[str] = []
        buf_len = 0
        char_offset = 0

        def flush():
            nonlocal buf, buf_len, char_offset
            if not buf:
                return
            joined = "\n\n".join(buf)
            sections.append({
                "index": len(sections),
                "text": joined,
                "start_char": char_offset,
                "end_char": char_offset + len(joined),
            })
            char_offset += len(joined)
            buf = []
            buf_len = 0

        for para in paragraphs:
            plen = len(para)
            if buf_len + plen > maximum and buf:
                flush()
            buf.append(para)
            buf_len += plen
            if buf_len >= target:
                flush()

        # Flush remainder — merge into last section if too small
        if buf:
            if buf_len < minimum and sections:
                last = sections[-1]
                merged = last["text"] + "\n\n" + "\n\n".join(buf)
                sections[-1] = {
                    "index": last["index"],
                    "text": merged,
                    "start_char": last["start_char"],
                    "end_char": last["start_char"] + len(merged),
                }
            else:
                flush()

        return sections


# ---------------------------------------------------------------------------
# vLLM caller (stdlib only)
# ---------------------------------------------------------------------------
class _VLLMCaller:
    """Thin synchronous wrapper for vLLM chat completions."""

    def __init__(self, endpoint: str, model: str):
        self.endpoint = endpoint.rstrip("/")
        self.model = model

    def chat(
        self,
        system: str,
        user: str,
        max_tokens: int = 400,
        temperature: float = 0.3,
        repetition_penalty: float = 1.1,
    ) -> str:
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
            "chat_template_kwargs": {"enable_thinking": False},
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
                    return _strip_think(data["choices"][0]["message"]["content"])
            except urllib.error.HTTPError as e:
                if e.code in (503, 429) and attempt < 2:
                    time.sleep(5)
                    continue
                raise
            except Exception:
                if attempt < 2:
                    time.sleep(5)
                    continue
                raise
        raise RuntimeError("vLLM unreachable after 3 attempts")


# ---------------------------------------------------------------------------
# CFRManager — the main class
# ---------------------------------------------------------------------------
class CFRManager:
    """Cognitive Focus and Resolution — hierarchical document comprehension."""

    # Inside containers, use /shared. On host, fall back to project-local path.
    _CONTAINER_DIR = Path("/shared/gaia_state/cfr")
    _HOST_DIR = Path("/gaia/GAIA_Project/gaia-core/shared/gaia_state/cfr")
    DEFAULT_STATE_DIR = _CONTAINER_DIR if _CONTAINER_DIR.parent.exists() else _HOST_DIR
    INDEX_FILENAME = "_index.json"

    # Chunking defaults
    CHUNK_TARGET = 3500
    CHUNK_MIN = 1500
    CHUNK_MAX = 5000

    # LLM budget
    SUMMARY_SYSTEM = (
        "You are GAIA, summarizing a section of a document. Write a concise "
        "100-200 word factual summary. Identify the main topic, key claims, "
        "and important details. Be precise. Do not editorialize."
    )
    SYNTHESIS_SYSTEM = (
        "You are GAIA, synthesizing your understanding of a document. Given "
        "summaries of all sections, write a cohesive 200-300 word overview "
        "that captures the document's overarching theme, key arguments, and "
        "most important details."
    )

    def __init__(
        self,
        state_dir: Optional[str] = None,
        vllm_endpoint: str = "http://gaia-prime:7777",
        model: str = "/models/Qwen3.5-4B-Abliterated-merged",
    ):
        self.state_dir = Path(state_dir) if state_dir else self.DEFAULT_STATE_DIR
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._vllm = _VLLMCaller(vllm_endpoint, model)

        # Track whether endpoint/model were explicitly provided
        self._explicit_endpoint = vllm_endpoint != "http://gaia-prime:7777"
        self._explicit_model = model != "/models/Qwen3.5-4B-Abliterated-merged"

        # Try loading config from gaia_constants.json
        self._load_config()

    def _load_config(self):
        """Load CFR config from gaia_constants.json if available."""
        try:
            for p in [
                Path("/gaia-common/gaia_common/constants/gaia_constants.json"),
                Path("/gaia/GAIA_Project/gaia-common/gaia_common/constants/gaia_constants.json"),
            ]:
                if p.exists():
                    cfg = json.loads(p.read_text()).get("CFR", {})
                    if cfg.get("state_dir"):
                        candidate_dir = Path(cfg["state_dir"])
                        # Only use config path if parent exists (i.e., we're in a container)
                        if candidate_dir.parent.exists():
                            self.state_dir = candidate_dir
                            self.state_dir.mkdir(parents=True, exist_ok=True)
                    if cfg.get("chunk_target_chars"):
                        self.CHUNK_TARGET = cfg["chunk_target_chars"]
                    if cfg.get("chunk_min_chars"):
                        self.CHUNK_MIN = cfg["chunk_min_chars"]
                    if cfg.get("chunk_max_chars"):
                        self.CHUNK_MAX = cfg["chunk_max_chars"]
                    endpoint_key = cfg.get("engine_endpoint") or cfg.get("vllm_endpoint")
                    if endpoint_key and not self._explicit_endpoint:
                        self._vllm.endpoint = endpoint_key.rstrip("/")
                    if cfg.get("model") and not self._explicit_model:
                        self._vllm.model = cfg["model"]
                    break
        except Exception:
            pass  # Use defaults

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------
    def _tree_path(self, doc_id: str) -> Path:
        return self.state_dir / f"{doc_id}.json"

    def _index_path(self) -> Path:
        return self.state_dir / self.INDEX_FILENAME

    def _save_tree(self, doc: CFRDocument) -> None:
        """Persist a resolution tree to disk."""
        data = {
            "doc_id": doc.doc_id,
            "source_path": doc.source_path,
            "created_at": doc.created_at,
            "updated_at": _now_iso(),
            "total_chars": doc.total_chars,
            "total_tokens_est": doc.total_tokens_est,
            "sections": [
                {
                    "index": s.index,
                    "start_char": s.start_char,
                    "end_char": s.end_char,
                    "full_text": s.full_text,
                    "state": s.state,
                    "summary": s.summary,
                    "topic": s.topic,
                    "token_est": s.token_est,
                    "summary_token_est": s.summary_token_est,
                }
                for s in doc.sections
            ],
            "synthesis": {
                "text": doc.synthesis.text,
                "updated_at": doc.synthesis.updated_at,
                "sections_covered": doc.synthesis.sections_covered,
            },
            "metadata": doc.metadata,
        }
        self._tree_path(doc.doc_id).write_text(json.dumps(data, indent=2))

    def _load_tree(self, doc_id: str) -> Optional[CFRDocument]:
        """Load a resolution tree from disk."""
        path = self._tree_path(doc_id)
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        sections = [
            CFRSection(
                index=s["index"],
                start_char=s["start_char"],
                end_char=s["end_char"],
                full_text=s["full_text"],
                state=s.get("state", "raw"),
                summary=s.get("summary", ""),
                topic=s.get("topic", ""),
            )
            for s in data["sections"]
        ]
        syn_data = data.get("synthesis", {})
        return CFRDocument(
            doc_id=data["doc_id"],
            source_path=data["source_path"],
            created_at=data["created_at"],
            updated_at=data.get("updated_at", ""),
            total_chars=data["total_chars"],
            total_tokens_est=data["total_tokens_est"],
            sections=sections,
            synthesis=CFRSynthesis(
                text=syn_data.get("text", ""),
                updated_at=syn_data.get("updated_at", ""),
                sections_covered=syn_data.get("sections_covered", []),
            ),
            metadata=data.get("metadata", {}),
        )

    def _update_index(self, doc: CFRDocument) -> None:
        """Update the document index."""
        idx_path = self._index_path()
        index = {}
        if idx_path.exists():
            index = json.loads(idx_path.read_text())
        docs = index.setdefault("documents", {})
        docs[doc.doc_id] = {
            "source_path": doc.source_path,
            "created_at": doc.created_at,
            "section_count": len(doc.sections),
            "total_tokens_est": doc.total_tokens_est,
            "synthesis_exists": bool(doc.synthesis.text),
        }
        idx_path.write_text(json.dumps(index, indent=2))

    # -----------------------------------------------------------------------
    # Core operations
    # -----------------------------------------------------------------------
    def ingest(
        self,
        file_path: str,
        doc_id: Optional[str] = None,
        chunk_target: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Ingest a document: chunk it, generate per-section summaries,
        store the resolution tree. Idempotent — skips sections with
        existing summaries on re-ingest.

        Returns: {"ok": True, "doc_id": str, "section_count": int, ...}
        """
        path = Path(file_path)
        if not path.exists():
            return {"ok": False, "error": f"File not found: {file_path}"}

        text = path.read_text()
        if not text.strip():
            return {"ok": False, "error": "File is empty"}

        # Generate doc_id from content hash if not provided
        if not doc_id:
            doc_id = hashlib.sha256(text.encode()).hexdigest()[:12]

        # Check for existing tree (resume support)
        existing = self._load_tree(doc_id)
        if existing and len(existing.sections) > 0:
            # Resume: only generate missing summaries
            logger.info("CFR: Resuming ingest for %s (%d sections exist)",
                        doc_id, len(existing.sections))
            doc = existing
            needs_summary = [s for s in doc.sections if s.state == "raw"]
        else:
            # Fresh ingest
            target = chunk_target or self.CHUNK_TARGET
            raw_sections = _Segmenter.segment(
                text, target=target, minimum=self.CHUNK_MIN, maximum=self.CHUNK_MAX
            )
            sections = [
                CFRSection(
                    index=s["index"],
                    start_char=s["start_char"],
                    end_char=s["end_char"],
                    full_text=s["text"],
                )
                for s in raw_sections
            ]
            doc = CFRDocument(
                doc_id=doc_id,
                source_path=str(path),
                created_at=_now_iso(),
                updated_at=_now_iso(),
                total_chars=len(text),
                total_tokens_est=_estimate_tokens(text),
                sections=sections,
                metadata={"chunk_target": target},
            )
            needs_summary = sections

        # Generate summaries for sections that need them
        for section in needs_summary:
            logger.info("CFR: Summarizing section %d/%d (%d tokens)...",
                        section.index + 1, len(doc.sections), section.token_est)
            summary = self._generate_summary(section.full_text)
            section.summary = summary
            section.summary_token_est = _estimate_tokens(summary)
            section.state = "compressed"

            # Extract topic from first sentence of summary
            first_sentence = summary.split(".")[0] if summary else ""
            section.topic = first_sentence[:100]

            # Save after each section (crash-safe)
            self._save_tree(doc)

        self._update_index(doc)
        logger.info("CFR: Ingest complete for %s — %d sections", doc_id, len(doc.sections))

        return {
            "ok": True,
            "doc_id": doc_id,
            "source_path": str(path),
            "section_count": len(doc.sections),
            "total_tokens_est": doc.total_tokens_est,
            "sections": [
                {"index": s.index, "state": s.state, "topic": s.topic,
                 "token_est": s.token_est, "summary_token_est": s.summary_token_est}
                for s in doc.sections
            ],
        }

    def focus(self, doc_id: str, section_index: int) -> Dict[str, Any]:
        """
        Load a section at full resolution with compressed siblings as context.

        Returns a working-memory payload ready for LLM consumption:
        {
            "focused_section": {"index", "full_text", "topic", "token_est"},
            "context_summaries": [{"index", "summary", "topic"}, ...],
            "synthesis": str or "",
            "total_context_tokens": int
        }
        """
        doc = self._load_tree(doc_id)
        if not doc:
            return {"ok": False, "error": f"Document not found: {doc_id}"}
        if section_index < 0 or section_index >= len(doc.sections):
            return {"ok": False, "error": f"Section index out of range: {section_index}"}

        # Mark states
        for s in doc.sections:
            if s.index == section_index:
                s.state = "focused"
            elif s.state == "focused":
                s.state = "compressed"  # Defocus previously focused sections
        self._save_tree(doc)

        target = doc.sections[section_index]
        siblings = [
            {"index": s.index, "summary": s.summary, "topic": s.topic}
            for s in doc.sections if s.index != section_index and s.summary
        ]

        context_tokens = (
            target.token_est
            + sum(s.summary_token_est for s in doc.sections if s.index != section_index)
            + _estimate_tokens(doc.synthesis.text)
        )

        return {
            "ok": True,
            "doc_id": doc_id,
            "focused_section": {
                "index": target.index,
                "full_text": target.full_text,
                "topic": target.topic,
                "token_est": target.token_est,
            },
            "context_summaries": siblings,
            "synthesis": doc.synthesis.text,
            "total_context_tokens": context_tokens,
        }

    def compress(self, doc_id: str, section_index: int) -> Dict[str, Any]:
        """
        Generate (or retrieve cached) summary for a section.
        Returns: {"ok": True, "summary": str, "token_est": int}
        """
        doc = self._load_tree(doc_id)
        if not doc:
            return {"ok": False, "error": f"Document not found: {doc_id}"}
        if section_index < 0 or section_index >= len(doc.sections):
            return {"ok": False, "error": f"Section index out of range: {section_index}"}

        section = doc.sections[section_index]

        # Generate summary if not already cached
        if not section.summary:
            section.summary = self._generate_summary(section.full_text)
            section.summary_token_est = _estimate_tokens(section.summary)
            section.topic = section.summary.split(".")[0][:100]

        section.state = "compressed"
        self._save_tree(doc)

        return {
            "ok": True,
            "doc_id": doc_id,
            "section_index": section_index,
            "summary": section.summary,
            "topic": section.topic,
            "token_est": section.summary_token_est,
        }

    def expand(self, doc_id: str, section_index: int) -> Dict[str, Any]:
        """
        Re-expand a section to full resolution. Free — no LLM call.
        Returns: {"ok": True, "full_text": str, "token_est": int}
        """
        doc = self._load_tree(doc_id)
        if not doc:
            return {"ok": False, "error": f"Document not found: {doc_id}"}
        if section_index < 0 or section_index >= len(doc.sections):
            return {"ok": False, "error": f"Section index out of range: {section_index}"}

        section = doc.sections[section_index]
        section.state = "focused"
        self._save_tree(doc)

        return {
            "ok": True,
            "doc_id": doc_id,
            "section_index": section_index,
            "full_text": section.full_text,
            "topic": section.topic,
            "token_est": section.token_est,
        }

    def synthesize(self, doc_id: str) -> Dict[str, Any]:
        """
        Generate a rolling synthesis across all section summaries.
        Returns: {"ok": True, "synthesis": str, "sections_covered": [int]}
        """
        doc = self._load_tree(doc_id)
        if not doc:
            return {"ok": False, "error": f"Document not found: {doc_id}"}

        # Collect all summaries
        summaries = []
        covered = []
        for s in doc.sections:
            if s.summary:
                summaries.append(f"Section {s.index + 1} ({s.topic}):\n{s.summary}")
                covered.append(s.index)

        if not summaries:
            return {"ok": False, "error": "No summaries available. Run ingest first."}

        combined = "\n\n".join(summaries)
        total_tokens = _estimate_tokens(combined)

        # If summaries fit in one call, synthesize directly
        if total_tokens < 8000:
            synthesis = self._vllm.chat(
                self.SYNTHESIS_SYSTEM,
                f"Here are summaries of all sections:\n\n{combined}\n\nWrite your synthesis.",
                max_tokens=800,
                temperature=0.3,
            )
        else:
            # Two-pass: synthesize halves, then merge
            mid = len(summaries) // 2
            first_half = "\n\n".join(summaries[:mid])
            second_half = "\n\n".join(summaries[mid:])

            syn1 = self._vllm.chat(
                self.SYNTHESIS_SYSTEM,
                f"Summarize these sections:\n\n{first_half}",
                max_tokens=500,
                temperature=0.3,
            )
            synthesis = self._vllm.chat(
                self.SYNTHESIS_SYSTEM,
                f"Prior synthesis (first half):\n{syn1}\n\nRemaining sections:\n\n{second_half}\n\nWrite a unified synthesis covering all sections.",
                max_tokens=800,
                temperature=0.3,
            )

        doc.synthesis = CFRSynthesis(
            text=synthesis,
            updated_at=_now_iso(),
            sections_covered=covered,
        )
        self._save_tree(doc)
        self._update_index(doc)

        return {
            "ok": True,
            "doc_id": doc_id,
            "synthesis": synthesis,
            "sections_covered": covered,
        }

    def rolling_context(self, doc_id: str, target_section: int) -> Dict[str, Any]:
        """
        Generate a relevance-weighted rolling summary of sections 0..N-1,
        emphasizing details relevant to section N's topic.

        For section 0, returns empty (no prior context).
        For section N, produces a compressed narrative of everything before N,
        weighted toward details that matter for N's topic.

        Also includes brief topic previews of sections N+1 onward.

        Returns: {
            "ok": True,
            "doc_id": str,
            "target_section": int,
            "rolling_summary": str,      # compressed prior context
            "upcoming_topics": [str],     # what's ahead
            "token_est": int
        }
        """
        doc = self._load_tree(doc_id)
        if not doc:
            return {"ok": False, "error": f"Document not found: {doc_id}"}
        if target_section < 0 or target_section >= len(doc.sections):
            return {"ok": False, "error": f"Section index out of range: {target_section}"}

        # Section 0 has no prior context
        if target_section == 0:
            upcoming = [
                s.topic[:80] for s in doc.sections[1:]
                if s.topic
            ]
            return {
                "ok": True,
                "doc_id": doc_id,
                "target_section": 0,
                "rolling_summary": "",
                "upcoming_topics": upcoming,
                "token_est": 0,
            }

        # Collect summaries of all prior sections
        prior_summaries = []
        for s in doc.sections[:target_section]:
            if s.summary:
                prior_summaries.append(
                    f"Section {s.index + 1} ({s.topic[:60]}): {s.summary}"
                )

        # Get target section's topic for relevance weighting
        target_topic = doc.sections[target_section].topic
        target_summary = doc.sections[target_section].summary

        # Upcoming topics (brief preview of what's ahead)
        upcoming = [
            s.topic[:80] for s in doc.sections[target_section + 1:]
            if s.topic
        ]

        if not prior_summaries:
            return {
                "ok": True,
                "doc_id": doc_id,
                "target_section": target_section,
                "rolling_summary": "",
                "upcoming_topics": upcoming,
                "token_est": 0,
            }

        combined_prior = "\n\n".join(prior_summaries)

        logger.info(
            "CFR: Generating rolling context for section %d "
            "(topic: %s, %d prior sections)...",
            target_section, target_topic[:40], len(prior_summaries),
        )

        rolling = self._vllm.chat(
            "You are GAIA, preparing context for yourself. Given summaries of "
            "previous sections of a document and the topic of the section you "
            "are about to read, write a 100-200 word rolling summary of "
            "everything so far. Emphasize details from earlier sections that "
            "are most relevant to the upcoming topic. Compress distant context "
            "more aggressively than recent context. Be factual and precise.",
            f"Upcoming section topic: {target_topic}\n"
            f"Upcoming section summary: {target_summary}\n\n"
            f"Prior sections:\n{combined_prior}\n\n"
            f"Write the rolling context summary.",
            max_tokens=500,
            temperature=0.3,
        )

        return {
            "ok": True,
            "doc_id": doc_id,
            "target_section": target_section,
            "rolling_summary": rolling,
            "upcoming_topics": upcoming,
            "token_est": _estimate_tokens(rolling),
        }

    def status(self, doc_id: str = "") -> Dict[str, Any]:
        """
        Show resolution state for a document, or list all documents.
        """
        if not doc_id:
            # List all documents
            idx_path = self._index_path()
            if not idx_path.exists():
                return {"ok": True, "documents": {}, "count": 0}
            index = json.loads(idx_path.read_text())
            docs = index.get("documents", {})
            return {"ok": True, "documents": docs, "count": len(docs)}

        doc = self._load_tree(doc_id)
        if not doc:
            return {"ok": False, "error": f"Document not found: {doc_id}"}

        focused_tokens = sum(s.token_est for s in doc.sections if s.state == "focused")
        compressed_tokens = sum(s.summary_token_est for s in doc.sections if s.state == "compressed")

        return {
            "ok": True,
            "doc_id": doc_id,
            "source_path": doc.source_path,
            "section_count": len(doc.sections),
            "total_tokens_est": doc.total_tokens_est,
            "focused_tokens": focused_tokens,
            "compressed_tokens": compressed_tokens,
            "synthesis_exists": bool(doc.synthesis.text),
            "sections": [
                {
                    "index": s.index,
                    "state": s.state,
                    "topic": s.topic,
                    "token_est": s.token_est,
                    "summary_token_est": s.summary_token_est,
                }
                for s in doc.sections
            ],
        }

    # -----------------------------------------------------------------------
    # Internal LLM helpers
    # -----------------------------------------------------------------------
    def _generate_summary(self, section_text: str) -> str:
        """Generate a 100-200 word factual summary of a section."""
        return self._vllm.chat(
            self.SUMMARY_SYSTEM,
            f"Summarize this text section:\n\n{section_text}",
            max_tokens=400,
            temperature=0.3,
        )
