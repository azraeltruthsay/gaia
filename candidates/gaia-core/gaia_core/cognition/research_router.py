"""Multi-source research routine — chains KB queries + web search + synthesis.

A single user turn fans out across all configured knowledge bases and a web
search in parallel, then synthesizes the findings into a structured brief.
"""
from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import gaia_core.utils.mcp_client as mcp_client
from gaia_core.config import get_config

logger = logging.getLogger(__name__)

RESEARCH_PATTERN = re.compile(
    r'^\s*(?:please\s+)?'
    r'(?:'
    r'research(?:\s+for\s+me)?|'
    r'look\s+(?:up|into)|'
    r'find\s+(?:me\s+)?(?:everything|all)\s+(?:we\s+have\s+)?(?:on|about)|'
    r'tell\s+me\s+everything\s+(?:we\s+(?:have|know)\s+)?(?:on|about)|'
    r'what\s+do\s+(?:we|you)\s+(?:have|know)\s+(?:on|about)|'
    r'do\s+(?:some\s+)?research\s+(?:on|about)'
    r')\s+'
    r'(?P<topic>.+?)'
    r'\s*\??\s*\.?\s*$',
    re.IGNORECASE,
)

DEFAULT_PER_KB_TOP_K = 3
WEB_MAX_RESULTS = 4
KB_QUERY_TIMEOUT_S = 8
WEB_QUERY_TIMEOUT_S = 12
# all-MiniLM-L6-v2 cosine: <0.2 is mostly random co-occurrence on common
# tokens. Drop KB hits below this — they pollute the synthesis prompt.
MIN_KB_SCORE = 0.20


def detect_research_intent(user_input: str) -> Optional[str]:
    """Return the research topic if the input is a research request, else None."""
    if not user_input:
        return None
    m = RESEARCH_PATTERN.match(user_input.strip())
    if not m:
        return None
    topic = (m.group("topic") or "").strip().rstrip("?.").strip()
    if len(topic) < 2:
        return None
    return topic


def _query_kb(kb_name: str, query: str, top_k: int) -> Tuple[str, List[Dict]]:
    try:
        rpc = mcp_client.call_jsonrpc(
            "query_knowledge",
            {"knowledge_base_name": kb_name, "query": query, "top_k": top_k},
            timeout=KB_QUERY_TIMEOUT_S,
        )
    except Exception as e:
        logger.debug("research_router: KB '%s' query exception: %s", kb_name, e)
        return kb_name, []
    if not rpc.get("ok"):
        return kb_name, []
    response = rpc.get("response") or {}
    result = response.get("result") if isinstance(response, dict) else response
    if not isinstance(result, list):
        return kb_name, []
    filtered = [h for h in result if (h.get("score") or 0.0) >= MIN_KB_SCORE]
    return kb_name, filtered


def _query_web(query: str) -> List[Dict]:
    try:
        rpc = mcp_client.call_jsonrpc(
            "web_search",
            {"query": query, "max_results": WEB_MAX_RESULTS},
            timeout=WEB_QUERY_TIMEOUT_S,
        )
    except Exception as e:
        logger.debug("research_router: web_search exception: %s", e)
        return []
    if not rpc.get("ok"):
        return []
    response = rpc.get("response") or {}
    result = response.get("result") if isinstance(response, dict) else response
    if isinstance(result, dict):
        return result.get("results") or []
    if isinstance(result, list):
        return result
    return []


def _build_synthesis_prompt(
    topic: str,
    kb_hits: Dict[str, List[Dict]],
    web_hits: List[Dict],
) -> str:
    parts: List[str] = [
        f"Research request: {topic}",
        "",
        "STRICT GROUNDING RULES:",
        "- Only state facts that appear in the SOURCES section below.",
        "- If the sources don't cover something, say 'sources don't say' — do "
        "NOT invent details.",
        "- Cite each fact with the KB name and filename it came from "
        "(e.g. 'per dnd_campaign/rupert_roads_character_sheet').",
        "- If local sources contradict web sources, side with local and note "
        "the discrepancy.",
        "- Two-paragraph maximum. End with one sentence on what's missing.",
        "",
        "=== SOURCES ===",
    ]

    has_local = any(bool(hits) for hits in kb_hits.values())
    if not has_local:
        parts.append("(no relevant local matches)")
    else:
        for kb, hits in kb_hits.items():
            if not hits:
                continue
            parts.append(f"\n[KB: {kb}]")
            for hit in hits[:DEFAULT_PER_KB_TOP_K]:
                fname = hit.get("filename", "?")
                score = hit.get("score", 0.0)
                text = (hit.get("text") or "").strip().replace("\n", " ")[:600]
                parts.append(f"- {fname} (score {score:.2f}): {text}")

    parts.append("\n=== WEB SEARCH ===")
    if not web_hits:
        parts.append("(no web results)")
    else:
        for hit in web_hits[:WEB_MAX_RESULTS]:
            title = hit.get("title", "?")
            url = hit.get("url", "?")
            snippet = (hit.get("snippet") or "").replace("\n", " ")[:300]
            parts.append(f"- [{title}]({url}): {snippet}")

    parts.append("\n=== RESEARCH BRIEF ===")
    parts.append("(Write the brief now. Cite each fact. Stay grounded.)")
    return "\n".join(parts)


def run_research(
    topic: str,
    *,
    model_pool=None,
    selected_model_name: str = "core",
    kb_filter: Optional[List[str]] = None,
    skip_web: bool = False,
) -> Dict[str, Any]:
    """Execute the research pipeline and return a structured result.

    Args:
        topic: What to research.
        model_pool: AgentCore.model_pool — needed for synthesis. If None,
            returns raw hits with no synthesis.
        selected_model_name: Model to use for synthesis.
        kb_filter: Optional whitelist of KB names; default = all configured.
        skip_web: If True, only query local KBs.

    Returns:
        Dict with topic, synthesis, kb_hits, web_hits, elapsed_ms, sources_queried.
    """
    t0 = time.time()
    config = get_config()
    constants = getattr(config, "constants", {}) or {}
    kbs = list((constants.get("KNOWLEDGE_BASES") or {}).keys())
    if kb_filter:
        kbs = [k for k in kbs if k in kb_filter]

    kb_hits: Dict[str, List[Dict]] = {}
    web_hits: List[Dict] = []
    sources_queried: List[str] = []

    if kbs:
        with ThreadPoolExecutor(max_workers=min(len(kbs), 6)) as exe:
            futures = {exe.submit(_query_kb, kb, topic, DEFAULT_PER_KB_TOP_K): kb for kb in kbs}
            for fut in as_completed(futures):
                try:
                    kb, hits = fut.result(timeout=KB_QUERY_TIMEOUT_S + 1)
                    kb_hits[kb] = hits
                    if hits:
                        sources_queried.append(f"kb:{kb}")
                except Exception as e:
                    logger.debug("research_router: KB future failed: %s", e)

    if not skip_web:
        web_hits = _query_web(topic)
        if web_hits:
            sources_queried.append("web")

    synthesis = ""
    if model_pool is not None:
        prompt = _build_synthesis_prompt(topic, kb_hits, web_hits)
        try:
            res = model_pool.forward_to_model(
                selected_model_name,
                messages=[
                    {"role": "system",
                     "content": "You are GAIA. Write a research brief grounded ONLY in the sources provided. Never invent details. Two paragraphs max."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=600,
                temperature=0.2,
            )
            synthesis = (res["choices"][0]["message"]["content"] or "").strip()
        except Exception as e:
            logger.exception("research_router: synthesis call failed: %s", e)

    elapsed_ms = (time.time() - t0) * 1000.0
    return {
        "topic": topic,
        "synthesis": synthesis,
        "kb_hits": kb_hits,
        "web_hits": web_hits,
        "elapsed_ms": elapsed_ms,
        "sources_queried": sources_queried,
    }


def format_research_response(result: Dict[str, Any]) -> str:
    """Render a research result into a single user-facing string."""
    topic = result.get("topic", "")
    synthesis = (result.get("synthesis") or "").strip()
    sources = result.get("sources_queried") or []
    kb_hits = result.get("kb_hits") or {}
    web_hits = result.get("web_hits") or []

    if synthesis:
        body = synthesis
    else:
        # No model available — return a structured raw brief.
        lines = [f"**Research: {topic}**", ""]
        any_local = any(hits for hits in kb_hits.values())
        if any_local:
            lines.append("**Local matches:**")
            for kb, hits in kb_hits.items():
                if not hits:
                    continue
                for h in hits[:DEFAULT_PER_KB_TOP_K]:
                    fname = h.get("filename", "?")
                    score = h.get("score", 0.0)
                    text = (h.get("text") or "").strip().replace("\n", " ")[:300]
                    lines.append(f"- `{kb}` · {fname} (score {score:.2f}): {text}")
            lines.append("")
        else:
            lines.append("(no relevant local matches)")
            lines.append("")
        if web_hits:
            lines.append("**Web matches:**")
            for h in web_hits[:WEB_MAX_RESULTS]:
                title = h.get("title", "?")
                url = h.get("url", "?")
                snippet = (h.get("snippet") or "").replace("\n", " ")[:200]
                lines.append(f"- [{title}]({url}): {snippet}")
        body = "\n".join(lines)

    footer = ""
    if sources:
        footer = f"\n\n_Sources queried: {', '.join(sources)}_"
    save_offer = (
        f"\n\nWant me to save this as a research note? Ask: "
        f"`Create a file at /knowledge/research/{_slug(topic)}.md with content: <paste the brief above>`."
    )
    return body + footer + save_offer


def _slug(text: str) -> str:
    s = re.sub(r'[^\w\s-]', '', (text or '').lower())
    s = re.sub(r'[\s_-]+', '-', s).strip('-')
    return s[:60] or "research"
