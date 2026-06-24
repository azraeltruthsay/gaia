"""
Knowledge Ingestion Pipeline for D&D Campaign Content

Detects incoming knowledge dumps (explicit save commands or heuristic auto-detect),
classifies content, checks for duplicates, formats as structured markdown documents,
and writes + embeds via MCP.
"""

import logging
import re
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

from gaia_core.utils import mcp_client

logger = logging.getLogger("GAIA.KnowledgeIngestion")

# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

# Patterns for explicit save commands
_SAVE_PATTERNS = [
    # Natural language: "save this about X", "remember this about X", "document this about X"
    re.compile(
        r"(?:save|remember|document|record|store|log)\s+(?:this|the\s+following)\s+(?:info(?:rmation)?\s+)?about\s+(?P<subject>.+?)(?:\s*:\s*(?P<content>.+))?$",
        re.IGNORECASE | re.DOTALL,
    ),
    # "save the following to my knowledge base: <content>"
    re.compile(
        r"(?:save|remember|document|record|store|log|add)\s+(?:the\s+following|this)\s+to\s+(?:my\s+)?knowledge\s*base\s*:\s*(?P<content>.+)",
        re.IGNORECASE | re.DOTALL,
    ),
    # Natural language without explicit subject: "save this", "remember this"
    re.compile(
        r"(?:save|remember|document|record|store|log)\s+(?:this|the\s+following)\s*(?::\s*(?P<content>.+))?$",
        re.IGNORECASE | re.DOTALL,
    ),
    # Legacy DOCUMENT format
    re.compile(
        r'GAIA,?\s*DOCUMENT\s+"(?P<subject>[^"]+)"\s+AS\s+"(?P<symbol>[^"]+)"\s+ABOUT\s+"(?P<content>.+)"',
        re.IGNORECASE | re.DOTALL,
    ),
]

# D&D entity signals for auto-detection heuristic
_DND_ENTITY_PATTERNS = re.compile(
    r"\b(?:"
    # Characters
    r"rupert(?:\s+roads)?|axuraud|mr\.?\s*bob|anton\s*snark|gilvestri|nathaniel|"
    r"thorne\s*weaver|zarut|skid|ninthalor|fjorlor|scantron|wendalis|bergrune|"
    # Locations
    r"bra[eē]n[eē]age|snowreach|rogue'?s?\s*end|ruadh\s*craic|rusthook|the\s+thorn|"
    r"dragonfjord|heartport|njivelun|heimr|strauthauk|co'?lire|vedania|qabaeth|"
    r"llasak|moon\s*gate|the\s+shine|antcelu|astral\s*sea|primordis|nondis|"
    # Items, factions, deities, events
    r"blueshot|r\.?s\.?s\.?\s*alice|ur\s*machine|arcani|"
    r"tower\s+faction|autonomes|jade\s*phoenix|"
    r"the\s+maid|the\s+healer|the\s+dreamer|"
    r"cernunnos|ayam|ayth\s*sehual|yesh'?thual|tharizdun|"
    r"candlelight\s*saga|the\s+flux|vaniss|"
    # D&D mechanics
    r"hit\s*points?|armor\s*class|saving\s*throw|ability\s*score|"
    r"stat\s*block|spell\s*slot|cantrip|"
    r"d20|d\d+|AC\s*\d|HP\s*\d|DC\s*\d"
    r")\b",
    re.IGNORECASE,
)

# Structural signals (bullets, headers, stat blocks)
_STRUCTURAL_PATTERN = re.compile(
    r"(?:^[-*]\s+.+$)|(?:^#{1,4}\s+.+$)|(?:^\*\*[^*]+\*\*\s*:)",
    re.MULTILINE,
)

_MIN_AUTO_DETECT_LENGTH = 300
_MIN_ENTITY_HITS = 2
_MIN_STRUCTURAL_HITS = 3


def detect_save_command(user_input: str) -> Optional[Dict[str, str]]:
    """
    Check if the user explicitly asked to save/document information.

    Returns:
        Dict with keys {subject, raw_content, symbol (optional)} or None.
    """
    for pattern in _SAVE_PATTERNS:
        m = pattern.search(user_input)
        if m:
            groups = m.groupdict()
            subject = (groups.get("subject") or "").strip()
            content = (groups.get("content") or "").strip()
            symbol = (groups.get("symbol") or "").strip()

            # Strip trailing conversational noise from subject
            subject = re.sub(
                r"(?:\s+(?:for me|please|if you (?:can|could|would|don't mind)|thanks|thank you))+\s*[?.!]*$",
                "", subject, flags=re.IGNORECASE,
            ).rstrip("?.! ")

            # If no explicit content portion, the whole message is the content
            if not content:
                content = user_input

            result = {"subject": subject or "untitled", "raw_content": content}
            if symbol:
                result["symbol"] = symbol
            logger.info(f"Explicit save command detected: subject='{result['subject']}'")
            return result
    return None


def detect_knowledge_dump(user_input: str, kb_name: str) -> bool:
    """
    Heuristic auto-detection: fires when kb_name is 'dnd_campaign', the message
    is long enough, and contains D&D entity names or structural signals.
    No LLM call — pure pattern matching.
    """
    if kb_name != "dnd_campaign":
        return False
    if len(user_input) < _MIN_AUTO_DETECT_LENGTH:
        return False

    entity_hits = len(_DND_ENTITY_PATTERNS.findall(user_input))
    structural_hits = len(_STRUCTURAL_PATTERN.findall(user_input))

    triggered = entity_hits >= _MIN_ENTITY_HITS or structural_hits >= _MIN_STRUCTURAL_HITS
    if triggered:
        logger.info(
            f"Knowledge dump auto-detected: entity_hits={entity_hits}, "
            f"structural_hits={structural_hits}, len={len(user_input)}"
        )
    return triggered


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

_CATEGORY_KEYWORDS = {
    "character": [
        "character", "backstory", "race", "class", "level", "ability score",
        "stat block", "hit points", "HP", "AC", "proficiency",
    ],
    "session_recap": [
        "session", "recap", "last time", "previously", "we played",
        "the party", "adventure log",
    ],
    "rules": [
        "rule", "mechanic", "homebrew", "house rule", "ruling",
        "spell slot", "action economy", "multiclass",
    ],
    "lore": [
        "lore", "history", "faction", "region", "deity", "god",
        "legend", "myth", "world", "continent", "city", "town",
    ],
}


def classify_content(text: str) -> Dict[str, str]:
    """
    Keyword-based categorization of D&D content.

    Returns:
        Dict with {category, tags, suggested_title, suggested_symbol}.
    """
    text_lower = text.lower()

    scores = {}
    for category, keywords in _CATEGORY_KEYWORDS.items():
        scores[category] = sum(1 for kw in keywords if kw.lower() in text_lower)

    category = max(scores, key=scores.get) if max(scores.values()) > 0 else "lore"

    # Extract a rough title from the first significant line
    first_line = ""
    for line in text.split("\n"):
        stripped = line.strip().lstrip("#").strip().strip("*").strip()
        if len(stripped) > 5:
            first_line = stripped
            break
    suggested_title = first_line[:80] if first_line else "Untitled D&D Content"

    # Build tags from matching keywords
    tags = ["dnd_campaign"]
    for kw in _CATEGORY_KEYWORDS.get(category, []):
        if kw.lower() in text_lower and kw.lower() not in tags:
            tags.append(kw.lower())
    tags = tags[:8]  # cap

    sanitized = re.sub(r"[^a-z0-9]+", "_", suggested_title.lower()).strip("_")[:40]
    suggested_symbol = f"DND_{category.upper()}_{sanitized.upper()}"

    return {
        "category": category,
        "tags": tags,
        "suggested_title": suggested_title,
        "suggested_symbol": suggested_symbol,
    }


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

_DEDUP_SIMILARITY_THRESHOLD = 0.85


def check_dedup(content: str, kb_name: str) -> Optional[Dict]:
    """
    Semantic similarity check against existing documents.

    Returns match info dict if a near-duplicate exists, otherwise None.
    """
    try:
        result = mcp_client.embedding_query(
            content[:500], top_k=1, knowledge_base_name=kb_name
        )
        if result.get("ok") and result.get("results"):
            top_hit = result["results"][0]
            similarity = top_hit.get("similarity", top_hit.get("score", 0))
            if similarity >= _DEDUP_SIMILARITY_THRESHOLD:
                logger.info(
                    f"Dedup match found: similarity={similarity:.3f}, "
                    f"source={top_hit.get('source', 'unknown')}"
                )
                return {
                    "similarity": similarity,
                    "source": top_hit.get("source", "unknown"),
                    "title": top_hit.get("title", top_hit.get("source", "unknown")),
                    "snippet": top_hit.get("text", "")[:200],
                }
    except Exception as e:
        logger.warning(f"Dedup check failed: {e}")
    return None


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _sanitize_filename(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:50]


def format_document(
    content: str,
    classification: Dict[str, str],
    subject: str = "",
) -> Tuple[str, str]:
    """
    Generate a markdown document with YAML front matter.

    Returns:
        (filename, document_string)
    """
    category = classification["category"]
    title = subject or classification["suggested_title"]
    symbol = classification["suggested_symbol"]
    tags = classification["tags"]
    now = datetime.now(timezone.utc)

    front_matter = (
        f"---\n"
        f"symbol: {symbol}\n"
        f'title: "{title}"\n'
        f"category: {category}\n"
        f"tags: [{', '.join(tags)}]\n"
        f"source: discord_ingestion\n"
        f'created: "{now.strftime("%Y-%m-%dT%H:%M:%SZ")}"\n'
        f'version: "1.0"\n'
        f"scope: dnd_campaign\n"
        f"---\n"
    )

    body = f"\n## {title}\n\n{content.strip()}\n"
    document = front_matter + body

    date_str = now.strftime("%Y%m%d")
    filename = f"{category}_{_sanitize_filename(subject or title)}_{date_str}.md"

    return filename, document


# ---------------------------------------------------------------------------
# Write + Embed
# ---------------------------------------------------------------------------

def write_and_embed(
    filename: str, doc_content: str, kb_name: str
) -> Dict[str, object]:
    """
    Two MCP calls:
      1. write_file to persist the document
      2. embed_documents to index it for RAG retrieval

    Returns dict with {ok, path, embed_ok, error}.
    """
    from gaia_core.config import get_config
    config = get_config()
    kb_config = config.constants.get("KNOWLEDGE_BASES", {}).get(kb_name, {})
    doc_dir = kb_config.get("doc_dir", f"projects/{kb_name}/core-documentation")
    file_path = f"/knowledge/{doc_dir}/{filename}"

    # Step 1: write_file via MCP
    write_result = mcp_client.call_jsonrpc(
        "write_file",
        {"path": file_path, "content": doc_content},
    )

    if not write_result.get("ok"):
        # Check for 403 (approval required) — the MCP layer handles this
        error = write_result.get("error", "write_file failed")
        logger.error(f"write_file failed for {file_path}: {error}")
        return {"ok": False, "path": file_path, "embed_ok": False, "error": error}

    logger.info(f"Document written to {file_path}")

    # Step 2: embed_documents via MCP
    embed_result = mcp_client.call_jsonrpc(
        "embed_documents",
        {"knowledge_base_name": kb_name, "file_path": file_path},
    )
    embed_ok = embed_result.get("ok", False)
    if embed_ok:
        logger.info(f"Document embedded into '{kb_name}' vector store")
    else:
        logger.warning(f"embed_documents failed: {embed_result.get('error', 'unknown')}")

    return {"ok": True, "path": file_path, "embed_ok": embed_ok, "error": None}


# ---------------------------------------------------------------------------
# High-level orchestration (used by agent_core)
# ---------------------------------------------------------------------------

def run_explicit_save(
    user_input: str, kb_name: str
) -> Optional[Dict]:
    """
    Full pipeline for an explicit save command.
    Returns result dict or None if no save command detected.
    """
    save_cmd = detect_save_command(user_input)
    if not save_cmd:
        return None

    classification = classify_content(save_cmd["raw_content"])

    # Override symbol if legacy format provided one
    if save_cmd.get("symbol"):
        classification["suggested_symbol"] = save_cmd["symbol"].upper().replace(" ", "_")

    # Dedup check
    dedup = check_dedup(save_cmd["raw_content"], kb_name)
    if dedup:
        return {
            "action": "dedup_blocked",
            "subject": save_cmd["subject"],
            "existing_doc": dedup,
        }

    filename, document = format_document(
        save_cmd["raw_content"], classification, subject=save_cmd["subject"]
    )
    result = write_and_embed(filename, document, kb_name)
    result["subject"] = save_cmd["subject"]
    result["action"] = "saved"
    return result


def run_auto_detect(
    user_input: str, kb_name: str
) -> Optional[Dict]:
    """
    Heuristic detection for knowledge dumps. Returns classification dict
    (to be attached as a DataField for the offer flow) or None.
    """
    if not detect_knowledge_dump(user_input, kb_name):
        return None

    classification = classify_content(user_input)

    # Dedup check — if duplicate exists, don't even offer
    dedup = check_dedup(user_input, kb_name)
    if dedup:
        logger.info(f"Auto-detect: duplicate found, not offering save. source={dedup['source']}")
        return None

    return classification


# ---------------------------------------------------------------------------
# LLM-Driven Domain Classification (Nano model)
# ---------------------------------------------------------------------------

# Short descriptive enrichments for known domains — the bare KB key ("system")
# embeds poorly, so we give the label a few words of meaning. Unknown KBs fall
# back to the humanized key (underscores → spaces). Extend as KBs are added.
_DOMAIN_DESCRIPTIONS = {
    "dnd_campaign": "tabletop Dungeons & Dragons campaign — characters, lore, quests, sessions",
    "system": "GAIA system internals — services, architecture, configuration, operations",
    "blueprints": "design blueprints and architecture plans for the cognitive system",
    "code": "source code, functions, APIs, programming, implementation",
    "general": "general knowledge that fits no specialized domain",
}


def classify_domain(text_snippet: str, available_kbs: list) -> str:
    """Classify which knowledge domain a text snippet belongs to via EMBEDDING
    nearest-neighbour — no LLM (86x: domain-from-fixed-label-set is structurally
    decidable). Encodes the snippet and each candidate domain (enriched with a
    short description), picks the highest cosine; below threshold → 'general'.
    Deterministic, cheap, reuses the shared CPU embed model (no GPU/model load).
    Degrades to a keyword heuristic, then 'general', if embeddings are unavailable."""
    if not available_kbs:
        available_kbs = ["dnd_campaign", "system", "blueprints", "general"]
    if "general" not in available_kbs:
        available_kbs = available_kbs + ["general"]

    snippet = (text_snippet or "").strip()
    if not snippet:
        return "general"

    try:
        from gaia_core.memory.session_history_indexer import _get_embed_model, _cosine_similarity
        import numpy as np

        embed_model = _get_embed_model()
        if embed_model is None:
            return _classify_domain_keyword(snippet, available_kbs)

        # Domain texts: enriched description when known, else the humanized key.
        domain_texts = [
            _DOMAIN_DESCRIPTIONS.get(kb, kb.replace("_", " ")) for kb in available_kbs
        ]
        vecs = embed_model.encode(
            [snippet[:1000]] + domain_texts,
            show_progress_bar=False, normalize_embeddings=True,
        )
        vecs = np.asarray(vecs, dtype=np.float32)
        q, domain_vecs = vecs[0], vecs[1:]

        best_kb, best_sim = "general", -1.0
        for kb, dv in zip(available_kbs, domain_vecs):
            sim = _cosine_similarity(q, dv)
            if sim > best_sim:
                best_kb, best_sim = kb, sim

        # Confidence floor: a weak best match means no domain really fits → general.
        _DOMAIN_SIM_FLOOR = 0.25
        if best_sim < _DOMAIN_SIM_FLOOR:
            logger.info("Domain classify: best '%s' sim %.2f < floor — using 'general' for '%s'",
                        best_kb, best_sim, snippet[:50])
            return "general"
        logger.info("Domain classify (embed): '%s' → %s (sim %.2f)", snippet[:50], best_kb, best_sim)
        return best_kb

    except Exception as e:
        logger.warning("Embedding domain classification failed: %s — keyword fallback", e)
        return _classify_domain_keyword(snippet, available_kbs)


# High-signal surface keywords per domain, for the embeddings-down fallback only.
# (The embed path handles semantics; this just needs to be reasonable when
# embeddings are unavailable.) The description words are folded in as well.
_DOMAIN_KEYWORDS = {
    "dnd_campaign": ["party", "dragon", "rogue", "initiative", "campaign", "dungeon",
                     "spell", "quest", "character", "session", "lore", "dice", "dnd"],
    "system": ["service", "orchestrator", "container", "config", "restart", "docker",
               "health", "gpu", "engine", "architecture", "deploy", "endpoint"],
    "blueprints": ["blueprint", "design", "plan", "architecture", "diagram", "spec"],
    "code": ["function", "class", "api", "import", "def ", "return", "code", "module"],
}


def _classify_domain_keyword(snippet: str, available_kbs: list) -> str:
    """Last-resort keyword heuristic when embeddings are unavailable. Scores the
    snippet against per-domain keyword banks (+ description words)."""
    low = snippet.lower()
    best_kb, best_hits = "general", 0
    for kb in available_kbs:
        terms = set(_DOMAIN_KEYWORDS.get(kb, []))
        terms |= {t for t in _DOMAIN_DESCRIPTIONS.get(kb, kb).replace("_", " ").lower().split()
                  if len(t) >= 4}
        hits = sum(1 for t in terms if t in low)
        if hits > best_hits:
            best_kb, best_hits = kb, hits
    return best_kb


# Backwards-compat alias (the function is no longer LLM-based; 86x).
classify_domain_llm = classify_domain


def run_attachment_ingestion(
    filename: str,
    text_content: str,
    user_hint: str = "",
) -> Dict:
    """
    Full ingestion pipeline for a file attachment.

    1. Classify domain via Nano LLM
    2. Override with user hint if present
    3. Classify content category (keyword-based)
    4. Dedup check
    5. Format and write + embed

    Returns dict with {action, kb_name, category, path, embed_ok, error}.
    """
    from gaia_core.behavior.persona_switcher import get_available_knowledge_bases

    available_kbs = get_available_knowledge_bases()

    # Step 1: LLM domain classification
    kb_name = classify_domain(text_content, available_kbs)

    # Step 2: User hint override (e.g., "save this D&D stuff")
    if user_hint:
        hint_lower = user_hint.lower()
        for kb in available_kbs:
            # Check if user explicitly mentioned a domain
            kb_keywords = {
                "dnd_campaign": ["d&d", "dnd", "campaign", "dungeon"],
                "system": ["system", "architecture", "config"],
                "blueprints": ["blueprint", "design", "plan"],
            }
            for kw in kb_keywords.get(kb, []):
                if kw in hint_lower:
                    logger.info("User hint override: '%s' → kb=%s", kw, kb)
                    kb_name = kb
                    break

    # Step 3: Keyword-based content classification within the domain
    classification = classify_content(text_content)

    # Step 4: Dedup check
    dedup = check_dedup(text_content, kb_name)
    if dedup:
        logger.info("Attachment dedup blocked: %s already exists (sim=%.3f)", filename, dedup["similarity"])
        return {
            "action": "dedup_blocked",
            "kb_name": kb_name,
            "category": classification["category"],
            "existing_doc": dedup,
            "path": None,
            "embed_ok": False,
            "error": None,
        }

    # Step 5: Format document and write + embed
    doc_filename, document = format_document(text_content, classification, subject=filename)
    result = write_and_embed(doc_filename, document, kb_name)

    return {
        "action": "saved",
        "kb_name": kb_name,
        "category": classification["category"],
        "path": result.get("path"),
        "embed_ok": result.get("embed_ok", False),
        "error": result.get("error"),
    }


# ---------------------------------------------------------------------------
# Knowledge Update Detection (casual updates to existing entities)
# ---------------------------------------------------------------------------

# Known character entities (extracted from _DND_ENTITY_PATTERNS)
_CHARACTER_ENTITIES = re.compile(
    r"\b(?P<entity>rupert(?:\s+roads)?|axuraud|mr\.?\s*bob|nathaniel|strauthauk|"
    r"anton\s*snark|gilvestri|thorne\s*weaver|zarut|skid|ninthalor|fjorlor)\b",
    re.IGNORECASE,
)

# State-change language patterns
_UPDATE_PATTERNS = [
    # "X is now level 9", "X is now a level 9 artificer"
    re.compile(r"(?P<entity>{ent})\s+is\s+now\s+(?P<signal>.+)", re.IGNORECASE),
    # "X has/got/learned/gained Y"
    re.compile(r"(?P<entity>{ent})\s+(?:has|got|learned|gained|acquired|took|picked\s+up)\s+(?P<signal>.+)", re.IGNORECASE),
    # "update/change X" or "X was updated/changed"
    re.compile(r"(?:update|change|modify|edit)\s+(?P<entity>{ent})\b.*", re.IGNORECASE),
    re.compile(r"(?P<entity>{ent})\s+(?:was|got)\s+(?:updated|changed|modified)\b.*", re.IGNORECASE),
    # "X leveled up / X hit level N / X reached level N"
    re.compile(r"(?P<entity>{ent})\s+(?:leveled?\s+up|hit\s+level|reached\s+level)\s*(?P<signal>.*)", re.IGNORECASE),
]

# Pre-compile with entity alternation injected
_ENTITY_ALT = r"(?:rupert(?:\s+roads)?|axuraud|mr\.?\s*bob|nathaniel|strauthauk|anton\s*snark|gilvestri|thorne\s*weaver|zarut|skid|ninthalor|fjorlor)"
_COMPILED_UPDATE_PATTERNS = [
    re.compile(p.pattern.replace("{ent}", _ENTITY_ALT), p.flags)
    for p in _UPDATE_PATTERNS
]

# Question guards — don't trigger on questions about state
_QUESTION_STARTS = re.compile(
    r"^\s*(?:is|are|was|were|do|does|did|has|have|had|can|could|will|would|should|what|when|where|who|how|why)\b",
    re.IGNORECASE,
)

_RETRIEVAL_SIMILARITY_THRESHOLD = 0.5


def detect_knowledge_update(
    user_input: str, kb_name: str
) -> Optional[Dict[str, str]]:
    """
    Detect casual knowledge updates referencing known entities.

    Fires for state-change language like "Rupert is now level 9" but NOT for
    questions like "Is Rupert level 9?".

    Returns:
        Dict with {entity, update_signal, raw_input} or None.
    """
    if kb_name != "dnd_campaign":
        return None

    # Question guard: skip trailing ? or question-word starts
    stripped = user_input.strip()
    if stripped.endswith("?"):
        return None
    if _QUESTION_STARTS.match(stripped):
        return None

    for pattern in _COMPILED_UPDATE_PATTERNS:
        m = pattern.search(user_input)
        if m:
            entity = m.group("entity").strip().lower()
            # Normalize entity name
            entity = re.sub(r"\s+", " ", entity)
            signal = m.group("signal").strip() if "signal" in m.groupdict() else ""
            logger.info(f"Knowledge update detected: entity={entity}, signal={signal!r}")
            return {
                "entity": entity,
                "update_signal": signal or user_input.strip(),
                "raw_input": user_input,
            }

    return None


def retrieve_entity_document(
    entity: str, kb_name: str
) -> Optional[Dict]:
    """
    Targeted RAG query to find the existing document for a character entity.

    Uses a lower similarity threshold (0.5) than dedup (0.85) to catch
    loosely matching documents.

    Returns:
        The top matching document dict or None.
    """
    try:
        result = mcp_client.embedding_query(
            f"{entity} character sheet",
            top_k=1,
            knowledge_base_name=kb_name,
        )
        if result.get("ok") and result.get("results"):
            top_hit = result["results"][0]
            similarity = top_hit.get("similarity", top_hit.get("score", 0))
            if similarity >= _RETRIEVAL_SIMILARITY_THRESHOLD:
                logger.info(
                    f"Entity document found: entity={entity}, "
                    f"similarity={similarity:.3f}, source={top_hit.get('source', 'unknown')}"
                )
                return top_hit
            else:
                logger.debug(
                    f"Entity document below threshold: entity={entity}, "
                    f"similarity={similarity:.3f}"
                )
    except Exception as e:
        logger.warning(f"Entity document retrieval failed for '{entity}': {e}")
    return None


def run_update_detect(
    user_input: str, kb_name: str
) -> Optional[Dict]:
    """
    Detect casual updates to existing entities and retrieve the existing doc.

    Returns:
        Dict with {entity, update_signal, raw_input, existing_doc} or None.
    """
    update_info = detect_knowledge_update(user_input, kb_name)
    if not update_info:
        return None

    existing_doc = retrieve_entity_document(update_info["entity"], kb_name)
    update_info["existing_doc"] = existing_doc
    return update_info
