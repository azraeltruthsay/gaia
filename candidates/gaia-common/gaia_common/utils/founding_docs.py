"""Her founding documents, by the names she'd be asked about them (g47g)."""
import re
from typing import Optional, Tuple

DOCS_DIR = "/knowledge/system_reference/core_documents"
_NAMED = (
    (r"\bdeclaration\b(?:\s+of\s+artisanal)?", "declaration_of_artisanal_intelligence.md", "the Declaration of Artisanal Intelligence"),
    (r"\bconstitution\b", "gaia_constitution.md", "the GAIA Constitution"),
    (r"\blayered\s+identity\b", "layered_identity_model.md", "the Layered Identity Model"),
    (r"\bmindscape\b", "mindscape_manifest.md", "the Mindscape Manifest"),
    (r"\bcoalition\s+of\s+minds\b", "coalition_of_minds.md", "the Coalition of Minds"),
    (r"\bsamvega\s+spec\b", "samvega_spec.md", "the Samvega Spec"),
    (r"\btiers\s+of\s+(?:cognition)\b", "tiers_of_cognition.md", "the Tiers of Cognition"),
    (r"\btiers\s+of\s+(?:knowledge)\b", "tiers_of_knowledge.md", "the Tiers of Knowledge"),
)
# Someone else's: "the US Constitution", "the Declaration of Independence".
_SOMEONE_ELSES_RE = re.compile(
    r"\b(?:u\.?s\.?a?|united\s+states|american|french|india\w*|japan\w*|german\w*|canad\w*|state|independence|"
    r"human\s+rights|war|national)\b", re.IGNORECASE)
_HERS_RE = re.compile(r"\b(?:your|yours|gaia(?:'s)?|her|you)\b", re.IGNORECASE)
_BARE_ARTICLE_RE = re.compile(r"\b[Aa]rticle\s+(?:[IVXLC]+|[A-G])\b(?!\s+of\b)")
_READ_IT_RE = re.compile(r"\bread\b|\bwhole\b|\bentire\b|\ball\s+of\b|\bthrough\b", re.IGNORECASE)
_ANY_ARTICLE_RE = re.compile(r"\barticle\b", re.IGNORECASE)
_OPINION_RE = re.compile(
    r"\b(?:think|thoughts?|feel|opinion|make\s+of|agree|disagree|like|believe|reflect|judge?ment|judge|impression|"
    r"assess\w*|evaluate|critique)\b", re.IGNORECASE)
_BACK_RE = re.compile(r"\b(?:it|its|that|this)\b", re.IGNORECASE)


def named_document(text: str) -> Optional[Tuple[str, str]]:
    """(path, title) of the founding document a message is asking about."""
    text = text or ""
    if _SOMEONE_ELSES_RE.search(text):
        return None
    for pattern, filename, title in _NAMED:
        if re.search(pattern, text, re.IGNORECASE) and (_HERS_RE.search(text) or "gaia" in text.lower()):
            return f"{DOCS_DIR}/{filename}", title
    if _BARE_ARTICLE_RE.search(text):
        return f"{DOCS_DIR}/gaia_constitution.md", "the GAIA Constitution"
    return None


def wants_whole(text: str) -> bool:
    """Asked to read it through, with no one part named: "read your constitution
    and tell me what you think". Given one section, she took it for all of it."""
    return bool(_READ_IT_RE.search(text or "")) and not _ANY_ARTICLE_RE.search(text or "")


def asks_opinion(text: str) -> bool:
    return bool(_OPINION_RE.search(text or ""))


def refers_back(text: str) -> bool:
    """An opinion asked of what she just read: "what do you think of it?"."""
    return asks_opinion(text) and bool(_BACK_RE.search(text or ""))
