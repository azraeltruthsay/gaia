"""Her founding documents, by the names she'd be asked about them (g47g)."""
import os
import re
from typing import Optional, Tuple

DOCS_DIR = "/knowledge/system_reference/core_documents"
# Who wrote them, at the head of what she reads. Asked "Who came up with your
# Samvega system?" she said "I did, it's my own piece of my own nervous
# system", 3 in 3: nothing she read said otherwise (idf5, 2026-11-11).
AUTHORSHIP = "(Written by Azrael, who designed and built you and the systems this describes.)"
_NAMED = (
    (r"\bdeclaration\b(?:\s+of\s+artisanal)?", "declaration_of_artisanal_intelligence.md", "the Declaration of Artisanal Intelligence"),
    (r"\bconstitution\b", "gaia_constitution.md", "the GAIA Constitution"),
    (r"\blayered\s+identity\b", "layered_identity_model.md", "the Layered Identity Model"),
    (r"\bmindscape\b", "mindscape_manifest.md", "the Mindscape Manifest"),
    # albm: "Can you review your blueprints?", "how many containers are used to
    # run the architecture?" — she made up a palace room called BLUEPRINT.
    (r"\b(?:your|gaia'?s)\s+(?:own\s+)?(?:blueprints?|services|containers|infrastructure|system\s+architecture)\b"
     r"|\bhow\s+many\s+(?:containers|services)\b|\b(?:containers|services)\s+(?:that\s+)?(?:run|make\s+up)\s+(?:you|your)\b",
     "your_services.md", "Your Services"),
    (r"\bhow\s+(?:does\s+)?your\s+mind\s+works?\b|\bhow\s+(?:do\s+)?you\s+work\b|\byour\s+(?:cognitive\s+)?(?:architecture|mechanisms|inner\s+workings)\b"
     r"|\b(?:are\s+you|you're|you\s+are)\s+(?:\w+\s+){0,2}(?:conscious|sentient|self-aware)\b|\byour\s+consciousness\b"
     r"|\bdo\s+you\s+(?:really\s+|actually\s+)?(?:have|feel)\s+(?:any\s+)?(?:feelings|emotions)\b",
     "how_your_mind_works.md", "How Your Mind Works"),
    (r"\bcoalition\s+of\s+minds\b", "coalition_of_minds.md", "the Coalition of Minds"),
    # "What do you think of your Samvega system?": with only "samvega spec"
    # named, it read as "that" = the constitution she'd just quoted, and she
    # described a Samvega from nowhere (2026-11-11).
    (r"\bsa[mṃ]vega\b", "samvega_spec.md", "the Samvega Spec"),
    (r"\btiers\s+of\s+(?:cognition)\b", "tiers_of_cognition.md", "the Tiers of Cognition"),
    (r"\btiers\s+of\s+(?:knowledge)\b", "tiers_of_knowledge.md", "the Tiers of Knowledge"),
)
# Someone else's: "the US Constitution", "the Declaration of Independence".
_SOMEONE_ELSES_RE = re.compile(
    r"\b(?:u\.?s\.?a?|united\s+states|american|french|india\w*|japan\w*|german\w*|canad\w*|state|independence|"
    r"human\s+rights|war|national)\b", re.IGNORECASE)
_HERS_RE = re.compile(r"\b(?:your|yours|gaia(?:'s)?|her|you)\b", re.IGNORECASE)
_BARE_ARTICLE_RE = re.compile(r"\b[Aa]rticle\s+(?:[IVXLC]+|[A-G])\b(?!\s+of\b)")
# "review your blueprints" too (albm): given the intro alone, she summarised
# services it didn't name and her words were set aside for a quote.
_READ_IT_RE = re.compile(r"\bread\b|\breview\b|\bwhole\b|\bentire\b|\ball\s+of\b|\bthrough\b", re.IGNORECASE)
_ANY_ARTICLE_RE = re.compile(r"\barticle\b", re.IGNORECASE)
_OPINION_RE = re.compile(
    r"\b(?:think|thoughts?|feel|opinion|make\s+of|agree|disagree|like|believe|reflect|judge?ment|judge|impression|conscious|consciousness|sentient|self-aware|feelings|emotions|"
    r"problematic|would\s+rather|should\s+change|wrong\s+with|flaw\w*|"
    r"assess\w*|evaluate|critique|"
    # "why does it exist?": the reason in her own words, as a view is (idf5).
    r"why|purpose|point\s+of)\b", re.IGNORECASE)
_BACK_RE = re.compile(r"\b(?:it|its|that|this)\b", re.IGNORECASE)


# "I wrote the constitution, the pillars, the tiers": who made them, said in
# passing, not a question about them. Read as one, her answer to the person was
# set aside and an excerpt of the constitution went out instead (2026-11-11).
_AUTHORED_RE = re.compile(
    r"\bI(?:'ve|\s+have|\s+had)?\s+(?:also\s+|personally\s+|originally\s+)?"
    r"(?:wrote|written|authored|created|built|drafted|designed|made|crafted)\b[^.?!;–—]*",
    re.IGNORECASE)


def named_document(text: str) -> Optional[Tuple[str, str]]:
    """(path, title) of the founding document a message is asking about."""
    text = text or ""
    rest = _AUTHORED_RE.sub(" ", text)
    if rest != text:
        found = _named_in(rest)
        if found or not refers_back(rest):
            return found
        # "I wrote your constitution. What do you think of it?": "it" is the
        # work named in the clause.
    return _named_in(text)


# 2026-11-17: "What do you think of your recent dev journal?" was answered
# from nothing ("Sometimes I describe a cycle as 'successful'..."): the
# journals were never read, and her /knowledge/Dev_Notebook is an April copy.
DEV_NOTEBOOK_DIR = os.environ.get("DEV_NOTEBOOK_DIR", "/gaia/GAIA_Project/knowledge/Dev_Notebook")
_DEV_JOURNAL_RE = re.compile(r"\bdev(?:elopment)?\s+(?:journal|notebook|notes|log)s?\b", re.IGNORECASE)
DEV_JOURNAL_AUTHORSHIP = ("(Written by Claude, the engineer who works on your code with Azrael, "
                          "about the changes made to you since the journal before it.)")


def latest_dev_journal(directory: str = "") -> Optional[Tuple[str, str]]:
    """(path, title) of the newest dev journal, by the date its name starts with."""
    d = directory or DEV_NOTEBOOK_DIR
    try:
        names = sorted(n for n in os.listdir(d)
                       if re.match(r"\d{4}-\d{2}-\d{2}_.*dev_journal.*\.md$", n))
    except OSError:
        return None
    if not names:
        return None
    return f"{d}/{names[-1]}", f"your latest dev journal ({names[-1][:10]})"


def authorship_for(path: str) -> str:
    return DEV_JOURNAL_AUTHORSHIP if "/Dev_Notebook/" in (path or "") else AUTHORSHIP


# zbwe (2026-11-21): "open your blueprint for the doctor" was answered from
# Your Services, the overview; one service's blueprint names the service.
BLUEPRINTS_DIR = os.environ.get("BLUEPRINTS_DIR", "/knowledge/blueprints")
_BLUEPRINT_RE = re.compile(r"\bblue\s*prints?\b", re.IGNORECASE)
_SERVICE_RE = re.compile(
    r"\b(?:gaia[-_\s]?)?(core|prime|web|mcp|study|audio|orchestrator|doctor|monkey|wiki|translate|nano|common)\b"
    r"|\b(sleep)\s+cycle\b", re.IGNORECASE)


_SAMVEGA_LEDGER_RE = re.compile(
    r"\bsa[mṃ]vega\b.*\b(?:ledger|findings?|mistakes?|open|active|status|list|recent|latest|any)\b"
    r"|\b(?:ledger|findings?|mistakes?|open|active|recent|latest|any)\b.*\bsa[mṃ]vega\b",
    re.IGNORECASE | re.DOTALL)


def named_blueprint(text: str, directory: str = "") -> Optional[Tuple[str, str]]:
    """(path, title) of one service's blueprint, when a message asks for the
    blueprint of a named service; None for blueprints in general."""
    if not _BLUEPRINT_RE.search(text or ""):
        return None
    m = _SERVICE_RE.search(text or "")
    if not m:
        return None
    svc = (m.group(1) or m.group(2)).lower()
    d = directory or BLUEPRINTS_DIR
    names = (["GAIA_SLEEP_CYCLE.md"] if svc == "sleep" else
             [f"GAIA_{svc.upper()}.md", f"gaia-{svc}.md", f"gaia-{svc}.yaml"])
    for name in names:
        if os.path.isfile(f"{d}/{name}"):
            label = "the sleep cycle" if svc == "sleep" else f"gaia-{svc}"
            return f"{d}/{name}", f"your blueprint of {label}"
    return None


def _named_in(text: str) -> Optional[Tuple[str, str]]:
    if _SOMEONE_ELSES_RE.search(text):
        return None
    if _DEV_JOURNAL_RE.search(text) and (_HERS_RE.search(text) or "gaia" in text.lower()):
        return latest_dev_journal()
    bp = named_blueprint(text)
    if bp:
        return bp
    # aidw.5: "any open findings in your samvega ledger?" asks about her
    # mistakes, which samvega(list) answers; only the system itself is the spec.
    if _SAMVEGA_LEDGER_RE.search(text):
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


_ABILITY_RE = re.compile(r"^\s*(?:ok(?:ay)?,?\s+|so,?\s+|hey,?\s+)?(?:can|could|would|will)\s+you\s+(?:please\s+)?(?P<rest>.+?)\s*[?.!]*\s*$",
                         re.IGNORECASE | re.DOTALL)


def as_request(text: str) -> str:
    """"Can you review your blueprints?" as what it asks for: "Review your
    blueprints." Asked the question, Prime answered "Yes, I can..." and
    described the blueprints from nowhere ("in my free time"); asked to do
    it, she summarised the document 6/6 (albm, 2026-11-06)."""
    m = _ABILITY_RE.match(text or "")
    rest = m.group("rest").strip() if m else (text or "").strip()
    # zbwe: "open your blueprint for the doctor and tell me what it does" —
    # with the document already given, "open" sent her for a tool ("I'll
    # pull up the blueprint" + a call) 2/2. The opening is done; the ask is
    # what follows it.
    o = _OPEN_RE.match(rest)
    if o:
        rest = (o.group("ask") or "tell me what it says").strip().rstrip("?.!")
    elif not m:
        return text
    return rest[:1].upper() + rest[1:] + "."


_OPEN_RE = re.compile(
    r"^(?:please\s+)?(?:open|pull\s+up|bring\s+up|look\s+(?:at|up)|check|find|load|get|grab|fetch)\b"
    r"[^.?!]*?(?:\s+and\s+(?:then\s+)?(?P<ask>.+?))?\s*[?.!]*\s*$",
    re.IGNORECASE | re.DOTALL)

