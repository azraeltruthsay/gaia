"""Fact-type ontology for the World Model (GAIA_Project-lw4).

Each KG triple can carry a `fact_type` that determines how its
confidence decays over time. The ontology is intentionally small and
explicit — eight classes with hand-picked half-lives — so retrieval
can multiply `confidence × decay(age, fact_type)` without an LLM in
the loop.

Half-lives are calibrated so that:

  - Weather (6h) gets refetched the same day.
  - News (7d) stays current for a week, then loses ground.
  - Market data (1h) is treated as essentially live-or-stale.
  - Political office (1y) is mostly stable across an election cycle.
  - Scientific consensus (50y) is treated as basically permanent
    relative to a single training run.
  - Biographical (None) doesn't decay — facts about a dead person
    don't get less true over time.
  - Temporary state (1d) for GAIA's own runtime state, sessions, etc.
  - Unknown (30d) is the fallback — long enough to survive a session,
    short enough that stale unknowns get refetched eventually.

These are seconds. `None` means "no decay" (biographical, identity,
etc.). Callers should treat that as `relevance = confidence`.
"""

from __future__ import annotations

from typing import Optional


# ── Fact-type constants (use these strings; don't pass raw strings) ──
WEATHER = "weather"
NEWS = "news"
MARKET_DATA = "market_data"
POLITICAL_OFFICE = "political_office"
SCIENTIFIC_CONSENSUS = "scientific_consensus"
BIOGRAPHICAL = "biographical"
TEMPORARY_STATE = "temporary_state"
UNKNOWN = "unknown"


# Affect/persona axes pinned at the same call site so the affect runtime
# can swap its placeholder decay onto Stage 7's kernel without inventing
# a parallel taxonomy. Same prefix names as affect_predicates.HALFLIFE_BY_PREFIX
# so the swap is mechanical.
AFFECT_FEELS = "affect_feels"
AFFECT_TRAIT = "affect_trait"
AFFECT_DRIVE = "affect_drive"
AFFECT_CURIOUS_ABOUT = "affect_curious_about"
AFFECT_TIRED_OF = "affect_tired_of"
AFFECT_BELIEVES_ABOUT = "affect_believes_about"


# ── Half-life table (seconds; None = no decay) ──────────────────────
_HOUR = 3600
_DAY = 24 * _HOUR
_YEAR = 365 * _DAY

HALFLIFE: dict[str, Optional[float]] = {
    WEATHER:              6 * _HOUR,
    NEWS:                 7 * _DAY,
    MARKET_DATA:          1 * _HOUR,
    POLITICAL_OFFICE:     1 * _YEAR,
    SCIENTIFIC_CONSENSUS: 50 * _YEAR,
    BIOGRAPHICAL:         None,        # no decay
    TEMPORARY_STATE:      1 * _DAY,
    UNKNOWN:              30 * _DAY,

    AFFECT_FEELS:         12 * _HOUR,
    AFFECT_DRIVE:         1 * _DAY,
    AFFECT_CURIOUS_ABOUT: 18 * _HOUR,
    AFFECT_TIRED_OF:      18 * _HOUR,
    AFFECT_TRAIT:         7 * _DAY,
    AFFECT_BELIEVES_ABOUT: 3 * _DAY,
}


VALID_FACT_TYPES: frozenset[str] = frozenset(HALFLIFE.keys())


def halflife_seconds(fact_type: Optional[str]) -> Optional[float]:
    """Return half-life for a fact_type. None for no-decay or unknown→default.

    A `None` fact_type or one not in the ontology falls through to UNKNOWN
    (30-day half-life). Pass BIOGRAPHICAL explicitly to disable decay.
    """
    if fact_type is None:
        return HALFLIFE[UNKNOWN]
    return HALFLIFE.get(fact_type, HALFLIFE[UNKNOWN])


def is_valid(fact_type: Optional[str]) -> bool:
    """True iff fact_type is in the ontology. None counts as invalid."""
    return fact_type is not None and fact_type in VALID_FACT_TYPES


# ── Heuristic classification by predicate name ──────────────────────
# Stage 7 uses a heuristic (not an LLM) to classify search-ingested
# triples. The mapping is by suffix/keyword on the predicate string;
# callers can override by passing fact_type= explicitly to add_triple().

_PREDICATE_HEURISTICS: list[tuple[str, str]] = [
    # (substring, fact_type) — first match wins
    ("weather",          WEATHER),
    ("temperature",      WEATHER),
    ("price",            MARKET_DATA),
    ("ticker",           MARKET_DATA),
    ("stock",            MARKET_DATA),
    ("market_cap",       MARKET_DATA),
    ("senator",          POLITICAL_OFFICE),
    ("president",        POLITICAL_OFFICE),
    ("governor",         POLITICAL_OFFICE),
    ("ceo",              POLITICAL_OFFICE),
    ("mayor",            POLITICAL_OFFICE),
    ("won_election",     POLITICAL_OFFICE),
    ("died_in",          BIOGRAPHICAL),
    ("born_in",          BIOGRAPHICAL),
    ("birthplace",       BIOGRAPHICAL),
    ("deathplace",       BIOGRAPHICAL),
    ("nationality",      BIOGRAPHICAL),
    ("authored",         BIOGRAPHICAL),
    ("invented",         BIOGRAPHICAL),
    ("speed_of_light",   SCIENTIFIC_CONSENSUS),
    ("gravitational",    SCIENTIFIC_CONSENSUS),
    ("law_of_",          SCIENTIFIC_CONSENSUS),
    ("constant",         SCIENTIFIC_CONSENSUS),
    ("breaking",         NEWS),
    ("headline",         NEWS),
    ("announced",        NEWS),
    ("released",         NEWS),
]


def classify_predicate(predicate: str) -> str:
    """Best-guess fact_type for a predicate string. Defaults to NEWS for
    search-ingested predicates that don't match any heuristic — most
    web-search content is news-shaped. Use UNKNOWN explicitly if you
    don't want news-class decay.
    """
    if not predicate:
        return UNKNOWN
    p = predicate.lower()
    for needle, ft in _PREDICATE_HEURISTICS:
        if needle in p:
            return ft
    return NEWS
