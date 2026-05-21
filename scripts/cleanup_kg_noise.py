#!/usr/bin/env python3
"""One-shot KG cleanup for World Model Stage 0 (hrp).

Removes the auto-extraction noise that accumulated before the
mempalace.py extractor was fixed:

  - Token-fragment entities: 'sup', 'bow', 'new', 'act' (and similar)
    — leftover from AAAK's first-3-chars truncation when feeding the
    KG, where codes like "SUP" leaked through as entity NAMES.

  - Topic-frequency 'related_to' triples — produced by the now-removed
    _extract_topics call site in MemPalace._extract_and_store_triples.

  - Palace-ID entities — synthetic '<date>_<hash>' subjects that aren't
    real entities (these were artifacts of the stored_in / memory_type
    triples that also got removed in the same fix).

Run-once after deploying the extractor fix. Safe to re-run; deletes
are idempotent.

Usage:
    docker exec gaia-mcp python3 /gaia/GAIA_Project/scripts/cleanup_kg_noise.py [--apply]

Default is dry-run (lists what would be deleted). Pass --apply to
actually delete.
"""

import argparse
import re
import sqlite3
import sys
from pathlib import Path


# Heuristics for noise identification.
# Entity name considered noise if any of:
#  - Length < 5 chars AND not in a small acronym allowlist
#  - Matches the palace_id shape (YYYY-MM-DD_<hex>)
#  - All-lowercase (real names are Title Case or ALL-CAPS acronyms)

_KEEP_SHORT = frozenset({
    "GAIA", "MCP", "GPU", "CPU", "RAM", "LLM",
    "API", "URL", "JSON", "HTTP", "HTTPS",
    "PST", "PDT", "UTC", "EST", "CST",
    "AI",
})

_PALACE_ID_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_[a-f0-9]{8,}$")


def looks_like_noise(entity_id: str, entity_name: str) -> tuple[bool, str]:
    """Return (is_noise, reason). Reason is a short tag for logging."""
    if _PALACE_ID_RE.match(entity_name or ""):
        return True, "palace_id"
    if _PALACE_ID_RE.match(entity_id or ""):
        return True, "palace_id"
    name = (entity_name or "").strip()
    if not name:
        return True, "empty_name"
    # Wing/room paths sneak in via the now-removed stored_in triples.
    if "/" in name and all(p.isalpha() for p in name.split("/")):
        return True, "wing_room_path"
    # Lowercase fragments like 'sup', 'bow', 'super'
    if name.islower() and len(name) < 8:
        return True, "lowercase_short"
    # Pure-alpha lowercase entities longer than 8 chars are also noise
    # (real lowercase entities are model IDs / compound names that
    # contain digits, dashes, or dots — 'qwen3.5-0.8b', 'gaia-core').
    if name.islower() and name.isalpha():
        return True, "lowercase_alpha"
    # 3-4 char ALL-CAPS that aren't real acronyms (SUP, BOW, NEW, ACT)
    if name.isupper() and len(name) < 5 and name not in _KEEP_SHORT:
        return True, "fragment_caps"
    # Mixed-case super-short (rare but possible)
    if len(name) < 4 and name not in _KEEP_SHORT:
        return True, "too_short"
    return False, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--apply", action="store_true",
        help="Actually delete. Default is dry-run.",
    )
    ap.add_argument(
        "--db", default="/shared/knowledge_graph/gaia_kg.sqlite3",
        help="Path to KG sqlite",
    )
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"ERROR: {db_path} does not exist", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    # Collect referenced-entity set first so we don't orphan a palace_id
    # that's still referenced as a (mentioned_in) triple object.
    referenced: "set[str]" = set()
    for row in cur.execute(
        "SELECT subject FROM triples "
        "WHERE predicate NOT IN ('stored_in', 'memory_type', 'related_to')"
    ):
        referenced.add(row[0])
    for row in cur.execute(
        "SELECT object FROM triples "
        "WHERE predicate NOT IN ('stored_in', 'memory_type', 'related_to')"
    ):
        referenced.add(row[0])

    # Find noise entities
    noise_ids = []
    keep_ids = []
    print("=== entity scan ===")
    for row in cur.execute("SELECT id, name FROM entities ORDER BY name"):
        eid, ename = row[0], row[1]
        is_noise, reason = looks_like_noise(eid, ename)
        # Preserve palace_id entities that are still referenced as
        # legitimate triple objects (mentioned_in points at them).
        if is_noise and reason == "palace_id" and eid in referenced:
            keep_ids.append((eid, ename))
            continue
        if is_noise:
            noise_ids.append((eid, ename, reason))
            print(f"  NOISE   [{reason:15s}] {eid:30s} name={ename!r}")
        else:
            keep_ids.append((eid, ename))
    print(f"\n  → {len(noise_ids)} noise entities, {len(keep_ids)} keep")

    # Find topic-frequency 'related_to' triples (the now-removed code path).
    # All such triples are noise — real semantic relationships should come
    # from a different mechanism.
    print("\n=== related_to triple scan ===")
    related_to_count = cur.execute(
        "SELECT COUNT(*) FROM triples WHERE predicate = 'related_to'"
    ).fetchone()[0]
    print(f"  → {related_to_count} 'related_to' triples (all noise)")

    # Find stored_in / memory_type triples (palace-id subjects, removed)
    print("\n=== palace-id triple scan ===")
    palace_triple_count = cur.execute(
        "SELECT COUNT(*) FROM triples WHERE predicate IN ('stored_in', 'memory_type')"
    ).fetchone()[0]
    print(f"  → {palace_triple_count} stored_in/memory_type triples (palace-id subjects)")

    # Triples involving noise entities (as subject or object)
    noise_id_set = {eid for eid, _, _ in noise_ids}
    triple_via_noise = 0
    for row in cur.execute("SELECT id, subject, predicate, object FROM triples"):
        if row[1] in noise_id_set or row[3] in noise_id_set:
            triple_via_noise += 1
    print(f"\n=== triples referencing noise entities: {triple_via_noise} ===")

    if not args.apply:
        print("\nDRY RUN — nothing deleted. Pass --apply to commit.")
        conn.close()
        return

    # Apply deletions
    print("\n=== applying deletions ===")
    cur.execute("DELETE FROM triples WHERE predicate = 'related_to'")
    print(f"  deleted {cur.rowcount} 'related_to' triples")
    cur.execute("DELETE FROM triples WHERE predicate IN ('stored_in', 'memory_type')")
    print(f"  deleted {cur.rowcount} stored_in/memory_type triples")
    if noise_id_set:
        placeholders = ",".join("?" for _ in noise_id_set)
        cur.execute(
            f"DELETE FROM triples WHERE subject IN ({placeholders}) "
            f"OR object IN ({placeholders})",
            list(noise_id_set) + list(noise_id_set),
        )
        print(f"  deleted {cur.rowcount} triples referencing noise entities")
        cur.execute(
            f"DELETE FROM entities WHERE id IN ({placeholders})",
            list(noise_id_set),
        )
        print(f"  deleted {cur.rowcount} noise entities")
    conn.commit()
    conn.close()
    print("\n  done.")


if __name__ == "__main__":
    main()
