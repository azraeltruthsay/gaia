#!/usr/bin/env python3
"""Kanka Campaign Cataloger — ingests Kanka world data into GAIA knowledge base.

Reads all entities from a Kanka campaign, writes structured markdown files
organized by entity type, and queues them for vector indexing.

Usage:
    python scripts/kanka_catalog.py --campaign 36156                   # Full catalog
    python scripts/kanka_catalog.py --campaign 36156 --type characters # Characters only
    python scripts/kanka_catalog.py --campaign 36156 --dry-run         # Preview only
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GAIA.Kanka.Catalog")

MCP_ENDPOINT = os.environ.get("MCP_ENDPOINT", "http://localhost:8765/jsonrpc")
OUTPUT_BASE = Path(os.environ.get("KNOWLEDGE_DIR", "knowledge")) / "dnd_campaign"

ENTITY_TYPES = [
    "characters", "locations", "organisations", "races", "families",
    "items", "journals", "quests", "creatures", "abilities",
]


def mcp_call(method: str, params: dict) -> dict:
    """Call a Kanka MCP tool."""
    payload = json.dumps({
        "jsonrpc": "2.0",
        "method": method,
        "params": params,
        "id": 1,
    }).encode()
    req = Request(MCP_ENDPOINT, data=payload,
                  headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=30) as resp:
        d = json.loads(resp.read().decode())
    if isinstance(d, list):
        d = d[0]
    return d.get("result", d)


def strip_html(text: str) -> str:
    """Remove HTML tags and clean up whitespace."""
    if not text:
        return ""
    clean = re.sub(r'<[^>]+>', '', text)
    clean = re.sub(r'\n\s*\n', '\n\n', clean)
    return clean.strip()


def fetch_all_entities(campaign_id: int, entity_type: str) -> list:
    """Fetch all entities of a type, handling pagination."""
    all_entities = []
    page = 1
    while True:
        result = mcp_call("kanka_list_entities", {
            "campaign_id": campaign_id,
            "entity_type": entity_type,
            "page": page,
        })
        entities = result.get("entities", [])
        if not entities:
            break
        all_entities.extend(entities)
        total = result.get("total", 0)
        if len(all_entities) >= total:
            break
        page += 1
        time.sleep(0.5)  # Rate limiting
    return all_entities


def fetch_entity_detail(campaign_id: int, entity_type: str, entity_id: int) -> dict:
    """Fetch full entity details including entry text."""
    result = mcp_call("kanka_get_entity", {
        "campaign_id": campaign_id,
        "entity_type": entity_type,
        "entity_id": entity_id,
        "related": True,
    })
    return result.get("entity", result)


def entity_to_markdown(entity: dict, entity_type: str) -> str:
    """Convert a Kanka entity to structured markdown."""
    name = entity.get("name", "Unknown")
    title = entity.get("title", "")
    entry = strip_html(entity.get("entry", "") or entity.get("entry_parsed", "") or "")
    etype = entity.get("type", "")

    lines = [f"# {name}"]
    if title:
        lines.append(f"*{title}*")
    lines.append("")

    # Metadata
    meta = []
    if etype:
        meta.append(f"**Type:** {etype}")
    if entity.get("age"):
        meta.append(f"**Age:** {entity['age']}")
    if entity.get("sex"):
        meta.append(f"**Sex:** {entity['sex']}")
    if entity.get("is_dead"):
        meta.append("**Status:** Dead")
    if entity.get("location_id"):
        meta.append(f"**Location ID:** {entity['location_id']}")
    if entity.get("races"):
        meta.append(f"**Race IDs:** {entity['races']}")
    if meta:
        lines.extend(meta)
        lines.append("")

    # Entry / backstory
    if entry:
        lines.append("## Description")
        lines.append(entry)
        lines.append("")

    # Tags
    tags = [f"kanka", entity_type.rstrip("s"), name.lower().replace(" ", "-")]
    lines.append(f"---")
    lines.append(f"Tags: {', '.join(tags)}")
    lines.append(f"Source: Kanka campaign, entity ID {entity.get('id', '?')}")
    lines.append(f"Last updated: {entity.get('updated_at', '?')}")

    return "\n".join(lines)


def catalog_campaign(campaign_id: int, entity_types: list, dry_run: bool = False):
    """Catalog all entities from a campaign into markdown files."""

    # Get campaign name
    campaigns = mcp_call("kanka_list_campaigns", {})
    campaign_name = "unknown"
    for c in campaigns.get("campaigns", []):
        if c["id"] == campaign_id:
            campaign_name = c["name"]
            break

    logger.info("Cataloging campaign: %s (ID: %d)", campaign_name, campaign_id)

    output_dir = OUTPUT_BASE / campaign_name.replace(" ", "_").lower()
    output_dir.mkdir(parents=True, exist_ok=True)

    total_entities = 0
    total_chars = 0

    for entity_type in entity_types:
        logger.info("Fetching %s...", entity_type)
        entities = fetch_all_entities(campaign_id, entity_type)
        logger.info("  Found %d %s", len(entities), entity_type)

        type_dir = output_dir / entity_type
        if not dry_run:
            type_dir.mkdir(parents=True, exist_ok=True)

        for i, entity in enumerate(entities):
            entity_id = entity.get("id")
            name = entity.get("name", "unknown")

            # Fetch full details
            try:
                detail = fetch_entity_detail(campaign_id, entity_type, entity_id)
                time.sleep(0.3)  # Rate limiting
            except Exception as e:
                logger.warning("  Failed to fetch %s/%d: %s", name, entity_id, e)
                detail = entity

            # Convert to markdown
            md = entity_to_markdown(detail, entity_type)
            total_chars += len(md)

            # Safe filename
            safe_name = re.sub(r'[^\w\-]', '_', name)[:80]
            filename = f"{safe_name}.md"

            if dry_run:
                if i < 3:
                    logger.info("  [DRY] %s (%d chars)", filename, len(md))
            else:
                (type_dir / filename).write_text(md, encoding="utf-8")
                if (i + 1) % 20 == 0:
                    logger.info("  Written %d/%d %s", i + 1, len(entities), entity_type)

            total_entities += 1

        if not dry_run:
            logger.info("  Written %d %s to %s", len(entities), entity_type, type_dir)

    # Write index
    if not dry_run:
        index_lines = [
            f"# {campaign_name} — Knowledge Index",
            f"",
            f"Cataloged from Kanka campaign ID {campaign_id}",
            f"Total entities: {total_entities}",
            f"Total content: {total_chars:,} characters",
            f"",
        ]
        for entity_type in entity_types:
            type_dir = output_dir / entity_type
            if type_dir.exists():
                count = len(list(type_dir.glob("*.md")))
                index_lines.append(f"- **{entity_type}**: {count} entries")
        index_lines.append(f"\n---\nGenerated: {time.strftime('%Y-%m-%d %H:%M UTC')}")
        (output_dir / "INDEX.md").write_text("\n".join(index_lines), encoding="utf-8")

    logger.info("Catalog complete: %d entities, %d chars", total_entities, total_chars)
    return {"entities": total_entities, "chars": total_chars, "output": str(output_dir)}


def main():
    parser = argparse.ArgumentParser(description="Kanka Campaign Cataloger")
    parser.add_argument("--campaign", type=int, required=True)
    parser.add_argument("--type", help="Single entity type to catalog")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    types = [args.type] if args.type else ENTITY_TYPES
    result = catalog_campaign(args.campaign, types, dry_run=args.dry_run)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
