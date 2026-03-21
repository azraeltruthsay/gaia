#!/usr/bin/env python3
"""
generate_journal_site.py — Generate a static HTML site from dev journal markdown files.

Reads all .md files from knowledge/Dev_Notebook/, generates a simple index page
and copies the markdown files for static serving. Output goes to /tmp/journal-site/
for the GitHub Pages workflow to pick up.

This is a lightweight generator — no MkDocs dependency. Produces plain HTML
with minimal styling that matches the blog's dark theme.
"""

import re
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_DIR = PROJECT_ROOT / "knowledge" / "Dev_Notebook"
OUTPUT_DIR = Path("/tmp/journal-site")

# Minimal dark theme CSS matching the blog
CSS = """
body { background: #0f1b23; color: #b0c4ce; font-family: -apple-system, system-ui, sans-serif; max-width: 900px; margin: 0 auto; padding: 2rem; line-height: 1.6; }
a { color: #4db8d1; text-decoration: none; } a:hover { color: #8fd4a8; text-decoration: underline; }
h1 { color: #4db8d1; border-bottom: 1px solid rgba(77,184,209,0.3); padding-bottom: 0.5rem; }
h2 { color: #5cb87a; } h3 { color: #7dd3e8; }
pre { background: #071520; padding: 1rem; border-radius: 8px; overflow-x: auto; }
code { background: rgba(13,59,84,0.3); color: #b8e6f0; padding: 0.2em 0.4em; border-radius: 4px; font-size: 0.9em; }
pre code { padding: 0; background: none; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
th, td { border: 1px solid rgba(77,184,209,0.2); padding: 0.5rem; text-align: left; }
th { background: #0a1520; }
.journal-entry { border: 1px solid rgba(77,184,209,0.2); border-radius: 8px; padding: 1rem; margin: 0.5rem 0; }
.journal-entry:hover { border-color: #5cb87a; }
.journal-date { color: #7dd3e8; font-size: 0.85em; }
.journal-title { font-size: 1.1em; }
.nav { margin-bottom: 2rem; padding: 1rem 0; border-bottom: 1px solid rgba(77,184,209,0.3); }
.nav a { margin-right: 1rem; color: #4db8d1; }
.badge { display: inline-block; padding: 0.1em 0.5em; border-radius: 12px; font-size: 0.8em; }
.badge-dev { background: rgba(77,184,209,0.15); color: #4db8d1; }
.badge-promo { background: rgba(92,184,122,0.15); color: #5cb87a; }
.badge-plan { background: rgba(125,211,232,0.15); color: #7dd3e8; }
"""


def parse_journal_entries() -> list[dict]:
    """Parse all journal files and return sorted entries."""
    entries = []
    for md_file in sorted(NOTEBOOK_DIR.glob("*.md"), reverse=True):
        name = md_file.stem
        # Extract date from filename (YYYY-MM-DD_*)
        date_match = re.match(r"(\d{4}-\d{2}-\d{2})_(.*)", name)
        if date_match:
            date_str = date_match.group(1)
            title_slug = date_match.group(2)
        else:
            date_str = "unknown"
            title_slug = name

        # Determine badge from filename
        if "promotion" in title_slug:
            badge = "promo"
        elif "plan" in title_slug:
            badge = "plan"
        else:
            badge = "dev"

        # Clean up title
        title = title_slug.replace("_", " ").title()

        entries.append({
            "filename": md_file.name,
            "date": date_str,
            "title": title,
            "badge": badge,
            "path": md_file,
        })

    return entries


def generate_index(entries: list[dict]) -> str:
    """Generate the index.html page."""
    rows = []
    current_month = ""
    for entry in entries:
        # Month header
        try:
            dt = datetime.strptime(entry["date"], "%Y-%m-%d")
            month = dt.strftime("%B %Y")
        except ValueError:
            month = "Unknown"

        if month != current_month:
            current_month = month
            rows.append(f'<h2>{month}</h2>')

        badge_html = f'<span class="badge badge-{entry["badge"]}">{entry["badge"]}</span>'
        rows.append(
            f'<div class="journal-entry">'
            f'  <span class="journal-date">{entry["date"]}</span> {badge_html}<br>'
            f'  <a class="journal-title" href="{entry["filename"]}">{entry["title"]}</a>'
            f'</div>'
        )

    entries_html = "\n".join(rows)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GAIA Dev Journal</title>
<style>{CSS}</style>
</head>
<body>
<div class="nav">
  <a href="../">Blog</a>
  <a href="../wiki/">Wiki</a>
  <a href="./">Journal</a>
</div>
<h1>GAIA Dev Journal</h1>
<p>Development notebooks from the GAIA project. Raw engineering notes, plans, and session journals.</p>
<p style="color:#6c7086;font-size:0.85em;">{len(entries)} entries</p>
{entries_html}
</body>
</html>"""


def main():
    entries = parse_journal_entries()
    if not entries:
        print("No journal entries found")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Write index
    index_html = generate_index(entries)
    (OUTPUT_DIR / "index.html").write_text(index_html)

    # Copy markdown files (served as raw text or rendered by browser)
    for entry in entries:
        src = entry["path"]
        dst = OUTPUT_DIR / src.name
        dst.write_text(src.read_text())

    print(f"Generated journal site: {len(entries)} entries -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
