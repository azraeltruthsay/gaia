#!/bin/bash
#
# generate_changelog.sh — Generate CHANGELOG.md from logs/changelog.jsonl
#
# Usage: ./scripts/generate_changelog.sh [output_path]
#

set -euo pipefail

GAIA_ROOT="/gaia/GAIA_Project"
JSONL="${GAIA_ROOT}/logs/changelog.jsonl"
OUTPUT="${1:-${GAIA_ROOT}/CHANGELOG.md}"

if [ ! -f "$JSONL" ]; then
    echo "No changelog.jsonl found at $JSONL"
    exit 0
fi

# Generate markdown grouped by date
python3 -c "
import json, sys
from collections import defaultdict

entries = []
with open('$JSONL') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue

# Group by date (newest first)
by_date = defaultdict(list)
for e in entries:
    date = e.get('timestamp', '')[:10]
    by_date[date].append(e)

type_icons = {
    'feat': '+', 'fix': '!', 'refactor': '~', 'docs': '#',
    'promote': '^', 'config': '%', 'manual': '*',
}

print('# CHANGELOG')
print()
print('> Auto-generated from \`logs/changelog.jsonl\`')
print()

for date in sorted(by_date.keys(), reverse=True):
    print(f'## {date}')
    print()
    for e in reversed(by_date[date]):
        icon = type_icons.get(e.get('type', 'manual'), '*')
        svc = e.get('service', '?')
        summary = e.get('summary', '')
        author = e.get('author', '')
        commit = e.get('commit_hash', '')
        prefix = f'[{icon} {e.get(\"type\",\"?\")}]'
        commit_ref = f' ({commit[:7]})' if commit else ''
        print(f'- **{prefix}** \`{svc}\` — {summary}{commit_ref}')
    print()
" > "$OUTPUT"

echo "Generated $OUTPUT ($(wc -l < "$OUTPUT") lines)"
