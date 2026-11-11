#!/usr/bin/env python3
"""One pass of the flattened codebase up to the NotebookLM notebook, deleting
nothing it didn't put there.

notebooklm_sync.py's start-up pass deletes every source that isn't a file in
GAIA_Condensed_flat, which takes the season 2 sources added by hand (GAIA's
voiced letters) with it, and it never re-uploads a changed file whose name it
already has. This uploads new files, replaces changed ones (new copy first,
then the old one goes), and only lists the rest.

Run on the host, signed in (`notebooklm login`):
    ~/.venvs/notebooklm/bin/python scripts/notebooklm_push_once.py --dry-run
    ~/.venvs/notebooklm/bin/python scripts/notebooklm_push_once.py

Content hashes are kept in STATE_PATH, so the next pass replaces only what
changed; with no state yet, every file already in the notebook is replaced.
"""

import argparse
import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from notebooklm_sync import (  # noqa: E402
    NOTEBOOK_NAME, UPLOAD_RATE_LIMIT, WATCH_DIRECTORY, NotebookLMClient, upload_with_validation)

STATE_PATH = Path(WATCH_DIRECTORY).parent / "notebooklm_push_state.json"


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def push(dry_run: bool) -> int:
    local = {p.name: p for p in sorted(Path(WATCH_DIRECTORY).iterdir()) if p.is_file() and p.stat().st_size}
    hashes = {name: _hash(p) for name, p in local.items()}
    try:
        pushed = json.loads(STATE_PATH.read_text())
    except (OSError, ValueError):
        pushed = {}

    async with NotebookLMClient.from_storage() as client:
        notebook = next((nb for nb in await client.notebooks.list() if nb.title == NOTEBOOK_NAME), None)
        if not notebook:
            print(f"No notebook titled {NOTEBOOK_NAME!r}; nothing done.")
            return 1
        remote = await client.sources.list(notebook.id) or []
        by_title = {}
        for s in remote:
            by_title.setdefault(s.title, []).append(s)

        new = [n for n in local if n not in by_title]
        changed = [n for n in local if n in by_title and pushed.get(n) != hashes[n]]
        others = sorted(t for t in by_title if t not in local)
        print(f"Notebook {notebook.id}: {len(remote)} sources. Local flat files: {len(local)}.")
        print(f"  new: {len(new)}   changed or never pushed from here: {len(changed)}   "
              f"unchanged: {len(local) - len(new) - len(changed)}")
        print(f"  in the notebook, not flattened files (left alone): {len(others)}")
        for t in others:
            print(f"    - {t}")
        if dry_run:
            return 0

        failed = 0
        for name in new + changed:
            ok = await upload_with_validation(client, notebook.id, name, str(local[name]))
            if ok:
                for old in by_title.get(name, []):
                    try:
                        await client.sources.delete(notebook.id, old.id)
                    except Exception as e:  # the new copy is in; the old one is a duplicate
                        print(f"  could not remove the old copy of {name}: {e}")
                pushed[name] = hashes[name]
                STATE_PATH.write_text(json.dumps(pushed, indent=1, sort_keys=True))
                print(f"  {'replaced' if name in by_title else 'added'}: {name}")
            else:
                failed += 1
            await asyncio.sleep(UPLOAD_RATE_LIMIT)
        print(f"Done: {len(new) + len(changed) - failed} pushed, {failed} failed.")
        return 1 if failed else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="show what would change, touch nothing")
    sys.exit(asyncio.run(push(parser.parse_args().dry_run)))


if __name__ == "__main__":
    main()
