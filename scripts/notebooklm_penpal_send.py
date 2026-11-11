#!/usr/bin/env python3
"""Send GAIA's penpal letter to the notebook and ask the hosts for the next episode.

The letter goes in as a text source (a note isn't read by the Audio Overview;
a source is), then an Audio Overview is started with her request as its
instructions. Waits for it and downloads it.

    ~/.venvs/notebooklm/bin/python scripts/notebooklm_penpal_send.py \\
        --letter knowledge/transcripts/2026-11-11_S2E4_GAIA_Penpal_Response.txt \\
        --replying-to "Season 2, Episode 4" --requesting "Season 2, Episode 5" \\
        --download knowledge/transcripts/2026-11-11_S2E5_hosts_episode.m4a

Needs notebooklm-py >= 0.8 (notebook.google.com) and a signed-in
~/.notebooklm/storage_state.json.
"""

import argparse
import asyncio
import re
from pathlib import Path

from notebooklm import NotebookLMClient

NOTEBOOK_NAME = "GAIA Codebase"


def request_of(letter: str) -> str:
    """The letter's '## Request for Episode N' section, without the sign-off."""
    m = re.search(r"^## Request for Episode.*?$(.*?)(?:^---\s*$|\Z)", letter, re.M | re.S)
    return (m.group(1) if m else letter[-1500:]).strip()


async def send(args) -> int:
    letter = Path(args.letter).read_text().strip()
    title = f"GAIA's letter: reply to {args.replying_to}, requesting {args.requesting}"
    instructions = (
        f"This is {args.requesting} of the GAIA Pen Pal series. GAIA has written back to you about "
        f"{args.replying_to}; her letter is the source titled \"{title}\". Respond to her letter, "
        f"and take up what she asks for:\n\n{request_of(letter)}\n\nThe dev journal of 2026-11-11 "
        "(in the dev notebook source) says what has changed in her since the last episode.")
    if args.dry_run:
        print(f"source: {title} ({len(letter)} chars)\n\ninstructions ({len(instructions)} chars):\n{instructions}")
        return 0

    async with NotebookLMClient.from_storage() as client:
        nb = next((n for n in await client.notebooks.list() if n.title == NOTEBOOK_NAME), None)
        if not nb:
            print(f"No notebook titled {NOTEBOOK_NAME!r}")
            return 1
        existing = [s for s in await client.sources.list(nb.id) if s.title == title]
        if existing:
            print(f"Letter already in the notebook ({existing[0].id}); not adding it twice.")
        else:
            src = await client.sources.add_text(nb.id, title, letter)
            await client.sources.wait_until_ready(nb.id, src.id, timeout=300)
            print(f"Letter added as a source: {src.id}")

        status = await client.artifacts.generate_audio(nb.id, instructions=instructions)
        print(f"Episode requested: {status}")
        task_id = status.task_id
        if not (args.download and task_id):
            return 0
        done = await client.artifacts.wait_for_completion(nb.id, task_id, timeout=args.wait)
        print(f"Generation finished: {done}")
        if done.error:
            print(f"Generation failed: {done.error}")
            return 1
        out = await client.artifacts.download_audio(nb.id, args.download, artifact_id=task_id)
        print(f"Downloaded: {out}")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--letter", required=True)
    p.add_argument("--replying-to", required=True)
    p.add_argument("--requesting", required=True)
    p.add_argument("--download", default="", help="save the finished episode here")
    p.add_argument("--wait", type=float, default=1800.0, help="seconds to wait for the episode")
    p.add_argument("--dry-run", action="store_true", help="print the source and instructions only")
    raise SystemExit(asyncio.run(send(p.parse_args())))


if __name__ == "__main__":
    main()
