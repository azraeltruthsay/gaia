Review the Triple-Note Chord status: $ARGUMENTS

## Overview

The GAIA Chord is a three-way collaboration:
- **Architect (Azrael)** — Vision, intent, final approval
- **Advisor (Gemini)** — Research, strategy, design docs in Dev_Notebook
- **Engineer (Claude)** — Implementation, validation, production deployment

This skill synchronizes the Chord by reading the Council Chamber, capturing Gemini's latest session, and reporting what's changed.

## Steps

1. **Read the Council Chamber** — `/gaia/GAIA_Project/COUNCIL_CHAMBER.md`
   - Check the Approved Work Queue for new items or status changes
   - Check Active Logs for new AAAK entries from the Advisor
   - Note any items marked [IN PROGRESS] or [NEXT]

2. **Capture Gemini's tmux session** — try `gaia_gemini` first, fall back to `gaia_gemini2`:
   ```bash
   tmux capture-pane -t gaia_gemini -p -S -100 2>/dev/null || tmux capture-pane -t gaia_gemini2 -p -S -100 2>/dev/null
   ```
   - Summarize what Gemini is currently working on
   - Note any design docs or research she's produced

3. **Check Dev_Notebook for new entries** — look for files modified today:
   ```bash
   find /gaia/GAIA_Project/knowledge/Dev_Notebook/ -mtime -1 -name "*.md" | sort
   ```

4. **Report** — concise summary:
   - What's new in the Chamber since last check
   - What Gemini is doing right now
   - Any pending dispatches or approval requests for the Architect
   - Any items ready for the Engineer to implement

## After Review

Update the Engineer Status section of the Council Chamber with current progress. Keep Gemini informed so work isn't duplicated.
