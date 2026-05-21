#!/usr/bin/env python3
import subprocess
import os
import datetime
import json
import sys
import re
import asyncio
import time
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DEV_NOTEBOOK_DIR = PROJECT_ROOT / "knowledge" / "Dev_Notebook"
LOG_FILE = PROJECT_ROOT / "logs" / "dev_journal_automation.log"
VENV_PYTHON = PROJECT_ROOT / "venv_notebooklm" / "bin" / "python"

# Config
# API key is read from the GEMINI_API_KEY env var. Real secret lives at
# /gaia/gaia-instance/secrets/env.gemini (mode 600) and is sourced by the
# post-commit hook before invoking this script. Never inline the key here.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    # Allow the script to be importable for tests; main() will abort.
    sys.stderr.write(
        "WARN: GEMINI_API_KEY not set — automation will abort. "
        "Source .env.gemini before invoking this script.\n"
    )
NOTEBOOK_NAME = "GAIA_Development"

# Tmux Target
# Where the AI dev-session transcript lives. Override via env if the tmux
# layout changes (window renames, pane reorder, etc.). The default points
# at the current Gemini CLI pane in the gaia_development session.
#
# Quick reference:
#   tmux list-panes -s -t gaia_development
#   tmux capture-pane -t <target> -p | tail -5   # to verify before setting
TMUX_TARGET = os.getenv("DEV_JOURNAL_TMUX_TARGET", "gaia_development:1.2")

def log(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = f"[{timestamp}] {message}"
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

def get_tmux_transcript():
    try:
        log(f"Capturing tmux transcript from {TMUX_TARGET}...")
        result = subprocess.run(
            ["tmux", "capture-pane", "-t", TMUX_TARGET, "-p", "-S", "-4000"],
            capture_output=True, text=True, check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        log(f"Error capturing tmux pane {TMUX_TARGET}: {e.stderr.strip() if e.stderr else e}")
        log("Hint: set DEV_JOURNAL_TMUX_TARGET=session:window.pane to override.")
        return None

def get_git_diff():
    try:
        log("Getting git diff for the last commit...")
        result = subprocess.run(
            ["git", "diff", "HEAD~1", "HEAD"],
            capture_output=True, text=True, cwd=PROJECT_ROOT, check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        log(f"Error getting git diff: {e}")
        return "First commit or diff failed."

async def upload_to_notebooklm(filepath):
    log(f"Attempting upload to NotebookLM...")
    
    upload_script = f"""
import asyncio
import os
import sys
from pathlib import Path
from notebooklm import NotebookLMClient
from notebooklm.paths import get_storage_path, get_browser_profile_dir

async def main():
    storage_path = get_storage_path()
    browser_profile = get_browser_profile_dir()
    
    async def try_upload(client):
        notebooks = await client.notebooks.list()
        target = next((nb for nb in notebooks if nb.title == "{NOTEBOOK_NAME}"), None)
        if not target:
            print("Target notebook not found.")
            return False
        await client.sources.add_file(target.id, "{filepath}")
        return True

    # Try 1: Normal storage
    try:
        async with await NotebookLMClient.from_storage(path=storage_path) as client:
            if await try_upload(client):
                print("UPLOAD_OK")
                return
    except Exception as e:
        print(f"Initial attempt failed: {{e}}")

    # Try 2: Refresh auth
    print("Attempting headless auth refresh...")
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            context = p.chromium.launch_persistent_context(
                user_data_dir=str(browser_profile),
                headless=True,
                args=["--disable-blink-features=AutomationControlled", "--password-store=basic"],
                ignore_default_args=["--enable-automation"],
            )
            page = context.new_page()
            page.goto("https://notebooklm.google.com/", wait_until="domcontentloaded", timeout=30000)
            if "accounts.google.com" not in page.url:
                context.storage_state(path=str(storage_path))
                storage_path.chmod(0o600)
                print("Auth refreshed.")
            context.close()
    except Exception as re_err:
        print(f"Auth refresh failed: {{re_err}}")

    # Try 3: Re-attempt upload after refresh
    try:
        async with await NotebookLMClient.from_storage(path=storage_path) as client:
            if await try_upload(client):
                print("UPLOAD_OK")
                return
    except Exception as e:
        print(f"Final attempt failed: {{e}}")

if __name__ == '__main__':
    asyncio.run(main())
"""
    
    temp_script = PROJECT_ROOT / "scripts" / "tmp_upload_hardened.py"
    with open(temp_script, "w") as f:
        f.write(upload_script)
        
    try:
        # Note: We use the same venv for both generation (if possible) and upload
        result = subprocess.run(
            [str(VENV_PYTHON), str(temp_script)],
            capture_output=True, text=True, check=True
        )
        if "UPLOAD_OK" in result.stdout:
            log("NotebookLM upload successful.")
        else:
            log(f"Upload output: {result.stdout}")
    except subprocess.CalledProcessError as e:
        log(f"Upload process failed: {e.stderr}")
    finally:
        if temp_script.exists():
            temp_script.unlink()

async def async_main():
    if not LOG_FILE.parent.exists():
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    log("--- Starting Hardened Dev Journal Automation ---")

    if not GEMINI_API_KEY:
        log("ABORT: GEMINI_API_KEY env var is empty. Source .env.gemini.")
        return
    
    transcript = get_tmux_transcript()
    if not transcript:
        log("Aborting: Could not capture transcript.")
        return
        
    diff = get_git_diff()
    
    # We invoke generation via the venv's python (it has google-generativeai).
    # The key is passed via env (GEMINI_API_KEY) — NOT inlined into the temp
    # script. Inlining would leave the secret on disk if the unlink in the
    # finally block didn't run (e.g. crash, SIGKILL).
    gen_script = """
import asyncio
import google.generativeai as genai
import os
import sys

async def main():
    key = os.getenv("GEMINI_API_KEY", "")
    if not key:
        print("API_ERROR: GEMINI_API_KEY not in subprocess env")
        return
    genai.configure(api_key=key)
    model = genai.GenerativeModel('gemini-flash-latest')
    prompt = sys.stdin.read()
    try:
        response = await model.generate_content_async(prompt)
        print(response.text)
    except Exception as e:
        print(f"API_ERROR: {e}")

if __name__ == '__main__':
    asyncio.run(main())
"""
    temp_gen = PROJECT_ROOT / "scripts" / "tmp_gen.py"
    with open(temp_gen, "w") as f:
        f.write(gen_script)
        
    prompt = f"""
You are an expert technical writer for the GAIA project. Synthesize the following session and diff into a Dev Journal entry.

### Claude Session Transcript (Last part):
{transcript[-10000:]}

### Git Diff:
{diff[:5000]}

### Requirements:
1. Title: A concise # header.
2. Date: **Date**: YYYY-MM-DD
3. Sections: Use ## for headers like Context, Implementation, and Conclusion.
4. Style: Technical, analytical, and honest.
5. Output ONLY the markdown content.
"""
    
    try:
        # Pass GEMINI_API_KEY explicitly via env so the subprocess can read it
        # via os.getenv. Inheriting os.environ wholesale would also work but
        # passing an explicit minimal env makes the dependency obvious.
        _subprocess_env = dict(os.environ)
        _subprocess_env["GEMINI_API_KEY"] = GEMINI_API_KEY
        result = subprocess.run(
            [str(VENV_PYTHON), str(temp_gen)],
            input=prompt, capture_output=True, text=True, check=True,
            env=_subprocess_env,
        )
        journal_content = result.stdout.strip()
        if "API_ERROR" in journal_content:
            log(f"Generation error: {journal_content}")
            return
    except Exception as e:
        log(f"Subprocess generation failed: {e}")
        return
    finally:
        if temp_gen.exists():
            temp_gen.unlink()

    if not journal_content:
        log("Aborting: Generated content is empty.")
        return
        
    title_match = re.search(r'^#\s+(.+)$', journal_content, re.MULTILINE)
    safe_title = re.sub(r'[^a-z0-9]+', '_', title_match.group(1).lower()).strip('_') if title_match else "automated_entry"
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    filename = f"{date_str}_{safe_title}.md"
    filepath = DEV_NOTEBOOK_DIR / filename
    
    if not DEV_NOTEBOOK_DIR.exists():
        DEV_NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
        
    with open(filepath, "w") as f:
        f.write(journal_content)
    log(f"Saved journal entry: {filepath}")
    
    await upload_to_notebooklm(filepath)
    log("--- Hardened Dev Journal Automation Finished ---")

if __name__ == "__main__":
    asyncio.run(async_main())
