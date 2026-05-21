# Dev Journal Automation Plan

## Objective
Automate the creation and upload of Dev Journal entries triggered by git commits. The automation will read the transcript of a Claude session running in a specific tmux pane, analyze the git diff, generate a markdown journal entry using Gemini, and upload it to the `GAIA_Development` notebook via NotebookLM.

## Context
- **Trigger**: Git `post-commit` hook.
- **Source Data**:
  - Tmux Pane: `gaia_development:development.1` (Claude session transcript).
  - Git: `git diff HEAD~1 HEAD` (recent changes).
- **Generator**: Gemini CLI (used programmatically).
- **Destination (Local)**: `knowledge/Dev_Notebook/YYYY-MM-DD_<title>.md`.
- **Destination (Remote)**: Google NotebookLM notebook named `GAIA_Development`.

## Implementation Steps

### 1. Create the Automation Script
Create `scripts/automate_dev_journal.py` to handle the core logic:
- **Capture Transcript**: Execute `tmux capture-pane -t gaia_development:development.1 -p -S -2000` to get the recent Claude conversation.
- **Capture Diff**: Execute `git diff HEAD~1 HEAD` to get the context of the commit.
- **Generate Content**: Format a prompt combining the transcript and diff, and pipe it to the `gemini` command line tool to generate the Dev Journal entry in Markdown format.
- **Save File**: Parse a title from the generated Markdown, format the current date, and save the file to `knowledge/Dev_Notebook/`.
- **Upload to NotebookLM**: Use the `notebooklm` Python library (from `venv_notebooklm`) to authenticate, locate the `GAIA_Development` notebook, and upload the newly created `.md` file as a source.

### 2. Update `post-commit` Hook
Modify `.git/hooks/post-commit` to execute the Python script:
- Trigger the script asynchronously (e.g., using `nohup` or `&`) so it does not block the user's `git commit` workflow.
- Ensure the script runs with the `venv_notebooklm` Python executable to have access to the necessary libraries.
- Log output to `logs/dev_journal_automation.log` for debugging.

## Verification
- Perform a test commit to verify the `post-commit` hook triggers.
- Check `logs/dev_journal_automation.log` for successful execution.
- Verify a new `.md` file is created in `knowledge/Dev_Notebook/`.
- (Manual) Verify the file appears in the `GAIA_Development` NotebookLM instance.
