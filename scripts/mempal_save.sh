#!/bin/bash
# mempal_save.sh — Save current session history to the Developer Palace
# Usage: ./mempal_save.sh "Short description of what was done"

DESCRIPTION=$1
PALACE_DIR="/gaia/GAIA_Project/knowledge/mempalace_dev"
VENV_PATH="/gaia/GAIA_Project/.venv"
SESSION_FILE="/tmp/mempal_current_session.md"

if [ -z "$DESCRIPTION" ]; then
    echo "Usage: ./mempal_save.sh \"Description of work\""
    exit 1
fi

# 1. Capture the latest context from the Chamber and TODO
# (In a real scenario, this script would also capture the active agent's 
# transcript, but for now we'll sync the manual files)
cp /gaia/GAIA_Project/COUNCIL_CHAMBER.md /gaia/GAIA_Project/knowledge/mempalace_dev/inbox/
cp /gaia/GAIA_Project/knowledge/Dev_Notebook/TODO.md /gaia/GAIA_Project/knowledge/mempalace_dev/inbox/

# 2. Add a timestamped 'Action Drawer' entry
DATE_STR=$(date +"%Y-%m-%d %H:%M")
echo -e "# Session Record: $DATE_STR\n\n## Action\n$DESCRIPTION\n\n## Source\nManual sync of Council Chamber and TODO" > "$SESSION_FILE"
cp "$SESSION_FILE" /gaia/GAIA_Project/knowledge/mempalace_dev/inbox/session_$(date +%s).md

# 3. Mine the inbox
source "$VENV_PATH/bin/activate"
mempalace --palace "$PALACE_DIR" mine /gaia/GAIA_Project/knowledge/mempalace_dev/inbox --wing GAIA_Sovereignty

echo "✓ Session memory filed in GAIA_Sovereignty Palace."
