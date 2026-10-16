#!/usr/bin/env bash
#
# bootstrap_install.sh — ONE-TIME privileged setup for gaia-ops-broker.
#
# Run this yourself with sudo. Claude Code has no root access and cannot run
# this for you — that's by design (see the approved plan). It:
#   1. Creates the gaia_admin system account + gaia-ops group (attribution
#      only — nologin, no password, never a login target).
#   2. Installs the two wrapper scripts to /usr/local/bin, root:root 0755.
#   3. Validates and installs the sudoers drop-in (visudo -cf first).
#   4. Creates /var/log/gaia-ops.log, root:gaia-ops 0640.
#   5. Installs + enables the systemd unit for the broker service.
#
# Safe to re-run — every step is idempotent.
#
# Usage: sudo bash bootstrap_install.sh
#
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
    echo "Run this with sudo: sudo bash $0" >&2
    exit 1
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REAL_USER="${SUDO_USER:-azrael}"

echo "== 1. gaia_admin account + gaia-ops group =="
if ! getent group gaia-ops >/dev/null; then
    groupadd --system gaia-ops
    echo "  created group gaia-ops"
else
    echo "  group gaia-ops already exists"
fi

if ! id gaia_admin >/dev/null 2>&1; then
    useradd --system --shell /usr/sbin/nologin --no-create-home \
        --gid gaia-ops --comment "GAIA ops attribution account (no login)" gaia_admin
    passwd -l gaia_admin >/dev/null
    echo "  created locked, nologin user gaia_admin"
else
    echo "  user gaia_admin already exists"
fi

if ! id -nG "$REAL_USER" | tr ' ' '\n' | grep -qx gaia-ops; then
    usermod -aG gaia-ops "$REAL_USER"
    echo "  added $REAL_USER to gaia-ops group (can read /var/log/gaia-ops.log)"
fi

echo "== 2. Wrapper scripts -> /usr/local/bin =="
install -o root -g root -m 0755 "$HERE/gaia-ops-exec" /usr/local/bin/gaia-ops-exec
install -o root -g root -m 0755 "$HERE/gaia-emergency-exec" /usr/local/bin/gaia-emergency-exec
echo "  installed gaia-ops-exec, gaia-emergency-exec"

echo "== 3. sudoers drop-in =="
TMP_SUDOERS="$(mktemp)"
trap 'rm -f "$TMP_SUDOERS"' EXIT
sed "s/^azrael /${REAL_USER} /" "$HERE/gaia-ops.sudoers" > "$TMP_SUDOERS"
if visudo -cf "$TMP_SUDOERS"; then
    install -o root -g root -m 0440 "$TMP_SUDOERS" /etc/sudoers.d/gaia-ops
    echo "  installed /etc/sudoers.d/gaia-ops (validated)"
else
    echo "  REFUSING to install sudoers file — visudo validation failed" >&2
    exit 1
fi

echo "== 4. Audit log =="
touch /var/log/gaia-ops.log
chown root:gaia-ops /var/log/gaia-ops.log
chmod 0640 /var/log/gaia-ops.log
echo "  /var/log/gaia-ops.log ready"

echo "== 5. systemd unit =="
SERVICE_SRC="$HERE/gaia-ops-broker.service"
SERVICE_DST="/etc/systemd/system/gaia-ops-broker.service"
sed "s/^User=azrael/User=${REAL_USER}/; s/^Group=azrael/Group=${REAL_USER}/" "$SERVICE_SRC" > "$SERVICE_DST"
chmod 0644 "$SERVICE_DST"
systemctl daemon-reload
systemctl enable gaia-ops-broker.service
echo "  installed + enabled gaia-ops-broker.service (not started yet — see below)"

echo ""
echo "== Bootstrap complete. Remaining manual steps before starting the service: =="
echo ""
echo "1. Create a SEPARATE Discord bot application (not GAIA's own bot):"
echo "   https://discord.com/developers/applications -> New Application"
echo "   -> Bot tab -> Reset Token, copy it"
echo "   -> Bot tab -> enable 'MESSAGE CONTENT INTENT' (required)"
echo "   -> OAuth2 -> URL Generator -> scope 'bot' -> invite it to a server"
echo "      you share with it (needed for it to DM you)"
echo ""
echo "2. Save the bot token (mode 600, your own user, not root):"
echo "   mkdir -p ~/.gaia/secrets && chmod 700 ~/.gaia/secrets"
echo "   echo 'YOUR_BOT_TOKEN' > ~/.gaia/secrets/claude_ops_bot_token"
echo "   chmod 600 ~/.gaia/secrets/claude_ops_bot_token"
echo ""
echo "3. Save your own Discord user ID (Settings -> Advanced -> enable Developer"
echo "   Mode, then right-click your name -> Copy User ID):"
echo "   echo 'YOUR_DISCORD_USER_ID' > ~/.gaia/secrets/ops_approver_id"
echo ""
echo "4. Enroll a TOTP secret (this is the actual approval credential --"
echo "   independent of Discord entirely) and add it to your authenticator app:"
echo "   cd $HERE && .venv/bin/python broker.py --enroll"
echo ""
echo "5. Start the service:"
echo "   sudo systemctl start gaia-ops-broker.service"
echo "   sudo systemctl status gaia-ops-broker.service"
echo "   journalctl -u gaia-ops-broker -f    # watch it connect"
