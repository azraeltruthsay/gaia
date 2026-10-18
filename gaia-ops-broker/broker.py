#!/usr/bin/env python3
"""
gaia-ops-broker — standalone Discord-DM approval gate for Claude Code's
privileged (T1/T0) host actions.

Deliberately NOT part of gaia-mcp or any GAIA container: this gates actions
taken by Claude Code (a different actor than GAIA itself), using a dedicated
Discord bot identity so approval DMs are never confused with GAIA's own
conversational bot. The notification/transport shape (DM the approver, wait
for a reply, open a time-boxed session) echoes gaia-mcp/gaia_mcp/approval.py's
ApprovalStore, but is an independent implementation — GAIA's own
ApprovalStore Blast Shield permanently blocks sudo/su/doas/pkexec for GAIA's
own self-directed tool calls, and that line must never be touched by or
merged with this system.

The credential is real TOTP (pyotp, same algorithm as Google
Authenticator/Authy), not a reversed-string challenge: an early version of
this broker asked the approver to reply with a server-generated challenge
spelled backwards, which is zero cryptographic value — anyone who can read
the Discord DM can also reply correctly, so it collapsed to ONE factor
(control of the Discord account) dressed up as two. TOTP's secret lives only
on the approver's phone, seeded once, independent of Discord entirely — a
compromised Discord account alone grants an attacker nothing.

This gate is not only (or even mainly) about hostile actors. It is a
deliberate human-in-the-loop friction point on an agent that is otherwise
trusted to act — the digital equivalent of the plastic cover over a fire
alarm pull: a second's pause and a second party's attention before a
privileged, hard-to-reverse action fires, independent of how much the agent
is trusted to act well. The tier split (T1 allow-listed / T0 unscoped) and
the session window exist so that pause can be dialed up or down by how
confident the action's blast radius warrants, not applied uniformly.

Flow:
  1. Claude Code POSTs /request {"command", "tier": "T1"|"T0", "reason"}.
  2. Broker DMs the approver via its own bot, describing the action.
  3. Approver replies with their current 6-digit TOTP code (or "cancel"/"deny").
  4. On a valid, unused code, broker writes a signed-by-possession session
     file that only the root-owned wrapper scripts (gaia-ops-exec /
     gaia-emergency-exec) read, valid for SESSION_TTL_SECONDS.

Run with --enroll once to generate a new TOTP secret and print the
provisioning URI/manual key for your authenticator app, before starting the
service for real.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import discord
import pyotp
from aiohttp import web

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] gaia-ops-broker: %(message)s",
)
log = logging.getLogger("gaia-ops-broker")

# ── Configuration (env-overridable for testing) ─────────────────────────────
BOT_TOKEN_FILE = Path(os.getenv("GAIA_OPS_BOT_TOKEN_FILE", os.path.expanduser("~/.gaia/secrets/claude_ops_bot_token")))
APPROVER_ID_FILE = Path(os.getenv("GAIA_OPS_APPROVER_ID_FILE", os.path.expanduser("~/.gaia/secrets/ops_approver_id")))
TOTP_SECRET_FILE = Path(os.getenv("GAIA_OPS_TOTP_SECRET_FILE", os.path.expanduser("~/.gaia/secrets/ops_totp_secret")))
SESSION_FILE = Path(os.getenv("GAIA_OPS_SESSION_FILE", "/run/gaia-ops-broker/session.json"))
API_HOST = os.getenv("GAIA_OPS_API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("GAIA_OPS_API_PORT", "8899"))
APPROVAL_TTL_SECONDS = int(os.getenv("GAIA_OPS_APPROVAL_TTL", "300"))  # time to reply
SESSION_TTL_SECONDS = int(os.getenv("GAIA_OPS_SESSION_TTL", "900"))  # approved window (15 min)

VALID_TIERS = {"T1", "T0"}


def _read_text(path: Path) -> str:
    try:
        return path.read_text().strip()
    except FileNotFoundError:
        log.error("Required file missing: %s", path)
        sys.exit(1)


class PendingApproval:
    """A single in-flight approval request. Only one at a time by design —
    this gates one operator's own actions, not a multi-tenant queue."""

    def __init__(self, command: str, tier: str, reason: str):
        self.command = command
        self.tier = tier
        self.reason = reason
        self.created_at = time.time()
        self.expiry = self.created_at + APPROVAL_TTL_SECONDS
        self.event = asyncio.Event()
        self.result: Optional[bool] = None  # True=approved, False=denied/expired

    def is_expired(self) -> bool:
        return time.time() > self.expiry


class OpsBroker:
    def __init__(self, bot_token: str, approver_id: int, totp_secret: str):
        self.approver_id = approver_id
        self.pending: Optional[PendingApproval] = None
        self.totp = pyotp.TOTP(totp_secret)
        self._last_used_code: Optional[str] = None
        self._last_used_at: float = 0.0

        intents = discord.Intents.default()
        intents.dm_messages = True
        intents.message_content = True  # must also be enabled in the Dev Portal
        self.client = discord.Client(intents=intents)
        self.bot_token = bot_token

        self.client.event(self.on_ready)
        self.client.event(self.on_message)

    async def on_ready(self):
        log.info("Connected to Discord as %s (approver_id=%s)", self.client.user, self.approver_id)

    async def on_message(self, message: discord.Message):
        if message.author.id == self.client.user.id:
            return
        if message.guild is not None:
            return  # DMs only
        if message.author.id != self.approver_id:
            log.warning("Ignored DM from non-approver id=%s", message.author.id)
            return

        pending = self.pending
        if pending is None or pending.event.is_set():
            return

        content = message.content.strip()
        normalized = content.lower()

        if normalized in ("cancel", "deny", "no", "abort"):
            pending.result = False
            pending.event.set()
            await message.channel.send(f"Cancelled. `{pending.command}` will not run.")
            log.info("Approval denied by user: %s", pending.command)
            return

        if pending.is_expired():
            pending.result = False
            pending.event.set()
            await message.channel.send("That request already expired. Ask again if you still want it to run.")
            return

        code = "".join(ch for ch in content if ch.isdigit())
        if not code or not self.totp.verify(code, valid_window=1):
            await message.channel.send(
                "That code didn't verify. Reply with your current 6-digit authenticator code, or say **cancel**."
            )
            return

        # Reject immediate reuse of the same code (it stays valid for ~30-60s
        # total, so without this a captured/observed code could be replayed
        # for a second request within that window).
        if code == self._last_used_code and (time.time() - self._last_used_at) < 90:
            await message.channel.send(
                "That code was already used for a previous approval. Wait for the next code and try again."
            )
            return
        self._last_used_code = code
        self._last_used_at = time.time()

        pending.result = True
        pending.event.set()
        self._write_session(pending.tier)
        await message.channel.send(
            f"Approved. `{pending.command}` ({pending.tier}) may now run — window open for "
            f"{SESSION_TTL_SECONDS // 60} minutes."
        )
        log.info("Approval granted: tier=%s command=%s", pending.tier, pending.command)

    def _write_session(self, tier: str):
        SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
        now = time.time()
        payload = {
            "tier": tier,
            "granted_at": now,
            "expires_at": now + SESSION_TTL_SECONDS,
        }
        tmp = SESSION_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload))
        os.chmod(tmp, 0o640)
        tmp.replace(SESSION_FILE)

    async def request_approval(self, command: str, tier: str, reason: str) -> dict:
        if tier not in VALID_TIERS:
            return {"approved": False, "error": f"invalid tier {tier!r}"}
        if self.pending is not None and not self.pending.event.is_set():
            return {"approved": False, "error": "another approval request is already pending"}

        pending = PendingApproval(command, tier, reason)
        self.pending = pending

        try:
            user = await self.client.fetch_user(self.approver_id)
            tier_banner = "EMERGENCY ROOT ACCESS REQUESTED" if tier == "T0" else "T1 Ops Approval Requested"
            warning_prefix = "⚠️ " if tier == "T0" else ""
            prompt = (
                f"**{warning_prefix}{tier_banner}**\n\n"
                f"Command: `{command}`\n"
                + (f"Reason: {reason}\n" if reason else "")
                + f"\nReply with your current **6-digit authenticator code** to approve, "
                f"or say **cancel** to deny.\n"
                f"_Expires in {APPROVAL_TTL_SECONDS // 60} minutes._"
            )
            await user.send(prompt)
            log.info("Sent approval DM: tier=%s command=%s", tier, command)

            try:
                await asyncio.wait_for(pending.event.wait(), timeout=APPROVAL_TTL_SECONDS + 5)
            except asyncio.TimeoutError:
                return {"approved": False, "error": "timed out waiting for reply"}

            if pending.result:
                return {"approved": True}
            return {"approved": False, "error": "denied"}
        except discord.Forbidden:
            log.error("Could not DM approver %s (no mutual guild, or DMs blocked)", self.approver_id)
            return {"approved": False, "error": "could not DM approver: no mutual Discord server, or DMs blocked"}
        finally:
            if self.pending is pending:
                self.pending = None

    def session_status(self) -> dict:
        try:
            data = json.loads(SESSION_FILE.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return {"valid": False}
        remaining = data.get("expires_at", 0) - time.time()
        return {"valid": remaining > 0, "remaining_seconds": max(0, int(remaining)), "tier": data.get("tier")}


async def make_app(broker: OpsBroker) -> web.Application:
    app = web.Application()

    async def handle_request(request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return web.json_response({"approved": False, "error": "invalid JSON body"}, status=400)

        command = str(body.get("command", "")).strip()
        tier = str(body.get("tier", "")).strip().upper()
        reason = str(body.get("reason", "")).strip()
        if not command:
            return web.json_response({"approved": False, "error": "command is required"}, status=400)

        result = await broker.request_approval(command, tier, reason)
        status = 200 if result.get("approved") else 409
        return web.json_response(result, status=status)

    async def handle_session(request: web.Request) -> web.Response:
        return web.json_response(broker.session_status())

    async def handle_health(request: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    app.router.add_post("/request", handle_request)
    app.router.add_get("/session", handle_session)
    app.router.add_get("/health", handle_health)
    return app


def enroll() -> None:
    """Generate a new TOTP secret and print how to add it to an authenticator
    app. Run this once before starting the service for real; re-running
    generates a NEW secret and invalidates the old one everywhere it was
    enrolled."""
    secret = pyotp.random_base32()
    TOTP_SECRET_FILE.parent.mkdir(parents=True, exist_ok=True)
    TOTP_SECRET_FILE.write_text(secret)
    os.chmod(TOTP_SECRET_FILE, 0o600)

    uri = pyotp.TOTP(secret).provisioning_uri(name="gaia-ops-broker", issuer_name="GAIA Ops Broker")
    print(f"Secret written to {TOTP_SECRET_FILE} (mode 600).")
    print()
    print("Add this to your authenticator app (Google Authenticator, Authy, etc.):")
    print(f"  Manual entry key: {secret}")
    print(f"  Provisioning URI: {uri}")
    print()
    print("Most apps can generate a scannable QR code from that URI if you paste")
    print("it into any online QR generator — just don't leave it sitting in your")
    print("clipboard/browser history any longer than you have to.")


async def main():
    bot_token = _read_text(BOT_TOKEN_FILE)
    approver_id = int(_read_text(APPROVER_ID_FILE))
    try:
        totp_secret = TOTP_SECRET_FILE.read_text().strip()
    except FileNotFoundError:
        print(f"No TOTP secret at {TOTP_SECRET_FILE} — run 'python broker.py --enroll' first.", file=sys.stderr)
        sys.exit(1)
    if not totp_secret:
        print(f"TOTP secret file {TOTP_SECRET_FILE} is empty — run 'python broker.py --enroll' again.", file=sys.stderr)
        sys.exit(1)

    broker = OpsBroker(bot_token, approver_id, totp_secret)
    app = await make_app(broker)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, API_HOST, API_PORT)
    await site.start()
    log.info("Local API listening on %s:%d", API_HOST, API_PORT)

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    try:
        loop.add_signal_handler(signal.SIGTERM, stop_event.set)
        loop.add_signal_handler(signal.SIGINT, stop_event.set)
    except (NotImplementedError, RuntimeError):
        pass  # platform without signal handler support in this loop — rely on KeyboardInterrupt

    bot_task = asyncio.create_task(broker.client.start(bot_token))
    stop_task = asyncio.create_task(stop_event.wait())

    await asyncio.wait({bot_task, stop_task}, return_when=asyncio.FIRST_COMPLETED)

    await runner.cleanup()
    if not broker.client.is_closed():
        await broker.client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enroll", action="store_true", help="Generate a new TOTP secret and exit")
    args = parser.parse_args()

    if args.enroll:
        enroll()
    else:
        asyncio.run(main())
