#!/usr/bin/env python3
"""
gaia-ops-broker — standalone Discord-DM approval gate for Claude Code's
privileged (T1/T0) host actions.

Deliberately NOT part of gaia-mcp or any GAIA container: this gates actions
taken by Claude Code (a different actor than GAIA itself), using a dedicated
Discord bot identity so approval DMs are never confused with GAIA's own
conversational bot. Mirrors the challenge/reversed-reply UX pattern from
gaia-mcp/gaia_mcp/approval.py's ApprovalStore, but is an independent
implementation — GAIA's own ApprovalStore Blast Shield permanently blocks
sudo/su/doas/pkexec for GAIA's own self-directed tool calls, and that line
must never be touched by or merged with this system.

Flow:
  1. Claude Code POSTs /request {"command", "tier": "T1"|"T0", "reason"}.
  2. Broker DMs the approver a 5-letter challenge via its own bot.
  3. Approver replies with the challenge REVERSED (or "cancel"/"deny").
  4. On match, broker writes a signed-by-possession session file that only
     the root-owned wrapper scripts (gaia-ops-exec / gaia-emergency-exec)
     read, valid for SESSION_TTL_SECONDS.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import signal
import string
import sys
import time
from pathlib import Path
from typing import Optional

import discord
from aiohttp import web

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] gaia-ops-broker: %(message)s",
)
log = logging.getLogger("gaia-ops-broker")

# ── Configuration (env-overridable for testing) ─────────────────────────────
BOT_TOKEN_FILE = Path(os.getenv("GAIA_OPS_BOT_TOKEN_FILE", os.path.expanduser("~/.gaia/secrets/claude_ops_bot_token")))
APPROVER_ID_FILE = Path(os.getenv("GAIA_OPS_APPROVER_ID_FILE", os.path.expanduser("~/.gaia/secrets/ops_approver_id")))
SESSION_FILE = Path(os.getenv("GAIA_OPS_SESSION_FILE", "/run/gaia-ops-broker/session.json"))
API_HOST = os.getenv("GAIA_OPS_API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("GAIA_OPS_API_PORT", "8899"))
CHALLENGE_TTL_SECONDS = int(os.getenv("GAIA_OPS_CHALLENGE_TTL", "300"))  # time to reply
SESSION_TTL_SECONDS = int(os.getenv("GAIA_OPS_SESSION_TTL", "900"))  # approved window (15 min)

VALID_TIERS = {"T1", "T0"}


def _read_text(path: Path) -> str:
    try:
        return path.read_text().strip()
    except FileNotFoundError:
        log.error("Required file missing: %s", path)
        sys.exit(1)


def _gen_challenge() -> str:
    """Same shape as gaia-mcp's ApprovalStore._gen_challenge (5 uppercase letters)."""
    return "".join(random.choice(string.ascii_uppercase) for _ in range(5))


class PendingApproval:
    """A single in-flight approval request. Only one at a time by design —
    this gates one operator's own actions, not a multi-tenant queue."""

    def __init__(self, command: str, tier: str, reason: str):
        self.command = command
        self.tier = tier
        self.reason = reason
        self.challenge = _gen_challenge()
        self.created_at = time.time()
        self.expiry = self.created_at + CHALLENGE_TTL_SECONDS
        self.event = asyncio.Event()
        self.result: Optional[bool] = None  # True=approved, False=denied/expired

    def is_expired(self) -> bool:
        return time.time() > self.expiry


class OpsBroker:
    def __init__(self, bot_token: str, approver_id: int):
        self.approver_id = approver_id
        self.pending: Optional[PendingApproval] = None

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

        expected = pending.challenge[::-1]
        if content.upper() != expected:
            await message.channel.send(
                f"That's not right. Challenge is `{pending.challenge}` — reply with it **reversed** "
                f"(e.g. `{expected}`), or say **cancel**."
            )
            return

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

        user = await self.client.fetch_user(self.approver_id)
        tier_banner = "EMERGENCY ROOT ACCESS REQUESTED" if tier == "T0" else "T1 Ops Approval Requested"
        warning_prefix = "⚠️ " if tier == "T0" else ""
        prompt = (
            f"**{warning_prefix}{tier_banner}**\n\n"
            f"Command: `{command}`\n"
            + (f"Reason: {reason}\n" if reason else "")
            + f"\n**Challenge:** `{pending.challenge}`\n"
            f"Reply with it **reversed** to approve, or say **cancel** to deny.\n"
            f"_Expires in {CHALLENGE_TTL_SECONDS // 60} minutes._"
        )
        await user.send(prompt)
        log.info("Sent approval DM: tier=%s command=%s", tier, command)

        try:
            await asyncio.wait_for(pending.event.wait(), timeout=CHALLENGE_TTL_SECONDS + 5)
        except asyncio.TimeoutError:
            return {"approved": False, "error": "timed out waiting for reply"}

        if pending.result:
            return {"approved": True}
        return {"approved": False, "error": "denied"}

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


async def main():
    bot_token = _read_text(BOT_TOKEN_FILE)
    approver_id = int(_read_text(APPROVER_ID_FILE))

    broker = OpsBroker(bot_token, approver_id)
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
    asyncio.run(main())
