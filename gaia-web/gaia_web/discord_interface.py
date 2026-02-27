"""
Discord Interface for GAIA Web Gateway

This module handles Discord bot integration as part of the Unified Interface Gateway.
Discord messages are received here, converted to CognitionPackets, sent to gaia-core
for processing, and responses are routed back to Discord.
"""

import os
import time
import uuid
import asyncio
import logging
import threading
from datetime import datetime
from typing import Optional, Dict, Any, Callable, Tuple

import httpx
# Import all necessary dataclasses for CognitionPacket
from gaia_common.protocols.cognition_packet import (
    CognitionPacket, Header, Persona, Origin, OutputRouting, DestinationTarget, Content, DataField,
    OutputDestination, PersonaRole, Routing, Model, OperationalStatus, SystemTask, Intent, Context,
    SessionHistoryRef, RelevantHistorySnippet, Cheatsheet, Constraints, Attachment, ReflectionLog,
    Sketchpad, ResponseFragment, Evaluation, Reasoning, SelectedTool, ToolExecutionResult,
    ToolRoutingState, ToolExecutionStatus, ToolCall, SidecarAction, Response, Safety, Signatures,
    Audit, Privacy, Governance, Vote, Council, TokenUsage, SystemResources, Metrics, Status,
    PacketState, TargetEngine
)

logger = logging.getLogger("GAIA.Web.Discord")

# Discord bot instance (module-level for access from main.py)
_bot = None
_bot_loop: Optional[asyncio.AbstractEventLoop] = None  # The bot's own event loop
_message_handler: Optional[Callable] = None
_voice_manager = None  # VoiceManager instance (set by start_discord_bot)
_dm_blocklist = None   # DMBlocklist instance (set by start_discord_bot)
_core_endpoint = ""    # Core endpoint URL (set by start_discord_bot)
_typing_wake_sent: Dict[str, float] = {}  # user_id -> last wake signal timestamp

# Pending approval waiters: (channel_id, user_id) -> waiter context dict
_pending_approvals: Dict[Tuple[str, str], dict] = {}
_MCP_BASE_URL = os.getenv("MCP_ENDPOINT", "http://gaia-mcp:8765/jsonrpc").rsplit("/", 1)[0]
_APPROVAL_STALE_SECONDS = 960  # 16 minutes (MCP TTL is 15 min)


def _run_on_bot_loop(coro, timeout: float = 30.0):
    """Schedule a coroutine on the bot's event loop from any thread.

    Returns the result or raises the exception from the coroutine.
    This is the safe way to call bot methods from uvicorn's event loop.
    """
    if _bot_loop is None or _bot_loop.is_closed():
        raise RuntimeError("Bot event loop not available")
    future = asyncio.run_coroutine_threadsafe(coro, _bot_loop)
    return future.result(timeout=timeout)


class DiscordInterface:
    """
    Discord interface for the GAIA Web Gateway.

    Handles:
    - Receiving messages from Discord (mentions and DMs)
    - Sending messages back to Discord channels/users
    - Converting between Discord events and CognitionPackets
    """

    def __init__(self, bot_token: str, core_endpoint: str, message_queue=None, core_fallback_endpoint: str = ""):
        self.bot_token = bot_token
        self.core_endpoint = core_endpoint
        self.core_fallback_endpoint = core_fallback_endpoint
        self.message_queue = message_queue
        self._bot = None
        self._loop = None

    async def start(self):
        """Start the Discord bot."""
        try:
            import discord
            from discord.ext import commands
        except ImportError:
            logger.error("discord.py not installed. Run: pip install discord.py")
            return False

        intents = discord.Intents.default()
        intents.message_content = True
        intents.guild_messages = True
        intents.dm_messages = True
        intents.members = True
        intents.voice_states = True
        intents.typing = True  # Receive typing events for speculative prime wake

        bot = commands.Bot(command_prefix="!", intents=intents)

        @bot.event
        async def on_ready():
            global _bot_loop
            _bot_loop = asyncio.get_running_loop()
            logger.info(f"Discord bot connected as {bot.user}")
            print(f"[DISCORD] Bot connected as {bot.user} — loop captured")
            await bot.change_presence(
                status=discord.Status.online,
                activity=discord.Activity(
                    type=discord.ActivityType.watching,
                    name="over the studio"
                )
            )
            logger.info("Discord bot presence set, bot is READY")

        @bot.event
        async def on_disconnect():
            logger.warning("Discord bot disconnected — discord.py will attempt reconnect")

        @bot.event
        async def on_resumed():
            logger.info("Discord bot session resumed")

        @bot.event
        async def on_message(message):
            if message.author == bot.user:
                return

            # Track user for voice whitelist (populate dashboard seen-users list)
            if _voice_manager is not None:
                try:
                    guild_id = str(message.guild.id) if message.guild else None
                    _voice_manager.whitelist.record_seen(
                        str(message.author.id), message.author.display_name, guild_id
                    )
                except Exception:
                    pass

            is_dm = message.guild is None

            # Blocklist: silently ignore DMs from blocked users
            if is_dm and _dm_blocklist and _dm_blocklist.is_blocked(str(message.author.id)):
                logger.info("Discord: blocked DM from user %s (%s)", message.author.id, message.author.display_name)
                return

            # Track DM user for blocklist dashboard
            if is_dm and _dm_blocklist:
                try:
                    _dm_blocklist.record_dm(str(message.author.id), message.author.display_name)
                except Exception:
                    pass

            # Check for pending approval waiter before normal processing
            waiter_key = (str(message.channel.id), str(message.author.id))
            if waiter_key in _pending_approvals:
                await self._handle_approval_response(
                    content=message.content.strip(),
                    waiter_key=waiter_key,
                    message_obj=message,
                    is_dm=is_dm,
                )
                return

            should_respond = False

            if is_dm:
                should_respond = True
            else:
                mentioned = bot.user in message.mentions
                addressed = message.content.lower().startswith(("@gaia", "gaia,", "gaia:"))
                should_respond = mentioned or addressed

            logger.debug(f"Discord message received: is_dm={is_dm}, should_respond={should_respond}, content={message.content[:50]}")

            if should_respond:
                content = message.content
                for mention in message.mentions:
                    content = content.replace(f"<@{mention.id}>", "").replace(f"<@!{mention.id}>", "")
                content = content.strip()

                if content:
                    logger.info(f"Discord processing message from {message.author.display_name}: {content[:50]}")
                    await self._handle_message(
                        content=content,
                        channel_id=str(message.channel.id),
                        user_id=str(message.author.id),
                        guild_id=str(message.guild.id) if message.guild else None,
                        author_name=message.author.display_name,
                        message_id=str(message.id),
                        is_dm=is_dm,
                        message_obj=message
                    )

            if not is_dm:
                await bot.process_commands(message)

        @bot.event
        async def on_voice_state_update(member, before, after):
            if _voice_manager is not None:
                try:
                    await _voice_manager.handle_voice_state_update(member, before, after)
                except Exception:
                    logger.error("Voice state update handler error", exc_info=True)

        @bot.event
        async def on_typing(channel, user, when):
            """Speculative prime wake: typing in a recent DM triggers preload."""
            if user == bot.user:
                return
            # Only DMs (DMChannel has no guild)
            if not isinstance(channel, discord.DMChannel):
                return
            uid = str(user.id)
            # Blocked users don't trigger wake
            if _dm_blocklist and _dm_blocklist.is_blocked(uid):
                return
            # Only wake if GAIA replied to this user within 48h
            if not (_dm_blocklist and _dm_blocklist.has_recent_gaia_reply(uid)):
                return
            # Debounce: one wake signal per 30s per user
            now = time.time()
            if now - _typing_wake_sent.get(uid, 0) < 30:
                return
            _typing_wake_sent[uid] = now
            # Fire-and-forget wake signal
            logger.info("Discord: typing wake trigger for DM user %s", uid)
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    await client.post(f"{_core_endpoint}/sleep/wake")
            except Exception:
                logger.debug("Typing wake signal failed", exc_info=True)

        @bot.command(name="call")
        async def voice_call(ctx):
            """Join the caller's voice channel for a voice conversation."""
            if _voice_manager is None:
                await ctx.send("Voice isn't set up right now.")
                return

            if not ctx.author.voice or not ctx.author.voice.channel:
                await ctx.send("Join a voice channel first, then try again.")
                return

            # Already connected?
            if _voice_manager._vc and _voice_manager._vc.is_connected():
                if _voice_manager._vc.channel == ctx.author.voice.channel:
                    await ctx.send("I'm already here with you!")
                else:
                    await ctx.send(
                        f"I'm in **{_voice_manager._channel_name}** right now. "
                        "Use `!hangup` first if you want me to switch."
                    )
                return

            channel = ctx.author.voice.channel
            await ctx.send(f"Joining **{channel.name}**...")
            _voice_manager._connected_user = ctx.author.display_name
            await _voice_manager._join_channel(channel)

        @bot.command(name="hangup")
        async def voice_hangup(ctx):
            """Disconnect GAIA from the current voice channel."""
            if _voice_manager is None:
                await ctx.send("Voice isn't set up right now.")
                return

            if not _voice_manager._vc or not _voice_manager._vc.is_connected():
                await ctx.send("I'm not in a voice channel.")
                return

            await _voice_manager.disconnect()
            await ctx.send("Disconnected. Talk to you later!")

        self._bot = bot
        global _bot
        _bot = bot

        try:
            await bot.start(self.bot_token)
        except asyncio.CancelledError:
            logger.warning("Discord bot cancelled (CancelledError) — will restart via watchdog")
        except Exception as e:
            logger.error(f"Discord bot error: {e}")

    async def _handle_message(
        self,
        content: str,
        channel_id: str,
        user_id: str,
        guild_id: Optional[str],
        author_name: str,
        message_id: str,
        is_dm: bool,
        message_obj: Any
    ):
        """Handle incoming Discord message by forwarding to gaia-core as a CognitionPacket."""
        logger.info(f"Discord: Received {'DM' if is_dm else 'channel message'} from {author_name}")

        # Check if GAIA is in a state that warrants a canned response or sleep-wake
        core_state = "active"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                check = await client.get(f"{self.core_endpoint}/sleep/distracted-check")
                if check.status_code == 200:
                    data = check.json()
                    core_state = data.get("state", "active")
                    canned = data.get("canned_response")
                    if canned:
                        logger.info("Discord: Canned response for state=%s", core_state)
                        await self._send_response(message_obj, canned, is_dm)
                        return
        except Exception:
            logger.debug("Distracted-check failed — proceeding normally", exc_info=True)

        # Sleep-aware: enqueue, wake, wait, then process normally
        if core_state in ("asleep", "drowsy") and self.message_queue is not None:
            from gaia_web.queue.message_queue import QueuedMessage

            qm = QueuedMessage(
                message_id=message_id,
                content=content,
                source="discord",
                session_id=f"discord_dm_{user_id}" if is_dm else f"discord_channel_{channel_id}",
                metadata={"author_name": author_name, "is_dm": is_dm},
            )
            await self.message_queue.enqueue(qm)
            logger.info("Discord: GAIA is asleep — queued message %s, waiting for wake", message_id)

            # Show typing indicator while waiting for wake
            try:
                async with message_obj.channel.typing():
                    woke = await self.message_queue.wait_for_active(timeout=120.0)
            except Exception:
                woke = await self.message_queue.wait_for_active(timeout=120.0)

            await self.message_queue.dequeue()

            if not woke:
                logger.warning("Discord: Wake timed out for message %s", message_id)
                await self._send_response(
                    message_obj,
                    "I'm having trouble waking up right now... give me another moment and try again.",
                    is_dm,
                )
                return

            logger.info("Discord: GAIA woke up — processing queued message %s", message_id)
            # Fall through to normal packet construction below

        # Show GAIA as online while processing this message
        await self._set_presence("online", "Thinking...")

        packet_id = str(uuid.uuid4())
        session_id = f"discord_dm_{user_id}" if is_dm else f"discord_channel_{channel_id}"
        current_time = datetime.now().isoformat()

        # Construct CognitionPacket
        packet = CognitionPacket(
            version="0.2", # Use appropriate version
            header=Header(
                datetime=current_time,
                session_id=session_id,
                packet_id=packet_id,
                sub_id="0", # Initial sub_id
                persona=Persona(
                    identity_id="default_user", # Placeholder
                    persona_id="default_persona", # Placeholder
                    role=PersonaRole.DEFAULT, # PersonaRole.DEFAULT for user messages
                    tone_hint="conversational"
                ),
                origin=Origin.USER,
                routing=Routing(
                    target_engine=TargetEngine.PRIME, # Default target
                    priority=5
                ),
                model=Model( # Placeholder model info
                    name="default_model",
                    provider="default_provider",
                    context_window_tokens=8192
                ),
                output_routing=OutputRouting(
                    primary=DestinationTarget(
                        destination=OutputDestination.DISCORD,
                        channel_id=channel_id,
                        user_id=user_id,
                        reply_to_message_id=message_id,
                        metadata={"is_dm": is_dm, "author_name": author_name}
                    ),
                    source_destination=OutputDestination.DISCORD,
                    addressed_to_gaia=True, # Assuming direct address
                ),
                operational_status=OperationalStatus(status="initialized")
            ),
            intent=Intent(user_intent="chat", system_task=SystemTask.GENERATE_DRAFT, confidence=0.0), # Placeholder
            context=Context(
                session_history_ref=SessionHistoryRef(type="discord_channel", value=session_id),
                cheatsheets=[],
                constraints=Constraints(max_tokens=2048, time_budget_ms=30000, safety_mode="strict"),
            ),
            content=Content(
                original_prompt=content,
                data_fields=[DataField(key="user_message", value=content, type="text")]
            ),
            reasoning=Reasoning(), # Empty for initial packet
            response=Response(candidate="", confidence=0.0, stream_proposal=False), # Empty response
            governance=Governance(
                safety=Safety(execution_allowed=False, dry_run=True) # Default
            ),
            metrics=Metrics(
                token_usage=TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
                latency_ms=0
            ),
            status=Status(finalized=False, state=PacketState.INITIALIZED, next_steps=[]),
            tool_routing=ToolRoutingState() # Empty for initial packet
        )
        packet.compute_hashes() # Compute hashes for integrity

        try:
            from gaia_web.utils.retry import post_with_retry

            fallback = f"{self.core_fallback_endpoint}/process_packet" if self.core_fallback_endpoint else None

            # Show typing indicator while GAIA processes the request
            async with message_obj.channel.typing():
                response = await post_with_retry(
                    f"{self.core_endpoint}/process_packet",
                    json=packet.to_serializable_dict(),
                    fallback_url=fallback,
                )

            # Expect a full CognitionPacket back
            completed_packet_dict = response.json()
            completed_packet = CognitionPacket.from_dict(completed_packet_dict)

            # Check for pending tool approval before sending normal response
            approval_info = self._extract_pending_approval(completed_packet)
            if approval_info:
                await self._send_approval_prompt(
                    channel_id=channel_id,
                    user_id=user_id,
                    message_obj=message_obj,
                    is_dm=is_dm,
                    approval_info=approval_info,
                )
                return

            gaia_response_text = completed_packet.response.candidate
            if not gaia_response_text:
                gaia_response_text = "GAIA processed your request but did not generate a text response."

            # Send response back to Discord
            await self._send_response(message_obj, gaia_response_text, is_dm)

            # Record GAIA reply for typing-wake 48h gate
            if is_dm and _dm_blocklist:
                try:
                    _dm_blocklist.record_gaia_reply(user_id)
                except Exception:
                    pass

        except httpx.TimeoutException:
            logger.error("Discord: Request to gaia-core timed out")
            await self._send_response(
                message_obj,
                "I'm sorry, GAIA Core took too long to respond. Please try again.",
                is_dm
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"Discord: GAIA Core returned error: {e.response.status_code} - {e.response.text}")
            await self._send_response(
                message_obj,
                "I encountered an error communicating with my core systems. Please try again in a moment.",
                is_dm
            )
        except Exception as e:
            logger.exception(f"Discord: An unexpected error occurred during packet processing: {e}")
            await self._send_response(
                message_obj,
                "An unexpected error occurred. Please try again.",
                is_dm
            )
        finally:
            # Return to idle presence after message processing completes
            await self._set_presence("idle", "over the studio")

    @staticmethod
    async def _set_presence(status: str, activity: str):
        """Update Discord bot presence directly from the bot event loop.

        Unlike change_presence_from_external (which cross-thread schedules),
        this is safe to await from on_message / _handle_message since we're
        already on the bot's own loop.
        """
        import discord as _discord

        if not _bot or not _bot.is_ready():
            return
        status_map = {
            "idle": _discord.Status.idle,
            "online": _discord.Status.online,
            "dnd": _discord.Status.dnd,
        }
        try:
            await _bot.change_presence(
                status=status_map.get(status, _discord.Status.online),
                activity=_discord.Activity(
                    type=_discord.ActivityType.watching, name=activity
                ),
            )
        except Exception:
            logger.debug("Presence update failed", exc_info=True)

    async def _send_response(self, message_obj: Any, content: str, is_dm: bool):
        """Send response back to Discord."""
        try:
            # Split long messages (Discord 2000 char limit)
            messages = self._split_message(content)
            for msg in messages:
                await message_obj.channel.send(msg)
            logger.info(f"Discord: Sent {len(messages)} message(s) to {'DM' if is_dm else 'channel'}")
        except Exception as e:
            logger.error(f"Discord: Failed to send response: {e}")

    def _split_message(self, content: str, max_length: int = 2000) -> list:
        """Split message into chunks respecting Discord's character limit."""
        if len(content) <= max_length:
            return [content]

        messages = []
        remaining = content

        while remaining:
            if len(remaining) <= max_length:
                messages.append(remaining)
                break

            split_point = remaining[:max_length].rfind('\n')
            if split_point < max_length // 2:
                split_point = remaining[:max_length].rfind(' ')
            if split_point < max_length // 2:
                split_point = max_length

            messages.append(remaining[:split_point])
            remaining = remaining[split_point:].lstrip()

        return messages

    # --- Approval Flow ---

    def _extract_pending_approval(self, packet: CognitionPacket) -> Optional[dict]:
        """Check if a completed packet contains a pending approval request."""
        if not packet.tool_routing or not packet.tool_routing.execution_result:
            return None
        result = packet.tool_routing.execution_result
        output = result.output
        if isinstance(output, dict) and output.get("pending_approval"):
            return {
                "action_id": output.get("action_id"),
                "challenge": output.get("challenge"),
                "proposal": output.get("proposal", ""),
                "tool_name": output.get("tool_name", "unknown"),
            }
        return None

    async def _send_approval_prompt(
        self,
        channel_id: str,
        user_id: str,
        message_obj: Any,
        is_dm: bool,
        approval_info: dict,
    ):
        """Send a challenge-response approval prompt to Discord and register a waiter."""
        tool_name = approval_info["tool_name"]
        challenge = approval_info["challenge"]
        proposal = approval_info.get("proposal", "")

        # Truncate proposal for Discord display
        if len(proposal) > 1200:
            proposal = proposal[:1200] + "\n... [truncated]"

        prompt_text = (
            f"**Tool Approval Required** \u2014 `{tool_name}`\n\n"
        )
        if proposal:
            prompt_text += f"```\n{proposal}\n```\n"
        prompt_text += (
            f"**Challenge:** `{challenge}`\n"
            f"Reply with the challenge **reversed** to approve, or say **cancel** to deny.\n"
            f"_This request expires in 15 minutes._"
        )

        await self._send_response(message_obj, prompt_text, is_dm)

        # Register waiter
        waiter_key = (channel_id, user_id)
        _pending_approvals[waiter_key] = {
            "action_id": approval_info["action_id"],
            "challenge": challenge,
            "tool_name": tool_name,
            "registered_at": time.time(),
        }
        logger.info(f"Discord: Registered approval waiter for {waiter_key}, tool={tool_name}")

        # Cleanup stale waiters while we're here
        self._cleanup_stale_approvals()

    async def _handle_approval_response(
        self,
        content: str,
        waiter_key: Tuple[str, str],
        message_obj: Any,
        is_dm: bool,
    ):
        """Handle a user's response to an approval challenge."""
        waiter = _pending_approvals.get(waiter_key)
        if not waiter:
            return

        normalized = content.strip().lower()

        # Cancel / deny
        if normalized in ("cancel", "deny", "no", "abort"):
            del _pending_approvals[waiter_key]
            logger.info(f"Discord: User cancelled approval for {waiter.get('tool_name')}")
            await self._send_response(
                message_obj,
                f"Cancelled `{waiter.get('tool_name')}` approval. The action will expire on the server.",
                is_dm,
            )
            return

        # Check challenge (case-insensitive)
        expected = waiter["challenge"][::-1]
        if content.strip().upper() != expected:
            # Wrong answer — re-register waiter for retry
            logger.info(f"Discord: Wrong challenge response for {waiter.get('tool_name')}")
            await self._send_response(
                message_obj,
                f"That's not right. The challenge is `{waiter['challenge']}` \u2014 reply with it **reversed** (e.g., `{expected}`), or say **cancel**.",
                is_dm,
            )
            return

        # Correct challenge — submit approval to MCP
        del _pending_approvals[waiter_key]
        action_id = waiter["action_id"]

        try:
            result = await self._submit_mcp_approval(action_id, content.strip().upper())
        except Exception as e:
            logger.error(f"Discord: MCP approval submission failed: {e}", exc_info=True)
            await self._send_response(
                message_obj,
                f"Approval submitted but MCP returned an error: {e}",
                is_dm,
            )
            return

        if result.get("ok"):
            summary = self._summarize_tool_result(result.get("result"))
            await self._send_response(
                message_obj,
                f"Approved. `{waiter['tool_name']}` executed successfully.\n{summary}",
                is_dm,
            )
        elif result.get("error") == "action not found or expired":
            await self._send_response(
                message_obj,
                f"This approval has expired. Please trigger the action again.",
                is_dm,
            )
        elif result.get("error") == "invalid approval string":
            await self._send_response(
                message_obj,
                f"The MCP server rejected the approval code. This shouldn't happen \u2014 please try again.",
                is_dm,
            )
        else:
            error_msg = result.get("error", "unknown error")
            await self._send_response(
                message_obj,
                f"Approved, but `{waiter['tool_name']}` failed: {error_msg}",
                is_dm,
            )

    async def _submit_mcp_approval(self, action_id: str, approval: str) -> dict:
        """Submit an approval to MCP's /approve_action endpoint."""
        url = f"{_MCP_BASE_URL}/approve_action"
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json={"action_id": action_id, "approval": approval})
            if resp.status_code == 404:
                return {"error": "action not found or expired"}
            if resp.status_code == 403:
                return {"error": "invalid approval string"}
            resp.raise_for_status()
            return resp.json()

    def _cleanup_stale_approvals(self):
        """Remove approval waiters older than _APPROVAL_STALE_SECONDS."""
        now = time.time()
        stale_keys = [
            k for k, v in _pending_approvals.items()
            if now - v.get("registered_at", 0) > _APPROVAL_STALE_SECONDS
        ]
        for k in stale_keys:
            del _pending_approvals[k]
        if stale_keys:
            logger.info(f"Discord: Cleaned up {len(stale_keys)} stale approval waiter(s)")

    @staticmethod
    def _summarize_tool_result(result: Any) -> str:
        """Format a tool execution result for Discord display."""
        if result is None:
            return ""
        if isinstance(result, dict):
            if result.get("ok") and result.get("result"):
                inner = result["result"]
                if isinstance(inner, str):
                    summary = inner[:500]
                else:
                    import json
                    summary = json.dumps(inner, indent=2, ensure_ascii=False)[:500]
            elif result.get("error"):
                summary = f"Error: {result['error']}"
            else:
                import json
                summary = json.dumps(result, indent=2, ensure_ascii=False)[:500]
        elif isinstance(result, str):
            summary = result[:500]
        else:
            summary = str(result)[:500]
        if summary:
            return f"```\n{summary}\n```"
        return ""


async def send_to_channel(channel_id: str, content: str, reply_to_message_id: Optional[str] = None) -> bool:
    """Send a message to a specific Discord channel (for autonomous messages).

    Safe to call from any event loop — schedules work on the bot's own loop.
    """
    global _bot
    if not _bot or not _bot.is_ready():
        logger.error("Discord bot not connected")
        return False

    async def _send():
        channel = _bot.get_channel(int(channel_id))
        if not channel:
            channel = await _bot.fetch_channel(int(channel_id))
        if not channel:
            logger.error(f"Could not find channel {channel_id}")
            return False
        messages = DiscordInterface(None, None)._split_message(content)
        for msg in messages:
            if reply_to_message_id:
                try:
                    reply_message = await channel.fetch_message(int(reply_to_message_id))
                    await reply_message.reply(msg)
                except Exception as e:
                    logger.warning(f"Failed to reply to message {reply_to_message_id}: {e}. Sending as regular message.")
                    await channel.send(msg)
            else:
                await channel.send(msg)
        return True

    try:
        return _run_on_bot_loop(_send(), timeout=30.0)
    except Exception as e:
        logger.error(f"Failed to send to channel {channel_id}: {e}")
        return False


async def send_to_user(user_id: str, content: str) -> bool:
    """Send a DM to a specific user (for autonomous messages).

    Safe to call from any event loop — schedules work on the bot's own loop.
    """
    global _bot
    if not _bot or not _bot.is_ready():
        logger.error("Discord bot not connected")
        return False

    async def _send():
        user = await _bot.fetch_user(int(user_id))
        if not user:
            logger.error(f"Could not find user {user_id}")
            return False
        messages = DiscordInterface(None, None)._split_message(content)
        for msg in messages:
            await user.send(msg)
        return True

    try:
        return _run_on_bot_loop(_send(), timeout=30.0)
    except Exception as e:
        logger.error(f"Failed to send DM to user {user_id}: {e}")
        return False


def start_discord_bot(bot_token: str, core_endpoint: str, message_queue=None, voice_manager=None, dm_blocklist=None, core_fallback_endpoint: str = "") -> bool:
    """
    Start the Discord bot in a background thread.

    Args:
        bot_token: Discord bot token
        core_endpoint: URL of gaia-core service (e.g., http://gaia-core:6415)
        message_queue: Optional MessageQueue instance for sleep/wake queueing
        voice_manager: Optional VoiceManager for voice auto-answer
        dm_blocklist: Optional DMBlocklist for DM user blocking + typing wake
        core_fallback_endpoint: Optional HA fallback URL for gaia-core

    Returns:
        True if started successfully
    """
    global _voice_manager, _dm_blocklist, _core_endpoint
    _voice_manager = voice_manager
    _dm_blocklist = dm_blocklist
    _core_endpoint = core_endpoint

    if not bot_token:
        logger.error("Cannot start Discord bot: no token provided")
        return False

    interface = DiscordInterface(bot_token, core_endpoint, message_queue=message_queue, core_fallback_endpoint=core_fallback_endpoint)

    def run_bot():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        backoff = 5
        max_backoff = 60
        while True:
            started_at = time.time()
            try:
                loop.run_until_complete(interface.start())
                # Clean exit via stop_discord_bot()
                break
            except Exception as e:
                logger.error(f"Discord bot error: {e} — reconnecting in {backoff}s")
            # Reset backoff if bot ran for >60s (stable session, not a crash loop)
            if time.time() - started_at > 60:
                backoff = 5
            time.sleep(backoff)
            backoff = min(backoff * 2, max_backoff)
            logger.info("Discord bot watchdog: restarting...")
        loop.close()

    thread = threading.Thread(target=run_bot, daemon=True, name="discord-bot")
    thread.start()

    logger.info("Discord bot started in background thread (with watchdog)")
    return True


def stop_discord_bot():
    """Stop the Discord bot."""
    global _bot, _bot_loop
    if _bot and _bot_loop and not _bot_loop.is_closed():
        future = asyncio.run_coroutine_threadsafe(_bot.close(), _bot_loop)
        try:
            future.result(timeout=10.0)  # Wait for clean close
        except Exception as e:
            logger.warning(f"Error during bot close: {e}")
    _bot = None
    _bot_loop = None


def is_bot_ready() -> bool:
    """Check if the Discord bot is connected and ready."""
    global _bot
    return _bot is not None and _bot.is_ready()


def get_discord_status() -> Dict[str, Any]:
    """Return Discord bot connectivity status for the dashboard."""
    global _bot, _bot_loop
    if _bot is None:
        return {"connected": False, "status": "not_started", "user": None, "guilds": 0}
    if not _bot.is_ready():
        return {"connected": False, "status": "connecting", "user": None, "guilds": 0}
    return {
        "connected": True,
        "status": "ready",
        "user": str(_bot.user) if _bot.user else None,
        "guilds": len(_bot.guilds) if _bot.guilds else 0,
        "latency_ms": round(_bot.latency * 1000, 1) if _bot.latency else None,
    }


def change_presence_from_external(activity_name: str, status_str: str | None = None):
    """Change bot presence from an external event loop (e.g. uvicorn).

    This is the safe way to call change_presence from the /presence endpoint.
    """
    global _bot, _bot_loop
    if not _bot or not _bot.is_ready():
        raise RuntimeError("Bot not connected")

    import discord as _discord

    status_map = {
        "idle": _discord.Status.idle,
        "online": _discord.Status.online,
        "dnd": _discord.Status.dnd,
        "invisible": _discord.Status.invisible,
    }
    effective_status = status_map.get(status_str, _discord.Status.online)

    async def _change():
        if effective_status == _discord.Status.invisible:
            await _bot.change_presence(status=effective_status, activity=None)
        else:
            await _bot.change_presence(
                status=effective_status,
                activity=_discord.Activity(type=_discord.ActivityType.watching, name=activity_name),
            )

    _run_on_bot_loop(_change(), timeout=10.0)
