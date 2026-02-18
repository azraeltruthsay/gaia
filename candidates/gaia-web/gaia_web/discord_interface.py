"""
Discord Interface for GAIA Web Gateway

This module handles Discord bot integration as part of the Unified Interface Gateway.
Discord messages are received here, converted to CognitionPackets, sent to gaia-core
for processing, and responses are routed back to Discord.
"""

import os
import uuid
import asyncio
import logging
import threading
from datetime import datetime
from typing import Optional, Dict, Any, Callable

import httpx
# Import all necessary dataclasses for CognitionPacket
from gaia_common.protocols.cognition_packet import (
    CognitionPacket, Header, Persona, Origin, OutputRouting, DestinationTarget, Content, DataField,
    OutputDestination, PersonaRole, Routing, Model, OperationalStatus, SystemTask, Intent, Context,
    SessionHistoryRef, RelevantHistorySnippet, Cheatsheet, Constraints, Attachment, ReflectionLog,
    Sketchpad, ResponseFragment, Evaluation, Reasoning, SelectedTool, ToolExecutionResult,
    ToolRoutingState, ToolCall, SidecarAction, Response, Safety, Signatures, Audit, Privacy,
    Governance, Vote, Council, TokenUsage, SystemResources, Metrics, Status, PacketState, TargetEngine
)

logger = logging.getLogger("GAIA.Web.Discord")

# Discord bot instance (module-level for access from main.py)
_bot = None
_message_handler: Optional[Callable] = None


class DiscordInterface:
    """
    Discord interface for the GAIA Web Gateway.

    Handles:
    - Receiving messages from Discord (mentions and DMs)
    - Sending messages back to Discord channels/users
    - Converting between Discord events and CognitionPackets
    """

    def __init__(self, bot_token: str, core_endpoint: str, message_queue=None):
        self.bot_token = bot_token
        self.core_endpoint = core_endpoint
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

        bot = commands.Bot(command_prefix="!", intents=intents)

        @bot.event
        async def on_ready():
            logger.info(f"Discord bot connected as {bot.user}")
            await bot.change_presence(
                status=discord.Status.online,
                activity=discord.Activity(
                    type=discord.ActivityType.watching,
                    name="over the studio"
                )
            )
            logger.info("Discord bot presence set, bot is READY")

        @bot.event
        async def on_message(message):
            if message.author == bot.user:
                return

            is_dm = message.guild is None
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

        self._bot = bot
        global _bot
        _bot = bot

        try:
            await bot.start(self.bot_token)
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
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(
                    f"{self.core_endpoint}/process_packet", # New endpoint name
                    json=packet.to_serializable_dict(), # Send as serializable dict
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()

                # Expect a full CognitionPacket back
                completed_packet_dict = response.json()
                completed_packet = CognitionPacket.from_dict(completed_packet_dict) # Deserialize back to object

                gaia_response_text = completed_packet.response.candidate # Extract LLM response
                if not gaia_response_text:
                    gaia_response_text = "GAIA processed your request but did not generate a text response."

                # Send response back to Discord
                await self._send_response(message_obj, gaia_response_text, is_dm)

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


async def send_to_channel(channel_id: str, content: str, reply_to_message_id: Optional[str] = None) -> bool:
    """Send a message to a specific Discord channel (for autonomous messages)."""
    global _bot
    if not _bot or not _bot.is_ready():
        logger.error("Discord bot not connected")
        return False

    try:
        channel = _bot.get_channel(int(channel_id))
        if not channel:
            channel = await _bot.fetch_channel(int(channel_id))
        if channel:
            messages = DiscordInterface(None, None)._split_message(content) # Use helper
            for msg in messages:
                if reply_to_message_id:
                    # Attempt to reply to a specific message if ID is provided
                    try:
                        reply_message = await channel.fetch_message(int(reply_to_message_id))
                        await reply_message.reply(msg)
                    except Exception as e:
                        logger.warning(f"Failed to reply to message {reply_to_message_id}: {e}. Sending as regular message.")
                        await channel.send(msg)
                else:
                    await channel.send(msg)
            return True
        else:
            logger.error(f"Could not find channel {channel_id}")
            return False
    except Exception as e:
        logger.error(f"Failed to send to channel {channel_id}: {e}")
        return False


async def send_to_user(user_id: str, content: str) -> bool:
    """Send a DM to a specific user (for autonomous messages)."""
    global _bot
    if not _bot or not _bot.is_ready():
        logger.error("Discord bot not connected")
        return False

    try:
        user = await _bot.fetch_user(int(user_id))
        if user:
            # Split long messages
            max_length = 2000
            if len(content) <= max_length:
                await user.send(content)
            else:
                remaining = content
                while remaining:
                    if len(remaining) <= max_length:
                        await user.send(remaining)
                        break
                    split_point = remaining[:max_length].rfind('\n')
                    if split_point < max_length // 2:
                        split_point = remaining[:max_length].rfind(' ')
                    if split_point < max_length // 2:
                        split_point = max_length
                    await user.send(remaining[:split_point])
                    remaining = remaining[split_point:].lstrip()
            return True
        else:
            logger.error(f"Could not find user {user_id}")
            return False
    except Exception as e:
        logger.error(f"Failed to send DM to user {user_id}: {e}")
        return False


def start_discord_bot(bot_token: str, core_endpoint: str, message_queue=None) -> bool:
    """
    Start the Discord bot in a background thread.

    Args:
        bot_token: Discord bot token
        core_endpoint: URL of gaia-core service (e.g., http://gaia-core:6415)
        message_queue: Optional MessageQueue instance for sleep/wake queueing

    Returns:
        True if started successfully
    """
    if not bot_token:
        logger.error("Cannot start Discord bot: no token provided")
        return False

    interface = DiscordInterface(bot_token, core_endpoint, message_queue=message_queue)

    def run_bot():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(interface.start())
        except Exception as e:
            logger.error(f"Discord bot thread error: {e}")
        finally:
            loop.close()

    thread = threading.Thread(target=run_bot, daemon=True, name="discord-bot")
    thread.start()

    logger.info("Discord bot started in background thread")
    return True


def stop_discord_bot():
    """Stop the Discord bot."""
    global _bot
    if _bot:
        try:
            asyncio.run_coroutine_threadsafe(_bot.close(), _bot.loop)
            logger.info("Discord bot stopped")
        except Exception as e:
            logger.error(f"Error stopping Discord bot: {e}")
        _bot = None


def is_bot_ready() -> bool:
    """Check if the Discord bot is connected and ready."""
    global _bot
    return _bot is not None and _bot.is_ready()
