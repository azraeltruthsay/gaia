"""
Discord Interface for GAIA Web Gateway

This module handles Discord bot integration as part of the Unified Interface Gateway.
Discord messages are received here, converted to CognitionPackets, sent to gaia-core
for processing, and responses are routed back to Discord.
"""

import os
import re
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
_bot_thread: Optional[threading.Thread] = None
_message_handler: Optional[Callable] = None


def _strip_think_tags(text: str) -> str:
    """
    Strip <think>...</think>, <thinking>...</thinking>, and other reflection tag
    variants from LLM output before sending to Discord.

    Mirrors the robust implementation in gaia-core's output_router.py but lives
    here as a safety net — the last stop before messages reach Discord.
    """
    if not text:
        return text

    result = text

    # Closed <think>/<thinking> blocks
    result = re.sub(r'<(?:think|thinking)>.*?</(?:think|thinking)>\s*', '', result, flags=re.DOTALL)

    # Unclosed <think>/<thinking> tags (model started thinking but never closed)
    result = re.sub(r'<(?:think|thinking)>.*$', '', result, flags=re.DOTALL)

    # Truncated/malformed tags like </think or <thinking attr="">
    result = re.sub(r'</?(?:think|thinking)[^>]*>', '', result)

    # Other common reflection tags that shouldn't be user-facing
    for tag in ('reflection', 'reasoning', 'internal', 'scratchpad', 'planning', 'analysis'):
        result = re.sub(rf'<{tag}>.*?</{tag}>\s*', '', result, flags=re.DOTALL | re.IGNORECASE)
        result = re.sub(rf'<{tag}>.*$', '', result, flags=re.DOTALL | re.IGNORECASE)
        result = re.sub(rf'</?{tag}[^>]*>', '', result, flags=re.IGNORECASE)

    # Qwen/DeepSeek-style: <|start_thinking|>...<|end_thinking|>
    result = re.sub(r'<\|start_thinking\|>.*?<\|end_thinking\|>\s*', '', result, flags=re.DOTALL)
    result = re.sub(r'<\|start_thinking\|>.*$', '', result, flags=re.DOTALL)
    result = re.sub(r'<\|(?:start_thinking|end_thinking)\|>', '', result)

    return result.strip()


def _build_status_footer(packet: 'CognitionPacket') -> str:
    """
    Build a compact Discord status footer from a completed CognitionPacket.

    Shows model name, observer activity, and deployment mode.
    """
    parts = []

    # Model name
    model_name = "unknown"
    try:
        model_name = packet.header.model.name or "unknown"
    except (AttributeError, TypeError):
        pass
    parts.append(f"**Model**: {model_name}")

    # Observer trace
    observer_summary = "none"
    try:
        traces = packet.status.observer_trace
        if traces:
            observer_summary = "; ".join(traces[-3:])  # last 3 entries max
            if len(traces) > 3:
                observer_summary = f"...{observer_summary}"
    except (AttributeError, TypeError):
        pass
    parts.append(f"**Observer**: {observer_summary}")

    # Mode: candidate vs live (check env var, default to live)
    mode = os.environ.get("GAIA_MODE", "live")
    parts.append(f"**Mode**: {mode}")

    return "\n---\n> " + " | ".join(parts)


class DiscordInterface:
    """
    Discord interface for the GAIA Web Gateway.

    Handles:
    - Receiving messages from Discord (mentions and DMs)
    - Sending messages back to Discord channels/users
    - Converting between Discord events and CognitionPackets
    """

    def __init__(self, bot_token: str, core_endpoint: str):
        self.bot_token = bot_token
        self.core_endpoint = core_endpoint
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
            print(f"[DISCORD] Bot connected as {bot.user}", flush=True)
            logger.info(f"Discord bot connected as {bot.user}")
            await bot.change_presence(
                status=discord.Status.online,
                activity=discord.Activity(
                    type=discord.ActivityType.watching,
                    name="over the studio"
                )
            )
            print(f"[DISCORD] Presence set, bot is READY", flush=True)

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

            print(f"[DISCORD] Message received: is_dm={is_dm}, should_respond={should_respond}, content={message.content[:50]}", flush=True)

            if should_respond:
                content = message.content
                for mention in message.mentions:
                    content = content.replace(f"<@{mention.id}>", "").replace(f"<@!{mention.id}>", "")
                content = content.strip()

                if content:
                    print(f"[DISCORD] Processing message from {message.author.display_name}: {content[:50]}", flush=True)
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
            async with httpx.AsyncClient(timeout=600.0) as client:
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

                # Strip think tags that may have leaked through from gaia-core
                gaia_response_text = _strip_think_tags(gaia_response_text)

                # Build and append status footer
                footer = _build_status_footer(completed_packet)
                if footer:
                    gaia_response_text = gaia_response_text + footer

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
                f"I encountered an error communicating with GAIA Core. Status: {e.response.status_code}\nDetails: {e.response.text}",
                is_dm
            )
        except Exception as e:
            logger.exception(f"Discord: An unexpected error occurred during packet processing: {e}")
            await self._send_response(
                message_obj,
                f"An unexpected error occurred: {e}. Please try again.",
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
                    except discord.NotFound:
                        await channel.send(f"Reply to message {reply_to_message_id} failed: message not found.\n{msg}")
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


def start_discord_bot(bot_token: str, core_endpoint: str) -> bool:
    """
    Start the Discord bot in a background thread.

    Args:
        bot_token: Discord bot token
        core_endpoint: URL of gaia-core service (e.g., http://gaia-core:6415)

    Returns:
        True if started successfully
    """
    global _bot_thread

    if not bot_token:
        logger.error("Cannot start Discord bot: no token provided")
        return False

    interface = DiscordInterface(bot_token, core_endpoint)

    def run_bot():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(interface.start())
        except Exception as e:
            logger.error(f"Discord bot thread error: {e}")
        finally:
            loop.close()

    _bot_thread = threading.Thread(target=run_bot, daemon=True, name="discord-bot")
    _bot_thread.start()

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
