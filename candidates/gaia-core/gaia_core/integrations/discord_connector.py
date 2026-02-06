"""
Discord Connector for GAIA Spinal Column

This module provides Discord integration as both:
- Output destination (send responses to Discord channels via webhook or bot)
- Input source (listen for @GAIA mentions and DMs, routing to AgentCore)
"""

import os
import logging
import asyncio
import threading
from typing import Optional, Dict, Any, Generator, Callable, List

from gaia_common.integrations.discord import DiscordConfig, DiscordWebhookSender
from gaia_common.protocols.cognition_packet import (
    CognitionPacket,
    OutputDestination,
    DestinationTarget,
)
from gaia_common.utils.destination_registry import DestinationConnector

logger = logging.getLogger("GAIA.Discord")

class DiscordConnector(DestinationConnector):
    """
    Discord connector for the GAIA spinal column.
    Supports both webhook output and (future) bot-based bidirectional communication.
    """

    def __init__(self, config: Optional[DiscordConfig] = None):
        super().__init__("discord", OutputDestination.DISCORD)
        self.config = config or DiscordConfig.from_env()
        self._webhook_sender: Optional[DiscordWebhookSender] = None
        self._bot_client = None  # Will hold discord.py client when available
        self._bot_thread: Optional[threading.Thread] = None
        self._message_callback: Optional[Callable[[str, str, Dict[str, Any]], None]] = None

        # Check if Discord integration is enabled
        if not self.config.enabled:
            logger.info("Discord integration disabled in config")
            self._enabled = False
            return

        # Initialize webhook sender if URL configured
        if self.config.webhook_url:
            self._webhook_sender = DiscordWebhookSender(
                self.config.webhook_url,
                self.config.bot_name,
                self.config.avatar_url
            )
            logger.info(f"Discord webhook sender initialized (bot_name={self.config.bot_name})")

    def send(self, content: str, target: DestinationTarget,
             packet: Optional[CognitionPacket] = None) -> bool:
        """Send content to Discord directly via bot or webhook."""
        try:
            is_dm = target.metadata.get("is_dm", False) if target.metadata else False
            
            if self._is_bot_connected() and (is_dm or target.channel_id):
                # Prefer sending via bot if connected and target is specific (DM or channel)
                logger.debug("DiscordConnector: Attempting to send via bot.")
                return self._send_via_bot(content, target, packet)
            elif self._webhook_sender:
                # Fallback to webhook if bot is not connected or target is not specific
                logger.debug("DiscordConnector: Attempting to send via webhook.")
                return self._send_via_webhook(content, target, packet)
            else:
                logger.error("DiscordConnector: No available method to send to Discord (bot not connected and no webhook).")
                return False
        except Exception as e:
            logger.error(f"DiscordConnector: Failed to send message: {e}")
            return False

    def send_stream(self, token_generator: Generator[str, None, None],
                    target: DestinationTarget,
                    packet: Optional[CognitionPacket] = None) -> bool:
        """
        Stream tokens to Discord.
        Note: Discord doesn't support true streaming, so we collect and send.
        """
        # Collect all tokens
        content = "".join(token_generator)
        return self.send(content, target, packet)

    def is_available(self) -> bool:
        """Check if Discord output is available."""
        return bool(self._webhook_sender) or self._is_bot_connected()

    def _send_via_webhook(self, content: str, target: DestinationTarget,
                          packet: Optional[CognitionPacket] = None) -> bool:
        """Send via webhook."""
        if not self._webhook_sender:
            return False

        thread_id = target.reply_to_message_id if target else None
        return self._webhook_sender.send(content, thread_id=thread_id)

    def _send_via_bot(self, content: str, target: DestinationTarget,
                      packet: Optional[CognitionPacket] = None) -> bool:
        """Send via bot (requires discord.py and running bot)."""
        if not self._bot_client or not self._is_bot_connected():
            logger.warning("Bot not connected, cannot send via bot")
            return False

        try:
            import asyncio

            # Check if this is a DM - prefer user_id for DMs
            is_dm = target.metadata.get("is_dm", False) if target.metadata else False
            user_id = target.user_id if target else None

            # Log what we're working with for debugging
            logger.debug(f"Bot send: is_dm={is_dm}, user_id={user_id}, channel_id={target.channel_id}")

            async def _send():
                # Split long messages
                messages = self._split_message_for_discord(content)

                # For DMs, use user.send() - Discord DM channels require going through the user object
                if is_dm and user_id:
                    try:
                        user = await self._bot_client.fetch_user(int(user_id))
                        if user:
                            for msg in messages:
                                await user.send(msg)
                            logger.info(f"Sent {len(messages)} DM message(s) to user {user_id}")
                            return True
                        else:
                            logger.error(f"Could not find user {user_id} for DM")
                            return False
                    except Exception as e:
                        logger.error(f"Failed to send DM to user {user_id}: {e}")
                        return False

                # For guild channel messages
                if target.channel_id:
                    try:
                        channel = self._bot_client.get_channel(int(target.channel_id))
                        if not channel:
                            channel = await self._bot_client.fetch_channel(int(target.channel_id))
                        if channel:
                            for msg in messages:
                                await channel.send(msg)
                            logger.info(f"Sent {len(messages)} message(s) to channel {target.channel_id}")
                            return True
                        else:
                            logger.error(f"Could not find channel {target.channel_id}")
                            return False
                    except Exception as e:
                        logger.error(f"Failed to send to channel {target.channel_id}: {e}")
                        return False

                logger.error(f"No valid send target: is_dm={is_dm}, user_id={user_id}, channel_id={target.channel_id}")
                return False

            # Run async send in the bot's event loop
            if self._bot_client.loop and self._bot_client.loop.is_running():
                future = asyncio.run_coroutine_threadsafe(_send(), self._bot_client.loop)
                # Don't block - let the send happen asynchronously
                # Add a callback to log any errors
                def _on_send_done(fut):
                    try:
                        result = fut.result()
                        if not result:
                            logger.warning("Bot send returned False")
                    except Exception as e:
                        logger.error(f"Bot send failed in callback: {e}")
                future.add_done_callback(_on_send_done)
                return True  # Optimistically return success
            else:
                logger.error("Bot event loop not running")
                return False

        except Exception as e:
            logger.error(f"Bot send failed: {e}", exc_info=True)
            return False

    def _split_message_for_discord(self, content: str) -> List[str]:
        """Split message into chunks respecting Discord's 2000 char limit."""
        max_length = self.config.max_message_length

        if len(content) <= max_length:
            return [content]

        messages = []
        remaining = content

        while remaining:
            if len(remaining) <= max_length:
                messages.append(remaining)
                break

            # Try to split at newline
            split_point = remaining[:max_length].rfind('\n')
            if split_point < max_length // 2:
                # No good newline, split at space
                split_point = remaining[:max_length].rfind(' ')
            if split_point < max_length // 2:
                # No good space, hard split
                split_point = max_length

            messages.append(remaining[:split_point])
            remaining = remaining[split_point:].lstrip()

        return messages

    def _is_bot_connected(self) -> bool:
        """Check if bot is connected."""
        return self._bot_client is not None and getattr(self._bot_client, 'is_ready', lambda: False)()

    def set_message_callback(self, callback: Callable[[str, str, Dict[str, Any]], None]) -> None:
        """
        Set callback for incoming Discord messages (both channel and DM).

        Args:
            callback: Function(message_content, author_id, metadata) called on each message
                     metadata includes 'is_dm' boolean to distinguish DMs from channel messages
        """
        self._message_callback = callback

    @staticmethod
    def generate_dm_session_id(user_id: str) -> str:
        """Generate a consistent session ID for DM conversations with a user."""
        return f"discord_dm_{user_id}"

    @staticmethod
    def is_dm_session(session_id: str) -> bool:
        """Check if a session ID represents a DM conversation."""
        return session_id.startswith("discord_dm_")

    def start_bot_listener(self) -> bool:
        """
        Start the Discord bot listener in a background thread.
        Requires discord.py to be installed and DISCORD_BOT_TOKEN to be set.

        Returns:
            True if started successfully
        """
        if not self.config.bot_token:
            logger.error("Cannot start bot listener: DISCORD_BOT_TOKEN not set")
            return False

        try:
            import discord
            from discord.ext import commands
        except ImportError:
            logger.error("discord.py not installed. Run: pip install discord.py")
            return False

        # Create bot with intents (including DM support)
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guild_messages = True
        intents.dm_messages = True  # Required for receiving DMs
        intents.members = True  # Helps with user lookup for DM responses

        bot = commands.Bot(command_prefix="!", intents=intents)

        @bot.event
        async def on_ready():
            logger.info(f"Discord bot connected as {bot.user}")
            # Set presence to show as online with a status
            await bot.change_presence(
                status=discord.Status.online,
                activity=discord.Activity(
                    type=discord.ActivityType.watching,
                    name="over the studio"
                )
            )

        @bot.event
        async def on_message(message):
            # Ignore own messages
            if message.author == bot.user:
                return

            # Detect if this is a DM (no guild means DM or group DM)
            is_dm = message.guild is None

            # For DMs, always respond (if DM responses are enabled)
            # For channels, check if GAIA was mentioned or addressed
            should_respond = False

            if is_dm:
                # DMs always get a response if enabled
                should_respond = self.config.respond_to_dms
                if not should_respond:
                    logger.debug(f"Ignoring DM from {message.author.id} - DM responses disabled")
            else:
                # Check if GAIA was mentioned or message starts with @GAIA
                mentioned = bot.user in message.mentions
                addressed = message.content.lower().startswith(("@gaia", "gaia,", "gaia:"))
                should_respond = mentioned or addressed

            if should_respond:
                # Clean the message content (remove mentions)
                content = message.content
                for mention in message.mentions:
                    content = content.replace(f"<@{mention.id}>", "").replace(f"<@!{mention.id}>", "")
                content = content.strip()

                if content and self._message_callback:
                    # Generate appropriate session ID
                    if is_dm:
                        session_id = self.generate_dm_session_id(str(message.author.id))
                    else:
                        session_id = f"discord_channel_{message.channel.id}"

                    metadata = {
                        "channel_id": str(message.channel.id),
                        "guild_id": str(message.guild.id) if message.guild else None,
                        "author_name": message.author.display_name,
                        "author_id": str(message.author.id),
                        "message_id": str(message.id),
                        "session_id": session_id,
                        "is_dm": is_dm,
                        "addressed_to_gaia": not is_dm,  # DMs are implicitly addressed
                        "source": "discord_dm" if is_dm else "discord_channel",
                        # Note: Don't store discord.Message object - it's not JSON serializable
                    }
                    logger.info(f"Processing {'DM' if is_dm else 'channel message'} from {message.author.display_name} ({message.author.id})")

                    # Run callback in a thread pool to avoid blocking the Discord event loop
                    # This allows the bot to maintain heartbeat while AgentCore processes
                    import concurrent.futures
                    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                    executor.submit(self._message_callback, content, str(message.author.id), metadata)
                    executor.shutdown(wait=False)  # Don't wait, let it run in background

            # Process commands if any (only in guild channels)
            if not is_dm:
                await bot.process_commands(message)

        # Run bot in background thread
        def run_bot():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(bot.start(self.config.bot_token))
            except Exception as e:
                logger.error(f"Discord bot error: {e}")
            finally:
                loop.close()

        self._bot_client = bot
        self._bot_thread = threading.Thread(target=run_bot, daemon=True)
        self._bot_thread.start()

        logger.info("Discord bot listener started in background thread")
        return True

    def stop_bot_listener(self) -> None:
        """Stop the Discord bot listener."""
        if self._bot_client:
            asyncio.run_coroutine_threadsafe(
                self._bot_client.close(),
                self._bot_client.loop
            )
            self._bot_client = None
            logger.info("Discord bot listener stopped")