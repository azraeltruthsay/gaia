"""
Discord Integration Components for GAIA

This module provides common components for Discord integration,
such as configuration and a simple webhook sender.
"""

import os
import logging
from typing import Optional, List, Dict
from dataclasses import dataclass

import requests

logger = logging.getLogger("GAIA.Discord")

# --- Configuration ---

@dataclass
class DiscordConfig:
    """Discord connector configuration."""
    enabled: bool = True
    webhook_url: Optional[str] = None
    bot_token: Optional[str] = None
    default_channel_id: Optional[str] = None
    guild_id: Optional[str] = None
    bot_name: str = "GAIA"
    avatar_url: Optional[str] = None
    max_message_length: int = 2000  # Discord's limit
    respond_to_dms: bool = True  # Whether to respond to direct messages

    @classmethod
    def from_constants(cls) -> 'DiscordConfig':
        """
        Load configuration from gaia_constants.json first, then fall back to environment variables.
        This allows centralized config management while still supporting env overrides.
        """
        config_values = {}

        # Try to load from gaia_constants.json
        try:
            from gaia_core.config import Config
            gaia_config = Config()
            # INTEGRATIONS lives in gaia_config.constants, not as a direct attribute
            integrations = gaia_config.constants.get('INTEGRATIONS', {})
            discord_config = integrations.get('discord', {})

            if discord_config:
                config_values = {
                    'enabled': discord_config.get('enabled', True),
                    'webhook_url': discord_config.get('webhook_url'),
                    'bot_token': discord_config.get('bot_token'),
                    'default_channel_id': discord_config.get('default_channel_id'),
                    'bot_name': discord_config.get('bot_name', 'GAIA'),
                    'avatar_url': discord_config.get('avatar_url'),
                }
                logger.debug("Loaded Discord config from gaia_constants.json")
        except Exception as e:
            logger.debug(f"Could not load Discord config from constants: {e}")

        # Environment variables override constants
        return cls(
            enabled=config_values.get('enabled', True),
            webhook_url=os.getenv("DISCORD_WEBHOOK_URL") or config_values.get('webhook_url'),
            bot_token=os.getenv("DISCORD_BOT_TOKEN") or config_values.get('bot_token'),
            default_channel_id=os.getenv("DISCORD_CHANNEL_ID") or config_values.get('default_channel_id'),
            guild_id=os.getenv("DISCORD_GUILD_ID"),
            bot_name=os.getenv("DISCORD_BOT_NAME") or config_values.get('bot_name', 'GAIA'),
            avatar_url=os.getenv("DISCORD_AVATAR_URL") or config_values.get('avatar_url'),
            respond_to_dms=os.getenv("DISCORD_RESPOND_TO_DMS", "true").lower() in ("true", "1", "yes"),
        )

    @classmethod
    def from_env(cls) -> 'DiscordConfig':
        """Load configuration - checks constants first, then environment variables."""
        return cls.from_constants()


# --- Webhook-based Output (Simple, no dependencies) ---

class DiscordWebhookSender:
    """
    Simple Discord webhook sender using requests.
    No external dependencies required beyond requests.
    """

    def __init__(self, webhook_url: str, bot_name: str = "GAIA", avatar_url: Optional[str] = None):
        self.webhook_url = webhook_url
        self.bot_name = bot_name
        self.avatar_url = avatar_url
        self.max_length = 2000

    def send(self, content: str, username: Optional[str] = None,
             thread_id: Optional[str] = None) -> bool:
        """
        Send a message via webhook.

        Args:
            content: Message content
            username: Override username (defaults to bot_name)
            thread_id: Thread ID for threaded replies

        Returns:
            True if sent successfully
        """
        if not self.webhook_url:
            logger.error("No webhook URL configured")
            return False

        # Split long messages
        messages = self._split_message(content)

        for msg in messages:
            payload = {
                "content": msg,
                "username": username or self.bot_name,
            }

            if self.avatar_url:
                payload["avatar_url"] = self.avatar_url

            # Build URL with thread_id if provided
            url = self.webhook_url
            if thread_id:
                url = f"{self.webhook_url}?thread_id={thread_id}"

            try:
                response = requests.post(
                    url,
                    json=payload,
                    timeout=10
                )
                response.raise_for_status()
            except requests.RequestException as e:
                logger.error(f"Discord webhook send failed: {e}")
                return False

        logger.info(f"Sent {len(messages)} message(s) to Discord webhook")
        return True

    def send_embed(self, title: str, description: str, color: int = 0x7289DA,
                   fields: Optional[List[Dict[str, str]]] = None,
                   footer: Optional[str] = None) -> bool:
        """
        Send an embed message via webhook.

        Args:
            title: Embed title
            description: Embed description
            color: Embed color (hex integer)
            fields: List of {"name": str, "value": str, "inline": bool}
            footer: Footer text

        Returns:
            True if sent successfully
        """
        if not self.webhook_url:
            logger.error("No webhook URL configured")
            return False

        embed = {
            "title": title,
            "description": description[:4096],  # Discord limit
            "color": color,
        }

        if fields:
            embed["fields"] = fields[:25]  # Discord limit

        if footer:
            embed["footer"] = {"text": footer[:2048]}

        payload = {
            "username": self.bot_name,
            "embeds": [embed],
        }

        if self.avatar_url:
            payload["avatar_url"] = self.avatar_url

        try:
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            logger.info("Sent embed to Discord webhook")
            return True
        except requests.RequestException as e:
            logger.error(f"Discord webhook embed failed: {e}")
            return False

    def _split_message(self, content: str) -> List[str]:
        """Split message into chunks respecting Discord's limit."""
        if len(content) <= self.max_length:
            return [content]

        messages = []
        remaining = content

        while remaining:
            if len(remaining) <= self.max_length:
                messages.append(remaining)
                break

            # Try to split at newline
            split_point = remaining[:self.max_length].rfind('\n')
            if split_point < self.max_length // 2:
                # No good newline, split at space
                split_point = remaining[:self.max_length].rfind(' ')
            if split_point < self.max_length // 2:
                # No good space, hard split
                split_point = self.max_length

            messages.append(remaining[:split_point])
            remaining = remaining[split_point:].lstrip()

        return messages
