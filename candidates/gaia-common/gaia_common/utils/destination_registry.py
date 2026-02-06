"""
Destination Registry - Spinal Column Output Routing

This module manages output destinations for GAIA's responses.
All outputs flow through this registry, which dispatches to registered
destination connectors (CLI, Web, Discord, etc.).

Architecture:
    AgentCore → ExternalVoice → StreamBus → OutputRouter → DestinationRegistry → Connectors

Each connector implements a simple interface:
    - send(content, metadata) -> bool
    - send_stream(token_generator, metadata) -> bool
    - is_available() -> bool
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Generator, Callable
from dataclasses import dataclass, field
from enum import Enum

from gaia_common.protocols.cognition_packet import (
    OutputDestination,
    DestinationTarget,
    OutputRouting,
    CognitionPacket
)

logger = logging.getLogger("GAIA.DestinationRegistry")


# --- Base Connector Interface ---

class DestinationConnector(ABC):
    """
    Abstract base class for destination connectors.
    Each connector handles delivery to a specific destination type.
    """

    def __init__(self, name: str, destination_type: OutputDestination):
        self.name = name
        self.destination_type = destination_type
        self._enabled = True

    @abstractmethod
    def send(self, content: str, target: DestinationTarget,
             packet: Optional[CognitionPacket] = None) -> bool:
        """
        Send content to this destination.

        Args:
            content: The response content to send
            target: Destination target with channel/user info
            packet: Optional packet for additional context

        Returns:
            True if sent successfully
        """
        pass

    @abstractmethod
    def send_stream(self, token_generator: Generator[str, None, None],
                    target: DestinationTarget,
                    packet: Optional[CognitionPacket] = None) -> bool:
        """
        Stream tokens to this destination.

        Args:
            token_generator: Generator yielding tokens
            target: Destination target with channel/user info
            packet: Optional packet for additional context

        Returns:
            True if streaming completed successfully
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this connector is available and ready."""
        pass

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled


# --- Built-in CLI Connector ---

class CLIConnector(DestinationConnector):
    """
    Connector for command-line interface output.
    This is the default destination for rescue shell and terminal interactions.
    """

    def __init__(self, output_fn: Optional[Callable] = None):
        super().__init__("cli", OutputDestination.CLI)
        self._output_fn = output_fn or print

    def send(self, content: str, target: DestinationTarget,
             packet: Optional[CognitionPacket] = None) -> bool:
        try:
            self._output_fn(content)
            return True
        except Exception as e:
            logger.error(f"CLI send failed: {e}")
            return False

    def send_stream(self, token_generator: Generator[str, None, None],
                    target: DestinationTarget,
                    packet: Optional[CognitionPacket] = None) -> bool:
        try:
            for token in token_generator:
                self._output_fn(token, end="", flush=True)
            self._output_fn("")  # Final newline
            return True
        except Exception as e:
            logger.error(f"CLI stream failed: {e}")
            return False

    def is_available(self) -> bool:
        return True  # CLI is always available


# --- Log Connector (for audit/logging only) ---

class LogConnector(DestinationConnector):
    """
    Connector that logs output without displaying to users.
    Useful for audit trails and background processing.
    """

    def __init__(self, log_level: int = logging.INFO):
        super().__init__("log", OutputDestination.LOG)
        self._log_level = log_level

    def send(self, content: str, target: DestinationTarget,
             packet: Optional[CognitionPacket] = None) -> bool:
        try:
            logger.log(self._log_level, f"[OUTPUT] {content[:500]}{'...' if len(content) > 500 else ''}")
            return True
        except Exception as e:
            logger.error(f"Log send failed: {e}")
            return False

    def send_stream(self, token_generator: Generator[str, None, None],
                    target: DestinationTarget,
                    packet: Optional[CognitionPacket] = None) -> bool:
        try:
            full_content = "".join(token_generator)
            return self.send(full_content, target, packet)
        except Exception as e:
            logger.error(f"Log stream failed: {e}")
            return False

    def is_available(self) -> bool:
        return True


# --- Destination Registry ---

class DestinationRegistry:
    """
    Central registry for all output destination connectors.
    Handles routing responses to appropriate destinations based on packet routing info.
    """

    _instance: Optional['DestinationRegistry'] = None

    def __init__(self):
        self._connectors: Dict[OutputDestination, DestinationConnector] = {}
        self._default_destination = OutputDestination.CLI
        self._hooks: List[Callable[[str, DestinationTarget, Optional[CognitionPacket]], None]] = []

        # Register built-in connectors
        self.register(CLIConnector())
        self.register(LogConnector())

    @classmethod
    def get_instance(cls) -> 'DestinationRegistry':
        """Get or create the singleton registry instance."""
        if cls._instance is None:
            cls._instance = DestinationRegistry()
        return cls._instance

    def register(self, connector: DestinationConnector) -> None:
        """Register a destination connector."""
        self._connectors[connector.destination_type] = connector
        logger.info(f"Registered destination connector: {connector.name} ({connector.destination_type.value})")

    def unregister(self, destination: OutputDestination) -> None:
        """Unregister a destination connector."""
        if destination in self._connectors:
            del self._connectors[destination]
            logger.info(f"Unregistered destination: {destination.value}")

    def get_connector(self, destination: OutputDestination) -> Optional[DestinationConnector]:
        """Get a connector by destination type."""
        return self._connectors.get(destination)

    def list_destinations(self) -> List[OutputDestination]:
        """List all registered destinations."""
        return list(self._connectors.keys())

    def list_available(self) -> List[OutputDestination]:
        """List all available (enabled and ready) destinations."""
        return [
            dest for dest, conn in self._connectors.items()
            if conn.enabled and conn.is_available()
        ]

    def add_hook(self, hook: Callable[[str, DestinationTarget, Optional[CognitionPacket]], None]) -> None:
        """Add a hook that's called for every output (for logging, metrics, etc.)."""
        self._hooks.append(hook)

    def set_default(self, destination: OutputDestination) -> None:
        """Set the default destination for packets without routing info."""
        self._default_destination = destination

    def route(self, content: str, packet: Optional[CognitionPacket] = None,
              override_destination: Optional[OutputDestination] = None) -> Dict[OutputDestination, bool]:
        """
        Route content to appropriate destinations based on packet routing or override.

        Args:
            content: The response content to route
            packet: Optional packet with routing information
            override_destination: Optional override to force a specific destination

        Returns:
            Dict mapping each destination to success status
        """
        results: Dict[OutputDestination, bool] = {}

        # Determine destinations
        targets: List[DestinationTarget] = []

        if override_destination:
            # Override takes precedence
            targets.append(DestinationTarget(destination=override_destination))
        elif packet and packet.header.output_routing:
            # Use packet routing
            routing = packet.header.output_routing
            targets.append(routing.primary)
            targets.extend(routing.secondary)
        else:
            # Fall back to default
            targets.append(DestinationTarget(destination=self._default_destination))

        # Route to each destination
        for target in targets:
            dest = target.destination

            # Handle broadcast
            if dest == OutputDestination.BROADCAST:
                for available_dest in self.list_available():
                    if available_dest != OutputDestination.LOG:  # Don't double-log
                        results[available_dest] = self._send_to_destination(
                            content, DestinationTarget(destination=available_dest), packet
                        )
            else:
                results[dest] = self._send_to_destination(content, target, packet)

        # Call hooks
        for hook in self._hooks:
            try:
                for target in targets:
                    hook(content, target, packet)
            except Exception as e:
                logger.warning(f"Output hook failed: {e}")

        return results

    def route_stream(self, token_generator: Generator[str, None, None],
                     packet: Optional[CognitionPacket] = None,
                     override_destination: Optional[OutputDestination] = None) -> Dict[OutputDestination, bool]:
        """
        Route streaming content to appropriate destinations.
        Note: Streaming only goes to the primary destination to avoid complexity.

        Args:
            token_generator: Generator yielding tokens
            packet: Optional packet with routing information
            override_destination: Optional override to force a specific destination

        Returns:
            Dict mapping each destination to success status
        """
        results: Dict[OutputDestination, bool] = {}

        # Determine primary destination
        if override_destination:
            target = DestinationTarget(destination=override_destination)
        elif packet and packet.header.output_routing:
            target = packet.header.output_routing.primary
        else:
            target = DestinationTarget(destination=self._default_destination)

        dest = target.destination
        connector = self._connectors.get(dest)

        if not connector:
            logger.warning(f"No connector for destination: {dest.value}")
            results[dest] = False
            return results

        if not connector.enabled or not connector.is_available():
            logger.warning(f"Destination not available: {dest.value}")
            results[dest] = False
            return results

        try:
            results[dest] = connector.send_stream(token_generator, target, packet)
        except Exception as e:
            logger.error(f"Stream routing failed for {dest.value}: {e}")
            results[dest] = False

        return results

    def _send_to_destination(self, content: str, target: DestinationTarget,
                             packet: Optional[CognitionPacket]) -> bool:
        """Send content to a specific destination."""
        connector = self._connectors.get(target.destination)

        if not connector:
            logger.warning(f"No connector for destination: {target.destination.value}")
            return False

        if not connector.enabled or not connector.is_available():
            logger.warning(f"Destination not available: {target.destination.value}")
            return False

        try:
            return connector.send(content, target, packet)
        except Exception as e:
            logger.error(f"Send failed for {target.destination.value}: {e}")
            return False


# --- Convenience Functions ---

def get_registry() -> DestinationRegistry:
    """Get the global destination registry instance."""
    return DestinationRegistry.get_instance()


def route_output(content: str, packet: Optional[CognitionPacket] = None,
                 destination: Optional[OutputDestination] = None) -> Dict[OutputDestination, bool]:
    """Convenience function to route output through the global registry."""
    return get_registry().route(content, packet, destination)


def register_connector(connector: DestinationConnector) -> None:
    """Convenience function to register a connector with the global registry."""
    get_registry().register(connector)
