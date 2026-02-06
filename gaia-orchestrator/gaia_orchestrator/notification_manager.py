"""
Notification manager for GAIA Orchestrator.

Manages WebSocket connections and broadcasts notifications
to connected clients (gaia-web, Discord bot, etc.).
"""

import asyncio
import logging
from typing import List, Set
from datetime import datetime

from fastapi import WebSocket

from .models.schemas import Notification, NotificationType

logger = logging.getLogger("GAIA.Orchestrator.Notifications")


class NotificationManager:
    """Manages WebSocket connections and notification broadcasting."""

    def __init__(self):
        self._connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()
        self._history: List[Notification] = []
        self._max_history = 100

    async def connect(self, websocket: WebSocket) -> None:
        """Register a new WebSocket connection."""
        async with self._lock:
            self._connections.add(websocket)
            logger.info(f"WebSocket connected. Total connections: {len(self._connections)}")

    async def disconnect(self, websocket: WebSocket) -> None:
        """Remove a WebSocket connection."""
        async with self._lock:
            self._connections.discard(websocket)
            logger.info(f"WebSocket disconnected. Total connections: {len(self._connections)}")

    async def broadcast(self, notification: Notification) -> int:
        """
        Broadcast a notification to all connected clients.

        Returns the number of clients that received the notification.
        """
        # Add to history
        self._history.append(notification)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        # Serialize notification
        message = notification.model_dump_json()

        # Send to all connections
        sent_count = 0
        dead_connections = []

        async with self._lock:
            for websocket in self._connections:
                try:
                    await websocket.send_text(message)
                    sent_count += 1
                except Exception as e:
                    logger.warning(f"Failed to send to WebSocket: {e}")
                    dead_connections.append(websocket)

            # Remove dead connections
            for ws in dead_connections:
                self._connections.discard(ws)

        logger.info(
            f"Broadcast notification {notification.notification_type.value} "
            f"to {sent_count}/{len(self._connections)} clients"
        )

        return sent_count

    async def send_to(self, websocket: WebSocket, notification: Notification) -> bool:
        """Send a notification to a specific client."""
        try:
            message = notification.model_dump_json()
            await websocket.send_text(message)
            return True
        except Exception as e:
            logger.warning(f"Failed to send to WebSocket: {e}")
            return False

    def get_history(self, limit: int = 50) -> List[Notification]:
        """Get recent notification history."""
        return self._history[-limit:]

    @property
    def connection_count(self) -> int:
        """Get number of active connections."""
        return len(self._connections)

    # Convenience methods for common notifications

    async def notify_oracle_fallback(
        self,
        fallback_model: str,
        original_role: str,
        reason: str = ""
    ) -> int:
        """Send Oracle fallback notification."""
        notification = Notification(
            notification_type=NotificationType.ORACLE_FALLBACK,
            title="Cloud Inference Active",
            message=f"Using {fallback_model} for {original_role}",
            data={
                "fallback_model": fallback_model,
                "original_role": original_role,
                "reason": reason,
            }
        )
        return await self.broadcast(notification)

    async def notify_gpu_released(self, previous_owner: str, reason: str = "") -> int:
        """Send GPU released notification."""
        notification = Notification(
            notification_type=NotificationType.GPU_RELEASED,
            title="GPU Released",
            message=f"GPU released by {previous_owner}",
            data={
                "previous_owner": previous_owner,
                "reason": reason,
            }
        )
        return await self.broadcast(notification)

    async def notify_gpu_acquired(self, new_owner: str, reason: str = "") -> int:
        """Send GPU acquired notification."""
        notification = Notification(
            notification_type=NotificationType.GPU_ACQUIRED,
            title="GPU Acquired",
            message=f"GPU acquired by {new_owner}",
            data={
                "new_owner": new_owner,
                "reason": reason,
            }
        )
        return await self.broadcast(notification)

    async def notify_handoff_started(
        self,
        handoff_id: str,
        handoff_type: str,
        source: str,
        destination: str
    ) -> int:
        """Send handoff started notification."""
        notification = Notification(
            notification_type=NotificationType.HANDOFF_STARTED,
            title="GPU Handoff Started",
            message=f"Transferring GPU from {source} to {destination}",
            data={
                "handoff_id": handoff_id,
                "handoff_type": handoff_type,
                "source": source,
                "destination": destination,
            }
        )
        return await self.broadcast(notification)

    async def notify_handoff_completed(
        self,
        handoff_id: str,
        handoff_type: str,
        source: str,
        destination: str
    ) -> int:
        """Send handoff completed notification."""
        notification = Notification(
            notification_type=NotificationType.HANDOFF_COMPLETED,
            title="GPU Handoff Complete",
            message=f"GPU transferred to {destination}",
            data={
                "handoff_id": handoff_id,
                "handoff_type": handoff_type,
                "source": source,
                "destination": destination,
            }
        )
        return await self.broadcast(notification)

    async def notify_handoff_failed(
        self,
        handoff_id: str,
        error: str
    ) -> int:
        """Send handoff failed notification."""
        notification = Notification(
            notification_type=NotificationType.HANDOFF_FAILED,
            title="GPU Handoff Failed",
            message=f"Handoff failed: {error}",
            data={
                "handoff_id": handoff_id,
                "error": error,
            }
        )
        return await self.broadcast(notification)

    async def notify_service_error(
        self,
        service: str,
        error: str
    ) -> int:
        """Send service error notification."""
        notification = Notification(
            notification_type=NotificationType.SERVICE_ERROR,
            title=f"{service} Error",
            message=error,
            data={
                "service": service,
                "error": error,
            }
        )
        return await self.broadcast(notification)
