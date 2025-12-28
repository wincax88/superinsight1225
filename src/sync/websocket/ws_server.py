"""
WebSocket Server Module.

Provides WebSocket server implementation for real-time data synchronization,
including connection management, authentication, message routing, and broadcasting.
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from fastapi import WebSocket, WebSocketDisconnect, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================

class ConnectionState(str, Enum):
    """WebSocket connection state."""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATED = "authenticated"
    SUBSCRIBED = "subscribed"
    DISCONNECTING = "disconnecting"
    DISCONNECTED = "disconnected"


class MessageType(str, Enum):
    """WebSocket message types."""
    # Client to server
    AUTH = "auth"
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    PUSH_DATA = "push_data"
    PING = "ping"

    # Server to client
    AUTH_SUCCESS = "auth_success"
    AUTH_FAILED = "auth_failed"
    SUBSCRIBED = "subscribed"
    UNSUBSCRIBED = "unsubscribed"
    DATA_UPDATE = "data_update"
    SYNC_EVENT = "sync_event"
    ERROR = "error"
    PONG = "pong"

    # Broadcast
    BROADCAST = "broadcast"
    SYSTEM_NOTIFICATION = "system_notification"


class SubscriptionType(str, Enum):
    """Subscription types for data streams."""
    SYNC_JOB = "sync_job"
    SYNC_EXECUTION = "sync_execution"
    DATA_CHANGES = "data_changes"
    CONFLICTS = "conflicts"
    ALERTS = "alerts"
    ALL = "all"


# =============================================================================
# Models
# =============================================================================

class WebSocketMessage(BaseModel):
    """WebSocket message model."""
    type: MessageType
    payload: Dict[str, Any] = Field(default_factory=dict)
    message_id: str = Field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:12]}")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = None

    class Config:
        use_enum_values = True


class AuthPayload(BaseModel):
    """Authentication payload."""
    token: str
    tenant_id: str
    user_id: Optional[str] = None


class SubscriptionPayload(BaseModel):
    """Subscription payload."""
    subscription_type: SubscriptionType
    filters: Dict[str, Any] = Field(default_factory=dict)
    job_id: Optional[str] = None
    execution_id: Optional[str] = None


@dataclass
class ConnectionInfo:
    """WebSocket connection information."""
    connection_id: str
    websocket: WebSocket
    state: ConnectionState = ConnectionState.CONNECTING
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None
    subscriptions: Set[str] = field(default_factory=set)
    connected_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = datetime.utcnow()


# =============================================================================
# Connection Manager
# =============================================================================

class WebSocketConnectionManager:
    """
    Manages WebSocket connections with authentication, subscriptions, and broadcasting.

    Features:
    - Connection lifecycle management
    - Authentication and authorization
    - Subscription-based message routing
    - Broadcasting to multiple clients
    - Connection health monitoring
    """

    def __init__(
        self,
        auth_handler: Optional[Callable] = None,
        max_connections_per_tenant: int = 100,
        ping_interval: float = 30.0,
        ping_timeout: float = 10.0
    ):
        """
        Initialize connection manager.

        Args:
            auth_handler: Optional authentication handler function
            max_connections_per_tenant: Maximum connections per tenant
            ping_interval: Interval for ping messages (seconds)
            ping_timeout: Timeout for ping responses (seconds)
        """
        self._connections: Dict[str, ConnectionInfo] = {}
        self._tenant_connections: Dict[str, Set[str]] = {}
        self._subscription_connections: Dict[str, Set[str]] = {}
        self._auth_handler = auth_handler
        self._max_connections_per_tenant = max_connections_per_tenant
        self._ping_interval = ping_interval
        self._ping_timeout = ping_timeout
        self._running = False
        self._health_check_task: Optional[asyncio.Task] = None
        self._message_handlers: Dict[MessageType, Callable] = {}

        # Register default handlers
        self._register_default_handlers()

    def _register_default_handlers(self) -> None:
        """Register default message handlers."""
        self._message_handlers[MessageType.PING] = self._handle_ping
        self._message_handlers[MessageType.AUTH] = self._handle_auth
        self._message_handlers[MessageType.SUBSCRIBE] = self._handle_subscribe
        self._message_handlers[MessageType.UNSUBSCRIBE] = self._handle_unsubscribe

    async def start(self) -> None:
        """Start the connection manager."""
        if self._running:
            return

        self._running = True
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        logger.info("WebSocket connection manager started")

    async def stop(self) -> None:
        """Stop the connection manager."""
        self._running = False

        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        # Close all connections
        for conn_id in list(self._connections.keys()):
            await self.disconnect(conn_id, reason="Server shutdown")

        logger.info("WebSocket connection manager stopped")

    async def connect(
        self,
        websocket: WebSocket,
        connection_id: Optional[str] = None
    ) -> ConnectionInfo:
        """
        Accept and register a new WebSocket connection.

        Args:
            websocket: FastAPI WebSocket instance
            connection_id: Optional custom connection ID

        Returns:
            ConnectionInfo for the new connection
        """
        await websocket.accept()

        conn_id = connection_id or f"ws_{uuid.uuid4().hex[:12]}"

        connection = ConnectionInfo(
            connection_id=conn_id,
            websocket=websocket,
            state=ConnectionState.CONNECTED
        )

        self._connections[conn_id] = connection

        logger.info(f"WebSocket connected: {conn_id}")

        # Send connection acknowledgment
        await self._send_message(connection, WebSocketMessage(
            type=MessageType.SYSTEM_NOTIFICATION,
            payload={
                "status": "connected",
                "connection_id": conn_id,
                "message": "Connection established. Please authenticate."
            }
        ))

        return connection

    async def disconnect(
        self,
        connection_id: str,
        reason: str = "Normal closure"
    ) -> None:
        """
        Disconnect and cleanup a WebSocket connection.

        Args:
            connection_id: ID of connection to disconnect
            reason: Reason for disconnection
        """
        connection = self._connections.pop(connection_id, None)

        if not connection:
            return

        connection.state = ConnectionState.DISCONNECTED

        # Remove from tenant tracking
        if connection.tenant_id:
            tenant_conns = self._tenant_connections.get(connection.tenant_id, set())
            tenant_conns.discard(connection_id)

        # Remove from subscription tracking
        for sub_key in connection.subscriptions:
            sub_conns = self._subscription_connections.get(sub_key, set())
            sub_conns.discard(connection_id)

        # Close websocket
        try:
            await connection.websocket.close(code=status.WS_1000_NORMAL_CLOSURE)
        except Exception as e:
            logger.debug(f"Error closing websocket: {e}")

        logger.info(f"WebSocket disconnected: {connection_id}, reason: {reason}")

    async def authenticate(
        self,
        connection_id: str,
        token: str,
        tenant_id: str,
        user_id: Optional[str] = None
    ) -> bool:
        """
        Authenticate a WebSocket connection.

        Args:
            connection_id: Connection to authenticate
            token: Authentication token
            tenant_id: Tenant ID
            user_id: Optional user ID

        Returns:
            True if authentication successful
        """
        connection = self._connections.get(connection_id)
        if not connection:
            return False

        # Check tenant connection limit
        tenant_conns = self._tenant_connections.get(tenant_id, set())
        if len(tenant_conns) >= self._max_connections_per_tenant:
            await self._send_message(connection, WebSocketMessage(
                type=MessageType.AUTH_FAILED,
                payload={
                    "error": "connection_limit_exceeded",
                    "message": f"Maximum connections ({self._max_connections_per_tenant}) exceeded for tenant"
                }
            ))
            return False

        # Custom authentication handler
        if self._auth_handler:
            try:
                auth_result = await self._auth_handler(token, tenant_id, user_id)
                if not auth_result:
                    await self._send_message(connection, WebSocketMessage(
                        type=MessageType.AUTH_FAILED,
                        payload={"error": "invalid_credentials"}
                    ))
                    return False
            except Exception as e:
                logger.error(f"Authentication error: {e}")
                await self._send_message(connection, WebSocketMessage(
                    type=MessageType.AUTH_FAILED,
                    payload={"error": "authentication_error", "message": str(e)}
                ))
                return False

        # Update connection info
        connection.tenant_id = tenant_id
        connection.user_id = user_id
        connection.state = ConnectionState.AUTHENTICATED

        # Track tenant connection
        if tenant_id not in self._tenant_connections:
            self._tenant_connections[tenant_id] = set()
        self._tenant_connections[tenant_id].add(connection_id)

        await self._send_message(connection, WebSocketMessage(
            type=MessageType.AUTH_SUCCESS,
            payload={
                "tenant_id": tenant_id,
                "user_id": user_id,
                "message": "Authentication successful"
            }
        ))

        logger.info(f"Connection authenticated: {connection_id}, tenant: {tenant_id}")
        return True

    async def subscribe(
        self,
        connection_id: str,
        subscription_type: SubscriptionType,
        filters: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Subscribe a connection to a data stream.

        Args:
            connection_id: Connection to subscribe
            subscription_type: Type of subscription
            filters: Optional filters for the subscription

        Returns:
            True if subscription successful
        """
        connection = self._connections.get(connection_id)
        if not connection or connection.state != ConnectionState.AUTHENTICATED:
            return False

        # Create subscription key
        sub_key = self._create_subscription_key(
            subscription_type,
            connection.tenant_id,
            filters
        )

        # Add to subscriptions
        connection.subscriptions.add(sub_key)

        if sub_key not in self._subscription_connections:
            self._subscription_connections[sub_key] = set()
        self._subscription_connections[sub_key].add(connection_id)

        connection.state = ConnectionState.SUBSCRIBED

        await self._send_message(connection, WebSocketMessage(
            type=MessageType.SUBSCRIBED,
            payload={
                "subscription_type": subscription_type.value,
                "subscription_key": sub_key,
                "filters": filters or {}
            }
        ))

        logger.debug(f"Connection {connection_id} subscribed to {sub_key}")
        return True

    async def unsubscribe(
        self,
        connection_id: str,
        subscription_type: SubscriptionType,
        filters: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Unsubscribe a connection from a data stream.

        Args:
            connection_id: Connection to unsubscribe
            subscription_type: Type of subscription
            filters: Optional filters for the subscription

        Returns:
            True if unsubscription successful
        """
        connection = self._connections.get(connection_id)
        if not connection:
            return False

        sub_key = self._create_subscription_key(
            subscription_type,
            connection.tenant_id,
            filters
        )

        connection.subscriptions.discard(sub_key)

        if sub_key in self._subscription_connections:
            self._subscription_connections[sub_key].discard(connection_id)

        await self._send_message(connection, WebSocketMessage(
            type=MessageType.UNSUBSCRIBED,
            payload={
                "subscription_type": subscription_type.value,
                "subscription_key": sub_key
            }
        ))

        return True

    async def broadcast_to_subscription(
        self,
        subscription_type: SubscriptionType,
        tenant_id: str,
        message: WebSocketMessage,
        filters: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Broadcast a message to all connections subscribed to a type.

        Args:
            subscription_type: Target subscription type
            tenant_id: Target tenant
            message: Message to broadcast
            filters: Optional filters

        Returns:
            Number of connections that received the message
        """
        sub_key = self._create_subscription_key(subscription_type, tenant_id, filters)
        connection_ids = self._subscription_connections.get(sub_key, set())

        sent_count = 0
        for conn_id in list(connection_ids):
            connection = self._connections.get(conn_id)
            if connection:
                try:
                    await self._send_message(connection, message)
                    sent_count += 1
                except Exception as e:
                    logger.warning(f"Failed to send to {conn_id}: {e}")

        return sent_count

    async def broadcast_to_tenant(
        self,
        tenant_id: str,
        message: WebSocketMessage
    ) -> int:
        """
        Broadcast a message to all connections of a tenant.

        Args:
            tenant_id: Target tenant
            message: Message to broadcast

        Returns:
            Number of connections that received the message
        """
        connection_ids = self._tenant_connections.get(tenant_id, set())

        sent_count = 0
        for conn_id in list(connection_ids):
            connection = self._connections.get(conn_id)
            if connection:
                try:
                    await self._send_message(connection, message)
                    sent_count += 1
                except Exception as e:
                    logger.warning(f"Failed to send to {conn_id}: {e}")

        return sent_count

    async def broadcast_all(self, message: WebSocketMessage) -> int:
        """
        Broadcast a message to all connected clients.

        Args:
            message: Message to broadcast

        Returns:
            Number of connections that received the message
        """
        sent_count = 0
        for connection in self._connections.values():
            try:
                await self._send_message(connection, message)
                sent_count += 1
            except Exception as e:
                logger.warning(f"Failed to send to {connection.connection_id}: {e}")

        return sent_count

    async def send_to_connection(
        self,
        connection_id: str,
        message: WebSocketMessage
    ) -> bool:
        """
        Send a message to a specific connection.

        Args:
            connection_id: Target connection
            message: Message to send

        Returns:
            True if message was sent successfully
        """
        connection = self._connections.get(connection_id)
        if not connection:
            return False

        try:
            await self._send_message(connection, message)
            return True
        except Exception as e:
            logger.warning(f"Failed to send to {connection_id}: {e}")
            return False

    async def handle_message(
        self,
        connection_id: str,
        message_data: Dict[str, Any]
    ) -> None:
        """
        Handle an incoming WebSocket message.

        Args:
            connection_id: Source connection
            message_data: Message data
        """
        connection = self._connections.get(connection_id)
        if not connection:
            return

        connection.update_activity()

        try:
            message = WebSocketMessage(**message_data)

            handler = self._message_handlers.get(message.type)
            if handler:
                await handler(connection, message)
            else:
                logger.warning(f"No handler for message type: {message.type}")
                await self._send_message(connection, WebSocketMessage(
                    type=MessageType.ERROR,
                    payload={"error": "unknown_message_type", "type": message.type}
                ))

        except Exception as e:
            logger.error(f"Error handling message: {e}")
            await self._send_message(connection, WebSocketMessage(
                type=MessageType.ERROR,
                payload={"error": "message_processing_error", "message": str(e)}
            ))

    def register_handler(
        self,
        message_type: MessageType,
        handler: Callable
    ) -> None:
        """
        Register a custom message handler.

        Args:
            message_type: Type of message to handle
            handler: Handler function (async)
        """
        self._message_handlers[message_type] = handler

    def get_connection(self, connection_id: str) -> Optional[ConnectionInfo]:
        """Get connection info by ID."""
        return self._connections.get(connection_id)

    def get_tenant_connections(self, tenant_id: str) -> List[ConnectionInfo]:
        """Get all connections for a tenant."""
        conn_ids = self._tenant_connections.get(tenant_id, set())
        return [
            self._connections[cid]
            for cid in conn_ids
            if cid in self._connections
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get connection manager statistics."""
        return {
            "total_connections": len(self._connections),
            "tenants": len(self._tenant_connections),
            "subscriptions": len(self._subscription_connections),
            "connections_by_state": self._get_connections_by_state(),
            "connections_by_tenant": {
                tenant: len(conns)
                for tenant, conns in self._tenant_connections.items()
            }
        }

    # =========================================================================
    # Private methods
    # =========================================================================

    async def _send_message(
        self,
        connection: ConnectionInfo,
        message: WebSocketMessage
    ) -> None:
        """Send a message to a connection."""
        try:
            await connection.websocket.send_json(message.model_dump(mode="json"))
        except Exception as e:
            logger.error(f"Error sending message to {connection.connection_id}: {e}")
            raise

    async def _handle_ping(
        self,
        connection: ConnectionInfo,
        message: WebSocketMessage
    ) -> None:
        """Handle ping message."""
        await self._send_message(connection, WebSocketMessage(
            type=MessageType.PONG,
            payload={"timestamp": datetime.utcnow().isoformat()},
            correlation_id=message.message_id
        ))

    async def _handle_auth(
        self,
        connection: ConnectionInfo,
        message: WebSocketMessage
    ) -> None:
        """Handle authentication message."""
        try:
            auth = AuthPayload(**message.payload)
            await self.authenticate(
                connection.connection_id,
                auth.token,
                auth.tenant_id,
                auth.user_id
            )
        except Exception as e:
            await self._send_message(connection, WebSocketMessage(
                type=MessageType.AUTH_FAILED,
                payload={"error": "invalid_auth_payload", "message": str(e)}
            ))

    async def _handle_subscribe(
        self,
        connection: ConnectionInfo,
        message: WebSocketMessage
    ) -> None:
        """Handle subscribe message."""
        try:
            sub = SubscriptionPayload(**message.payload)
            await self.subscribe(
                connection.connection_id,
                sub.subscription_type,
                sub.filters
            )
        except Exception as e:
            await self._send_message(connection, WebSocketMessage(
                type=MessageType.ERROR,
                payload={"error": "subscription_failed", "message": str(e)}
            ))

    async def _handle_unsubscribe(
        self,
        connection: ConnectionInfo,
        message: WebSocketMessage
    ) -> None:
        """Handle unsubscribe message."""
        try:
            sub = SubscriptionPayload(**message.payload)
            await self.unsubscribe(
                connection.connection_id,
                sub.subscription_type,
                sub.filters
            )
        except Exception as e:
            await self._send_message(connection, WebSocketMessage(
                type=MessageType.ERROR,
                payload={"error": "unsubscription_failed", "message": str(e)}
            ))

    def _create_subscription_key(
        self,
        subscription_type: SubscriptionType,
        tenant_id: Optional[str],
        filters: Optional[Dict[str, Any]]
    ) -> str:
        """Create a unique subscription key."""
        base = f"{subscription_type.value}:{tenant_id or 'global'}"

        if filters:
            filter_str = ":".join(f"{k}={v}" for k, v in sorted(filters.items()))
            return f"{base}:{filter_str}"

        return base

    def _get_connections_by_state(self) -> Dict[str, int]:
        """Get connection count by state."""
        counts: Dict[str, int] = {}
        for conn in self._connections.values():
            state = conn.state.value
            counts[state] = counts.get(state, 0) + 1
        return counts

    async def _health_check_loop(self) -> None:
        """Periodic health check for connections."""
        while self._running:
            try:
                await asyncio.sleep(self._ping_interval)

                for conn_id in list(self._connections.keys()):
                    connection = self._connections.get(conn_id)
                    if not connection:
                        continue

                    # Check for stale connections
                    inactive_time = (datetime.utcnow() - connection.last_activity).total_seconds()
                    if inactive_time > self._ping_interval * 3:
                        logger.warning(f"Connection {conn_id} is stale, disconnecting")
                        await self.disconnect(conn_id, reason="Connection timeout")
                        continue

                    # Send ping
                    try:
                        await self._send_message(connection, WebSocketMessage(
                            type=MessageType.PING,
                            payload={"timestamp": datetime.utcnow().isoformat()}
                        ))
                    except Exception:
                        await self.disconnect(conn_id, reason="Ping failed")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")


# =============================================================================
# Global Connection Manager Instance
# =============================================================================

_connection_manager: Optional[WebSocketConnectionManager] = None


def get_connection_manager() -> WebSocketConnectionManager:
    """Get the global connection manager instance."""
    global _connection_manager
    if _connection_manager is None:
        _connection_manager = WebSocketConnectionManager()
    return _connection_manager


def set_connection_manager(manager: WebSocketConnectionManager) -> None:
    """Set the global connection manager instance."""
    global _connection_manager
    _connection_manager = manager


__all__ = [
    "WebSocketConnectionManager",
    "ConnectionInfo",
    "ConnectionState",
    "MessageType",
    "SubscriptionType",
    "WebSocketMessage",
    "AuthPayload",
    "SubscriptionPayload",
    "get_connection_manager",
    "set_connection_manager",
]
