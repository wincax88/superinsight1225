"""
Knowledge Graph WebSocket API.

Task 14: Real-time updates via WebSocket for Knowledge Graph system.
Provides live streaming of graph changes to connected clients.
"""

import asyncio
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional, Set
from uuid import UUID, uuid4
from dataclasses import dataclass, field

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/knowledge-graph/ws", tags=["Knowledge Graph WebSocket"])


# =============================================================================
# Event Types and Models
# =============================================================================

class EventType(str, Enum):
    """Types of graph events."""
    ENTITY_CREATED = "entity.created"
    ENTITY_UPDATED = "entity.updated"
    ENTITY_DELETED = "entity.deleted"
    RELATION_CREATED = "relation.created"
    RELATION_UPDATED = "relation.updated"
    RELATION_DELETED = "relation.deleted"
    QUERY_RESULT = "query.result"
    EXTRACTION_COMPLETE = "extraction.complete"
    HEARTBEAT = "heartbeat"
    ERROR = "error"
    SUBSCRIBED = "subscribed"
    UNSUBSCRIBED = "unsubscribed"


class SubscriptionType(str, Enum):
    """Types of subscriptions."""
    ALL = "all"
    ENTITIES = "entities"
    RELATIONS = "relations"
    ENTITY = "entity"
    ENTITY_TYPE = "entity_type"
    RELATION_TYPE = "relation_type"
    QUERY = "query"


class GraphEvent(BaseModel):
    """Graph event message."""
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    event_type: EventType
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    data: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}


class SubscribeRequest(BaseModel):
    """Subscription request."""
    subscription_type: SubscriptionType
    filters: Dict[str, Any] = {}


class UnsubscribeRequest(BaseModel):
    """Unsubscription request."""
    subscription_id: str


class ClientMessage(BaseModel):
    """Message from client."""
    action: str
    payload: Dict[str, Any] = {}
    request_id: Optional[str] = None


# =============================================================================
# Subscription Management
# =============================================================================

@dataclass
class Subscription:
    """Represents a client subscription."""
    id: str
    subscription_type: SubscriptionType
    filters: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ConnectedClient:
    """Represents a connected WebSocket client."""
    id: str
    websocket: WebSocket
    subscriptions: Dict[str, Subscription] = field(default_factory=dict)
    tenant_id: Optional[str] = None
    connected_at: datetime = field(default_factory=datetime.utcnow)
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)


class ConnectionManager:
    """Manages WebSocket connections and subscriptions."""

    def __init__(self):
        self._clients: Dict[str, ConnectedClient] = {}
        self._lock = asyncio.Lock()
        self._heartbeat_interval = 30  # seconds
        self._heartbeat_task: Optional[asyncio.Task] = None

    async def connect(
        self,
        websocket: WebSocket,
        client_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> str:
        """Accept a new WebSocket connection."""
        await websocket.accept()

        client_id = client_id or str(uuid4())

        async with self._lock:
            client = ConnectedClient(
                id=client_id,
                websocket=websocket,
                tenant_id=tenant_id,
            )
            self._clients[client_id] = client

        logger.info(f"Client connected: {client_id} (tenant: {tenant_id})")

        # Send welcome message
        await self.send_to_client(client_id, GraphEvent(
            event_type=EventType.SUBSCRIBED,
            data={
                "client_id": client_id,
                "message": "Connected to Knowledge Graph WebSocket",
            },
        ))

        # Start heartbeat if not running
        if self._heartbeat_task is None or self._heartbeat_task.done():
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        return client_id

    async def disconnect(self, client_id: str) -> None:
        """Handle client disconnection."""
        async with self._lock:
            if client_id in self._clients:
                del self._clients[client_id]
                logger.info(f"Client disconnected: {client_id}")

    async def subscribe(
        self,
        client_id: str,
        subscription_type: SubscriptionType,
        filters: Dict[str, Any] = None,
    ) -> str:
        """Add a subscription for a client."""
        subscription_id = str(uuid4())

        async with self._lock:
            if client_id not in self._clients:
                raise ValueError(f"Client not found: {client_id}")

            subscription = Subscription(
                id=subscription_id,
                subscription_type=subscription_type,
                filters=filters or {},
            )
            self._clients[client_id].subscriptions[subscription_id] = subscription

        logger.info(
            f"Client {client_id} subscribed: {subscription_type} "
            f"(id: {subscription_id})"
        )

        return subscription_id

    async def unsubscribe(self, client_id: str, subscription_id: str) -> bool:
        """Remove a subscription."""
        async with self._lock:
            if client_id not in self._clients:
                return False

            client = self._clients[client_id]
            if subscription_id in client.subscriptions:
                del client.subscriptions[subscription_id]
                logger.info(f"Client {client_id} unsubscribed: {subscription_id}")
                return True

        return False

    async def send_to_client(
        self,
        client_id: str,
        event: GraphEvent,
    ) -> bool:
        """Send an event to a specific client."""
        async with self._lock:
            if client_id not in self._clients:
                return False

            client = self._clients[client_id]

        try:
            await client.websocket.send_json(event.model_dump(mode="json"))
            return True
        except Exception as e:
            logger.error(f"Failed to send to client {client_id}: {e}")
            await self.disconnect(client_id)
            return False

    async def broadcast(
        self,
        event: GraphEvent,
        tenant_id: Optional[str] = None,
    ) -> int:
        """Broadcast an event to matching subscribers."""
        sent_count = 0

        async with self._lock:
            clients = list(self._clients.values())

        for client in clients:
            # Filter by tenant if specified
            if tenant_id and client.tenant_id != tenant_id:
                continue

            # Check if client has matching subscription
            if self._matches_subscription(client, event):
                try:
                    await client.websocket.send_json(event.model_dump(mode="json"))
                    sent_count += 1
                except Exception as e:
                    logger.error(f"Failed to broadcast to {client.id}: {e}")
                    await self.disconnect(client.id)

        return sent_count

    def _matches_subscription(
        self,
        client: ConnectedClient,
        event: GraphEvent,
    ) -> bool:
        """Check if event matches any client subscription."""
        if not client.subscriptions:
            return False

        for subscription in client.subscriptions.values():
            if self._subscription_matches_event(subscription, event):
                return True

        return False

    def _subscription_matches_event(
        self,
        subscription: Subscription,
        event: GraphEvent,
    ) -> bool:
        """Check if a subscription matches an event."""
        sub_type = subscription.subscription_type
        filters = subscription.filters

        # Match all events
        if sub_type == SubscriptionType.ALL:
            return True

        # Match entity events
        if sub_type == SubscriptionType.ENTITIES:
            return event.event_type.value.startswith("entity.")

        # Match relation events
        if sub_type == SubscriptionType.RELATIONS:
            return event.event_type.value.startswith("relation.")

        # Match specific entity
        if sub_type == SubscriptionType.ENTITY:
            entity_id = filters.get("entity_id")
            if entity_id:
                event_entity_id = event.data.get("entity_id") or event.data.get("id")
                return str(event_entity_id) == str(entity_id)

        # Match entity type
        if sub_type == SubscriptionType.ENTITY_TYPE:
            entity_type = filters.get("entity_type")
            if entity_type:
                event_type = event.data.get("entity_type") or event.data.get("type")
                return event_type == entity_type

        # Match relation type
        if sub_type == SubscriptionType.RELATION_TYPE:
            relation_type = filters.get("relation_type")
            if relation_type:
                event_rel_type = event.data.get("relation_type") or event.data.get("type")
                return event_rel_type == relation_type

        return False

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats to all clients."""
        while True:
            await asyncio.sleep(self._heartbeat_interval)

            async with self._lock:
                if not self._clients:
                    break

                clients = list(self._clients.items())

            for client_id, client in clients:
                try:
                    heartbeat = GraphEvent(
                        event_type=EventType.HEARTBEAT,
                        data={"timestamp": datetime.utcnow().isoformat()},
                    )
                    await client.websocket.send_json(heartbeat.model_dump(mode="json"))
                    client.last_heartbeat = datetime.utcnow()
                except Exception:
                    await self.disconnect(client_id)

    def get_client_count(self) -> int:
        """Get number of connected clients."""
        return len(self._clients)

    def get_subscription_count(self) -> int:
        """Get total number of active subscriptions."""
        return sum(len(c.subscriptions) for c in self._clients.values())

    async def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        async with self._lock:
            clients = list(self._clients.values())

        return {
            "connected_clients": len(clients),
            "total_subscriptions": sum(len(c.subscriptions) for c in clients),
            "clients": [
                {
                    "id": c.id,
                    "tenant_id": c.tenant_id,
                    "subscriptions": len(c.subscriptions),
                    "connected_at": c.connected_at.isoformat(),
                }
                for c in clients
            ],
        }


# Global connection manager instance
manager = ConnectionManager()


# =============================================================================
# Event Publishing Functions
# =============================================================================

async def publish_entity_created(
    entity_id: str,
    entity_type: str,
    entity_name: str,
    properties: Dict[str, Any] = None,
    tenant_id: Optional[str] = None,
) -> int:
    """Publish entity created event."""
    event = GraphEvent(
        event_type=EventType.ENTITY_CREATED,
        data={
            "id": entity_id,
            "entity_type": entity_type,
            "name": entity_name,
            "properties": properties or {},
        },
        metadata={"tenant_id": tenant_id} if tenant_id else {},
    )
    return await manager.broadcast(event, tenant_id)


async def publish_entity_updated(
    entity_id: str,
    changes: Dict[str, Any],
    tenant_id: Optional[str] = None,
) -> int:
    """Publish entity updated event."""
    event = GraphEvent(
        event_type=EventType.ENTITY_UPDATED,
        data={
            "entity_id": entity_id,
            "changes": changes,
        },
        metadata={"tenant_id": tenant_id} if tenant_id else {},
    )
    return await manager.broadcast(event, tenant_id)


async def publish_entity_deleted(
    entity_id: str,
    tenant_id: Optional[str] = None,
) -> int:
    """Publish entity deleted event."""
    event = GraphEvent(
        event_type=EventType.ENTITY_DELETED,
        data={"entity_id": entity_id},
        metadata={"tenant_id": tenant_id} if tenant_id else {},
    )
    return await manager.broadcast(event, tenant_id)


async def publish_relation_created(
    relation_id: str,
    source_id: str,
    target_id: str,
    relation_type: str,
    properties: Dict[str, Any] = None,
    tenant_id: Optional[str] = None,
) -> int:
    """Publish relation created event."""
    event = GraphEvent(
        event_type=EventType.RELATION_CREATED,
        data={
            "id": relation_id,
            "source_id": source_id,
            "target_id": target_id,
            "relation_type": relation_type,
            "properties": properties or {},
        },
        metadata={"tenant_id": tenant_id} if tenant_id else {},
    )
    return await manager.broadcast(event, tenant_id)


async def publish_relation_updated(
    relation_id: str,
    changes: Dict[str, Any],
    tenant_id: Optional[str] = None,
) -> int:
    """Publish relation updated event."""
    event = GraphEvent(
        event_type=EventType.RELATION_UPDATED,
        data={
            "relation_id": relation_id,
            "changes": changes,
        },
        metadata={"tenant_id": tenant_id} if tenant_id else {},
    )
    return await manager.broadcast(event, tenant_id)


async def publish_relation_deleted(
    relation_id: str,
    tenant_id: Optional[str] = None,
) -> int:
    """Publish relation deleted event."""
    event = GraphEvent(
        event_type=EventType.RELATION_DELETED,
        data={"relation_id": relation_id},
        metadata={"tenant_id": tenant_id} if tenant_id else {},
    )
    return await manager.broadcast(event, tenant_id)


async def publish_extraction_complete(
    extraction_id: str,
    entities_count: int,
    relations_count: int,
    tenant_id: Optional[str] = None,
) -> int:
    """Publish extraction complete event."""
    event = GraphEvent(
        event_type=EventType.EXTRACTION_COMPLETE,
        data={
            "extraction_id": extraction_id,
            "entities_count": entities_count,
            "relations_count": relations_count,
        },
        metadata={"tenant_id": tenant_id} if tenant_id else {},
    )
    return await manager.broadcast(event, tenant_id)


# =============================================================================
# WebSocket Endpoint
# =============================================================================

@router.websocket("/stream")
async def websocket_endpoint(
    websocket: WebSocket,
    client_id: Optional[str] = Query(None),
    tenant_id: Optional[str] = Query(None),
    token: Optional[str] = Query(None),
):
    """
    WebSocket endpoint for real-time graph updates.

    Connect to receive live updates about entities and relations.

    Query Parameters:
        - client_id: Optional client identifier (generated if not provided)
        - tenant_id: Optional tenant identifier for multi-tenancy
        - token: Optional authentication token

    Messages from client:
        - {"action": "subscribe", "payload": {"subscription_type": "all"}}
        - {"action": "subscribe", "payload": {"subscription_type": "entity", "filters": {"entity_id": "..."}}}
        - {"action": "unsubscribe", "payload": {"subscription_id": "..."}}
        - {"action": "ping"}

    Messages to client:
        - {"event_type": "entity.created", "data": {...}}
        - {"event_type": "entity.updated", "data": {...}}
        - {"event_type": "relation.created", "data": {...}}
        - {"event_type": "heartbeat", "data": {...}}
    """
    # TODO: Add token validation here
    # if token:
    #     validate_token(token)

    client_id = await manager.connect(websocket, client_id, tenant_id)

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()

            try:
                message = json.loads(data)
                action = message.get("action", "").lower()
                payload = message.get("payload", {})
                request_id = message.get("request_id")

                if action == "subscribe":
                    subscription_type = SubscriptionType(
                        payload.get("subscription_type", "all")
                    )
                    filters = payload.get("filters", {})

                    subscription_id = await manager.subscribe(
                        client_id,
                        subscription_type,
                        filters,
                    )

                    await manager.send_to_client(client_id, GraphEvent(
                        event_type=EventType.SUBSCRIBED,
                        data={
                            "subscription_id": subscription_id,
                            "subscription_type": subscription_type.value,
                            "filters": filters,
                            "request_id": request_id,
                        },
                    ))

                elif action == "unsubscribe":
                    subscription_id = payload.get("subscription_id")
                    if subscription_id:
                        success = await manager.unsubscribe(client_id, subscription_id)

                        await manager.send_to_client(client_id, GraphEvent(
                            event_type=EventType.UNSUBSCRIBED,
                            data={
                                "subscription_id": subscription_id,
                                "success": success,
                                "request_id": request_id,
                            },
                        ))

                elif action == "ping":
                    await manager.send_to_client(client_id, GraphEvent(
                        event_type=EventType.HEARTBEAT,
                        data={
                            "pong": True,
                            "timestamp": datetime.utcnow().isoformat(),
                            "request_id": request_id,
                        },
                    ))

                else:
                    await manager.send_to_client(client_id, GraphEvent(
                        event_type=EventType.ERROR,
                        data={
                            "message": f"Unknown action: {action}",
                            "request_id": request_id,
                        },
                    ))

            except json.JSONDecodeError:
                await manager.send_to_client(client_id, GraphEvent(
                    event_type=EventType.ERROR,
                    data={"message": "Invalid JSON message"},
                ))
            except ValueError as e:
                await manager.send_to_client(client_id, GraphEvent(
                    event_type=EventType.ERROR,
                    data={"message": str(e)},
                ))

    except WebSocketDisconnect:
        await manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
        await manager.disconnect(client_id)


# =============================================================================
# REST Endpoints for WebSocket Management
# =============================================================================

@router.get("/stats")
async def get_websocket_stats():
    """Get WebSocket connection statistics."""
    return await manager.get_stats()


@router.post("/broadcast")
async def broadcast_event(
    event_type: str,
    data: Dict[str, Any],
    tenant_id: Optional[str] = None,
):
    """Broadcast a custom event to subscribers (admin only)."""
    try:
        event = GraphEvent(
            event_type=EventType(event_type),
            data=data,
        )
        sent_count = await manager.broadcast(event, tenant_id)
        return {
            "success": True,
            "sent_to": sent_count,
            "event_id": event.event_id,
        }
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid event type: {event_type}")


# =============================================================================
# Get manager instance for external use
# =============================================================================

def get_connection_manager() -> ConnectionManager:
    """Get the global connection manager instance."""
    return manager
