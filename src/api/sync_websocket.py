"""
WebSocket API Routes for Real-time Sync.

Provides WebSocket endpoints for real-time data synchronization,
including connection management, subscriptions, and stream processing.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from src.sync.gateway.auth import (
    AuthToken,
    PermissionLevel,
    ResourceType,
    get_tenant_id,
    sync_auth_handler,
)
from src.sync.websocket.ws_server import (
    WebSocketConnectionManager,
    ConnectionState,
    MessageType,
    SubscriptionType,
    WebSocketMessage,
)
from src.sync.websocket.stream_processor import (
    StreamProcessor,
    StreamProcessorManager,
    StreamMessage,
    StreamFilter,
    FilterRule,
    BackpressureController,
    BackpressureStrategy,
    RetryPolicy,
    create_sync_data_processor,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/sync/ws", tags=["sync-websocket"])

# Global instances
_connection_manager: Optional[WebSocketConnectionManager] = None
_processor_manager: Optional[StreamProcessorManager] = None


def get_connection_manager() -> WebSocketConnectionManager:
    """Get or create the global connection manager."""
    global _connection_manager
    if _connection_manager is None:
        _connection_manager = WebSocketConnectionManager()
    return _connection_manager


def get_processor_manager() -> StreamProcessorManager:
    """Get or create the global processor manager."""
    global _processor_manager
    if _processor_manager is None:
        _processor_manager = StreamProcessorManager()
    return _processor_manager


# ============================================================================
# Request/Response Models
# ============================================================================


class ConnectionInfo(BaseModel):
    """Connection information response."""

    connection_id: str
    state: str
    tenant_id: Optional[str]
    user_id: Optional[str]
    connected_at: datetime
    last_activity: datetime
    subscriptions: List[str]


class ConnectionListResponse(BaseModel):
    """Response for listing connections."""

    items: List[ConnectionInfo]
    total: int


class SubscriptionRequest(BaseModel):
    """Request to create a subscription."""

    subscription_type: str = Field(
        ..., description="Type of subscription: sync_events, data_changes, conflicts, job_status, metrics"
    )
    filters: Dict[str, Any] = Field(
        default_factory=dict, description="Optional filters for the subscription"
    )


class SubscriptionResponse(BaseModel):
    """Response for subscription operations."""

    success: bool
    subscription_type: str
    message: str


class ProcessorConfig(BaseModel):
    """Configuration for stream processor."""

    processor_id: str
    source_types: List[str] = Field(default_factory=list)
    max_buffer_size: int = Field(default=1000, ge=100, le=10000)
    batch_size: int = Field(default=10, ge=1, le=100)
    batch_timeout_seconds: float = Field(default=1.0, ge=0.1, le=10.0)
    max_retries: int = Field(default=3, ge=0, le=10)
    backpressure_strategy: str = Field(default="drop_oldest")


class ProcessorResponse(BaseModel):
    """Response for processor operations."""

    processor_id: str
    status: str
    message: str


class ProcessorMetrics(BaseModel):
    """Metrics for a stream processor."""

    processor_id: str
    state: str
    messages_received: int
    messages_processed: int
    messages_dropped: int
    messages_filtered: int
    processing_errors: int
    backpressure_events: int
    average_processing_time_ms: float
    throughput_per_second: float
    buffer_utilization: float
    is_backpressure_active: bool
    dead_letter_queue_size: int


class BroadcastRequest(BaseModel):
    """Request to broadcast a message."""

    subscription_type: str
    message_type: str
    data: Dict[str, Any]
    tenant_id: Optional[str] = None


class BroadcastResponse(BaseModel):
    """Response for broadcast operations."""

    success: bool
    recipients_count: int
    message: str


# ============================================================================
# WebSocket Endpoint
# ============================================================================


@router.websocket("/connect")
async def websocket_endpoint(websocket: WebSocket):
    """
    Main WebSocket endpoint for real-time sync.

    Connection flow:
    1. Client connects to this endpoint
    2. Client sends authentication message with JWT token
    3. Server validates token and sets up connection
    4. Client can subscribe to different event types
    5. Server pushes real-time updates to subscribed clients
    """
    manager = get_connection_manager()
    connection_id = None

    try:
        # Accept connection
        connection_info = await manager.connect(websocket)
        connection_id = connection_info.connection_id

        logger.info(f"WebSocket connected: {connection_id}")

        # Send connection acknowledgment
        await manager.send_message(
            connection_id,
            WebSocketMessage(
                type=MessageType.CONNECTED,
                data={
                    "connection_id": connection_id,
                    "server_time": datetime.utcnow().isoformat(),
                },
            ),
        )

        # Main message loop
        while True:
            try:
                # Receive message with timeout for health checks
                message_data = await asyncio.wait_for(
                    websocket.receive_json(), timeout=60.0
                )

                message = WebSocketMessage(**message_data)
                await _handle_message(manager, connection_id, message)

            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                await manager.send_message(
                    connection_id,
                    WebSocketMessage(type=MessageType.PING, data={}),
                )

    except WebSocketDisconnect as e:
        logger.info(f"WebSocket disconnected: {connection_id}, code: {e.code}")
    except Exception as e:
        logger.error(f"WebSocket error for {connection_id}: {e}")
    finally:
        if connection_id:
            await manager.disconnect(connection_id, "Connection closed")


async def _handle_message(
    manager: WebSocketConnectionManager,
    connection_id: str,
    message: WebSocketMessage,
) -> None:
    """Handle incoming WebSocket messages."""

    if message.type == MessageType.AUTH:
        # Handle authentication
        auth_data = message.data
        token = auth_data.get("token", "")
        tenant_id = auth_data.get("tenant_id", "")
        user_id = auth_data.get("user_id")

        success = await manager.authenticate(connection_id, token, tenant_id, user_id)

        await manager.send_message(
            connection_id,
            WebSocketMessage(
                type=MessageType.AUTH_RESPONSE,
                data={
                    "success": success,
                    "message": "Authentication successful" if success else "Authentication failed",
                },
            ),
        )

    elif message.type == MessageType.SUBSCRIBE:
        # Handle subscription
        sub_type_str = message.data.get("subscription_type", "")
        filters = message.data.get("filters", {})

        try:
            sub_type = SubscriptionType(sub_type_str)
            success = await manager.subscribe(connection_id, sub_type, filters)

            await manager.send_message(
                connection_id,
                WebSocketMessage(
                    type=MessageType.SUBSCRIPTION_RESPONSE,
                    data={
                        "success": success,
                        "subscription_type": sub_type_str,
                        "message": "Subscribed successfully" if success else "Subscription failed",
                    },
                ),
            )
        except ValueError:
            await manager.send_message(
                connection_id,
                WebSocketMessage(
                    type=MessageType.ERROR,
                    data={
                        "error": "invalid_subscription_type",
                        "message": f"Unknown subscription type: {sub_type_str}",
                    },
                ),
            )

    elif message.type == MessageType.UNSUBSCRIBE:
        # Handle unsubscription
        sub_type_str = message.data.get("subscription_type", "")

        try:
            sub_type = SubscriptionType(sub_type_str)
            success = await manager.unsubscribe(connection_id, sub_type)

            await manager.send_message(
                connection_id,
                WebSocketMessage(
                    type=MessageType.SUBSCRIPTION_RESPONSE,
                    data={
                        "success": success,
                        "subscription_type": sub_type_str,
                        "message": "Unsubscribed successfully" if success else "Unsubscription failed",
                    },
                ),
            )
        except ValueError:
            await manager.send_message(
                connection_id,
                WebSocketMessage(
                    type=MessageType.ERROR,
                    data={
                        "error": "invalid_subscription_type",
                        "message": f"Unknown subscription type: {sub_type_str}",
                    },
                ),
            )

    elif message.type == MessageType.PONG:
        # Handle pong response (keep-alive acknowledgment)
        pass

    else:
        logger.warning(f"Unknown message type from {connection_id}: {message.type}")


# ============================================================================
# REST API Endpoints for WebSocket Management
# ============================================================================


@router.get("/connections", response_model=ConnectionListResponse)
async def list_connections(
    tenant_id: str = Depends(get_tenant_id),
    auth: AuthToken = Depends(
        sync_auth_handler.require_permission(ResourceType.SYNC_JOB, PermissionLevel.READ)
    ),
):
    """
    List all active WebSocket connections for the tenant.

    This endpoint is useful for monitoring and debugging WebSocket connections.
    """
    manager = get_connection_manager()
    connections = []

    for conn_id, info in manager.connections.items():
        if info.tenant_id == tenant_id:
            connections.append(
                ConnectionInfo(
                    connection_id=conn_id,
                    state=info.state.value,
                    tenant_id=info.tenant_id,
                    user_id=info.user_id,
                    connected_at=info.connected_at,
                    last_activity=info.last_activity,
                    subscriptions=[s.value for s in info.subscriptions],
                )
            )

    return ConnectionListResponse(items=connections, total=len(connections))


@router.delete("/connections/{connection_id}")
async def disconnect_connection(
    connection_id: str,
    tenant_id: str = Depends(get_tenant_id),
    auth: AuthToken = Depends(
        sync_auth_handler.require_permission(ResourceType.SYNC_JOB, PermissionLevel.ADMIN)
    ),
):
    """
    Force disconnect a WebSocket connection.

    This endpoint is useful for administrative purposes.
    """
    manager = get_connection_manager()
    info = manager.connections.get(connection_id)

    if not info:
        raise HTTPException(status_code=404, detail="Connection not found")

    if info.tenant_id != tenant_id:
        raise HTTPException(status_code=403, detail="Connection belongs to different tenant")

    await manager.disconnect(connection_id, "Admin disconnect")

    return {"success": True, "message": f"Connection {connection_id} disconnected"}


@router.post("/broadcast", response_model=BroadcastResponse)
async def broadcast_message(
    request: BroadcastRequest,
    tenant_id: str = Depends(get_tenant_id),
    auth: AuthToken = Depends(
        sync_auth_handler.require_permission(ResourceType.SYNC_JOB, PermissionLevel.WRITE)
    ),
):
    """
    Broadcast a message to all subscribed connections.

    The message is sent to all connections that are subscribed to the specified
    subscription type and match the optional filters.
    """
    manager = get_connection_manager()

    try:
        sub_type = SubscriptionType(request.subscription_type)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid subscription type: {request.subscription_type}",
        )

    message = WebSocketMessage(
        type=MessageType(request.message_type),
        data=request.data,
    )

    recipients = await manager.broadcast_to_subscription(
        sub_type, request.tenant_id or tenant_id, message
    )

    return BroadcastResponse(
        success=True,
        recipients_count=recipients,
        message=f"Message broadcast to {recipients} connections",
    )


# ============================================================================
# Stream Processor Management Endpoints
# ============================================================================


@router.post("/processors", response_model=ProcessorResponse)
async def create_processor(
    config: ProcessorConfig,
    tenant_id: str = Depends(get_tenant_id),
    auth: AuthToken = Depends(
        sync_auth_handler.require_permission(ResourceType.SYNC_JOB, PermissionLevel.WRITE)
    ),
):
    """
    Create a new stream processor.

    Stream processors handle real-time data transformation and filtering
    before broadcasting to WebSocket clients.
    """
    processor_manager = get_processor_manager()
    manager = get_connection_manager()

    async def broadcast_handler(message: StreamMessage) -> None:
        """Handler that broadcasts processed messages to WebSocket clients."""
        try:
            sub_type = SubscriptionType(message.type)
        except ValueError:
            sub_type = SubscriptionType.SYNC_EVENTS

        ws_message = WebSocketMessage(
            type=MessageType.DATA,
            data={
                "id": message.id,
                "source": message.source,
                "type": message.type,
                "payload": message.data,
                "timestamp": message.timestamp.isoformat(),
            },
        )

        await manager.broadcast_to_subscription(sub_type, tenant_id, ws_message)

    try:
        strategy = BackpressureStrategy(config.backpressure_strategy)
    except ValueError:
        strategy = BackpressureStrategy.DROP_OLDEST

    filter_rules = []
    if config.source_types:
        filter_rules.append(
            FilterRule(field="type", operator="in", value=config.source_types)
        )

    processor = StreamProcessor(
        processor_id=config.processor_id,
        handler=broadcast_handler,
        filter=StreamFilter(rules=filter_rules),
        backpressure_controller=BackpressureController(
            max_buffer_size=config.max_buffer_size,
            strategy=strategy,
        ),
        retry_policy=RetryPolicy(max_retries=config.max_retries),
        batch_size=config.batch_size,
        batch_timeout_seconds=config.batch_timeout_seconds,
    )

    try:
        await processor_manager.register_processor(processor)
        await processor.start()

        return ProcessorResponse(
            processor_id=config.processor_id,
            status="running",
            message="Processor created and started successfully",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/processors", response_model=List[ProcessorMetrics])
async def list_processors(
    tenant_id: str = Depends(get_tenant_id),
    auth: AuthToken = Depends(
        sync_auth_handler.require_permission(ResourceType.SYNC_JOB, PermissionLevel.READ)
    ),
):
    """
    List all stream processors and their metrics.
    """
    processor_manager = get_processor_manager()
    metrics = processor_manager.get_all_metrics()

    return [
        ProcessorMetrics(processor_id=proc_id, **proc_metrics)
        for proc_id, proc_metrics in metrics.items()
    ]


@router.get("/processors/{processor_id}", response_model=ProcessorMetrics)
async def get_processor(
    processor_id: str,
    tenant_id: str = Depends(get_tenant_id),
    auth: AuthToken = Depends(
        sync_auth_handler.require_permission(ResourceType.SYNC_JOB, PermissionLevel.READ)
    ),
):
    """
    Get metrics for a specific stream processor.
    """
    processor_manager = get_processor_manager()
    processor = processor_manager.get_processor(processor_id)

    if not processor:
        raise HTTPException(status_code=404, detail="Processor not found")

    metrics = processor.get_metrics()
    return ProcessorMetrics(processor_id=processor_id, **metrics)


@router.post("/processors/{processor_id}/pause", response_model=ProcessorResponse)
async def pause_processor(
    processor_id: str,
    tenant_id: str = Depends(get_tenant_id),
    auth: AuthToken = Depends(
        sync_auth_handler.require_permission(ResourceType.SYNC_JOB, PermissionLevel.WRITE)
    ),
):
    """
    Pause a stream processor.
    """
    processor_manager = get_processor_manager()
    processor = processor_manager.get_processor(processor_id)

    if not processor:
        raise HTTPException(status_code=404, detail="Processor not found")

    await processor.pause()

    return ProcessorResponse(
        processor_id=processor_id,
        status="paused",
        message="Processor paused successfully",
    )


@router.post("/processors/{processor_id}/resume", response_model=ProcessorResponse)
async def resume_processor(
    processor_id: str,
    tenant_id: str = Depends(get_tenant_id),
    auth: AuthToken = Depends(
        sync_auth_handler.require_permission(ResourceType.SYNC_JOB, PermissionLevel.WRITE)
    ),
):
    """
    Resume a paused stream processor.
    """
    processor_manager = get_processor_manager()
    processor = processor_manager.get_processor(processor_id)

    if not processor:
        raise HTTPException(status_code=404, detail="Processor not found")

    await processor.resume()

    return ProcessorResponse(
        processor_id=processor_id,
        status="running",
        message="Processor resumed successfully",
    )


@router.delete("/processors/{processor_id}", response_model=ProcessorResponse)
async def delete_processor(
    processor_id: str,
    tenant_id: str = Depends(get_tenant_id),
    auth: AuthToken = Depends(
        sync_auth_handler.require_permission(ResourceType.SYNC_JOB, PermissionLevel.ADMIN)
    ),
):
    """
    Stop and delete a stream processor.
    """
    processor_manager = get_processor_manager()
    processor = processor_manager.get_processor(processor_id)

    if not processor:
        raise HTTPException(status_code=404, detail="Processor not found")

    await processor_manager.unregister_processor(processor_id)

    return ProcessorResponse(
        processor_id=processor_id,
        status="deleted",
        message="Processor stopped and deleted successfully",
    )


@router.get("/processors/{processor_id}/dead-letters")
async def get_dead_letters(
    processor_id: str,
    limit: int = Query(100, ge=1, le=1000),
    tenant_id: str = Depends(get_tenant_id),
    auth: AuthToken = Depends(
        sync_auth_handler.require_permission(ResourceType.SYNC_JOB, PermissionLevel.READ)
    ),
):
    """
    Get messages from the dead letter queue.

    These are messages that failed processing after all retry attempts.
    """
    processor_manager = get_processor_manager()
    processor = processor_manager.get_processor(processor_id)

    if not processor:
        raise HTTPException(status_code=404, detail="Processor not found")

    messages = processor.get_dead_letter_messages(limit)

    return {"items": messages, "total": len(messages)}


# ============================================================================
# Health Check Endpoint
# ============================================================================


@router.get("/health")
async def websocket_health():
    """
    Health check endpoint for WebSocket service.

    Returns the status of the WebSocket connection manager and stream processors.
    """
    manager = get_connection_manager()
    processor_manager = get_processor_manager()

    connection_count = len(manager.connections)
    authenticated_count = sum(
        1 for info in manager.connections.values()
        if info.state == ConnectionState.AUTHENTICATED
    )

    processor_count = len(processor_manager.processors)
    running_processors = sum(
        1 for p in processor_manager.processors.values()
        if p.state.value == "running"
    )

    return {
        "status": "healthy",
        "websocket": {
            "total_connections": connection_count,
            "authenticated_connections": authenticated_count,
        },
        "processors": {
            "total": processor_count,
            "running": running_processors,
        },
        "timestamp": datetime.utcnow().isoformat(),
    }
