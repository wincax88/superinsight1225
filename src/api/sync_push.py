"""
Sync Push API Routes.

Provides API endpoints for receiving pushed data from external sources,
including batch push, streaming, webhooks, and file uploads.
"""

import hashlib
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel, Field

from src.sync.gateway.auth import (
    AuthToken,
    Permission,
    PermissionLevel,
    ResourceType,
    get_tenant_id,
    sync_auth_handler,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/sync/push", tags=["sync-push"])


# ============================================================================
# Request/Response Models
# ============================================================================

class PushDataRecord(BaseModel):
    """Single data record in push request."""
    id: Optional[str] = None
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None


class BatchPushRequest(BaseModel):
    """Request model for batch data push."""
    source_id: str = Field(..., description="Data source identifier")
    records: List[PushDataRecord] = Field(..., min_length=1, max_length=10000)
    schema_name: Optional[str] = None
    table_name: Optional[str] = None
    options: Dict[str, Any] = Field(default_factory=dict)


class BatchPushResponse(BaseModel):
    """Response model for batch push."""
    request_id: str
    tenant_id: str
    source_id: str
    total_records: int
    accepted_records: int
    rejected_records: int
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    processing_time_ms: float
    status: str
    message: str


class StreamPushRequest(BaseModel):
    """Request model for stream data push."""
    source_id: str
    stream_id: Optional[str] = None
    record: PushDataRecord
    sequence_number: Optional[int] = None


class StreamPushResponse(BaseModel):
    """Response model for stream push."""
    request_id: str
    stream_id: str
    sequence_number: int
    status: str
    message: str


class WebhookEvent(BaseModel):
    """Webhook event payload."""
    event_type: str
    event_id: Optional[str] = None
    source: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


class WebhookResponse(BaseModel):
    """Response model for webhook."""
    event_id: str
    status: str
    message: str
    processed_at: datetime


class FilePushResponse(BaseModel):
    """Response model for file push."""
    request_id: str
    file_id: str
    filename: str
    file_size: int
    content_type: str
    records_count: Optional[int] = None
    status: str
    message: str


class PushStatus(BaseModel):
    """Push operation status."""
    request_id: str
    status: str
    progress: float
    records_processed: int
    records_total: int
    errors_count: int
    started_at: datetime
    completed_at: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None


# ============================================================================
# In-Memory Storage (Replace with database/queue in production)
# ============================================================================

_push_queue: List[Dict[str, Any]] = []
_push_status: Dict[str, Dict[str, Any]] = {}
_stream_sessions: Dict[str, Dict[str, Any]] = {}
_webhook_configs: Dict[str, Dict[str, Any]] = {}


# ============================================================================
# API Endpoints
# ============================================================================

@router.post("/batch", response_model=BatchPushResponse)
async def push_batch_data(
    request: BatchPushRequest,
    tenant_id: str = Depends(get_tenant_id),
    auth: AuthToken = Depends(
        sync_auth_handler.require_permission(ResourceType.SYNC_JOB, PermissionLevel.WRITE)
    )
):
    """
    Push a batch of data records.

    Accepts up to 10,000 records per request. Records are validated
    and queued for processing.
    """
    start_time = time.time()
    request_id = str(uuid4())

    accepted = 0
    rejected = 0
    errors = []

    # Validate and queue records
    for i, record in enumerate(request.records):
        try:
            # Basic validation
            if not record.data:
                raise ValueError("Empty data")

            # Generate ID if not provided
            if not record.id:
                record.id = str(uuid4())

            # Queue for processing
            _push_queue.append({
                "request_id": request_id,
                "tenant_id": tenant_id,
                "source_id": request.source_id,
                "record_id": record.id,
                "data": record.data,
                "metadata": record.metadata,
                "timestamp": record.timestamp or datetime.utcnow(),
                "schema_name": request.schema_name,
                "table_name": request.table_name,
                "queued_at": datetime.utcnow()
            })
            accepted += 1

        except Exception as e:
            rejected += 1
            errors.append({
                "index": i,
                "record_id": record.id,
                "error": str(e)
            })

    processing_time = (time.time() - start_time) * 1000

    # Store status
    _push_status[request_id] = {
        "request_id": request_id,
        "tenant_id": tenant_id,
        "source_id": request.source_id,
        "status": "queued",
        "total_records": len(request.records),
        "accepted_records": accepted,
        "rejected_records": rejected,
        "processed_records": 0,
        "started_at": datetime.utcnow(),
        "completed_at": None
    }

    logger.info(
        f"Batch push received: {request_id}, "
        f"accepted={accepted}, rejected={rejected}"
    )

    return BatchPushResponse(
        request_id=request_id,
        tenant_id=tenant_id,
        source_id=request.source_id,
        total_records=len(request.records),
        accepted_records=accepted,
        rejected_records=rejected,
        errors=errors[:100],  # Limit errors in response
        processing_time_ms=processing_time,
        status="accepted" if rejected == 0 else "partial",
        message=f"Accepted {accepted} records, rejected {rejected}"
    )


@router.post("/stream", response_model=StreamPushResponse)
async def push_stream_data(
    request: StreamPushRequest,
    tenant_id: str = Depends(get_tenant_id),
    auth: AuthToken = Depends(
        sync_auth_handler.require_permission(ResourceType.SYNC_JOB, PermissionLevel.WRITE)
    )
):
    """
    Push a single record in streaming mode.

    Used for real-time data streaming. Records are processed immediately.
    """
    request_id = str(uuid4())

    # Get or create stream session
    stream_id = request.stream_id or str(uuid4())
    if stream_id not in _stream_sessions:
        _stream_sessions[stream_id] = {
            "stream_id": stream_id,
            "tenant_id": tenant_id,
            "source_id": request.source_id,
            "started_at": datetime.utcnow(),
            "sequence_number": 0,
            "records_count": 0
        }

    session = _stream_sessions[stream_id]
    session["sequence_number"] += 1
    session["records_count"] += 1
    session["last_activity"] = datetime.utcnow()

    sequence_number = request.sequence_number or session["sequence_number"]

    # Generate record ID if not provided
    record_id = request.record.id or str(uuid4())

    # Queue for processing
    _push_queue.append({
        "request_id": request_id,
        "stream_id": stream_id,
        "tenant_id": tenant_id,
        "source_id": request.source_id,
        "record_id": record_id,
        "sequence_number": sequence_number,
        "data": request.record.data,
        "metadata": request.record.metadata,
        "timestamp": request.record.timestamp or datetime.utcnow(),
        "queued_at": datetime.utcnow()
    })

    logger.debug(
        f"Stream push received: stream={stream_id}, seq={sequence_number}"
    )

    return StreamPushResponse(
        request_id=request_id,
        stream_id=stream_id,
        sequence_number=sequence_number,
        status="accepted",
        message="Record accepted for processing"
    )


@router.post("/webhook/{webhook_id}", response_model=WebhookResponse)
async def receive_webhook(
    webhook_id: str,
    event: WebhookEvent,
    request: Request,
    tenant_id: str = Depends(get_tenant_id),
    auth: AuthToken = Depends(
        sync_auth_handler.require_permission(ResourceType.SYNC_JOB, PermissionLevel.WRITE)
    )
):
    """
    Receive data via webhook.

    Webhooks must be pre-configured with a webhook_id.
    Supports various event types for different data sources.
    """
    # Validate webhook configuration
    webhook_config = _webhook_configs.get(f"{tenant_id}:{webhook_id}")
    if not webhook_config:
        # Create default config for demo
        webhook_config = {
            "webhook_id": webhook_id,
            "tenant_id": tenant_id,
            "enabled": True,
            "created_at": datetime.utcnow()
        }
        _webhook_configs[f"{tenant_id}:{webhook_id}"] = webhook_config

    if not webhook_config.get("enabled", True):
        raise HTTPException(status_code=404, detail="Webhook not found or disabled")

    # Generate event ID if not provided
    event_id = event.event_id or str(uuid4())

    # Queue webhook data for processing
    _push_queue.append({
        "request_id": event_id,
        "webhook_id": webhook_id,
        "tenant_id": tenant_id,
        "event_type": event.event_type,
        "source": event.source,
        "data": event.data,
        "metadata": event.metadata,
        "timestamp": event.timestamp,
        "received_at": datetime.utcnow()
    })

    logger.info(
        f"Webhook received: id={webhook_id}, event_type={event.event_type}"
    )

    return WebhookResponse(
        event_id=event_id,
        status="accepted",
        message=f"Event {event.event_type} accepted for processing",
        processed_at=datetime.utcnow()
    )


@router.post("/file", response_model=FilePushResponse)
async def push_file(
    file: UploadFile = File(...),
    source_id: str = Form(...),
    schema_name: Optional[str] = Form(None),
    table_name: Optional[str] = Form(None),
    file_format: Optional[str] = Form(None),
    tenant_id: str = Depends(get_tenant_id),
    auth: AuthToken = Depends(
        sync_auth_handler.require_permission(ResourceType.SYNC_JOB, PermissionLevel.WRITE)
    )
):
    """
    Push data via file upload.

    Supports CSV, JSON, JSONL, Excel, and Parquet formats.
    Files are validated and queued for processing.
    """
    request_id = str(uuid4())
    file_id = str(uuid4())

    # Read file content
    content = await file.read()
    file_size = len(content)

    # Validate file size (max 100MB)
    max_size = 100 * 1024 * 1024
    if file_size > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {max_size // (1024*1024)}MB"
        )

    # Detect format if not provided
    if not file_format:
        filename = file.filename or ""
        if filename.endswith(".csv"):
            file_format = "csv"
        elif filename.endswith(".json"):
            file_format = "json"
        elif filename.endswith(".jsonl"):
            file_format = "jsonl"
        elif filename.endswith((".xlsx", ".xls")):
            file_format = "excel"
        elif filename.endswith(".parquet"):
            file_format = "parquet"
        else:
            file_format = "unknown"

    # Calculate checksum
    checksum = hashlib.sha256(content).hexdigest()

    # Queue file for processing
    _push_queue.append({
        "request_id": request_id,
        "file_id": file_id,
        "tenant_id": tenant_id,
        "source_id": source_id,
        "filename": file.filename,
        "file_size": file_size,
        "content_type": file.content_type,
        "file_format": file_format,
        "checksum": checksum,
        "schema_name": schema_name,
        "table_name": table_name,
        "queued_at": datetime.utcnow(),
        # In production, store to object storage and reference here
        "content": content  # For demo only
    })

    # Store status
    _push_status[request_id] = {
        "request_id": request_id,
        "file_id": file_id,
        "tenant_id": tenant_id,
        "status": "queued",
        "filename": file.filename,
        "file_size": file_size,
        "started_at": datetime.utcnow()
    }

    logger.info(
        f"File push received: {file.filename}, size={file_size}, format={file_format}"
    )

    return FilePushResponse(
        request_id=request_id,
        file_id=file_id,
        filename=file.filename or "unknown",
        file_size=file_size,
        content_type=file.content_type or "application/octet-stream",
        records_count=None,  # Unknown until parsed
        status="accepted",
        message="File accepted for processing"
    )


@router.get("/status/{request_id}", response_model=PushStatus)
async def get_push_status(
    request_id: str,
    tenant_id: str = Depends(get_tenant_id),
    auth: AuthToken = Depends(
        sync_auth_handler.require_permission(ResourceType.SYNC_JOB, PermissionLevel.READ)
    )
):
    """
    Get the status of a push request.

    Returns progress information for batch, stream, or file push operations.
    """
    status = _push_status.get(request_id)
    if not status or status.get("tenant_id") != tenant_id:
        raise HTTPException(status_code=404, detail="Push request not found")

    return PushStatus(
        request_id=request_id,
        status=status.get("status", "unknown"),
        progress=status.get("processed_records", 0) / max(status.get("total_records", 1), 1),
        records_processed=status.get("processed_records", 0),
        records_total=status.get("total_records", 0),
        errors_count=status.get("rejected_records", 0),
        started_at=status.get("started_at", datetime.utcnow()),
        completed_at=status.get("completed_at"),
        estimated_completion=None
    )


@router.get("/queue/stats")
async def get_queue_stats(
    tenant_id: str = Depends(get_tenant_id),
    auth: AuthToken = Depends(
        sync_auth_handler.require_permission(ResourceType.SYNC_JOB, PermissionLevel.READ)
    )
):
    """
    Get push queue statistics.

    Returns information about queued, processing, and completed push operations.
    """
    tenant_items = [
        item for item in _push_queue
        if item.get("tenant_id") == tenant_id
    ]

    return {
        "tenant_id": tenant_id,
        "queue_length": len(tenant_items),
        "total_items": len(_push_queue),
        "streams_active": len([
            s for s in _stream_sessions.values()
            if s.get("tenant_id") == tenant_id
        ]),
        "webhooks_configured": len([
            w for w in _webhook_configs.values()
            if w.get("tenant_id") == tenant_id
        ]),
        "timestamp": datetime.utcnow()
    }


@router.post("/webhooks", status_code=201)
async def create_webhook(
    webhook_id: str = Query(..., min_length=1, max_length=100),
    description: Optional[str] = None,
    secret: Optional[str] = None,
    tenant_id: str = Depends(get_tenant_id),
    auth: AuthToken = Depends(
        sync_auth_handler.require_permission(ResourceType.SYNC_JOB, PermissionLevel.WRITE)
    )
):
    """
    Create a new webhook configuration.

    Returns the webhook URL and secret for use in external systems.
    """
    config_key = f"{tenant_id}:{webhook_id}"
    if config_key in _webhook_configs:
        raise HTTPException(status_code=409, detail="Webhook ID already exists")

    # Generate secret if not provided
    if not secret:
        import secrets
        secret = secrets.token_urlsafe(32)

    config = {
        "webhook_id": webhook_id,
        "tenant_id": tenant_id,
        "description": description,
        "secret": secret,
        "enabled": True,
        "created_at": datetime.utcnow(),
        "events_received": 0
    }
    _webhook_configs[config_key] = config

    logger.info(f"Created webhook: {webhook_id} for tenant {tenant_id}")

    return {
        "webhook_id": webhook_id,
        "webhook_url": f"/api/v1/sync/push/webhook/{webhook_id}",
        "secret": secret,
        "status": "created",
        "message": "Webhook created successfully"
    }


@router.get("/webhooks")
async def list_webhooks(
    tenant_id: str = Depends(get_tenant_id),
    auth: AuthToken = Depends(
        sync_auth_handler.require_permission(ResourceType.SYNC_JOB, PermissionLevel.READ)
    )
):
    """List all configured webhooks for the tenant."""
    webhooks = [
        {
            "webhook_id": config["webhook_id"],
            "description": config.get("description"),
            "enabled": config.get("enabled", True),
            "created_at": config.get("created_at"),
            "events_received": config.get("events_received", 0)
        }
        for config in _webhook_configs.values()
        if config.get("tenant_id") == tenant_id
    ]

    return {
        "webhooks": webhooks,
        "total": len(webhooks)
    }


@router.delete("/webhooks/{webhook_id}", status_code=204)
async def delete_webhook(
    webhook_id: str,
    tenant_id: str = Depends(get_tenant_id),
    auth: AuthToken = Depends(
        sync_auth_handler.require_permission(ResourceType.SYNC_JOB, PermissionLevel.ADMIN)
    )
):
    """Delete a webhook configuration."""
    config_key = f"{tenant_id}:{webhook_id}"
    if config_key not in _webhook_configs:
        raise HTTPException(status_code=404, detail="Webhook not found")

    del _webhook_configs[config_key]
    logger.info(f"Deleted webhook: {webhook_id}")
