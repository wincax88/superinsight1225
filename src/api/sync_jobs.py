"""
Sync Jobs API Routes.

Provides API endpoints for managing synchronization jobs,
including CRUD operations and job control (start, stop, pause, resume).
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from src.sync.gateway.auth import (
    AuthToken,
    Permission,
    PermissionLevel,
    ResourceType,
    get_current_user,
    get_tenant_id,
    sync_auth_handler,
)
from src.sync.models import (
    ConflictResolutionStrategy,
    SyncDirection,
    SyncFrequency,
    SyncJobStatus,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/sync/jobs", tags=["sync-jobs"])


# ============================================================================
# Request/Response Models
# ============================================================================

class DataSourceReference(BaseModel):
    """Reference to a data source."""
    source_id: UUID
    source_name: Optional[str] = None


class TargetConfig(BaseModel):
    """Target configuration for sync job."""
    target_type: str = Field(..., description="Target type: internal, external, cloud")
    connection_config: Dict[str, Any] = Field(default_factory=dict)
    table_mapping: Dict[str, str] = Field(default_factory=dict)


class SyncJobCreate(BaseModel):
    """Request model for creating a sync job."""
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    source_id: UUID
    target_config: TargetConfig
    direction: SyncDirection = SyncDirection.PULL
    frequency: SyncFrequency = SyncFrequency.MANUAL
    schedule_cron: Optional[str] = None
    schedule_timezone: str = "UTC"
    batch_size: int = Field(default=1000, ge=1, le=100000)
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_delay: int = Field(default=60, ge=1)
    timeout: int = Field(default=3600, ge=60)
    enable_incremental: bool = True
    incremental_field: Optional[str] = None
    conflict_resolution: ConflictResolutionStrategy = ConflictResolutionStrategy.TIMESTAMP_BASED
    transformation_rules: List[Dict[str, Any]] = Field(default_factory=list)
    filter_conditions: Dict[str, Any] = Field(default_factory=dict)
    enable_encryption: bool = True
    enable_compression: bool = False


class SyncJobUpdate(BaseModel):
    """Request model for updating a sync job."""
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = None
    target_config: Optional[TargetConfig] = None
    direction: Optional[SyncDirection] = None
    frequency: Optional[SyncFrequency] = None
    status: Optional[SyncJobStatus] = None
    schedule_cron: Optional[str] = None
    schedule_timezone: Optional[str] = None
    batch_size: Optional[int] = Field(None, ge=1, le=100000)
    max_retries: Optional[int] = Field(None, ge=0, le=10)
    retry_delay: Optional[int] = Field(None, ge=1)
    timeout: Optional[int] = Field(None, ge=60)
    enable_incremental: Optional[bool] = None
    incremental_field: Optional[str] = None
    conflict_resolution: Optional[ConflictResolutionStrategy] = None
    transformation_rules: Optional[List[Dict[str, Any]]] = None
    filter_conditions: Optional[Dict[str, Any]] = None
    enable_encryption: Optional[bool] = None
    enable_compression: Optional[bool] = None


class SyncJobResponse(BaseModel):
    """Response model for sync job."""
    id: UUID
    tenant_id: str
    name: str
    description: Optional[str]
    source_id: UUID
    target_config: Dict[str, Any]
    direction: str
    frequency: str
    status: str
    schedule_cron: Optional[str]
    schedule_timezone: str
    batch_size: int
    max_retries: int
    retry_delay: int
    timeout: int
    enable_incremental: bool
    incremental_field: Optional[str]
    last_sync_value: Optional[str]
    conflict_resolution: str
    transformation_rules: List[Dict[str, Any]]
    filter_conditions: Dict[str, Any]
    enable_encryption: bool
    enable_compression: bool
    total_executions: int
    successful_executions: int
    failed_executions: int
    total_records_synced: int
    created_at: datetime
    updated_at: datetime
    last_executed_at: Optional[datetime]
    next_scheduled_at: Optional[datetime]
    created_by: Optional[str]


class SyncJobListResponse(BaseModel):
    """Response model for sync job list."""
    items: List[SyncJobResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class JobExecutionRequest(BaseModel):
    """Request model for job execution control."""
    force: bool = False
    skip_validation: bool = False


class JobExecutionResponse(BaseModel):
    """Response model for job execution control."""
    job_id: UUID
    execution_id: Optional[UUID] = None
    status: str
    message: str
    started_at: Optional[datetime] = None


# ============================================================================
# In-Memory Storage (Replace with database in production)
# ============================================================================

_sync_jobs: Dict[str, Dict[str, Any]] = {}


def _get_job(job_id: str, tenant_id: str) -> Optional[Dict[str, Any]]:
    """Get job by ID and tenant."""
    job = _sync_jobs.get(job_id)
    if job and job.get("tenant_id") == tenant_id:
        return job
    return None


def _list_jobs(tenant_id: str) -> List[Dict[str, Any]]:
    """List jobs for tenant."""
    return [
        job for job in _sync_jobs.values()
        if job.get("tenant_id") == tenant_id
    ]


# ============================================================================
# API Endpoints
# ============================================================================

@router.get("", response_model=SyncJobListResponse)
async def list_sync_jobs(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: Optional[SyncJobStatus] = None,
    direction: Optional[SyncDirection] = None,
    search: Optional[str] = None,
    tenant_id: str = Depends(get_tenant_id),
    auth: AuthToken = Depends(
        sync_auth_handler.require_permission(ResourceType.SYNC_JOB, PermissionLevel.READ)
    )
):
    """
    List all sync jobs for the current tenant.

    Supports filtering by status, direction, and search term.
    """
    jobs = _list_jobs(tenant_id)

    # Apply filters
    if status:
        jobs = [j for j in jobs if j.get("status") == status.value]
    if direction:
        jobs = [j for j in jobs if j.get("direction") == direction.value]
    if search:
        search_lower = search.lower()
        jobs = [
            j for j in jobs
            if search_lower in j.get("name", "").lower()
            or search_lower in (j.get("description") or "").lower()
        ]

    # Pagination
    total = len(jobs)
    total_pages = (total + page_size - 1) // page_size
    start = (page - 1) * page_size
    end = start + page_size
    paginated_jobs = jobs[start:end]

    return SyncJobListResponse(
        items=[SyncJobResponse(**job) for job in paginated_jobs],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages
    )


@router.post("", response_model=SyncJobResponse, status_code=201)
async def create_sync_job(
    job_data: SyncJobCreate,
    tenant_id: str = Depends(get_tenant_id),
    auth: AuthToken = Depends(
        sync_auth_handler.require_permission(ResourceType.SYNC_JOB, PermissionLevel.WRITE)
    )
):
    """
    Create a new sync job.

    The job is created in DRAFT status and must be activated to start syncing.
    """
    from uuid import uuid4

    job_id = str(uuid4())
    now = datetime.utcnow()

    job = {
        "id": job_id,
        "tenant_id": tenant_id,
        "name": job_data.name,
        "description": job_data.description,
        "source_id": str(job_data.source_id),
        "target_config": job_data.target_config.model_dump(),
        "direction": job_data.direction.value,
        "frequency": job_data.frequency.value,
        "status": SyncJobStatus.DRAFT.value,
        "schedule_cron": job_data.schedule_cron,
        "schedule_timezone": job_data.schedule_timezone,
        "batch_size": job_data.batch_size,
        "max_retries": job_data.max_retries,
        "retry_delay": job_data.retry_delay,
        "timeout": job_data.timeout,
        "enable_incremental": job_data.enable_incremental,
        "incremental_field": job_data.incremental_field,
        "last_sync_value": None,
        "conflict_resolution": job_data.conflict_resolution.value,
        "transformation_rules": job_data.transformation_rules,
        "filter_conditions": job_data.filter_conditions,
        "enable_encryption": job_data.enable_encryption,
        "enable_compression": job_data.enable_compression,
        "total_executions": 0,
        "successful_executions": 0,
        "failed_executions": 0,
        "total_records_synced": 0,
        "created_at": now,
        "updated_at": now,
        "last_executed_at": None,
        "next_scheduled_at": None,
        "created_by": auth.user_id
    }

    _sync_jobs[job_id] = job
    logger.info(f"Created sync job {job_id} for tenant {tenant_id}")

    return SyncJobResponse(**job)


@router.get("/{job_id}", response_model=SyncJobResponse)
async def get_sync_job(
    job_id: UUID,
    tenant_id: str = Depends(get_tenant_id),
    auth: AuthToken = Depends(
        sync_auth_handler.require_permission(ResourceType.SYNC_JOB, PermissionLevel.READ)
    )
):
    """Get a specific sync job by ID."""
    job = _get_job(str(job_id), tenant_id)
    if not job:
        raise HTTPException(status_code=404, detail="Sync job not found")
    return SyncJobResponse(**job)


@router.put("/{job_id}", response_model=SyncJobResponse)
async def update_sync_job(
    job_id: UUID,
    job_data: SyncJobUpdate,
    tenant_id: str = Depends(get_tenant_id),
    auth: AuthToken = Depends(
        sync_auth_handler.require_permission(ResourceType.SYNC_JOB, PermissionLevel.WRITE)
    )
):
    """
    Update a sync job.

    Only non-null fields in the request will be updated.
    """
    job = _get_job(str(job_id), tenant_id)
    if not job:
        raise HTTPException(status_code=404, detail="Sync job not found")

    # Update fields
    update_data = job_data.model_dump(exclude_none=True)
    for key, value in update_data.items():
        if key == "target_config" and value:
            job["target_config"] = value
        elif key in ["direction", "frequency", "status", "conflict_resolution"] and value:
            job[key] = value.value if hasattr(value, "value") else value
        else:
            job[key] = value

    job["updated_at"] = datetime.utcnow()

    logger.info(f"Updated sync job {job_id}")
    return SyncJobResponse(**job)


@router.delete("/{job_id}", status_code=204)
async def delete_sync_job(
    job_id: UUID,
    tenant_id: str = Depends(get_tenant_id),
    auth: AuthToken = Depends(
        sync_auth_handler.require_permission(ResourceType.SYNC_JOB, PermissionLevel.ADMIN)
    )
):
    """
    Delete a sync job.

    Jobs that are currently running cannot be deleted.
    """
    job = _get_job(str(job_id), tenant_id)
    if not job:
        raise HTTPException(status_code=404, detail="Sync job not found")

    if job.get("status") == SyncJobStatus.ACTIVE.value:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete an active job. Stop it first."
        )

    del _sync_jobs[str(job_id)]
    logger.info(f"Deleted sync job {job_id}")


@router.post("/{job_id}/start", response_model=JobExecutionResponse)
async def start_sync_job(
    job_id: UUID,
    request: JobExecutionRequest = JobExecutionRequest(),
    tenant_id: str = Depends(get_tenant_id),
    auth: AuthToken = Depends(
        sync_auth_handler.require_permission(ResourceType.SYNC_JOB, PermissionLevel.WRITE)
    )
):
    """
    Start a sync job execution.

    Creates a new execution and begins the synchronization process.
    """
    from uuid import uuid4

    job = _get_job(str(job_id), tenant_id)
    if not job:
        raise HTTPException(status_code=404, detail="Sync job not found")

    if job.get("status") == SyncJobStatus.DISABLED.value:
        raise HTTPException(status_code=400, detail="Job is disabled")

    # Create execution
    execution_id = uuid4()
    now = datetime.utcnow()

    job["status"] = SyncJobStatus.ACTIVE.value
    job["last_executed_at"] = now
    job["total_executions"] = job.get("total_executions", 0) + 1
    job["updated_at"] = now

    logger.info(f"Started sync job {job_id}, execution {execution_id}")

    return JobExecutionResponse(
        job_id=job_id,
        execution_id=execution_id,
        status="started",
        message="Sync job started successfully",
        started_at=now
    )


@router.post("/{job_id}/stop", response_model=JobExecutionResponse)
async def stop_sync_job(
    job_id: UUID,
    tenant_id: str = Depends(get_tenant_id),
    auth: AuthToken = Depends(
        sync_auth_handler.require_permission(ResourceType.SYNC_JOB, PermissionLevel.WRITE)
    )
):
    """Stop a running sync job."""
    job = _get_job(str(job_id), tenant_id)
    if not job:
        raise HTTPException(status_code=404, detail="Sync job not found")

    if job.get("status") != SyncJobStatus.ACTIVE.value:
        raise HTTPException(status_code=400, detail="Job is not running")

    job["status"] = SyncJobStatus.PAUSED.value
    job["updated_at"] = datetime.utcnow()

    logger.info(f"Stopped sync job {job_id}")

    return JobExecutionResponse(
        job_id=job_id,
        status="stopped",
        message="Sync job stopped successfully"
    )


@router.post("/{job_id}/pause", response_model=JobExecutionResponse)
async def pause_sync_job(
    job_id: UUID,
    tenant_id: str = Depends(get_tenant_id),
    auth: AuthToken = Depends(
        sync_auth_handler.require_permission(ResourceType.SYNC_JOB, PermissionLevel.WRITE)
    )
):
    """Pause a running sync job."""
    job = _get_job(str(job_id), tenant_id)
    if not job:
        raise HTTPException(status_code=404, detail="Sync job not found")

    if job.get("status") != SyncJobStatus.ACTIVE.value:
        raise HTTPException(status_code=400, detail="Job is not running")

    job["status"] = SyncJobStatus.PAUSED.value
    job["updated_at"] = datetime.utcnow()

    logger.info(f"Paused sync job {job_id}")

    return JobExecutionResponse(
        job_id=job_id,
        status="paused",
        message="Sync job paused successfully"
    )


@router.post("/{job_id}/resume", response_model=JobExecutionResponse)
async def resume_sync_job(
    job_id: UUID,
    tenant_id: str = Depends(get_tenant_id),
    auth: AuthToken = Depends(
        sync_auth_handler.require_permission(ResourceType.SYNC_JOB, PermissionLevel.WRITE)
    )
):
    """Resume a paused sync job."""
    job = _get_job(str(job_id), tenant_id)
    if not job:
        raise HTTPException(status_code=404, detail="Sync job not found")

    if job.get("status") != SyncJobStatus.PAUSED.value:
        raise HTTPException(status_code=400, detail="Job is not paused")

    job["status"] = SyncJobStatus.ACTIVE.value
    job["updated_at"] = datetime.utcnow()

    logger.info(f"Resumed sync job {job_id}")

    return JobExecutionResponse(
        job_id=job_id,
        status="resumed",
        message="Sync job resumed successfully"
    )


@router.get("/{job_id}/executions")
async def list_job_executions(
    job_id: UUID,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    tenant_id: str = Depends(get_tenant_id),
    auth: AuthToken = Depends(
        sync_auth_handler.require_permission(ResourceType.SYNC_EXECUTION, PermissionLevel.READ)
    )
):
    """List execution history for a sync job."""
    job = _get_job(str(job_id), tenant_id)
    if not job:
        raise HTTPException(status_code=404, detail="Sync job not found")

    # In production, query from database
    return {
        "items": [],
        "total": 0,
        "page": page,
        "page_size": page_size,
        "total_pages": 0
    }


@router.get("/{job_id}/conflicts")
async def list_job_conflicts(
    job_id: UUID,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: Optional[str] = None,
    tenant_id: str = Depends(get_tenant_id),
    auth: AuthToken = Depends(
        sync_auth_handler.require_permission(ResourceType.DATA_CONFLICT, PermissionLevel.READ)
    )
):
    """List data conflicts for a sync job."""
    job = _get_job(str(job_id), tenant_id)
    if not job:
        raise HTTPException(status_code=404, detail="Sync job not found")

    # In production, query from database
    return {
        "items": [],
        "total": 0,
        "page": page,
        "page_size": page_size,
        "total_pages": 0
    }
