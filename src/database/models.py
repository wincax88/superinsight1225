"""
SQLAlchemy ORM models for SuperInsight Platform database tables.

These models define the database schema using SQLAlchemy ORM.
"""

from datetime import datetime
from uuid import uuid4
from sqlalchemy import String, Text, Float, Integer, DateTime, ForeignKey, Enum as SQLEnum, JSON, Boolean
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.sql import func
import enum
from typing import Optional, List

from src.database.connection import Base
from src.config.settings import settings


def get_json_type():
    """Get appropriate JSON type based on database backend"""
    if settings.database.database_url.startswith('sqlite'):
        return JSON  # SQLite uses JSON type
    else:
        return JSONB  # PostgreSQL uses JSONB type


def get_uuid_type():
    """Get appropriate UUID type based on database backend"""
    if settings.database.database_url.startswith('sqlite'):
        return String(36)  # SQLite uses string for UUID
    else:
        return UUID(as_uuid=True)  # PostgreSQL uses UUID type


def get_uuid_default():
    """Get appropriate UUID default based on database backend"""
    if settings.database.database_url.startswith('sqlite'):
        return lambda: str(uuid4())  # SQLite needs string UUID
    else:
        return uuid4  # PostgreSQL can use UUID directly


class TaskStatus(str, enum.Enum):
    """Task status enumeration."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    REVIEWED = "reviewed"


class IssueSeverity(str, enum.Enum):
    """Quality issue severity enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IssueStatus(str, enum.Enum):
    """Quality issue status enumeration."""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"


class SyncStatus(str, enum.Enum):
    """Document sync status enumeration."""
    PENDING = "pending"
    SYNCING = "syncing"
    SYNCED = "synced"
    CONFLICT = "conflict"
    FAILED = "failed"


class DocumentModel(Base):
    """
    Document table for storing source documents.

    Stores documents from various sources (database, file, API) with JSONB support.
    Extended with sync-related fields for data synchronization system.
    """
    __tablename__ = "documents"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    source_type: Mapped[str] = mapped_column(String(50), nullable=False)
    source_config: Mapped[dict] = mapped_column(JSONB, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    document_metadata: Mapped[dict] = mapped_column("metadata", JSONB, default={})
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Sync-related fields (Phase 1.2 extension)
    tenant_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, index=True)
    sync_status: Mapped[Optional[SyncStatus]] = mapped_column(SQLEnum(SyncStatus), nullable=True, default=None)
    sync_version: Mapped[int] = mapped_column(Integer, default=1)
    sync_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)  # SHA-256 content hash
    last_synced_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    sync_source_id: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)  # External source record ID
    sync_job_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)  # UUID of sync job
    is_from_sync: Mapped[bool] = mapped_column(Boolean, default=False)  # Whether document came from sync
    sync_metadata: Mapped[dict] = mapped_column(JSONB, default={})  # Additional sync-related metadata

    # Relationship to tasks
    tasks: Mapped[List["TaskModel"]] = relationship("TaskModel", back_populates="document")


class TaskModel(Base):
    """
    Task table for managing annotation tasks.

    Links documents to annotation projects and tracks progress.
    Extended with sync-related fields for tracking synchronization status.
    """
    __tablename__ = "tasks"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    document_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    project_id: Mapped[str] = mapped_column(String(100), nullable=False)
    status: Mapped[TaskStatus] = mapped_column(SQLEnum(TaskStatus), default=TaskStatus.PENDING)
    annotations: Mapped[list] = mapped_column(JSONB, default=[])
    ai_predictions: Mapped[list] = mapped_column(JSONB, default=[])
    quality_score: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Sync-related fields (Phase 1.2 extension)
    tenant_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, index=True)
    sync_status: Mapped[Optional[SyncStatus]] = mapped_column(SQLEnum(SyncStatus), nullable=True, default=None)
    sync_version: Mapped[int] = mapped_column(Integer, default=1)
    last_synced_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    sync_execution_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)  # UUID of sync execution
    is_from_sync: Mapped[bool] = mapped_column(Boolean, default=False)  # Whether task came from sync
    sync_metadata: Mapped[dict] = mapped_column(JSONB, default={})  # Additional sync-related metadata

    # Relationships
    document: Mapped["DocumentModel"] = relationship("DocumentModel", back_populates="tasks")
    quality_issues: Mapped[List["QualityIssueModel"]] = relationship("QualityIssueModel", back_populates="task")
    billing_records: Mapped[List["BillingRecordModel"]] = relationship("BillingRecordModel", back_populates="task")


class BillingRecordModel(Base):
    """
    Billing records table for tracking annotation costs.
    
    Stores billing information per tenant, user, and task.
    """
    __tablename__ = "billing_records"
    
    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    tenant_id: Mapped[str] = mapped_column(String(100), nullable=False)
    user_id: Mapped[str] = mapped_column(String(100), nullable=False)
    task_id: Mapped[Optional[UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("tasks.id"), nullable=True)
    annotation_count: Mapped[int] = mapped_column(Integer, default=0)
    time_spent: Mapped[int] = mapped_column(Integer, default=0)  # in seconds
    cost: Mapped[float] = mapped_column(Float, default=0.0)
    billing_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.current_date())
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    
    # Relationship
    task: Mapped[Optional["TaskModel"]] = relationship("TaskModel", back_populates="billing_records")


class QualityIssueModel(Base):
    """
    Quality issues table for tracking quality problems and work orders.
    
    Manages quality issues found during annotation review.
    """
    __tablename__ = "quality_issues"
    
    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    task_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tasks.id"), nullable=False)
    issue_type: Mapped[str] = mapped_column(String(50), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    severity: Mapped[IssueSeverity] = mapped_column(SQLEnum(IssueSeverity), default=IssueSeverity.MEDIUM)
    status: Mapped[IssueStatus] = mapped_column(SQLEnum(IssueStatus), default=IssueStatus.OPEN)
    assignee_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Relationship
    task: Mapped["TaskModel"] = relationship("TaskModel", back_populates="quality_issues")

# Import security models to ensure they are registered with SQLAlchemy
from src.security.models import (
    UserModel, ProjectPermissionModel, IPWhitelistModel,
    AuditLogModel, DataMaskingRuleModel
)

# Import sync models to ensure they are registered with SQLAlchemy
from src.sync.models import (
    DataSourceModel, SyncJobModel, SyncExecutionModel,
    DataConflictModel, SyncRuleModel, TransformationRuleModel,
    IndustryDatasetModel, SyncAuditLogModel, DataQualityScoreModel
)