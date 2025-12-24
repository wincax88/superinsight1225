"""
SQLAlchemy ORM models for Data Sync System.

These models define the database schema for the data synchronization system,
including sync jobs, executions, conflicts, audit logs, data sources, rules,
transformations, and industry datasets.
"""

from datetime import datetime
from uuid import uuid4
from sqlalchemy import (
    String, Text, Float, Integer, DateTime, ForeignKey,
    Enum as SQLEnum, JSON, Boolean, BigInteger, Index
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.sql import func
import enum
from typing import Optional, List

from src.database.connection import Base


# ============================================================================
# Enumerations
# ============================================================================

class SyncDirection(str, enum.Enum):
    """Sync direction enumeration."""
    LOCAL_TO_CLOUD = "local_to_cloud"
    CLOUD_TO_LOCAL = "cloud_to_local"
    BIDIRECTIONAL = "bidirectional"
    PULL = "pull"
    PUSH = "push"


class SyncFrequency(str, enum.Enum):
    """Sync frequency enumeration."""
    REAL_TIME = "real_time"
    SCHEDULED = "scheduled"
    MANUAL = "manual"
    ON_CHANGE = "on_change"


class SyncJobStatus(str, enum.Enum):
    """Sync job status enumeration."""
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    DISABLED = "disabled"
    ARCHIVED = "archived"


class SyncExecutionStatus(str, enum.Enum):
    """Sync execution status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PARTIAL = "partial"


class ConflictType(str, enum.Enum):
    """Data conflict type enumeration."""
    VERSION_CONFLICT = "version_conflict"
    CONTENT_CONFLICT = "content_conflict"
    SCHEMA_CONFLICT = "schema_conflict"
    DELETE_CONFLICT = "delete_conflict"
    CONSTRAINT_CONFLICT = "constraint_conflict"


class ConflictResolutionStrategy(str, enum.Enum):
    """Conflict resolution strategy enumeration."""
    TIMESTAMP_BASED = "timestamp_based"
    SOURCE_WINS = "source_wins"
    TARGET_WINS = "target_wins"
    MANUAL = "manual"
    FIELD_MERGE = "field_merge"
    BUSINESS_RULE = "business_rule"


class ConflictStatus(str, enum.Enum):
    """Conflict status enumeration."""
    PENDING = "pending"
    AUTO_RESOLVED = "auto_resolved"
    MANUALLY_RESOLVED = "manually_resolved"
    ESCALATED = "escalated"
    IGNORED = "ignored"


class DataSourceType(str, enum.Enum):
    """Data source type enumeration."""
    MYSQL = "mysql"
    POSTGRESQL = "postgresql"
    ORACLE = "oracle"
    MONGODB = "mongodb"
    SQLSERVER = "sqlserver"
    REST_API = "rest_api"
    GRAPHQL_API = "graphql_api"
    SOAP_API = "soap_api"
    FTP = "ftp"
    SFTP = "sftp"
    S3 = "s3"
    LOCAL_FILE = "local_file"
    WEBHOOK = "webhook"


class DataSourceStatus(str, enum.Enum):
    """Data source status enumeration."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    TESTING = "testing"


class TransformationType(str, enum.Enum):
    """Transformation type enumeration."""
    FIELD_MAPPING = "field_mapping"
    DATA_TYPE_CONVERSION = "data_type_conversion"
    VALUE_TRANSFORMATION = "value_transformation"
    AGGREGATION = "aggregation"
    FILTERING = "filtering"
    ENRICHMENT = "enrichment"
    NORMALIZATION = "normalization"
    CUSTOM_SCRIPT = "custom_script"


class DatasetCategory(str, enum.Enum):
    """Industry dataset category enumeration."""
    FINANCE = "finance"
    HEALTHCARE = "healthcare"
    LEGAL = "legal"
    GENERAL = "general"
    TECHNOLOGY = "technology"
    RETAIL = "retail"
    MANUFACTURING = "manufacturing"
    EDUCATION = "education"


class DatasetStatus(str, enum.Enum):
    """Industry dataset status enumeration."""
    AVAILABLE = "available"
    DOWNLOADING = "downloading"
    PROCESSING = "processing"
    INTEGRATED = "integrated"
    ERROR = "error"
    DEPRECATED = "deprecated"


class AuditAction(str, enum.Enum):
    """Sync audit action enumeration."""
    JOB_CREATED = "job_created"
    JOB_UPDATED = "job_updated"
    JOB_DELETED = "job_deleted"
    JOB_STARTED = "job_started"
    JOB_STOPPED = "job_stopped"
    JOB_PAUSED = "job_paused"
    JOB_RESUMED = "job_resumed"
    SYNC_STARTED = "sync_started"
    SYNC_COMPLETED = "sync_completed"
    SYNC_FAILED = "sync_failed"
    CONFLICT_DETECTED = "conflict_detected"
    CONFLICT_RESOLVED = "conflict_resolved"
    DATA_PUSHED = "data_pushed"
    DATA_PULLED = "data_pulled"
    SOURCE_CONNECTED = "source_connected"
    SOURCE_DISCONNECTED = "source_disconnected"


# ============================================================================
# Data Source Model
# ============================================================================

class DataSourceModel(Base):
    """
    Data source configuration table.

    Stores configuration for various data sources including databases, APIs, and file systems.
    """
    __tablename__ = "data_sources"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    tenant_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    source_type: Mapped[DataSourceType] = mapped_column(SQLEnum(DataSourceType), nullable=False)
    status: Mapped[DataSourceStatus] = mapped_column(SQLEnum(DataSourceStatus), default=DataSourceStatus.INACTIVE)

    # Connection configuration (encrypted)
    connection_config: Mapped[dict] = mapped_column(JSONB, nullable=False, default={})
    # Schema/metadata information
    schema_config: Mapped[dict] = mapped_column(JSONB, nullable=True, default={})

    # Connection pool settings
    pool_size: Mapped[int] = mapped_column(Integer, default=5)
    max_overflow: Mapped[int] = mapped_column(Integer, default=10)
    connection_timeout: Mapped[int] = mapped_column(Integer, default=30)

    # Health check
    last_health_check: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    health_check_status: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    created_by: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # Relationships
    sync_jobs: Mapped[List["SyncJobModel"]] = relationship("SyncJobModel", back_populates="data_source", foreign_keys="SyncJobModel.source_id")

    __table_args__ = (
        Index('idx_data_sources_tenant_type', 'tenant_id', 'source_type'),
    )


# ============================================================================
# Sync Job Model
# ============================================================================

class SyncJobModel(Base):
    """
    Sync job configuration table.

    Defines synchronization jobs with scheduling, direction, and rules.
    """
    __tablename__ = "sync_jobs"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    tenant_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Source and target
    source_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("data_sources.id"), nullable=False)
    target_config: Mapped[dict] = mapped_column(JSONB, nullable=False, default={})

    # Sync configuration
    direction: Mapped[SyncDirection] = mapped_column(SQLEnum(SyncDirection), default=SyncDirection.PULL)
    frequency: Mapped[SyncFrequency] = mapped_column(SQLEnum(SyncFrequency), default=SyncFrequency.MANUAL)
    status: Mapped[SyncJobStatus] = mapped_column(SQLEnum(SyncJobStatus), default=SyncJobStatus.DRAFT)

    # Schedule (cron expression for scheduled jobs)
    schedule_cron: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    schedule_timezone: Mapped[str] = mapped_column(String(50), default="UTC")

    # Sync settings
    batch_size: Mapped[int] = mapped_column(Integer, default=1000)
    max_retries: Mapped[int] = mapped_column(Integer, default=3)
    retry_delay: Mapped[int] = mapped_column(Integer, default=60)  # seconds
    timeout: Mapped[int] = mapped_column(Integer, default=3600)  # seconds

    # Incremental sync settings
    enable_incremental: Mapped[bool] = mapped_column(Boolean, default=True)
    incremental_field: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    last_sync_value: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    # Conflict resolution
    conflict_resolution: Mapped[ConflictResolutionStrategy] = mapped_column(
        SQLEnum(ConflictResolutionStrategy),
        default=ConflictResolutionStrategy.TIMESTAMP_BASED
    )

    # Data transformation
    transformation_rules: Mapped[list] = mapped_column(JSONB, default=[])

    # Filtering
    filter_conditions: Mapped[dict] = mapped_column(JSONB, default={})

    # Encryption and compression
    enable_encryption: Mapped[bool] = mapped_column(Boolean, default=True)
    enable_compression: Mapped[bool] = mapped_column(Boolean, default=False)

    # Statistics
    total_executions: Mapped[int] = mapped_column(Integer, default=0)
    successful_executions: Mapped[int] = mapped_column(Integer, default=0)
    failed_executions: Mapped[int] = mapped_column(Integer, default=0)
    total_records_synced: Mapped[int] = mapped_column(BigInteger, default=0)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    last_executed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    next_scheduled_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_by: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # Relationships
    data_source: Mapped["DataSourceModel"] = relationship("DataSourceModel", back_populates="sync_jobs", foreign_keys=[source_id])
    executions: Mapped[List["SyncExecutionModel"]] = relationship("SyncExecutionModel", back_populates="sync_job")
    sync_rules: Mapped[List["SyncRuleModel"]] = relationship("SyncRuleModel", back_populates="sync_job")

    __table_args__ = (
        Index('idx_sync_jobs_tenant_status', 'tenant_id', 'status'),
        Index('idx_sync_jobs_next_scheduled', 'next_scheduled_at'),
    )


# ============================================================================
# Sync Execution Model
# ============================================================================

class SyncExecutionModel(Base):
    """
    Sync execution history table.

    Records each execution of a sync job with detailed metrics.
    """
    __tablename__ = "sync_executions"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    job_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("sync_jobs.id"), nullable=False, index=True)
    tenant_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)

    # Execution status
    status: Mapped[SyncExecutionStatus] = mapped_column(SQLEnum(SyncExecutionStatus), default=SyncExecutionStatus.PENDING)

    # Timing
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Record counts
    records_total: Mapped[int] = mapped_column(BigInteger, default=0)
    records_processed: Mapped[int] = mapped_column(BigInteger, default=0)
    records_inserted: Mapped[int] = mapped_column(BigInteger, default=0)
    records_updated: Mapped[int] = mapped_column(BigInteger, default=0)
    records_deleted: Mapped[int] = mapped_column(BigInteger, default=0)
    records_skipped: Mapped[int] = mapped_column(BigInteger, default=0)
    records_failed: Mapped[int] = mapped_column(BigInteger, default=0)

    # Data transfer metrics
    bytes_transferred: Mapped[int] = mapped_column(BigInteger, default=0)

    # Error tracking
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    error_details: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)

    # Checkpoint for resume
    checkpoint_data: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    # Execution context
    triggered_by: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)  # user_id, system, schedule
    trigger_type: Mapped[str] = mapped_column(String(50), default="manual")  # manual, scheduled, webhook, cdc

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    sync_job: Mapped["SyncJobModel"] = relationship("SyncJobModel", back_populates="executions")
    conflicts: Mapped[List["DataConflictModel"]] = relationship("DataConflictModel", back_populates="execution")

    __table_args__ = (
        Index('idx_sync_executions_job_status', 'job_id', 'status'),
        Index('idx_sync_executions_tenant_time', 'tenant_id', 'created_at'),
    )


# ============================================================================
# Data Conflict Model
# ============================================================================

class DataConflictModel(Base):
    """
    Data conflict records table.

    Stores detected conflicts during synchronization with resolution status.
    """
    __tablename__ = "data_conflicts"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    execution_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("sync_executions.id"), nullable=False, index=True)
    tenant_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)

    # Conflict identification
    conflict_type: Mapped[ConflictType] = mapped_column(SQLEnum(ConflictType), nullable=False)
    status: Mapped[ConflictStatus] = mapped_column(SQLEnum(ConflictStatus), default=ConflictStatus.PENDING)

    # Record information
    record_id: Mapped[str] = mapped_column(String(500), nullable=False)  # Primary key of conflicting record
    table_name: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    field_name: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)

    # Conflict data
    source_value: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    target_value: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    source_version: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    target_version: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    source_timestamp: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    target_timestamp: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Resolution
    resolution_strategy: Mapped[Optional[ConflictResolutionStrategy]] = mapped_column(
        SQLEnum(ConflictResolutionStrategy),
        nullable=True
    )
    resolved_value: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    resolved_by: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    resolution_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Priority and severity
    priority: Mapped[int] = mapped_column(Integer, default=5)  # 1-10, higher is more urgent

    # Timestamps
    detected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    execution: Mapped["SyncExecutionModel"] = relationship("SyncExecutionModel", back_populates="conflicts")

    __table_args__ = (
        Index('idx_data_conflicts_execution_status', 'execution_id', 'status'),
        Index('idx_data_conflicts_tenant_pending', 'tenant_id', 'status'),
    )


# ============================================================================
# Sync Rule Model
# ============================================================================

class SyncRuleModel(Base):
    """
    Sync rules configuration table.

    Defines fine-grained synchronization rules for specific tables or fields.
    """
    __tablename__ = "sync_rules"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    job_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("sync_jobs.id"), nullable=False, index=True)
    tenant_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Rule scope
    table_pattern: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)  # Regex pattern for table names
    field_pattern: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)  # Regex pattern for field names

    # Rule configuration
    direction: Mapped[SyncDirection] = mapped_column(SQLEnum(SyncDirection), nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    priority: Mapped[int] = mapped_column(Integer, default=0)  # Higher priority rules applied first

    # Sync behavior
    sync_deletes: Mapped[bool] = mapped_column(Boolean, default=False)
    sync_updates: Mapped[bool] = mapped_column(Boolean, default=True)
    sync_inserts: Mapped[bool] = mapped_column(Boolean, default=True)

    # Filtering conditions (JSON query format)
    filter_conditions: Mapped[dict] = mapped_column(JSONB, default={})

    # Conflict resolution override
    conflict_resolution: Mapped[Optional[ConflictResolutionStrategy]] = mapped_column(
        SQLEnum(ConflictResolutionStrategy),
        nullable=True
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    sync_job: Mapped["SyncJobModel"] = relationship("SyncJobModel", back_populates="sync_rules")


# ============================================================================
# Transformation Rule Model
# ============================================================================

class TransformationRuleModel(Base):
    """
    Data transformation rules table.

    Defines how data should be transformed during synchronization.
    """
    __tablename__ = "transformation_rules"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    tenant_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Transformation type
    transformation_type: Mapped[TransformationType] = mapped_column(SQLEnum(TransformationType), nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    priority: Mapped[int] = mapped_column(Integer, default=0)

    # Source and target
    source_field: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    target_field: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)

    # Transformation configuration
    transformation_config: Mapped[dict] = mapped_column(JSONB, nullable=False, default={})

    # Custom script (for CUSTOM_SCRIPT type)
    custom_script: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    script_language: Mapped[str] = mapped_column(String(50), default="python")

    # Validation
    validation_rules: Mapped[dict] = mapped_column(JSONB, default={})

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


# ============================================================================
# Industry Dataset Model
# ============================================================================

class IndustryDatasetModel(Base):
    """
    Industry datasets metadata table.

    Stores metadata about available industry datasets for data enrichment and noise dilution.
    """
    __tablename__ = "industry_datasets"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    tenant_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, index=True)  # Null for global datasets

    # Dataset identification
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    display_name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    version: Mapped[str] = mapped_column(String(50), default="1.0.0")

    # Source information
    source_platform: Mapped[str] = mapped_column(String(100), nullable=False)  # huggingface, kaggle, github
    source_url: Mapped[str] = mapped_column(String(500), nullable=False)
    source_identifier: Mapped[str] = mapped_column(String(200), nullable=False)  # dataset id on platform

    # Category and domain
    category: Mapped[DatasetCategory] = mapped_column(SQLEnum(DatasetCategory), nullable=False)
    domain_tags: Mapped[list] = mapped_column(JSONB, default=[])  # ["qa", "sentiment", "ner", etc.]
    language: Mapped[str] = mapped_column(String(50), default="zh")  # ISO 639-1 code

    # Status and availability
    status: Mapped[DatasetStatus] = mapped_column(SQLEnum(DatasetStatus), default=DatasetStatus.AVAILABLE)
    is_public: Mapped[bool] = mapped_column(Boolean, default=True)

    # Dataset statistics
    total_records: Mapped[int] = mapped_column(BigInteger, default=0)
    file_size_bytes: Mapped[int] = mapped_column(BigInteger, default=0)

    # Quality metrics
    quality_score: Mapped[float] = mapped_column(Float, default=0.0)  # 0-1 score
    quality_metrics: Mapped[dict] = mapped_column(JSONB, default={})  # detailed quality breakdown

    # Integration information
    local_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    integration_config: Mapped[dict] = mapped_column(JSONB, default={})  # how to integrate with customer data

    # Dilution parameters
    recommended_dilution_ratio: Mapped[float] = mapped_column(Float, default=0.3)  # 30% default
    min_dilution_ratio: Mapped[float] = mapped_column(Float, default=0.1)
    max_dilution_ratio: Mapped[float] = mapped_column(Float, default=0.5)

    # Usage tracking
    download_count: Mapped[int] = mapped_column(Integer, default=0)
    integration_count: Mapped[int] = mapped_column(Integer, default=0)
    last_downloaded_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    last_integrated_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # License information
    license_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    license_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index('idx_industry_datasets_category', 'category'),
        Index('idx_industry_datasets_status', 'status'),
        Index('idx_industry_datasets_platform', 'source_platform'),
    )


# ============================================================================
# Sync Audit Log Model
# ============================================================================

class SyncAuditLogModel(Base):
    """
    Sync operation audit logs table.

    Records all synchronization-related actions for compliance and debugging.
    """
    __tablename__ = "sync_audit_logs"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    tenant_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)

    # Action identification
    action: Mapped[AuditAction] = mapped_column(SQLEnum(AuditAction), nullable=False)

    # Related entities
    job_id: Mapped[Optional[UUID]] = mapped_column(UUID(as_uuid=True), nullable=True, index=True)
    execution_id: Mapped[Optional[UUID]] = mapped_column(UUID(as_uuid=True), nullable=True)
    source_id: Mapped[Optional[UUID]] = mapped_column(UUID(as_uuid=True), nullable=True)

    # Actor information
    actor_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)  # user_id or system
    actor_type: Mapped[str] = mapped_column(String(50), default="user")  # user, system, scheduler
    actor_ip: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    # Action details
    action_details: Mapped[dict] = mapped_column(JSONB, default={})

    # Before/After state (for changes)
    state_before: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    state_after: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    # Result
    success: Mapped[bool] = mapped_column(Boolean, default=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Timestamp
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index('idx_sync_audit_logs_tenant_action', 'tenant_id', 'action'),
        Index('idx_sync_audit_logs_tenant_time', 'tenant_id', 'created_at'),
        Index('idx_sync_audit_logs_job', 'job_id'),
    )


# ============================================================================
# Data Quality Score Model (for AI-friendly dataset tracking)
# ============================================================================

class DataQualityScoreModel(Base):
    """
    Data quality scores table.

    Tracks quality metrics for datasets before and after processing.
    """
    __tablename__ = "data_quality_scores"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    tenant_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)

    # Reference to data batch or dataset
    dataset_id: Mapped[Optional[UUID]] = mapped_column(UUID(as_uuid=True), nullable=True)
    batch_id: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)

    # Quality stage
    stage: Mapped[str] = mapped_column(String(50), nullable=False)  # raw, cleaned, annotated, augmented, final

    # Overall score
    overall_score: Mapped[float] = mapped_column(Float, nullable=False)  # 0-1

    # Detailed metrics
    completeness_score: Mapped[float] = mapped_column(Float, default=0.0)  # Data completeness
    consistency_score: Mapped[float] = mapped_column(Float, default=0.0)  # Format consistency
    accuracy_score: Mapped[float] = mapped_column(Float, default=0.0)  # Data accuracy
    relevancy_score: Mapped[float] = mapped_column(Float, default=0.0)  # Relevancy to domain
    noise_ratio: Mapped[float] = mapped_column(Float, default=0.0)  # Noise percentage
    duplicate_ratio: Mapped[float] = mapped_column(Float, default=0.0)  # Duplicate percentage

    # Sample counts
    total_samples: Mapped[int] = mapped_column(Integer, default=0)
    valid_samples: Mapped[int] = mapped_column(Integer, default=0)
    invalid_samples: Mapped[int] = mapped_column(Integer, default=0)

    # AI performance metrics (if applicable)
    ai_accuracy_improvement: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # Accuracy improvement percentage
    ai_response_time_improvement: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # Speed improvement percentage

    # Additional metrics
    metrics_details: Mapped[dict] = mapped_column(JSONB, default={})

    # Timestamps
    evaluated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index('idx_data_quality_scores_tenant_stage', 'tenant_id', 'stage'),
    )
