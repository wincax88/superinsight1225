"""
Ticket management data models for SuperInsight platform.

Contains both SQLAlchemy ORM models for database persistence
and Pydantic models for API request/response handling.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from uuid import UUID, uuid4
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from sqlalchemy import String, Text, Float, Integer, DateTime, ForeignKey, Enum as SQLEnum, Boolean
from sqlalchemy.dialects.postgresql import UUID as PGUUID, JSONB
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.sql import func

from src.database.connection import Base


# ==================== Enumerations ====================

class TicketStatus(str, Enum):
    """Ticket status enumeration."""
    OPEN = "open"                    # 待处理
    ASSIGNED = "assigned"            # 已分配
    IN_PROGRESS = "in_progress"      # 处理中
    PENDING_REVIEW = "pending_review"  # 待审核
    RESOLVED = "resolved"            # 已解决
    CLOSED = "closed"                # 已关闭
    ESCALATED = "escalated"          # 已升级


class TicketPriority(str, Enum):
    """Ticket priority enumeration."""
    CRITICAL = "critical"    # 紧急 - 4小时 SLA
    HIGH = "high"            # 高 - 8小时 SLA
    MEDIUM = "medium"        # 中 - 24小时 SLA
    LOW = "low"              # 低 - 72小时 SLA


class TicketType(str, Enum):
    """Ticket type enumeration."""
    QUALITY_ISSUE = "quality_issue"          # 质量问题
    ANNOTATION_ERROR = "annotation_error"    # 标注错误
    DATA_REPAIR = "data_repair"              # 数据修复
    REVIEW_REQUEST = "review_request"        # 审核请求
    TRAINING_FEEDBACK = "training_feedback"  # 培训反馈
    CUSTOMER_COMPLAINT = "customer_complaint"  # 客户投诉
    SYSTEM_ERROR = "system_error"            # 系统错误


# ==================== SQLAlchemy ORM Models ====================

class TicketModel(Base):
    """
    Ticket table for managing work orders.

    Extends quality issue concept with SLA, assignment, and lifecycle management.
    """
    __tablename__ = "tickets"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)

    # Basic info
    ticket_type: Mapped[TicketType] = mapped_column(SQLEnum(TicketType), nullable=False)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    priority: Mapped[TicketPriority] = mapped_column(SQLEnum(TicketPriority), default=TicketPriority.MEDIUM)
    status: Mapped[TicketStatus] = mapped_column(SQLEnum(TicketStatus), default=TicketStatus.OPEN)

    # Foreign keys
    quality_issue_id: Mapped[Optional[UUID]] = mapped_column(PGUUID(as_uuid=True), ForeignKey("quality_issues.id"), nullable=True)
    task_id: Mapped[Optional[UUID]] = mapped_column(PGUUID(as_uuid=True), ForeignKey("tasks.id"), nullable=True)

    # Assignment
    assigned_to: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    assigned_by: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    assigned_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Skill requirements for smart dispatch
    skill_requirements: Mapped[dict] = mapped_column(JSONB, default={})
    workload_weight: Mapped[float] = mapped_column(Float, default=1.0)

    # SLA tracking
    sla_deadline: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    sla_breached: Mapped[bool] = mapped_column(Boolean, default=False)
    escalation_level: Mapped[int] = mapped_column(Integer, default=0)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Multi-tenant support
    tenant_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, index=True)
    created_by: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # Additional metadata
    metadata: Mapped[dict] = mapped_column(JSONB, default={})
    resolution_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class AnnotatorSkillModel(Base):
    """
    Annotator skill matrix for intelligent dispatch.

    Tracks skill levels, performance history, and workload capacity.
    """
    __tablename__ = "annotator_skills"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)

    # User identification
    user_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    tenant_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, index=True)

    # Skill information
    skill_type: Mapped[str] = mapped_column(String(50), nullable=False)
    skill_level: Mapped[float] = mapped_column(Float, default=0.5)  # 0-1 scale

    # Performance metrics
    total_tasks: Mapped[int] = mapped_column(Integer, default=0)
    completed_tasks: Mapped[int] = mapped_column(Integer, default=0)
    success_rate: Mapped[float] = mapped_column(Float, default=0.0)  # 0-1 scale
    avg_resolution_time: Mapped[int] = mapped_column(Integer, default=0)  # seconds
    avg_quality_score: Mapped[float] = mapped_column(Float, default=0.0)  # 0-1 scale

    # Workload
    current_workload: Mapped[int] = mapped_column(Integer, default=0)  # active tickets
    max_workload: Mapped[int] = mapped_column(Integer, default=10)  # max concurrent tickets

    # Availability
    is_available: Mapped[bool] = mapped_column(Boolean, default=True)
    last_active_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class TicketHistoryModel(Base):
    """
    Ticket history for audit trail.

    Tracks all status changes and actions on tickets.
    """
    __tablename__ = "ticket_history"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    ticket_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("tickets.id"), nullable=False)

    action: Mapped[str] = mapped_column(String(50), nullable=False)  # created, assigned, status_changed, etc.
    old_value: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    new_value: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    performed_by: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


# ==================== Pydantic Business Models ====================

class Ticket(BaseModel):
    """Pydantic model for ticket data transfer."""

    id: UUID = Field(default_factory=uuid4, description="Unique ticket identifier")
    ticket_type: TicketType = Field(..., description="Type of ticket")
    title: str = Field(..., min_length=1, max_length=200, description="Ticket title")
    description: Optional[str] = Field(None, description="Detailed description")
    priority: TicketPriority = Field(default=TicketPriority.MEDIUM, description="Priority level")
    status: TicketStatus = Field(default=TicketStatus.OPEN, description="Current status")

    quality_issue_id: Optional[UUID] = Field(None, description="Related quality issue")
    task_id: Optional[UUID] = Field(None, description="Related task")

    assigned_to: Optional[str] = Field(None, description="Assigned user ID")
    assigned_by: Optional[str] = Field(None, description="User who assigned")
    assigned_at: Optional[datetime] = Field(None, description="Assignment time")

    skill_requirements: Dict[str, Any] = Field(default_factory=dict, description="Required skills")
    workload_weight: float = Field(default=1.0, ge=0.1, le=10.0, description="Workload weight")

    sla_deadline: Optional[datetime] = Field(None, description="SLA deadline")
    sla_breached: bool = Field(default=False, description="SLA breach flag")
    escalation_level: int = Field(default=0, ge=0, description="Escalation level")

    tenant_id: Optional[str] = Field(None, description="Tenant identifier")
    created_by: Optional[str] = Field(None, description="Creator user ID")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation time")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update time")
    resolved_at: Optional[datetime] = Field(None, description="Resolution time")

    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    resolution_notes: Optional[str] = Field(None, description="Resolution notes")

    @field_validator('title')
    @classmethod
    def validate_title(cls, v):
        """Validate that title is not empty."""
        if not v or not v.strip():
            raise ValueError('Title cannot be empty')
        return v.strip()

    def to_dict(self) -> Dict[str, Any]:
        """Convert ticket to dictionary for JSON serialization."""
        return {
            "id": str(self.id),
            "ticket_type": self.ticket_type.value,
            "title": self.title,
            "description": self.description,
            "priority": self.priority.value,
            "status": self.status.value,
            "quality_issue_id": str(self.quality_issue_id) if self.quality_issue_id else None,
            "task_id": str(self.task_id) if self.task_id else None,
            "assigned_to": self.assigned_to,
            "assigned_by": self.assigned_by,
            "assigned_at": self.assigned_at.isoformat() if self.assigned_at else None,
            "skill_requirements": self.skill_requirements,
            "workload_weight": self.workload_weight,
            "sla_deadline": self.sla_deadline.isoformat() if self.sla_deadline else None,
            "sla_breached": self.sla_breached,
            "escalation_level": self.escalation_level,
            "tenant_id": self.tenant_id,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "metadata": self.metadata,
            "resolution_notes": self.resolution_notes,
        }

    model_config = ConfigDict(
        json_encoders={
            UUID: str,
            datetime: lambda v: v.isoformat(),
        }
    )


class AnnotatorSkill(BaseModel):
    """Pydantic model for annotator skill data transfer."""

    id: UUID = Field(default_factory=uuid4, description="Unique skill record identifier")
    user_id: str = Field(..., description="User identifier")
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")

    skill_type: str = Field(..., description="Skill type identifier")
    skill_level: float = Field(default=0.5, ge=0.0, le=1.0, description="Skill level (0-1)")

    total_tasks: int = Field(default=0, ge=0, description="Total tasks assigned")
    completed_tasks: int = Field(default=0, ge=0, description="Completed tasks")
    success_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Success rate")
    avg_resolution_time: int = Field(default=0, ge=0, description="Avg resolution time (seconds)")
    avg_quality_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Avg quality score")

    current_workload: int = Field(default=0, ge=0, description="Current active tickets")
    max_workload: int = Field(default=10, ge=1, description="Max concurrent tickets")

    is_available: bool = Field(default=True, description="Availability status")
    last_active_at: Optional[datetime] = Field(None, description="Last activity time")

    created_at: datetime = Field(default_factory=datetime.now, description="Creation time")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update time")

    @property
    def available_capacity(self) -> int:
        """Calculate available workload capacity."""
        return max(0, self.max_workload - self.current_workload)

    @property
    def utilization_rate(self) -> float:
        """Calculate workload utilization rate."""
        if self.max_workload == 0:
            return 1.0
        return self.current_workload / self.max_workload

    def to_dict(self) -> Dict[str, Any]:
        """Convert skill record to dictionary."""
        return {
            "id": str(self.id),
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
            "skill_type": self.skill_type,
            "skill_level": self.skill_level,
            "total_tasks": self.total_tasks,
            "completed_tasks": self.completed_tasks,
            "success_rate": self.success_rate,
            "avg_resolution_time": self.avg_resolution_time,
            "avg_quality_score": self.avg_quality_score,
            "current_workload": self.current_workload,
            "max_workload": self.max_workload,
            "available_capacity": self.available_capacity,
            "utilization_rate": self.utilization_rate,
            "is_available": self.is_available,
            "last_active_at": self.last_active_at.isoformat() if self.last_active_at else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    model_config = ConfigDict(
        json_encoders={
            UUID: str,
            datetime: lambda v: v.isoformat(),
        }
    )


class TicketHistory(BaseModel):
    """Pydantic model for ticket history records."""

    id: UUID = Field(default_factory=uuid4)
    ticket_id: UUID = Field(..., description="Related ticket ID")
    action: str = Field(..., description="Action performed")
    old_value: Optional[str] = Field(None, description="Previous value")
    new_value: Optional[str] = Field(None, description="New value")
    performed_by: Optional[str] = Field(None, description="User who performed action")
    notes: Optional[str] = Field(None, description="Additional notes")
    created_at: datetime = Field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert history record to dictionary."""
        return {
            "id": str(self.id),
            "ticket_id": str(self.ticket_id),
            "action": self.action,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "performed_by": self.performed_by,
            "notes": self.notes,
            "created_at": self.created_at.isoformat(),
        }


# ==================== SLA Configuration ====================

class SLAConfig:
    """SLA configuration for different priority levels (standard mode)."""

    # SLA deadlines in seconds (standard mode - user confirmed)
    SLA_DEADLINES = {
        TicketPriority.CRITICAL: 4 * 3600,   # 4 hours
        TicketPriority.HIGH: 8 * 3600,       # 8 hours
        TicketPriority.MEDIUM: 24 * 3600,    # 24 hours
        TicketPriority.LOW: 72 * 3600,       # 72 hours
    }

    # Warning thresholds (percentage of SLA time remaining)
    WARNING_THRESHOLD = 0.25  # 25% time remaining

    # Escalation intervals
    ESCALATION_INTERVALS = {
        TicketPriority.CRITICAL: 1 * 3600,   # 1 hour
        TicketPriority.HIGH: 2 * 3600,       # 2 hours
        TicketPriority.MEDIUM: 4 * 3600,     # 4 hours
        TicketPriority.LOW: 8 * 3600,        # 8 hours
    }

    @classmethod
    def get_sla_deadline(cls, priority: TicketPriority, created_at: datetime) -> datetime:
        """Calculate SLA deadline based on priority."""
        sla_seconds = cls.SLA_DEADLINES.get(priority, cls.SLA_DEADLINES[TicketPriority.MEDIUM])
        return created_at + timedelta(seconds=sla_seconds)

    @classmethod
    def get_time_remaining(cls, sla_deadline: datetime) -> int:
        """Get remaining time in seconds (negative if breached)."""
        return int((sla_deadline - datetime.now()).total_seconds())

    @classmethod
    def is_sla_breached(cls, sla_deadline: datetime) -> bool:
        """Check if SLA has been breached."""
        return datetime.now() > sla_deadline

    @classmethod
    def needs_warning(cls, sla_deadline: datetime, priority: TicketPriority) -> bool:
        """Check if SLA warning should be sent."""
        total_sla = cls.SLA_DEADLINES.get(priority, cls.SLA_DEADLINES[TicketPriority.MEDIUM])
        remaining = cls.get_time_remaining(sla_deadline)
        return 0 < remaining <= (total_sla * cls.WARNING_THRESHOLD)
