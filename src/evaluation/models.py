"""
Performance evaluation data models for SuperInsight platform.

Contains both SQLAlchemy ORM models for database persistence
and Pydantic models for API request/response handling.
"""

from datetime import datetime, date
from typing import Dict, Any, List, Optional
from uuid import UUID, uuid4
from enum import Enum
from pydantic import BaseModel, Field, field_validator, ConfigDict
from sqlalchemy import String, Text, Float, Integer, DateTime, Date, ForeignKey, Enum as SQLEnum, Boolean
from sqlalchemy.dialects.postgresql import UUID as PGUUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from src.database.connection import Base


# ==================== Enumerations ====================

class PerformanceStatus(str, Enum):
    """Performance record status enumeration."""
    DRAFT = "draft"              # 草稿
    PENDING_REVIEW = "pending_review"  # 待审核
    APPROVED = "approved"        # 已批准
    DISPUTED = "disputed"        # 有争议
    FINALIZED = "finalized"      # 已定稿


class AppealStatus(str, Enum):
    """Appeal status enumeration."""
    SUBMITTED = "submitted"      # 已提交
    UNDER_REVIEW = "under_review"  # 审核中
    APPROVED = "approved"        # 申诉通过
    REJECTED = "rejected"        # 申诉驳回
    WITHDRAWN = "withdrawn"      # 已撤回


class PerformancePeriod(str, Enum):
    """Performance evaluation period."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"


# ==================== SQLAlchemy ORM Models ====================

class PerformanceRecordModel(Base):
    """
    Performance record table for tracking annotator performance.

    Stores multi-dimensional performance metrics for each evaluation period.
    """
    __tablename__ = "performance_records"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)

    # User identification
    user_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    tenant_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, index=True)

    # Period information
    period_type: Mapped[PerformancePeriod] = mapped_column(SQLEnum(PerformancePeriod), default=PerformancePeriod.MONTHLY)
    period_start: Mapped[date] = mapped_column(Date, nullable=False)
    period_end: Mapped[date] = mapped_column(Date, nullable=False)

    # Quality dimension (40% weight)
    quality_score: Mapped[float] = mapped_column(Float, default=0.0)  # 0-1 scale
    accuracy_rate: Mapped[float] = mapped_column(Float, default=0.0)  # 准确率
    consistency_score: Mapped[float] = mapped_column(Float, default=0.0)  # 一致性
    error_rate: Mapped[float] = mapped_column(Float, default=0.0)  # 错误率

    # Efficiency dimension (30% weight)
    completion_rate: Mapped[float] = mapped_column(Float, default=0.0)  # 完成率
    avg_resolution_time: Mapped[int] = mapped_column(Integer, default=0)  # 平均解决时间(秒)
    tasks_completed: Mapped[int] = mapped_column(Integer, default=0)  # 完成任务数
    tasks_assigned: Mapped[int] = mapped_column(Integer, default=0)  # 分配任务数

    # Compliance dimension (20% weight)
    sla_compliance_rate: Mapped[float] = mapped_column(Float, default=0.0)  # SLA 合规率
    attendance_rate: Mapped[float] = mapped_column(Float, default=1.0)  # 出勤率
    rule_violations: Mapped[int] = mapped_column(Integer, default=0)  # 规则违反次数

    # Improvement dimension (10% weight)
    improvement_rate: Mapped[float] = mapped_column(Float, default=0.0)  # 改进率
    training_completion: Mapped[float] = mapped_column(Float, default=0.0)  # 培训完成率
    feedback_score: Mapped[float] = mapped_column(Float, default=0.0)  # 反馈评分

    # Overall scores
    overall_score: Mapped[float] = mapped_column(Float, default=0.0)  # 综合得分
    rank: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # 排名
    percentile: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # 百分位

    # Status and workflow
    status: Mapped[PerformanceStatus] = mapped_column(SQLEnum(PerformanceStatus), default=PerformanceStatus.DRAFT)
    reviewed_by: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    reviewed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Metadata
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    calculation_details: Mapped[dict] = mapped_column(JSONB, default={})
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class AppealModel(Base):
    """
    Appeal table for performance evaluation disputes.

    Tracks appeal submissions, reviews, and resolutions.
    """
    __tablename__ = "performance_appeals"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)

    # References
    performance_record_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("performance_records.id"), nullable=False)
    user_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    tenant_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, index=True)

    # Appeal content
    appeal_type: Mapped[str] = mapped_column(String(50), nullable=False)  # score_dispute, data_error, etc.
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    disputed_fields: Mapped[list] = mapped_column(JSONB, default=[])  # List of field names
    supporting_evidence: Mapped[dict] = mapped_column(JSONB, default={})  # Evidence data

    # Status and resolution
    status: Mapped[AppealStatus] = mapped_column(SQLEnum(AppealStatus), default=AppealStatus.SUBMITTED)
    reviewer_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    review_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    resolution: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Score adjustments if approved
    original_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    adjusted_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    adjustment_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Timestamps
    submitted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    reviewed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)


class PerformanceHistoryModel(Base):
    """
    Performance history for trend analysis.

    Stores daily snapshots for detailed trend tracking.
    """
    __tablename__ = "performance_history"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    tenant_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, index=True)

    record_date: Mapped[date] = mapped_column(Date, nullable=False)
    quality_score: Mapped[float] = mapped_column(Float, default=0.0)
    efficiency_score: Mapped[float] = mapped_column(Float, default=0.0)
    tasks_completed: Mapped[int] = mapped_column(Integer, default=0)
    avg_resolution_time: Mapped[int] = mapped_column(Integer, default=0)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


# ==================== Pydantic Business Models ====================

class PerformanceRecord(BaseModel):
    """Pydantic model for performance record data transfer."""

    id: UUID = Field(default_factory=uuid4, description="Unique record identifier")
    user_id: str = Field(..., description="User identifier")
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")

    period_type: PerformancePeriod = Field(default=PerformancePeriod.MONTHLY)
    period_start: date = Field(..., description="Period start date")
    period_end: date = Field(..., description="Period end date")

    # Quality metrics
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    accuracy_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    consistency_score: float = Field(default=0.0, ge=0.0, le=1.0)
    error_rate: float = Field(default=0.0, ge=0.0, le=1.0)

    # Efficiency metrics
    completion_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    avg_resolution_time: int = Field(default=0, ge=0)
    tasks_completed: int = Field(default=0, ge=0)
    tasks_assigned: int = Field(default=0, ge=0)

    # Compliance metrics
    sla_compliance_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    attendance_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    rule_violations: int = Field(default=0, ge=0)

    # Improvement metrics
    improvement_rate: float = Field(default=0.0, ge=-1.0, le=1.0)
    training_completion: float = Field(default=0.0, ge=0.0, le=1.0)
    feedback_score: float = Field(default=0.0, ge=0.0, le=1.0)

    # Overall
    overall_score: float = Field(default=0.0, ge=0.0, le=1.0)
    rank: Optional[int] = Field(None, ge=1)
    percentile: Optional[float] = Field(None, ge=0.0, le=100.0)

    status: PerformanceStatus = Field(default=PerformanceStatus.DRAFT)
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[datetime] = None

    notes: Optional[str] = None
    calculation_details: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary."""
        return {
            "id": str(self.id),
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
            "period_type": self.period_type.value,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "quality_score": self.quality_score,
            "accuracy_rate": self.accuracy_rate,
            "consistency_score": self.consistency_score,
            "error_rate": self.error_rate,
            "completion_rate": self.completion_rate,
            "avg_resolution_time": self.avg_resolution_time,
            "tasks_completed": self.tasks_completed,
            "tasks_assigned": self.tasks_assigned,
            "sla_compliance_rate": self.sla_compliance_rate,
            "attendance_rate": self.attendance_rate,
            "rule_violations": self.rule_violations,
            "improvement_rate": self.improvement_rate,
            "training_completion": self.training_completion,
            "feedback_score": self.feedback_score,
            "overall_score": self.overall_score,
            "rank": self.rank,
            "percentile": self.percentile,
            "status": self.status.value,
            "reviewed_by": self.reviewed_by,
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
            "notes": self.notes,
            "calculation_details": self.calculation_details,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    model_config = ConfigDict(
        json_encoders={
            UUID: str,
            datetime: lambda v: v.isoformat(),
            date: lambda v: v.isoformat(),
        }
    )


class Appeal(BaseModel):
    """Pydantic model for appeal data transfer."""

    id: UUID = Field(default_factory=uuid4)
    performance_record_id: UUID = Field(..., description="Related performance record")
    user_id: str = Field(..., description="User submitting appeal")
    tenant_id: Optional[str] = None

    appeal_type: str = Field(..., description="Type of appeal")
    reason: str = Field(..., min_length=10, description="Appeal reason")
    disputed_fields: List[str] = Field(default_factory=list)
    supporting_evidence: Dict[str, Any] = Field(default_factory=dict)

    status: AppealStatus = Field(default=AppealStatus.SUBMITTED)
    reviewer_id: Optional[str] = None
    review_notes: Optional[str] = None
    resolution: Optional[str] = None

    original_score: Optional[float] = None
    adjusted_score: Optional[float] = None
    adjustment_reason: Optional[str] = None

    submitted_at: datetime = Field(default_factory=datetime.now)
    reviewed_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert appeal to dictionary."""
        return {
            "id": str(self.id),
            "performance_record_id": str(self.performance_record_id),
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
            "appeal_type": self.appeal_type,
            "reason": self.reason,
            "disputed_fields": self.disputed_fields,
            "supporting_evidence": self.supporting_evidence,
            "status": self.status.value,
            "reviewer_id": self.reviewer_id,
            "review_notes": self.review_notes,
            "resolution": self.resolution,
            "original_score": self.original_score,
            "adjusted_score": self.adjusted_score,
            "adjustment_reason": self.adjustment_reason,
            "submitted_at": self.submitted_at.isoformat(),
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
        }

    model_config = ConfigDict(
        json_encoders={
            UUID: str,
            datetime: lambda v: v.isoformat(),
        }
    )


# ==================== Performance Weights Configuration ====================

class PerformanceWeights:
    """Configuration for performance score calculation weights."""

    # Dimension weights (must sum to 1.0)
    DIMENSION_WEIGHTS = {
        "quality": 0.40,       # 质量权重 40%
        "efficiency": 0.30,    # 效率权重 30%
        "compliance": 0.20,    # 合规权重 20%
        "improvement": 0.10,   # 改进权重 10%
    }

    # Sub-metric weights within each dimension
    QUALITY_WEIGHTS = {
        "accuracy_rate": 0.40,
        "consistency_score": 0.30,
        "error_rate": 0.30,  # Inverted (lower is better)
    }

    EFFICIENCY_WEIGHTS = {
        "completion_rate": 0.40,
        "resolution_time": 0.30,  # Inverted (lower is better)
        "throughput": 0.30,
    }

    COMPLIANCE_WEIGHTS = {
        "sla_compliance": 0.50,
        "attendance": 0.30,
        "violations": 0.20,  # Inverted (lower is better)
    }

    IMPROVEMENT_WEIGHTS = {
        "improvement_rate": 0.40,
        "training": 0.30,
        "feedback": 0.30,
    }

    # Thresholds for performance levels
    LEVEL_THRESHOLDS = {
        "excellent": 0.90,  # 优秀
        "good": 0.80,       # 良好
        "average": 0.60,    # 合格
        "poor": 0.40,       # 需改进
        # Below 0.40 is "unacceptable"
    }

    @classmethod
    def get_performance_level(cls, score: float) -> str:
        """Get performance level based on score."""
        if score >= cls.LEVEL_THRESHOLDS["excellent"]:
            return "excellent"
        elif score >= cls.LEVEL_THRESHOLDS["good"]:
            return "good"
        elif score >= cls.LEVEL_THRESHOLDS["average"]:
            return "average"
        elif score >= cls.LEVEL_THRESHOLDS["poor"]:
            return "poor"
        else:
            return "unacceptable"
