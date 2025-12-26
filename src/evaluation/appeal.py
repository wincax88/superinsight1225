"""
Appeal management for performance evaluation disputes.

Provides:
- Appeal submission
- Review workflow
- Score adjustments
- Statistics tracking
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from uuid import UUID

from sqlalchemy import select, and_, func
from sqlalchemy.orm import Session

from src.database.connection import db_manager
from src.evaluation.models import (
    PerformanceRecordModel,
    AppealModel,
    Appeal,
    AppealStatus,
    PerformanceStatus,
)

logger = logging.getLogger(__name__)


class AppealManager:
    """
    Appeal management for performance evaluation disputes.

    Handles appeal submission, review, and resolution workflow.
    """

    # Valid appeal types
    APPEAL_TYPES = [
        "score_dispute",        # 分数争议
        "data_error",           # 数据错误
        "calculation_error",    # 计算错误
        "missing_data",         # 数据缺失
        "unfair_evaluation",    # 不公平评估
        "system_issue",         # 系统问题
        "other",                # 其他
    ]

    # Review SLA (in hours)
    REVIEW_SLA = {
        "score_dispute": 48,       # 48 hours
        "data_error": 24,          # 24 hours
        "calculation_error": 24,   # 24 hours
        "missing_data": 48,        # 48 hours
        "unfair_evaluation": 72,   # 72 hours
        "system_issue": 24,        # 24 hours
        "other": 72,               # 72 hours
    }

    def __init__(self):
        """Initialize the appeal manager."""
        pass

    async def submit_appeal(
        self,
        performance_record_id: UUID,
        user_id: str,
        appeal_type: str,
        reason: str,
        disputed_fields: Optional[List[str]] = None,
        supporting_evidence: Optional[Dict[str, Any]] = None,
        tenant_id: Optional[str] = None
    ) -> Appeal:
        """
        Submit a performance evaluation appeal.

        Args:
            performance_record_id: ID of the disputed performance record
            user_id: User submitting the appeal
            appeal_type: Type of appeal
            reason: Detailed reason for appeal
            disputed_fields: List of disputed metric fields
            supporting_evidence: Evidence supporting the appeal
            tenant_id: Optional tenant identifier

        Returns:
            Created Appeal object
        """
        if appeal_type not in self.APPEAL_TYPES:
            raise ValueError(f"Invalid appeal type: {appeal_type}")

        try:
            with db_manager.get_session() as session:
                # Verify performance record exists
                record = session.execute(
                    select(PerformanceRecordModel).where(
                        PerformanceRecordModel.id == performance_record_id
                    )
                ).scalar_one_or_none()

                if not record:
                    raise ValueError("Performance record not found")

                # Check if user is allowed to appeal
                if record.user_id != user_id:
                    raise ValueError("Only the evaluated user can submit an appeal")

                # Check if record is in appealable state
                if record.status == PerformanceStatus.FINALIZED:
                    raise ValueError("Cannot appeal finalized performance records")

                # Check for existing pending appeal
                existing = session.execute(
                    select(AppealModel).where(
                        and_(
                            AppealModel.performance_record_id == performance_record_id,
                            AppealModel.status.in_([AppealStatus.SUBMITTED, AppealStatus.UNDER_REVIEW])
                        )
                    )
                ).scalar_one_or_none()

                if existing:
                    raise ValueError("An appeal is already pending for this record")

                # Create appeal
                appeal = Appeal(
                    performance_record_id=performance_record_id,
                    user_id=user_id,
                    tenant_id=tenant_id,
                    appeal_type=appeal_type,
                    reason=reason,
                    disputed_fields=disputed_fields or [],
                    supporting_evidence=supporting_evidence or {},
                    original_score=record.overall_score,
                )

                # Save to database
                appeal_model = AppealModel(
                    id=appeal.id,
                    performance_record_id=appeal.performance_record_id,
                    user_id=appeal.user_id,
                    tenant_id=appeal.tenant_id,
                    appeal_type=appeal.appeal_type,
                    reason=appeal.reason,
                    disputed_fields=appeal.disputed_fields,
                    supporting_evidence=appeal.supporting_evidence,
                    original_score=appeal.original_score,
                    status=AppealStatus.SUBMITTED,
                )
                session.add(appeal_model)

                # Update performance record status
                record.status = PerformanceStatus.DISPUTED
                record.updated_at = datetime.now()

                session.commit()

                logger.info(f"Appeal submitted: {appeal.id} for record {performance_record_id}")
                return appeal

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Error submitting appeal: {e}")
            raise

    async def review_appeal(
        self,
        appeal_id: UUID,
        reviewer_id: str,
        decision: str,
        review_notes: Optional[str] = None,
        adjusted_score: Optional[float] = None,
        adjustment_reason: Optional[str] = None
    ) -> bool:
        """
        Review an appeal and make a decision.

        Args:
            appeal_id: Appeal UUID
            reviewer_id: User reviewing the appeal
            decision: "approved" or "rejected"
            review_notes: Optional review notes
            adjusted_score: New score if approved
            adjustment_reason: Reason for score adjustment

        Returns:
            True if review successful
        """
        if decision not in ["approved", "rejected"]:
            raise ValueError("Decision must be 'approved' or 'rejected'")

        try:
            with db_manager.get_session() as session:
                appeal = session.execute(
                    select(AppealModel).where(AppealModel.id == appeal_id)
                ).scalar_one_or_none()

                if not appeal:
                    raise ValueError("Appeal not found")

                if appeal.status not in [AppealStatus.SUBMITTED, AppealStatus.UNDER_REVIEW]:
                    raise ValueError("Appeal is not in reviewable state")

                now = datetime.now()

                # Update appeal
                appeal.status = AppealStatus.APPROVED if decision == "approved" else AppealStatus.REJECTED
                appeal.reviewer_id = reviewer_id
                appeal.review_notes = review_notes
                appeal.reviewed_at = now
                appeal.resolved_at = now

                if decision == "approved" and adjusted_score is not None:
                    appeal.adjusted_score = adjusted_score
                    appeal.adjustment_reason = adjustment_reason

                    # Update performance record
                    record = session.execute(
                        select(PerformanceRecordModel).where(
                            PerformanceRecordModel.id == appeal.performance_record_id
                        )
                    ).scalar_one_or_none()

                    if record:
                        record.overall_score = adjusted_score
                        record.status = PerformanceStatus.APPROVED
                        record.reviewed_by = reviewer_id
                        record.reviewed_at = now
                        record.notes = f"Score adjusted via appeal. Original: {appeal.original_score}, New: {adjusted_score}"
                else:
                    # Rejected - restore record status
                    record = session.execute(
                        select(PerformanceRecordModel).where(
                            PerformanceRecordModel.id == appeal.performance_record_id
                        )
                    ).scalar_one_or_none()

                    if record:
                        record.status = PerformanceStatus.APPROVED
                        record.reviewed_by = reviewer_id
                        record.reviewed_at = now

                session.commit()

                logger.info(f"Appeal {appeal_id} {decision} by {reviewer_id}")
                return True

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Error reviewing appeal: {e}")
            return False

    async def start_review(
        self,
        appeal_id: UUID,
        reviewer_id: str
    ) -> bool:
        """
        Start reviewing an appeal (mark as under review).

        Args:
            appeal_id: Appeal UUID
            reviewer_id: User starting the review

        Returns:
            True if successful
        """
        try:
            with db_manager.get_session() as session:
                appeal = session.execute(
                    select(AppealModel).where(AppealModel.id == appeal_id)
                ).scalar_one_or_none()

                if not appeal:
                    return False

                if appeal.status != AppealStatus.SUBMITTED:
                    return False

                appeal.status = AppealStatus.UNDER_REVIEW
                appeal.reviewer_id = reviewer_id

                session.commit()
                return True

        except Exception as e:
            logger.error(f"Error starting review: {e}")
            return False

    async def withdraw_appeal(
        self,
        appeal_id: UUID,
        user_id: str
    ) -> bool:
        """
        Withdraw an appeal.

        Args:
            appeal_id: Appeal UUID
            user_id: User withdrawing (must be appeal owner)

        Returns:
            True if withdrawn successfully
        """
        try:
            with db_manager.get_session() as session:
                appeal = session.execute(
                    select(AppealModel).where(AppealModel.id == appeal_id)
                ).scalar_one_or_none()

                if not appeal:
                    return False

                if appeal.user_id != user_id:
                    raise ValueError("Only the appeal owner can withdraw")

                if appeal.status not in [AppealStatus.SUBMITTED, AppealStatus.UNDER_REVIEW]:
                    raise ValueError("Cannot withdraw an appeal that is already resolved")

                appeal.status = AppealStatus.WITHDRAWN
                appeal.resolved_at = datetime.now()

                # Restore performance record status
                record = session.execute(
                    select(PerformanceRecordModel).where(
                        PerformanceRecordModel.id == appeal.performance_record_id
                    )
                ).scalar_one_or_none()

                if record:
                    record.status = PerformanceStatus.PENDING_REVIEW

                session.commit()
                return True

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Error withdrawing appeal: {e}")
            return False

    async def get_appeal(self, appeal_id: UUID) -> Optional[Appeal]:
        """
        Get an appeal by ID.

        Args:
            appeal_id: Appeal UUID

        Returns:
            Appeal if found
        """
        try:
            with db_manager.get_session() as session:
                appeal_model = session.execute(
                    select(AppealModel).where(AppealModel.id == appeal_id)
                ).scalar_one_or_none()

                if appeal_model:
                    return self._to_pydantic(appeal_model)
                return None

        except Exception as e:
            logger.error(f"Error getting appeal: {e}")
            return None

    async def list_appeals(
        self,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        status: Optional[AppealStatus] = None,
        appeal_type: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> Tuple[List[Appeal], int]:
        """
        List appeals with filters.

        Args:
            user_id: Filter by user
            tenant_id: Filter by tenant
            status: Filter by status
            appeal_type: Filter by type
            limit: Max results
            offset: Pagination offset

        Returns:
            Tuple of (appeals, total_count)
        """
        try:
            with db_manager.get_session() as session:
                query = select(AppealModel)
                count_query = select(func.count(AppealModel.id))

                filters = []
                if user_id:
                    filters.append(AppealModel.user_id == user_id)
                if tenant_id:
                    filters.append(AppealModel.tenant_id == tenant_id)
                if status:
                    filters.append(AppealModel.status == status)
                if appeal_type:
                    filters.append(AppealModel.appeal_type == appeal_type)

                if filters:
                    query = query.where(and_(*filters))
                    count_query = count_query.where(and_(*filters))

                total = session.execute(count_query).scalar()

                query = query.order_by(AppealModel.submitted_at.desc()).limit(limit).offset(offset)
                appeals = session.execute(query).scalars().all()

                return [self._to_pydantic(a) for a in appeals], total

        except Exception as e:
            logger.error(f"Error listing appeals: {e}")
            return [], 0

    async def get_pending_appeals(
        self,
        reviewer_id: Optional[str] = None,
        tenant_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get pending appeals for review.

        Args:
            reviewer_id: Optional filter by assigned reviewer
            tenant_id: Optional tenant filter

        Returns:
            List of pending appeals with details
        """
        try:
            with db_manager.get_session() as session:
                query = select(AppealModel).where(
                    AppealModel.status.in_([AppealStatus.SUBMITTED, AppealStatus.UNDER_REVIEW])
                )

                if reviewer_id:
                    query = query.where(AppealModel.reviewer_id == reviewer_id)
                if tenant_id:
                    query = query.where(AppealModel.tenant_id == tenant_id)

                query = query.order_by(AppealModel.submitted_at.asc())
                appeals = session.execute(query).scalars().all()

                result = []
                for appeal in appeals:
                    # Calculate time since submission
                    age_hours = (datetime.now() - appeal.submitted_at).total_seconds() / 3600
                    sla_hours = self.REVIEW_SLA.get(appeal.appeal_type, 72)
                    overdue = age_hours > sla_hours

                    result.append({
                        "id": str(appeal.id),
                        "user_id": appeal.user_id,
                        "appeal_type": appeal.appeal_type,
                        "status": appeal.status.value,
                        "submitted_at": appeal.submitted_at.isoformat(),
                        "age_hours": round(age_hours, 1),
                        "sla_hours": sla_hours,
                        "overdue": overdue,
                        "reviewer_id": appeal.reviewer_id,
                    })

                return result

        except Exception as e:
            logger.error(f"Error getting pending appeals: {e}")
            return []

    async def get_appeal_statistics(
        self,
        tenant_id: Optional[str] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get appeal statistics.

        Args:
            tenant_id: Optional tenant filter
            days: Analysis period in days

        Returns:
            Statistics dictionary
        """
        try:
            with db_manager.get_session() as session:
                cutoff = datetime.now() - datetime.timedelta(days=days) if hasattr(datetime, 'timedelta') else datetime.now()

                base_filter = []
                if tenant_id:
                    base_filter.append(AppealModel.tenant_id == tenant_id)

                # Total appeals
                total_query = select(func.count(AppealModel.id))
                if base_filter:
                    total_query = total_query.where(and_(*base_filter))
                total = session.execute(total_query).scalar() or 0

                # By status
                by_status = {}
                for status in AppealStatus:
                    count = session.execute(
                        select(func.count(AppealModel.id)).where(
                            and_(*base_filter, AppealModel.status == status) if base_filter
                            else AppealModel.status == status
                        )
                    ).scalar() or 0
                    by_status[status.value] = count

                # By type
                by_type = {}
                for appeal_type in self.APPEAL_TYPES:
                    count = session.execute(
                        select(func.count(AppealModel.id)).where(
                            and_(*base_filter, AppealModel.appeal_type == appeal_type) if base_filter
                            else AppealModel.appeal_type == appeal_type
                        )
                    ).scalar() or 0
                    by_type[appeal_type] = count

                # Approval rate
                approved = by_status.get("approved", 0)
                resolved = approved + by_status.get("rejected", 0)
                approval_rate = approved / resolved if resolved > 0 else 0

                return {
                    "period_days": days,
                    "total_appeals": total,
                    "by_status": by_status,
                    "by_type": by_type,
                    "pending_count": by_status.get("submitted", 0) + by_status.get("under_review", 0),
                    "resolved_count": resolved,
                    "approval_rate": approval_rate,
                    "generated_at": datetime.now().isoformat(),
                }

        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}

    def _to_pydantic(self, model: AppealModel) -> Appeal:
        """Convert SQLAlchemy model to Pydantic model."""
        return Appeal(
            id=model.id,
            performance_record_id=model.performance_record_id,
            user_id=model.user_id,
            tenant_id=model.tenant_id,
            appeal_type=model.appeal_type,
            reason=model.reason,
            disputed_fields=model.disputed_fields,
            supporting_evidence=model.supporting_evidence,
            status=model.status,
            reviewer_id=model.reviewer_id,
            review_notes=model.review_notes,
            resolution=model.resolution,
            original_score=model.original_score,
            adjusted_score=model.adjusted_score,
            adjustment_reason=model.adjustment_reason,
            submitted_at=model.submitted_at,
            reviewed_at=model.reviewed_at,
            resolved_at=model.resolved_at,
        )
