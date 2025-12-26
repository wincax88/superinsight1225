"""
Performance calculation engine for SuperInsight platform.

Provides multi-dimensional performance evaluation with:
- Quality assessment
- Efficiency metrics
- Compliance tracking
- Improvement analysis
"""

import logging
from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Optional, Tuple
from uuid import UUID

from sqlalchemy import select, and_, func
from sqlalchemy.orm import Session

from src.database.connection import db_manager
from src.evaluation.models import (
    PerformanceRecordModel,
    PerformanceHistoryModel,
    PerformanceRecord,
    PerformanceStatus,
    PerformancePeriod,
    PerformanceWeights,
)
from src.ticket.models import TicketModel, TicketStatus, AnnotatorSkillModel

logger = logging.getLogger(__name__)


class PerformanceEngine:
    """
    Performance calculation engine.

    Calculates multi-dimensional performance scores based on:
    - Quality (40%): accuracy, consistency, error rate
    - Efficiency (30%): completion rate, resolution time, throughput
    - Compliance (20%): SLA adherence, attendance, rule violations
    - Improvement (10%): improvement rate, training, feedback
    """

    def __init__(self):
        """Initialize the performance engine."""
        self.weights = PerformanceWeights()

    async def calculate_performance(
        self,
        user_id: str,
        period_start: date,
        period_end: date,
        tenant_id: Optional[str] = None,
        period_type: PerformancePeriod = PerformancePeriod.MONTHLY
    ) -> PerformanceRecord:
        """
        Calculate comprehensive performance for a user over a period.

        Args:
            user_id: User identifier
            period_start: Start of evaluation period
            period_end: End of evaluation period
            tenant_id: Optional tenant filter
            period_type: Type of period (daily, weekly, monthly, quarterly)

        Returns:
            PerformanceRecord with calculated metrics
        """
        try:
            with db_manager.get_session() as session:
                # Collect raw metrics
                quality_metrics = await self._calculate_quality_metrics(
                    session, user_id, period_start, period_end, tenant_id
                )
                efficiency_metrics = await self._calculate_efficiency_metrics(
                    session, user_id, period_start, period_end, tenant_id
                )
                compliance_metrics = await self._calculate_compliance_metrics(
                    session, user_id, period_start, period_end, tenant_id
                )
                improvement_metrics = await self._calculate_improvement_metrics(
                    session, user_id, period_start, period_end, tenant_id
                )

                # Calculate dimension scores
                quality_score = self._calculate_dimension_score(
                    quality_metrics, self.weights.QUALITY_WEIGHTS
                )
                efficiency_score = self._calculate_dimension_score(
                    efficiency_metrics, self.weights.EFFICIENCY_WEIGHTS
                )
                compliance_score = self._calculate_dimension_score(
                    compliance_metrics, self.weights.COMPLIANCE_WEIGHTS
                )
                improvement_score = self._calculate_dimension_score(
                    improvement_metrics, self.weights.IMPROVEMENT_WEIGHTS
                )

                # Calculate overall score
                overall_score = (
                    quality_score * self.weights.DIMENSION_WEIGHTS["quality"] +
                    efficiency_score * self.weights.DIMENSION_WEIGHTS["efficiency"] +
                    compliance_score * self.weights.DIMENSION_WEIGHTS["compliance"] +
                    improvement_score * self.weights.DIMENSION_WEIGHTS["improvement"]
                )

                # Create performance record
                record = PerformanceRecord(
                    user_id=user_id,
                    tenant_id=tenant_id,
                    period_type=period_type,
                    period_start=period_start,
                    period_end=period_end,
                    # Quality metrics
                    quality_score=quality_score,
                    accuracy_rate=quality_metrics.get("accuracy_rate", 0.0),
                    consistency_score=quality_metrics.get("consistency_score", 0.0),
                    error_rate=quality_metrics.get("error_rate", 0.0),
                    # Efficiency metrics
                    completion_rate=efficiency_metrics.get("completion_rate", 0.0),
                    avg_resolution_time=int(efficiency_metrics.get("avg_resolution_time", 0)),
                    tasks_completed=int(efficiency_metrics.get("tasks_completed", 0)),
                    tasks_assigned=int(efficiency_metrics.get("tasks_assigned", 0)),
                    # Compliance metrics
                    sla_compliance_rate=compliance_metrics.get("sla_compliance", 0.0),
                    attendance_rate=compliance_metrics.get("attendance", 1.0),
                    rule_violations=int(compliance_metrics.get("violations", 0)),
                    # Improvement metrics
                    improvement_rate=improvement_metrics.get("improvement_rate", 0.0),
                    training_completion=improvement_metrics.get("training", 0.0),
                    feedback_score=improvement_metrics.get("feedback", 0.0),
                    # Overall
                    overall_score=overall_score,
                    status=PerformanceStatus.DRAFT,
                    calculation_details={
                        "quality_score": quality_score,
                        "efficiency_score": efficiency_score,
                        "compliance_score": compliance_score,
                        "improvement_score": improvement_score,
                        "weights": self.weights.DIMENSION_WEIGHTS,
                        "level": self.weights.get_performance_level(overall_score),
                    }
                )

                # Save to database
                record_model = PerformanceRecordModel(
                    id=record.id,
                    user_id=record.user_id,
                    tenant_id=record.tenant_id,
                    period_type=record.period_type,
                    period_start=record.period_start,
                    period_end=record.period_end,
                    quality_score=record.quality_score,
                    accuracy_rate=record.accuracy_rate,
                    consistency_score=record.consistency_score,
                    error_rate=record.error_rate,
                    completion_rate=record.completion_rate,
                    avg_resolution_time=record.avg_resolution_time,
                    tasks_completed=record.tasks_completed,
                    tasks_assigned=record.tasks_assigned,
                    sla_compliance_rate=record.sla_compliance_rate,
                    attendance_rate=record.attendance_rate,
                    rule_violations=record.rule_violations,
                    improvement_rate=record.improvement_rate,
                    training_completion=record.training_completion,
                    feedback_score=record.feedback_score,
                    overall_score=record.overall_score,
                    status=record.status,
                    calculation_details=record.calculation_details,
                )
                session.add(record_model)
                session.commit()

                logger.info(f"Calculated performance for {user_id}: {overall_score:.3f}")
                return record

        except Exception as e:
            logger.error(f"Error calculating performance for {user_id}: {e}")
            raise

    async def _calculate_quality_metrics(
        self,
        session: Session,
        user_id: str,
        period_start: date,
        period_end: date,
        tenant_id: Optional[str]
    ) -> Dict[str, float]:
        """Calculate quality dimension metrics."""
        try:
            # Get completed tickets for the user in period
            query = select(TicketModel).where(
                and_(
                    TicketModel.assigned_to == user_id,
                    TicketModel.status.in_([TicketStatus.RESOLVED, TicketStatus.CLOSED]),
                    TicketModel.resolved_at >= datetime.combine(period_start, datetime.min.time()),
                    TicketModel.resolved_at <= datetime.combine(period_end, datetime.max.time()),
                )
            )
            if tenant_id:
                query = query.where(TicketModel.tenant_id == tenant_id)

            tickets = session.execute(query).scalars().all()

            if not tickets:
                return {"accuracy_rate": 0.0, "consistency_score": 0.0, "error_rate": 0.0}

            # Get annotator skill record for quality metrics
            skill = session.execute(
                select(AnnotatorSkillModel).where(
                    AnnotatorSkillModel.user_id == user_id
                )
            ).scalar_one_or_none()

            accuracy_rate = skill.avg_quality_score if skill else 0.0
            success_rate = skill.success_rate if skill else 0.0

            # Calculate error rate from SLA breaches
            breached = sum(1 for t in tickets if t.sla_breached)
            error_rate = breached / len(tickets) if tickets else 0.0

            return {
                "accuracy_rate": min(1.0, accuracy_rate),
                "consistency_score": min(1.0, success_rate),
                "error_rate": min(1.0, error_rate),
            }

        except Exception as e:
            logger.error(f"Error calculating quality metrics: {e}")
            return {"accuracy_rate": 0.0, "consistency_score": 0.0, "error_rate": 0.0}

    async def _calculate_efficiency_metrics(
        self,
        session: Session,
        user_id: str,
        period_start: date,
        period_end: date,
        tenant_id: Optional[str]
    ) -> Dict[str, float]:
        """Calculate efficiency dimension metrics."""
        try:
            # Get all tickets assigned to user in period
            assigned_query = select(func.count(TicketModel.id)).where(
                and_(
                    TicketModel.assigned_to == user_id,
                    TicketModel.assigned_at >= datetime.combine(period_start, datetime.min.time()),
                    TicketModel.assigned_at <= datetime.combine(period_end, datetime.max.time()),
                )
            )
            if tenant_id:
                assigned_query = assigned_query.where(TicketModel.tenant_id == tenant_id)

            tasks_assigned = session.execute(assigned_query).scalar() or 0

            # Get completed tickets
            completed_query = select(TicketModel).where(
                and_(
                    TicketModel.assigned_to == user_id,
                    TicketModel.status.in_([TicketStatus.RESOLVED, TicketStatus.CLOSED]),
                    TicketModel.resolved_at >= datetime.combine(period_start, datetime.min.time()),
                    TicketModel.resolved_at <= datetime.combine(period_end, datetime.max.time()),
                )
            )
            if tenant_id:
                completed_query = completed_query.where(TicketModel.tenant_id == tenant_id)

            completed_tickets = session.execute(completed_query).scalars().all()
            tasks_completed = len(completed_tickets)

            # Calculate completion rate
            completion_rate = tasks_completed / tasks_assigned if tasks_assigned > 0 else 0.0

            # Calculate average resolution time
            resolution_times = []
            for t in completed_tickets:
                if t.resolved_at and t.created_at:
                    resolution_times.append((t.resolved_at - t.created_at).total_seconds())

            avg_resolution_time = sum(resolution_times) / len(resolution_times) if resolution_times else 0

            # Normalize resolution time (assume 8 hours = 28800s is baseline for 0.5 score)
            baseline_time = 28800  # 8 hours
            resolution_score = 1.0 - min(1.0, avg_resolution_time / (baseline_time * 2))

            # Calculate throughput score (tasks per day)
            days = (period_end - period_start).days + 1
            throughput = tasks_completed / days if days > 0 else 0
            throughput_score = min(1.0, throughput / 5)  # 5 tasks/day = 1.0

            return {
                "completion_rate": min(1.0, completion_rate),
                "avg_resolution_time": avg_resolution_time,
                "resolution_time": max(0.0, resolution_score),
                "tasks_completed": tasks_completed,
                "tasks_assigned": tasks_assigned,
                "throughput": throughput_score,
            }

        except Exception as e:
            logger.error(f"Error calculating efficiency metrics: {e}")
            return {"completion_rate": 0.0, "resolution_time": 0.0, "throughput": 0.0}

    async def _calculate_compliance_metrics(
        self,
        session: Session,
        user_id: str,
        period_start: date,
        period_end: date,
        tenant_id: Optional[str]
    ) -> Dict[str, float]:
        """Calculate compliance dimension metrics."""
        try:
            # Get all resolved tickets
            query = select(TicketModel).where(
                and_(
                    TicketModel.assigned_to == user_id,
                    TicketModel.status.in_([TicketStatus.RESOLVED, TicketStatus.CLOSED]),
                    TicketModel.resolved_at >= datetime.combine(period_start, datetime.min.time()),
                    TicketModel.resolved_at <= datetime.combine(period_end, datetime.max.time()),
                )
            )
            if tenant_id:
                query = query.where(TicketModel.tenant_id == tenant_id)

            tickets = session.execute(query).scalars().all()

            if not tickets:
                return {"sla_compliance": 1.0, "attendance": 1.0, "violations": 0}

            # Calculate SLA compliance
            compliant = sum(1 for t in tickets if not t.sla_breached)
            sla_compliance = compliant / len(tickets)

            # Attendance - assume 1.0 for now (would integrate with HR system)
            attendance = 1.0

            # Rule violations - count SLA breaches as violations
            violations = sum(1 for t in tickets if t.sla_breached)
            violation_score = 1.0 - min(1.0, violations / 10)  # 10+ violations = 0

            return {
                "sla_compliance": sla_compliance,
                "attendance": attendance,
                "violations": violations,
                "violation_score": violation_score,
            }

        except Exception as e:
            logger.error(f"Error calculating compliance metrics: {e}")
            return {"sla_compliance": 0.0, "attendance": 1.0, "violations": 0}

    async def _calculate_improvement_metrics(
        self,
        session: Session,
        user_id: str,
        period_start: date,
        period_end: date,
        tenant_id: Optional[str]
    ) -> Dict[str, float]:
        """Calculate improvement dimension metrics."""
        try:
            # Get previous period performance
            prev_period_end = period_start - timedelta(days=1)
            prev_period_start = prev_period_end - (period_end - period_start)

            prev_record = session.execute(
                select(PerformanceRecordModel).where(
                    and_(
                        PerformanceRecordModel.user_id == user_id,
                        PerformanceRecordModel.period_end == prev_period_end
                    )
                )
            ).scalar_one_or_none()

            # Calculate improvement rate
            if prev_record:
                # Get current quality score
                skill = session.execute(
                    select(AnnotatorSkillModel).where(
                        AnnotatorSkillModel.user_id == user_id
                    )
                ).scalar_one_or_none()

                current_quality = skill.avg_quality_score if skill else 0.0
                prev_quality = prev_record.quality_score

                if prev_quality > 0:
                    improvement_rate = (current_quality - prev_quality) / prev_quality
                else:
                    improvement_rate = current_quality
            else:
                improvement_rate = 0.0

            # Training completion (placeholder - would integrate with LMS)
            training_completion = 0.8  # Default assumption

            # Feedback score (placeholder - would integrate with feedback system)
            feedback_score = 0.7  # Default assumption

            return {
                "improvement_rate": max(-1.0, min(1.0, improvement_rate)),
                "training": training_completion,
                "feedback": feedback_score,
            }

        except Exception as e:
            logger.error(f"Error calculating improvement metrics: {e}")
            return {"improvement_rate": 0.0, "training": 0.0, "feedback": 0.0}

    def _calculate_dimension_score(
        self,
        metrics: Dict[str, float],
        weights: Dict[str, float]
    ) -> float:
        """Calculate weighted score for a dimension."""
        score = 0.0
        total_weight = 0.0

        for metric_name, weight in weights.items():
            value = metrics.get(metric_name, 0.0)

            # Handle inverted metrics (where lower is better)
            if metric_name in ["error_rate", "resolution_time", "violations"]:
                value = 1.0 - min(1.0, value)

            score += value * weight
            total_weight += weight

        return score / total_weight if total_weight > 0 else 0.0

    async def compare_before_after(
        self,
        ticket_id: UUID
    ) -> Dict[str, Any]:
        """
        Compare quality before and after a repair/resolution.

        Args:
            ticket_id: Ticket UUID

        Returns:
            Comparison results
        """
        try:
            with db_manager.get_session() as session:
                ticket = session.execute(
                    select(TicketModel).where(TicketModel.id == ticket_id)
                ).scalar_one_or_none()

                if not ticket:
                    return {"error": "Ticket not found"}

                if not ticket.resolved_at:
                    return {"error": "Ticket not resolved"}

                # Calculate time to resolution
                resolution_time = (ticket.resolved_at - ticket.created_at).total_seconds()

                # Check SLA compliance
                sla_compliant = not ticket.sla_breached

                return {
                    "ticket_id": str(ticket_id),
                    "resolution_time_seconds": int(resolution_time),
                    "sla_compliant": sla_compliant,
                    "escalation_level": ticket.escalation_level,
                    "priority": ticket.priority.value,
                    "quality_improved": True,  # Placeholder
                }

        except Exception as e:
            logger.error(f"Error comparing before/after: {e}")
            return {"error": str(e)}

    async def detect_regression(
        self,
        user_id: str,
        periods: int = 3,
        threshold: float = 0.1
    ) -> List[Dict[str, Any]]:
        """
        Detect performance regression over recent periods.

        Args:
            user_id: User identifier
            periods: Number of periods to analyze
            threshold: Decline threshold to flag

        Returns:
            List of detected regressions
        """
        regressions = []

        try:
            with db_manager.get_session() as session:
                records = session.execute(
                    select(PerformanceRecordModel)
                    .where(PerformanceRecordModel.user_id == user_id)
                    .order_by(PerformanceRecordModel.period_end.desc())
                    .limit(periods)
                ).scalars().all()

                if len(records) < 2:
                    return []

                # Compare each period with previous
                for i in range(len(records) - 1):
                    current = records[i]
                    previous = records[i + 1]

                    if previous.overall_score > 0:
                        decline = (previous.overall_score - current.overall_score) / previous.overall_score

                        if decline > threshold:
                            regressions.append({
                                "period": f"{current.period_start} to {current.period_end}",
                                "current_score": current.overall_score,
                                "previous_score": previous.overall_score,
                                "decline_rate": decline,
                                "affected_dimensions": self._identify_declining_dimensions(current, previous),
                            })

        except Exception as e:
            logger.error(f"Error detecting regression: {e}")

        return regressions

    def _identify_declining_dimensions(
        self,
        current: PerformanceRecordModel,
        previous: PerformanceRecordModel
    ) -> List[str]:
        """Identify which dimensions have declined."""
        declining = []

        if current.quality_score < previous.quality_score * 0.9:
            declining.append("quality")
        if current.completion_rate < previous.completion_rate * 0.9:
            declining.append("efficiency")
        if current.sla_compliance_rate < previous.sla_compliance_rate * 0.9:
            declining.append("compliance")

        return declining

    async def get_performance_ranking(
        self,
        tenant_id: Optional[str] = None,
        period_start: Optional[date] = None,
        period_end: Optional[date] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get performance ranking for users.

        Args:
            tenant_id: Optional tenant filter
            period_start: Optional period start filter
            period_end: Optional period end filter
            limit: Maximum results

        Returns:
            Ranked list of users by performance
        """
        try:
            with db_manager.get_session() as session:
                query = select(PerformanceRecordModel)

                filters = []
                if tenant_id:
                    filters.append(PerformanceRecordModel.tenant_id == tenant_id)
                if period_start:
                    filters.append(PerformanceRecordModel.period_start >= period_start)
                if period_end:
                    filters.append(PerformanceRecordModel.period_end <= period_end)

                if filters:
                    query = query.where(and_(*filters))

                query = query.order_by(
                    PerformanceRecordModel.overall_score.desc()
                ).limit(limit)

                records = session.execute(query).scalars().all()

                ranking = []
                for i, record in enumerate(records, 1):
                    ranking.append({
                        "rank": i,
                        "user_id": record.user_id,
                        "overall_score": record.overall_score,
                        "quality_score": record.quality_score,
                        "completion_rate": record.completion_rate,
                        "sla_compliance_rate": record.sla_compliance_rate,
                        "level": self.weights.get_performance_level(record.overall_score),
                        "period": f"{record.period_start} to {record.period_end}",
                    })

                return ranking

        except Exception as e:
            logger.error(f"Error getting ranking: {e}")
            return []

    async def get_user_performance_history(
        self,
        user_id: str,
        periods: int = 12
    ) -> List[Dict[str, Any]]:
        """
        Get performance history for trend analysis.

        Args:
            user_id: User identifier
            periods: Number of periods to retrieve

        Returns:
            List of historical performance records
        """
        try:
            with db_manager.get_session() as session:
                records = session.execute(
                    select(PerformanceRecordModel)
                    .where(PerformanceRecordModel.user_id == user_id)
                    .order_by(PerformanceRecordModel.period_end.desc())
                    .limit(periods)
                ).scalars().all()

                return [
                    {
                        "period_start": r.period_start.isoformat(),
                        "period_end": r.period_end.isoformat(),
                        "overall_score": r.overall_score,
                        "quality_score": r.quality_score,
                        "completion_rate": r.completion_rate,
                        "sla_compliance_rate": r.sla_compliance_rate,
                        "level": self.weights.get_performance_level(r.overall_score),
                    }
                    for r in reversed(records)
                ]

        except Exception as e:
            logger.error(f"Error getting history: {e}")
            return []
