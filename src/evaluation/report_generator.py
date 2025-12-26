"""
Performance report generator for SuperInsight platform.

Provides:
- Individual performance reports
- Team performance reports
- Trend analysis
- Improvement recommendations
"""

import logging
from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Optional
from uuid import UUID

from sqlalchemy import select, and_, func
from sqlalchemy.orm import Session

from src.database.connection import db_manager
from src.evaluation.models import (
    PerformanceRecordModel,
    PerformanceHistoryModel,
    PerformanceRecord,
    PerformanceWeights,
    PerformancePeriod,
)

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Performance report generator.

    Generates comprehensive performance reports with:
    - Individual assessments
    - Team comparisons
    - Trend analysis
    - Improvement recommendations
    """

    def __init__(self):
        """Initialize the report generator."""
        self.weights = PerformanceWeights()

    async def generate_individual_report(
        self,
        user_id: str,
        period_start: date,
        period_end: date,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate individual performance report.

        Args:
            user_id: User identifier
            period_start: Report period start
            period_end: Report period end
            tenant_id: Optional tenant filter

        Returns:
            Comprehensive individual report
        """
        try:
            with db_manager.get_session() as session:
                # Get current period record
                record = session.execute(
                    select(PerformanceRecordModel).where(
                        and_(
                            PerformanceRecordModel.user_id == user_id,
                            PerformanceRecordModel.period_start >= period_start,
                            PerformanceRecordModel.period_end <= period_end
                        )
                    ).order_by(PerformanceRecordModel.created_at.desc())
                ).scalar_one_or_none()

                if not record:
                    return {"error": "No performance data found for this period"}

                # Get historical records for comparison
                history = session.execute(
                    select(PerformanceRecordModel).where(
                        PerformanceRecordModel.user_id == user_id
                    ).order_by(PerformanceRecordModel.period_end.desc()).limit(6)
                ).scalars().all()

                # Calculate trends
                trends = self._calculate_trends(list(reversed(history)))

                # Get ranking info
                ranking = await self._get_user_ranking(session, record, tenant_id)

                # Generate recommendations
                recommendations = self._generate_recommendations(record)

                # Build report
                report = {
                    "report_type": "individual",
                    "user_id": user_id,
                    "period": {
                        "start": period_start.isoformat(),
                        "end": period_end.isoformat(),
                    },
                    "summary": {
                        "overall_score": record.overall_score,
                        "level": self.weights.get_performance_level(record.overall_score),
                        "rank": ranking.get("rank"),
                        "percentile": ranking.get("percentile"),
                        "total_users": ranking.get("total_users"),
                    },
                    "dimensions": {
                        "quality": {
                            "score": record.quality_score,
                            "accuracy_rate": record.accuracy_rate,
                            "consistency_score": record.consistency_score,
                            "error_rate": record.error_rate,
                        },
                        "efficiency": {
                            "completion_rate": record.completion_rate,
                            "avg_resolution_time": record.avg_resolution_time,
                            "tasks_completed": record.tasks_completed,
                            "tasks_assigned": record.tasks_assigned,
                        },
                        "compliance": {
                            "sla_compliance_rate": record.sla_compliance_rate,
                            "attendance_rate": record.attendance_rate,
                            "rule_violations": record.rule_violations,
                        },
                        "improvement": {
                            "improvement_rate": record.improvement_rate,
                            "training_completion": record.training_completion,
                            "feedback_score": record.feedback_score,
                        },
                    },
                    "trends": trends,
                    "recommendations": recommendations,
                    "generated_at": datetime.now().isoformat(),
                }

                return report

        except Exception as e:
            logger.error(f"Error generating individual report: {e}")
            return {"error": str(e)}

    async def generate_team_report(
        self,
        tenant_id: str,
        period_start: date,
        period_end: date
    ) -> Dict[str, Any]:
        """
        Generate team performance report.

        Args:
            tenant_id: Tenant/team identifier
            period_start: Report period start
            period_end: Report period end

        Returns:
            Team performance report
        """
        try:
            with db_manager.get_session() as session:
                # Get all records for the period
                records = session.execute(
                    select(PerformanceRecordModel).where(
                        and_(
                            PerformanceRecordModel.tenant_id == tenant_id,
                            PerformanceRecordModel.period_start >= period_start,
                            PerformanceRecordModel.period_end <= period_end
                        )
                    )
                ).scalars().all()

                if not records:
                    return {"error": "No performance data found for this period"}

                # Calculate team averages
                team_avg = self._calculate_team_averages(records)

                # Get top performers
                top_performers = sorted(records, key=lambda r: r.overall_score, reverse=True)[:5]

                # Get users needing improvement
                needs_improvement = [r for r in records if r.overall_score < 0.6]

                # Calculate distribution
                distribution = self._calculate_score_distribution(records)

                report = {
                    "report_type": "team",
                    "tenant_id": tenant_id,
                    "period": {
                        "start": period_start.isoformat(),
                        "end": period_end.isoformat(),
                    },
                    "summary": {
                        "total_members": len(records),
                        "avg_score": team_avg["overall"],
                        "avg_level": self.weights.get_performance_level(team_avg["overall"]),
                    },
                    "averages": team_avg,
                    "top_performers": [
                        {
                            "user_id": r.user_id,
                            "overall_score": r.overall_score,
                            "level": self.weights.get_performance_level(r.overall_score),
                        }
                        for r in top_performers
                    ],
                    "needs_improvement": [
                        {
                            "user_id": r.user_id,
                            "overall_score": r.overall_score,
                            "weak_areas": self._identify_weak_areas(r),
                        }
                        for r in needs_improvement
                    ],
                    "distribution": distribution,
                    "generated_at": datetime.now().isoformat(),
                }

                return report

        except Exception as e:
            logger.error(f"Error generating team report: {e}")
            return {"error": str(e)}

    async def generate_comparison_report(
        self,
        user_ids: List[str],
        period_start: date,
        period_end: date,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate comparison report for multiple users.

        Args:
            user_ids: List of user identifiers to compare
            period_start: Report period start
            period_end: Report period end
            tenant_id: Optional tenant filter

        Returns:
            Comparison report
        """
        try:
            with db_manager.get_session() as session:
                records = session.execute(
                    select(PerformanceRecordModel).where(
                        and_(
                            PerformanceRecordModel.user_id.in_(user_ids),
                            PerformanceRecordModel.period_start >= period_start,
                            PerformanceRecordModel.period_end <= period_end
                        )
                    )
                ).scalars().all()

                if not records:
                    return {"error": "No performance data found"}

                # Build comparison data
                comparisons = []
                for record in records:
                    comparisons.append({
                        "user_id": record.user_id,
                        "overall_score": record.overall_score,
                        "quality_score": record.quality_score,
                        "completion_rate": record.completion_rate,
                        "sla_compliance_rate": record.sla_compliance_rate,
                        "improvement_rate": record.improvement_rate,
                        "level": self.weights.get_performance_level(record.overall_score),
                    })

                # Sort by overall score
                comparisons.sort(key=lambda x: x["overall_score"], reverse=True)

                # Add rankings
                for i, comp in enumerate(comparisons, 1):
                    comp["rank"] = i

                return {
                    "report_type": "comparison",
                    "period": {
                        "start": period_start.isoformat(),
                        "end": period_end.isoformat(),
                    },
                    "users_compared": len(comparisons),
                    "comparisons": comparisons,
                    "generated_at": datetime.now().isoformat(),
                }

        except Exception as e:
            logger.error(f"Error generating comparison report: {e}")
            return {"error": str(e)}

    async def generate_trend_report(
        self,
        user_id: str,
        periods: int = 12,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate trend analysis report.

        Args:
            user_id: User identifier
            periods: Number of periods to analyze
            tenant_id: Optional tenant filter

        Returns:
            Trend analysis report
        """
        try:
            with db_manager.get_session() as session:
                records = session.execute(
                    select(PerformanceRecordModel).where(
                        PerformanceRecordModel.user_id == user_id
                    ).order_by(PerformanceRecordModel.period_end.desc()).limit(periods)
                ).scalars().all()

                if len(records) < 2:
                    return {"error": "Insufficient data for trend analysis"}

                records = list(reversed(records))

                # Calculate trends for each metric
                overall_trend = self._calculate_metric_trend([r.overall_score for r in records])
                quality_trend = self._calculate_metric_trend([r.quality_score for r in records])
                efficiency_trend = self._calculate_metric_trend([r.completion_rate for r in records])
                compliance_trend = self._calculate_metric_trend([r.sla_compliance_rate for r in records])

                # Build timeline data
                timeline = []
                for record in records:
                    timeline.append({
                        "period_end": record.period_end.isoformat(),
                        "overall_score": record.overall_score,
                        "quality_score": record.quality_score,
                        "completion_rate": record.completion_rate,
                        "sla_compliance_rate": record.sla_compliance_rate,
                    })

                return {
                    "report_type": "trend",
                    "user_id": user_id,
                    "periods_analyzed": len(records),
                    "trends": {
                        "overall": overall_trend,
                        "quality": quality_trend,
                        "efficiency": efficiency_trend,
                        "compliance": compliance_trend,
                    },
                    "timeline": timeline,
                    "insights": self._generate_trend_insights(overall_trend, quality_trend),
                    "generated_at": datetime.now().isoformat(),
                }

        except Exception as e:
            logger.error(f"Error generating trend report: {e}")
            return {"error": str(e)}

    def _calculate_trends(self, records: List[PerformanceRecordModel]) -> Dict[str, Any]:
        """Calculate trend data from historical records."""
        if len(records) < 2:
            return {"direction": "stable", "change": 0}

        scores = [r.overall_score for r in records]
        first = scores[0]
        last = scores[-1]

        change = (last - first) / first if first > 0 else 0

        if change > 0.05:
            direction = "improving"
        elif change < -0.05:
            direction = "declining"
        else:
            direction = "stable"

        return {
            "direction": direction,
            "change_rate": change,
            "history": [
                {"period": r.period_end.isoformat(), "score": r.overall_score}
                for r in records
            ],
        }

    async def _get_user_ranking(
        self,
        session: Session,
        record: PerformanceRecordModel,
        tenant_id: Optional[str]
    ) -> Dict[str, Any]:
        """Get user's ranking among peers."""
        try:
            query = select(PerformanceRecordModel).where(
                and_(
                    PerformanceRecordModel.period_start == record.period_start,
                    PerformanceRecordModel.period_end == record.period_end
                )
            )

            if tenant_id:
                query = query.where(PerformanceRecordModel.tenant_id == tenant_id)

            all_records = session.execute(query).scalars().all()

            if not all_records:
                return {}

            # Sort by score
            sorted_records = sorted(all_records, key=lambda r: r.overall_score, reverse=True)

            # Find user's rank
            rank = None
            for i, r in enumerate(sorted_records, 1):
                if r.user_id == record.user_id:
                    rank = i
                    break

            total = len(sorted_records)
            percentile = ((total - rank + 1) / total) * 100 if rank else None

            return {
                "rank": rank,
                "total_users": total,
                "percentile": round(percentile, 1) if percentile else None,
            }

        except Exception:
            return {}

    def _generate_recommendations(
        self,
        record: PerformanceRecordModel
    ) -> List[Dict[str, Any]]:
        """Generate improvement recommendations based on performance."""
        recommendations = []

        # Quality recommendations
        if record.quality_score < 0.7:
            recommendations.append({
                "area": "quality",
                "priority": "high",
                "suggestion": "Focus on accuracy improvement through additional training",
                "target_improvement": 0.1,
            })

        if record.error_rate > 0.2:
            recommendations.append({
                "area": "quality",
                "priority": "high",
                "suggestion": "Reduce error rate by reviewing common mistakes",
                "target_improvement": -0.1,
            })

        # Efficiency recommendations
        if record.completion_rate < 0.8:
            recommendations.append({
                "area": "efficiency",
                "priority": "medium",
                "suggestion": "Improve task completion rate through better time management",
                "target_improvement": 0.1,
            })

        if record.avg_resolution_time > 28800:  # > 8 hours
            recommendations.append({
                "area": "efficiency",
                "priority": "medium",
                "suggestion": "Work on reducing resolution time",
                "target_improvement": -3600,  # 1 hour reduction
            })

        # Compliance recommendations
        if record.sla_compliance_rate < 0.9:
            recommendations.append({
                "area": "compliance",
                "priority": "high",
                "suggestion": "Improve SLA compliance by prioritizing urgent tickets",
                "target_improvement": 0.05,
            })

        # Improvement recommendations
        if record.training_completion < 0.8:
            recommendations.append({
                "area": "improvement",
                "priority": "low",
                "suggestion": "Complete pending training modules",
                "target_improvement": 0.2,
            })

        return recommendations

    def _calculate_team_averages(
        self,
        records: List[PerformanceRecordModel]
    ) -> Dict[str, float]:
        """Calculate team average metrics."""
        if not records:
            return {}

        n = len(records)
        return {
            "overall": sum(r.overall_score for r in records) / n,
            "quality": sum(r.quality_score for r in records) / n,
            "completion_rate": sum(r.completion_rate for r in records) / n,
            "sla_compliance": sum(r.sla_compliance_rate for r in records) / n,
            "improvement_rate": sum(r.improvement_rate for r in records) / n,
        }

    def _calculate_score_distribution(
        self,
        records: List[PerformanceRecordModel]
    ) -> Dict[str, int]:
        """Calculate score distribution across performance levels."""
        distribution = {
            "excellent": 0,
            "good": 0,
            "average": 0,
            "poor": 0,
            "unacceptable": 0,
        }

        for record in records:
            level = self.weights.get_performance_level(record.overall_score)
            distribution[level] += 1

        return distribution

    def _identify_weak_areas(
        self,
        record: PerformanceRecordModel
    ) -> List[str]:
        """Identify areas needing improvement."""
        weak_areas = []

        if record.quality_score < 0.6:
            weak_areas.append("quality")
        if record.completion_rate < 0.6:
            weak_areas.append("efficiency")
        if record.sla_compliance_rate < 0.8:
            weak_areas.append("compliance")
        if record.improvement_rate < 0:
            weak_areas.append("improvement")

        return weak_areas

    def _calculate_metric_trend(self, values: List[float]) -> Dict[str, Any]:
        """Calculate trend for a single metric."""
        if len(values) < 2:
            return {"direction": "stable", "slope": 0}

        # Simple linear regression
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n

        numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        slope = numerator / denominator if denominator != 0 else 0

        if slope > 0.01:
            direction = "improving"
        elif slope < -0.01:
            direction = "declining"
        else:
            direction = "stable"

        return {
            "direction": direction,
            "slope": round(slope, 4),
            "start_value": values[0],
            "end_value": values[-1],
            "change": values[-1] - values[0],
        }

    def _generate_trend_insights(
        self,
        overall_trend: Dict[str, Any],
        quality_trend: Dict[str, Any]
    ) -> List[str]:
        """Generate insights from trend analysis."""
        insights = []

        if overall_trend["direction"] == "improving":
            insights.append("Overall performance is on an upward trend")
        elif overall_trend["direction"] == "declining":
            insights.append("Overall performance has been declining - review and address issues")

        if quality_trend["direction"] == "improving":
            insights.append("Quality metrics are improving - training is effective")
        elif quality_trend["direction"] == "declining":
            insights.append("Quality needs attention - consider additional training")

        if overall_trend["direction"] == "stable" and quality_trend["direction"] == "stable":
            insights.append("Performance is stable - look for opportunities to improve")

        return insights
