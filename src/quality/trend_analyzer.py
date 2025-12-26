"""
Quality trend analysis for SuperInsight Platform.

Provides:
- Quality metric trend analysis
- Decline detection
- Quality prediction
- Root cause analysis
"""

import logging
from datetime import datetime, timedelta, date
from typing import Dict, Any, List, Optional
from collections import defaultdict
import statistics

from sqlalchemy import select, and_, func
from sqlalchemy.orm import Session

from src.database.connection import db_manager
from src.database.models import QualityIssueModel, IssueSeverity, IssueStatus

logger = logging.getLogger(__name__)


class QualityTrendAnalyzer:
    """
    Quality trend analysis engine.

    Analyzes quality metrics over time to identify trends,
    detect quality declines, and predict future quality levels.
    """

    # Trend detection thresholds
    DECLINE_THRESHOLD = 0.1  # 10% decline triggers alert
    SIGNIFICANT_CHANGE_THRESHOLD = 0.05  # 5% change is significant

    # Prediction parameters
    DEFAULT_PREDICTION_DAYS = 7
    MIN_DATA_POINTS = 5  # Minimum data points for prediction

    def __init__(self):
        """Initialize the trend analyzer."""
        pass

    async def analyze_trends(
        self,
        tenant_id: Optional[str] = None,
        days: int = 30,
        granularity: str = "daily"
    ) -> Dict[str, Any]:
        """
        Analyze quality trends over a specified period.

        Args:
            tenant_id: Optional tenant filter
            days: Number of days to analyze
            granularity: Time granularity (daily, weekly)

        Returns:
            Trend analysis results
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            with db_manager.get_session() as session:
                # Get quality issues in the period
                query = select(QualityIssueModel).where(
                    QualityIssueModel.created_at >= start_date
                )

                if tenant_id:
                    # Note: tenant filtering would require joining with tasks
                    pass

                issues = session.execute(query).scalars().all()

                # Group by date
                daily_metrics = defaultdict(lambda: {
                    "total_issues": 0,
                    "critical": 0,
                    "high": 0,
                    "medium": 0,
                    "low": 0,
                    "resolved": 0,
                    "open": 0
                })

                for issue in issues:
                    day_key = issue.created_at.date().isoformat()
                    daily_metrics[day_key]["total_issues"] += 1

                    # By severity
                    if issue.severity == IssueSeverity.CRITICAL:
                        daily_metrics[day_key]["critical"] += 1
                    elif issue.severity == IssueSeverity.HIGH:
                        daily_metrics[day_key]["high"] += 1
                    elif issue.severity == IssueSeverity.MEDIUM:
                        daily_metrics[day_key]["medium"] += 1
                    else:
                        daily_metrics[day_key]["low"] += 1

                    # By status
                    if issue.status == IssueStatus.RESOLVED:
                        daily_metrics[day_key]["resolved"] += 1
                    else:
                        daily_metrics[day_key]["open"] += 1

                # Calculate trends
                sorted_dates = sorted(daily_metrics.keys())
                issue_counts = [daily_metrics[d]["total_issues"] for d in sorted_dates]

                trend_direction = self._calculate_trend_direction(issue_counts)
                trend_slope = self._calculate_slope(issue_counts)

                # Calculate averages
                avg_daily_issues = statistics.mean(issue_counts) if issue_counts else 0

                # Weekly aggregation if requested
                weekly_metrics = {}
                if granularity == "weekly":
                    weekly_metrics = self._aggregate_weekly(daily_metrics)

                return {
                    "period": {
                        "start": start_date.isoformat(),
                        "end": end_date.isoformat(),
                        "days": days
                    },
                    "summary": {
                        "total_issues": sum(issue_counts),
                        "avg_daily_issues": round(avg_daily_issues, 2),
                        "trend_direction": trend_direction,
                        "trend_slope": round(trend_slope, 4),
                        "data_points": len(sorted_dates)
                    },
                    "daily_metrics": dict(daily_metrics) if granularity == "daily" else None,
                    "weekly_metrics": weekly_metrics if weekly_metrics else None,
                    "severity_distribution": self._calculate_severity_distribution(issues),
                    "resolution_rate": self._calculate_resolution_rate(issues),
                    "generated_at": datetime.now().isoformat()
                }

        except Exception as e:
            logger.error(f"Error analyzing trends: {e}")
            return {"error": str(e)}

    async def detect_quality_decline(
        self,
        tenant_id: Optional[str] = None,
        threshold: float = None,
        window_days: int = 7
    ) -> List[Dict[str, Any]]:
        """
        Detect quality decline alerts.

        Args:
            tenant_id: Optional tenant filter
            threshold: Decline threshold (default 0.1 = 10%)
            window_days: Comparison window in days

        Returns:
            List of decline alerts
        """
        threshold = threshold or self.DECLINE_THRESHOLD
        alerts = []

        try:
            end_date = datetime.now()
            mid_date = end_date - timedelta(days=window_days)
            start_date = mid_date - timedelta(days=window_days)

            with db_manager.get_session() as session:
                # Get issues for both periods
                previous_query = select(func.count(QualityIssueModel.id)).where(
                    and_(
                        QualityIssueModel.created_at >= start_date,
                        QualityIssueModel.created_at < mid_date
                    )
                )

                current_query = select(func.count(QualityIssueModel.id)).where(
                    and_(
                        QualityIssueModel.created_at >= mid_date,
                        QualityIssueModel.created_at <= end_date
                    )
                )

                previous_count = session.execute(previous_query).scalar() or 0
                current_count = session.execute(current_query).scalar() or 0

                # Check for significant increase in issues (indicates quality decline)
                if previous_count > 0:
                    change_rate = (current_count - previous_count) / previous_count

                    if change_rate > threshold:
                        alerts.append({
                            "type": "issue_increase",
                            "severity": "high" if change_rate > 2 * threshold else "medium",
                            "message": f"Quality issues increased by {change_rate * 100:.1f}%",
                            "details": {
                                "previous_period_count": previous_count,
                                "current_period_count": current_count,
                                "change_rate": round(change_rate, 4),
                                "window_days": window_days
                            },
                            "detected_at": datetime.now().isoformat()
                        })

                # Check for critical issue spike
                critical_previous = session.execute(
                    select(func.count(QualityIssueModel.id)).where(
                        and_(
                            QualityIssueModel.created_at >= start_date,
                            QualityIssueModel.created_at < mid_date,
                            QualityIssueModel.severity == IssueSeverity.CRITICAL
                        )
                    )
                ).scalar() or 0

                critical_current = session.execute(
                    select(func.count(QualityIssueModel.id)).where(
                        and_(
                            QualityIssueModel.created_at >= mid_date,
                            QualityIssueModel.created_at <= end_date,
                            QualityIssueModel.severity == IssueSeverity.CRITICAL
                        )
                    )
                ).scalar() or 0

                if critical_current > critical_previous and critical_current > 0:
                    alerts.append({
                        "type": "critical_issue_spike",
                        "severity": "critical",
                        "message": f"Critical issues increased from {critical_previous} to {critical_current}",
                        "details": {
                            "previous_critical": critical_previous,
                            "current_critical": critical_current
                        },
                        "detected_at": datetime.now().isoformat()
                    })

                # Check resolution rate decline
                resolved_previous = session.execute(
                    select(func.count(QualityIssueModel.id)).where(
                        and_(
                            QualityIssueModel.created_at >= start_date,
                            QualityIssueModel.created_at < mid_date,
                            QualityIssueModel.status == IssueStatus.RESOLVED
                        )
                    )
                ).scalar() or 0

                resolved_current = session.execute(
                    select(func.count(QualityIssueModel.id)).where(
                        and_(
                            QualityIssueModel.created_at >= mid_date,
                            QualityIssueModel.created_at <= end_date,
                            QualityIssueModel.status == IssueStatus.RESOLVED
                        )
                    )
                ).scalar() or 0

                prev_rate = resolved_previous / previous_count if previous_count > 0 else 0
                curr_rate = resolved_current / current_count if current_count > 0 else 0

                if prev_rate > 0 and curr_rate < prev_rate * (1 - threshold):
                    alerts.append({
                        "type": "resolution_rate_decline",
                        "severity": "medium",
                        "message": f"Resolution rate declined from {prev_rate * 100:.1f}% to {curr_rate * 100:.1f}%",
                        "details": {
                            "previous_rate": round(prev_rate, 4),
                            "current_rate": round(curr_rate, 4)
                        },
                        "detected_at": datetime.now().isoformat()
                    })

                return alerts

        except Exception as e:
            logger.error(f"Error detecting quality decline: {e}")
            return []

    async def predict_quality(
        self,
        tenant_id: Optional[str] = None,
        days_ahead: int = None,
        history_days: int = 30
    ) -> Dict[str, Any]:
        """
        Predict future quality metrics.

        Args:
            tenant_id: Optional tenant filter
            days_ahead: Days to predict ahead
            history_days: Historical data to use

        Returns:
            Quality prediction results
        """
        days_ahead = days_ahead or self.DEFAULT_PREDICTION_DAYS

        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=history_days)

            with db_manager.get_session() as session:
                # Get daily issue counts
                query = select(
                    func.date(QualityIssueModel.created_at).label("day"),
                    func.count(QualityIssueModel.id).label("count")
                ).where(
                    QualityIssueModel.created_at >= start_date
                ).group_by(
                    func.date(QualityIssueModel.created_at)
                ).order_by("day")

                results = session.execute(query).all()

                if len(results) < self.MIN_DATA_POINTS:
                    return {
                        "status": "insufficient_data",
                        "message": f"Need at least {self.MIN_DATA_POINTS} data points, got {len(results)}",
                        "predictions": []
                    }

                # Extract values for prediction
                daily_counts = [r.count for r in results]

                # Simple linear regression prediction
                slope = self._calculate_slope(daily_counts)
                last_value = daily_counts[-1]
                avg_value = statistics.mean(daily_counts)

                predictions = []
                for i in range(1, days_ahead + 1):
                    pred_date = end_date + timedelta(days=i)
                    # Linear prediction with bounds
                    predicted = max(0, last_value + slope * i)

                    predictions.append({
                        "date": pred_date.date().isoformat(),
                        "predicted_issues": round(predicted, 1),
                        "confidence": max(0.5, 1 - (0.05 * i))  # Decrease confidence over time
                    })

                # Trend assessment
                if slope > 0.5:
                    trend_assessment = "worsening"
                elif slope < -0.5:
                    trend_assessment = "improving"
                else:
                    trend_assessment = "stable"

                return {
                    "status": "success",
                    "historical_data": {
                        "days_analyzed": len(results),
                        "avg_daily_issues": round(avg_value, 2),
                        "trend_slope": round(slope, 4)
                    },
                    "predictions": predictions,
                    "trend_assessment": trend_assessment,
                    "generated_at": datetime.now().isoformat()
                }

        except Exception as e:
            logger.error(f"Error predicting quality: {e}")
            return {"error": str(e)}

    async def identify_root_causes(
        self,
        tenant_id: Optional[str] = None,
        days: int = 30,
        min_occurrences: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Identify root causes of quality issues.

        Args:
            tenant_id: Optional tenant filter
            days: Analysis period
            min_occurrences: Minimum occurrences to consider

        Returns:
            List of identified root causes
        """
        try:
            start_date = datetime.now() - timedelta(days=days)

            with db_manager.get_session() as session:
                # Group by issue type
                query = select(
                    QualityIssueModel.issue_type,
                    func.count(QualityIssueModel.id).label("count"),
                    func.avg(
                        func.extract("epoch",
                            func.coalesce(
                                QualityIssueModel.resolved_at - QualityIssueModel.created_at,
                                func.now() - QualityIssueModel.created_at
                            )
                        )
                    ).label("avg_resolution_time")
                ).where(
                    QualityIssueModel.created_at >= start_date
                ).group_by(
                    QualityIssueModel.issue_type
                ).having(
                    func.count(QualityIssueModel.id) >= min_occurrences
                ).order_by(
                    func.count(QualityIssueModel.id).desc()
                )

                results = session.execute(query).all()

                root_causes = []
                total_issues = sum(r.count for r in results)

                for result in results:
                    percentage = (result.count / total_issues * 100) if total_issues > 0 else 0
                    avg_time_hours = result.avg_resolution_time / 3600 if result.avg_resolution_time else 0

                    # Determine impact level
                    if percentage >= 30:
                        impact = "critical"
                    elif percentage >= 15:
                        impact = "high"
                    elif percentage >= 5:
                        impact = "medium"
                    else:
                        impact = "low"

                    root_causes.append({
                        "issue_type": result.issue_type,
                        "occurrence_count": result.count,
                        "percentage": round(percentage, 2),
                        "avg_resolution_time_hours": round(avg_time_hours, 2),
                        "impact_level": impact,
                        "recommended_actions": self._get_recommended_actions(result.issue_type)
                    })

                return root_causes

        except Exception as e:
            logger.error(f"Error identifying root causes: {e}")
            return []

    async def get_quality_score_trend(
        self,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get quality score trend over time.

        Args:
            user_id: Optional user filter
            tenant_id: Optional tenant filter
            days: Analysis period

        Returns:
            Quality score trend data
        """
        try:
            # This would integrate with the evaluation module
            from src.evaluation.performance import PerformanceEngine

            engine = PerformanceEngine()
            history = await engine.get_user_performance_history(user_id) if user_id else []

            if not history:
                return {
                    "status": "no_data",
                    "message": "No performance history available",
                    "trend": []
                }

            # Extract quality scores
            trend_data = []
            for record in history[-days:]:  # Last N records
                trend_data.append({
                    "date": record.get("period_end"),
                    "quality_score": record.get("quality_score", 0),
                    "overall_score": record.get("overall_score", 0)
                })

            return {
                "status": "success",
                "user_id": user_id,
                "trend": trend_data,
                "generated_at": datetime.now().isoformat()
            }

        except ImportError:
            return {
                "status": "error",
                "message": "Performance module not available"
            }
        except Exception as e:
            logger.error(f"Error getting quality score trend: {e}")
            return {"error": str(e)}

    def _calculate_trend_direction(self, values: List[float]) -> str:
        """Calculate overall trend direction."""
        if len(values) < 2:
            return "stable"

        slope = self._calculate_slope(values)

        if abs(slope) < self.SIGNIFICANT_CHANGE_THRESHOLD:
            return "stable"
        elif slope > 0:
            return "increasing"
        else:
            return "decreasing"

    def _calculate_slope(self, values: List[float]) -> float:
        """Calculate linear regression slope."""
        if len(values) < 2:
            return 0.0

        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = statistics.mean(values)

        numerator = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        return numerator / denominator if denominator != 0 else 0.0

    def _aggregate_weekly(self, daily_metrics: Dict) -> Dict:
        """Aggregate daily metrics into weekly."""
        weekly = defaultdict(lambda: {
            "total_issues": 0,
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0
        })

        for day_str, metrics in daily_metrics.items():
            day = date.fromisoformat(day_str)
            week_start = day - timedelta(days=day.weekday())
            week_key = week_start.isoformat()

            for key in ["total_issues", "critical", "high", "medium", "low"]:
                weekly[week_key][key] += metrics[key]

        return dict(weekly)

    def _calculate_severity_distribution(self, issues: List) -> Dict[str, int]:
        """Calculate severity distribution."""
        distribution = {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0
        }

        for issue in issues:
            if issue.severity == IssueSeverity.CRITICAL:
                distribution["critical"] += 1
            elif issue.severity == IssueSeverity.HIGH:
                distribution["high"] += 1
            elif issue.severity == IssueSeverity.MEDIUM:
                distribution["medium"] += 1
            else:
                distribution["low"] += 1

        return distribution

    def _calculate_resolution_rate(self, issues: List) -> float:
        """Calculate issue resolution rate."""
        if not issues:
            return 0.0

        resolved = sum(1 for i in issues if i.status == IssueStatus.RESOLVED)
        return resolved / len(issues)

    def _get_recommended_actions(self, issue_type: str) -> List[str]:
        """Get recommended actions for an issue type."""
        recommendations = {
            "accuracy": [
                "Review annotation guidelines",
                "Provide additional training on accuracy standards",
                "Implement double-check workflow"
            ],
            "consistency": [
                "Calibrate annotator team",
                "Update standardization rules",
                "Add consistency checks to quality rules"
            ],
            "completeness": [
                "Review task requirements",
                "Add mandatory field validation",
                "Update training materials"
            ],
            "formatting": [
                "Update format specifications",
                "Add automated format validation",
                "Provide format examples"
            ],
            "timeliness": [
                "Review SLA targets",
                "Optimize task distribution",
                "Add workload monitoring"
            ]
        }

        return recommendations.get(issue_type.lower(), [
            "Investigate issue pattern",
            "Update quality rules",
            "Provide targeted training"
        ])
