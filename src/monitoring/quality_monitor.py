"""
Real-time quality monitoring for SuperInsight Platform.

Provides:
- Real-time quality metrics tracking
- Anomaly detection
- Dashboard data aggregation
- Alert triggering
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from collections import defaultdict, deque
from dataclasses import dataclass, field
import statistics
import asyncio

from sqlalchemy import select, func, and_

from src.database.connection import db_manager
from src.database.models import QualityIssueModel, IssueSeverity, IssueStatus

logger = logging.getLogger(__name__)


@dataclass
class QualityMetricPoint:
    """Quality metric data point."""
    timestamp: datetime
    value: float
    metric_type: str
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityAnomaly:
    """Detected quality anomaly."""
    anomaly_type: str
    severity: str  # low, medium, high, critical
    metric_name: str
    current_value: float
    expected_value: float
    deviation: float
    detected_at: datetime
    context: Dict[str, Any] = field(default_factory=dict)


class QualityMonitor:
    """
    Real-time quality monitoring engine.

    Tracks quality metrics in real-time, detects anomalies,
    and provides dashboard data.
    """

    # Metric retention settings
    MAX_HISTORY_POINTS = 1000
    AGGREGATION_INTERVALS = {
        "1m": 60,
        "5m": 300,
        "1h": 3600,
        "1d": 86400
    }

    # Anomaly detection thresholds
    ANOMALY_THRESHOLDS = {
        "z_score": 3.0,  # Standard deviations for anomaly
        "sudden_change": 0.30,  # 30% sudden change
        "trend_threshold": 0.15  # 15% trend deviation
    }

    # Alert thresholds
    ALERT_THRESHOLDS = {
        "quality_score": {"warning": 0.75, "critical": 0.60},
        "error_rate": {"warning": 0.15, "critical": 0.25},
        "resolution_time": {"warning": 4 * 3600, "critical": 8 * 3600},  # seconds
        "issue_count": {"warning": 50, "critical": 100}  # per hour
    }

    def __init__(self):
        """Initialize the quality monitor."""
        self._metrics_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.MAX_HISTORY_POINTS)
        )
        self._anomalies: List[QualityAnomaly] = []
        self._is_monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._alert_callbacks: List[callable] = []

    async def start_monitoring(self, interval: int = 60):
        """
        Start real-time monitoring.

        Args:
            interval: Collection interval in seconds
        """
        if self._is_monitoring:
            return

        self._is_monitoring = True
        self._monitor_task = asyncio.create_task(
            self._monitoring_loop(interval)
        )
        logger.info("Quality monitoring started")

    async def stop_monitoring(self):
        """Stop real-time monitoring."""
        self._is_monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Quality monitoring stopped")

    async def _monitoring_loop(self, interval: int):
        """Main monitoring loop."""
        while self._is_monitoring:
            try:
                await self._collect_metrics()
                await self._detect_anomalies()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(interval)

    async def _collect_metrics(self):
        """Collect current quality metrics."""
        now = datetime.now()

        try:
            with db_manager.get_session() as session:
                # Get recent issue counts
                hour_ago = now - timedelta(hours=1)

                total_issues = session.execute(
                    select(func.count(QualityIssueModel.id)).where(
                        QualityIssueModel.created_at >= hour_ago
                    )
                ).scalar() or 0

                # Get issues by severity
                critical_issues = session.execute(
                    select(func.count(QualityIssueModel.id)).where(
                        and_(
                            QualityIssueModel.created_at >= hour_ago,
                            QualityIssueModel.severity == IssueSeverity.CRITICAL
                        )
                    )
                ).scalar() or 0

                # Get resolution rate
                resolved_issues = session.execute(
                    select(func.count(QualityIssueModel.id)).where(
                        and_(
                            QualityIssueModel.created_at >= hour_ago,
                            QualityIssueModel.status == IssueStatus.RESOLVED
                        )
                    )
                ).scalar() or 0

                resolution_rate = resolved_issues / total_issues if total_issues > 0 else 1.0

                # Store metrics
                self._record_metric("issues_per_hour", total_issues, now)
                self._record_metric("critical_issues", critical_issues, now)
                self._record_metric("resolution_rate", resolution_rate, now)

                # Calculate error rate (approximation)
                error_rate = critical_issues / total_issues if total_issues > 0 else 0.0
                self._record_metric("error_rate", error_rate, now)

        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")

    def _record_metric(
        self,
        metric_name: str,
        value: float,
        timestamp: datetime,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        """Record a metric point."""
        point = QualityMetricPoint(
            timestamp=timestamp,
            value=value,
            metric_type=metric_name,
            tenant_id=tenant_id,
            user_id=user_id
        )
        self._metrics_history[metric_name].append(point)

    async def _detect_anomalies(self):
        """Detect anomalies in collected metrics."""
        for metric_name, history in self._metrics_history.items():
            if len(history) < 10:
                continue

            values = [p.value for p in history]
            current = values[-1]

            # Z-score anomaly detection
            if len(values) >= 30:
                mean = statistics.mean(values[:-1])
                stdev = statistics.stdev(values[:-1]) if len(values) > 2 else 0

                if stdev > 0:
                    z_score = abs(current - mean) / stdev
                    if z_score > self.ANOMALY_THRESHOLDS["z_score"]:
                        self._record_anomaly(
                            metric_name=metric_name,
                            anomaly_type="z_score",
                            current=current,
                            expected=mean,
                            deviation=z_score
                        )

            # Sudden change detection
            if len(values) >= 2:
                prev = values[-2]
                if prev > 0:
                    change = abs(current - prev) / prev
                    if change > self.ANOMALY_THRESHOLDS["sudden_change"]:
                        self._record_anomaly(
                            metric_name=metric_name,
                            anomaly_type="sudden_change",
                            current=current,
                            expected=prev,
                            deviation=change
                        )

    def _record_anomaly(
        self,
        metric_name: str,
        anomaly_type: str,
        current: float,
        expected: float,
        deviation: float
    ):
        """Record a detected anomaly."""
        # Determine severity
        if deviation > 5:
            severity = "critical"
        elif deviation > 3:
            severity = "high"
        elif deviation > 2:
            severity = "medium"
        else:
            severity = "low"

        anomaly = QualityAnomaly(
            anomaly_type=anomaly_type,
            severity=severity,
            metric_name=metric_name,
            current_value=current,
            expected_value=expected,
            deviation=deviation,
            detected_at=datetime.now()
        )

        self._anomalies.append(anomaly)

        # Trigger alert callbacks
        for callback in self._alert_callbacks:
            try:
                callback(anomaly)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")

        logger.warning(
            f"Anomaly detected: {metric_name} - {anomaly_type} "
            f"(current={current:.2f}, expected={expected:.2f}, deviation={deviation:.2f})"
        )

    async def get_realtime_metrics(
        self,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get current real-time metrics.

        Args:
            tenant_id: Optional tenant filter

        Returns:
            Real-time metrics dictionary
        """
        now = datetime.now()

        try:
            with db_manager.get_session() as session:
                # Current hour metrics
                hour_ago = now - timedelta(hours=1)

                query = select(func.count(QualityIssueModel.id)).where(
                    QualityIssueModel.created_at >= hour_ago
                )
                total_issues = session.execute(query).scalar() or 0

                # By severity
                severity_counts = {}
                for severity in IssueSeverity:
                    count = session.execute(
                        select(func.count(QualityIssueModel.id)).where(
                            and_(
                                QualityIssueModel.created_at >= hour_ago,
                                QualityIssueModel.severity == severity
                            )
                        )
                    ).scalar() or 0
                    severity_counts[severity.value] = count

                # By status
                status_counts = {}
                for status in IssueStatus:
                    count = session.execute(
                        select(func.count(QualityIssueModel.id)).where(
                            and_(
                                QualityIssueModel.created_at >= hour_ago,
                                QualityIssueModel.status == status
                            )
                        )
                    ).scalar() or 0
                    status_counts[status.value] = count

                # Calculate rates
                resolved = status_counts.get("resolved", 0)
                resolution_rate = resolved / total_issues if total_issues > 0 else 1.0

                critical = severity_counts.get("critical", 0)
                critical_rate = critical / total_issues if total_issues > 0 else 0.0

                return {
                    "timestamp": now.isoformat(),
                    "period": "last_hour",
                    "total_issues": total_issues,
                    "by_severity": severity_counts,
                    "by_status": status_counts,
                    "resolution_rate": round(resolution_rate, 4),
                    "critical_rate": round(critical_rate, 4),
                    "alert_status": self._get_alert_status(
                        total_issues, resolution_rate, critical_rate
                    )
                }

        except Exception as e:
            logger.error(f"Error getting realtime metrics: {e}")
            return {"error": str(e)}

    def _get_alert_status(
        self,
        issue_count: int,
        resolution_rate: float,
        critical_rate: float
    ) -> Dict[str, str]:
        """Determine alert status based on metrics."""
        status = {}

        # Issue count alerts
        if issue_count >= self.ALERT_THRESHOLDS["issue_count"]["critical"]:
            status["issue_count"] = "critical"
        elif issue_count >= self.ALERT_THRESHOLDS["issue_count"]["warning"]:
            status["issue_count"] = "warning"
        else:
            status["issue_count"] = "normal"

        # Resolution rate (inverted - lower is worse)
        quality_score = resolution_rate
        if quality_score <= self.ALERT_THRESHOLDS["quality_score"]["critical"]:
            status["quality"] = "critical"
        elif quality_score <= self.ALERT_THRESHOLDS["quality_score"]["warning"]:
            status["quality"] = "warning"
        else:
            status["quality"] = "normal"

        # Error rate
        if critical_rate >= self.ALERT_THRESHOLDS["error_rate"]["critical"]:
            status["error_rate"] = "critical"
        elif critical_rate >= self.ALERT_THRESHOLDS["error_rate"]["warning"]:
            status["error_rate"] = "warning"
        else:
            status["error_rate"] = "normal"

        # Overall status
        if "critical" in status.values():
            status["overall"] = "critical"
        elif "warning" in status.values():
            status["overall"] = "warning"
        else:
            status["overall"] = "normal"

        return status

    async def check_anomalies(
        self,
        tenant_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get recent anomalies.

        Args:
            tenant_id: Optional tenant filter

        Returns:
            List of anomalies
        """
        # Get anomalies from last hour
        cutoff = datetime.now() - timedelta(hours=1)
        recent = [
            a for a in self._anomalies
            if a.detected_at >= cutoff
        ]

        return [
            {
                "anomaly_type": a.anomaly_type,
                "severity": a.severity,
                "metric_name": a.metric_name,
                "current_value": a.current_value,
                "expected_value": a.expected_value,
                "deviation": round(a.deviation, 4),
                "detected_at": a.detected_at.isoformat()
            }
            for a in recent
        ]

    async def get_quality_dashboard(
        self,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive quality dashboard data.

        Args:
            tenant_id: Optional tenant filter

        Returns:
            Dashboard data dictionary
        """
        now = datetime.now()

        try:
            # Get realtime metrics
            realtime = await self.get_realtime_metrics(tenant_id)

            # Get historical data for charts
            historical = self._get_historical_charts()

            # Get anomalies
            anomalies = await self.check_anomalies(tenant_id)

            # Get top issues
            top_issues = await self._get_top_issues(tenant_id)

            return {
                "generated_at": now.isoformat(),
                "realtime": realtime,
                "historical": historical,
                "anomalies": {
                    "count": len(anomalies),
                    "items": anomalies[:10]  # Top 10
                },
                "top_issues": top_issues,
                "health_score": self._calculate_health_score(realtime)
            }

        except Exception as e:
            logger.error(f"Error getting dashboard: {e}")
            return {"error": str(e)}

    def _get_historical_charts(self) -> Dict[str, List]:
        """Get historical data for charts."""
        charts = {}

        for metric_name, history in self._metrics_history.items():
            if history:
                charts[metric_name] = [
                    {
                        "timestamp": p.timestamp.isoformat(),
                        "value": p.value
                    }
                    for p in list(history)[-100:]  # Last 100 points
                ]

        return charts

    async def _get_top_issues(
        self,
        tenant_id: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get top issue types."""
        try:
            with db_manager.get_session() as session:
                day_ago = datetime.now() - timedelta(days=1)

                query = select(
                    QualityIssueModel.issue_type,
                    func.count(QualityIssueModel.id).label("count")
                ).where(
                    QualityIssueModel.created_at >= day_ago
                ).group_by(
                    QualityIssueModel.issue_type
                ).order_by(
                    func.count(QualityIssueModel.id).desc()
                ).limit(limit)

                results = session.execute(query).all()

                return [
                    {"issue_type": r.issue_type, "count": r.count}
                    for r in results
                ]

        except Exception as e:
            logger.error(f"Error getting top issues: {e}")
            return []

    def _calculate_health_score(self, realtime: Dict[str, Any]) -> float:
        """Calculate overall health score (0-100)."""
        score = 100.0

        alert_status = realtime.get("alert_status", {})

        # Deductions for alerts
        deductions = {
            "critical": 30,
            "warning": 15
        }

        for key, status in alert_status.items():
            if key == "overall":
                continue
            if status in deductions:
                score -= deductions[status]

        return max(0, min(100, score))

    def register_alert_callback(self, callback: callable):
        """Register a callback for anomaly alerts."""
        self._alert_callbacks.append(callback)

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of collected metrics."""
        summary = {}

        for metric_name, history in self._metrics_history.items():
            if history:
                values = [p.value for p in history]
                summary[metric_name] = {
                    "current": values[-1] if values else 0,
                    "avg": statistics.mean(values) if values else 0,
                    "min": min(values) if values else 0,
                    "max": max(values) if values else 0,
                    "count": len(values)
                }

        return summary

    def update_thresholds(self, thresholds: Dict[str, Dict[str, float]]):
        """Update alert thresholds."""
        for key, value in thresholds.items():
            if key in self.ALERT_THRESHOLDS:
                self.ALERT_THRESHOLDS[key].update(value)
                logger.info(f"Updated threshold {key}: {value}")
