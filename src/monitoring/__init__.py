"""
Monitoring module for SuperInsight Platform.

Provides comprehensive monitoring capabilities including:
- Quality monitoring and dashboards
- Alert management and notifications
- Advanced anomaly detection with ML
- SLA compliance monitoring
- Capacity planning and prediction
- Report generation and scheduling
"""

def get_quality_monitor():
    """Get QualityMonitor instance with lazy import."""
    from .quality_monitor import QualityMonitor
    return QualityMonitor()


def get_alert_manager():
    """Get AlertManager instance with lazy import."""
    from .alert_manager import AlertManager
    return AlertManager()


def get_anomaly_detector():
    """Get AdvancedAnomalyDetector instance with lazy import."""
    from .advanced_anomaly_detection import advanced_anomaly_detector
    return advanced_anomaly_detector


def get_report_service():
    """Get MonitoringReportService instance with lazy import."""
    from .report_service import monitoring_report_service
    return monitoring_report_service


__all__ = [
    "get_quality_monitor",
    "get_alert_manager",
    "get_anomaly_detector",
    "get_report_service",
]
