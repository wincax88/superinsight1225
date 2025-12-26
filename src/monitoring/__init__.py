"""
Monitoring module for SuperInsight Platform.

Provides quality monitoring, dashboards, and alerting.
"""

def get_quality_monitor():
    """Get QualityMonitor instance with lazy import."""
    from .quality_monitor import QualityMonitor
    return QualityMonitor()


def get_alert_manager():
    """Get AlertManager instance with lazy import."""
    from .alert_manager import AlertManager
    return AlertManager()


__all__ = [
    "get_quality_monitor",
    "get_alert_manager",
]
