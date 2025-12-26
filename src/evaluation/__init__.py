"""
Performance evaluation module for SuperInsight Platform.

Provides performance assessment, appeal management, and reporting
functionality for annotator performance tracking.
"""

from .models import (
    PerformanceRecordModel,
    AppealModel,
    PerformanceRecord,
    Appeal,
    PerformanceStatus,
    AppealStatus,
    PerformancePeriod,
)

# Lazy imports for services to avoid circular dependency issues
def get_performance_engine():
    """Get PerformanceEngine instance with lazy import."""
    from .performance import PerformanceEngine
    return PerformanceEngine()

def get_appeal_manager():
    """Get AppealManager instance with lazy import."""
    from .appeal import AppealManager
    return AppealManager()

def get_report_generator():
    """Get ReportGenerator instance with lazy import."""
    from .report_generator import ReportGenerator
    return ReportGenerator()

__all__ = [
    # Models
    "PerformanceRecordModel",
    "AppealModel",
    "PerformanceRecord",
    "Appeal",
    "PerformanceStatus",
    "AppealStatus",
    "PerformancePeriod",
    # Service getters
    "get_performance_engine",
    "get_appeal_manager",
    "get_report_generator",
]
