"""
Quality management module for SuperInsight Platform.

Provides quality assessment, rule management, issue tracking,
and data repair functionality.
"""

# Import repair module directly to avoid Ragas import issues
from .repair import DataRepairService, RepairRecord, RepairType, RepairStatus, RepairApprovalWorkflow

# Lazy import for manager to avoid Ragas dependency issues
def get_quality_manager():
    """Get QualityManager instance with lazy import."""
    from .manager import QualityManager
    return QualityManager()

def get_quality_rule():
    """Get QualityRule class with lazy import."""
    from .manager import QualityRule
    return QualityRule

def get_quality_rule_type():
    """Get QualityRuleType enum with lazy import."""
    from .manager import QualityRuleType
    return QualityRuleType

def get_quality_report():
    """Get QualityReport class with lazy import."""
    from .manager import QualityReport
    return QualityReport

def get_trend_analyzer():
    """Get QualityTrendAnalyzer instance with lazy import."""
    from .trend_analyzer import QualityTrendAnalyzer
    return QualityTrendAnalyzer()


def get_auto_retrain_trigger():
    """Get AutoRetrainTrigger instance with lazy import."""
    from .auto_retrain import AutoRetrainTrigger
    return AutoRetrainTrigger()


__all__ = [
    "DataRepairService",
    "RepairRecord",
    "RepairType",
    "RepairStatus",
    "RepairApprovalWorkflow",
    "get_quality_manager",
    "get_quality_rule",
    "get_quality_rule_type",
    "get_quality_report",
    "get_trend_analyzer",
    "get_auto_retrain_trigger",
]