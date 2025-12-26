"""
Billing system module for SuperInsight platform.

Provides billing tracking, calculation, and reporting functionality.
"""

from src.billing.models import (
    BillingRecord,
    BillingRule,
    Bill,
    BillingReport,
    BillingMode
)
from src.billing.service import BillingSystem
from src.billing.analytics import BillingAnalytics


def get_quality_pricing_engine():
    """Get QualityPricingEngine instance with lazy import."""
    from .quality_pricing import QualityPricingEngine
    return QualityPricingEngine()


def get_incentive_manager():
    """Get IncentiveManager instance with lazy import."""
    from .incentive_manager import IncentiveManager
    return IncentiveManager()


__all__ = [
    'BillingRecord',
    'BillingRule',
    'Bill',
    'BillingReport',
    'BillingMode',
    'BillingSystem',
    'BillingAnalytics',
    'get_quality_pricing_engine',
    'get_incentive_manager',
]