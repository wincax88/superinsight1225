"""
Knowledge Management Module for SuperInsight Platform.

Provides knowledge base management, rule learning, and case library maintenance.
"""

from .models import (
    KnowledgeEntry,
    KnowledgeRule,
    CaseEntry,
    KnowledgeCategory,
    RuleType,
    CaseStatus,
    KnowledgeUpdateResult,
    QualityScore
)
from .knowledge_base import KnowledgeBase, get_knowledge_base
from .rule_engine import RuleEngine, get_rule_engine
from .case_library import CaseLibrary, get_case_library
from .auto_updater import KnowledgeAutoUpdater, get_auto_updater

__all__ = [
    # Models
    "KnowledgeEntry",
    "KnowledgeRule",
    "CaseEntry",
    "KnowledgeCategory",
    "RuleType",
    "CaseStatus",
    "KnowledgeUpdateResult",
    "QualityScore",
    # Core Classes
    "KnowledgeBase",
    "get_knowledge_base",
    "RuleEngine",
    "get_rule_engine",
    "CaseLibrary",
    "get_case_library",
    "KnowledgeAutoUpdater",
    "get_auto_updater",
]

__version__ = "1.0.0"
