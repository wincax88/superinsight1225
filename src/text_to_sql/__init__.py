"""
Text-to-SQL Module for SuperInsight Platform.

Provides natural language to SQL query generation capabilities
using integrated AI models.
"""

from .models import (
    SQLGenerationRequest,
    SQLGenerationResponse,
    QueryPlan,
    SQLValidationResult,
    TableInfo,
    ColumnInfo,
    QueryComplexity
)
from .schema_manager import SchemaManager
from .llm_adapter import LLMAdapter, get_llm_adapter
from .sql_generator import SQLGenerator
from .advanced_sql import AdvancedSQLGenerator

__all__ = [
    # Models
    "SQLGenerationRequest",
    "SQLGenerationResponse",
    "QueryPlan",
    "SQLValidationResult",
    "TableInfo",
    "ColumnInfo",
    "QueryComplexity",
    # Core Classes
    "SchemaManager",
    "LLMAdapter",
    "get_llm_adapter",
    "SQLGenerator",
    "AdvancedSQLGenerator",
]

__version__ = "1.0.0"
