"""
Data models for Text-to-SQL module.

Defines request/response models and supporting data structures.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, field_validator


class QueryComplexity(str, Enum):
    """Query complexity levels."""
    SIMPLE = "simple"           # Single table, basic conditions
    MODERATE = "moderate"       # Joins, aggregations
    COMPLEX = "complex"         # Subqueries, CTEs, window functions
    ADVANCED = "advanced"       # Multiple subqueries, complex analytics


class DatabaseDialect(str, Enum):
    """Supported database dialects."""
    MYSQL = "mysql"
    POSTGRESQL = "postgresql"
    ORACLE = "oracle"
    SQLITE = "sqlite"


class ColumnInfo(BaseModel):
    """Column information model."""

    name: str = Field(..., description="Column name")
    data_type: str = Field(..., description="Column data type")
    nullable: bool = Field(default=True, description="Whether column allows NULL")
    primary_key: bool = Field(default=False, description="Whether column is primary key")
    foreign_key: Optional[str] = Field(None, description="Foreign key reference (table.column)")
    default_value: Optional[str] = Field(None, description="Default value")
    description: Optional[str] = Field(None, description="Column description/comment")

    def to_schema_string(self) -> str:
        """Convert to schema description string."""
        parts = [f"{self.name} {self.data_type}"]
        if self.primary_key:
            parts.append("PRIMARY KEY")
        if not self.nullable:
            parts.append("NOT NULL")
        if self.foreign_key:
            parts.append(f"REFERENCES {self.foreign_key}")
        return " ".join(parts)


class TableInfo(BaseModel):
    """Table information model."""

    name: str = Field(..., description="Table name")
    schema_name: Optional[str] = Field(None, description="Schema name")
    columns: List[ColumnInfo] = Field(default_factory=list, description="Table columns")
    primary_keys: List[str] = Field(default_factory=list, description="Primary key columns")
    foreign_keys: Dict[str, str] = Field(default_factory=dict, description="Foreign key mappings")
    row_count: Optional[int] = Field(None, description="Approximate row count")
    description: Optional[str] = Field(None, description="Table description/comment")

    def get_column(self, name: str) -> Optional[ColumnInfo]:
        """Get column by name."""
        for col in self.columns:
            if col.name.lower() == name.lower():
                return col
        return None

    def to_schema_string(self) -> str:
        """Convert to CREATE TABLE schema string."""
        cols = ",\n  ".join(col.to_schema_string() for col in self.columns)
        return f"CREATE TABLE {self.name} (\n  {cols}\n)"

    def to_description_string(self) -> str:
        """Convert to natural language description."""
        col_descs = []
        for col in self.columns:
            desc = f"- {col.name} ({col.data_type})"
            if col.description:
                desc += f": {col.description}"
            col_descs.append(desc)
        return f"Table: {self.name}\nColumns:\n" + "\n".join(col_descs)


class SQLGenerationRequest(BaseModel):
    """Request model for SQL generation."""

    id: UUID = Field(default_factory=uuid4, description="Request ID")
    query: str = Field(..., min_length=1, description="Natural language query")
    database_id: Optional[str] = Field(None, description="Target database identifier")
    dialect: DatabaseDialect = Field(default=DatabaseDialect.POSTGRESQL, description="SQL dialect")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    max_results: Optional[int] = Field(None, ge=1, le=10000, description="Maximum rows to return")
    include_explanation: bool = Field(default=True, description="Include query explanation")
    validate_sql: bool = Field(default=True, description="Validate generated SQL")

    @field_validator('query')
    @classmethod
    def validate_query(cls, v):
        """Validate query is not empty."""
        if not v.strip():
            raise ValueError('Query cannot be empty')
        return v.strip()


class QueryPlan(BaseModel):
    """Query execution plan model."""

    tables_involved: List[str] = Field(default_factory=list, description="Tables used in query")
    columns_selected: List[str] = Field(default_factory=list, description="Columns selected")
    join_conditions: List[str] = Field(default_factory=list, description="JOIN conditions")
    filter_conditions: List[str] = Field(default_factory=list, description="WHERE conditions")
    aggregations: List[str] = Field(default_factory=list, description="Aggregation functions")
    group_by: List[str] = Field(default_factory=list, description="GROUP BY columns")
    order_by: List[str] = Field(default_factory=list, description="ORDER BY columns")
    has_subquery: bool = Field(default=False, description="Contains subqueries")
    has_cte: bool = Field(default=False, description="Contains CTEs")
    has_window_function: bool = Field(default=False, description="Contains window functions")
    complexity: QueryComplexity = Field(default=QueryComplexity.SIMPLE, description="Query complexity")

    def analyze_complexity(self) -> QueryComplexity:
        """Analyze and return query complexity."""
        if self.has_cte or (self.has_subquery and len(self.join_conditions) > 2):
            return QueryComplexity.ADVANCED
        elif self.has_subquery or self.has_window_function:
            return QueryComplexity.COMPLEX
        elif self.join_conditions or self.aggregations:
            return QueryComplexity.MODERATE
        return QueryComplexity.SIMPLE


class SQLValidationResult(BaseModel):
    """SQL validation result model."""

    is_valid: bool = Field(..., description="Whether SQL is valid")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    is_safe: bool = Field(default=True, description="Whether SQL is safe to execute")
    safety_issues: List[str] = Field(default_factory=list, description="Safety concerns")
    optimizations: List[str] = Field(default_factory=list, description="Optimization suggestions")


class SQLGenerationResponse(BaseModel):
    """Response model for SQL generation."""

    id: UUID = Field(default_factory=uuid4, description="Response ID")
    request_id: UUID = Field(..., description="Original request ID")
    sql: str = Field(..., description="Generated SQL query")
    formatted_sql: Optional[str] = Field(None, description="Formatted SQL query")
    explanation: Optional[str] = Field(None, description="Natural language explanation")
    query_plan: Optional[QueryPlan] = Field(None, description="Query execution plan")
    validation: Optional[SQLValidationResult] = Field(None, description="Validation result")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence score")
    alternatives: List[str] = Field(default_factory=list, description="Alternative SQL queries")
    processing_time: float = Field(default=0.0, description="Processing time in seconds")
    model_used: Optional[str] = Field(None, description="AI model used for generation")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v):
        """Validate confidence score."""
        if not 0.0 <= v <= 1.0:
            raise ValueError('Confidence must be between 0.0 and 1.0')
        return v


class SchemaContext(BaseModel):
    """Database schema context for SQL generation."""

    tables: List[TableInfo] = Field(default_factory=list, description="Available tables")
    relationships: Dict[str, List[str]] = Field(default_factory=dict, description="Table relationships")
    common_queries: List[str] = Field(default_factory=list, description="Common query patterns")
    business_terms: Dict[str, str] = Field(default_factory=dict, description="Business term mappings")

    def get_relevant_tables(self, keywords: List[str]) -> List[TableInfo]:
        """Find tables relevant to given keywords."""
        relevant = []
        keywords_lower = [k.lower() for k in keywords]

        for table in self.tables:
            # Check table name
            if any(kw in table.name.lower() for kw in keywords_lower):
                relevant.append(table)
                continue

            # Check column names
            for col in table.columns:
                if any(kw in col.name.lower() for kw in keywords_lower):
                    relevant.append(table)
                    break

        return relevant

    def to_prompt_context(self) -> str:
        """Convert to prompt context string."""
        parts = ["Database Schema:"]
        for table in self.tables:
            parts.append(table.to_description_string())

        if self.relationships:
            parts.append("\nTable Relationships:")
            for table, related in self.relationships.items():
                parts.append(f"- {table} -> {', '.join(related)}")

        if self.business_terms:
            parts.append("\nBusiness Terms:")
            for term, definition in self.business_terms.items():
                parts.append(f"- {term}: {definition}")

        return "\n".join(parts)


class ExecutionResult(BaseModel):
    """SQL execution result model."""

    success: bool = Field(..., description="Whether execution succeeded")
    data: List[Dict[str, Any]] = Field(default_factory=list, description="Result data")
    row_count: int = Field(default=0, description="Number of rows returned")
    column_names: List[str] = Field(default_factory=list, description="Column names")
    execution_time: float = Field(default=0.0, description="Execution time in seconds")
    error: Optional[str] = Field(None, description="Error message if failed")
    warnings: List[str] = Field(default_factory=list, description="Execution warnings")
