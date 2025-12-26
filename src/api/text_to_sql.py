"""
FastAPI endpoints for Text-to-SQL in SuperInsight Platform.

Provides RESTful API for natural language to SQL query generation.
"""

import logging
import time
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, HTTPException, status, Query, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.text_to_sql import (
    SQLGenerationRequest,
    SQLGenerationResponse,
    SQLValidationResult,
    QueryPlan,
    TableInfo,
    DatabaseDialect
)
from src.text_to_sql.sql_generator import SQLGenerator, get_sql_generator
from src.text_to_sql.advanced_sql import AdvancedSQLGenerator, get_advanced_sql_generator
from src.text_to_sql.schema_manager import SchemaManager, get_schema_manager

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/text-to-sql", tags=["Text-to-SQL"])


# Request/Response Models
class GenerateSQLRequest(BaseModel):
    """Request model for SQL generation."""
    query: str = Field(..., min_length=1, description="Natural language query")
    database_id: Optional[str] = Field(None, description="Target database identifier")
    connection_string: Optional[str] = Field(None, description="Database connection string")
    dialect: DatabaseDialect = Field(default=DatabaseDialect.POSTGRESQL, description="SQL dialect")
    max_results: Optional[int] = Field(None, ge=1, le=10000, description="Maximum rows to return")
    include_explanation: bool = Field(default=True, description="Include query explanation")
    validate_sql: bool = Field(default=True, description="Validate generated SQL")
    use_advanced: bool = Field(default=False, description="Use advanced SQL generation")


class ExecuteSQLRequest(BaseModel):
    """Request model for SQL execution."""
    sql: str = Field(..., min_length=1, description="SQL query to execute")
    connection_string: str = Field(..., description="Database connection string")
    max_rows: int = Field(default=100, ge=1, le=10000, description="Maximum rows to return")
    timeout: int = Field(default=30, ge=1, le=300, description="Query timeout in seconds")


class ValidateSQLRequest(BaseModel):
    """Request model for SQL validation."""
    sql: str = Field(..., min_length=1, description="SQL query to validate")
    dialect: DatabaseDialect = Field(default=DatabaseDialect.POSTGRESQL, description="SQL dialect")


class SchemaRequest(BaseModel):
    """Request model for schema retrieval."""
    connection_string: str = Field(..., description="Database connection string")
    schema_name: Optional[str] = Field(None, description="Schema name to load")
    include_tables: Optional[List[str]] = Field(None, description="Tables to include")
    exclude_tables: Optional[List[str]] = Field(None, description="Tables to exclude")


class GenerateSQLResponse(BaseModel):
    """Response model for SQL generation."""
    success: bool = Field(..., description="Generation success status")
    sql: str = Field(..., description="Generated SQL query")
    formatted_sql: Optional[str] = Field(None, description="Formatted SQL")
    explanation: Optional[str] = Field(None, description="Query explanation")
    confidence: float = Field(..., description="Confidence score")
    processing_time: float = Field(..., description="Processing time in seconds")
    query_plan: Optional[QueryPlan] = Field(None, description="Query execution plan")
    validation: Optional[SQLValidationResult] = Field(None, description="Validation result")
    model_used: Optional[str] = Field(None, description="AI model used")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ExecuteSQLResponse(BaseModel):
    """Response model for SQL execution."""
    success: bool = Field(..., description="Execution success status")
    data: List[Dict[str, Any]] = Field(default_factory=list, description="Query results")
    row_count: int = Field(default=0, description="Number of rows returned")
    column_names: List[str] = Field(default_factory=list, description="Column names")
    execution_time: float = Field(default=0.0, description="Execution time in seconds")
    error: Optional[str] = Field(None, description="Error message if failed")
    warnings: List[str] = Field(default_factory=list, description="Execution warnings")


class SchemaResponse(BaseModel):
    """Response model for schema retrieval."""
    success: bool = Field(..., description="Retrieval success status")
    tables: List[TableInfo] = Field(default_factory=list, description="Table information")
    table_count: int = Field(default=0, description="Number of tables")
    relationships: Dict[str, List[str]] = Field(default_factory=dict, description="Table relationships")
    error: Optional[str] = Field(None, description="Error message if failed")


# Service instances
_sql_generator: Optional[SQLGenerator] = None
_advanced_generator: Optional[AdvancedSQLGenerator] = None
_schema_manager: Optional[SchemaManager] = None


def get_generator() -> SQLGenerator:
    """Get SQL generator instance."""
    global _sql_generator
    if _sql_generator is None:
        _sql_generator = get_sql_generator()
    return _sql_generator


def get_advanced() -> AdvancedSQLGenerator:
    """Get advanced SQL generator instance."""
    global _advanced_generator
    if _advanced_generator is None:
        _advanced_generator = get_advanced_sql_generator()
    return _advanced_generator


def get_manager() -> SchemaManager:
    """Get schema manager instance."""
    global _schema_manager
    if _schema_manager is None:
        _schema_manager = get_schema_manager()
    return _schema_manager


# Endpoints
@router.post("/generate", response_model=GenerateSQLResponse)
async def generate_sql(request: GenerateSQLRequest) -> GenerateSQLResponse:
    """
    Generate SQL from natural language query.

    Converts a natural language query into SQL using AI models.
    """
    try:
        logger.info(f"Generating SQL for query: {request.query[:50]}...")
        start_time = time.time()

        # Get appropriate generator
        if request.use_advanced:
            generator = get_advanced()
        else:
            generator = get_generator()

        # Set connection if provided
        if request.connection_string:
            generator.set_connection(request.connection_string)

        # Create generation request
        gen_request = SQLGenerationRequest(
            query=request.query,
            database_id=request.database_id,
            dialect=request.dialect,
            max_results=request.max_results,
            include_explanation=request.include_explanation,
            validate_sql=request.validate_sql
        )

        # Generate SQL
        response = await generator.generate_sql(gen_request)

        return GenerateSQLResponse(
            success=bool(response.sql),
            sql=response.sql,
            formatted_sql=response.formatted_sql,
            explanation=response.explanation,
            confidence=response.confidence,
            processing_time=response.processing_time,
            query_plan=response.query_plan,
            validation=response.validation,
            model_used=response.model_used,
            metadata=response.metadata
        )

    except Exception as e:
        logger.error(f"SQL generation failed: {e}")
        return GenerateSQLResponse(
            success=False,
            sql="",
            confidence=0.0,
            processing_time=time.time() - start_time,
            metadata={"error": str(e)}
        )


@router.post("/execute", response_model=ExecuteSQLResponse)
async def execute_sql(request: ExecuteSQLRequest) -> ExecuteSQLResponse:
    """
    Execute a SQL query and return results.

    Executes the provided SQL query against the specified database.
    Only SELECT queries are allowed for security.
    """
    try:
        logger.info(f"Executing SQL query: {request.sql[:50]}...")
        start_time = time.time()

        # Validate SQL is SELECT only
        sql_upper = request.sql.upper().strip()
        if not (sql_upper.startswith("SELECT") or sql_upper.startswith("WITH")):
            return ExecuteSQLResponse(
                success=False,
                error="Only SELECT queries are allowed",
                execution_time=time.time() - start_time
            )

        # Check for dangerous operations
        forbidden = ["INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", "TRUNCATE"]
        for keyword in forbidden:
            if keyword in sql_upper:
                return ExecuteSQLResponse(
                    success=False,
                    error=f"Forbidden operation: {keyword}",
                    execution_time=time.time() - start_time
                )

        # Execute query using schema manager
        manager = get_manager()
        engine = manager.get_engine(request.connection_string)

        from sqlalchemy import text

        data = []
        column_names = []
        warnings = []

        with engine.connect() as conn:
            # Set timeout (PostgreSQL specific)
            try:
                conn.execute(text(f"SET statement_timeout = {request.timeout * 1000}"))
            except Exception:
                pass  # Not all databases support this

            result = conn.execute(text(request.sql))

            # Get column names
            column_names = list(result.keys())

            # Fetch rows
            rows = result.fetchmany(request.max_rows)
            data = [dict(zip(column_names, row)) for row in rows]

            # Check if more rows available
            if result.fetchone() is not None:
                warnings.append(f"Result truncated to {request.max_rows} rows")

        execution_time = time.time() - start_time

        return ExecuteSQLResponse(
            success=True,
            data=data,
            row_count=len(data),
            column_names=column_names,
            execution_time=execution_time,
            warnings=warnings
        )

    except Exception as e:
        logger.error(f"SQL execution failed: {e}")
        return ExecuteSQLResponse(
            success=False,
            error=str(e),
            execution_time=time.time() - start_time
        )


@router.post("/validate", response_model=SQLValidationResult)
async def validate_sql(request: ValidateSQLRequest) -> SQLValidationResult:
    """
    Validate a SQL query.

    Checks SQL syntax, safety, and provides optimization suggestions.
    """
    try:
        logger.info(f"Validating SQL: {request.sql[:50]}...")

        generator = get_generator()
        result = generator._validate_sql(request.sql, request.dialect)

        return result

    except Exception as e:
        logger.error(f"SQL validation failed: {e}")
        return SQLValidationResult(
            is_valid=False,
            errors=[str(e)],
            is_safe=False
        )


@router.post("/schema", response_model=SchemaResponse)
async def get_schema(request: SchemaRequest) -> SchemaResponse:
    """
    Get database schema information.

    Retrieves table and column information from the database.
    """
    try:
        logger.info("Loading database schema...")

        manager = get_manager()
        context = manager.load_schema(
            request.connection_string,
            schema_name=request.schema_name,
            include_tables=request.include_tables,
            exclude_tables=request.exclude_tables
        )

        return SchemaResponse(
            success=True,
            tables=context.tables,
            table_count=len(context.tables),
            relationships=context.relationships
        )

    except Exception as e:
        logger.error(f"Schema loading failed: {e}")
        return SchemaResponse(
            success=False,
            error=str(e)
        )


@router.get("/schema/tables", response_model=List[str])
async def list_tables(connection_string: str = Query(..., description="Database connection string")) -> List[str]:
    """
    List all tables in the database.
    """
    try:
        manager = get_manager()
        context = manager.load_schema(connection_string)
        return [table.name for table in context.tables]

    except Exception as e:
        logger.error(f"Failed to list tables: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list tables: {str(e)}"
        )


@router.get("/schema/table/{table_name}", response_model=TableInfo)
async def get_table_info(
    table_name: str,
    connection_string: str = Query(..., description="Database connection string")
) -> TableInfo:
    """
    Get information for a specific table.
    """
    try:
        manager = get_manager()
        table = manager.get_table_info(connection_string, table_name)

        if not table:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Table not found: {table_name}"
            )

        return table

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get table info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get table info: {str(e)}"
        )


@router.get("/metrics")
async def get_metrics() -> JSONResponse:
    """
    Get Text-to-SQL service metrics.
    """
    try:
        generator = get_generator()
        gen_stats = generator.get_statistics()

        from src.text_to_sql.llm_adapter import get_llm_adapter
        llm_adapter = get_llm_adapter()
        llm_stats = llm_adapter.get_statistics()

        return JSONResponse(content={
            "success": True,
            "metrics": {
                "generator": gen_stats,
                "llm_adapter": llm_stats
            }
        })

    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"success": False, "error": str(e)}
        )


@router.get("/health")
async def health_check() -> JSONResponse:
    """
    Health check endpoint for Text-to-SQL service.
    """
    try:
        from src.text_to_sql.llm_adapter import get_llm_adapter
        llm_adapter = get_llm_adapter()
        llm_available = await llm_adapter.check_availability()

        return JSONResponse(content={
            "status": "healthy" if llm_available else "degraded",
            "components": {
                "sql_generator": "available",
                "schema_manager": "available",
                "llm_adapter": "available" if llm_available else "unavailable"
            }
        })

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )


@router.post("/analyze")
async def analyze_query(
    query: str = Query(..., description="Natural language query to analyze")
) -> JSONResponse:
    """
    Analyze a natural language query without generating SQL.

    Returns intent analysis and complexity estimation.
    """
    try:
        generator = get_generator()

        # Parse intent
        intent = generator._parse_intent(query)

        # Estimate complexity
        complexity = generator._estimate_complexity(intent)

        # Extract keywords
        keywords = generator._extract_keywords(query)

        return JSONResponse(content={
            "success": True,
            "query": query,
            "analysis": {
                "intent": intent,
                "complexity": complexity.value,
                "keywords": keywords
            }
        })

    except Exception as e:
        logger.error(f"Query analysis failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"success": False, "error": str(e)}
        )


@router.get("/dialects")
async def get_supported_dialects() -> JSONResponse:
    """
    Get list of supported SQL dialects.
    """
    return JSONResponse(content={
        "dialects": [d.value for d in DatabaseDialect]
    })
