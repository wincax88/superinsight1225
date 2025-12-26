"""
SQL Generator for Text-to-SQL.

Core engine for converting natural language queries to SQL.
"""

import logging
import re
import time
from typing import Dict, Any, List, Optional, Tuple
from uuid import uuid4

from .models import (
    SQLGenerationRequest,
    SQLGenerationResponse,
    QueryPlan,
    SQLValidationResult,
    QueryComplexity,
    DatabaseDialect
)
from .schema_manager import SchemaManager, get_schema_manager
from .llm_adapter import LLMAdapter, get_llm_adapter

logger = logging.getLogger(__name__)


# SQL Keywords for intent parsing
SELECT_KEYWORDS = ["查询", "查找", "获取", "显示", "列出", "select", "find", "get", "show", "list", "fetch"]
AGGREGATE_KEYWORDS = ["统计", "计数", "总数", "平均", "最大", "最小", "求和", "count", "sum", "avg", "max", "min", "total"]
JOIN_KEYWORDS = ["关联", "连接", "join", "with", "和", "及"]
FILTER_KEYWORDS = ["筛选", "过滤", "条件", "where", "filter", "when", "if"]
ORDER_KEYWORDS = ["排序", "排列", "order", "sort", "按"]
GROUP_KEYWORDS = ["分组", "按...分", "group by", "group"]
LIMIT_KEYWORDS = ["前", "top", "limit", "first", "最近"]


class SQLGenerator:
    """
    Core SQL generation engine.

    Features:
    - Intent parsing from natural language
    - Table and column identification
    - SQL clause building
    - Query validation
    """

    def __init__(self,
                 connection_string: Optional[str] = None,
                 schema_manager: Optional[SchemaManager] = None,
                 llm_adapter: Optional[LLMAdapter] = None):
        """
        Initialize SQL Generator.

        Args:
            connection_string: Database connection string
            schema_manager: Schema manager instance
            llm_adapter: LLM adapter instance
        """
        self.connection_string = connection_string
        self._schema_manager = schema_manager or get_schema_manager()
        self._llm_adapter = llm_adapter or get_llm_adapter()

        # Query statistics
        self._query_count = 0
        self._success_count = 0
        self._total_time = 0.0

    async def generate_sql(self, request: SQLGenerationRequest) -> SQLGenerationResponse:
        """
        Generate SQL from natural language request.

        Args:
            request: SQL generation request

        Returns:
            SQL generation response
        """
        start_time = time.time()

        try:
            # Parse query intent
            intent = self._parse_intent(request.query)
            logger.debug(f"Parsed intent: {intent}")

            # Get schema context
            schema_context = ""
            if self.connection_string:
                keywords = self._extract_keywords(request.query)
                schema_context = self._schema_manager.build_schema_context(
                    self.connection_string,
                    keywords
                )

            # Determine query complexity
            complexity = self._estimate_complexity(intent)

            # Generate SQL using LLM
            if complexity in [QueryComplexity.COMPLEX, QueryComplexity.ADVANCED]:
                result = await self._llm_adapter.generate_complex_sql(
                    request.query,
                    schema_context,
                    request.dialect.value
                )
            elif request.include_explanation:
                result = await self._llm_adapter.generate_sql_with_explanation(
                    request.query,
                    schema_context,
                    request.dialect.value
                )
            else:
                result = await self._llm_adapter.generate_sql_simple(
                    request.query,
                    schema_context,
                    request.dialect.value
                )

            if not result.get("success"):
                raise Exception(result.get("error", "SQL generation failed"))

            sql = result.get("sql", "")
            explanation = result.get("explanation", "")

            # Format SQL
            formatted_sql = self._format_sql(sql)

            # Build query plan
            query_plan = self._build_query_plan(sql, intent)

            # Validate if requested
            validation = None
            if request.validate_sql:
                validation = self._validate_sql(sql, request.dialect)

            # Apply max_results limit if specified
            if request.max_results and "LIMIT" not in sql.upper():
                sql = self._add_limit(sql, request.max_results, request.dialect)
                formatted_sql = self._format_sql(sql)

            processing_time = time.time() - start_time
            self._query_count += 1
            self._success_count += 1
            self._total_time += processing_time

            return SQLGenerationResponse(
                request_id=request.id,
                sql=sql,
                formatted_sql=formatted_sql,
                explanation=explanation,
                query_plan=query_plan,
                validation=validation,
                confidence=result.get("confidence", 0.8),
                processing_time=processing_time,
                model_used=result.get("model_used"),
                metadata={
                    "intent": intent,
                    "complexity": complexity.value,
                    "schema_tables_used": query_plan.tables_involved if query_plan else []
                }
            )

        except Exception as e:
            self._query_count += 1
            logger.error(f"SQL generation failed: {e}")

            return SQLGenerationResponse(
                request_id=request.id,
                sql="",
                confidence=0.0,
                processing_time=time.time() - start_time,
                validation=SQLValidationResult(
                    is_valid=False,
                    errors=[str(e)]
                ),
                metadata={"error": str(e)}
            )

    def _parse_intent(self, query: str) -> Dict[str, Any]:
        """
        Parse query intent from natural language.

        Args:
            query: Natural language query

        Returns:
            Intent dictionary
        """
        query_lower = query.lower()

        intent = {
            "action": "select",
            "has_aggregation": False,
            "has_join": False,
            "has_filter": False,
            "has_order": False,
            "has_group": False,
            "has_limit": False,
            "aggregations": [],
            "filters": [],
            "order_direction": None
        }

        # Check for aggregations
        for keyword in AGGREGATE_KEYWORDS:
            if keyword in query_lower:
                intent["has_aggregation"] = True
                intent["aggregations"].append(keyword)

        # Check for joins
        for keyword in JOIN_KEYWORDS:
            if keyword in query_lower:
                intent["has_join"] = True
                break

        # Check for filters
        for keyword in FILTER_KEYWORDS:
            if keyword in query_lower:
                intent["has_filter"] = True
                break

        # Check for ordering
        for keyword in ORDER_KEYWORDS:
            if keyword in query_lower:
                intent["has_order"] = True
                # Determine direction
                if any(d in query_lower for d in ["降序", "desc", "倒序", "从高到低"]):
                    intent["order_direction"] = "DESC"
                elif any(d in query_lower for d in ["升序", "asc", "正序", "从低到高"]):
                    intent["order_direction"] = "ASC"
                break

        # Check for grouping
        for keyword in GROUP_KEYWORDS:
            if keyword in query_lower:
                intent["has_group"] = True
                break

        # Check for limits
        for keyword in LIMIT_KEYWORDS:
            if keyword in query_lower:
                intent["has_limit"] = True
                # Try to extract limit number
                numbers = re.findall(r'\d+', query)
                if numbers:
                    intent["limit_value"] = int(numbers[0])
                break

        return intent

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from query for schema filtering."""
        # Remove common stop words
        stop_words = {"的", "是", "在", "和", "了", "有", "我", "要", "查询", "查找",
                     "the", "is", "in", "and", "of", "a", "an", "to", "for"}

        # Tokenize
        words = re.findall(r'\w+', query.lower())

        # Filter stop words and short words
        keywords = [w for w in words if w not in stop_words and len(w) > 1]

        return keywords

    def _estimate_complexity(self, intent: Dict[str, Any]) -> QueryComplexity:
        """Estimate query complexity from intent."""
        score = 0

        if intent.get("has_join"):
            score += 2
        if intent.get("has_aggregation"):
            score += 1
        if intent.get("has_group"):
            score += 1
        if intent.get("has_filter"):
            score += 1
        if intent.get("has_order"):
            score += 0.5

        # Check for subquery indicators
        subquery_keywords = ["子查询", "嵌套", "subquery", "nested", "in (select"]
        if any(k in str(intent).lower() for k in subquery_keywords):
            score += 3

        if score >= 5:
            return QueryComplexity.ADVANCED
        elif score >= 3:
            return QueryComplexity.COMPLEX
        elif score >= 1:
            return QueryComplexity.MODERATE
        return QueryComplexity.SIMPLE

    def _build_query_plan(self, sql: str, intent: Dict[str, Any]) -> QueryPlan:
        """Build query execution plan from SQL."""
        sql_upper = sql.upper()

        # Extract tables
        tables = self._extract_tables(sql)

        # Extract columns
        columns = self._extract_columns(sql)

        # Extract join conditions
        joins = []
        if "JOIN" in sql_upper:
            join_pattern = r'(?:INNER|LEFT|RIGHT|FULL|CROSS)?\s*JOIN\s+(\w+)\s+(?:AS\s+\w+\s+)?ON\s+([^)]+?)(?=\s+(?:LEFT|RIGHT|INNER|FULL|CROSS|JOIN|WHERE|GROUP|ORDER|LIMIT|$))'
            join_matches = re.findall(join_pattern, sql, re.IGNORECASE)
            for match in join_matches:
                joins.append(f"JOIN {match[0]} ON {match[1]}")

        # Extract where conditions
        filters = []
        where_match = re.search(r'WHERE\s+(.+?)(?=\s+GROUP\s+BY|\s+ORDER\s+BY|\s+LIMIT|\s*;?\s*$)', sql, re.IGNORECASE | re.DOTALL)
        if where_match:
            filters.append(where_match.group(1).strip())

        # Extract aggregations
        aggregations = []
        agg_pattern = r'(COUNT|SUM|AVG|MAX|MIN)\s*\([^)]+\)'
        agg_matches = re.findall(agg_pattern, sql, re.IGNORECASE)
        aggregations.extend(agg_matches)

        # Extract GROUP BY
        group_by = []
        group_match = re.search(r'GROUP\s+BY\s+(.+?)(?=\s+HAVING|\s+ORDER\s+BY|\s+LIMIT|\s*;?\s*$)', sql, re.IGNORECASE)
        if group_match:
            group_by = [g.strip() for g in group_match.group(1).split(",")]

        # Extract ORDER BY
        order_by = []
        order_match = re.search(r'ORDER\s+BY\s+(.+?)(?=\s+LIMIT|\s*;?\s*$)', sql, re.IGNORECASE)
        if order_match:
            order_by = [o.strip() for o in order_match.group(1).split(",")]

        # Check for subqueries and CTEs
        has_subquery = bool(re.search(r'\(\s*SELECT', sql, re.IGNORECASE))
        has_cte = sql_upper.strip().startswith("WITH")
        has_window = bool(re.search(r'OVER\s*\(', sql, re.IGNORECASE))

        plan = QueryPlan(
            tables_involved=tables,
            columns_selected=columns,
            join_conditions=joins,
            filter_conditions=filters,
            aggregations=aggregations,
            group_by=group_by,
            order_by=order_by,
            has_subquery=has_subquery,
            has_cte=has_cte,
            has_window_function=has_window
        )

        plan.complexity = plan.analyze_complexity()

        return plan

    def _extract_tables(self, sql: str) -> List[str]:
        """Extract table names from SQL."""
        tables = set()

        # FROM clause
        from_pattern = r'FROM\s+([a-zA-Z_][\w]*)'
        tables.update(re.findall(from_pattern, sql, re.IGNORECASE))

        # JOIN clauses
        join_pattern = r'JOIN\s+([a-zA-Z_][\w]*)'
        tables.update(re.findall(join_pattern, sql, re.IGNORECASE))

        return list(tables)

    def _extract_columns(self, sql: str) -> List[str]:
        """Extract column names from SQL SELECT clause."""
        columns = []

        # Match SELECT clause
        select_match = re.search(r'SELECT\s+(.+?)\s+FROM', sql, re.IGNORECASE | re.DOTALL)
        if select_match:
            select_clause = select_match.group(1)

            if select_clause.strip() == "*":
                columns = ["*"]
            else:
                # Parse columns (handling functions, aliases, etc.)
                # This is a simplified extraction
                parts = select_clause.split(",")
                for part in parts:
                    # Get alias if present
                    alias_match = re.search(r'(?:AS\s+)?(\w+)\s*$', part.strip(), re.IGNORECASE)
                    if alias_match:
                        columns.append(alias_match.group(1))

        return columns

    def _validate_sql(self, sql: str, dialect: DatabaseDialect) -> SQLValidationResult:
        """Validate SQL query."""
        errors = []
        warnings = []
        safety_issues = []
        optimizations = []

        sql_upper = sql.upper()

        # Basic syntax checks
        if not sql.strip():
            errors.append("SQL query is empty")

        if not any(sql_upper.startswith(kw) for kw in ["SELECT", "WITH"]):
            if any(kw in sql_upper for kw in ["INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", "TRUNCATE"]):
                safety_issues.append("Query contains write operations - only SELECT queries are allowed")
            else:
                errors.append("Query must start with SELECT or WITH")

        # Check for balanced parentheses
        if sql.count("(") != sql.count(")"):
            errors.append("Unbalanced parentheses")

        # Check for unclosed quotes
        single_quotes = sql.count("'")
        if single_quotes % 2 != 0:
            errors.append("Unclosed single quote")

        double_quotes = sql.count('"')
        if double_quotes % 2 != 0:
            errors.append("Unclosed double quote")

        # Safety checks
        dangerous_patterns = [
            (r';\s*DROP', "Potential SQL injection: DROP statement"),
            (r';\s*DELETE', "Potential SQL injection: DELETE statement"),
            (r';\s*UPDATE', "Potential SQL injection: UPDATE statement"),
            (r'--', "SQL comment detected - potential injection"),
            (r'/\*', "SQL block comment detected - potential injection"),
        ]

        for pattern, message in dangerous_patterns:
            if re.search(pattern, sql, re.IGNORECASE):
                safety_issues.append(message)

        # Performance warnings
        if "SELECT *" in sql_upper:
            warnings.append("Using SELECT * may impact performance - consider specifying columns")
            optimizations.append("Replace SELECT * with specific column names")

        if "LIKE '%'" in sql_upper or "LIKE '%" in sql_upper:
            warnings.append("Leading wildcard in LIKE may prevent index usage")
            optimizations.append("Avoid leading wildcards in LIKE patterns if possible")

        if "ORDER BY" in sql_upper and "LIMIT" not in sql_upper:
            warnings.append("ORDER BY without LIMIT may be slow on large tables")
            optimizations.append("Consider adding LIMIT clause")

        is_valid = len(errors) == 0
        is_safe = len(safety_issues) == 0

        return SQLValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            is_safe=is_safe,
            safety_issues=safety_issues,
            optimizations=optimizations
        )

    def _format_sql(self, sql: str) -> str:
        """Format SQL for readability."""
        if not sql:
            return ""

        # Simple formatting - uppercase keywords
        keywords = ["SELECT", "FROM", "WHERE", "JOIN", "LEFT JOIN", "RIGHT JOIN",
                   "INNER JOIN", "OUTER JOIN", "ON", "AND", "OR", "ORDER BY",
                   "GROUP BY", "HAVING", "LIMIT", "OFFSET", "WITH", "AS",
                   "UNION", "INTERSECT", "EXCEPT", "INSERT", "UPDATE", "DELETE"]

        formatted = sql
        for kw in keywords:
            pattern = rf'\b{kw}\b'
            formatted = re.sub(pattern, kw, formatted, flags=re.IGNORECASE)

        # Add newlines before major clauses
        major_clauses = ["FROM", "WHERE", "JOIN", "LEFT JOIN", "RIGHT JOIN",
                        "ORDER BY", "GROUP BY", "HAVING", "LIMIT"]
        for clause in major_clauses:
            formatted = re.sub(rf'\s+{clause}\b', f'\n{clause}', formatted, flags=re.IGNORECASE)

        return formatted.strip()

    def _add_limit(self, sql: str, limit: int, dialect: DatabaseDialect) -> str:
        """Add LIMIT clause to SQL."""
        sql = sql.rstrip().rstrip(";")

        if dialect == DatabaseDialect.ORACLE:
            # Oracle uses FETCH FIRST or ROWNUM
            if "FETCH" not in sql.upper():
                sql += f" FETCH FIRST {limit} ROWS ONLY"
        else:
            # MySQL, PostgreSQL, SQLite use LIMIT
            sql += f" LIMIT {limit}"

        return sql + ";"

    def set_connection(self, connection_string: str) -> None:
        """Set database connection string."""
        self.connection_string = connection_string

    def get_statistics(self) -> Dict[str, Any]:
        """Get generation statistics."""
        return {
            "query_count": self._query_count,
            "success_count": self._success_count,
            "success_rate": self._success_count / max(1, self._query_count),
            "total_time": self._total_time,
            "average_time": self._total_time / max(1, self._query_count)
        }


# Global instance
_sql_generator: Optional[SQLGenerator] = None


def get_sql_generator(connection_string: Optional[str] = None) -> SQLGenerator:
    """Get or create global SQLGenerator instance."""
    global _sql_generator

    if _sql_generator is None:
        _sql_generator = SQLGenerator(connection_string)
    elif connection_string:
        _sql_generator.set_connection(connection_string)

    return _sql_generator
