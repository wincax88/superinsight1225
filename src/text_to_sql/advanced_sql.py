"""
Advanced SQL Generator for Text-to-SQL.

Handles complex queries including subqueries, CTEs, and window functions.
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum

from .models import (
    SQLGenerationRequest,
    SQLGenerationResponse,
    QueryPlan,
    QueryComplexity,
    DatabaseDialect
)
from .sql_generator import SQLGenerator
from .llm_adapter import LLMAdapter, get_llm_adapter

logger = logging.getLogger(__name__)


class WindowFunctionType(str, Enum):
    """Window function types."""
    ROW_NUMBER = "ROW_NUMBER"
    RANK = "RANK"
    DENSE_RANK = "DENSE_RANK"
    NTILE = "NTILE"
    LAG = "LAG"
    LEAD = "LEAD"
    FIRST_VALUE = "FIRST_VALUE"
    LAST_VALUE = "LAST_VALUE"
    SUM = "SUM"
    AVG = "AVG"
    COUNT = "COUNT"
    MIN = "MIN"
    MAX = "MAX"


class AdvancedSQLGenerator(SQLGenerator):
    """
    Advanced SQL generator for complex queries.

    Extends SQLGenerator with:
    - Subquery generation
    - CTE (Common Table Expressions) generation
    - Window function support
    - Query optimization
    """

    # Prompt templates for advanced queries
    SUBQUERY_PROMPT = """Generate a SQL query with subqueries for:
{user_query}

Schema:
{schema_context}

Requirements:
1. Use subqueries where appropriate
2. Ensure correct correlation between outer and inner queries
3. Use {dialect} syntax

SQL:"""

    CTE_PROMPT = """Generate a SQL query using CTEs (WITH clause) for:
{user_query}

Schema:
{schema_context}

Requirements:
1. Use CTEs for better readability
2. Name CTEs descriptively
3. Use {dialect} syntax

SQL:"""

    WINDOW_PROMPT = """Generate a SQL query with window functions for:
{user_query}

Schema:
{schema_context}

Requirements:
1. Use appropriate window functions (ROW_NUMBER, RANK, LAG, LEAD, etc.)
2. Define correct PARTITION BY and ORDER BY clauses
3. Use {dialect} syntax

SQL:"""

    async def generate_subquery(self,
                               user_query: str,
                               schema_context: str,
                               dialect: str = "postgresql") -> Dict[str, Any]:
        """
        Generate SQL with subqueries.

        Args:
            user_query: Natural language query
            schema_context: Database schema context
            dialect: SQL dialect

        Returns:
            Generation result with subquery SQL
        """
        prompt = self.SUBQUERY_PROMPT.format(
            user_query=user_query,
            schema_context=schema_context,
            dialect=dialect
        )

        result = await self._llm_adapter.generate_sql(prompt)

        if result.get("success"):
            sql = result.get("sql", "")
            # Validate subquery structure
            if not self._validate_subquery_structure(sql):
                result["warnings"] = result.get("warnings", [])
                result["warnings"].append("Generated SQL may not contain proper subqueries")

        return result

    async def generate_cte(self,
                          user_query: str,
                          schema_context: str,
                          dialect: str = "postgresql") -> Dict[str, Any]:
        """
        Generate SQL with CTEs.

        Args:
            user_query: Natural language query
            schema_context: Database schema context
            dialect: SQL dialect

        Returns:
            Generation result with CTE SQL
        """
        prompt = self.CTE_PROMPT.format(
            user_query=user_query,
            schema_context=schema_context,
            dialect=dialect
        )

        result = await self._llm_adapter.generate_sql(prompt)

        if result.get("success"):
            sql = result.get("sql", "")
            # Validate CTE structure
            if not sql.upper().strip().startswith("WITH"):
                # Try to convert to CTE format
                sql = self._convert_to_cte(sql)
                result["sql"] = sql

        return result

    async def generate_window_function(self,
                                       user_query: str,
                                       schema_context: str,
                                       function_type: Optional[WindowFunctionType] = None,
                                       dialect: str = "postgresql") -> Dict[str, Any]:
        """
        Generate SQL with window functions.

        Args:
            user_query: Natural language query
            schema_context: Database schema context
            function_type: Preferred window function type
            dialect: SQL dialect

        Returns:
            Generation result with window function SQL
        """
        # Enhance prompt with function type hint if provided
        enhanced_query = user_query
        if function_type:
            enhanced_query = f"{user_query} (use {function_type.value} window function)"

        prompt = self.WINDOW_PROMPT.format(
            user_query=enhanced_query,
            schema_context=schema_context,
            dialect=dialect
        )

        result = await self._llm_adapter.generate_sql(prompt)

        if result.get("success"):
            sql = result.get("sql", "")
            # Validate window function structure
            if "OVER" not in sql.upper():
                result["warnings"] = result.get("warnings", [])
                result["warnings"].append("Generated SQL may not contain window functions")

        return result

    def optimize_query(self, sql: str, dialect: DatabaseDialect) -> Tuple[str, List[str]]:
        """
        Optimize SQL query.

        Args:
            sql: SQL query to optimize
            dialect: SQL dialect

        Returns:
            Tuple of (optimized SQL, list of optimizations applied)
        """
        optimizations = []
        optimized = sql

        # 1. Replace SELECT * with column list if possible
        if "SELECT *" in sql.upper():
            optimizations.append("Consider replacing SELECT * with specific columns")

        # 2. Check for missing indexes hints
        if "JOIN" in sql.upper() and "INDEX" not in sql.upper():
            optimizations.append("Consider adding index hints for large table joins")

        # 3. Convert correlated subqueries to JOINs where possible
        if self._has_correlated_subquery(sql):
            converted, was_converted = self._convert_correlated_to_join(sql)
            if was_converted:
                optimized = converted
                optimizations.append("Converted correlated subquery to JOIN for better performance")

        # 4. Simplify redundant conditions
        optimized = self._simplify_conditions(optimized)
        if optimized != sql:
            optimizations.append("Simplified redundant conditions")

        # 5. Add query hints for specific dialects
        if dialect == DatabaseDialect.MYSQL:
            if "FORCE INDEX" not in optimized.upper() and "USE INDEX" not in optimized.upper():
                optimizations.append("Consider using FORCE INDEX or USE INDEX for large tables")

        # 6. Check for potential N+1 issues
        if self._detect_n_plus_one_pattern(sql):
            optimizations.append("Warning: Potential N+1 query pattern detected")

        return optimized, optimizations

    def build_cte(self,
                  cte_name: str,
                  cte_query: str,
                  main_query: str,
                  recursive: bool = False) -> str:
        """
        Build a CTE query.

        Args:
            cte_name: Name for the CTE
            cte_query: The CTE query
            main_query: The main query using the CTE
            recursive: Whether the CTE is recursive

        Returns:
            Complete CTE SQL
        """
        recursive_keyword = "RECURSIVE " if recursive else ""
        return f"WITH {recursive_keyword}{cte_name} AS (\n{cte_query}\n)\n{main_query}"

    def build_window_function(self,
                              function_type: WindowFunctionType,
                              column: str,
                              partition_by: Optional[List[str]] = None,
                              order_by: Optional[List[str]] = None,
                              frame: Optional[str] = None) -> str:
        """
        Build a window function expression.

        Args:
            function_type: Type of window function
            column: Column to apply function to
            partition_by: PARTITION BY columns
            order_by: ORDER BY columns
            frame: Window frame specification

        Returns:
            Window function SQL expression
        """
        # Build function call
        if function_type in [WindowFunctionType.ROW_NUMBER, WindowFunctionType.RANK,
                            WindowFunctionType.DENSE_RANK]:
            func_call = f"{function_type.value}()"
        elif function_type == WindowFunctionType.NTILE:
            func_call = f"NTILE(4)"  # Default to quartiles
        elif function_type in [WindowFunctionType.LAG, WindowFunctionType.LEAD]:
            func_call = f"{function_type.value}({column}, 1)"
        else:
            func_call = f"{function_type.value}({column})"

        # Build OVER clause
        over_parts = []

        if partition_by:
            over_parts.append(f"PARTITION BY {', '.join(partition_by)}")

        if order_by:
            over_parts.append(f"ORDER BY {', '.join(order_by)}")

        if frame:
            over_parts.append(frame)

        over_clause = " ".join(over_parts)

        return f"{func_call} OVER ({over_clause})"

    def build_subquery(self,
                      subquery: str,
                      alias: str,
                      subquery_type: str = "inline") -> str:
        """
        Build a subquery expression.

        Args:
            subquery: The subquery SQL
            alias: Alias for the subquery
            subquery_type: Type: 'inline', 'exists', 'in', 'scalar'

        Returns:
            Subquery SQL expression
        """
        subquery = subquery.rstrip(";").strip()

        if subquery_type == "inline":
            return f"(\n{subquery}\n) AS {alias}"
        elif subquery_type == "exists":
            return f"EXISTS (\n{subquery}\n)"
        elif subquery_type == "in":
            return f"IN (\n{subquery}\n)"
        elif subquery_type == "scalar":
            return f"(\n{subquery}\n)"
        else:
            return f"(\n{subquery}\n) AS {alias}"

    def _validate_subquery_structure(self, sql: str) -> bool:
        """Check if SQL contains valid subquery structure."""
        # Look for subquery patterns
        patterns = [
            r'\(\s*SELECT',  # Standard subquery
            r'IN\s*\(\s*SELECT',  # IN subquery
            r'EXISTS\s*\(\s*SELECT',  # EXISTS subquery
            r'FROM\s*\(\s*SELECT',  # Inline view
        ]

        for pattern in patterns:
            if re.search(pattern, sql, re.IGNORECASE):
                return True

        return False

    def _convert_to_cte(self, sql: str) -> str:
        """Attempt to convert SQL with subqueries to CTE format."""
        # Find inline views (FROM (SELECT ...))
        pattern = r'FROM\s*\(\s*(SELECT[^)]+)\s*\)\s+(?:AS\s+)?(\w+)'
        matches = re.findall(pattern, sql, re.IGNORECASE | re.DOTALL)

        if not matches:
            return sql

        ctes = []
        modified_sql = sql

        for i, (subquery, alias) in enumerate(matches):
            cte_name = alias if alias else f"cte_{i+1}"
            ctes.append(f"{cte_name} AS (\n{subquery.strip()}\n)")

            # Replace subquery with CTE reference
            old_pattern = rf'FROM\s*\(\s*{re.escape(subquery)}\s*\)\s+(?:AS\s+)?{re.escape(alias)}'
            modified_sql = re.sub(old_pattern, f"FROM {cte_name}", modified_sql, flags=re.IGNORECASE | re.DOTALL)

        if ctes:
            return f"WITH {', '.join(ctes)}\n{modified_sql}"

        return sql

    def _has_correlated_subquery(self, sql: str) -> bool:
        """Check if SQL contains correlated subqueries."""
        # Look for outer table references in subqueries
        # This is a simplified check
        pattern = r'\(\s*SELECT[^)]+WHERE[^)]+\.\w+\s*=\s*\w+\.\w+[^)]*\)'
        return bool(re.search(pattern, sql, re.IGNORECASE))

    def _convert_correlated_to_join(self, sql: str) -> Tuple[str, bool]:
        """Attempt to convert correlated subquery to JOIN."""
        # This is a complex operation - simplified version
        # In production, this would need more sophisticated parsing
        return sql, False

    def _simplify_conditions(self, sql: str) -> str:
        """Simplify redundant conditions in SQL."""
        # Remove duplicate conditions
        simplified = sql

        # Pattern: WHERE a = b AND a = b
        simplified = re.sub(
            r'(\w+\s*=\s*\w+)\s+AND\s+\1',
            r'\1',
            simplified,
            flags=re.IGNORECASE
        )

        # Pattern: 1 = 1 AND
        simplified = re.sub(
            r'1\s*=\s*1\s+AND\s+',
            '',
            simplified,
            flags=re.IGNORECASE
        )

        return simplified

    def _detect_n_plus_one_pattern(self, sql: str) -> bool:
        """Detect potential N+1 query patterns."""
        # Look for patterns that might indicate N+1
        # This is a heuristic check
        patterns = [
            r'WHERE\s+\w+\.id\s*=\s*\?',  # Parameterized single ID lookup
            r'WHERE\s+\w+_id\s*=\s*\d+',  # Direct ID lookup
        ]

        for pattern in patterns:
            if re.search(pattern, sql, re.IGNORECASE):
                return True

        return False

    def analyze_query_complexity(self, sql: str) -> Dict[str, Any]:
        """
        Analyze SQL query complexity in detail.

        Args:
            sql: SQL query to analyze

        Returns:
            Complexity analysis
        """
        analysis = {
            "complexity_score": 0,
            "complexity_level": QueryComplexity.SIMPLE.value,
            "features": {
                "tables": 0,
                "joins": 0,
                "subqueries": 0,
                "ctes": 0,
                "window_functions": 0,
                "aggregations": 0,
                "conditions": 0,
            },
            "recommendations": []
        }

        sql_upper = sql.upper()

        # Count tables
        from_count = len(re.findall(r'\bFROM\b', sql_upper))
        join_count = len(re.findall(r'\bJOIN\b', sql_upper))
        analysis["features"]["tables"] = from_count + join_count
        analysis["features"]["joins"] = join_count

        # Count subqueries
        subquery_count = len(re.findall(r'\(\s*SELECT', sql_upper))
        analysis["features"]["subqueries"] = subquery_count

        # Count CTEs
        cte_count = len(re.findall(r'\bWITH\b.*?\bAS\s*\(', sql_upper))
        analysis["features"]["ctes"] = cte_count

        # Count window functions
        window_count = len(re.findall(r'\bOVER\s*\(', sql_upper))
        analysis["features"]["window_functions"] = window_count

        # Count aggregations
        agg_count = len(re.findall(r'\b(COUNT|SUM|AVG|MAX|MIN)\s*\(', sql_upper))
        analysis["features"]["aggregations"] = agg_count

        # Count conditions
        condition_count = len(re.findall(r'\b(AND|OR)\b', sql_upper))
        analysis["features"]["conditions"] = condition_count

        # Calculate complexity score
        score = (
            analysis["features"]["tables"] * 1 +
            analysis["features"]["joins"] * 2 +
            analysis["features"]["subqueries"] * 3 +
            analysis["features"]["ctes"] * 2 +
            analysis["features"]["window_functions"] * 2 +
            analysis["features"]["aggregations"] * 1 +
            analysis["features"]["conditions"] * 0.5
        )

        analysis["complexity_score"] = score

        # Determine complexity level
        if score >= 15:
            analysis["complexity_level"] = QueryComplexity.ADVANCED.value
        elif score >= 8:
            analysis["complexity_level"] = QueryComplexity.COMPLEX.value
        elif score >= 3:
            analysis["complexity_level"] = QueryComplexity.MODERATE.value
        else:
            analysis["complexity_level"] = QueryComplexity.SIMPLE.value

        # Add recommendations
        if analysis["features"]["subqueries"] > 2:
            analysis["recommendations"].append("Consider using CTEs to improve readability")

        if analysis["features"]["joins"] > 3:
            analysis["recommendations"].append("Ensure proper indexes exist for join columns")

        if analysis["features"]["conditions"] > 5:
            analysis["recommendations"].append("Complex WHERE clause - verify logic correctness")

        return analysis


# Global instance
_advanced_generator: Optional[AdvancedSQLGenerator] = None


def get_advanced_sql_generator(connection_string: Optional[str] = None) -> AdvancedSQLGenerator:
    """Get or create global AdvancedSQLGenerator instance."""
    global _advanced_generator

    if _advanced_generator is None:
        _advanced_generator = AdvancedSQLGenerator(connection_string)
    elif connection_string:
        _advanced_generator.set_connection(connection_string)

    return _advanced_generator
