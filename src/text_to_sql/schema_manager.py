"""
Database Schema Manager for Text-to-SQL.

Manages database metadata, table relationships, and schema context.
"""

import logging
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
import hashlib
import json

from sqlalchemy import create_engine, text, MetaData, Table, inspect
from sqlalchemy.engine import Engine

from .models import (
    TableInfo,
    ColumnInfo,
    SchemaContext,
    DatabaseDialect
)

logger = logging.getLogger(__name__)


class SchemaManager:
    """
    Manages database schema information for SQL generation.

    Features:
    - Schema loading and caching
    - Table relationship detection
    - Semantic column mapping
    - Business term dictionary
    """

    def __init__(self, cache_ttl: int = 3600):
        """
        Initialize SchemaManager.

        Args:
            cache_ttl: Cache time-to-live in seconds (default: 1 hour)
        """
        self.cache_ttl = cache_ttl
        self._schema_cache: Dict[str, Dict[str, Any]] = {}
        self._relationship_cache: Dict[str, Dict[str, List[str]]] = {}
        self._business_terms: Dict[str, str] = self._load_default_business_terms()
        self._engines: Dict[str, Engine] = {}

    def _load_default_business_terms(self) -> Dict[str, str]:
        """Load default business term mappings."""
        return {
            # Common Chinese-English mappings
            "用户": "users",
            "订单": "orders",
            "产品": "products",
            "客户": "customers",
            "销售": "sales",
            "金额": "amount",
            "数量": "quantity",
            "日期": "date",
            "时间": "time",
            "状态": "status",
            "类型": "type",
            "名称": "name",
            "价格": "price",
            "总计": "total",
            "平均": "avg",
            "最大": "max",
            "最小": "min",
            "统计": "count",
            # Common abbreviations
            "qty": "quantity",
            "amt": "amount",
            "dt": "date",
            "tm": "time",
        }

    def add_business_term(self, term: str, mapping: str) -> None:
        """Add a business term mapping."""
        self._business_terms[term.lower()] = mapping.lower()

    def get_engine(self, connection_string: str) -> Engine:
        """Get or create database engine."""
        engine_key = hashlib.md5(connection_string.encode()).hexdigest()

        if engine_key not in self._engines:
            self._engines[engine_key] = create_engine(
                connection_string,
                pool_pre_ping=True,
                pool_recycle=3600
            )

        return self._engines[engine_key]

    def load_schema(self, connection_string: str,
                   schema_name: Optional[str] = None,
                   include_tables: Optional[List[str]] = None,
                   exclude_tables: Optional[List[str]] = None) -> SchemaContext:
        """
        Load database schema.

        Args:
            connection_string: Database connection string
            schema_name: Schema to load (default: all)
            include_tables: Tables to include (default: all)
            exclude_tables: Tables to exclude (default: none)

        Returns:
            SchemaContext with loaded schema information
        """
        cache_key = self._get_cache_key(connection_string, schema_name)

        # Check cache
        if cache_key in self._schema_cache:
            cached = self._schema_cache[cache_key]
            if datetime.now() - cached["timestamp"] < timedelta(seconds=self.cache_ttl):
                logger.debug(f"Using cached schema for {cache_key[:8]}...")
                return cached["context"]

        try:
            engine = self.get_engine(connection_string)
            inspector = inspect(engine)

            tables: List[TableInfo] = []
            exclude_set = set(exclude_tables or [])
            include_set = set(include_tables) if include_tables else None

            # Get table names
            table_names = inspector.get_table_names(schema=schema_name)

            for table_name in table_names:
                # Filter tables
                if table_name in exclude_set:
                    continue
                if include_set and table_name not in include_set:
                    continue

                table_info = self._load_table_info(inspector, table_name, schema_name)
                if table_info:
                    tables.append(table_info)

            # Detect relationships
            relationships = self._detect_relationships(tables)

            # Build context
            context = SchemaContext(
                tables=tables,
                relationships=relationships,
                business_terms=self._business_terms
            )

            # Cache result
            self._schema_cache[cache_key] = {
                "context": context,
                "timestamp": datetime.now()
            }

            logger.info(f"Loaded schema with {len(tables)} tables")
            return context

        except Exception as e:
            logger.error(f"Failed to load schema: {e}")
            raise

    def _load_table_info(self, inspector, table_name: str,
                        schema_name: Optional[str]) -> Optional[TableInfo]:
        """Load information for a single table."""
        try:
            # Get columns
            columns = []
            pk_columns = inspector.get_pk_constraint(table_name, schema=schema_name)
            pk_names = set(pk_columns.get("constrained_columns", []))

            # Get foreign keys
            fk_info = inspector.get_foreign_keys(table_name, schema=schema_name)
            fk_map = {}
            for fk in fk_info:
                for i, col in enumerate(fk.get("constrained_columns", [])):
                    ref_table = fk.get("referred_table", "")
                    ref_cols = fk.get("referred_columns", [])
                    if i < len(ref_cols):
                        fk_map[col] = f"{ref_table}.{ref_cols[i]}"

            for col in inspector.get_columns(table_name, schema=schema_name):
                column_info = ColumnInfo(
                    name=col["name"],
                    data_type=str(col["type"]),
                    nullable=col.get("nullable", True),
                    primary_key=col["name"] in pk_names,
                    foreign_key=fk_map.get(col["name"]),
                    default_value=str(col.get("default")) if col.get("default") else None,
                    description=col.get("comment")
                )
                columns.append(column_info)

            # Get row count estimate
            row_count = self._estimate_row_count(inspector, table_name, schema_name)

            return TableInfo(
                name=table_name,
                schema_name=schema_name,
                columns=columns,
                primary_keys=list(pk_names),
                foreign_keys=fk_map,
                row_count=row_count
            )

        except Exception as e:
            logger.warning(f"Failed to load table {table_name}: {e}")
            return None

    def _estimate_row_count(self, inspector, table_name: str,
                           schema_name: Optional[str]) -> Optional[int]:
        """Estimate row count for a table."""
        try:
            engine = inspector.bind
            with engine.connect() as conn:
                # Use database-specific methods for estimation
                dialect = engine.dialect.name

                if dialect == "postgresql":
                    query = text(f"""
                        SELECT reltuples::bigint
                        FROM pg_class
                        WHERE relname = :table_name
                    """)
                    result = conn.execute(query, {"table_name": table_name})
                elif dialect == "mysql":
                    query = text(f"""
                        SELECT table_rows
                        FROM information_schema.tables
                        WHERE table_name = :table_name
                    """)
                    result = conn.execute(query, {"table_name": table_name})
                else:
                    return None

                row = result.fetchone()
                return int(row[0]) if row else None

        except Exception:
            return None

    def _detect_relationships(self, tables: List[TableInfo]) -> Dict[str, List[str]]:
        """Detect relationships between tables."""
        relationships: Dict[str, List[str]] = {}

        for table in tables:
            related = []

            # Check foreign keys
            for col, ref in table.foreign_keys.items():
                ref_table = ref.split(".")[0]
                if ref_table not in related:
                    related.append(ref_table)

            # Check column naming patterns (e.g., user_id -> users)
            for col in table.columns:
                if col.name.endswith("_id"):
                    potential_table = col.name[:-3] + "s"  # Simple pluralization
                    for t in tables:
                        if t.name.lower() == potential_table.lower():
                            if t.name not in related:
                                related.append(t.name)

            if related:
                relationships[table.name] = related

        return relationships

    def get_table_info(self, connection_string: str,
                      table_name: str) -> Optional[TableInfo]:
        """Get information for a specific table."""
        context = self.load_schema(connection_string, include_tables=[table_name])
        return context.tables[0] if context.tables else None

    def get_column_info(self, connection_string: str,
                       table_name: str,
                       column_name: str) -> Optional[ColumnInfo]:
        """Get information for a specific column."""
        table = self.get_table_info(connection_string, table_name)
        if table:
            return table.get_column(column_name)
        return None

    def find_related_tables(self, connection_string: str,
                           table_name: str) -> List[str]:
        """Find tables related to the given table."""
        context = self.load_schema(connection_string)

        related = set()

        # Direct relationships
        if table_name in context.relationships:
            related.update(context.relationships[table_name])

        # Reverse relationships
        for t, refs in context.relationships.items():
            if table_name in refs:
                related.add(t)

        return list(related)

    def build_schema_context(self, connection_string: str,
                            query_keywords: Optional[List[str]] = None) -> str:
        """
        Build schema context string for LLM prompt.

        Args:
            connection_string: Database connection string
            query_keywords: Keywords from user query for filtering

        Returns:
            Schema context string for prompt
        """
        context = self.load_schema(connection_string)

        if query_keywords:
            # Filter to relevant tables
            relevant_tables = context.get_relevant_tables(query_keywords)
            if relevant_tables:
                filtered_context = SchemaContext(
                    tables=relevant_tables,
                    relationships={
                        k: v for k, v in context.relationships.items()
                        if k in [t.name for t in relevant_tables]
                    },
                    business_terms=context.business_terms
                )
                return filtered_context.to_prompt_context()

        return context.to_prompt_context()

    def map_business_term(self, term: str) -> Optional[str]:
        """Map a business term to technical term."""
        return self._business_terms.get(term.lower())

    def expand_query_terms(self, query: str) -> str:
        """Expand business terms in a query to technical terms."""
        expanded = query
        for term, mapping in self._business_terms.items():
            if term in query.lower():
                # Add technical term alongside business term
                expanded = expanded.replace(term, f"{term}({mapping})")
        return expanded

    def _get_cache_key(self, connection_string: str,
                      schema_name: Optional[str]) -> str:
        """Generate cache key for schema."""
        key_data = f"{connection_string}:{schema_name or 'default'}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def clear_cache(self, connection_string: Optional[str] = None) -> None:
        """Clear schema cache."""
        if connection_string:
            cache_key = self._get_cache_key(connection_string, None)
            self._schema_cache.pop(cache_key, None)
        else:
            self._schema_cache.clear()
        logger.info("Schema cache cleared")

    def close(self) -> None:
        """Close all database connections."""
        for engine in self._engines.values():
            engine.dispose()
        self._engines.clear()
        logger.info("Schema manager connections closed")


# Global instance
_schema_manager: Optional[SchemaManager] = None


def get_schema_manager() -> SchemaManager:
    """Get or create global SchemaManager instance."""
    global _schema_manager
    if _schema_manager is None:
        _schema_manager = SchemaManager()
    return _schema_manager
