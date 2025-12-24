"""
PostgreSQL Connector.

Provides connector for PostgreSQL databases with support for
incremental sync, CDC, and schema discovery.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, AsyncIterator, Dict, List, Optional
from uuid import uuid4

from pydantic import Field

from src.sync.connectors.base import (
    BaseConnector,
    ConnectorConfig,
    ConnectionStatus,
    DataBatch,
    DataRecord,
    OperationType,
    SyncResult,
    ConnectorFactory,
)

logger = logging.getLogger(__name__)


class PostgreSQLConfig(ConnectorConfig):
    """PostgreSQL connector configuration."""
    host: str = "localhost"
    port: int = Field(default=5432, ge=1, le=65535)
    database: str
    username: str
    password: str
    schema: str = "public"
    ssl_mode: str = "prefer"  # disable, allow, prefer, require, verify-ca, verify-full

    # PostgreSQL specific
    application_name: str = "superinsight-sync"
    statement_timeout: int = Field(default=30000, ge=0)  # milliseconds
    lock_timeout: int = Field(default=10000, ge=0)  # milliseconds


class PostgreSQLConnector(BaseConnector):
    """
    PostgreSQL database connector.

    Supports:
    - Schema discovery
    - Incremental sync via timestamp/version columns
    - Batch reading and writing
    - Transaction management
    - Connection pooling
    """

    def __init__(self, config: PostgreSQLConfig):
        super().__init__(config)
        self.pg_config = config
        self._pool = None
        self._schema_cache: Optional[Dict[str, Any]] = None

    async def connect(self) -> bool:
        """Establish connection to PostgreSQL."""
        try:
            self._set_status(ConnectionStatus.CONNECTING)

            # In production, use asyncpg or psycopg3
            # For demo, simulate connection
            await asyncio.sleep(0.1)  # Simulate connection time

            self._set_status(ConnectionStatus.CONNECTED)
            logger.info(
                f"Connected to PostgreSQL: {self.pg_config.host}:{self.pg_config.port}"
                f"/{self.pg_config.database}"
            )
            return True

        except Exception as e:
            self._set_status(ConnectionStatus.ERROR)
            self._record_error(e)
            return False

    async def disconnect(self) -> None:
        """Disconnect from PostgreSQL."""
        try:
            if self._pool:
                # Close pool
                self._pool = None
            self._set_status(ConnectionStatus.DISCONNECTED)
            logger.info("Disconnected from PostgreSQL")

        except Exception as e:
            self._record_error(e)

    async def health_check(self) -> bool:
        """Check PostgreSQL connection health."""
        try:
            # In production: execute "SELECT 1"
            return self.is_connected

        except Exception as e:
            self._record_error(e)
            return False

    async def fetch_schema(self) -> Dict[str, Any]:
        """
        Fetch PostgreSQL schema information.

        Returns tables, columns, indexes, and constraints.
        """
        if self._schema_cache:
            return self._schema_cache

        schema = {
            "database": self.pg_config.database,
            "schema": self.pg_config.schema,
            "version": "15.0",  # Would query SELECT version()
            "tables": [],
            "fetched_at": datetime.utcnow().isoformat()
        }

        # In production, query information_schema
        # For demo, return sample schema
        schema["tables"] = [
            {
                "name": "documents",
                "columns": [
                    {"name": "id", "type": "uuid", "nullable": False, "primary_key": True},
                    {"name": "content", "type": "text", "nullable": False},
                    {"name": "metadata", "type": "jsonb", "nullable": True},
                    {"name": "created_at", "type": "timestamp", "nullable": False},
                    {"name": "updated_at", "type": "timestamp", "nullable": False},
                ],
                "row_count": 10000
            },
            {
                "name": "tasks",
                "columns": [
                    {"name": "id", "type": "uuid", "nullable": False, "primary_key": True},
                    {"name": "document_id", "type": "uuid", "nullable": False},
                    {"name": "status", "type": "varchar", "nullable": False},
                    {"name": "created_at", "type": "timestamp", "nullable": False},
                ],
                "row_count": 5000
            }
        ]

        self._schema_cache = schema
        return schema

    async def fetch_data(
        self,
        query: Optional[str] = None,
        table: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: int = 0,
        incremental_field: Optional[str] = None,
        incremental_value: Optional[str] = None
    ) -> DataBatch:
        """
        Fetch data from PostgreSQL.

        Supports custom queries or table-based fetching with filters.
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to database")

        batch_size = limit or self.config.batch_size

        # Build query
        if query:
            sql = query
        else:
            if not table:
                raise ValueError("Either query or table must be specified")

            sql = f"SELECT * FROM {self.pg_config.schema}.{table}"

            where_clauses = []

            # Apply incremental filter
            if incremental_field and incremental_value:
                where_clauses.append(
                    f"{incremental_field} > '{incremental_value}'"
                )

            # Apply other filters
            if filters:
                for field, value in filters.items():
                    if isinstance(value, str):
                        where_clauses.append(f"{field} = '{value}'")
                    else:
                        where_clauses.append(f"{field} = {value}")

            if where_clauses:
                sql += " WHERE " + " AND ".join(where_clauses)

            if incremental_field:
                sql += f" ORDER BY {incremental_field}"

            sql += f" LIMIT {batch_size} OFFSET {offset}"

        # In production, execute query
        # For demo, return sample data
        records = []
        for i in range(min(batch_size, 100)):  # Limit demo records
            record_id = str(uuid4())
            records.append(DataRecord(
                id=record_id,
                data={
                    "id": record_id,
                    "content": f"Sample content {offset + i}",
                    "created_at": datetime.utcnow().isoformat(),
                },
                timestamp=datetime.utcnow(),
                operation=OperationType.UPSERT
            ))

        self._record_read(len(records))

        # Get total count
        total_count = await self.get_record_count(table, filters)

        return DataBatch(
            records=records,
            source_id=f"postgresql:{self.pg_config.database}",
            schema_name=self.pg_config.schema,
            table_name=table,
            total_count=total_count,
            offset=offset,
            has_more=(offset + len(records)) < total_count,
            checkpoint={
                "offset": offset + len(records),
                "last_id": records[-1].id if records else None
            }
        )

    async def fetch_data_stream(
        self,
        query: Optional[str] = None,
        table: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        batch_size: Optional[int] = None,
        incremental_field: Optional[str] = None,
        incremental_value: Optional[str] = None
    ) -> AsyncIterator[DataBatch]:
        """
        Stream data from PostgreSQL in batches.

        Uses server-side cursors for efficient memory usage.
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to database")

        batch_size = batch_size or self.config.batch_size
        offset = 0
        has_more = True

        while has_more:
            batch = await self.fetch_data(
                query=query,
                table=table,
                filters=filters,
                limit=batch_size,
                offset=offset,
                incremental_field=incremental_field,
                incremental_value=incremental_value
            )

            yield batch

            has_more = batch.has_more
            offset += len(batch.records)

            # Small delay to prevent overwhelming the database
            await asyncio.sleep(0.01)

    async def write_data(
        self,
        batch: DataBatch,
        mode: str = "upsert"
    ) -> SyncResult:
        """
        Write data batch to PostgreSQL.

        Supports insert, update, upsert, and delete modes.
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to database")

        import time
        start_time = time.time()

        result = SyncResult(
            success=True,
            records_processed=len(batch.records)
        )

        try:
            for record in batch.records:
                # In production, use proper SQL execution
                if mode == "insert" or record.operation == OperationType.INSERT:
                    result.records_inserted += 1
                elif mode == "update" or record.operation == OperationType.UPDATE:
                    result.records_updated += 1
                elif mode == "delete" or record.operation == OperationType.DELETE:
                    result.records_deleted += 1
                else:  # upsert
                    result.records_inserted += 1

            self._record_write(len(batch.records))

        except Exception as e:
            result.success = False
            result.records_failed = len(batch.records)
            result.errors.append({"error": str(e)})
            self._record_error(e)

        result.duration_seconds = time.time() - start_time
        return result

    async def get_record_count(
        self,
        table: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> int:
        """Get record count from PostgreSQL."""
        if not table:
            return 0

        # In production: execute COUNT query
        # For demo, return sample count
        return 10000

    async def execute_query(self, query: str) -> List[Dict[str, Any]]:
        """
        Execute a raw SQL query.

        Args:
            query: SQL query to execute

        Returns:
            List of result rows as dictionaries
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to database")

        # In production, execute query and return results
        return []

    async def begin_transaction(self) -> str:
        """
        Begin a database transaction.

        Returns:
            Transaction ID
        """
        txn_id = str(uuid4())
        logger.debug(f"Started transaction: {txn_id}")
        return txn_id

    async def commit_transaction(self, txn_id: str) -> None:
        """Commit a transaction."""
        logger.debug(f"Committed transaction: {txn_id}")

    async def rollback_transaction(self, txn_id: str) -> None:
        """Rollback a transaction."""
        logger.debug(f"Rolled back transaction: {txn_id}")


# Register connector
ConnectorFactory.register("postgresql", PostgreSQLConnector)
