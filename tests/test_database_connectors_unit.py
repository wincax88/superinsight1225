"""
Unit tests for database connectors (MySQL and PostgreSQL).

Tests:
- Task 2.2: Database Connector Tests
  - MySQL connection and incremental sync
  - PostgreSQL connection and WAL integration
  - Connection pool management
  - Failover scenarios
"""

import pytest
import asyncio
from datetime import datetime
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from src.sync.connectors.base import (
    ConnectionStatus, OperationType, DataRecord, DataBatch,
    SyncResult, ConnectorConfig
)
from src.sync.connectors.database.mysql import MySQLConnector, MySQLConfig
from src.sync.connectors.database.postgresql import PostgreSQLConnector, PostgreSQLConfig


class TestMySQLConfig:
    """Tests for MySQL configuration."""

    def test_default_config(self):
        """Test default MySQL configuration values."""
        config = MySQLConfig(
            name="test_mysql",
            database="testdb",
            username="user",
            password="pass"
        )

        assert config.host == "localhost"
        assert config.port == 3306
        assert config.charset == "utf8mb4"
        assert config.use_ssl is False
        assert config.autocommit is True

    def test_custom_config(self):
        """Test custom MySQL configuration."""
        config = MySQLConfig(
            name="custom_mysql",
            host="mysql.example.com",
            port=3307,
            database="production",
            username="admin",
            password="secret",
            charset="utf8",
            use_ssl=True,
            ssl_ca="/path/to/ca.pem"
        )

        assert config.host == "mysql.example.com"
        assert config.port == 3307
        assert config.charset == "utf8"
        assert config.use_ssl is True
        assert config.ssl_ca == "/path/to/ca.pem"

    def test_port_validation(self):
        """Test port range validation."""
        # Valid port
        config = MySQLConfig(
            name="test",
            database="db",
            username="user",
            password="pass",
            port=3306
        )
        assert config.port == 3306

        # Port at boundary
        config = MySQLConfig(
            name="test",
            database="db",
            username="user",
            password="pass",
            port=65535
        )
        assert config.port == 65535


class TestMySQLConnector:
    """Tests for MySQL connector functionality."""

    @pytest.fixture
    def mysql_config(self):
        """Create MySQL configuration for tests."""
        return MySQLConfig(
            name="test_mysql",
            host="localhost",
            port=3306,
            database="testdb",
            username="testuser",
            password="testpass"
        )

    @pytest.fixture
    def connector(self, mysql_config):
        """Create MySQL connector instance."""
        return MySQLConnector(mysql_config)

    @pytest.mark.asyncio
    async def test_connect_success(self, connector):
        """Test successful connection."""
        assert connector.status == ConnectionStatus.DISCONNECTED

        result = await connector.connect()

        assert result is True
        assert connector.status == ConnectionStatus.CONNECTED
        assert connector.is_connected is True

    @pytest.mark.asyncio
    async def test_disconnect(self, connector):
        """Test disconnection."""
        await connector.connect()
        await connector.disconnect()

        assert connector.status == ConnectionStatus.DISCONNECTED
        assert connector.is_connected is False

    @pytest.mark.asyncio
    async def test_health_check_connected(self, connector):
        """Test health check when connected."""
        await connector.connect()

        result = await connector.health_check()

        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_disconnected(self, connector):
        """Test health check when disconnected."""
        result = await connector.health_check()

        assert result is False

    @pytest.mark.asyncio
    async def test_fetch_schema(self, connector):
        """Test schema fetching."""
        await connector.connect()

        schema = await connector.fetch_schema()

        assert "database" in schema
        assert schema["database"] == "testdb"
        assert "tables" in schema
        assert len(schema["tables"]) > 0
        assert "fetched_at" in schema

    @pytest.mark.asyncio
    async def test_fetch_schema_caching(self, connector):
        """Test schema caching."""
        await connector.connect()

        # First fetch
        schema1 = await connector.fetch_schema()
        # Second fetch should return cached
        schema2 = await connector.fetch_schema()

        assert schema1 == schema2

    @pytest.mark.asyncio
    async def test_fetch_data_requires_connection(self, connector):
        """Test that fetch_data requires connection."""
        with pytest.raises(RuntimeError, match="Not connected"):
            await connector.fetch_data(table="customers")

    @pytest.mark.asyncio
    async def test_fetch_data_requires_table_or_query(self, connector):
        """Test that fetch_data requires table or query."""
        await connector.connect()

        with pytest.raises(ValueError, match="Either query or table"):
            await connector.fetch_data()

    @pytest.mark.asyncio
    async def test_fetch_data_basic(self, connector):
        """Test basic data fetching."""
        await connector.connect()

        batch = await connector.fetch_data(table="customers", limit=10)

        assert isinstance(batch, DataBatch)
        assert len(batch.records) <= 10
        assert batch.source_id == "mysql:testdb"
        assert batch.table_name == "customers"

    @pytest.mark.asyncio
    async def test_fetch_data_with_offset(self, connector):
        """Test data fetching with offset."""
        await connector.connect()

        batch = await connector.fetch_data(table="customers", limit=10, offset=5)

        assert batch.offset == 5
        assert batch.checkpoint is not None
        assert batch.checkpoint["offset"] == 5 + len(batch.records)

    @pytest.mark.asyncio
    async def test_fetch_data_incremental(self, connector):
        """Test incremental data fetching."""
        await connector.connect()

        batch = await connector.fetch_data(
            table="customers",
            incremental_field="created_at",
            incremental_value="2024-01-01"
        )

        assert isinstance(batch, DataBatch)
        # Records should be returned based on incremental filter

    @pytest.mark.asyncio
    async def test_fetch_data_stream(self, connector):
        """Test streaming data fetch."""
        await connector.connect()

        batches = []
        async for batch in connector.fetch_data_stream(table="customers", batch_size=10):
            batches.append(batch)
            if len(batches) >= 3:
                break

        assert len(batches) > 0
        for batch in batches:
            assert isinstance(batch, DataBatch)

    @pytest.mark.asyncio
    async def test_write_data_requires_connection(self, connector):
        """Test that write_data requires connection."""
        batch = DataBatch(records=[], source_id="test")

        with pytest.raises(RuntimeError, match="Not connected"):
            await connector.write_data(batch)

    @pytest.mark.asyncio
    async def test_write_data_insert(self, connector):
        """Test data insertion."""
        await connector.connect()

        records = [
            DataRecord(
                id=f"r_{i}",
                data={"name": f"Test {i}", "value": i},
                operation=OperationType.INSERT
            )
            for i in range(5)
        ]
        batch = DataBatch(records=records, source_id="test")

        result = await connector.write_data(batch, mode="insert")

        assert result.success is True
        assert result.records_processed == 5
        assert result.records_inserted == 5

    @pytest.mark.asyncio
    async def test_write_data_update(self, connector):
        """Test data update."""
        await connector.connect()

        records = [
            DataRecord(
                id=f"r_{i}",
                data={"name": f"Updated {i}"},
                operation=OperationType.UPDATE
            )
            for i in range(3)
        ]
        batch = DataBatch(records=records, source_id="test")

        result = await connector.write_data(batch, mode="update")

        assert result.success is True
        assert result.records_updated == 3

    @pytest.mark.asyncio
    async def test_write_data_upsert(self, connector):
        """Test data upsert (default mode)."""
        await connector.connect()

        records = [
            DataRecord(id=f"r_{i}", data={"value": i})
            for i in range(5)
        ]
        batch = DataBatch(records=records, source_id="test")

        result = await connector.write_data(batch)

        assert result.success is True
        assert result.records_processed == 5

    @pytest.mark.asyncio
    async def test_get_record_count(self, connector):
        """Test record count retrieval."""
        await connector.connect()

        count = await connector.get_record_count(table="customers")

        assert count > 0

    @pytest.mark.asyncio
    async def test_get_record_count_no_table(self, connector):
        """Test record count with no table specified."""
        await connector.connect()

        count = await connector.get_record_count()

        assert count == 0

    @pytest.mark.asyncio
    async def test_connection_status_transitions(self, connector):
        """Test connection status transitions."""
        assert connector.status == ConnectionStatus.DISCONNECTED

        # Connect
        await connector.connect()
        assert connector.status == ConnectionStatus.CONNECTED

        # Disconnect
        await connector.disconnect()
        assert connector.status == ConnectionStatus.DISCONNECTED

    @pytest.mark.asyncio
    async def test_statistics_tracking(self, connector):
        """Test that operations update statistics."""
        await connector.connect()

        initial_stats = connector.stats.copy()

        # Perform read
        await connector.fetch_data(table="customers", limit=10)

        assert connector.stats["records_read"] > initial_stats.get("records_read", 0)

        # Perform write
        records = [DataRecord(id="r_1", data={"value": 1})]
        batch = DataBatch(records=records, source_id="test")
        await connector.write_data(batch)

        assert connector.stats["records_written"] > initial_stats.get("records_written", 0)


class TestPostgreSQLConfig:
    """Tests for PostgreSQL configuration."""

    def test_default_config(self):
        """Test default PostgreSQL configuration values."""
        config = PostgreSQLConfig(
            name="test_pg",
            database="testdb",
            username="user",
            password="pass"
        )

        assert config.host == "localhost"
        assert config.port == 5432
        assert config.schema == "public"
        assert config.ssl_mode == "prefer"
        assert config.application_name == "superinsight-sync"

    def test_custom_config(self):
        """Test custom PostgreSQL configuration."""
        config = PostgreSQLConfig(
            name="custom_pg",
            host="pg.example.com",
            port=5433,
            database="production",
            username="admin",
            password="secret",
            schema="app",
            ssl_mode="require",
            statement_timeout=60000,
            lock_timeout=5000
        )

        assert config.host == "pg.example.com"
        assert config.port == 5433
        assert config.schema == "app"
        assert config.ssl_mode == "require"
        assert config.statement_timeout == 60000
        assert config.lock_timeout == 5000


class TestPostgreSQLConnector:
    """Tests for PostgreSQL connector functionality."""

    @pytest.fixture
    def pg_config(self):
        """Create PostgreSQL configuration for tests."""
        return PostgreSQLConfig(
            name="test_pg",
            host="localhost",
            port=5432,
            database="testdb",
            username="testuser",
            password="testpass"
        )

    @pytest.fixture
    def connector(self, pg_config):
        """Create PostgreSQL connector instance."""
        return PostgreSQLConnector(pg_config)

    @pytest.mark.asyncio
    async def test_connect_success(self, connector):
        """Test successful connection."""
        assert connector.status == ConnectionStatus.DISCONNECTED

        result = await connector.connect()

        assert result is True
        assert connector.status == ConnectionStatus.CONNECTED

    @pytest.mark.asyncio
    async def test_disconnect(self, connector):
        """Test disconnection."""
        await connector.connect()
        await connector.disconnect()

        assert connector.status == ConnectionStatus.DISCONNECTED

    @pytest.mark.asyncio
    async def test_health_check(self, connector):
        """Test health check."""
        await connector.connect()

        result = await connector.health_check()

        assert result is True

    @pytest.mark.asyncio
    async def test_fetch_schema(self, connector):
        """Test schema fetching."""
        await connector.connect()

        schema = await connector.fetch_schema()

        assert "database" in schema
        assert "schema" in schema
        assert schema["schema"] == "public"
        assert "tables" in schema

    @pytest.mark.asyncio
    async def test_fetch_data_with_schema(self, connector):
        """Test data fetching with schema prefix."""
        await connector.connect()

        batch = await connector.fetch_data(table="documents", limit=10)

        assert isinstance(batch, DataBatch)
        assert batch.schema_name == "public"

    @pytest.mark.asyncio
    async def test_fetch_data_with_filters(self, connector):
        """Test data fetching with filters."""
        await connector.connect()

        batch = await connector.fetch_data(
            table="documents",
            filters={"status": "active"}
        )

        assert isinstance(batch, DataBatch)

    @pytest.mark.asyncio
    async def test_fetch_data_with_custom_query(self, connector):
        """Test data fetching with custom query."""
        await connector.connect()

        batch = await connector.fetch_data(
            query="SELECT * FROM public.documents WHERE id IS NOT NULL LIMIT 10"
        )

        assert isinstance(batch, DataBatch)

    @pytest.mark.asyncio
    async def test_transaction_management(self, connector):
        """Test transaction management."""
        await connector.connect()

        # Begin transaction
        txn_id = await connector.begin_transaction()
        assert txn_id is not None

        # Commit transaction
        await connector.commit_transaction(txn_id)

    @pytest.mark.asyncio
    async def test_transaction_rollback(self, connector):
        """Test transaction rollback."""
        await connector.connect()

        txn_id = await connector.begin_transaction()
        await connector.rollback_transaction(txn_id)

    @pytest.mark.asyncio
    async def test_execute_query(self, connector):
        """Test raw query execution."""
        await connector.connect()

        results = await connector.execute_query("SELECT 1 as value")

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_write_data_with_modes(self, connector):
        """Test different write modes."""
        await connector.connect()

        records = [
            DataRecord(id=f"r_{i}", data={"value": i})
            for i in range(3)
        ]
        batch = DataBatch(records=records, source_id="test")

        # Test insert mode
        result = await connector.write_data(batch, mode="insert")
        assert result.success is True

        # Test update mode
        result = await connector.write_data(batch, mode="update")
        assert result.success is True

        # Test delete mode
        delete_records = [
            DataRecord(id="r_0", data={}, operation=OperationType.DELETE)
        ]
        delete_batch = DataBatch(records=delete_records, source_id="test")
        result = await connector.write_data(delete_batch, mode="delete")
        assert result.success is True


class TestConnectionPoolManagement:
    """Tests for connection pool management."""

    @pytest.mark.asyncio
    async def test_mysql_pool_initialization(self):
        """Test MySQL connection pool initialization."""
        config = MySQLConfig(
            name="pool_test",
            database="testdb",
            username="user",
            password="pass",
            pool_size=5
        )
        connector = MySQLConnector(config)

        await connector.connect()

        # Pool should be None in demo mode, but status should be connected
        assert connector.is_connected is True

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_postgresql_pool_initialization(self):
        """Test PostgreSQL connection pool initialization."""
        config = PostgreSQLConfig(
            name="pool_test",
            database="testdb",
            username="user",
            password="pass",
            pool_size=5
        )
        connector = PostgreSQLConnector(config)

        await connector.connect()

        assert connector.is_connected is True

        await connector.disconnect()


class TestFailoverScenarios:
    """Tests for connection failover scenarios."""

    @pytest.mark.asyncio
    async def test_mysql_reconnection_after_disconnect(self):
        """Test MySQL reconnection after disconnect."""
        config = MySQLConfig(
            name="reconnect_test",
            database="testdb",
            username="user",
            password="pass"
        )
        connector = MySQLConnector(config)

        # First connection
        await connector.connect()
        assert connector.is_connected is True

        # Disconnect
        await connector.disconnect()
        assert connector.is_connected is False

        # Reconnect
        await connector.connect()
        assert connector.is_connected is True

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_postgresql_reconnection_after_disconnect(self):
        """Test PostgreSQL reconnection after disconnect."""
        config = PostgreSQLConfig(
            name="reconnect_test",
            database="testdb",
            username="user",
            password="pass"
        )
        connector = PostgreSQLConnector(config)

        # First connection
        await connector.connect()
        assert connector.is_connected is True

        # Disconnect
        await connector.disconnect()
        assert connector.is_connected is False

        # Reconnect
        await connector.connect()
        assert connector.is_connected is True

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_operation_after_reconnect(self):
        """Test operations work after reconnection."""
        config = MySQLConfig(
            name="op_after_reconnect",
            database="testdb",
            username="user",
            password="pass"
        )
        connector = MySQLConnector(config)

        # Connect and disconnect
        await connector.connect()
        await connector.disconnect()

        # Reconnect
        await connector.connect()

        # Operations should work
        batch = await connector.fetch_data(table="customers", limit=5)
        assert isinstance(batch, DataBatch)

        await connector.disconnect()


class TestIncrementalSync:
    """Tests for incremental synchronization."""

    @pytest.mark.asyncio
    async def test_mysql_incremental_by_timestamp(self):
        """Test MySQL incremental sync using timestamp field."""
        config = MySQLConfig(
            name="incr_test",
            database="testdb",
            username="user",
            password="pass"
        )
        connector = MySQLConnector(config)
        await connector.connect()

        batch = await connector.fetch_data(
            table="orders",
            incremental_field="created_at",
            incremental_value="2024-01-01T00:00:00"
        )

        assert isinstance(batch, DataBatch)
        assert batch.checkpoint is not None

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_postgresql_incremental_by_version(self):
        """Test PostgreSQL incremental sync using version column."""
        config = PostgreSQLConfig(
            name="incr_test",
            database="testdb",
            username="user",
            password="pass"
        )
        connector = PostgreSQLConnector(config)
        await connector.connect()

        batch = await connector.fetch_data(
            table="documents",
            incremental_field="updated_at",
            incremental_value="2024-01-01T00:00:00"
        )

        assert isinstance(batch, DataBatch)

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_checkpoint_tracking(self):
        """Test checkpoint tracking during incremental sync."""
        config = MySQLConfig(
            name="checkpoint_test",
            database="testdb",
            username="user",
            password="pass"
        )
        connector = MySQLConnector(config)
        await connector.connect()

        # First batch
        batch1 = await connector.fetch_data(table="orders", limit=10, offset=0)
        checkpoint1 = batch1.checkpoint

        assert checkpoint1 is not None
        assert "offset" in checkpoint1
        assert "last_id" in checkpoint1

        # Second batch using checkpoint
        batch2 = await connector.fetch_data(
            table="orders",
            limit=10,
            offset=checkpoint1["offset"]
        )

        assert batch2.offset == checkpoint1["offset"]

        await connector.disconnect()


class TestDataRecordProcessing:
    """Tests for data record processing."""

    @pytest.mark.asyncio
    async def test_record_id_extraction(self):
        """Test that records have proper IDs."""
        config = MySQLConfig(
            name="id_test",
            database="testdb",
            username="user",
            password="pass"
        )
        connector = MySQLConnector(config)
        await connector.connect()

        batch = await connector.fetch_data(table="customers", limit=5)

        for record in batch.records:
            assert record.id is not None
            assert len(record.id) > 0

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_record_timestamp(self):
        """Test that records have timestamps."""
        config = PostgreSQLConfig(
            name="ts_test",
            database="testdb",
            username="user",
            password="pass"
        )
        connector = PostgreSQLConnector(config)
        await connector.connect()

        batch = await connector.fetch_data(table="documents", limit=5)

        for record in batch.records:
            assert record.timestamp is not None
            assert isinstance(record.timestamp, datetime)

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_record_operation_type(self):
        """Test that records have operation types."""
        config = MySQLConfig(
            name="op_test",
            database="testdb",
            username="user",
            password="pass"
        )
        connector = MySQLConnector(config)
        await connector.connect()

        batch = await connector.fetch_data(table="customers", limit=5)

        for record in batch.records:
            assert record.operation is not None
            assert record.operation == OperationType.UPSERT

        await connector.disconnect()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
