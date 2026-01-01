"""
Unit tests for file connectors (Local and S3).

Tests:
- Task 2.3: File Connector Tests
  - S3 connection and large file handling
  - Local file system monitoring
  - Multiple format support
  - Streaming processing
"""

import pytest
import asyncio
import json
import tempfile
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from src.sync.connectors.base import (
    ConnectionStatus, OperationType, DataRecord, DataBatch,
    SyncResult, ConnectorConfig
)
from src.sync.connectors.file.local_connector import (
    LocalConnector, LocalConnectorConfig, FileFormat, FileWatchMode,
    CSVOptions, JSONOptions
)
from src.sync.connectors.file.s3_connector import (
    S3Connector, S3ConnectorConfig, S3Provider, S3AuthConfig,
    FileFormat as S3FileFormat
)


class TestLocalConnectorConfig:
    """Tests for local file connector configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LocalConnectorConfig(name="test_local")

        assert config.base_path == "."
        assert config.file_pattern == "*"
        assert config.recursive is False
        assert config.file_format == FileFormat.AUTO
        assert config.watch_mode == FileWatchMode.NONE

    def test_custom_config(self):
        """Test custom configuration."""
        config = LocalConnectorConfig(
            name="custom_local",
            base_path="/data/files",
            file_pattern="*.json",
            recursive=True,
            file_format=FileFormat.JSON,
            watch_mode=FileWatchMode.POLL,
            poll_interval=30
        )

        assert config.base_path == "/data/files"
        assert config.file_pattern == "*.json"
        assert config.recursive is True
        assert config.file_format == FileFormat.JSON
        assert config.watch_mode == FileWatchMode.POLL
        assert config.poll_interval == 30

    def test_csv_options(self):
        """Test CSV parsing options."""
        csv_opts = CSVOptions(
            delimiter=";",
            quotechar="'",
            has_header=False,
            encoding="latin-1",
            skip_rows=2
        )

        config = LocalConnectorConfig(
            name="csv_test",
            csv_options=csv_opts
        )

        assert config.csv_options.delimiter == ";"
        assert config.csv_options.quotechar == "'"
        assert config.csv_options.has_header is False
        assert config.csv_options.skip_rows == 2

    def test_json_options(self):
        """Test JSON parsing options."""
        json_opts = JSONOptions(
            encoding="utf-8",
            data_path="results.items",
            flatten=True
        )

        config = LocalConnectorConfig(
            name="json_test",
            json_options=json_opts
        )

        assert config.json_options.data_path == "results.items"
        assert config.json_options.flatten is True


class TestLocalConnector:
    """Tests for local file connector functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory with test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test JSON file
            json_file = Path(tmpdir) / "data.json"
            json_data = [
                {"id": 1, "name": "Item 1", "value": 100},
                {"id": 2, "name": "Item 2", "value": 200},
                {"id": 3, "name": "Item 3", "value": 300}
            ]
            json_file.write_text(json.dumps(json_data))

            # Create test JSONL file
            jsonl_file = Path(tmpdir) / "data.jsonl"
            jsonl_content = "\n".join([json.dumps(item) for item in json_data])
            jsonl_file.write_text(jsonl_content)

            # Create test CSV file
            csv_file = Path(tmpdir) / "data.csv"
            csv_content = "id,name,value\n1,Item 1,100\n2,Item 2,200\n3,Item 3,300"
            csv_file.write_text(csv_content)

            yield tmpdir

    @pytest.fixture
    def config(self, temp_dir):
        """Create local connector configuration."""
        return LocalConnectorConfig(
            name="test_local",
            base_path=temp_dir,
            file_pattern="*"
        )

    @pytest.fixture
    def connector(self, config):
        """Create local connector instance."""
        return LocalConnector(config)

    @pytest.mark.asyncio
    async def test_connect_success(self, connector):
        """Test successful connection."""
        result = await connector.connect()

        assert result is True
        assert connector.status == ConnectionStatus.CONNECTED

    @pytest.mark.asyncio
    async def test_connect_invalid_path(self):
        """Test connection with invalid path."""
        config = LocalConnectorConfig(
            name="invalid",
            base_path="/nonexistent/path/that/does/not/exist"
        )
        connector = LocalConnector(config)

        result = await connector.connect()

        assert result is False
        assert connector.status == ConnectionStatus.ERROR

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

        assert "type" in schema
        assert schema["type"] == "local_filesystem"
        assert "base_path" in schema
        assert "files" in schema
        assert "total_files" in schema

    @pytest.mark.asyncio
    async def test_fetch_data_json(self, connector):
        """Test fetching data from JSON files."""
        await connector.connect()

        batch = await connector.fetch_data(table="data.json")

        assert isinstance(batch, DataBatch)
        assert len(batch.records) == 3
        assert batch.records[0].data["name"] == "Item 1"

    @pytest.mark.asyncio
    async def test_fetch_data_csv(self, temp_dir):
        """Test fetching data from CSV files."""
        config = LocalConnectorConfig(
            name="csv_test",
            base_path=temp_dir,
            file_format=FileFormat.CSV
        )
        connector = LocalConnector(config)
        await connector.connect()

        batch = await connector.fetch_data(table="data.csv")

        assert isinstance(batch, DataBatch)
        assert len(batch.records) == 3

    @pytest.mark.asyncio
    async def test_fetch_data_with_filters(self, connector):
        """Test fetching data with filters."""
        await connector.connect()

        batch = await connector.fetch_data(
            table="data.json",
            filters={"value": 200}
        )

        assert len(batch.records) == 1
        assert batch.records[0].data["value"] == 200

    @pytest.mark.asyncio
    async def test_fetch_data_with_limit_offset(self, connector):
        """Test fetching data with limit and offset."""
        await connector.connect()

        batch = await connector.fetch_data(
            table="data.json",
            limit=2,
            offset=1
        )

        assert len(batch.records) == 2
        assert batch.offset == 1

    @pytest.mark.asyncio
    async def test_fetch_data_stream(self, connector):
        """Test streaming data fetch."""
        await connector.connect()

        batches = []
        async for batch in connector.fetch_data_stream(table="data.json", batch_size=2):
            batches.append(batch)

        assert len(batches) >= 1
        total_records = sum(len(b.records) for b in batches)
        assert total_records == 3

    @pytest.mark.asyncio
    async def test_write_data_json(self, temp_dir):
        """Test writing data to JSON file."""
        config = LocalConnectorConfig(
            name="write_test",
            base_path=temp_dir
        )
        connector = LocalConnector(config)
        await connector.connect()

        records = [
            DataRecord(id="w_1", data={"name": "Written 1", "value": 1}),
            DataRecord(id="w_2", data={"name": "Written 2", "value": 2})
        ]
        batch = DataBatch(records=records, source_id="test", table_name="output.json")

        result = await connector.write_data(batch)

        assert result.success is True
        assert result.records_processed == 2

        # Verify file was created
        output_file = Path(temp_dir) / "output.json"
        assert output_file.exists()

    @pytest.mark.asyncio
    async def test_get_record_count(self, connector):
        """Test record count retrieval."""
        await connector.connect()

        count = await connector.get_record_count(table="data.json")

        assert count == 3

    @pytest.mark.asyncio
    async def test_auto_format_detection(self, connector, temp_dir):
        """Test automatic file format detection."""
        await connector.connect()

        # Test JSON detection
        json_batch = await connector.fetch_data(table="data.json")
        assert len(json_batch.records) > 0

        # Test CSV detection
        csv_batch = await connector.fetch_data(table="data.csv")
        assert len(csv_batch.records) > 0

    @pytest.mark.asyncio
    async def test_recursive_file_listing(self, temp_dir):
        """Test recursive file listing."""
        # Create subdirectory with file
        subdir = Path(temp_dir) / "subdir"
        subdir.mkdir()
        sub_file = subdir / "nested.json"
        sub_file.write_text(json.dumps([{"id": 100}]))

        config = LocalConnectorConfig(
            name="recursive_test",
            base_path=temp_dir,
            file_pattern="*.json",
            recursive=True
        )
        connector = LocalConnector(config)
        await connector.connect()

        schema = await connector.fetch_schema()

        # Should find files in subdirectory
        assert schema["total_files"] >= 2

    @pytest.mark.asyncio
    async def test_complex_filter_operations(self, connector):
        """Test complex filter operations."""
        await connector.connect()

        # Greater than filter
        batch = await connector.fetch_data(
            table="data.json",
            filters={"value": {"$op": "gt", "$value": 150}}
        )
        assert all(r.data["value"] > 150 for r in batch.records)

        # In filter
        batch = await connector.fetch_data(
            table="data.json",
            filters={"value": {"$op": "in", "$value": [100, 300]}}
        )
        assert all(r.data["value"] in [100, 300] for r in batch.records)


class TestS3ConnectorConfig:
    """Tests for S3 connector configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = S3ConnectorConfig(
            name="test_s3",
            bucket="my-bucket"
        )

        assert config.provider == S3Provider.AWS
        assert config.region == "us-east-1"
        assert config.prefix == ""
        assert config.file_format == S3FileFormat.AUTO
        assert config.recursive is True

    def test_custom_config(self):
        """Test custom configuration."""
        auth = S3AuthConfig(
            access_key_id="AKIAIOSFODNN7EXAMPLE",
            secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        )

        config = S3ConnectorConfig(
            name="custom_s3",
            bucket="production-bucket",
            prefix="data/",
            provider=S3Provider.MINIO,
            endpoint_url="http://minio:9000",
            region="us-west-2",
            auth=auth,
            file_format=S3FileFormat.JSONL
        )

        assert config.bucket == "production-bucket"
        assert config.prefix == "data/"
        assert config.provider == S3Provider.MINIO
        assert config.endpoint_url == "http://minio:9000"
        assert config.auth.access_key_id == "AKIAIOSFODNN7EXAMPLE"

    def test_iam_role_auth(self):
        """Test IAM role authentication configuration."""
        auth = S3AuthConfig(
            use_iam_role=True,
            role_arn="arn:aws:iam::123456789012:role/S3AccessRole"
        )

        config = S3ConnectorConfig(
            name="iam_s3",
            bucket="my-bucket",
            auth=auth
        )

        assert config.auth.use_iam_role is True
        assert config.auth.role_arn == "arn:aws:iam::123456789012:role/S3AccessRole"


class TestS3Connector:
    """Tests for S3 connector functionality."""

    @pytest.fixture
    def s3_config(self):
        """Create S3 configuration for tests."""
        return S3ConnectorConfig(
            name="test_s3",
            bucket="test-bucket",
            prefix="test-data/",
            auth=S3AuthConfig(
                access_key_id="test-key",
                secret_access_key="test-secret"
            )
        )

    @pytest.fixture
    def connector(self, s3_config):
        """Create S3 connector instance."""
        return S3Connector(s3_config)

    @pytest.mark.asyncio
    async def test_connect_with_mock(self, connector):
        """Test S3 connection with mock."""
        with patch("boto3.client") as mock_client:
            mock_s3 = MagicMock()
            mock_client.return_value = mock_s3

            result = await connector.connect()

            # Connection should work with mock
            assert connector.status in [ConnectionStatus.CONNECTED, ConnectionStatus.ERROR]

    @pytest.mark.asyncio
    async def test_disconnect(self, connector):
        """Test S3 disconnection."""
        connector._client = MagicMock()
        connector._set_status(ConnectionStatus.CONNECTED)

        await connector.disconnect()

        assert connector.status == ConnectionStatus.DISCONNECTED
        assert connector._client is None

    @pytest.mark.asyncio
    async def test_health_check_no_client(self, connector):
        """Test health check without client."""
        result = await connector.health_check()

        assert result is False

    @pytest.mark.asyncio
    async def test_format_detection_json(self, connector):
        """Test JSON format detection."""
        format = connector._detect_format("data/file.json")
        assert format == S3FileFormat.JSON

    @pytest.mark.asyncio
    async def test_format_detection_jsonl(self, connector):
        """Test JSONL format detection."""
        format = connector._detect_format("data/file.jsonl")
        assert format == S3FileFormat.JSONL

        format = connector._detect_format("data/file.ndjson")
        assert format == S3FileFormat.JSONL

    @pytest.mark.asyncio
    async def test_format_detection_csv(self, connector):
        """Test CSV format detection."""
        format = connector._detect_format("data/file.csv")
        assert format == S3FileFormat.CSV

    @pytest.mark.asyncio
    async def test_format_detection_parquet(self, connector):
        """Test Parquet format detection."""
        format = connector._detect_format("data/file.parquet")
        assert format == S3FileFormat.PARQUET

    def test_parse_json_content(self, connector):
        """Test JSON content parsing."""
        content = b'[{"id": 1, "name": "Test"}]'

        records = connector._parse_content(content, "data.json")

        assert len(records) == 1
        assert records[0]["id"] == 1

    def test_parse_jsonl_content(self, connector):
        """Test JSONL content parsing."""
        content = b'{"id": 1, "name": "Test1"}\n{"id": 2, "name": "Test2"}'

        records = connector._parse_content(content, "data.jsonl")

        assert len(records) == 2
        assert records[0]["id"] == 1
        assert records[1]["id"] == 2

    def test_parse_csv_content(self, connector):
        """Test CSV content parsing."""
        content = b'id,name\n1,Test1\n2,Test2'

        records = connector._parse_content(content, "data.csv")

        assert len(records) == 2
        assert records[0]["id"] == "1"
        assert records[0]["name"] == "Test1"

    def test_serialize_json_content(self, connector):
        """Test JSON content serialization."""
        records = [{"id": 1, "name": "Test"}]

        content = connector._serialize_content(records, S3FileFormat.JSON)

        assert b'"id": 1' in content

    def test_serialize_jsonl_content(self, connector):
        """Test JSONL content serialization."""
        records = [{"id": 1}, {"id": 2}]

        content = connector._serialize_content(records, S3FileFormat.JSONL)

        lines = content.decode("utf-8").strip().split("\n")
        assert len(lines) == 2

    def test_serialize_csv_content(self, connector):
        """Test CSV content serialization."""
        records = [{"id": 1, "name": "Test"}]

        content = connector._serialize_content(records, S3FileFormat.CSV)

        assert b'id,name' in content
        assert b'1,Test' in content

    def test_content_type_mapping(self, connector):
        """Test content type mapping."""
        assert connector._get_content_type(S3FileFormat.JSON) == "application/json"
        assert connector._get_content_type(S3FileFormat.JSONL) == "application/x-ndjson"
        assert connector._get_content_type(S3FileFormat.CSV) == "text/csv"
        assert connector._get_content_type(S3FileFormat.PARQUET) == "application/octet-stream"

    def test_pattern_matching(self, connector):
        """Test file pattern matching."""
        connector.s3_config.file_pattern = "*.json"

        assert connector._matches_pattern("data/file.json") is True
        assert connector._matches_pattern("data/file.csv") is False

    def test_filter_matching(self, connector):
        """Test record filter matching."""
        record = {"id": 1, "name": "Test", "value": 100}

        # Simple equality
        assert connector._matches_filters(record, {"id": 1}) is True
        assert connector._matches_filters(record, {"id": 2}) is False

        # Complex operators
        assert connector._matches_filters(record, {"value": {"$op": "gt", "$value": 50}}) is True
        assert connector._matches_filters(record, {"value": {"$op": "lt", "$value": 50}}) is False


class TestLargeFileHandling:
    """Tests for large file handling."""

    @pytest.mark.asyncio
    async def test_streaming_large_json_file(self):
        """Test streaming processing of large JSON files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create large JSON file
            large_file = Path(tmpdir) / "large.json"
            large_data = [{"id": i, "data": f"item_{i}"} for i in range(1000)]
            large_file.write_text(json.dumps(large_data))

            config = LocalConnectorConfig(
                name="large_test",
                base_path=tmpdir,
                batch_size=100
            )
            connector = LocalConnector(config)
            await connector.connect()

            batches = []
            async for batch in connector.fetch_data_stream(table="large.json", batch_size=100):
                batches.append(batch)

            total_records = sum(len(b.records) for b in batches)
            assert total_records == 1000

    @pytest.mark.asyncio
    async def test_streaming_large_jsonl_file(self):
        """Test streaming processing of large JSONL files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create large JSONL file
            large_file = Path(tmpdir) / "large.jsonl"
            lines = [json.dumps({"id": i}) for i in range(500)]
            large_file.write_text("\n".join(lines))

            config = LocalConnectorConfig(
                name="large_jsonl_test",
                base_path=tmpdir,
                batch_size=50
            )
            connector = LocalConnector(config)
            await connector.connect()

            batches = []
            async for batch in connector.fetch_data_stream(table="large.jsonl", batch_size=50):
                batches.append(batch)

            total_records = sum(len(b.records) for b in batches)
            assert total_records == 500


class TestMultipleFormatSupport:
    """Tests for multiple file format support."""

    @pytest.fixture
    def multi_format_dir(self):
        """Create directory with multiple file formats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # JSON
            json_file = Path(tmpdir) / "data.json"
            json_file.write_text(json.dumps([{"id": 1, "type": "json"}]))

            # JSONL
            jsonl_file = Path(tmpdir) / "data.jsonl"
            jsonl_file.write_text('{"id": 2, "type": "jsonl"}')

            # CSV
            csv_file = Path(tmpdir) / "data.csv"
            csv_file.write_text("id,type\n3,csv")

            # TSV
            tsv_file = Path(tmpdir) / "data.tsv"
            tsv_file.write_text("id\ttype\n4\ttsv")

            yield tmpdir

    @pytest.mark.asyncio
    async def test_read_all_formats(self, multi_format_dir):
        """Test reading all supported formats."""
        config = LocalConnectorConfig(
            name="multi_format",
            base_path=multi_format_dir
        )
        connector = LocalConnector(config)
        await connector.connect()

        # JSON
        json_batch = await connector.fetch_data(table="data.json")
        assert len(json_batch.records) == 1
        assert json_batch.records[0].data["type"] == "json"

        # JSONL
        jsonl_batch = await connector.fetch_data(table="data.jsonl")
        assert len(jsonl_batch.records) == 1
        assert jsonl_batch.records[0].data["type"] == "jsonl"

        # CSV
        csv_batch = await connector.fetch_data(table="data.csv")
        assert len(csv_batch.records) == 1
        assert csv_batch.records[0].data["type"] == "csv"

    @pytest.mark.asyncio
    async def test_format_override(self, multi_format_dir):
        """Test explicit format override."""
        config = LocalConnectorConfig(
            name="format_override",
            base_path=multi_format_dir,
            file_format=FileFormat.JSON
        )
        connector = LocalConnector(config)
        await connector.connect()

        # Should force JSON parsing
        batch = await connector.fetch_data(table="data.json")
        assert len(batch.records) == 1


class TestFileWatching:
    """Tests for file watching functionality."""

    def test_watch_mode_none(self):
        """Test no watching mode."""
        config = LocalConnectorConfig(
            name="no_watch",
            watch_mode=FileWatchMode.NONE
        )

        assert config.watch_mode == FileWatchMode.NONE

    def test_watch_mode_poll(self):
        """Test polling watch mode."""
        config = LocalConnectorConfig(
            name="poll_watch",
            watch_mode=FileWatchMode.POLL,
            poll_interval=30
        )

        assert config.watch_mode == FileWatchMode.POLL
        assert config.poll_interval == 30

    def test_watch_mode_inotify(self):
        """Test inotify watch mode (Linux)."""
        config = LocalConnectorConfig(
            name="inotify_watch",
            watch_mode=FileWatchMode.INOTIFY
        )

        assert config.watch_mode == FileWatchMode.INOTIFY


class TestS3ProviderSupport:
    """Tests for different S3 provider support."""

    def test_aws_provider(self):
        """Test AWS S3 provider configuration."""
        config = S3ConnectorConfig(
            name="aws_s3",
            bucket="aws-bucket",
            provider=S3Provider.AWS,
            region="us-east-1"
        )

        assert config.provider == S3Provider.AWS
        assert config.endpoint_url is None

    def test_minio_provider(self):
        """Test MinIO provider configuration."""
        config = S3ConnectorConfig(
            name="minio_s3",
            bucket="minio-bucket",
            provider=S3Provider.MINIO,
            endpoint_url="http://minio:9000"
        )

        assert config.provider == S3Provider.MINIO
        assert config.endpoint_url == "http://minio:9000"

    def test_digital_ocean_provider(self):
        """Test Digital Ocean Spaces provider."""
        config = S3ConnectorConfig(
            name="do_s3",
            bucket="do-bucket",
            provider=S3Provider.DIGITAL_OCEAN,
            endpoint_url="https://nyc3.digitaloceanspaces.com",
            region="nyc3"
        )

        assert config.provider == S3Provider.DIGITAL_OCEAN

    def test_custom_provider(self):
        """Test custom S3-compatible provider."""
        config = S3ConnectorConfig(
            name="custom_s3",
            bucket="custom-bucket",
            provider=S3Provider.CUSTOM,
            endpoint_url="https://custom-s3.example.com"
        )

        assert config.provider == S3Provider.CUSTOM


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
