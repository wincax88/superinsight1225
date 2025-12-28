"""
S3 Compatible Storage Connector Module.

Provides connector for synchronizing data from S3-compatible object storage
(AWS S3, MinIO, Ceph, etc.) with support for various file formats.
"""

import asyncio
import hashlib
import io
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from pydantic import Field

from ..base import (
    BaseConnector,
    ConnectionStatus,
    ConnectorConfig,
    DataBatch,
    DataRecord,
    OperationType,
    SyncResult,
)

logger = logging.getLogger(__name__)


class S3Provider(str, Enum):
    """S3-compatible storage providers."""
    AWS = "aws"
    MINIO = "minio"
    CEPH = "ceph"
    WASABI = "wasabi"
    DIGITAL_OCEAN = "digital_ocean"
    BACKBLAZE = "backblaze"
    CUSTOM = "custom"


class FileFormat(str, Enum):
    """Supported file formats."""
    JSON = "json"
    JSONL = "jsonl"
    CSV = "csv"
    PARQUET = "parquet"
    AUTO = "auto"


@dataclass
class S3AuthConfig:
    """S3 authentication configuration."""
    access_key_id: Optional[str] = None
    secret_access_key: Optional[str] = None
    session_token: Optional[str] = None
    use_iam_role: bool = False
    role_arn: Optional[str] = None


class S3ConnectorConfig(ConnectorConfig):
    """S3 connector configuration."""

    # Provider settings
    provider: S3Provider = S3Provider.AWS
    endpoint_url: Optional[str] = None  # Custom endpoint for non-AWS

    # Bucket settings
    bucket: str
    prefix: str = ""  # Object key prefix

    # Authentication
    region: str = "us-east-1"
    auth: S3AuthConfig = Field(default_factory=S3AuthConfig)

    # File settings
    file_pattern: str = "*"  # Glob-like pattern
    file_format: FileFormat = FileFormat.AUTO
    recursive: bool = True

    # CSV options
    csv_delimiter: str = ","
    csv_has_header: bool = True

    # JSON options
    json_data_path: str = ""

    # Record settings
    id_field: str = "id"
    timestamp_field: Optional[str] = None

    # Processing options
    archive_prefix: Optional[str] = None  # Move processed files here
    delete_after_sync: bool = False


class S3Connector(BaseConnector):
    """
    S3 Compatible Storage Connector.

    Supports:
    - AWS S3 and S3-compatible storage (MinIO, Ceph, etc.)
    - Multiple file formats (JSON, JSONL, CSV, Parquet)
    - Prefix filtering and glob patterns
    - Streaming for large files
    - Archive/delete after processing
    """

    def __init__(self, config: S3ConnectorConfig):
        """Initialize S3 connector."""
        super().__init__(config)
        self.s3_config = config
        self._client = None
        self._resource = None

    async def connect(self) -> bool:
        """Establish S3 connection."""
        try:
            self._set_status(ConnectionStatus.CONNECTING)

            # Import boto3 here to make it optional
            import boto3
            from botocore.config import Config

            # Build configuration
            boto_config = Config(
                connect_timeout=self.s3_config.connection_timeout,
                read_timeout=self.s3_config.read_timeout,
                retries={
                    "max_attempts": self.s3_config.max_retries,
                    "mode": "adaptive"
                }
            )

            # Build client kwargs
            client_kwargs = {
                "config": boto_config,
                "region_name": self.s3_config.region
            }

            # Custom endpoint for non-AWS providers
            if self.s3_config.endpoint_url:
                client_kwargs["endpoint_url"] = self.s3_config.endpoint_url

            # Authentication
            auth = self.s3_config.auth
            if not auth.use_iam_role:
                if auth.access_key_id and auth.secret_access_key:
                    client_kwargs["aws_access_key_id"] = auth.access_key_id
                    client_kwargs["aws_secret_access_key"] = auth.secret_access_key
                    if auth.session_token:
                        client_kwargs["aws_session_token"] = auth.session_token

            # Create S3 client (run in executor as boto3 is sync)
            loop = asyncio.get_event_loop()
            self._client = await loop.run_in_executor(
                None,
                lambda: boto3.client("s3", **client_kwargs)
            )

            # Verify bucket access
            await loop.run_in_executor(
                None,
                lambda: self._client.head_bucket(Bucket=self.s3_config.bucket)
            )

            self._set_status(ConnectionStatus.CONNECTED)
            logger.info(f"Connected to S3 bucket: {self.s3_config.bucket}")
            return True

        except Exception as e:
            self._record_error(e)
            self._set_status(ConnectionStatus.ERROR)
            return False

    async def disconnect(self) -> None:
        """Close S3 connection."""
        self._client = None
        self._set_status(ConnectionStatus.DISCONNECTED)
        logger.info("Disconnected from S3")

    async def health_check(self) -> bool:
        """Check if S3 bucket is accessible."""
        if not self._client:
            return False

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self._client.head_bucket(Bucket=self.s3_config.bucket)
            )
            return True
        except Exception as e:
            logger.warning(f"S3 health check failed: {e}")
            return False

    async def fetch_schema(self) -> Dict[str, Any]:
        """Fetch S3 bucket schema information."""
        schema = {
            "type": "s3",
            "bucket": self.s3_config.bucket,
            "prefix": self.s3_config.prefix,
            "provider": self.s3_config.provider.value,
            "objects": [],
            "inferred_schema": {}
        }

        try:
            objects = await self._list_objects(limit=100)
            schema["objects"] = [obj["key"] for obj in objects]
            schema["total_objects"] = len(objects)

            # Infer schema from first file
            if objects:
                first_obj = objects[0]
                content = await self._get_object(first_obj["key"])
                records = self._parse_content(content, first_obj["key"])
                if records:
                    schema["inferred_schema"] = {
                        k: type(v).__name__ for k, v in records[0].items()
                    }
        except Exception as e:
            logger.warning(f"Could not fetch schema: {e}")

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
        Fetch data from S3 objects.

        Args:
            query: Object key pattern
            table: Specific object key or prefix
            filters: Record filters
            limit: Maximum records
            offset: Record offset
            incremental_field: Field for incremental sync
            incremental_value: Last sync value

        Returns:
            DataBatch containing records from S3 objects
        """
        prefix = table or query or self.s3_config.prefix
        objects = await self._list_objects(prefix=prefix)

        all_records = []
        for obj in objects:
            if limit and len(all_records) >= limit + offset:
                break

            try:
                content = await self._get_object(obj["key"])
                file_records = self._parse_content(content, obj["key"])

                # Add S3 metadata to each record
                for record in file_records:
                    record["_s3_key"] = obj["key"]
                    record["_s3_last_modified"] = obj.get("last_modified")

                all_records.extend(file_records)
            except Exception as e:
                logger.error(f"Error reading object {obj['key']}: {e}")

        # Apply filters
        if filters:
            all_records = [r for r in all_records if self._matches_filters(r, filters)]

        # Apply incremental filter
        if incremental_field and incremental_value:
            all_records = [
                r for r in all_records
                if str(r.get(incremental_field, "")) > incremental_value
            ]

        # Convert to DataRecords
        records = []
        for idx, record in enumerate(all_records):
            if idx < offset:
                continue
            if limit and len(records) >= limit:
                break

            record_id = str(record.get(self.s3_config.id_field, ""))
            if not record_id:
                record_id = hashlib.md5(
                    json.dumps(record, sort_keys=True, default=str).encode()
                ).hexdigest()[:16]

            timestamp = None
            if self.s3_config.timestamp_field:
                ts_value = record.get(self.s3_config.timestamp_field)
                if ts_value:
                    try:
                        if isinstance(ts_value, datetime):
                            timestamp = ts_value
                        elif isinstance(ts_value, str):
                            timestamp = datetime.fromisoformat(ts_value.replace("Z", "+00:00"))
                    except (ValueError, AttributeError):
                        pass

            records.append(DataRecord(
                id=record_id,
                data=record,
                timestamp=timestamp,
                metadata={"source": self.s3_config.name, "bucket": self.s3_config.bucket}
            ))

        total_count = len(all_records)
        has_more = offset + len(records) < total_count

        self._record_read(len(records))

        return DataBatch(
            records=records,
            source_id=self.s3_config.name,
            table_name=prefix,
            total_count=total_count,
            offset=offset,
            has_more=has_more,
            checkpoint={
                "offset": offset + len(records),
                "objects_processed": [obj["key"] for obj in objects],
                "timestamp": datetime.utcnow().isoformat()
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
        """Stream data from S3 in batches."""
        batch_size = batch_size or self.s3_config.batch_size
        prefix = table or query or self.s3_config.prefix
        objects = await self._list_objects(prefix=prefix)

        current_batch = []
        batch_num = 0

        for obj in objects:
            try:
                content = await self._get_object(obj["key"])
                file_records = self._parse_content(content, obj["key"])

                for record in file_records:
                    # Apply filters
                    if filters and not self._matches_filters(record, filters):
                        continue

                    # Apply incremental filter
                    if incremental_field and incremental_value:
                        if str(record.get(incremental_field, "")) <= incremental_value:
                            continue

                    record_id = str(record.get(self.s3_config.id_field, ""))
                    if not record_id:
                        record_id = hashlib.md5(
                            json.dumps(record, sort_keys=True, default=str).encode()
                        ).hexdigest()[:16]

                    current_batch.append(DataRecord(
                        id=record_id,
                        data=record,
                        metadata={
                            "source": self.s3_config.name,
                            "bucket": self.s3_config.bucket,
                            "key": obj["key"]
                        }
                    ))

                    if len(current_batch) >= batch_size:
                        yield DataBatch(
                            records=current_batch,
                            source_id=self.s3_config.name,
                            table_name=prefix,
                            total_count=len(current_batch),
                            offset=batch_num * batch_size,
                            has_more=True
                        )
                        self._record_read(len(current_batch))
                        current_batch = []
                        batch_num += 1

            except Exception as e:
                logger.error(f"Error reading object {obj['key']}: {e}")

        # Yield remaining records
        if current_batch:
            yield DataBatch(
                records=current_batch,
                source_id=self.s3_config.name,
                table_name=prefix,
                total_count=len(current_batch),
                offset=batch_num * batch_size,
                has_more=False
            )
            self._record_read(len(current_batch))

    async def write_data(
        self,
        batch: DataBatch,
        mode: str = "upsert"
    ) -> SyncResult:
        """
        Write data to S3.

        Args:
            batch: DataBatch to write
            mode: Write mode

        Returns:
            SyncResult with write statistics
        """
        start_time = time.time()
        result = SyncResult(success=True)

        try:
            # Prepare data
            records_data = [r.data for r in batch.records]

            # Determine output key
            output_key = batch.table_name or f"{self.s3_config.prefix}/output.json"
            if not output_key.startswith(self.s3_config.prefix):
                output_key = f"{self.s3_config.prefix}/{output_key}"

            # Serialize data
            file_format = self._detect_format(output_key)
            content = self._serialize_content(records_data, file_format)

            # Upload to S3
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self._client.put_object(
                    Bucket=self.s3_config.bucket,
                    Key=output_key,
                    Body=content,
                    ContentType=self._get_content_type(file_format)
                )
            )

            result.records_processed = len(batch.records)
            result.records_inserted = len(batch.records)
            self._record_write(len(batch.records))

        except Exception as e:
            result.success = False
            result.errors.append({
                "error": str(e),
                "key": output_key if 'output_key' in locals() else "unknown"
            })

        result.duration_seconds = time.time() - start_time
        return result

    async def get_record_count(
        self,
        table: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> int:
        """Get total record count from S3 objects."""
        prefix = table or self.s3_config.prefix
        objects = await self._list_objects(prefix=prefix)

        total = 0
        for obj in objects:
            try:
                content = await self._get_object(obj["key"])
                records = self._parse_content(content, obj["key"])
                if filters:
                    records = [r for r in records if self._matches_filters(r, filters)]
                total += len(records)
            except Exception:
                pass

        return total

    async def _list_objects(
        self,
        prefix: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """List objects in bucket."""
        prefix = prefix or self.s3_config.prefix
        objects = []

        loop = asyncio.get_event_loop()
        paginator = self._client.get_paginator("list_objects_v2")

        async def list_objects_sync():
            result = []
            for page in paginator.paginate(
                Bucket=self.s3_config.bucket,
                Prefix=prefix
            ):
                for obj in page.get("Contents", []):
                    # Apply pattern filter
                    if self._matches_pattern(obj["Key"]):
                        result.append({
                            "key": obj["Key"],
                            "size": obj["Size"],
                            "last_modified": obj["LastModified"],
                            "etag": obj["ETag"]
                        })
                        if limit and len(result) >= limit:
                            return result
            return result

        objects = await loop.run_in_executor(None, list_objects_sync)
        return objects

    async def _get_object(self, key: str) -> bytes:
        """Get object content."""
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._client.get_object(
                Bucket=self.s3_config.bucket,
                Key=key
            )
        )
        return response["Body"].read()

    def _matches_pattern(self, key: str) -> bool:
        """Check if key matches file pattern."""
        import fnmatch
        pattern = self.s3_config.file_pattern
        filename = key.split("/")[-1]
        return fnmatch.fnmatch(filename, pattern)

    def _detect_format(self, key: str) -> FileFormat:
        """Detect file format from key."""
        if self.s3_config.file_format != FileFormat.AUTO:
            return self.s3_config.file_format

        key_lower = key.lower()
        if key_lower.endswith(".json"):
            return FileFormat.JSON
        elif key_lower.endswith(".jsonl") or key_lower.endswith(".ndjson"):
            return FileFormat.JSONL
        elif key_lower.endswith(".csv"):
            return FileFormat.CSV
        elif key_lower.endswith(".parquet"):
            return FileFormat.PARQUET

        return FileFormat.JSON

    def _parse_content(self, content: bytes, key: str) -> List[Dict[str, Any]]:
        """Parse file content to records."""
        file_format = self._detect_format(key)

        if file_format == FileFormat.JSON:
            data = json.loads(content.decode("utf-8"))
            if self.s3_config.json_data_path:
                for part in self.s3_config.json_data_path.split("."):
                    if isinstance(data, dict):
                        data = data.get(part, [])
            return data if isinstance(data, list) else [data]

        elif file_format == FileFormat.JSONL:
            records = []
            for line in content.decode("utf-8").strip().split("\n"):
                if line:
                    records.append(json.loads(line))
            return records

        elif file_format == FileFormat.CSV:
            import csv
            reader = csv.DictReader(
                io.StringIO(content.decode("utf-8")),
                delimiter=self.s3_config.csv_delimiter
            )
            return list(reader)

        elif file_format == FileFormat.PARQUET:
            try:
                import pyarrow.parquet as pq
                table = pq.read_table(io.BytesIO(content))
                return table.to_pylist()
            except ImportError:
                raise ImportError("pyarrow is required for Parquet support")

        return []

    def _serialize_content(
        self,
        records: List[Dict[str, Any]],
        file_format: FileFormat
    ) -> bytes:
        """Serialize records to bytes."""
        if file_format == FileFormat.JSON:
            return json.dumps(records, default=str, indent=2).encode("utf-8")

        elif file_format == FileFormat.JSONL:
            lines = [json.dumps(r, default=str) for r in records]
            return "\n".join(lines).encode("utf-8")

        elif file_format == FileFormat.CSV:
            if not records:
                return b""
            import csv
            output = io.StringIO()
            writer = csv.DictWriter(
                output,
                fieldnames=list(records[0].keys()),
                delimiter=self.s3_config.csv_delimiter
            )
            writer.writeheader()
            writer.writerows(records)
            return output.getvalue().encode("utf-8")

        elif file_format == FileFormat.PARQUET:
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq
                table = pa.Table.from_pylist(records)
                output = io.BytesIO()
                pq.write_table(table, output)
                return output.getvalue()
            except ImportError:
                raise ImportError("pyarrow is required for Parquet support")

        return json.dumps(records, default=str).encode("utf-8")

    def _get_content_type(self, file_format: FileFormat) -> str:
        """Get content type for file format."""
        content_types = {
            FileFormat.JSON: "application/json",
            FileFormat.JSONL: "application/x-ndjson",
            FileFormat.CSV: "text/csv",
            FileFormat.PARQUET: "application/octet-stream"
        }
        return content_types.get(file_format, "application/octet-stream")

    def _matches_filters(
        self,
        record: Dict[str, Any],
        filters: Dict[str, Any]
    ) -> bool:
        """Check if record matches filters."""
        for key, value in filters.items():
            record_value = record.get(key)
            if isinstance(value, dict):
                op = value.get("$op", "eq")
                filter_value = value.get("$value")
                if op == "eq" and record_value != filter_value:
                    return False
                elif op == "ne" and record_value == filter_value:
                    return False
                elif op == "gt" and not (record_value > filter_value):
                    return False
                elif op == "lt" and not (record_value < filter_value):
                    return False
            else:
                if record_value != value:
                    return False
        return True


# Register connector
from ..base import ConnectorFactory
ConnectorFactory.register("s3", S3Connector)
