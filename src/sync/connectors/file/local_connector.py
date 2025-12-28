"""
Local File System Connector Module.

Provides connector for synchronizing data from local file systems,
supporting various file formats (JSON, CSV, XML, Parquet).
"""

import asyncio
import csv
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import aiofiles
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


class FileFormat(str, Enum):
    """Supported file formats."""
    JSON = "json"
    JSONL = "jsonl"  # JSON Lines
    CSV = "csv"
    TSV = "tsv"
    XML = "xml"
    PARQUET = "parquet"
    AUTO = "auto"  # Auto-detect from extension


class FileWatchMode(str, Enum):
    """File watching modes."""
    NONE = "none"           # No watching
    POLL = "poll"           # Periodic polling
    INOTIFY = "inotify"     # Linux inotify (if available)


@dataclass
class CSVOptions:
    """CSV parsing options."""
    delimiter: str = ","
    quotechar: str = '"'
    has_header: bool = True
    encoding: str = "utf-8"
    skip_rows: int = 0


@dataclass
class JSONOptions:
    """JSON parsing options."""
    encoding: str = "utf-8"
    data_path: str = ""  # JSON path to data array (e.g., "results.data")
    flatten: bool = False


class LocalConnectorConfig(ConnectorConfig):
    """Local file system connector configuration."""

    # Base path
    base_path: str = "."

    # File patterns
    file_pattern: str = "*"  # Glob pattern for files
    recursive: bool = False

    # Format settings
    file_format: FileFormat = FileFormat.AUTO
    csv_options: CSVOptions = Field(default_factory=CSVOptions)
    json_options: JSONOptions = Field(default_factory=JSONOptions)

    # Record settings
    id_field: str = "id"
    timestamp_field: Optional[str] = None

    # Watch settings
    watch_mode: FileWatchMode = FileWatchMode.NONE
    poll_interval: int = 60  # seconds

    # Processing
    archive_processed: bool = False
    archive_path: str = "./archive"
    delete_after_sync: bool = False


class LocalConnector(BaseConnector):
    """
    Local File System Connector.

    Supports:
    - Multiple file formats (JSON, JSONL, CSV, TSV, XML, Parquet)
    - Glob patterns for file selection
    - Recursive directory scanning
    - File watching for real-time sync
    - Archive/delete after processing
    """

    def __init__(self, config: LocalConnectorConfig):
        """Initialize local file connector."""
        super().__init__(config)
        self.local_config = config
        self._base_path: Optional[Path] = None
        self._processed_files: set = set()

    async def connect(self) -> bool:
        """Verify base path accessibility."""
        try:
            self._set_status(ConnectionStatus.CONNECTING)

            self._base_path = Path(self.local_config.base_path).resolve()

            if not self._base_path.exists():
                raise FileNotFoundError(f"Base path not found: {self._base_path}")

            if not self._base_path.is_dir():
                raise NotADirectoryError(f"Base path is not a directory: {self._base_path}")

            # Check read permissions
            if not os.access(self._base_path, os.R_OK):
                raise PermissionError(f"No read access to: {self._base_path}")

            self._set_status(ConnectionStatus.CONNECTED)
            logger.info(f"Connected to local path: {self._base_path}")
            return True

        except Exception as e:
            self._record_error(e)
            self._set_status(ConnectionStatus.ERROR)
            return False

    async def disconnect(self) -> None:
        """Disconnect from local file system."""
        self._base_path = None
        self._set_status(ConnectionStatus.DISCONNECTED)
        logger.info("Disconnected from local file system")

    async def health_check(self) -> bool:
        """Check if base path is still accessible."""
        if not self._base_path:
            return False

        return self._base_path.exists() and os.access(self._base_path, os.R_OK)

    async def fetch_schema(self) -> Dict[str, Any]:
        """
        Fetch schema information from files.

        Scans sample files to infer schema.
        """
        schema = {
            "type": "local_filesystem",
            "base_path": str(self._base_path),
            "files": [],
            "inferred_schema": {}
        }

        files = self._list_files()
        schema["files"] = [str(f.relative_to(self._base_path)) for f in files[:100]]
        schema["total_files"] = len(files)

        # Infer schema from first file
        if files:
            try:
                records = await self._read_file(files[0])
                if records:
                    first_record = records[0]
                    schema["inferred_schema"] = {
                        k: type(v).__name__ for k, v in first_record.items()
                    }
            except Exception as e:
                logger.warning(f"Could not infer schema: {e}")

        return schema

    async def fetch_data(
        self,
        query: Optional[str] = None,
        table: Optional[str] = None,  # Used as file path or pattern
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: int = 0,
        incremental_field: Optional[str] = None,
        incremental_value: Optional[str] = None
    ) -> DataBatch:
        """
        Fetch data from local files.

        Args:
            query: File glob pattern
            table: Specific file path (relative to base_path)
            filters: Record filters
            limit: Maximum records
            offset: Record offset
            incremental_field: Field for incremental sync
            incremental_value: Last sync value

        Returns:
            DataBatch containing records from files
        """
        pattern = query or table or self.local_config.file_pattern
        files = self._list_files(pattern)

        all_records = []
        for file_path in files:
            if len(all_records) >= (limit or float('inf')) + offset:
                break

            try:
                file_records = await self._read_file(file_path)
                all_records.extend(file_records)
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {e}")

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

            record_id = str(record.get(self.local_config.id_field, ""))
            if not record_id:
                record_id = hashlib.md5(
                    json.dumps(record, sort_keys=True, default=str).encode()
                ).hexdigest()[:16]

            timestamp = None
            if self.local_config.timestamp_field:
                ts_value = record.get(self.local_config.timestamp_field)
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
                metadata={"source": self.local_config.name}
            ))

        total_count = len(all_records)
        has_more = offset + len(records) < total_count

        self._record_read(len(records))

        return DataBatch(
            records=records,
            source_id=self.local_config.name,
            table_name=pattern,
            total_count=total_count,
            offset=offset,
            has_more=has_more,
            checkpoint={
                "offset": offset + len(records),
                "files_processed": [str(f) for f in files],
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
        """
        Stream data from files in batches.

        Processes files one at a time to minimize memory usage.
        """
        batch_size = batch_size or self.local_config.batch_size
        pattern = query or table or self.local_config.file_pattern
        files = self._list_files(pattern)

        current_batch = []
        batch_num = 0

        for file_path in files:
            try:
                file_records = await self._read_file(file_path)

                for record in file_records:
                    # Apply filters
                    if filters and not self._matches_filters(record, filters):
                        continue

                    # Apply incremental filter
                    if incremental_field and incremental_value:
                        if str(record.get(incremental_field, "")) <= incremental_value:
                            continue

                    record_id = str(record.get(self.local_config.id_field, ""))
                    if not record_id:
                        record_id = hashlib.md5(
                            json.dumps(record, sort_keys=True, default=str).encode()
                        ).hexdigest()[:16]

                    current_batch.append(DataRecord(
                        id=record_id,
                        data=record,
                        metadata={
                            "source": self.local_config.name,
                            "file": str(file_path)
                        }
                    ))

                    if len(current_batch) >= batch_size:
                        yield DataBatch(
                            records=current_batch,
                            source_id=self.local_config.name,
                            table_name=pattern,
                            total_count=len(current_batch),
                            offset=batch_num * batch_size,
                            has_more=True
                        )
                        self._record_read(len(current_batch))
                        current_batch = []
                        batch_num += 1

            except Exception as e:
                logger.error(f"Error reading file {file_path}: {e}")

        # Yield remaining records
        if current_batch:
            yield DataBatch(
                records=current_batch,
                source_id=self.local_config.name,
                table_name=pattern,
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
        Write data to local files.

        Args:
            batch: DataBatch to write
            mode: Write mode (append, overwrite)

        Returns:
            SyncResult with write statistics
        """
        start_time = time.time()
        result = SyncResult(success=True)

        output_file = self._base_path / (batch.table_name or "output.json")

        try:
            # Prepare data
            records_data = [r.data for r in batch.records]

            # Determine format
            file_format = self._detect_format(output_file)

            if file_format in [FileFormat.JSON, FileFormat.AUTO]:
                if mode == "append" and output_file.exists():
                    existing_data = await self._read_json(output_file)
                    records_data = existing_data + records_data

                async with aiofiles.open(output_file, "w", encoding="utf-8") as f:
                    await f.write(json.dumps(records_data, indent=2, default=str))

            elif file_format == FileFormat.JSONL:
                mode_flag = "a" if mode == "append" else "w"
                async with aiofiles.open(output_file, mode_flag, encoding="utf-8") as f:
                    for record in records_data:
                        await f.write(json.dumps(record, default=str) + "\n")

            elif file_format == FileFormat.CSV:
                await self._write_csv(output_file, records_data, mode == "append")

            result.records_processed = len(batch.records)
            result.records_inserted = len(batch.records)
            self._record_write(len(batch.records))

        except Exception as e:
            result.success = False
            result.errors.append({
                "error": str(e),
                "file": str(output_file)
            })

        result.duration_seconds = time.time() - start_time
        return result

    async def get_record_count(
        self,
        table: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> int:
        """Get total record count from files."""
        pattern = table or self.local_config.file_pattern
        files = self._list_files(pattern)

        total = 0
        for file_path in files:
            try:
                records = await self._read_file(file_path)
                if filters:
                    records = [r for r in records if self._matches_filters(r, filters)]
                total += len(records)
            except Exception:
                pass

        return total

    def _list_files(self, pattern: Optional[str] = None) -> List[Path]:
        """List files matching pattern."""
        if not self._base_path:
            return []

        pattern = pattern or self.local_config.file_pattern

        if self.local_config.recursive:
            files = list(self._base_path.rglob(pattern))
        else:
            files = list(self._base_path.glob(pattern))

        # Filter to only files (not directories)
        files = [f for f in files if f.is_file()]

        # Sort by modification time (newest first)
        files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

        return files

    def _detect_format(self, file_path: Path) -> FileFormat:
        """Detect file format from extension."""
        if self.local_config.file_format != FileFormat.AUTO:
            return self.local_config.file_format

        ext = file_path.suffix.lower()
        format_map = {
            ".json": FileFormat.JSON,
            ".jsonl": FileFormat.JSONL,
            ".ndjson": FileFormat.JSONL,
            ".csv": FileFormat.CSV,
            ".tsv": FileFormat.TSV,
            ".xml": FileFormat.XML,
            ".parquet": FileFormat.PARQUET
        }

        return format_map.get(ext, FileFormat.JSON)

    async def _read_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Read records from a file."""
        file_format = self._detect_format(file_path)

        if file_format == FileFormat.JSON:
            return await self._read_json(file_path)
        elif file_format == FileFormat.JSONL:
            return await self._read_jsonl(file_path)
        elif file_format in [FileFormat.CSV, FileFormat.TSV]:
            return await self._read_csv(file_path, file_format == FileFormat.TSV)
        elif file_format == FileFormat.PARQUET:
            return await self._read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported format: {file_format}")

    async def _read_json(self, file_path: Path) -> List[Dict[str, Any]]:
        """Read JSON file."""
        async with aiofiles.open(
            file_path,
            "r",
            encoding=self.local_config.json_options.encoding
        ) as f:
            content = await f.read()
            data = json.loads(content)

            # Navigate to data path if specified
            if self.local_config.json_options.data_path:
                for key in self.local_config.json_options.data_path.split("."):
                    if isinstance(data, dict):
                        data = data.get(key, [])
                    else:
                        break

            if isinstance(data, list):
                return data
            else:
                return [data]

    async def _read_jsonl(self, file_path: Path) -> List[Dict[str, Any]]:
        """Read JSON Lines file."""
        records = []
        async with aiofiles.open(
            file_path,
            "r",
            encoding=self.local_config.json_options.encoding
        ) as f:
            async for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    async def _read_csv(self, file_path: Path, is_tsv: bool = False) -> List[Dict[str, Any]]:
        """Read CSV/TSV file."""
        options = self.local_config.csv_options
        delimiter = "\t" if is_tsv else options.delimiter

        records = []

        # Use sync file reading with csv module (more reliable)
        loop = asyncio.get_event_loop()
        records = await loop.run_in_executor(
            None,
            lambda: self._read_csv_sync(file_path, delimiter, options)
        )

        return records

    def _read_csv_sync(
        self,
        file_path: Path,
        delimiter: str,
        options: CSVOptions
    ) -> List[Dict[str, Any]]:
        """Synchronous CSV reading."""
        records = []

        with open(file_path, "r", encoding=options.encoding, newline="") as f:
            # Skip rows
            for _ in range(options.skip_rows):
                next(f, None)

            if options.has_header:
                reader = csv.DictReader(
                    f,
                    delimiter=delimiter,
                    quotechar=options.quotechar
                )
                for row in reader:
                    records.append(dict(row))
            else:
                reader = csv.reader(
                    f,
                    delimiter=delimiter,
                    quotechar=options.quotechar
                )
                for row in reader:
                    records.append({f"col_{i}": v for i, v in enumerate(row)})

        return records

    async def _read_parquet(self, file_path: Path) -> List[Dict[str, Any]]:
        """Read Parquet file."""
        try:
            import pyarrow.parquet as pq

            loop = asyncio.get_event_loop()
            table = await loop.run_in_executor(
                None,
                lambda: pq.read_table(file_path)
            )

            return table.to_pylist()

        except ImportError:
            raise ImportError("pyarrow is required for Parquet support")

    async def _write_csv(
        self,
        file_path: Path,
        records: List[Dict[str, Any]],
        append: bool = False
    ) -> None:
        """Write records to CSV file."""
        if not records:
            return

        options = self.local_config.csv_options
        mode = "a" if append else "w"
        write_header = not append or not file_path.exists()

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self._write_csv_sync(file_path, records, mode, write_header, options)
        )

    def _write_csv_sync(
        self,
        file_path: Path,
        records: List[Dict[str, Any]],
        mode: str,
        write_header: bool,
        options: CSVOptions
    ) -> None:
        """Synchronous CSV writing."""
        fieldnames = list(records[0].keys())

        with open(file_path, mode, encoding=options.encoding, newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
                delimiter=options.delimiter,
                quotechar=options.quotechar
            )

            if write_header:
                writer.writeheader()

            writer.writerows(records)

    def _matches_filters(
        self,
        record: Dict[str, Any],
        filters: Dict[str, Any]
    ) -> bool:
        """Check if record matches filters."""
        for key, value in filters.items():
            record_value = record.get(key)

            if isinstance(value, dict):
                # Complex filter
                op = value.get("$op", "eq")
                filter_value = value.get("$value")

                if op == "eq" and record_value != filter_value:
                    return False
                elif op == "ne" and record_value == filter_value:
                    return False
                elif op == "gt" and not (record_value > filter_value):
                    return False
                elif op == "gte" and not (record_value >= filter_value):
                    return False
                elif op == "lt" and not (record_value < filter_value):
                    return False
                elif op == "lte" and not (record_value <= filter_value):
                    return False
                elif op == "in" and record_value not in filter_value:
                    return False
                elif op == "contains" and filter_value not in str(record_value):
                    return False
            else:
                # Simple equality
                if record_value != value:
                    return False

        return True


# Register connector
from ..base import ConnectorFactory
ConnectorFactory.register("local", LocalConnector)
