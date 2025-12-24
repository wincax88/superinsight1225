"""
Base Connector Module.

Provides abstract base class and common utilities for data source connectors.
"""

import asyncio
import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator, Dict, Generator, List, Optional, Union

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ConnectionStatus(str, Enum):
    """Connection status enumeration."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    RECONNECTING = "reconnecting"


class OperationType(str, Enum):
    """Data operation type."""
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    UPSERT = "upsert"


@dataclass
class DataRecord:
    """Represents a single data record."""
    id: str
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[datetime] = None
    operation: OperationType = OperationType.UPSERT
    version: Optional[str] = None
    hash: Optional[str] = None

    def compute_hash(self) -> str:
        """Compute hash of record data for change detection."""
        import json
        data_str = json.dumps(self.data, sort_keys=True, default=str)
        self.hash = hashlib.sha256(data_str.encode()).hexdigest()
        return self.hash


@dataclass
class DataBatch:
    """Represents a batch of data records."""
    records: List[DataRecord]
    batch_id: str = ""
    source_id: str = ""
    schema_name: Optional[str] = None
    table_name: Optional[str] = None
    total_count: int = 0
    offset: int = 0
    has_more: bool = False
    checkpoint: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.batch_id:
            import uuid
            self.batch_id = str(uuid.uuid4())
        if self.total_count == 0:
            self.total_count = len(self.records)


@dataclass
class SyncResult:
    """Result of a sync operation."""
    success: bool
    records_processed: int = 0
    records_inserted: int = 0
    records_updated: int = 0
    records_deleted: int = 0
    records_failed: int = 0
    records_skipped: int = 0
    errors: List[Dict[str, Any]] = field(default_factory=list)
    duration_seconds: float = 0.0
    checkpoint: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConnectorConfig(BaseModel):
    """Base configuration for connectors."""
    name: str
    description: Optional[str] = None
    enabled: bool = True

    # Connection settings
    connection_timeout: int = Field(default=30, ge=1)
    read_timeout: int = Field(default=60, ge=1)
    max_retries: int = Field(default=3, ge=0)
    retry_delay: int = Field(default=5, ge=1)

    # Pool settings
    pool_size: int = Field(default=5, ge=1)
    max_overflow: int = Field(default=10, ge=0)

    # Batch settings
    batch_size: int = Field(default=1000, ge=1)
    max_batch_size: int = Field(default=10000, ge=1)

    # Extra configuration
    extra: Dict[str, Any] = Field(default_factory=dict)


class BaseConnector(ABC):
    """
    Abstract base class for data source connectors.

    Provides common interface for connecting to and syncing data from
    various data sources (databases, APIs, file systems).
    """

    def __init__(self, config: ConnectorConfig):
        """
        Initialize connector.

        Args:
            config: Connector configuration
        """
        self.config = config
        self._status = ConnectionStatus.DISCONNECTED
        self._last_error: Optional[Exception] = None
        self._connected_at: Optional[datetime] = None
        self._stats = {
            "total_reads": 0,
            "total_writes": 0,
            "total_errors": 0,
            "bytes_read": 0,
            "bytes_written": 0
        }

    @property
    def status(self) -> ConnectionStatus:
        """Get current connection status."""
        return self._status

    @property
    def is_connected(self) -> bool:
        """Check if connector is connected."""
        return self._status == ConnectionStatus.CONNECTED

    @property
    def last_error(self) -> Optional[Exception]:
        """Get last error."""
        return self._last_error

    @property
    def stats(self) -> Dict[str, Any]:
        """Get connector statistics."""
        return {
            **self._stats,
            "status": self._status.value,
            "connected_at": self._connected_at,
            "uptime_seconds": (
                (datetime.utcnow() - self._connected_at).total_seconds()
                if self._connected_at else 0
            )
        }

    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection to data source.

        Returns:
            True if connection successful
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from data source."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if connection is healthy.

        Returns:
            True if connection is healthy
        """
        pass

    @abstractmethod
    async def fetch_schema(self) -> Dict[str, Any]:
        """
        Fetch schema information from data source.

        Returns:
            Schema information dictionary
        """
        pass

    @abstractmethod
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
        Fetch data from source.

        Args:
            query: Custom query (if supported)
            table: Table/collection name
            filters: Filter conditions
            limit: Maximum records to fetch
            offset: Offset for pagination
            incremental_field: Field for incremental sync
            incremental_value: Last sync value for incremental sync

        Returns:
            DataBatch containing fetched records
        """
        pass

    @abstractmethod
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
        Stream data from source in batches.

        Args:
            query: Custom query (if supported)
            table: Table/collection name
            filters: Filter conditions
            batch_size: Records per batch
            incremental_field: Field for incremental sync
            incremental_value: Last sync value for incremental sync

        Yields:
            DataBatch objects
        """
        pass

    @abstractmethod
    async def write_data(
        self,
        batch: DataBatch,
        mode: str = "upsert"
    ) -> SyncResult:
        """
        Write data to destination.

        Args:
            batch: DataBatch to write
            mode: Write mode (insert, update, upsert, delete)

        Returns:
            SyncResult with write statistics
        """
        pass

    @abstractmethod
    async def get_record_count(
        self,
        table: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Get total record count.

        Args:
            table: Table/collection name
            filters: Filter conditions

        Returns:
            Record count
        """
        pass

    async def test_connection(self) -> Dict[str, Any]:
        """
        Test connection and return diagnostics.

        Returns:
            Dictionary with connection test results
        """
        import time
        start_time = time.time()

        result = {
            "success": False,
            "status": self._status.value,
            "latency_ms": 0,
            "error": None,
            "details": {}
        }

        try:
            if not self.is_connected:
                await self.connect()

            healthy = await self.health_check()
            result["success"] = healthy
            result["status"] = self._status.value

            if healthy:
                # Get additional info
                try:
                    schema = await self.fetch_schema()
                    result["details"]["schema"] = {
                        "tables": len(schema.get("tables", [])),
                        "version": schema.get("version")
                    }
                except Exception:
                    pass

        except Exception as e:
            result["success"] = False
            result["error"] = str(e)
            self._last_error = e

        result["latency_ms"] = (time.time() - start_time) * 1000
        return result

    async def reconnect(self) -> bool:
        """
        Reconnect to data source.

        Returns:
            True if reconnection successful
        """
        self._status = ConnectionStatus.RECONNECTING
        await self.disconnect()
        return await self.connect()

    def _set_status(self, status: ConnectionStatus) -> None:
        """Update connection status."""
        self._status = status
        if status == ConnectionStatus.CONNECTED:
            self._connected_at = datetime.utcnow()
        logger.debug(f"Connector status changed to: {status.value}")

    def _record_error(self, error: Exception) -> None:
        """Record an error."""
        self._last_error = error
        self._stats["total_errors"] += 1
        logger.error(f"Connector error: {error}")

    def _record_read(self, record_count: int, byte_count: int = 0) -> None:
        """Record read operation stats."""
        self._stats["total_reads"] += record_count
        self._stats["bytes_read"] += byte_count

    def _record_write(self, record_count: int, byte_count: int = 0) -> None:
        """Record write operation stats."""
        self._stats["total_writes"] += record_count
        self._stats["bytes_written"] += byte_count

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()


class ConnectorFactory:
    """Factory for creating connector instances."""

    _connectors: Dict[str, type] = {}

    @classmethod
    def register(cls, connector_type: str, connector_class: type) -> None:
        """Register a connector type."""
        cls._connectors[connector_type] = connector_class
        logger.info(f"Registered connector type: {connector_type}")

    @classmethod
    def create(
        cls,
        connector_type: str,
        config: Union[ConnectorConfig, Dict[str, Any]]
    ) -> BaseConnector:
        """
        Create a connector instance.

        Args:
            connector_type: Type of connector
            config: Connector configuration

        Returns:
            Connector instance

        Raises:
            ValueError: If connector type not registered
        """
        if connector_type not in cls._connectors:
            raise ValueError(f"Unknown connector type: {connector_type}")

        connector_class = cls._connectors[connector_type]

        if isinstance(config, dict):
            config = ConnectorConfig(**config)

        return connector_class(config)

    @classmethod
    def list_types(cls) -> List[str]:
        """List registered connector types."""
        return list(cls._connectors.keys())
