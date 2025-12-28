"""
Database CDC (Change Data Capture) Module.

Provides real-time change detection and capture for databases,
supporting MySQL binlog, PostgreSQL WAL, and polling-based CDC.
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Set

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CDCOperation(str, Enum):
    """CDC operation types."""
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    TRUNCATE = "truncate"
    DDL = "ddl"


class CDCMode(str, Enum):
    """CDC capture modes."""
    BINLOG = "binlog"      # MySQL binary log
    WAL = "wal"            # PostgreSQL WAL
    OPLOG = "oplog"        # MongoDB oplog
    POLLING = "polling"    # Polling-based CDC
    TRIGGER = "trigger"    # Trigger-based CDC


@dataclass
class ChangeEvent:
    """Represents a single change event."""
    id: str
    operation: CDCOperation
    table: str
    schema: Optional[str] = None
    database: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    before: Optional[Dict[str, Any]] = None  # Previous state (for updates/deletes)
    after: Optional[Dict[str, Any]] = None   # New state (for inserts/updates)
    primary_key: Optional[Dict[str, Any]] = None
    position: Optional[Dict[str, Any]] = None  # Position for resumption
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "operation": self.operation.value,
            "table": self.table,
            "schema": self.schema,
            "database": self.database,
            "timestamp": self.timestamp.isoformat(),
            "before": self.before,
            "after": self.after,
            "primary_key": self.primary_key,
            "position": self.position,
            "metadata": self.metadata
        }


@dataclass
class CDCPosition:
    """Position marker for CDC stream."""
    source: str
    position: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "position": self.position,
            "timestamp": self.timestamp.isoformat()
        }


class CDCConfig(BaseModel):
    """Base CDC configuration."""
    mode: CDCMode
    name: str
    tables: List[str] = Field(default_factory=list)  # Tables to capture
    exclude_tables: List[str] = Field(default_factory=list)
    operations: List[CDCOperation] = Field(
        default_factory=lambda: [CDCOperation.INSERT, CDCOperation.UPDATE, CDCOperation.DELETE]
    )

    # Connection settings
    host: str = "localhost"
    port: int = 3306
    database: str = ""
    username: Optional[str] = None
    password: Optional[str] = None

    # Capture settings
    batch_size: int = 1000
    poll_interval_seconds: float = 1.0
    include_schema: bool = True

    # Position tracking
    position_file: Optional[str] = None
    start_position: Optional[Dict[str, Any]] = None


class BaseCDC(ABC):
    """
    Abstract base class for CDC implementations.

    Provides common interface for different CDC mechanisms.
    """

    def __init__(self, config: CDCConfig):
        self.config = config
        self._running = False
        self._position: Optional[CDCPosition] = None
        self._handlers: List[Callable[[ChangeEvent], None]] = []
        self._error_handlers: List[Callable[[Exception], None]] = []
        self._stats = {
            "events_captured": 0,
            "events_processed": 0,
            "events_failed": 0,
            "started_at": None,
            "last_event_at": None
        }

    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the data source."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the data source."""
        pass

    @abstractmethod
    async def start_capture(self) -> None:
        """Start capturing changes."""
        pass

    @abstractmethod
    async def stop_capture(self) -> None:
        """Stop capturing changes."""
        pass

    @abstractmethod
    async def get_changes(
        self,
        from_position: Optional[CDCPosition] = None
    ) -> AsyncIterator[ChangeEvent]:
        """
        Get changes from the source.

        Args:
            from_position: Position to start from (optional)

        Yields:
            ChangeEvent objects
        """
        pass

    def on_change(self, handler: Callable[[ChangeEvent], None]) -> None:
        """Register a change event handler."""
        self._handlers.append(handler)

    def on_error(self, handler: Callable[[Exception], None]) -> None:
        """Register an error handler."""
        self._error_handlers.append(handler)

    async def emit_event(self, event: ChangeEvent) -> None:
        """Emit event to all handlers."""
        self._stats["events_captured"] += 1
        self._stats["last_event_at"] = datetime.utcnow()

        for handler in self._handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
                self._stats["events_processed"] += 1
            except Exception as e:
                self._stats["events_failed"] += 1
                await self._handle_error(e)

    async def _handle_error(self, error: Exception) -> None:
        """Handle errors."""
        for handler in self._error_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(error)
                else:
                    handler(error)
            except Exception:
                pass

    @property
    def stats(self) -> Dict[str, Any]:
        """Get capture statistics."""
        return {
            **self._stats,
            "running": self._running,
            "position": self._position.to_dict() if self._position else None
        }

    def _should_capture_table(self, table: str) -> bool:
        """Check if table should be captured."""
        if self.config.exclude_tables and table in self.config.exclude_tables:
            return False
        if self.config.tables:
            return table in self.config.tables
        return True

    def _should_capture_operation(self, operation: CDCOperation) -> bool:
        """Check if operation should be captured."""
        return operation in self.config.operations


class MySQLBinlogCDC(BaseCDC):
    """
    MySQL binlog-based CDC implementation.

    Uses mysql-replication library to capture changes from MySQL binary log.
    """

    def __init__(self, config: CDCConfig):
        super().__init__(config)
        self._stream = None
        self._server_id = 100

    async def connect(self) -> bool:
        """Connect to MySQL and setup binlog streaming."""
        try:
            from pymysqlreplication import BinLogStreamReader
            from pymysqlreplication.row_event import (
                WriteRowsEvent,
                UpdateRowsEvent,
                DeleteRowsEvent,
            )

            connection_settings = {
                "host": self.config.host,
                "port": self.config.port,
                "user": self.config.username,
                "passwd": self.config.password,
            }

            # Start position
            resume_stream = False
            log_file = None
            log_pos = None

            if self.config.start_position:
                log_file = self.config.start_position.get("log_file")
                log_pos = self.config.start_position.get("log_pos")
                resume_stream = True

            self._stream = BinLogStreamReader(
                connection_settings=connection_settings,
                server_id=self._server_id,
                only_events=[WriteRowsEvent, UpdateRowsEvent, DeleteRowsEvent],
                only_schemas=[self.config.database] if self.config.database else None,
                only_tables=self.config.tables if self.config.tables else None,
                resume_stream=resume_stream,
                log_file=log_file,
                log_pos=log_pos,
                blocking=True,
            )

            logger.info("Connected to MySQL binlog")
            return True

        except ImportError:
            logger.error("pymysqlreplication is required for MySQL CDC")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to MySQL binlog: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from MySQL binlog."""
        if self._stream:
            self._stream.close()
            self._stream = None
        logger.info("Disconnected from MySQL binlog")

    async def start_capture(self) -> None:
        """Start capturing changes from binlog."""
        from pymysqlreplication.row_event import (
            WriteRowsEvent,
            UpdateRowsEvent,
            DeleteRowsEvent,
        )

        self._running = True
        self._stats["started_at"] = datetime.utcnow()

        loop = asyncio.get_event_loop()

        while self._running and self._stream:
            try:
                # Run blocking binlog read in executor
                event = await loop.run_in_executor(
                    None,
                    lambda: next(self._stream, None)
                )

                if event is None:
                    await asyncio.sleep(0.1)
                    continue

                # Determine operation type
                if isinstance(event, WriteRowsEvent):
                    operation = CDCOperation.INSERT
                elif isinstance(event, UpdateRowsEvent):
                    operation = CDCOperation.UPDATE
                elif isinstance(event, DeleteRowsEvent):
                    operation = CDCOperation.DELETE
                else:
                    continue

                if not self._should_capture_operation(operation):
                    continue

                # Process rows
                for row in event.rows:
                    change_event = self._create_change_event(
                        operation, event, row
                    )
                    await self.emit_event(change_event)

                # Update position
                self._position = CDCPosition(
                    source="mysql_binlog",
                    position={
                        "log_file": self._stream.log_file,
                        "log_pos": self._stream.log_pos
                    }
                )

            except StopIteration:
                await asyncio.sleep(0.1)
            except Exception as e:
                await self._handle_error(e)
                await asyncio.sleep(1)

    async def stop_capture(self) -> None:
        """Stop capturing changes."""
        self._running = False

    async def get_changes(
        self,
        from_position: Optional[CDCPosition] = None
    ) -> AsyncIterator[ChangeEvent]:
        """Get changes from binlog."""
        # This is handled by start_capture in streaming mode
        # For batch mode, would need different implementation
        raise NotImplementedError("Use start_capture for streaming CDC")

    def _create_change_event(
        self,
        operation: CDCOperation,
        binlog_event: Any,
        row: Dict[str, Any]
    ) -> ChangeEvent:
        """Create ChangeEvent from binlog row event."""
        import uuid

        before = None
        after = None

        if operation == CDCOperation.INSERT:
            after = row.get("values", row)
        elif operation == CDCOperation.UPDATE:
            before = row.get("before_values", {})
            after = row.get("after_values", {})
        elif operation == CDCOperation.DELETE:
            before = row.get("values", row)

        return ChangeEvent(
            id=f"mysql_{uuid.uuid4().hex[:12]}",
            operation=operation,
            table=binlog_event.table,
            schema=binlog_event.schema,
            database=binlog_event.schema,
            timestamp=datetime.utcnow(),
            before=before,
            after=after,
            position={
                "log_file": self._stream.log_file if self._stream else None,
                "log_pos": self._stream.log_pos if self._stream else None
            },
            metadata={
                "server_id": binlog_event.packet.server_id,
                "event_type": type(binlog_event).__name__
            }
        )


class PostgreSQLWALCDC(BaseCDC):
    """
    PostgreSQL WAL-based CDC implementation.

    Uses logical replication to capture changes from PostgreSQL.
    """

    def __init__(self, config: CDCConfig):
        super().__init__(config)
        self._connection = None
        self._slot_name = f"cdc_{config.name}"
        self._publication_name = f"pub_{config.name}"

    async def connect(self) -> bool:
        """Connect to PostgreSQL for logical replication."""
        try:
            import psycopg2
            from psycopg2 import extras

            connection_string = (
                f"host={self.config.host} "
                f"port={self.config.port} "
                f"dbname={self.config.database} "
                f"user={self.config.username} "
                f"password={self.config.password} "
                "replication=database"
            )

            self._connection = psycopg2.connect(
                connection_string,
                connection_factory=extras.LogicalReplicationConnection
            )

            # Create replication slot if not exists
            cursor = self._connection.cursor()
            try:
                cursor.create_replication_slot(
                    self._slot_name,
                    output_plugin='pgoutput'
                )
            except psycopg2.ProgrammingError:
                # Slot already exists
                pass

            logger.info("Connected to PostgreSQL WAL")
            return True

        except ImportError:
            logger.error("psycopg2 is required for PostgreSQL CDC")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from PostgreSQL."""
        if self._connection:
            self._connection.close()
            self._connection = None
        logger.info("Disconnected from PostgreSQL")

    async def start_capture(self) -> None:
        """Start capturing changes from WAL."""
        import psycopg2

        self._running = True
        self._stats["started_at"] = datetime.utcnow()

        cursor = self._connection.cursor()

        # Start replication
        start_lsn = None
        if self.config.start_position:
            start_lsn = self.config.start_position.get("lsn")

        cursor.start_replication(
            slot_name=self._slot_name,
            start_lsn=start_lsn,
            options={
                'publication_names': self._publication_name,
                'proto_version': '1'
            }
        )

        loop = asyncio.get_event_loop()

        while self._running:
            try:
                # Read message from replication stream
                msg = await loop.run_in_executor(
                    None,
                    lambda: cursor.read_message()
                )

                if msg is None:
                    await asyncio.sleep(0.1)
                    continue

                # Parse and process message
                change_events = self._parse_wal_message(msg)

                for event in change_events:
                    if self._should_capture_table(event.table):
                        if self._should_capture_operation(event.operation):
                            await self.emit_event(event)

                # Send feedback
                msg.cursor.send_feedback(flush_lsn=msg.data_start)

                # Update position
                self._position = CDCPosition(
                    source="postgresql_wal",
                    position={
                        "lsn": str(msg.data_start)
                    }
                )

            except psycopg2.DatabaseError as e:
                await self._handle_error(e)
                await asyncio.sleep(1)

    async def stop_capture(self) -> None:
        """Stop capturing changes."""
        self._running = False

    async def get_changes(
        self,
        from_position: Optional[CDCPosition] = None
    ) -> AsyncIterator[ChangeEvent]:
        """Get changes from WAL."""
        raise NotImplementedError("Use start_capture for streaming CDC")

    def _parse_wal_message(self, msg: Any) -> List[ChangeEvent]:
        """Parse WAL message into change events."""
        import uuid

        events = []
        # This is a simplified parser - actual implementation would need
        # to properly decode pgoutput protocol

        payload = msg.payload

        # Parse the payload based on pgoutput format
        # This is a placeholder - actual parsing is complex

        return events


class PollingCDC(BaseCDC):
    """
    Polling-based CDC implementation.

    Uses timestamp or version columns to detect changes.
    Works with any database but has higher latency.
    """

    def __init__(self, config: CDCConfig):
        super().__init__(config)
        self._connection = None
        self._timestamp_column: str = "updated_at"
        self._version_column: Optional[str] = None
        self._last_values: Dict[str, Any] = {}

    async def connect(self) -> bool:
        """Connect to database."""
        try:
            import asyncpg

            self._connection = await asyncpg.connect(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.username,
                password=self.config.password
            )

            logger.info("Connected to database for polling CDC")
            return True

        except ImportError:
            logger.error("asyncpg is required for polling CDC")
            return False
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from database."""
        if self._connection:
            await self._connection.close()
            self._connection = None
        logger.info("Disconnected from database")

    async def start_capture(self) -> None:
        """Start polling for changes."""
        self._running = True
        self._stats["started_at"] = datetime.utcnow()

        # Initialize last values for each table
        for table in self.config.tables:
            self._last_values[table] = await self._get_max_timestamp(table)

        while self._running:
            try:
                for table in self.config.tables:
                    if not self._should_capture_table(table):
                        continue

                    changes = await self._poll_table(table)

                    for change in changes:
                        await self.emit_event(change)

                await asyncio.sleep(self.config.poll_interval_seconds)

            except Exception as e:
                await self._handle_error(e)
                await asyncio.sleep(1)

    async def stop_capture(self) -> None:
        """Stop polling."""
        self._running = False

    async def get_changes(
        self,
        from_position: Optional[CDCPosition] = None
    ) -> AsyncIterator[ChangeEvent]:
        """Get changes since position."""
        for table in self.config.tables:
            if not self._should_capture_table(table):
                continue

            since = None
            if from_position and from_position.position:
                since = from_position.position.get(table)

            changes = await self._poll_table(table, since=since)

            for change in changes:
                yield change

    async def _get_max_timestamp(self, table: str) -> Optional[datetime]:
        """Get maximum timestamp value from table."""
        query = f"""
            SELECT MAX({self._timestamp_column})
            FROM {table}
        """
        try:
            result = await self._connection.fetchval(query)
            return result
        except Exception:
            return None

    async def _poll_table(
        self,
        table: str,
        since: Optional[datetime] = None
    ) -> List[ChangeEvent]:
        """Poll table for changes."""
        import uuid

        events = []
        last_timestamp = since or self._last_values.get(table)

        if last_timestamp:
            query = f"""
                SELECT *
                FROM {table}
                WHERE {self._timestamp_column} > $1
                ORDER BY {self._timestamp_column}
                LIMIT {self.config.batch_size}
            """
            rows = await self._connection.fetch(query, last_timestamp)
        else:
            query = f"""
                SELECT *
                FROM {table}
                ORDER BY {self._timestamp_column}
                LIMIT {self.config.batch_size}
            """
            rows = await self._connection.fetch(query)

        for row in rows:
            row_dict = dict(row)

            # Determine operation (polling can only reliably detect changes)
            operation = CDCOperation.UPDATE

            event = ChangeEvent(
                id=f"poll_{uuid.uuid4().hex[:12]}",
                operation=operation,
                table=table,
                database=self.config.database,
                timestamp=row_dict.get(self._timestamp_column, datetime.utcnow()),
                after=row_dict,
                metadata={
                    "cdc_mode": "polling"
                }
            )
            events.append(event)

            # Update last timestamp
            if self._timestamp_column in row_dict:
                self._last_values[table] = row_dict[self._timestamp_column]

        return events


class CDCManager:
    """
    Manager for multiple CDC streams.

    Coordinates multiple CDC instances and provides unified interface.
    """

    def __init__(self):
        self._cdcs: Dict[str, BaseCDC] = {}
        self._tasks: Dict[str, asyncio.Task] = {}

    def register(self, cdc: BaseCDC) -> None:
        """Register a CDC instance."""
        self._cdcs[cdc.config.name] = cdc
        logger.info(f"Registered CDC: {cdc.config.name}")

    def unregister(self, name: str) -> None:
        """Unregister a CDC instance."""
        if name in self._cdcs:
            del self._cdcs[name]

    async def start_all(self) -> None:
        """Start all registered CDC instances."""
        for name, cdc in self._cdcs.items():
            if await cdc.connect():
                task = asyncio.create_task(cdc.start_capture())
                self._tasks[name] = task
                logger.info(f"Started CDC: {name}")

    async def stop_all(self) -> None:
        """Stop all CDC instances."""
        for name, cdc in self._cdcs.items():
            await cdc.stop_capture()
            await cdc.disconnect()

        for task in self._tasks.values():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._tasks.clear()
        logger.info("Stopped all CDC instances")

    async def start(self, name: str) -> bool:
        """Start a specific CDC instance."""
        cdc = self._cdcs.get(name)
        if not cdc:
            return False

        if await cdc.connect():
            task = asyncio.create_task(cdc.start_capture())
            self._tasks[name] = task
            return True

        return False

    async def stop(self, name: str) -> bool:
        """Stop a specific CDC instance."""
        cdc = self._cdcs.get(name)
        if not cdc:
            return False

        await cdc.stop_capture()
        await cdc.disconnect()

        if name in self._tasks:
            self._tasks[name].cancel()
            del self._tasks[name]

        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get stats for all CDC instances."""
        return {
            name: cdc.stats
            for name, cdc in self._cdcs.items()
        }


def create_cdc(config: CDCConfig) -> BaseCDC:
    """Factory function to create appropriate CDC instance."""
    if config.mode == CDCMode.BINLOG:
        return MySQLBinlogCDC(config)
    elif config.mode == CDCMode.WAL:
        return PostgreSQLWALCDC(config)
    elif config.mode == CDCMode.POLLING:
        return PollingCDC(config)
    else:
        raise ValueError(f"Unsupported CDC mode: {config.mode}")


__all__ = [
    "BaseCDC",
    "MySQLBinlogCDC",
    "PostgreSQLWALCDC",
    "PollingCDC",
    "CDCManager",
    "CDCConfig",
    "CDCMode",
    "CDCOperation",
    "CDCPosition",
    "ChangeEvent",
    "create_cdc",
]
