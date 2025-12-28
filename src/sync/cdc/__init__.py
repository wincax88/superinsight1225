"""
CDC (Change Data Capture) Module.

Provides Change Data Capture listeners for MySQL Binlog, PostgreSQL WAL,
MongoDB Oplog, and polling-based CDC.
"""

from .database_cdc import (
    BaseCDC,
    MySQLBinlogCDC,
    PostgreSQLWALCDC,
    PollingCDC,
    CDCManager,
    CDCConfig,
    CDCMode,
    CDCOperation,
    CDCPosition,
    ChangeEvent,
    create_cdc,
)

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
