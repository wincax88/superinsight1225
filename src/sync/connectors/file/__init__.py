"""
File Connectors Module.

Provides connectors for FTP, SFTP, S3, and local file systems.
"""

from .local_connector import (
    LocalConnector,
    LocalConnectorConfig,
    FileFormat,
    FileWatchMode,
    CSVOptions,
    JSONOptions,
)

from .s3_connector import (
    S3Connector,
    S3ConnectorConfig,
    S3Provider,
    S3AuthConfig,
    FileFormat as S3FileFormat,
)

__all__ = [
    # Local connector
    "LocalConnector",
    "LocalConnectorConfig",
    "FileFormat",
    "FileWatchMode",
    "CSVOptions",
    "JSONOptions",
    # S3 connector
    "S3Connector",
    "S3ConnectorConfig",
    "S3Provider",
    "S3AuthConfig",
]
