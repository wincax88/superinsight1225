"""
Base classes for data extraction in SuperInsight Platform.

Provides secure, read-only data extraction from various sources.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import ssl
from urllib.parse import urlparse

# Optional database drivers - import only if available
try:
    import pymysql
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False

try:
    import psycopg2
    POSTGRESQL_AVAILABLE = True
except ImportError:
    POSTGRESQL_AVAILABLE = False

try:
    import cx_Oracle
    ORACLE_AVAILABLE = True
except ImportError:
    ORACLE_AVAILABLE = False

from src.models.document import Document

logger = logging.getLogger(__name__)


class SourceType(str, Enum):
    """Supported data source types."""
    DATABASE = "database"
    FILE = "file"
    API = "api"


class DatabaseType(str, Enum):
    """Supported database types."""
    MYSQL = "mysql"
    POSTGRESQL = "postgresql"
    ORACLE = "oracle"


class FileType(str, Enum):
    """Supported file types."""
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    HTML = "html"


@dataclass
class ConnectionConfig:
    """Base configuration for data source connections."""
    source_type: SourceType
    connection_timeout: int = 30
    read_timeout: int = 60
    use_ssl: bool = True
    verify_ssl: bool = True
    
    def validate(self) -> bool:
        """Validate connection configuration."""
        if self.connection_timeout <= 0:
            raise ValueError("connection_timeout must be positive")
        if self.read_timeout <= 0:
            raise ValueError("read_timeout must be positive")
        return True


@dataclass
class DatabaseConfig:
    """Database connection configuration."""
    host: str
    port: int
    database: str
    username: str
    password: str
    database_type: DatabaseType
    read_only: bool = True
    connection_timeout: int = 30
    read_timeout: int = 60
    use_ssl: bool = True
    verify_ssl: bool = True
    
    def __post_init__(self):
        self.source_type = SourceType.DATABASE
        self.validate()
    
    def validate(self) -> bool:
        """Validate database configuration."""
        if self.connection_timeout <= 0:
            raise ValueError("connection_timeout must be positive")
        if self.read_timeout <= 0:
            raise ValueError("read_timeout must be positive")
        
        if not self.host:
            raise ValueError("host is required")
        if not (1 <= self.port <= 65535):
            raise ValueError("port must be between 1 and 65535")
        if not self.database:
            raise ValueError("database name is required")
        if not self.username:
            raise ValueError("username is required")
        if not self.password:
            raise ValueError("password is required")
        if not self.read_only:
            logger.warning("Database connection is not read-only - this may violate security requirements")
        
        return True
    
    def get_connection_string(self) -> str:
        """Generate database connection string."""
        if self.database_type == DatabaseType.MYSQL:
            if not MYSQL_AVAILABLE:
                raise ImportError("pymysql is required for MySQL connections. Install with: pip install pymysql")
            protocol = "mysql+pymysql"
        elif self.database_type == DatabaseType.POSTGRESQL:
            if not POSTGRESQL_AVAILABLE:
                raise ImportError("psycopg2 is required for PostgreSQL connections. Install with: pip install psycopg2-binary")
            protocol = "postgresql+psycopg2"
        elif self.database_type == DatabaseType.ORACLE:
            if not ORACLE_AVAILABLE:
                raise ImportError("cx_Oracle is required for Oracle connections. Install with: pip install cx_Oracle")
            protocol = "oracle+cx_oracle"
        else:
            raise ValueError(f"Unsupported database type: {self.database_type}")
        
        ssl_params = ""
        if self.use_ssl:
            if self.database_type == DatabaseType.MYSQL:
                ssl_params = "?ssl_disabled=false"
            elif self.database_type == DatabaseType.POSTGRESQL:
                ssl_params = "?sslmode=require"
        
        return f"{protocol}://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}{ssl_params}"


@dataclass
class FileConfig:
    """File extraction configuration."""
    file_path: str
    file_type: FileType
    encoding: str = "utf-8"
    connection_timeout: int = 30
    read_timeout: int = 60
    use_ssl: bool = True
    verify_ssl: bool = True
    
    def __post_init__(self):
        self.source_type = SourceType.FILE
        self.validate()
    
    def validate(self) -> bool:
        """Validate file configuration."""
        if self.connection_timeout <= 0:
            raise ValueError("connection_timeout must be positive")
        if self.read_timeout <= 0:
            raise ValueError("read_timeout must be positive")
        
        if not self.file_path:
            raise ValueError("file_path is required")
        
        return True


@dataclass
class APIConfig:
    """API extraction configuration."""
    base_url: str
    headers: Dict[str, str]
    auth_token: Optional[str] = None
    connection_timeout: int = 30
    read_timeout: int = 60
    use_ssl: bool = True
    verify_ssl: bool = True
    
    def __post_init__(self):
        self.source_type = SourceType.API
        self.validate()
    
    def validate(self) -> bool:
        """Validate API configuration."""
        if self.connection_timeout <= 0:
            raise ValueError("connection_timeout must be positive")
        if self.read_timeout <= 0:
            raise ValueError("read_timeout must be positive")
        
        if not self.base_url:
            raise ValueError("base_url is required")
        
        # Validate URL format
        parsed = urlparse(self.base_url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError("base_url must be a valid URL")
        
        # Enforce HTTPS for security
        if parsed.scheme != "https" and not self.base_url.startswith("http://localhost"):
            raise ValueError("API connections must use HTTPS (except localhost)")
        
        return True


class ExtractionResult:
    """Result of data extraction operation."""
    
    def __init__(self, success: bool, documents: List[Document] = None, error: str = None):
        self.success = success
        self.documents = documents or []
        self.error = error
        self.extracted_count = len(self.documents)
    
    def __repr__(self) -> str:
        if self.success:
            return f"ExtractionResult(success=True, extracted_count={self.extracted_count})"
        else:
            return f"ExtractionResult(success=False, error='{self.error}')"


class SecurityValidator:
    """Security validation for data extraction operations."""
    
    @staticmethod
    def validate_read_only_connection(config) -> bool:
        """Validate that connection is configured for read-only access."""
        if hasattr(config, 'read_only') and hasattr(config, 'database_type'):
            if not config.read_only:
                raise ValueError("Database connections must be read-only")
        return True
    
    @staticmethod
    def validate_ssl_configuration(config) -> bool:
        """Validate SSL/TLS configuration."""
        if not config.use_ssl:
            logger.warning("SSL/TLS is disabled - connection may not be secure")
        
        if config.use_ssl and not config.verify_ssl:
            logger.warning("SSL certificate verification is disabled")
        
        return True
    
    @staticmethod
    def validate_connection_limits(config) -> bool:
        """Validate connection timeout limits."""
        max_timeout = 300  # 5 minutes
        
        if config.connection_timeout > max_timeout:
            raise ValueError(f"connection_timeout cannot exceed {max_timeout} seconds")
        
        if config.read_timeout > max_timeout:
            raise ValueError(f"read_timeout cannot exceed {max_timeout} seconds")
        
        return True


class BaseExtractor(ABC):
    """Base class for all data extractors."""
    
    def __init__(self, config):
        self.config = config
        self._validate_security()
    
    def _validate_security(self) -> None:
        """Validate security requirements for the extractor."""
        SecurityValidator.validate_read_only_connection(self.config)
        SecurityValidator.validate_ssl_configuration(self.config)
        SecurityValidator.validate_connection_limits(self.config)
    
    @abstractmethod
    def test_connection(self) -> bool:
        """Test connection to the data source."""
        pass
    
    @abstractmethod
    def extract_data(self, query: Optional[str] = None, **kwargs) -> ExtractionResult:
        """Extract data from the source."""
        pass
    
    def validate_connection(self) -> bool:
        """Validate connection configuration and test connectivity."""
        try:
            self.config.validate()
            return self.test_connection()
        except Exception as e:
            logger.error(f"Connection validation failed: {e}")
            return False


class DataExtractor:
    """Main data extractor class that coordinates different extraction types."""
    
    def __init__(self):
        self._extractors: Dict[str, BaseExtractor] = {}
    
    def register_extractor(self, source_id: str, extractor: BaseExtractor) -> None:
        """Register an extractor for a specific source."""
        self._extractors[source_id] = extractor
        logger.info(f"Registered extractor for source: {source_id}")
    
    def get_extractor(self, source_id: str) -> Optional[BaseExtractor]:
        """Get registered extractor by source ID."""
        return self._extractors.get(source_id)
    
    def extract_from_source(self, source_id: str, **kwargs) -> ExtractionResult:
        """Extract data from a registered source."""
        extractor = self.get_extractor(source_id)
        if not extractor:
            return ExtractionResult(
                success=False,
                error=f"No extractor registered for source: {source_id}"
            )
        
        try:
            return extractor.extract_data(**kwargs)
        except Exception as e:
            logger.error(f"Extraction failed for source {source_id}: {e}")
            return ExtractionResult(
                success=False,
                error=str(e)
            )
    
    def validate_all_connections(self) -> Dict[str, bool]:
        """Validate all registered extractors."""
        results = {}
        for source_id, extractor in self._extractors.items():
            try:
                results[source_id] = extractor.validate_connection()
            except Exception as e:
                logger.error(f"Validation failed for {source_id}: {e}")
                results[source_id] = False
        
        return results