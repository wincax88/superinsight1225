"""
Factory for creating data extractors based on configuration.

Provides a unified interface for creating different types of extractors.
"""

import logging
from typing import Optional, Dict, Any, Union
from urllib.parse import urlparse

from .base import (
    BaseExtractor, 
    DatabaseConfig, 
    FileConfig, 
    APIConfig,
    SourceType,
    DatabaseType,
    FileType
)
from .database import DatabaseExtractor
from .file import FileExtractor, WebExtractor, detect_file_type
from .api import APIExtractor, GraphQLExtractor, WebhookExtractor

logger = logging.getLogger(__name__)


class ExtractorFactory:
    """Factory class for creating data extractors."""
    
    @staticmethod
    def create_database_extractor(
        host: str,
        port: int,
        database: str,
        username: str,
        password: str,
        database_type: str,
        use_ssl: bool = True,
        connection_timeout: int = 30,
        read_timeout: int = 60
    ) -> DatabaseExtractor:
        """Create a database extractor."""
        try:
            # Convert string to enum
            db_type = DatabaseType(database_type.lower())
            
            config = DatabaseConfig(
                host=host,
                port=port,
                database=database,
                username=username,
                password=password,
                database_type=db_type,
                use_ssl=use_ssl,
                connection_timeout=connection_timeout,
                read_timeout=read_timeout,
                read_only=True
            )
            
            return DatabaseExtractor(config)
            
        except Exception as e:
            logger.error(f"Failed to create database extractor: {e}")
            raise
    
    @staticmethod
    def create_file_extractor(
        file_path: str,
        file_type: Optional[str] = None,
        encoding: str = "utf-8",
        use_ssl: bool = True,
        connection_timeout: int = 30,
        read_timeout: int = 60
    ) -> FileExtractor:
        """Create a file extractor."""
        try:
            # Auto-detect file type if not provided
            if file_type is None:
                detected_type = detect_file_type(file_path)
                if detected_type is None:
                    raise ValueError(f"Could not detect file type for: {file_path}")
                file_type_enum = detected_type
            else:
                file_type_enum = FileType(file_type.lower())
            
            config = FileConfig(
                file_path=file_path,
                file_type=file_type_enum,
                encoding=encoding,
                use_ssl=use_ssl,
                connection_timeout=connection_timeout,
                read_timeout=read_timeout
            )
            
            return FileExtractor(config)
            
        except Exception as e:
            logger.error(f"Failed to create file extractor: {e}")
            raise
    
    @staticmethod
    def create_web_extractor(
        base_url: str,
        max_pages: int = 10,
        connection_timeout: int = 30,
        read_timeout: int = 60
    ) -> WebExtractor:
        """Create a web content extractor."""
        try:
            return WebExtractor(
                base_url=base_url,
                max_pages=max_pages
            )
            
        except Exception as e:
            logger.error(f"Failed to create web extractor: {e}")
            raise
    
    @staticmethod
    def create_api_extractor(
        base_url: str,
        headers: Optional[Dict[str, str]] = None,
        auth_token: Optional[str] = None,
        use_ssl: bool = True,
        connection_timeout: int = 30,
        read_timeout: int = 60
    ) -> APIExtractor:
        """Create an API extractor."""
        try:
            config = APIConfig(
                base_url=base_url,
                headers=headers or {},
                auth_token=auth_token,
                use_ssl=use_ssl,
                connection_timeout=connection_timeout,
                read_timeout=read_timeout
            )
            
            return APIExtractor(config)
            
        except Exception as e:
            logger.error(f"Failed to create API extractor: {e}")
            raise
    
    @staticmethod
    def create_graphql_extractor(
        base_url: str,
        headers: Optional[Dict[str, str]] = None,
        auth_token: Optional[str] = None,
        use_ssl: bool = True,
        connection_timeout: int = 30,
        read_timeout: int = 60
    ) -> GraphQLExtractor:
        """Create a GraphQL API extractor."""
        try:
            config = APIConfig(
                base_url=base_url,
                headers=headers or {'Content-Type': 'application/json'},
                auth_token=auth_token,
                use_ssl=use_ssl,
                connection_timeout=connection_timeout,
                read_timeout=read_timeout
            )
            
            return GraphQLExtractor(config)
            
        except Exception as e:
            logger.error(f"Failed to create GraphQL extractor: {e}")
            raise
    
    @staticmethod
    def create_webhook_extractor(
        webhook_url: str,
        secret_key: Optional[str] = None
    ) -> WebhookExtractor:
        """Create a webhook extractor."""
        try:
            return WebhookExtractor(
                webhook_url=webhook_url,
                secret_key=secret_key
            )
            
        except Exception as e:
            logger.error(f"Failed to create webhook extractor: {e}")
            raise
    
    @staticmethod
    def create_from_config(config_dict: Dict[str, Any]) -> BaseExtractor:
        """Create extractor from configuration dictionary."""
        try:
            source_type = config_dict.get('source_type')
            
            if source_type == 'database':
                return ExtractorFactory.create_database_extractor(
                    host=config_dict['host'],
                    port=config_dict['port'],
                    database=config_dict['database'],
                    username=config_dict['username'],
                    password=config_dict['password'],
                    database_type=config_dict['database_type'],
                    use_ssl=config_dict.get('use_ssl', True),
                    connection_timeout=config_dict.get('connection_timeout', 30),
                    read_timeout=config_dict.get('read_timeout', 60)
                )
            
            elif source_type == 'file':
                return ExtractorFactory.create_file_extractor(
                    file_path=config_dict['file_path'],
                    file_type=config_dict.get('file_type'),
                    encoding=config_dict.get('encoding', 'utf-8'),
                    use_ssl=config_dict.get('use_ssl', True),
                    connection_timeout=config_dict.get('connection_timeout', 30),
                    read_timeout=config_dict.get('read_timeout', 60)
                )
            
            elif source_type == 'web':
                return ExtractorFactory.create_web_extractor(
                    base_url=config_dict['base_url'],
                    max_pages=config_dict.get('max_pages', 10),
                    connection_timeout=config_dict.get('connection_timeout', 30),
                    read_timeout=config_dict.get('read_timeout', 60)
                )
            
            elif source_type == 'api':
                return ExtractorFactory.create_api_extractor(
                    base_url=config_dict['base_url'],
                    headers=config_dict.get('headers'),
                    auth_token=config_dict.get('auth_token'),
                    use_ssl=config_dict.get('use_ssl', True),
                    connection_timeout=config_dict.get('connection_timeout', 30),
                    read_timeout=config_dict.get('read_timeout', 60)
                )
            
            elif source_type == 'graphql':
                return ExtractorFactory.create_graphql_extractor(
                    base_url=config_dict['base_url'],
                    headers=config_dict.get('headers'),
                    auth_token=config_dict.get('auth_token'),
                    use_ssl=config_dict.get('use_ssl', True),
                    connection_timeout=config_dict.get('connection_timeout', 30),
                    read_timeout=config_dict.get('read_timeout', 60)
                )
            
            elif source_type == 'webhook':
                return ExtractorFactory.create_webhook_extractor(
                    webhook_url=config_dict['webhook_url'],
                    secret_key=config_dict.get('secret_key')
                )
            
            else:
                raise ValueError(f"Unsupported source type: {source_type}")
                
        except Exception as e:
            logger.error(f"Failed to create extractor from config: {e}")
            raise
    
    @staticmethod
    def create_from_url(url: str, **kwargs) -> BaseExtractor:
        """Create extractor by auto-detecting type from URL."""
        try:
            parsed = urlparse(url)
            
            # Web URLs
            if parsed.scheme in ['http', 'https']:
                # Check if it's a file URL
                file_type = detect_file_type(url)
                if file_type:
                    return ExtractorFactory.create_file_extractor(
                        file_path=url,
                        file_type=file_type.value,
                        **kwargs
                    )
                else:
                    # Assume it's a web page
                    return ExtractorFactory.create_web_extractor(
                        base_url=url,
                        **kwargs
                    )
            
            # Database URLs
            elif parsed.scheme in ['mysql', 'postgresql', 'oracle']:
                # Parse database URL components
                return ExtractorFactory.create_database_extractor(
                    host=parsed.hostname,
                    port=parsed.port or 5432,
                    database=parsed.path.lstrip('/'),
                    username=parsed.username,
                    password=parsed.password,
                    database_type=parsed.scheme,
                    **kwargs
                )
            
            # Local files
            else:
                return ExtractorFactory.create_file_extractor(
                    file_path=url,
                    **kwargs
                )
                
        except Exception as e:
            logger.error(f"Failed to create extractor from URL {url}: {e}")
            raise


# Convenience functions
def create_mysql_extractor(host: str, port: int, database: str, 
                          username: str, password: str, **kwargs) -> DatabaseExtractor:
    """Create MySQL extractor."""
    return ExtractorFactory.create_database_extractor(
        host=host, port=port, database=database,
        username=username, password=password,
        database_type='mysql', **kwargs
    )


def create_postgresql_extractor(host: str, port: int, database: str,
                               username: str, password: str, **kwargs) -> DatabaseExtractor:
    """Create PostgreSQL extractor."""
    return ExtractorFactory.create_database_extractor(
        host=host, port=port, database=database,
        username=username, password=password,
        database_type='postgresql', **kwargs
    )


def create_oracle_extractor(host: str, port: int, database: str,
                           username: str, password: str, **kwargs) -> DatabaseExtractor:
    """Create Oracle extractor."""
    return ExtractorFactory.create_database_extractor(
        host=host, port=port, database=database,
        username=username, password=password,
        database_type='oracle', **kwargs
    )