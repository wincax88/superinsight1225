"""
Data extraction module for SuperInsight Platform.

Provides secure, read-only data extraction from various sources including
databases, files, and APIs.
"""

from .base import (
    DataExtractor,
    BaseExtractor,
    ConnectionConfig,
    DatabaseConfig,
    FileConfig,
    APIConfig,
    ExtractionResult,
    SourceType,
    DatabaseType,
    FileType,
    SecurityValidator
)

from .database import DatabaseExtractor
from .file import FileExtractor, WebExtractor, detect_file_type
from .api import APIExtractor, GraphQLExtractor, WebhookExtractor
from .factory import ExtractorFactory

__all__ = [
    # Base classes
    'DataExtractor',
    'BaseExtractor',
    'ExtractionResult',
    
    # Configuration classes
    'ConnectionConfig',
    'DatabaseConfig', 
    'FileConfig',
    'APIConfig',
    
    # Enums
    'SourceType',
    'DatabaseType',
    'FileType',
    
    # Extractors
    'DatabaseExtractor',
    'FileExtractor',
    'WebExtractor',
    'APIExtractor',
    'GraphQLExtractor',
    'WebhookExtractor',
    
    # Factory
    'ExtractorFactory',
    
    # Utilities
    'SecurityValidator',
    'detect_file_type'
]