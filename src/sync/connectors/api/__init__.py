"""
API Connectors Module.

Provides connectors for REST, GraphQL, and SOAP APIs.
"""

from .rest_connector import (
    RESTConnector,
    RESTConnectorConfig,
    AuthType,
    AuthConfig,
    PaginationType,
    PaginationConfig,
    RateLimitConfig,
    HttpMethod,
    ApiKeyLocation,
)

__all__ = [
    "RESTConnector",
    "RESTConnectorConfig",
    "AuthType",
    "AuthConfig",
    "PaginationType",
    "PaginationConfig",
    "RateLimitConfig",
    "HttpMethod",
    "ApiKeyLocation",
]
