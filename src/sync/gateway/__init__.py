"""
Sync Gateway Module.

Provides unified API gateway functionality for the data synchronization system,
including authentication, rate limiting, request routing, and security controls.
"""

from src.sync.gateway.router import SyncGatewayRouter
from src.sync.gateway.middleware import SyncGatewayMiddleware
from src.sync.gateway.auth import SyncAuthHandler
from src.sync.gateway.rate_limiter import SyncRateLimiter
from src.sync.gateway.security import SyncSecurityHandler

__all__ = [
    "SyncGatewayRouter",
    "SyncGatewayMiddleware",
    "SyncAuthHandler",
    "SyncRateLimiter",
    "SyncSecurityHandler",
]
