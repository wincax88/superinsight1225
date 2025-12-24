"""
Sync Gateway Middleware.

Provides middleware components for request/response processing,
logging, CORS handling, and security headers.
"""

import json
import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.middleware.cors import CORSMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


@dataclass
class MiddlewareConfig:
    """Configuration for gateway middleware."""
    # CORS settings
    cors_enabled: bool = True
    cors_origins: List[str] = None
    cors_methods: List[str] = None
    cors_headers: List[str] = None
    cors_credentials: bool = True
    cors_max_age: int = 600

    # Security headers
    security_headers_enabled: bool = True
    content_security_policy: Optional[str] = None
    strict_transport_security: bool = True
    x_frame_options: str = "DENY"
    x_content_type_options: str = "nosniff"
    x_xss_protection: str = "1; mode=block"
    referrer_policy: str = "strict-origin-when-cross-origin"

    # Request logging
    request_logging_enabled: bool = True
    log_request_body: bool = False
    log_response_body: bool = False
    max_body_log_size: int = 1024

    # Request ID
    request_id_header: str = "X-Request-ID"
    generate_request_id: bool = True

    # Timing
    timing_header: str = "X-Response-Time"
    add_timing_header: bool = True

    def __post_init__(self):
        if self.cors_origins is None:
            self.cors_origins = ["*"]
        if self.cors_methods is None:
            self.cors_methods = ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"]
        if self.cors_headers is None:
            self.cors_headers = ["*"]


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging requests and responses."""

    def __init__(
        self,
        app: ASGIApp,
        config: Optional[MiddlewareConfig] = None
    ):
        super().__init__(app)
        self.config = config or MiddlewareConfig()
        self._request_logs: List[Dict[str, Any]] = []
        self._max_logs = 10000

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint
    ) -> Response:
        """Process request and log details."""
        if not self.config.request_logging_enabled:
            return await call_next(request)

        # Generate or extract request ID
        request_id = request.headers.get(self.config.request_id_header)
        if not request_id and self.config.generate_request_id:
            request_id = str(uuid.uuid4())

        # Store request ID in state for access in handlers
        request.state.request_id = request_id

        # Record start time
        start_time = time.time()

        # Extract request details
        log_entry = {
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat(),
            "method": request.method,
            "path": str(request.url.path),
            "query_string": str(request.url.query),
            "client_ip": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent"),
            "content_type": request.headers.get("content-type"),
            "content_length": request.headers.get("content-length"),
        }

        # Optionally log request body
        if self.config.log_request_body:
            try:
                body = await request.body()
                if len(body) <= self.config.max_body_log_size:
                    log_entry["request_body"] = body.decode("utf-8", errors="ignore")
                else:
                    log_entry["request_body"] = f"[Body too large: {len(body)} bytes]"
            except Exception:
                log_entry["request_body"] = "[Could not read body]"

        # Process request
        try:
            response = await call_next(request)
            log_entry["status_code"] = response.status_code
            log_entry["success"] = response.status_code < 400

            # Add request ID to response headers
            response.headers[self.config.request_id_header] = request_id

            # Add timing header
            if self.config.add_timing_header:
                duration_ms = (time.time() - start_time) * 1000
                response.headers[self.config.timing_header] = f"{duration_ms:.2f}ms"
                log_entry["duration_ms"] = duration_ms

        except Exception as e:
            log_entry["error"] = str(e)
            log_entry["success"] = False
            log_entry["duration_ms"] = (time.time() - start_time) * 1000
            raise

        finally:
            # Store log entry
            self._store_log(log_entry)

            # Log to standard logger
            log_level = logging.INFO if log_entry.get("success", True) else logging.WARNING
            logger.log(
                log_level,
                f"{log_entry['method']} {log_entry['path']} - "
                f"{log_entry.get('status_code', 'N/A')} - "
                f"{log_entry.get('duration_ms', 0):.2f}ms"
            )

        return response

    def _store_log(self, entry: Dict[str, Any]) -> None:
        """Store log entry with size limiting."""
        self._request_logs.append(entry)
        if len(self._request_logs) > self._max_logs:
            self._request_logs = self._request_logs[-self._max_logs // 2:]

    def get_logs(
        self,
        limit: int = 100,
        method: Optional[str] = None,
        path_prefix: Optional[str] = None,
        success_only: Optional[bool] = None,
        min_duration_ms: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Get request logs with optional filtering."""
        logs = self._request_logs

        if method:
            logs = [l for l in logs if l.get("method") == method]

        if path_prefix:
            logs = [l for l in logs if l.get("path", "").startswith(path_prefix)]

        if success_only is not None:
            logs = [l for l in logs if l.get("success") == success_only]

        if min_duration_ms is not None:
            logs = [l for l in logs if l.get("duration_ms", 0) >= min_duration_ms]

        return logs[-limit:]


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware for adding security headers to responses."""

    def __init__(
        self,
        app: ASGIApp,
        config: Optional[MiddlewareConfig] = None
    ):
        super().__init__(app)
        self.config = config or MiddlewareConfig()

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint
    ) -> Response:
        """Add security headers to response."""
        response = await call_next(request)

        if not self.config.security_headers_enabled:
            return response

        # X-Frame-Options
        if self.config.x_frame_options:
            response.headers["X-Frame-Options"] = self.config.x_frame_options

        # X-Content-Type-Options
        if self.config.x_content_type_options:
            response.headers["X-Content-Type-Options"] = self.config.x_content_type_options

        # X-XSS-Protection
        if self.config.x_xss_protection:
            response.headers["X-XSS-Protection"] = self.config.x_xss_protection

        # Referrer-Policy
        if self.config.referrer_policy:
            response.headers["Referrer-Policy"] = self.config.referrer_policy

        # Strict-Transport-Security (HSTS)
        if self.config.strict_transport_security:
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains"
            )

        # Content-Security-Policy
        if self.config.content_security_policy:
            response.headers["Content-Security-Policy"] = (
                self.config.content_security_policy
            )

        # Remove potentially dangerous headers
        response.headers.pop("X-Powered-By", None)
        response.headers.pop("Server", None)

        return response


class TenantContextMiddleware(BaseHTTPMiddleware):
    """Middleware for extracting and validating tenant context."""

    TENANT_HEADER = "X-Tenant-ID"

    def __init__(
        self,
        app: ASGIApp,
        require_tenant: bool = False,
        allowed_tenants: Optional[Set[str]] = None
    ):
        super().__init__(app)
        self.require_tenant = require_tenant
        self.allowed_tenants = allowed_tenants

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint
    ) -> Response:
        """Extract tenant ID and validate."""
        tenant_id = request.headers.get(self.TENANT_HEADER)

        # Check if tenant is required
        if self.require_tenant and not tenant_id:
            return Response(
                content=json.dumps({
                    "error": "Tenant ID required",
                    "error_code": "TENANT_REQUIRED"
                }),
                status_code=400,
                media_type="application/json"
            )

        # Validate against allowed tenants
        if tenant_id and self.allowed_tenants:
            if tenant_id not in self.allowed_tenants:
                return Response(
                    content=json.dumps({
                        "error": "Invalid tenant ID",
                        "error_code": "INVALID_TENANT"
                    }),
                    status_code=403,
                    media_type="application/json"
                )

        # Store in request state
        request.state.tenant_id = tenant_id

        return await call_next(request)


class SyncGatewayMiddleware:
    """
    Composite middleware manager for Sync Gateway.

    Manages all middleware components and provides easy setup for FastAPI app.
    """

    def __init__(self, config: Optional[MiddlewareConfig] = None):
        self.config = config or MiddlewareConfig()
        self._logging_middleware: Optional[RequestLoggingMiddleware] = None

    def setup(self, app: FastAPI) -> None:
        """
        Setup all middleware components for the FastAPI app.

        Args:
            app: FastAPI application instance
        """
        # Add CORS middleware (must be added last to wrap everything)
        if self.config.cors_enabled:
            app.add_middleware(
                CORSMiddleware,
                allow_origins=self.config.cors_origins,
                allow_credentials=self.config.cors_credentials,
                allow_methods=self.config.cors_methods,
                allow_headers=self.config.cors_headers,
                max_age=self.config.cors_max_age
            )

        # Add security headers middleware
        if self.config.security_headers_enabled:
            app.add_middleware(
                SecurityHeadersMiddleware,
                config=self.config
            )

        # Add request logging middleware
        if self.config.request_logging_enabled:
            self._logging_middleware = RequestLoggingMiddleware(
                app=app,
                config=self.config
            )
            app.add_middleware(RequestLoggingMiddleware, config=self.config)

        logger.info("Sync Gateway middleware configured")

    def get_request_logs(self, **kwargs) -> List[Dict[str, Any]]:
        """Get request logs from logging middleware."""
        if self._logging_middleware:
            return self._logging_middleware.get_logs(**kwargs)
        return []


def create_middleware_config(
    cors_origins: Optional[List[str]] = None,
    enable_logging: bool = True,
    enable_security_headers: bool = True,
    **kwargs
) -> MiddlewareConfig:
    """
    Factory function to create middleware configuration.

    Args:
        cors_origins: List of allowed CORS origins
        enable_logging: Enable request logging
        enable_security_headers: Enable security headers

    Returns:
        Configured MiddlewareConfig instance
    """
    return MiddlewareConfig(
        cors_origins=cors_origins or ["*"],
        request_logging_enabled=enable_logging,
        security_headers_enabled=enable_security_headers,
        **kwargs
    )
