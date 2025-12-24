"""
Sync Gateway Router.

Provides unified API gateway routing for the data synchronization system.
Handles request routing, load balancing, and service discovery.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ServiceStatus(str, Enum):
    """Service health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class LoadBalanceStrategy(str, Enum):
    """Load balancing strategy."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    RANDOM = "random"
    WEIGHTED = "weighted"


@dataclass
class ServiceEndpoint:
    """Represents a service endpoint for routing."""
    id: str
    name: str
    url: str
    weight: int = 1
    status: ServiceStatus = ServiceStatus.UNKNOWN
    active_connections: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    last_health_check: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RouteConfig:
    """Route configuration for API gateway."""
    path: str
    method: str
    service_name: str
    endpoints: List[ServiceEndpoint] = field(default_factory=list)
    load_balance_strategy: LoadBalanceStrategy = LoadBalanceStrategy.ROUND_ROBIN
    timeout: int = 30
    retries: int = 3
    circuit_breaker_enabled: bool = True
    rate_limit: Optional[int] = None  # requests per minute
    auth_required: bool = True
    cache_enabled: bool = False
    cache_ttl: int = 60  # seconds


class RequestContext(BaseModel):
    """Request context for gateway processing."""
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None
    start_time: float = Field(default_factory=time.time)
    path: str = ""
    method: str = ""
    client_ip: Optional[str] = None
    user_agent: Optional[str] = None
    headers: Dict[str, str] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GatewayResponse(BaseModel):
    """Standard gateway response format."""
    success: bool
    request_id: str
    data: Optional[Any] = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    duration_ms: Optional[float] = None


class SyncGatewayRouter:
    """
    Unified API Gateway Router for Data Sync System.

    Provides:
    - Dynamic route registration and management
    - Load balancing across service endpoints
    - Request/response logging
    - Circuit breaker pattern
    - Service health monitoring
    """

    def __init__(self):
        self.router = APIRouter(prefix="/api/v1/sync", tags=["sync-gateway"])
        self._routes: Dict[str, RouteConfig] = {}
        self._services: Dict[str, List[ServiceEndpoint]] = {}
        self._round_robin_index: Dict[str, int] = {}
        self._circuit_breakers: Dict[str, Dict[str, Any]] = {}
        self._request_log: List[Dict[str, Any]] = []
        self._max_log_size = 10000

        # Register default routes
        self._register_default_routes()

    def _register_default_routes(self):
        """Register default gateway routes."""

        @self.router.get("/health")
        async def gateway_health():
            """Gateway health check endpoint."""
            return {
                "status": "healthy",
                "service": "sync-gateway",
                "timestamp": datetime.utcnow().isoformat(),
                "routes_count": len(self._routes),
                "services_count": len(self._services)
            }

        @self.router.get("/routes")
        async def list_routes():
            """List all registered routes."""
            return {
                "routes": [
                    {
                        "path": route.path,
                        "method": route.method,
                        "service": route.service_name,
                        "endpoints_count": len(route.endpoints),
                        "load_balance": route.load_balance_strategy.value
                    }
                    for route in self._routes.values()
                ]
            }

        @self.router.get("/services")
        async def list_services():
            """List all registered services with their endpoints."""
            return {
                "services": {
                    name: [
                        {
                            "id": ep.id,
                            "url": ep.url,
                            "status": ep.status.value,
                            "weight": ep.weight,
                            "active_connections": ep.active_connections,
                            "avg_response_time": ep.avg_response_time
                        }
                        for ep in endpoints
                    ]
                    for name, endpoints in self._services.items()
                }
            }

        @self.router.get("/metrics")
        async def gateway_metrics():
            """Get gateway metrics."""
            total_requests = sum(
                ep.total_requests
                for endpoints in self._services.values()
                for ep in endpoints
            )
            failed_requests = sum(
                ep.failed_requests
                for endpoints in self._services.values()
                for ep in endpoints
            )

            return {
                "total_requests": total_requests,
                "failed_requests": failed_requests,
                "success_rate": (
                    (total_requests - failed_requests) / total_requests * 100
                    if total_requests > 0 else 100.0
                ),
                "services": len(self._services),
                "endpoints": sum(
                    len(endpoints) for endpoints in self._services.values()
                ),
                "circuit_breakers": {
                    name: cb.get("state", "closed")
                    for name, cb in self._circuit_breakers.items()
                }
            }

    def register_service(
        self,
        name: str,
        endpoints: List[Dict[str, Any]]
    ) -> None:
        """
        Register a service with its endpoints.

        Args:
            name: Service name
            endpoints: List of endpoint configurations
        """
        self._services[name] = [
            ServiceEndpoint(
                id=ep.get("id", str(uuid4())),
                name=name,
                url=ep["url"],
                weight=ep.get("weight", 1),
                metadata=ep.get("metadata", {})
            )
            for ep in endpoints
        ]
        self._round_robin_index[name] = 0
        self._circuit_breakers[name] = {
            "state": "closed",
            "failures": 0,
            "last_failure": None,
            "threshold": 5,
            "reset_timeout": 60
        }
        logger.info(f"Registered service '{name}' with {len(endpoints)} endpoints")

    def register_route(self, config: RouteConfig) -> None:
        """
        Register a route configuration.

        Args:
            config: Route configuration
        """
        route_key = f"{config.method}:{config.path}"
        self._routes[route_key] = config
        logger.info(f"Registered route: {route_key} -> {config.service_name}")

    def unregister_service(self, name: str) -> bool:
        """
        Unregister a service.

        Args:
            name: Service name

        Returns:
            True if service was removed, False if not found
        """
        if name in self._services:
            del self._services[name]
            del self._round_robin_index[name]
            del self._circuit_breakers[name]
            logger.info(f"Unregistered service: {name}")
            return True
        return False

    def get_endpoint(
        self,
        service_name: str,
        strategy: LoadBalanceStrategy = LoadBalanceStrategy.ROUND_ROBIN
    ) -> Optional[ServiceEndpoint]:
        """
        Get an endpoint using the specified load balancing strategy.

        Args:
            service_name: Name of the service
            strategy: Load balancing strategy

        Returns:
            Selected endpoint or None if no healthy endpoints
        """
        endpoints = self._services.get(service_name, [])
        healthy_endpoints = [
            ep for ep in endpoints
            if ep.status != ServiceStatus.UNHEALTHY
        ]

        if not healthy_endpoints:
            return None

        if strategy == LoadBalanceStrategy.ROUND_ROBIN:
            return self._round_robin_select(service_name, healthy_endpoints)
        elif strategy == LoadBalanceStrategy.LEAST_CONNECTIONS:
            return self._least_connections_select(healthy_endpoints)
        elif strategy == LoadBalanceStrategy.WEIGHTED:
            return self._weighted_select(healthy_endpoints)
        else:
            import random
            return random.choice(healthy_endpoints)

    def _round_robin_select(
        self,
        service_name: str,
        endpoints: List[ServiceEndpoint]
    ) -> ServiceEndpoint:
        """Round-robin endpoint selection."""
        index = self._round_robin_index.get(service_name, 0)
        endpoint = endpoints[index % len(endpoints)]
        self._round_robin_index[service_name] = (index + 1) % len(endpoints)
        return endpoint

    def _least_connections_select(
        self,
        endpoints: List[ServiceEndpoint]
    ) -> ServiceEndpoint:
        """Least connections endpoint selection."""
        return min(endpoints, key=lambda ep: ep.active_connections)

    def _weighted_select(
        self,
        endpoints: List[ServiceEndpoint]
    ) -> ServiceEndpoint:
        """Weighted random endpoint selection."""
        import random
        total_weight = sum(ep.weight for ep in endpoints)
        r = random.uniform(0, total_weight)
        current_weight = 0
        for ep in endpoints:
            current_weight += ep.weight
            if r <= current_weight:
                return ep
        return endpoints[-1]

    def check_circuit_breaker(self, service_name: str) -> bool:
        """
        Check if circuit breaker allows request.

        Args:
            service_name: Service name

        Returns:
            True if request allowed, False if circuit is open
        """
        cb = self._circuit_breakers.get(service_name)
        if not cb:
            return True

        if cb["state"] == "open":
            # Check if we should try to close
            if cb["last_failure"]:
                elapsed = time.time() - cb["last_failure"]
                if elapsed > cb["reset_timeout"]:
                    cb["state"] = "half-open"
                    return True
            return False

        return True

    def record_success(self, service_name: str, response_time: float) -> None:
        """Record successful request for circuit breaker."""
        cb = self._circuit_breakers.get(service_name)
        if cb:
            if cb["state"] == "half-open":
                cb["state"] = "closed"
            cb["failures"] = 0

        # Update endpoint metrics
        endpoints = self._services.get(service_name, [])
        for ep in endpoints:
            ep.total_requests += 1
            # Simple moving average for response time
            ep.avg_response_time = (
                (ep.avg_response_time * (ep.total_requests - 1) + response_time)
                / ep.total_requests
            )

    def record_failure(self, service_name: str) -> None:
        """Record failed request for circuit breaker."""
        cb = self._circuit_breakers.get(service_name)
        if cb:
            cb["failures"] += 1
            cb["last_failure"] = time.time()
            if cb["failures"] >= cb["threshold"]:
                cb["state"] = "open"
                logger.warning(f"Circuit breaker opened for service: {service_name}")

        # Update endpoint metrics
        endpoints = self._services.get(service_name, [])
        for ep in endpoints:
            ep.failed_requests += 1

    def log_request(self, context: RequestContext, response: GatewayResponse) -> None:
        """Log request for monitoring and debugging."""
        log_entry = {
            "request_id": context.request_id,
            "tenant_id": context.tenant_id,
            "user_id": context.user_id,
            "path": context.path,
            "method": context.method,
            "client_ip": context.client_ip,
            "start_time": context.start_time,
            "duration_ms": response.duration_ms,
            "success": response.success,
            "error": response.error,
            "timestamp": datetime.utcnow().isoformat()
        }

        self._request_log.append(log_entry)

        # Trim log if too large
        if len(self._request_log) > self._max_log_size:
            self._request_log = self._request_log[-self._max_log_size // 2:]

    def get_request_logs(
        self,
        limit: int = 100,
        tenant_id: Optional[str] = None,
        success_only: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """Get request logs with optional filtering."""
        logs = self._request_log

        if tenant_id:
            logs = [l for l in logs if l.get("tenant_id") == tenant_id]

        if success_only is not None:
            logs = [l for l in logs if l.get("success") == success_only]

        return logs[-limit:]

    async def health_check_services(self) -> Dict[str, ServiceStatus]:
        """
        Perform health checks on all registered services.

        Returns:
            Dict mapping service names to their health status
        """
        import aiohttp

        results = {}

        async def check_endpoint(endpoint: ServiceEndpoint) -> ServiceStatus:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{endpoint.url}/health",
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as response:
                        if response.status == 200:
                            return ServiceStatus.HEALTHY
                        elif response.status < 500:
                            return ServiceStatus.DEGRADED
                        else:
                            return ServiceStatus.UNHEALTHY
            except Exception:
                return ServiceStatus.UNHEALTHY

        for service_name, endpoints in self._services.items():
            statuses = await asyncio.gather(*[
                check_endpoint(ep) for ep in endpoints
            ])

            # Update endpoint statuses
            for ep, status in zip(endpoints, statuses):
                ep.status = status
                ep.last_health_check = datetime.utcnow()

            # Aggregate service status
            healthy_count = sum(1 for s in statuses if s == ServiceStatus.HEALTHY)
            if healthy_count == len(statuses):
                results[service_name] = ServiceStatus.HEALTHY
            elif healthy_count > 0:
                results[service_name] = ServiceStatus.DEGRADED
            else:
                results[service_name] = ServiceStatus.UNHEALTHY

        return results

    def create_request_context(self, request: Request) -> RequestContext:
        """Create request context from FastAPI request."""
        return RequestContext(
            path=str(request.url.path),
            method=request.method,
            client_ip=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
            headers=dict(request.headers)
        )

    def create_response(
        self,
        context: RequestContext,
        data: Any = None,
        error: Optional[str] = None,
        error_code: Optional[str] = None,
        success: bool = True
    ) -> GatewayResponse:
        """Create standardized gateway response."""
        duration_ms = (time.time() - context.start_time) * 1000
        return GatewayResponse(
            success=success,
            request_id=context.request_id,
            data=data,
            error=error,
            error_code=error_code,
            duration_ms=duration_ms
        )

    def get_router(self) -> APIRouter:
        """Get the FastAPI router."""
        return self.router


# Global gateway router instance
sync_gateway_router = SyncGatewayRouter()
