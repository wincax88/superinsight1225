"""
Performance Optimization Module for AI Agent System.

Provides caching, concurrent processing, performance monitoring,
and optimization utilities for the agent system.
"""

import asyncio
import functools
import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, Generic, List, Optional, TypeVar, Union

from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ============================================================================
# Caching System
# ============================================================================


class CacheStrategy(str, Enum):
    """Cache eviction strategies."""

    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time-To-Live based
    FIFO = "fifo"  # First In First Out


@dataclass
class CacheEntry(Generic[T]):
    """Single cache entry with metadata."""

    key: str
    value: T
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    ttl_seconds: Optional[float] = None

    @property
    def is_expired(self) -> bool:
        if self.ttl_seconds is None:
            return False
        age = (datetime.utcnow() - self.created_at).total_seconds()
        return age > self.ttl_seconds

    def access(self) -> T:
        self.last_accessed = datetime.utcnow()
        self.access_count += 1
        return self.value


class CacheMetrics(BaseModel):
    """Metrics for cache performance."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    max_size: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class InMemoryCache(Generic[T]):
    """High-performance in-memory cache with multiple eviction strategies."""

    def __init__(
        self,
        max_size: int = 1000,
        strategy: CacheStrategy = CacheStrategy.LRU,
        default_ttl: Optional[float] = None,
    ):
        self.max_size = max_size
        self.strategy = strategy
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry[T]] = OrderedDict()
        self._metrics = CacheMetrics(max_size=max_size)
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[T]:
        """Get a value from cache."""
        async with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._metrics.misses += 1
                return None

            if entry.is_expired:
                del self._cache[key]
                self._metrics.misses += 1
                self._metrics.evictions += 1
                return None

            self._metrics.hits += 1

            # Update access order for LRU
            if self.strategy == CacheStrategy.LRU:
                self._cache.move_to_end(key)

            return entry.access()

    async def set(
        self, key: str, value: T, ttl: Optional[float] = None
    ) -> None:
        """Set a value in cache."""
        async with self._lock:
            ttl = ttl or self.default_ttl

            # Evict if necessary
            while len(self._cache) >= self.max_size:
                self._evict_one()

            self._cache[key] = CacheEntry(
                key=key, value=value, ttl_seconds=ttl
            )
            self._metrics.size = len(self._cache)

    async def delete(self, key: str) -> bool:
        """Delete a key from cache."""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._metrics.size = len(self._cache)
                return True
            return False

    async def clear(self) -> int:
        """Clear the entire cache."""
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._metrics.size = 0
            return count

    def _evict_one(self) -> None:
        """Evict one entry based on strategy."""
        if not self._cache:
            return

        if self.strategy == CacheStrategy.LRU:
            self._cache.popitem(last=False)
        elif self.strategy == CacheStrategy.LFU:
            # Find least frequently used
            min_key = min(
                self._cache.keys(),
                key=lambda k: self._cache[k].access_count,
            )
            del self._cache[min_key]
        elif self.strategy == CacheStrategy.FIFO:
            self._cache.popitem(last=False)
        elif self.strategy == CacheStrategy.TTL:
            # Evict expired first, then oldest
            expired = [
                k for k, v in self._cache.items() if v.is_expired
            ]
            if expired:
                del self._cache[expired[0]]
            else:
                self._cache.popitem(last=False)

        self._metrics.evictions += 1

    def get_metrics(self) -> CacheMetrics:
        """Get cache metrics."""
        self._metrics.size = len(self._cache)
        return self._metrics.copy()


class ResponseCache:
    """Specialized cache for agent responses."""

    def __init__(
        self,
        max_size: int = 500,
        default_ttl: float = 3600,  # 1 hour
    ):
        self._cache: InMemoryCache[Dict[str, Any]] = InMemoryCache(
            max_size=max_size,
            strategy=CacheStrategy.LRU,
            default_ttl=default_ttl,
        )

    @staticmethod
    def _generate_key(
        query: str,
        context: Optional[Dict[str, Any]] = None,
        tenant_id: Optional[str] = None,
    ) -> str:
        """Generate a cache key from request parameters."""
        key_data = {
            "query": query.strip().lower(),
            "context": json.dumps(context or {}, sort_keys=True),
            "tenant_id": tenant_id or "",
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()

    async def get_response(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        tenant_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get cached response for a query."""
        key = self._generate_key(query, context, tenant_id)
        return await self._cache.get(key)

    async def cache_response(
        self,
        query: str,
        response: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        tenant_id: Optional[str] = None,
        ttl: Optional[float] = None,
    ) -> None:
        """Cache a response for a query."""
        key = self._generate_key(query, context, tenant_id)
        await self._cache.set(key, response, ttl)

    async def invalidate(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        tenant_id: Optional[str] = None,
    ) -> bool:
        """Invalidate a cached response."""
        key = self._generate_key(query, context, tenant_id)
        return await self._cache.delete(key)

    async def clear_all(self) -> int:
        """Clear all cached responses."""
        return await self._cache.clear()

    def get_metrics(self) -> CacheMetrics:
        """Get cache metrics."""
        return self._cache.get_metrics()


def cached_response(
    ttl: float = 3600,
    key_builder: Optional[Callable[..., str]] = None,
):
    """Decorator for caching async function responses."""

    def decorator(
        func: Callable[..., Coroutine[Any, Any, T]]
    ) -> Callable[..., Coroutine[Any, Any, T]]:
        cache: InMemoryCache[T] = InMemoryCache(
            max_size=1000, strategy=CacheStrategy.LRU, default_ttl=ttl
        )

        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            # Build cache key
            if key_builder:
                key = key_builder(*args, **kwargs)
            else:
                key_data = {"args": str(args), "kwargs": str(kwargs)}
                key = hashlib.md5(
                    json.dumps(key_data, sort_keys=True).encode()
                ).hexdigest()

            # Try cache
            result = await cache.get(key)
            if result is not None:
                return result

            # Execute and cache
            result = await func(*args, **kwargs)
            await cache.set(key, result, ttl)
            return result

        wrapper.cache = cache  # type: ignore
        return wrapper

    return decorator


# ============================================================================
# Concurrent Processing
# ============================================================================


class ConcurrencyMode(str, Enum):
    """Concurrency execution modes."""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    BATCH = "batch"


@dataclass
class TaskResult(Generic[T]):
    """Result of a concurrent task execution."""

    task_id: str
    success: bool
    result: Optional[T] = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class ConcurrentExecutor:
    """High-performance concurrent task executor."""

    def __init__(
        self,
        max_workers: int = 10,
        max_concurrent: int = 50,
        timeout: float = 30.0,
    ):
        self.max_workers = max_workers
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._active_tasks: Dict[str, asyncio.Task] = {}

    async def execute_async(
        self,
        task_id: str,
        func: Callable[..., Coroutine[Any, Any, T]],
        *args,
        **kwargs,
    ) -> TaskResult[T]:
        """Execute an async function with concurrency control."""
        async with self._semaphore:
            started_at = datetime.utcnow()
            start_time = time.time()

            try:
                task = asyncio.create_task(func(*args, **kwargs))
                self._active_tasks[task_id] = task

                result = await asyncio.wait_for(task, timeout=self.timeout)

                return TaskResult(
                    task_id=task_id,
                    success=True,
                    result=result,
                    execution_time_ms=(time.time() - start_time) * 1000,
                    started_at=started_at,
                    completed_at=datetime.utcnow(),
                )
            except asyncio.TimeoutError:
                return TaskResult(
                    task_id=task_id,
                    success=False,
                    error=f"Task timed out after {self.timeout}s",
                    execution_time_ms=(time.time() - start_time) * 1000,
                    started_at=started_at,
                    completed_at=datetime.utcnow(),
                )
            except Exception as e:
                return TaskResult(
                    task_id=task_id,
                    success=False,
                    error=str(e),
                    execution_time_ms=(time.time() - start_time) * 1000,
                    started_at=started_at,
                    completed_at=datetime.utcnow(),
                )
            finally:
                self._active_tasks.pop(task_id, None)

    async def execute_sync(
        self,
        task_id: str,
        func: Callable[..., T],
        *args,
        **kwargs,
    ) -> TaskResult[T]:
        """Execute a sync function in thread pool."""
        async with self._semaphore:
            started_at = datetime.utcnow()
            start_time = time.time()

            try:
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(
                        self._executor,
                        functools.partial(func, *args, **kwargs),
                    ),
                    timeout=self.timeout,
                )

                return TaskResult(
                    task_id=task_id,
                    success=True,
                    result=result,
                    execution_time_ms=(time.time() - start_time) * 1000,
                    started_at=started_at,
                    completed_at=datetime.utcnow(),
                )
            except asyncio.TimeoutError:
                return TaskResult(
                    task_id=task_id,
                    success=False,
                    error=f"Task timed out after {self.timeout}s",
                    execution_time_ms=(time.time() - start_time) * 1000,
                    started_at=started_at,
                    completed_at=datetime.utcnow(),
                )
            except Exception as e:
                return TaskResult(
                    task_id=task_id,
                    success=False,
                    error=str(e),
                    execution_time_ms=(time.time() - start_time) * 1000,
                    started_at=started_at,
                    completed_at=datetime.utcnow(),
                )

    async def execute_batch(
        self,
        tasks: List[tuple],  # [(task_id, func, args, kwargs), ...]
        mode: ConcurrencyMode = ConcurrencyMode.PARALLEL,
    ) -> List[TaskResult]:
        """Execute multiple tasks."""
        if mode == ConcurrencyMode.SEQUENTIAL:
            results = []
            for task_id, func, args, kwargs in tasks:
                if asyncio.iscoroutinefunction(func):
                    result = await self.execute_async(
                        task_id, func, *args, **kwargs
                    )
                else:
                    result = await self.execute_sync(
                        task_id, func, *args, **kwargs
                    )
                results.append(result)
            return results

        elif mode == ConcurrencyMode.PARALLEL:
            coros = []
            for task_id, func, args, kwargs in tasks:
                if asyncio.iscoroutinefunction(func):
                    coros.append(
                        self.execute_async(task_id, func, *args, **kwargs)
                    )
                else:
                    coros.append(
                        self.execute_sync(task_id, func, *args, **kwargs)
                    )
            return await asyncio.gather(*coros)

        else:
            raise ValueError(f"Unsupported mode: {mode}")

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task."""
        task = self._active_tasks.get(task_id)
        if task and not task.done():
            task.cancel()
            return True
        return False

    def get_active_count(self) -> int:
        """Get count of active tasks."""
        return len(self._active_tasks)

    def shutdown(self) -> None:
        """Shutdown the executor."""
        self._executor.shutdown(wait=False)


# ============================================================================
# Performance Monitoring
# ============================================================================


class MetricType(str, Enum):
    """Types of performance metrics."""

    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    CACHE_HIT_RATE = "cache_hit_rate"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"


@dataclass
class PerformanceMetric:
    """Single performance metric."""

    name: str
    type: MetricType
    value: float
    unit: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class LatencyStats:
    """Latency statistics."""

    count: int = 0
    total_ms: float = 0.0
    min_ms: float = float("inf")
    max_ms: float = 0.0
    values: List[float] = field(default_factory=list)

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.count if self.count > 0 else 0.0

    @property
    def p50_ms(self) -> float:
        if not self.values:
            return 0.0
        sorted_values = sorted(self.values)
        idx = len(sorted_values) // 2
        return sorted_values[idx]

    @property
    def p95_ms(self) -> float:
        if not self.values:
            return 0.0
        sorted_values = sorted(self.values)
        idx = int(len(sorted_values) * 0.95)
        return sorted_values[min(idx, len(sorted_values) - 1)]

    @property
    def p99_ms(self) -> float:
        if not self.values:
            return 0.0
        sorted_values = sorted(self.values)
        idx = int(len(sorted_values) * 0.99)
        return sorted_values[min(idx, len(sorted_values) - 1)]

    def record(self, latency_ms: float) -> None:
        self.count += 1
        self.total_ms += latency_ms
        self.min_ms = min(self.min_ms, latency_ms)
        self.max_ms = max(self.max_ms, latency_ms)
        self.values.append(latency_ms)

        # Keep only last 10000 values
        if len(self.values) > 10000:
            self.values = self.values[-5000:]


class PerformanceMonitor:
    """Comprehensive performance monitoring for agent system."""

    def __init__(self, window_size_seconds: int = 300):
        self.window_size = window_size_seconds
        self._latency_stats: Dict[str, LatencyStats] = {}
        self._counters: Dict[str, int] = {}
        self._gauges: Dict[str, float] = {}
        self._start_time = datetime.utcnow()
        self._lock = asyncio.Lock()

    async def record_latency(
        self, operation: str, latency_ms: float, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Record operation latency."""
        async with self._lock:
            key = f"{operation}:{json.dumps(labels or {}, sort_keys=True)}"
            if key not in self._latency_stats:
                self._latency_stats[key] = LatencyStats()
            self._latency_stats[key].record(latency_ms)

    async def increment_counter(
        self, name: str, value: int = 1, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Increment a counter metric."""
        async with self._lock:
            key = f"{name}:{json.dumps(labels or {}, sort_keys=True)}"
            self._counters[key] = self._counters.get(key, 0) + value

    async def set_gauge(
        self, name: str, value: float, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Set a gauge metric."""
        async with self._lock:
            key = f"{name}:{json.dumps(labels or {}, sort_keys=True)}"
            self._gauges[key] = value

    def get_latency_stats(self, operation: str) -> Optional[LatencyStats]:
        """Get latency statistics for an operation."""
        for key, stats in self._latency_stats.items():
            if key.startswith(f"{operation}:"):
                return stats
        return None

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics."""
        uptime = (datetime.utcnow() - self._start_time).total_seconds()

        latency_summary = {}
        for key, stats in self._latency_stats.items():
            operation = key.split(":")[0]
            latency_summary[operation] = {
                "count": stats.count,
                "avg_ms": round(stats.avg_ms, 2),
                "min_ms": round(stats.min_ms, 2) if stats.min_ms != float("inf") else 0,
                "max_ms": round(stats.max_ms, 2),
                "p50_ms": round(stats.p50_ms, 2),
                "p95_ms": round(stats.p95_ms, 2),
                "p99_ms": round(stats.p99_ms, 2),
            }

        return {
            "uptime_seconds": uptime,
            "latency": latency_summary,
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "timestamp": datetime.utcnow().isoformat(),
        }

    async def reset(self) -> None:
        """Reset all metrics."""
        async with self._lock:
            self._latency_stats.clear()
            self._counters.clear()
            self._gauges.clear()
            self._start_time = datetime.utcnow()


def measure_latency(operation: str, monitor: Optional[PerformanceMonitor] = None):
    """Decorator to measure function latency."""

    def decorator(
        func: Callable[..., Coroutine[Any, Any, T]]
    ) -> Callable[..., Coroutine[Any, Any, T]]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            start_time = time.time()
            try:
                return await func(*args, **kwargs)
            finally:
                latency_ms = (time.time() - start_time) * 1000
                if monitor:
                    await monitor.record_latency(operation, latency_ms)
                else:
                    logger.debug(f"{operation} completed in {latency_ms:.2f}ms")

        return wrapper

    return decorator


# ============================================================================
# Health Check and Service Discovery
# ============================================================================


class HealthStatus(str, Enum):
    """Health check status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health status of a single component."""

    name: str
    status: HealthStatus
    message: str = ""
    latency_ms: float = 0.0
    last_check: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemHealth:
    """Overall system health."""

    status: HealthStatus
    components: List[ComponentHealth]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    version: str = "1.0.0"

    @classmethod
    def aggregate(cls, components: List[ComponentHealth], version: str = "1.0.0") -> "SystemHealth":
        """Aggregate component health into system health."""
        if not components:
            return cls(status=HealthStatus.UNKNOWN, components=[], version=version)

        statuses = [c.status for c in components]

        if all(s == HealthStatus.HEALTHY for s in statuses):
            overall = HealthStatus.HEALTHY
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            overall = HealthStatus.UNHEALTHY
        else:
            overall = HealthStatus.DEGRADED

        return cls(status=overall, components=components, version=version)


class HealthChecker:
    """Health check manager for agent system."""

    def __init__(self):
        self._checks: Dict[str, Callable[[], Coroutine[Any, Any, ComponentHealth]]] = {}
        self._last_results: Dict[str, ComponentHealth] = {}

    def register_check(
        self,
        name: str,
        check_fn: Callable[[], Coroutine[Any, Any, ComponentHealth]],
    ) -> None:
        """Register a health check function."""
        self._checks[name] = check_fn

    async def check_component(self, name: str) -> ComponentHealth:
        """Run a single health check."""
        check_fn = self._checks.get(name)
        if not check_fn:
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNKNOWN,
                message=f"No health check registered for {name}",
            )

        start_time = time.time()
        try:
            result = await asyncio.wait_for(check_fn(), timeout=5.0)
            result.latency_ms = (time.time() - start_time) * 1000
            self._last_results[name] = result
            return result
        except asyncio.TimeoutError:
            result = ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message="Health check timed out",
                latency_ms=(time.time() - start_time) * 1000,
            )
            self._last_results[name] = result
            return result
        except Exception as e:
            result = ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                latency_ms=(time.time() - start_time) * 1000,
            )
            self._last_results[name] = result
            return result

    async def check_all(self) -> SystemHealth:
        """Run all health checks."""
        tasks = [self.check_component(name) for name in self._checks]
        components = await asyncio.gather(*tasks)
        return SystemHealth.aggregate(list(components))

    def get_last_results(self) -> Dict[str, ComponentHealth]:
        """Get last health check results."""
        return dict(self._last_results)


@dataclass
class ServiceInstance:
    """Service instance information."""

    service_id: str
    service_name: str
    host: str
    port: int
    health_status: HealthStatus = HealthStatus.UNKNOWN
    metadata: Dict[str, Any] = field(default_factory=dict)
    registered_at: datetime = field(default_factory=datetime.utcnow)
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"

    @property
    def is_healthy(self) -> bool:
        return self.health_status == HealthStatus.HEALTHY


class ServiceRegistry:
    """Simple in-memory service registry for service discovery."""

    def __init__(self, heartbeat_timeout: float = 30.0):
        self._services: Dict[str, Dict[str, ServiceInstance]] = {}
        self._heartbeat_timeout = heartbeat_timeout
        self._lock = asyncio.Lock()

    async def register(self, instance: ServiceInstance) -> None:
        """Register a service instance."""
        async with self._lock:
            if instance.service_name not in self._services:
                self._services[instance.service_name] = {}
            self._services[instance.service_name][instance.service_id] = instance
            logger.info(
                f"Registered service {instance.service_name}/{instance.service_id} "
                f"at {instance.address}"
            )

    async def deregister(self, service_name: str, service_id: str) -> bool:
        """Deregister a service instance."""
        async with self._lock:
            if service_name in self._services:
                if service_id in self._services[service_name]:
                    del self._services[service_name][service_id]
                    logger.info(f"Deregistered service {service_name}/{service_id}")
                    return True
            return False

    async def heartbeat(self, service_name: str, service_id: str) -> bool:
        """Update heartbeat for a service instance."""
        async with self._lock:
            if service_name in self._services:
                instance = self._services[service_name].get(service_id)
                if instance:
                    instance.last_heartbeat = datetime.utcnow()
                    return True
            return False

    async def update_health(
        self, service_name: str, service_id: str, status: HealthStatus
    ) -> bool:
        """Update health status for a service instance."""
        async with self._lock:
            if service_name in self._services:
                instance = self._services[service_name].get(service_id)
                if instance:
                    instance.health_status = status
                    return True
            return False

    async def get_instances(
        self, service_name: str, healthy_only: bool = True
    ) -> List[ServiceInstance]:
        """Get all instances of a service."""
        async with self._lock:
            instances = list(self._services.get(service_name, {}).values())

            # Filter out stale instances
            now = datetime.utcnow()
            active_instances = [
                inst
                for inst in instances
                if (now - inst.last_heartbeat).total_seconds() < self._heartbeat_timeout
            ]

            if healthy_only:
                return [inst for inst in active_instances if inst.is_healthy]
            return active_instances

    async def get_all_services(self) -> Dict[str, List[ServiceInstance]]:
        """Get all registered services."""
        async with self._lock:
            return {
                name: list(instances.values())
                for name, instances in self._services.items()
            }


# ============================================================================
# Load Balancer
# ============================================================================


class LoadBalanceStrategy(str, Enum):
    """Load balancing strategies."""

    ROUND_ROBIN = "round_robin"
    RANDOM = "random"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED = "weighted"


class LoadBalancer:
    """Simple load balancer for service instances."""

    def __init__(
        self,
        registry: ServiceRegistry,
        strategy: LoadBalanceStrategy = LoadBalanceStrategy.ROUND_ROBIN,
    ):
        self.registry = registry
        self.strategy = strategy
        self._counters: Dict[str, int] = {}
        self._connections: Dict[str, int] = {}
        self._lock = asyncio.Lock()

    async def get_instance(self, service_name: str) -> Optional[ServiceInstance]:
        """Get a service instance using the configured strategy."""
        instances = await self.registry.get_instances(service_name, healthy_only=True)

        if not instances:
            return None

        async with self._lock:
            if self.strategy == LoadBalanceStrategy.ROUND_ROBIN:
                idx = self._counters.get(service_name, 0)
                instance = instances[idx % len(instances)]
                self._counters[service_name] = idx + 1
                return instance

            elif self.strategy == LoadBalanceStrategy.RANDOM:
                import random
                return random.choice(instances)

            elif self.strategy == LoadBalanceStrategy.LEAST_CONNECTIONS:
                return min(
                    instances,
                    key=lambda i: self._connections.get(i.service_id, 0),
                )

            elif self.strategy == LoadBalanceStrategy.WEIGHTED:
                # Use weight from metadata, default to 1
                weighted_instances = []
                for inst in instances:
                    weight = inst.metadata.get("weight", 1)
                    weighted_instances.extend([inst] * weight)
                if weighted_instances:
                    import random
                    return random.choice(weighted_instances)

        return instances[0] if instances else None

    async def acquire(self, service_id: str) -> None:
        """Acquire a connection (for least connections strategy)."""
        async with self._lock:
            self._connections[service_id] = self._connections.get(service_id, 0) + 1

    async def release(self, service_id: str) -> None:
        """Release a connection."""
        async with self._lock:
            if service_id in self._connections:
                self._connections[service_id] = max(
                    0, self._connections[service_id] - 1
                )


# ============================================================================
# Global Instances
# ============================================================================


_response_cache: Optional[ResponseCache] = None
_performance_monitor: Optional[PerformanceMonitor] = None
_concurrent_executor: Optional[ConcurrentExecutor] = None
_health_checker: Optional[HealthChecker] = None
_service_registry: Optional[ServiceRegistry] = None


def get_response_cache() -> ResponseCache:
    """Get or create global response cache."""
    global _response_cache
    if _response_cache is None:
        _response_cache = ResponseCache(max_size=500, default_ttl=3600)
    return _response_cache


def get_performance_monitor() -> PerformanceMonitor:
    """Get or create global performance monitor."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor


def get_concurrent_executor() -> ConcurrentExecutor:
    """Get or create global concurrent executor."""
    global _concurrent_executor
    if _concurrent_executor is None:
        _concurrent_executor = ConcurrentExecutor()
    return _concurrent_executor


def get_health_checker() -> HealthChecker:
    """Get or create global health checker."""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker


def get_service_registry() -> ServiceRegistry:
    """Get or create global service registry."""
    global _service_registry
    if _service_registry is None:
        _service_registry = ServiceRegistry()
    return _service_registry


# ============================================================================
# Utility Functions
# ============================================================================


async def create_default_health_checks(health_checker: HealthChecker) -> None:
    """Register default health checks for agent components."""

    async def check_cache() -> ComponentHealth:
        cache = get_response_cache()
        metrics = cache.get_metrics()
        return ComponentHealth(
            name="response_cache",
            status=HealthStatus.HEALTHY,
            message=f"Hit rate: {metrics.hit_rate:.2%}",
            metadata={
                "size": metrics.size,
                "max_size": metrics.max_size,
                "hit_rate": metrics.hit_rate,
            },
        )

    async def check_executor() -> ComponentHealth:
        executor = get_concurrent_executor()
        active = executor.get_active_count()
        return ComponentHealth(
            name="concurrent_executor",
            status=HealthStatus.HEALTHY if active < 40 else HealthStatus.DEGRADED,
            message=f"Active tasks: {active}",
            metadata={"active_tasks": active},
        )

    async def check_monitor() -> ComponentHealth:
        monitor = get_performance_monitor()
        metrics = monitor.get_all_metrics()
        return ComponentHealth(
            name="performance_monitor",
            status=HealthStatus.HEALTHY,
            message=f"Uptime: {metrics['uptime_seconds']:.0f}s",
            metadata={"uptime_seconds": metrics["uptime_seconds"]},
        )

    health_checker.register_check("response_cache", check_cache)
    health_checker.register_check("concurrent_executor", check_executor)
    health_checker.register_check("performance_monitor", check_monitor)
