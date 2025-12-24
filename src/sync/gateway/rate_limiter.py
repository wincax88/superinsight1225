"""
Sync Gateway Rate Limiter.

Provides intelligent rate limiting based on tenant, user, and IP,
with support for multiple rate limiting strategies.
"""

import asyncio
import hashlib
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from fastapi import HTTPException, Request, Response
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


class RateLimitStrategy(str, Enum):
    """Rate limiting strategy types."""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"


class RateLimitScope(str, Enum):
    """Rate limiting scope."""
    GLOBAL = "global"
    TENANT = "tenant"
    USER = "user"
    IP = "ip"
    ENDPOINT = "endpoint"


@dataclass
class RateLimitRule:
    """Rate limit rule configuration."""
    name: str
    scope: RateLimitScope
    requests: int  # Number of requests allowed
    window_seconds: int  # Time window in seconds
    strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW
    burst_multiplier: float = 1.5  # Allow burst up to this multiplier
    path_pattern: Optional[str] = None  # Optional path pattern to match
    methods: Optional[List[str]] = None  # Optional HTTP methods to match
    enabled: bool = True
    priority: int = 0  # Higher priority rules evaluated first


@dataclass
class RateLimitResult:
    """Result of rate limit check."""
    allowed: bool
    remaining: int
    limit: int
    reset_at: float  # Unix timestamp
    retry_after: Optional[int] = None  # Seconds to wait
    rule_name: Optional[str] = None


class TokenBucket:
    """Token bucket rate limiter implementation."""

    def __init__(self, capacity: int, refill_rate: float):
        """
        Initialize token bucket.

        Args:
            capacity: Maximum tokens in bucket
            refill_rate: Tokens added per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()
        self._lock = asyncio.Lock()

    async def consume(self, tokens: int = 1) -> Tuple[bool, int]:
        """
        Try to consume tokens from bucket.

        Returns:
            Tuple of (success, remaining_tokens)
        """
        async with self._lock:
            now = time.time()
            elapsed = now - self.last_refill

            # Refill tokens
            self.tokens = min(
                self.capacity,
                self.tokens + elapsed * self.refill_rate
            )
            self.last_refill = now

            # Try to consume
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True, int(self.tokens)
            return False, int(self.tokens)

    async def get_tokens(self) -> int:
        """Get current token count."""
        async with self._lock:
            now = time.time()
            elapsed = now - self.last_refill
            return min(
                self.capacity,
                int(self.tokens + elapsed * self.refill_rate)
            )


class SlidingWindowCounter:
    """Sliding window rate limiter implementation."""

    def __init__(self, window_size: int, limit: int):
        """
        Initialize sliding window.

        Args:
            window_size: Window size in seconds
            limit: Maximum requests in window
        """
        self.window_size = window_size
        self.limit = limit
        self.requests: List[float] = []
        self._lock = asyncio.Lock()

    async def record(self) -> Tuple[bool, int]:
        """
        Record a request and check if allowed.

        Returns:
            Tuple of (allowed, remaining)
        """
        async with self._lock:
            now = time.time()
            cutoff = now - self.window_size

            # Remove expired requests
            self.requests = [ts for ts in self.requests if ts > cutoff]

            # Check if under limit
            if len(self.requests) < self.limit:
                self.requests.append(now)
                return True, self.limit - len(self.requests)

            return False, 0

    async def get_count(self) -> int:
        """Get current request count in window."""
        async with self._lock:
            now = time.time()
            cutoff = now - self.window_size
            return len([ts for ts in self.requests if ts > cutoff])


class RateLimitStore:
    """
    In-memory rate limit state store.

    In production, replace with Redis for distributed rate limiting.
    """

    def __init__(self):
        self._buckets: Dict[str, TokenBucket] = {}
        self._windows: Dict[str, SlidingWindowCounter] = {}
        self._fixed_windows: Dict[str, Dict[str, Any]] = {}
        self._cleanup_interval = 300  # 5 minutes
        self._last_cleanup = time.time()

    def _make_key(
        self,
        rule: RateLimitRule,
        scope_value: str
    ) -> str:
        """Generate storage key for rate limit state."""
        return f"{rule.name}:{rule.scope.value}:{scope_value}"

    async def check_and_record(
        self,
        rule: RateLimitRule,
        scope_value: str
    ) -> RateLimitResult:
        """
        Check rate limit and record request.

        Args:
            rule: Rate limit rule to check
            scope_value: Value for the scope (tenant_id, user_id, ip, etc.)

        Returns:
            RateLimitResult with check outcome
        """
        key = self._make_key(rule, scope_value)
        now = time.time()

        if rule.strategy == RateLimitStrategy.TOKEN_BUCKET:
            return await self._check_token_bucket(key, rule)
        elif rule.strategy == RateLimitStrategy.SLIDING_WINDOW:
            return await self._check_sliding_window(key, rule)
        else:
            return await self._check_fixed_window(key, rule)

    async def _check_token_bucket(
        self,
        key: str,
        rule: RateLimitRule
    ) -> RateLimitResult:
        """Check using token bucket strategy."""
        if key not in self._buckets:
            # tokens/second = requests/window
            refill_rate = rule.requests / rule.window_seconds
            capacity = int(rule.requests * rule.burst_multiplier)
            self._buckets[key] = TokenBucket(capacity, refill_rate)

        bucket = self._buckets[key]
        allowed, remaining = await bucket.consume()

        return RateLimitResult(
            allowed=allowed,
            remaining=remaining,
            limit=bucket.capacity,
            reset_at=time.time() + (bucket.capacity - remaining) / bucket.refill_rate,
            retry_after=int(1 / bucket.refill_rate) if not allowed else None,
            rule_name=rule.name
        )

    async def _check_sliding_window(
        self,
        key: str,
        rule: RateLimitRule
    ) -> RateLimitResult:
        """Check using sliding window strategy."""
        if key not in self._windows:
            self._windows[key] = SlidingWindowCounter(
                rule.window_seconds,
                rule.requests
            )

        window = self._windows[key]
        allowed, remaining = await window.record()

        return RateLimitResult(
            allowed=allowed,
            remaining=remaining,
            limit=rule.requests,
            reset_at=time.time() + rule.window_seconds,
            retry_after=rule.window_seconds if not allowed else None,
            rule_name=rule.name
        )

    async def _check_fixed_window(
        self,
        key: str,
        rule: RateLimitRule
    ) -> RateLimitResult:
        """Check using fixed window strategy."""
        now = time.time()
        window_start = int(now / rule.window_seconds) * rule.window_seconds
        window_key = f"{key}:{window_start}"

        if window_key not in self._fixed_windows:
            self._fixed_windows[window_key] = {
                "count": 0,
                "window_start": window_start
            }

        state = self._fixed_windows[window_key]
        state["count"] += 1

        allowed = state["count"] <= rule.requests
        remaining = max(0, rule.requests - state["count"])

        return RateLimitResult(
            allowed=allowed,
            remaining=remaining,
            limit=rule.requests,
            reset_at=window_start + rule.window_seconds,
            retry_after=int(window_start + rule.window_seconds - now) if not allowed else None,
            rule_name=rule.name
        )

    async def cleanup(self) -> int:
        """Clean up expired state. Returns count of cleaned entries."""
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return 0

        self._last_cleanup = now
        cleaned = 0

        # Clean fixed windows
        expired_keys = [
            k for k, v in self._fixed_windows.items()
            if now - v.get("window_start", 0) > 3600  # 1 hour
        ]
        for key in expired_keys:
            del self._fixed_windows[key]
            cleaned += 1

        return cleaned


class SyncRateLimiter:
    """
    Intelligent rate limiter for Sync Gateway.

    Features:
    - Multiple rate limiting strategies
    - Tenant, user, and IP-based limits
    - Endpoint-specific rules
    - Burst handling
    - DDoS protection
    """

    def __init__(self):
        self._store = RateLimitStore()
        self._rules: List[RateLimitRule] = []
        self._default_rules = self._create_default_rules()

    def _create_default_rules(self) -> List[RateLimitRule]:
        """Create default rate limit rules."""
        return [
            # Global rate limit
            RateLimitRule(
                name="global_limit",
                scope=RateLimitScope.GLOBAL,
                requests=10000,
                window_seconds=60,
                strategy=RateLimitStrategy.SLIDING_WINDOW,
                priority=0
            ),
            # Per-tenant limit
            RateLimitRule(
                name="tenant_limit",
                scope=RateLimitScope.TENANT,
                requests=1000,
                window_seconds=60,
                strategy=RateLimitStrategy.TOKEN_BUCKET,
                burst_multiplier=2.0,
                priority=10
            ),
            # Per-user limit
            RateLimitRule(
                name="user_limit",
                scope=RateLimitScope.USER,
                requests=100,
                window_seconds=60,
                strategy=RateLimitStrategy.SLIDING_WINDOW,
                priority=20
            ),
            # Per-IP limit (anti-DDoS)
            RateLimitRule(
                name="ip_limit",
                scope=RateLimitScope.IP,
                requests=200,
                window_seconds=60,
                strategy=RateLimitStrategy.FIXED_WINDOW,
                priority=30
            ),
            # Sync job execution limit
            RateLimitRule(
                name="sync_execution_limit",
                scope=RateLimitScope.TENANT,
                requests=10,
                window_seconds=60,
                strategy=RateLimitStrategy.TOKEN_BUCKET,
                path_pattern="/api/v1/sync/jobs/*/start",
                priority=40
            ),
            # Push API limit
            RateLimitRule(
                name="push_api_limit",
                scope=RateLimitScope.TENANT,
                requests=100,
                window_seconds=60,
                strategy=RateLimitStrategy.SLIDING_WINDOW,
                path_pattern="/api/v1/sync/push/*",
                methods=["POST"],
                priority=40
            )
        ]

    def add_rule(self, rule: RateLimitRule) -> None:
        """Add a rate limit rule."""
        self._rules.append(rule)
        self._rules.sort(key=lambda r: r.priority, reverse=True)
        logger.info(f"Added rate limit rule: {rule.name}")

    def remove_rule(self, name: str) -> bool:
        """Remove a rate limit rule by name."""
        for i, rule in enumerate(self._rules):
            if rule.name == name:
                del self._rules[i]
                logger.info(f"Removed rate limit rule: {name}")
                return True
        return False

    def get_rules(self) -> List[RateLimitRule]:
        """Get all active rules."""
        return self._rules + self._default_rules

    def _match_rule(self, rule: RateLimitRule, request: Request) -> bool:
        """Check if rule matches the request."""
        if not rule.enabled:
            return False

        # Check path pattern
        if rule.path_pattern:
            import fnmatch
            if not fnmatch.fnmatch(str(request.url.path), rule.path_pattern):
                return False

        # Check methods
        if rule.methods:
            if request.method not in rule.methods:
                return False

        return True

    def _get_scope_value(
        self,
        rule: RateLimitRule,
        request: Request,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Optional[str]:
        """Get the scope value for rate limiting."""
        if rule.scope == RateLimitScope.GLOBAL:
            return "global"
        elif rule.scope == RateLimitScope.TENANT:
            return tenant_id
        elif rule.scope == RateLimitScope.USER:
            return user_id
        elif rule.scope == RateLimitScope.IP:
            return request.client.host if request.client else None
        elif rule.scope == RateLimitScope.ENDPOINT:
            return f"{request.method}:{request.url.path}"
        return None

    async def check_rate_limit(
        self,
        request: Request,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> RateLimitResult:
        """
        Check rate limit for a request.

        Args:
            request: FastAPI request
            tenant_id: Current tenant ID
            user_id: Current user ID

        Returns:
            RateLimitResult
        """
        all_rules = self._rules + self._default_rules

        for rule in all_rules:
            if not self._match_rule(rule, request):
                continue

            scope_value = self._get_scope_value(rule, request, tenant_id, user_id)
            if not scope_value:
                continue

            result = await self._store.check_and_record(rule, scope_value)

            if not result.allowed:
                return result

        # All checks passed
        return RateLimitResult(
            allowed=True,
            remaining=-1,  # Unknown for multiple rules
            limit=-1,
            reset_at=time.time() + 60
        )

    def create_response_headers(self, result: RateLimitResult) -> Dict[str, str]:
        """Create rate limit response headers."""
        headers = {
            "X-RateLimit-Limit": str(result.limit),
            "X-RateLimit-Remaining": str(result.remaining),
            "X-RateLimit-Reset": str(int(result.reset_at))
        }

        if result.retry_after:
            headers["Retry-After"] = str(result.retry_after)

        if result.rule_name:
            headers["X-RateLimit-Rule"] = result.rule_name

        return headers


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for enforcing rate limits."""

    def __init__(
        self,
        app: ASGIApp,
        rate_limiter: Optional[SyncRateLimiter] = None,
        exclude_paths: Optional[List[str]] = None
    ):
        super().__init__(app)
        self.rate_limiter = rate_limiter or SyncRateLimiter()
        self.exclude_paths = exclude_paths or ["/health", "/metrics"]

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint
    ) -> Response:
        """Apply rate limiting to request."""
        # Skip excluded paths
        path = str(request.url.path)
        for exclude in self.exclude_paths:
            if path.startswith(exclude):
                return await call_next(request)

        # Get tenant and user from request state if available
        tenant_id = getattr(request.state, "tenant_id", None)
        user_id = getattr(request.state, "user_id", None)

        # Also try headers
        if not tenant_id:
            tenant_id = request.headers.get("X-Tenant-ID")

        # Check rate limit
        result = await self.rate_limiter.check_rate_limit(
            request, tenant_id, user_id
        )

        if not result.allowed:
            headers = self.rate_limiter.create_response_headers(result)
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Rate limit exceeded",
                    "error_code": "RATE_LIMIT_EXCEEDED",
                    "retry_after": result.retry_after,
                    "rule": result.rule_name
                },
                headers=headers
            )

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        headers = self.rate_limiter.create_response_headers(result)
        for key, value in headers.items():
            response.headers[key] = value

        return response


# Global rate limiter instance
sync_rate_limiter = SyncRateLimiter()
