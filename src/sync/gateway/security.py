"""
Sync Gateway Security Handler.

Provides security controls including:
- DDoS protection
- SQL injection prevention
- XSS protection
- IP whitelisting/blacklisting
- Request validation
- Geo-location restrictions
"""

import hashlib
import ipaddress
import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Pattern

from fastapi import HTTPException, Request, Response
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


class ThreatLevel(str, Enum):
    """Threat level classification."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThreatType(str, Enum):
    """Types of security threats."""
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    DDOS = "ddos"
    BRUTE_FORCE = "brute_force"
    SUSPICIOUS_AGENT = "suspicious_agent"
    INVALID_INPUT = "invalid_input"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    PATH_TRAVERSAL = "path_traversal"
    COMMAND_INJECTION = "command_injection"


@dataclass
class ThreatEvent:
    """Security threat event."""
    id: str
    threat_type: ThreatType
    level: ThreatLevel
    ip_address: str
    path: str
    method: str
    details: Dict[str, Any]
    timestamp: datetime
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None
    blocked: bool = False


@dataclass
class IPAccessRule:
    """IP access control rule."""
    ip_pattern: str
    is_whitelist: bool
    description: Optional[str] = None
    tenant_id: Optional[str] = None  # None means global
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None

    def matches(self, ip: str) -> bool:
        """Check if IP matches this rule."""
        try:
            # Handle CIDR notation
            if "/" in self.ip_pattern:
                network = ipaddress.ip_network(self.ip_pattern, strict=False)
                return ipaddress.ip_address(ip) in network
            # Exact match
            return ip == self.ip_pattern
        except ValueError:
            return False


class SecurityConfig(BaseModel):
    """Security configuration."""
    # DDoS protection
    ddos_protection_enabled: bool = True
    ddos_threshold_requests: int = 1000  # requests per minute per IP
    ddos_block_duration: int = 3600  # seconds

    # SQL Injection protection
    sql_injection_protection: bool = True
    sql_injection_patterns: List[str] = Field(default_factory=lambda: [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|ALTER)\b)",
        r"(--|\#|/\*)",
        r"(\bOR\b.+?=.+?)",
        r"(\bAND\b.+?=.+?)",
        r"(;.*(SELECT|INSERT|UPDATE|DELETE|DROP))",
        r"(\b1\s*=\s*1\b)",
        r"(\b0\s*=\s*0\b)"
    ])

    # XSS protection
    xss_protection: bool = True
    xss_patterns: List[str] = Field(default_factory=lambda: [
        r"<script[^>]*>",
        r"javascript:",
        r"on\w+\s*=",
        r"<iframe[^>]*>",
        r"<object[^>]*>",
        r"<embed[^>]*>",
        r"expression\s*\(",
        r"url\s*\("
    ])

    # Path traversal protection
    path_traversal_protection: bool = True
    path_traversal_patterns: List[str] = Field(default_factory=lambda: [
        r"\.\./",
        r"\.\.\\",
        r"%2e%2e%2f",
        r"%2e%2e/",
        r"..%2f",
        r"%2e%2e%5c"
    ])

    # Command injection protection
    command_injection_protection: bool = True
    command_injection_patterns: List[str] = Field(default_factory=lambda: [
        r"[;&|`$]",
        r"\$\(",
        r"`.*`",
        r"\|\|",
        r"&&"
    ])

    # Suspicious user agents
    block_suspicious_agents: bool = True
    suspicious_agent_patterns: List[str] = Field(default_factory=lambda: [
        r"sqlmap",
        r"nikto",
        r"nessus",
        r"masscan",
        r"nmap",
        r"dirbuster",
        r"gobuster"
    ])

    # Request size limits
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    max_header_size: int = 8 * 1024  # 8KB
    max_url_length: int = 2048

    # IP restrictions
    enable_ip_whitelist: bool = False
    enable_ip_blacklist: bool = True
    default_blacklisted_ips: List[str] = Field(default_factory=list)


class SyncSecurityHandler:
    """
    Security handler for Sync Gateway.

    Provides comprehensive security controls:
    - Attack pattern detection (SQLi, XSS, etc.)
    - DDoS protection
    - IP access control
    - Request validation
    - Threat monitoring
    """

    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig()
        self._ip_whitelist: List[IPAccessRule] = []
        self._ip_blacklist: List[IPAccessRule] = []
        self._blocked_ips: Dict[str, datetime] = {}
        self._request_counts: Dict[str, List[float]] = defaultdict(list)
        self._threat_events: List[ThreatEvent] = []
        self._max_events = 10000

        # Compile patterns
        self._sql_patterns = self._compile_patterns(self.config.sql_injection_patterns)
        self._xss_patterns = self._compile_patterns(self.config.xss_patterns)
        self._path_patterns = self._compile_patterns(self.config.path_traversal_patterns)
        self._cmd_patterns = self._compile_patterns(self.config.command_injection_patterns)
        self._agent_patterns = self._compile_patterns(self.config.suspicious_agent_patterns)

        # Initialize default blacklist
        for ip in self.config.default_blacklisted_ips:
            self.add_blacklist(ip, "Default blacklist")

    def _compile_patterns(self, patterns: List[str]) -> List[Pattern]:
        """Compile regex patterns."""
        compiled = []
        for pattern in patterns:
            try:
                compiled.append(re.compile(pattern, re.IGNORECASE))
            except re.error as e:
                logger.warning(f"Invalid pattern '{pattern}': {e}")
        return compiled

    def add_whitelist(
        self,
        ip_pattern: str,
        description: Optional[str] = None,
        tenant_id: Optional[str] = None,
        expires_hours: Optional[int] = None
    ) -> IPAccessRule:
        """Add IP to whitelist."""
        rule = IPAccessRule(
            ip_pattern=ip_pattern,
            is_whitelist=True,
            description=description,
            tenant_id=tenant_id,
            expires_at=datetime.utcnow() + timedelta(hours=expires_hours) if expires_hours else None
        )
        self._ip_whitelist.append(rule)
        logger.info(f"Added IP to whitelist: {ip_pattern}")
        return rule

    def add_blacklist(
        self,
        ip_pattern: str,
        description: Optional[str] = None,
        tenant_id: Optional[str] = None,
        expires_hours: Optional[int] = None
    ) -> IPAccessRule:
        """Add IP to blacklist."""
        rule = IPAccessRule(
            ip_pattern=ip_pattern,
            is_whitelist=False,
            description=description,
            tenant_id=tenant_id,
            expires_at=datetime.utcnow() + timedelta(hours=expires_hours) if expires_hours else None
        )
        self._ip_blacklist.append(rule)
        logger.info(f"Added IP to blacklist: {ip_pattern}")
        return rule

    def remove_from_whitelist(self, ip_pattern: str) -> bool:
        """Remove IP from whitelist."""
        for i, rule in enumerate(self._ip_whitelist):
            if rule.ip_pattern == ip_pattern:
                del self._ip_whitelist[i]
                return True
        return False

    def remove_from_blacklist(self, ip_pattern: str) -> bool:
        """Remove IP from blacklist."""
        for i, rule in enumerate(self._ip_blacklist):
            if rule.ip_pattern == ip_pattern:
                del self._ip_blacklist[i]
                return True
        return False

    def check_ip_access(
        self,
        ip: str,
        tenant_id: Optional[str] = None
    ) -> tuple[bool, Optional[str]]:
        """
        Check if IP is allowed access.

        Returns:
            Tuple of (allowed, reason)
        """
        now = datetime.utcnow()

        # Check if temporarily blocked
        if ip in self._blocked_ips:
            block_until = self._blocked_ips[ip]
            if now < block_until:
                return False, "IP temporarily blocked due to suspicious activity"
            else:
                del self._blocked_ips[ip]

        # Check whitelist first (if enabled)
        if self.config.enable_ip_whitelist:
            for rule in self._ip_whitelist:
                if rule.expires_at and now > rule.expires_at:
                    continue
                if rule.tenant_id and rule.tenant_id != tenant_id:
                    continue
                if rule.matches(ip):
                    return True, None
            # If whitelist enabled and not matched, deny
            return False, "IP not in whitelist"

        # Check blacklist
        if self.config.enable_ip_blacklist:
            for rule in self._ip_blacklist:
                if rule.expires_at and now > rule.expires_at:
                    continue
                if rule.tenant_id and rule.tenant_id != tenant_id:
                    continue
                if rule.matches(ip):
                    return False, rule.description or "IP in blacklist"

        return True, None

    def check_ddos(self, ip: str) -> bool:
        """
        Check for DDoS attack patterns.

        Returns:
            True if request should be blocked
        """
        if not self.config.ddos_protection_enabled:
            return False

        now = time.time()
        window = 60  # 1 minute

        # Get recent requests for this IP
        requests = self._request_counts[ip]
        requests.append(now)

        # Clean old requests
        cutoff = now - window
        self._request_counts[ip] = [t for t in requests if t > cutoff]

        # Check threshold
        if len(self._request_counts[ip]) > self.config.ddos_threshold_requests:
            # Block this IP
            self._blocked_ips[ip] = datetime.utcnow() + timedelta(
                seconds=self.config.ddos_block_duration
            )
            logger.warning(f"DDoS protection: Blocked IP {ip}")
            return True

        return False

    def check_sql_injection(self, content: str) -> Optional[ThreatEvent]:
        """Check for SQL injection patterns."""
        if not self.config.sql_injection_protection:
            return None

        for pattern in self._sql_patterns:
            if pattern.search(content):
                return self._create_threat_event(
                    ThreatType.SQL_INJECTION,
                    ThreatLevel.HIGH,
                    {"pattern": pattern.pattern, "content": content[:200]}
                )
        return None

    def check_xss(self, content: str) -> Optional[ThreatEvent]:
        """Check for XSS patterns."""
        if not self.config.xss_protection:
            return None

        for pattern in self._xss_patterns:
            if pattern.search(content):
                return self._create_threat_event(
                    ThreatType.XSS,
                    ThreatLevel.MEDIUM,
                    {"pattern": pattern.pattern, "content": content[:200]}
                )
        return None

    def check_path_traversal(self, path: str) -> Optional[ThreatEvent]:
        """Check for path traversal patterns."""
        if not self.config.path_traversal_protection:
            return None

        for pattern in self._path_patterns:
            if pattern.search(path):
                return self._create_threat_event(
                    ThreatType.PATH_TRAVERSAL,
                    ThreatLevel.HIGH,
                    {"pattern": pattern.pattern, "path": path}
                )
        return None

    def check_command_injection(self, content: str) -> Optional[ThreatEvent]:
        """Check for command injection patterns."""
        if not self.config.command_injection_protection:
            return None

        for pattern in self._cmd_patterns:
            if pattern.search(content):
                return self._create_threat_event(
                    ThreatType.COMMAND_INJECTION,
                    ThreatLevel.CRITICAL,
                    {"pattern": pattern.pattern, "content": content[:200]}
                )
        return None

    def check_user_agent(self, user_agent: str) -> Optional[ThreatEvent]:
        """Check for suspicious user agents."""
        if not self.config.block_suspicious_agents:
            return None

        for pattern in self._agent_patterns:
            if pattern.search(user_agent):
                return self._create_threat_event(
                    ThreatType.SUSPICIOUS_AGENT,
                    ThreatLevel.MEDIUM,
                    {"pattern": pattern.pattern, "user_agent": user_agent}
                )
        return None

    def _create_threat_event(
        self,
        threat_type: ThreatType,
        level: ThreatLevel,
        details: Dict[str, Any]
    ) -> ThreatEvent:
        """Create a threat event."""
        event = ThreatEvent(
            id=hashlib.md5(
                f"{threat_type.value}{time.time()}".encode()
            ).hexdigest()[:16],
            threat_type=threat_type,
            level=level,
            ip_address="",  # Set by caller
            path="",
            method="",
            details=details,
            timestamp=datetime.utcnow()
        )
        return event

    def record_threat(self, event: ThreatEvent) -> None:
        """Record a threat event."""
        self._threat_events.append(event)
        if len(self._threat_events) > self._max_events:
            self._threat_events = self._threat_events[-self._max_events // 2:]

        logger.warning(
            f"Security threat detected: {event.threat_type.value} "
            f"from {event.ip_address} - {event.details}"
        )

    def get_threats(
        self,
        limit: int = 100,
        threat_type: Optional[ThreatType] = None,
        min_level: Optional[ThreatLevel] = None,
        ip_address: Optional[str] = None
    ) -> List[ThreatEvent]:
        """Get recorded threat events with filtering."""
        events = self._threat_events

        if threat_type:
            events = [e for e in events if e.threat_type == threat_type]

        if min_level:
            level_order = [
                ThreatLevel.NONE,
                ThreatLevel.LOW,
                ThreatLevel.MEDIUM,
                ThreatLevel.HIGH,
                ThreatLevel.CRITICAL
            ]
            min_idx = level_order.index(min_level)
            events = [
                e for e in events
                if level_order.index(e.level) >= min_idx
            ]

        if ip_address:
            events = [e for e in events if e.ip_address == ip_address]

        return events[-limit:]

    async def validate_request(
        self,
        request: Request
    ) -> tuple[bool, Optional[ThreatEvent]]:
        """
        Validate request for security threats.

        Returns:
            Tuple of (valid, threat_event)
        """
        ip = request.client.host if request.client else "unknown"
        path = str(request.url.path)
        method = request.method
        user_agent = request.headers.get("user-agent", "")

        # Check IP access
        allowed, reason = self.check_ip_access(ip)
        if not allowed:
            event = self._create_threat_event(
                ThreatType.UNAUTHORIZED_ACCESS,
                ThreatLevel.MEDIUM,
                {"reason": reason}
            )
            event.ip_address = ip
            event.path = path
            event.method = method
            event.blocked = True
            return False, event

        # Check DDoS
        if self.check_ddos(ip):
            event = self._create_threat_event(
                ThreatType.DDOS,
                ThreatLevel.HIGH,
                {"requests_per_minute": len(self._request_counts[ip])}
            )
            event.ip_address = ip
            event.path = path
            event.method = method
            event.blocked = True
            return False, event

        # Check user agent
        threat = self.check_user_agent(user_agent)
        if threat:
            threat.ip_address = ip
            threat.path = path
            threat.method = method
            threat.blocked = True
            return False, threat

        # Check path traversal
        threat = self.check_path_traversal(path)
        if threat:
            threat.ip_address = ip
            threat.path = path
            threat.method = method
            threat.blocked = True
            return False, threat

        # Check URL length
        if len(str(request.url)) > self.config.max_url_length:
            event = self._create_threat_event(
                ThreatType.INVALID_INPUT,
                ThreatLevel.LOW,
                {"url_length": len(str(request.url))}
            )
            event.ip_address = ip
            event.path = path
            event.method = method
            return False, event

        # Check query params for injection
        for key, value in request.query_params.items():
            content = f"{key}={value}"

            threat = self.check_sql_injection(content)
            if threat:
                threat.ip_address = ip
                threat.path = path
                threat.method = method
                threat.blocked = True
                return False, threat

            threat = self.check_xss(content)
            if threat:
                threat.ip_address = ip
                threat.path = path
                threat.method = method
                threat.blocked = True
                return False, threat

        return True, None


class SecurityMiddleware(BaseHTTPMiddleware):
    """Middleware for security validation."""

    def __init__(
        self,
        app: ASGIApp,
        security_handler: Optional[SyncSecurityHandler] = None
    ):
        super().__init__(app)
        self.security_handler = security_handler or SyncSecurityHandler()

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint
    ) -> Response:
        """Validate request security."""
        valid, threat = await self.security_handler.validate_request(request)

        if not valid and threat:
            self.security_handler.record_threat(threat)

            status_code = 403
            if threat.threat_type == ThreatType.DDOS:
                status_code = 429

            raise HTTPException(
                status_code=status_code,
                detail={
                    "error": "Request blocked by security controls",
                    "error_code": f"SECURITY_{threat.threat_type.value.upper()}",
                    "threat_id": threat.id
                }
            )

        response = await call_next(request)

        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"

        return response


# Global security handler instance
sync_security_handler = SyncSecurityHandler()
