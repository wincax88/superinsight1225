"""
Sync Gateway Authentication and Authorization.

Provides authentication handlers for API Key, JWT, and OAuth 2.0,
along with fine-grained permission control.
"""

import hashlib
import hmac
import logging
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import jwt
from fastapi import Depends, HTTPException, Request, Security
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AuthMethod(str, Enum):
    """Authentication method types."""
    API_KEY = "api_key"
    JWT = "jwt"
    OAUTH2 = "oauth2"
    HMAC = "hmac"


class PermissionLevel(str, Enum):
    """Permission levels for access control."""
    NONE = "none"
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"


class ResourceType(str, Enum):
    """Resource types for permission control."""
    SYNC_JOB = "sync_job"
    DATA_SOURCE = "data_source"
    SYNC_EXECUTION = "sync_execution"
    DATA_CONFLICT = "data_conflict"
    TRANSFORMATION = "transformation"
    DATASET = "dataset"
    AUDIT_LOG = "audit_log"


@dataclass
class Permission:
    """Permission definition."""
    resource_type: ResourceType
    level: PermissionLevel
    resource_id: Optional[str] = None  # Specific resource or None for all
    conditions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuthToken:
    """Authenticated token information."""
    token_id: str
    user_id: str
    tenant_id: str
    auth_method: AuthMethod
    permissions: List[Permission]
    issued_at: datetime
    expires_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        return datetime.utcnow() > self.expires_at

    def has_permission(
        self,
        resource_type: ResourceType,
        level: PermissionLevel,
        resource_id: Optional[str] = None
    ) -> bool:
        """Check if token has the required permission."""
        for perm in self.permissions:
            if perm.resource_type != resource_type:
                continue

            # Check level (higher levels include lower ones)
            level_order = [
                PermissionLevel.NONE,
                PermissionLevel.READ,
                PermissionLevel.WRITE,
                PermissionLevel.ADMIN
            ]
            if level_order.index(perm.level) < level_order.index(level):
                continue

            # Check resource ID if specified
            if perm.resource_id and resource_id:
                if perm.resource_id != resource_id:
                    continue

            return True

        return False


class TokenPayload(BaseModel):
    """JWT token payload structure."""
    sub: str  # user_id
    tenant_id: str
    permissions: List[Dict[str, Any]] = Field(default_factory=list)
    iat: int  # issued at
    exp: int  # expiration
    jti: str  # token id
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AuthConfig(BaseModel):
    """Authentication configuration."""
    # JWT settings
    jwt_secret_key: str = "your-secret-key-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = 60
    jwt_refresh_expiry_days: int = 7

    # API Key settings
    api_key_header: str = "X-API-Key"
    api_key_prefix: str = "sk_"

    # HMAC settings
    hmac_header: str = "X-HMAC-Signature"
    hmac_timestamp_header: str = "X-Timestamp"
    hmac_max_age_seconds: int = 300

    # OAuth settings
    oauth_token_url: Optional[str] = None
    oauth_client_id: Optional[str] = None
    oauth_client_secret: Optional[str] = None


class SyncAuthHandler:
    """
    Authentication handler for Sync Gateway.

    Supports multiple authentication methods:
    - API Key authentication
    - JWT token authentication
    - HMAC signature verification
    - OAuth 2.0 (placeholder)
    """

    def __init__(self, config: Optional[AuthConfig] = None):
        self.config = config or AuthConfig()
        self._api_keys: Dict[str, Dict[str, Any]] = {}  # In production, use database
        self._revoked_tokens: Set[str] = set()
        self._refresh_tokens: Dict[str, Dict[str, Any]] = {}

        # Security handlers
        self._bearer = HTTPBearer(auto_error=False)
        self._api_key_header = APIKeyHeader(
            name=self.config.api_key_header,
            auto_error=False
        )

    def generate_api_key(
        self,
        user_id: str,
        tenant_id: str,
        permissions: List[Permission],
        name: Optional[str] = None,
        expires_days: Optional[int] = None
    ) -> Tuple[str, str]:
        """
        Generate a new API key.

        Returns:
            Tuple of (api_key, key_id)
        """
        key_id = secrets.token_hex(8)
        key_secret = secrets.token_hex(32)
        api_key = f"{self.config.api_key_prefix}{key_id}_{key_secret}"

        # Hash the key for storage
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        expires_at = None
        if expires_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_days)

        self._api_keys[key_hash] = {
            "key_id": key_id,
            "user_id": user_id,
            "tenant_id": tenant_id,
            "permissions": permissions,
            "name": name,
            "created_at": datetime.utcnow(),
            "expires_at": expires_at,
            "last_used_at": None
        }

        logger.info(f"Generated API key {key_id} for user {user_id}")
        return api_key, key_id

    def validate_api_key(self, api_key: str) -> Optional[AuthToken]:
        """Validate an API key and return auth token."""
        if not api_key or not api_key.startswith(self.config.api_key_prefix):
            return None

        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        key_data = self._api_keys.get(key_hash)

        if not key_data:
            return None

        # Check expiration
        if key_data.get("expires_at") and datetime.utcnow() > key_data["expires_at"]:
            return None

        # Update last used
        key_data["last_used_at"] = datetime.utcnow()

        return AuthToken(
            token_id=key_data["key_id"],
            user_id=key_data["user_id"],
            tenant_id=key_data["tenant_id"],
            auth_method=AuthMethod.API_KEY,
            permissions=key_data["permissions"],
            issued_at=key_data["created_at"],
            expires_at=key_data.get("expires_at") or datetime.utcnow() + timedelta(days=365),
            metadata={"name": key_data.get("name")}
        )

    def revoke_api_key(self, key_id: str) -> bool:
        """Revoke an API key by its ID."""
        for key_hash, key_data in list(self._api_keys.items()):
            if key_data["key_id"] == key_id:
                del self._api_keys[key_hash]
                logger.info(f"Revoked API key {key_id}")
                return True
        return False

    def generate_jwt_token(
        self,
        user_id: str,
        tenant_id: str,
        permissions: List[Permission],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, str]:
        """
        Generate JWT access and refresh tokens.

        Returns:
            Tuple of (access_token, refresh_token)
        """
        now = datetime.utcnow()
        token_id = secrets.token_hex(16)

        # Create access token
        access_payload = TokenPayload(
            sub=user_id,
            tenant_id=tenant_id,
            permissions=[
                {
                    "resource_type": p.resource_type.value,
                    "level": p.level.value,
                    "resource_id": p.resource_id
                }
                for p in permissions
            ],
            iat=int(now.timestamp()),
            exp=int((now + timedelta(minutes=self.config.jwt_expiry_minutes)).timestamp()),
            jti=token_id,
            metadata=metadata or {}
        )

        access_token = jwt.encode(
            access_payload.model_dump(),
            self.config.jwt_secret_key,
            algorithm=self.config.jwt_algorithm
        )

        # Create refresh token
        refresh_token_id = secrets.token_hex(16)
        refresh_payload = {
            "sub": user_id,
            "tenant_id": tenant_id,
            "jti": refresh_token_id,
            "type": "refresh",
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(days=self.config.jwt_refresh_expiry_days)).timestamp())
        }

        refresh_token = jwt.encode(
            refresh_payload,
            self.config.jwt_secret_key,
            algorithm=self.config.jwt_algorithm
        )

        # Store refresh token info
        self._refresh_tokens[refresh_token_id] = {
            "user_id": user_id,
            "tenant_id": tenant_id,
            "permissions": permissions,
            "created_at": now,
            "metadata": metadata or {}
        }

        logger.info(f"Generated JWT token for user {user_id}")
        return access_token, refresh_token

    def validate_jwt_token(self, token: str) -> Optional[AuthToken]:
        """Validate a JWT token and return auth token."""
        try:
            payload = jwt.decode(
                token,
                self.config.jwt_secret_key,
                algorithms=[self.config.jwt_algorithm]
            )

            token_id = payload.get("jti")

            # Check if revoked
            if token_id in self._revoked_tokens:
                return None

            # Parse permissions
            permissions = [
                Permission(
                    resource_type=ResourceType(p["resource_type"]),
                    level=PermissionLevel(p["level"]),
                    resource_id=p.get("resource_id")
                )
                for p in payload.get("permissions", [])
            ]

            return AuthToken(
                token_id=token_id,
                user_id=payload["sub"],
                tenant_id=payload["tenant_id"],
                auth_method=AuthMethod.JWT,
                permissions=permissions,
                issued_at=datetime.fromtimestamp(payload["iat"]),
                expires_at=datetime.fromtimestamp(payload["exp"]),
                metadata=payload.get("metadata", {})
            )

        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
            return None

    def refresh_jwt_token(self, refresh_token: str) -> Optional[Tuple[str, str]]:
        """
        Refresh JWT tokens using a refresh token.

        Returns:
            New (access_token, refresh_token) tuple or None
        """
        try:
            payload = jwt.decode(
                refresh_token,
                self.config.jwt_secret_key,
                algorithms=[self.config.jwt_algorithm]
            )

            if payload.get("type") != "refresh":
                return None

            refresh_token_id = payload.get("jti")
            refresh_data = self._refresh_tokens.get(refresh_token_id)

            if not refresh_data:
                return None

            # Generate new tokens
            new_access, new_refresh = self.generate_jwt_token(
                user_id=refresh_data["user_id"],
                tenant_id=refresh_data["tenant_id"],
                permissions=refresh_data["permissions"],
                metadata=refresh_data.get("metadata")
            )

            # Revoke old refresh token
            del self._refresh_tokens[refresh_token_id]

            return new_access, new_refresh

        except jwt.InvalidTokenError:
            return None

    def revoke_jwt_token(self, token_id: str) -> None:
        """Revoke a JWT token."""
        self._revoked_tokens.add(token_id)
        logger.info(f"Revoked JWT token {token_id}")

    def validate_hmac_signature(
        self,
        request_body: bytes,
        signature: str,
        timestamp: str,
        secret_key: str
    ) -> bool:
        """
        Validate HMAC signature for request.

        Args:
            request_body: Raw request body bytes
            signature: Provided HMAC signature
            timestamp: Request timestamp
            secret_key: Shared secret key
        """
        try:
            # Check timestamp freshness
            ts = int(timestamp)
            current_ts = int(time.time())
            if abs(current_ts - ts) > self.config.hmac_max_age_seconds:
                logger.warning("HMAC timestamp too old")
                return False

            # Compute expected signature
            message = f"{timestamp}.{request_body.decode()}"
            expected_signature = hmac.new(
                secret_key.encode(),
                message.encode(),
                hashlib.sha256
            ).hexdigest()

            # Constant-time comparison
            return hmac.compare_digest(signature, expected_signature)

        except Exception as e:
            logger.warning(f"HMAC validation error: {e}")
            return False

    async def authenticate(self, request: Request) -> Optional[AuthToken]:
        """
        Authenticate request using available methods.

        Tries in order: JWT Bearer > API Key > HMAC
        """
        # Try JWT Bearer token
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]
            auth_token = self.validate_jwt_token(token)
            if auth_token:
                return auth_token

        # Try API Key
        api_key = request.headers.get(self.config.api_key_header)
        if api_key:
            auth_token = self.validate_api_key(api_key)
            if auth_token:
                return auth_token

        return None

    def require_auth(self, required_permissions: Optional[List[Permission]] = None):
        """
        Dependency for requiring authentication.

        Usage:
            @router.get("/protected")
            async def protected_route(auth: AuthToken = Depends(auth_handler.require_auth())):
                pass
        """
        async def dependency(request: Request) -> AuthToken:
            auth_token = await self.authenticate(request)

            if not auth_token:
                raise HTTPException(
                    status_code=401,
                    detail="Authentication required",
                    headers={"WWW-Authenticate": "Bearer"}
                )

            if auth_token.is_expired:
                raise HTTPException(
                    status_code=401,
                    detail="Token expired",
                    headers={"WWW-Authenticate": "Bearer"}
                )

            # Check permissions
            if required_permissions:
                for required in required_permissions:
                    if not auth_token.has_permission(
                        required.resource_type,
                        required.level,
                        required.resource_id
                    ):
                        raise HTTPException(
                            status_code=403,
                            detail=f"Insufficient permissions for {required.resource_type.value}"
                        )

            return auth_token

        return dependency

    def require_permission(
        self,
        resource_type: ResourceType,
        level: PermissionLevel = PermissionLevel.READ
    ):
        """
        Decorator/dependency for checking specific permission.

        Usage:
            @router.get("/sync-jobs")
            async def list_jobs(
                auth: AuthToken = Depends(
                    auth_handler.require_permission(ResourceType.SYNC_JOB, PermissionLevel.READ)
                )
            ):
                pass
        """
        return self.require_auth([
            Permission(resource_type=resource_type, level=level)
        ])


# Global auth handler instance
sync_auth_handler = SyncAuthHandler()


def get_current_user(
    auth_token: AuthToken = Depends(sync_auth_handler.require_auth())
) -> AuthToken:
    """FastAPI dependency to get current authenticated user."""
    return auth_token


def get_tenant_id(
    auth_token: AuthToken = Depends(sync_auth_handler.require_auth())
) -> str:
    """FastAPI dependency to get current tenant ID."""
    return auth_token.tenant_id
