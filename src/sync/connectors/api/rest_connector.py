"""
REST API Connector Module.

Provides connector for synchronizing data from REST APIs with support for
various authentication methods, pagination strategies, and rate limiting.
"""

import asyncio
import hashlib
import hmac
import json
import logging
import time
from base64 import b64encode
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Union
from urllib.parse import urlencode, urljoin, urlparse

import aiohttp
from pydantic import BaseModel, Field

from ..base import (
    BaseConnector,
    ConnectionStatus,
    ConnectorConfig,
    DataBatch,
    DataRecord,
    OperationType,
    SyncResult,
)

logger = logging.getLogger(__name__)


class AuthType(str, Enum):
    """Authentication types for REST API."""
    NONE = "none"
    API_KEY = "api_key"
    BASIC = "basic"
    BEARER = "bearer"
    OAUTH2 = "oauth2"
    HMAC = "hmac"
    CUSTOM = "custom"


class ApiKeyLocation(str, Enum):
    """Where to place API key."""
    HEADER = "header"
    QUERY = "query"


class PaginationType(str, Enum):
    """Pagination strategies."""
    NONE = "none"
    OFFSET = "offset"           # offset + limit
    PAGE = "page"               # page + page_size
    CURSOR = "cursor"           # cursor-based
    LINK_HEADER = "link_header" # RFC 5988 Link header
    NEXT_URL = "next_url"       # next URL in response


class HttpMethod(str, Enum):
    """HTTP methods."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    requests_per_second: float = 10.0
    burst_size: int = 20
    retry_on_429: bool = True
    max_retries: int = 3
    retry_delay_seconds: float = 1.0


@dataclass
class AuthConfig:
    """Authentication configuration."""
    auth_type: AuthType = AuthType.NONE

    # API Key auth
    api_key: Optional[str] = None
    api_key_name: str = "X-API-Key"
    api_key_location: ApiKeyLocation = ApiKeyLocation.HEADER

    # Basic auth
    username: Optional[str] = None
    password: Optional[str] = None

    # Bearer token
    bearer_token: Optional[str] = None

    # OAuth2
    oauth2_token_url: Optional[str] = None
    oauth2_client_id: Optional[str] = None
    oauth2_client_secret: Optional[str] = None
    oauth2_scope: Optional[str] = None
    oauth2_grant_type: str = "client_credentials"

    # HMAC
    hmac_key: Optional[str] = None
    hmac_secret: Optional[str] = None
    hmac_algorithm: str = "sha256"

    # Custom headers
    custom_headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class PaginationConfig:
    """Pagination configuration."""
    pagination_type: PaginationType = PaginationType.NONE

    # Offset pagination
    offset_param: str = "offset"
    limit_param: str = "limit"

    # Page pagination
    page_param: str = "page"
    page_size_param: str = "page_size"

    # Cursor pagination
    cursor_param: str = "cursor"
    cursor_response_path: str = "next_cursor"

    # Next URL pagination
    next_url_path: str = "next"

    # Response data location
    data_path: str = "data"           # JSON path to data array
    total_count_path: str = "total"   # JSON path to total count

    # Limits
    default_page_size: int = 100
    max_page_size: int = 1000


class RESTConnectorConfig(ConnectorConfig):
    """REST API connector configuration."""

    # Base URL
    base_url: str

    # Default endpoint for data
    default_endpoint: str = "/"

    # Request settings
    default_method: HttpMethod = HttpMethod.GET
    content_type: str = "application/json"
    accept: str = "application/json"

    # Authentication
    auth: AuthConfig = Field(default_factory=AuthConfig)

    # Pagination
    pagination: PaginationConfig = Field(default_factory=PaginationConfig)

    # Rate limiting
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)

    # Request/Response
    verify_ssl: bool = True
    follow_redirects: bool = True
    max_redirects: int = 5

    # Record mapping
    id_field: str = "id"
    timestamp_field: Optional[str] = None

    # Custom headers
    headers: Dict[str, str] = Field(default_factory=dict)


class RESTConnector(BaseConnector):
    """
    REST API Connector.

    Supports:
    - Multiple authentication methods (API Key, Basic, Bearer, OAuth2, HMAC)
    - Various pagination strategies (offset, page, cursor, Link header)
    - Rate limiting with retry logic
    - Custom data extraction and mapping
    - Streaming for large datasets
    """

    def __init__(self, config: RESTConnectorConfig):
        """Initialize REST connector."""
        super().__init__(config)
        self.rest_config = config
        self._session: Optional[aiohttp.ClientSession] = None
        self._oauth2_token: Optional[str] = None
        self._oauth2_expires: Optional[datetime] = None
        self._rate_limiter = RateLimiter(config.rate_limit)

    async def connect(self) -> bool:
        """Establish HTTP session."""
        try:
            self._set_status(ConnectionStatus.CONNECTING)

            # Create session with default headers
            headers = {
                "Content-Type": self.rest_config.content_type,
                "Accept": self.rest_config.accept,
                "User-Agent": "SuperInsight-Sync/1.0"
            }
            headers.update(self.rest_config.headers)

            # Add authentication headers
            auth_headers = await self._get_auth_headers()
            headers.update(auth_headers)

            timeout = aiohttp.ClientTimeout(
                total=self.rest_config.connection_timeout + self.rest_config.read_timeout,
                connect=self.rest_config.connection_timeout,
                sock_read=self.rest_config.read_timeout
            )

            connector = aiohttp.TCPConnector(
                limit=self.rest_config.pool_size,
                ssl=self.rest_config.verify_ssl if self.rest_config.verify_ssl else False
            )

            self._session = aiohttp.ClientSession(
                headers=headers,
                timeout=timeout,
                connector=connector
            )

            self._set_status(ConnectionStatus.CONNECTED)
            logger.info(f"Connected to REST API: {self.rest_config.base_url}")
            return True

        except Exception as e:
            self._record_error(e)
            self._set_status(ConnectionStatus.ERROR)
            return False

    async def disconnect(self) -> None:
        """Close HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None
        self._set_status(ConnectionStatus.DISCONNECTED)
        logger.info("Disconnected from REST API")

    async def health_check(self) -> bool:
        """Check if API is accessible."""
        if not self._session:
            return False

        try:
            url = urljoin(self.rest_config.base_url, self.rest_config.default_endpoint)
            async with self._session.head(url) as response:
                return response.status < 500
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False

    async def fetch_schema(self) -> Dict[str, Any]:
        """
        Fetch API schema information.

        Attempts to retrieve OpenAPI/Swagger specification if available.
        """
        schema = {
            "type": "rest_api",
            "base_url": self.rest_config.base_url,
            "endpoints": [],
            "version": None
        }

        # Try common OpenAPI endpoints
        openapi_paths = [
            "/openapi.json",
            "/swagger.json",
            "/api/openapi.json",
            "/api/v1/openapi.json",
            "/v1/openapi.json"
        ]

        for path in openapi_paths:
            try:
                url = urljoin(self.rest_config.base_url, path)
                async with self._session.get(url) as response:
                    if response.status == 200:
                        openapi_spec = await response.json()
                        schema["openapi"] = openapi_spec
                        schema["version"] = openapi_spec.get("info", {}).get("version")
                        schema["endpoints"] = list(openapi_spec.get("paths", {}).keys())
                        break
            except Exception:
                continue

        return schema

    async def fetch_data(
        self,
        query: Optional[str] = None,
        table: Optional[str] = None,  # Used as endpoint
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: int = 0,
        incremental_field: Optional[str] = None,
        incremental_value: Optional[str] = None
    ) -> DataBatch:
        """
        Fetch data from REST API endpoint.

        Args:
            query: Custom query string or JSON body
            table: API endpoint (relative to base_url)
            filters: Query parameters
            limit: Maximum records to fetch
            offset: Offset for pagination
            incremental_field: Field for incremental sync
            incremental_value: Last sync value

        Returns:
            DataBatch containing fetched records
        """
        endpoint = table or self.rest_config.default_endpoint
        params = filters or {}

        # Apply pagination
        page_size = min(
            limit or self.rest_config.pagination.default_page_size,
            self.rest_config.pagination.max_page_size
        )

        pagination = self.rest_config.pagination
        if pagination.pagination_type == PaginationType.OFFSET:
            params[pagination.offset_param] = offset
            params[pagination.limit_param] = page_size
        elif pagination.pagination_type == PaginationType.PAGE:
            page = (offset // page_size) + 1
            params[pagination.page_param] = page
            params[pagination.page_size_param] = page_size
        elif pagination.pagination_type == PaginationType.CURSOR:
            if offset > 0:
                params[pagination.cursor_param] = str(offset)
            params[pagination.limit_param] = page_size

        # Apply incremental filter
        if incremental_field and incremental_value:
            params[incremental_field] = incremental_value

        # Make request
        response_data = await self._make_request(
            method=self.rest_config.default_method,
            endpoint=endpoint,
            params=params,
            body=json.loads(query) if query else None
        )

        # Extract records from response
        records = self._extract_records(response_data)
        total_count = self._extract_total_count(response_data, len(records))

        # Check if more data available
        has_more = (offset + len(records)) < total_count

        # Build checkpoint
        checkpoint = {
            "offset": offset + len(records),
            "timestamp": datetime.utcnow().isoformat()
        }

        if pagination.pagination_type == PaginationType.CURSOR:
            next_cursor = self._get_nested_value(response_data, pagination.cursor_response_path)
            if next_cursor:
                checkpoint["cursor"] = next_cursor
                has_more = True

        self._record_read(len(records))

        return DataBatch(
            records=records,
            source_id=self.rest_config.name,
            table_name=endpoint,
            total_count=total_count,
            offset=offset,
            has_more=has_more,
            checkpoint=checkpoint
        )

    async def fetch_data_stream(
        self,
        query: Optional[str] = None,
        table: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        batch_size: Optional[int] = None,
        incremental_field: Optional[str] = None,
        incremental_value: Optional[str] = None
    ) -> AsyncIterator[DataBatch]:
        """
        Stream data from REST API in batches.

        Handles pagination automatically to fetch all available data.
        """
        page_size = batch_size or self.rest_config.pagination.default_page_size
        offset = 0
        cursor = None
        has_more = True

        while has_more:
            # Apply rate limiting
            await self._rate_limiter.acquire()

            params = filters.copy() if filters else {}

            # Handle different pagination types
            pagination = self.rest_config.pagination
            if pagination.pagination_type == PaginationType.CURSOR and cursor:
                params[pagination.cursor_param] = cursor

            batch = await self.fetch_data(
                query=query,
                table=table,
                filters=params,
                limit=page_size,
                offset=offset if pagination.pagination_type != PaginationType.CURSOR else 0,
                incremental_field=incremental_field,
                incremental_value=incremental_value
            )

            if batch.records:
                yield batch

            has_more = batch.has_more
            offset = batch.checkpoint.get("offset", offset + len(batch.records))
            cursor = batch.checkpoint.get("cursor")

            # Safety check to prevent infinite loops
            if not batch.records:
                break

    async def write_data(
        self,
        batch: DataBatch,
        mode: str = "upsert"
    ) -> SyncResult:
        """
        Write data to REST API.

        Args:
            batch: DataBatch to write
            mode: Write mode (insert, update, upsert, delete)

        Returns:
            SyncResult with write statistics
        """
        start_time = time.time()
        result = SyncResult(success=True)

        endpoint = batch.table_name or self.rest_config.default_endpoint

        for record in batch.records:
            try:
                await self._rate_limiter.acquire()

                method = self._get_method_for_operation(record.operation, mode)
                record_endpoint = endpoint

                # For updates/deletes, append record ID to endpoint
                if record.operation in [OperationType.UPDATE, OperationType.DELETE]:
                    record_endpoint = f"{endpoint}/{record.id}"

                body = record.data if method != HttpMethod.DELETE else None

                response = await self._make_request(
                    method=method,
                    endpoint=record_endpoint,
                    body=body
                )

                if record.operation == OperationType.INSERT:
                    result.records_inserted += 1
                elif record.operation == OperationType.UPDATE:
                    result.records_updated += 1
                elif record.operation == OperationType.DELETE:
                    result.records_deleted += 1
                else:
                    result.records_updated += 1

                result.records_processed += 1

            except Exception as e:
                result.records_failed += 1
                result.errors.append({
                    "record_id": record.id,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                })

        result.duration_seconds = time.time() - start_time
        result.success = result.records_failed == 0

        self._record_write(result.records_processed)

        return result

    async def get_record_count(
        self,
        table: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Get total record count from API.

        Attempts to fetch count from metadata or first page.
        """
        endpoint = table or self.rest_config.default_endpoint
        params = filters.copy() if filters else {}

        # Request minimal data to get count
        pagination = self.rest_config.pagination
        if pagination.pagination_type in [PaginationType.OFFSET, PaginationType.CURSOR]:
            params[pagination.limit_param] = 1
        elif pagination.pagination_type == PaginationType.PAGE:
            params[pagination.page_size_param] = 1

        try:
            response = await self._make_request(
                method=HttpMethod.GET,
                endpoint=endpoint,
                params=params
            )
            return self._extract_total_count(response, 0)
        except Exception:
            return 0

    async def _make_request(
        self,
        method: HttpMethod,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        body: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Make HTTP request with retry logic."""
        if not self._session:
            raise RuntimeError("Not connected")

        url = urljoin(self.rest_config.base_url, endpoint)
        request_headers = headers or {}

        # Refresh OAuth2 token if needed
        if self.rest_config.auth.auth_type == AuthType.OAUTH2:
            auth_headers = await self._get_auth_headers()
            request_headers.update(auth_headers)

        retries = 0
        last_error = None

        while retries <= self.rest_config.max_retries:
            try:
                async with self._session.request(
                    method=method.value,
                    url=url,
                    params=params,
                    json=body,
                    headers=request_headers,
                    allow_redirects=self.rest_config.follow_redirects
                ) as response:

                    # Handle rate limiting
                    if response.status == 429:
                        if self.rest_config.rate_limit.retry_on_429:
                            retry_after = int(response.headers.get("Retry-After",
                                self.rest_config.rate_limit.retry_delay_seconds))
                            await asyncio.sleep(retry_after)
                            retries += 1
                            continue
                        else:
                            raise Exception("Rate limited by API")

                    response.raise_for_status()

                    # Parse response
                    content_type = response.headers.get("Content-Type", "")
                    if "application/json" in content_type:
                        return await response.json()
                    else:
                        text = await response.text()
                        return {"data": text}

            except aiohttp.ClientError as e:
                last_error = e
                retries += 1
                if retries <= self.rest_config.max_retries:
                    await asyncio.sleep(self.rest_config.retry_delay * retries)

        raise last_error or Exception("Request failed after retries")

    async def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers based on config."""
        auth = self.rest_config.auth
        headers = {}

        if auth.auth_type == AuthType.API_KEY:
            if auth.api_key_location == ApiKeyLocation.HEADER:
                headers[auth.api_key_name] = auth.api_key

        elif auth.auth_type == AuthType.BASIC:
            credentials = b64encode(
                f"{auth.username}:{auth.password}".encode()
            ).decode()
            headers["Authorization"] = f"Basic {credentials}"

        elif auth.auth_type == AuthType.BEARER:
            headers["Authorization"] = f"Bearer {auth.bearer_token}"

        elif auth.auth_type == AuthType.OAUTH2:
            token = await self._get_oauth2_token()
            headers["Authorization"] = f"Bearer {token}"

        elif auth.auth_type == AuthType.HMAC:
            # HMAC signature is typically computed per-request
            pass

        # Add custom headers
        headers.update(auth.custom_headers)

        return headers

    async def _get_oauth2_token(self) -> str:
        """Get or refresh OAuth2 access token."""
        # Check if current token is still valid
        if self._oauth2_token and self._oauth2_expires:
            if datetime.utcnow() < self._oauth2_expires:
                return self._oauth2_token

        auth = self.rest_config.auth

        # Request new token
        async with aiohttp.ClientSession() as session:
            data = {
                "grant_type": auth.oauth2_grant_type,
                "client_id": auth.oauth2_client_id,
                "client_secret": auth.oauth2_client_secret
            }
            if auth.oauth2_scope:
                data["scope"] = auth.oauth2_scope

            async with session.post(
                auth.oauth2_token_url,
                data=data
            ) as response:
                response.raise_for_status()
                token_data = await response.json()

                self._oauth2_token = token_data["access_token"]
                expires_in = token_data.get("expires_in", 3600)
                self._oauth2_expires = datetime.utcnow() + \
                    timedelta(seconds=expires_in - 60)  # Refresh 1 min early

                return self._oauth2_token

    def _extract_records(self, response: Dict[str, Any]) -> List[DataRecord]:
        """Extract records from API response."""
        data_path = self.rest_config.pagination.data_path
        data = self._get_nested_value(response, data_path)

        if data is None:
            data = response if isinstance(response, list) else [response]
        elif not isinstance(data, list):
            data = [data]

        records = []
        for item in data:
            record_id = str(item.get(self.rest_config.id_field, ""))
            if not record_id:
                record_id = hashlib.md5(json.dumps(item, sort_keys=True).encode()).hexdigest()[:16]

            timestamp = None
            if self.rest_config.timestamp_field:
                ts_value = item.get(self.rest_config.timestamp_field)
                if ts_value:
                    try:
                        timestamp = datetime.fromisoformat(ts_value.replace("Z", "+00:00"))
                    except (ValueError, AttributeError):
                        pass

            records.append(DataRecord(
                id=record_id,
                data=item,
                timestamp=timestamp,
                metadata={"source": self.rest_config.name}
            ))

        return records

    def _extract_total_count(self, response: Dict[str, Any], default: int) -> int:
        """Extract total count from response."""
        count_path = self.rest_config.pagination.total_count_path
        count = self._get_nested_value(response, count_path)

        if count is not None:
            try:
                return int(count)
            except (ValueError, TypeError):
                pass

        return default

    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Any:
        """Get nested value from dict using dot notation path."""
        if not path:
            return data

        keys = path.split(".")
        value = data

        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            elif isinstance(value, list) and key.isdigit():
                idx = int(key)
                value = value[idx] if idx < len(value) else None
            else:
                return None

            if value is None:
                return None

        return value

    def _get_method_for_operation(
        self,
        operation: OperationType,
        mode: str
    ) -> HttpMethod:
        """Get HTTP method for data operation."""
        if operation == OperationType.INSERT:
            return HttpMethod.POST
        elif operation == OperationType.UPDATE:
            return HttpMethod.PUT
        elif operation == OperationType.DELETE:
            return HttpMethod.DELETE
        elif operation == OperationType.UPSERT:
            if mode == "insert":
                return HttpMethod.POST
            else:
                return HttpMethod.PUT
        return HttpMethod.POST


class RateLimiter:
    """Token bucket rate limiter for API requests."""

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self._tokens = config.burst_size
        self._last_update = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquire a token, waiting if necessary."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_update

            # Add tokens based on elapsed time
            self._tokens = min(
                self.config.burst_size,
                self._tokens + elapsed * self.config.requests_per_second
            )
            self._last_update = now

            if self._tokens < 1:
                # Wait for token to become available
                wait_time = (1 - self._tokens) / self.config.requests_per_second
                await asyncio.sleep(wait_time)
                self._tokens = 0
            else:
                self._tokens -= 1


# Additional imports needed
from datetime import timedelta

# Register connector
from ..base import ConnectorFactory
ConnectorFactory.register("rest", RESTConnector)
