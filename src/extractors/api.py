"""
API data extractor for SuperInsight Platform.

Provides secure extraction from REST APIs with authentication support.
"""

import logging
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
import json

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.extractors.base import BaseExtractor, APIConfig, ExtractionResult
from src.models.document import Document

logger = logging.getLogger(__name__)


class APIExtractor(BaseExtractor):
    """API extractor for REST endpoints with security features."""
    
    def __init__(self, config: APIConfig):
        super().__init__(config)
        self.config: APIConfig = config
        self._session = None
    
    def _create_session(self) -> requests.Session:
        """Create HTTP session with retry strategy and security settings."""
        if self._session is not None:
            return self._session
        
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set default headers
        session.headers.update(self.config.headers)
        
        # Add authentication if provided
        if self.config.auth_token:
            session.headers.update({
                'Authorization': f'Bearer {self.config.auth_token}'
            })
        
        # Configure timeouts and SSL
        session.timeout = (self.config.connection_timeout, self.config.read_timeout)
        session.verify = self.config.verify_ssl
        
        self._session = session
        return session
    
    def test_connection(self) -> bool:
        """Test API connectivity."""
        try:
            session = self._create_session()
            
            # Try to access the base URL or a health endpoint
            test_urls = [
                f"{self.config.base_url}/health",
                f"{self.config.base_url}/status",
                f"{self.config.base_url}/ping",
                self.config.base_url
            ]
            
            for url in test_urls:
                try:
                    response = session.head(url)
                    if response.status_code in [200, 404]:  # 404 is OK for test
                        logger.info(f"API connection test successful: {url}")
                        return True
                except requests.exceptions.RequestException:
                    continue
            
            logger.warning("API connection test failed for all endpoints")
            return False
            
        except Exception as e:
            logger.error(f"API connection test failed: {e}")
            return False
    
    def extract_data(self, endpoint: str = "", method: str = "GET", 
                    params: Optional[Dict[str, Any]] = None,
                    data: Optional[Dict[str, Any]] = None,
                    paginate: bool = True, **kwargs) -> ExtractionResult:
        """Extract data from API endpoint."""
        try:
            session = self._create_session()
            url = f"{self.config.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
            
            documents = []
            
            if paginate:
                # Handle pagination
                documents = self._extract_with_pagination(session, url, method, params, data)
            else:
                # Single request
                response_data = self._make_request(session, url, method, params, data)
                if response_data:
                    doc = self._create_document_from_response(response_data, url, endpoint)
                    documents.append(doc)
            
            logger.info(f"Extracted {len(documents)} documents from API endpoint: {endpoint}")
            return ExtractionResult(success=True, documents=documents)
            
        except Exception as e:
            logger.error(f"API extraction failed: {e}")
            return ExtractionResult(success=False, error=str(e))
    
    def _make_request(self, session: requests.Session, url: str, method: str,
                     params: Optional[Dict[str, Any]] = None,
                     data: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Make a single API request."""
        try:
            if method.upper() == "GET":
                response = session.get(url, params=params)
            elif method.upper() == "POST":
                response = session.post(url, json=data, params=params)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            
            # Try to parse JSON response
            try:
                return response.json()
            except json.JSONDecodeError:
                # If not JSON, return text content
                return {"content": response.text, "content_type": response.headers.get("content-type")}
                
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return None
    
    def _extract_with_pagination(self, session: requests.Session, url: str, method: str,
                                params: Optional[Dict[str, Any]] = None,
                                data: Optional[Dict[str, Any]] = None,
                                max_pages: int = 10) -> List[Document]:
        """Extract data with automatic pagination handling."""
        documents = []
        current_page = 1
        params = params or {}
        
        while current_page <= max_pages:
            # Add pagination parameters
            page_params = params.copy()
            page_params.update({
                'page': current_page,
                'limit': 100,  # Default page size
                'per_page': 100
            })
            
            response_data = self._make_request(session, url, method, page_params, data)
            
            if not response_data:
                break
            
            # Handle different pagination response formats
            items = self._extract_items_from_response(response_data)
            
            if not items:
                break
            
            # Create documents from items
            for i, item in enumerate(items):
                doc = self._create_document_from_item(item, url, current_page, i)
                documents.append(doc)
            
            # Check if there are more pages
            if not self._has_more_pages(response_data, len(items)):
                break
            
            current_page += 1
        
        return documents
    
    def _extract_items_from_response(self, response_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract items from API response, handling different formats."""
        # Common pagination response formats
        if isinstance(response_data, list):
            return response_data
        
        # Check common item keys
        for key in ['data', 'items', 'results', 'records']:
            if key in response_data and isinstance(response_data[key], list):
                return response_data[key]
        
        # If no list found, treat entire response as single item
        return [response_data]
    
    def _has_more_pages(self, response_data: Dict[str, Any], items_count: int) -> bool:
        """Determine if there are more pages to fetch."""
        # If we got fewer items than expected, probably no more pages
        if items_count < 100:
            return False
        
        # Check common pagination indicators
        if isinstance(response_data, dict):
            # Check for explicit pagination info
            pagination_keys = ['pagination', 'meta', 'page_info']
            for key in pagination_keys:
                if key in response_data:
                    page_info = response_data[key]
                    if isinstance(page_info, dict):
                        # Check various pagination indicators
                        if 'has_next' in page_info:
                            return page_info['has_next']
                        if 'next_page' in page_info:
                            return page_info['next_page'] is not None
                        if 'total_pages' in page_info and 'current_page' in page_info:
                            return page_info['current_page'] < page_info['total_pages']
        
        # Default: assume more pages if we got a full page
        return items_count >= 100
    
    def _create_document_from_response(self, response_data: Dict[str, Any], 
                                     url: str, endpoint: str) -> Document:
        """Create document from entire API response."""
        return Document(
            source_type="api",
            source_config={
                "base_url": self.config.base_url,
                "endpoint": endpoint,
                "url": url
            },
            content=json.dumps(response_data, ensure_ascii=False, indent=2),
            metadata={
                "endpoint": endpoint,
                "response_type": "full_response",
                "extraction_time": datetime.now().isoformat()
            }
        )
    
    def _create_document_from_item(self, item: Dict[str, Any], url: str, 
                                  page: int, index: int) -> Document:
        """Create document from individual item in paginated response."""
        return Document(
            source_type="api",
            source_config={
                "base_url": self.config.base_url,
                "url": url,
                "page": page,
                "index": index
            },
            content=json.dumps(item, ensure_ascii=False, indent=2),
            metadata={
                "page_number": page,
                "item_index": index,
                "response_type": "paginated_item",
                "extraction_time": datetime.now().isoformat()
            }
        )
    
    def extract_from_multiple_endpoints(self, endpoints: List[str], 
                                      method: str = "GET",
                                      params: Optional[Dict[str, Any]] = None) -> ExtractionResult:
        """Extract data from multiple API endpoints."""
        try:
            all_documents = []
            
            for endpoint in endpoints:
                result = self.extract_data(endpoint, method, params, paginate=False)
                if result.success:
                    all_documents.extend(result.documents)
                else:
                    logger.warning(f"Failed to extract from endpoint {endpoint}: {result.error}")
            
            logger.info(f"Extracted {len(all_documents)} documents from {len(endpoints)} endpoints")
            return ExtractionResult(success=True, documents=all_documents)
            
        except Exception as e:
            logger.error(f"Multi-endpoint extraction failed: {e}")
            return ExtractionResult(success=False, error=str(e))
    
    def close(self) -> None:
        """Close HTTP session."""
        if self._session:
            self._session.close()
            self._session = None
            logger.info("API session closed")


class GraphQLExtractor(APIExtractor):
    """GraphQL API extractor."""
    
    def extract_graphql(self, query: str, variables: Optional[Dict[str, Any]] = None) -> ExtractionResult:
        """Extract data using GraphQL query."""
        try:
            payload = {
                "query": query,
                "variables": variables or {}
            }
            
            return self.extract_data(
                endpoint="graphql",
                method="POST",
                data=payload,
                paginate=False
            )
            
        except Exception as e:
            logger.error(f"GraphQL extraction failed: {e}")
            return ExtractionResult(success=False, error=str(e))


class WebhookExtractor(BaseExtractor):
    """Extractor for webhook-based data collection."""
    
    def __init__(self, webhook_url: str, secret_key: Optional[str] = None):
        config = APIConfig(
            base_url=webhook_url,
            headers={'Content-Type': 'application/json'}
        )
        super().__init__(config)
        self.webhook_url = webhook_url
        self.secret_key = secret_key
        self.received_data = []
    
    def test_connection(self) -> bool:
        """Test webhook endpoint accessibility."""
        try:
            response = requests.get(self.webhook_url, timeout=self.config.connection_timeout)
            return response.status_code in [200, 405]  # 405 Method Not Allowed is OK
        except Exception as e:
            logger.error(f"Webhook connection test failed: {e}")
            return False
    
    def extract_data(self, **kwargs) -> ExtractionResult:
        """Return collected webhook data."""
        try:
            documents = []
            
            for i, data in enumerate(self.received_data):
                document = Document(
                    source_type="api",
                    source_config={
                        "webhook_url": self.webhook_url,
                        "data_index": i
                    },
                    content=json.dumps(data, ensure_ascii=False, indent=2),
                    metadata={
                        "webhook_url": self.webhook_url,
                        "received_at": data.get("received_at", datetime.now().isoformat()),
                        "data_type": "webhook_payload"
                    }
                )
                documents.append(document)
            
            logger.info(f"Retrieved {len(documents)} webhook payloads")
            return ExtractionResult(success=True, documents=documents)
            
        except Exception as e:
            logger.error(f"Webhook data extraction failed: {e}")
            return ExtractionResult(success=False, error=str(e))
    
    def receive_webhook_data(self, payload: Dict[str, Any]) -> bool:
        """Receive and store webhook data."""
        try:
            # Add timestamp
            payload["received_at"] = datetime.now().isoformat()
            
            # Verify signature if secret key is provided
            if self.secret_key:
                # Implement webhook signature verification here
                pass
            
            self.received_data.append(payload)
            logger.info("Webhook data received and stored")
            return True
            
        except Exception as e:
            logger.error(f"Failed to receive webhook data: {e}")
            return False
    
    def clear_received_data(self) -> None:
        """Clear stored webhook data."""
        self.received_data.clear()
        logger.info("Webhook data cleared")