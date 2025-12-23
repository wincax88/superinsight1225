"""
Database cache service for SuperInsight Platform.

Provides Redis-like caching functionality with fallback to in-memory cache.
"""

import json
import logging
import pickle
from typing import Any, Optional, Dict, List
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CacheStats:
    """Cache statistics."""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0


class CacheService:
    """Cache service with TTL support and statistics."""
    
    def __init__(self, max_size: int = 10000, default_ttl: int = 3600):
        """Initialize cache service."""
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.stats = CacheStats()
        
        # Access tracking for LRU eviction
        self.access_times: Dict[str, datetime] = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key in self.cache:
            entry = self.cache[key]
            
            # Check if expired
            if datetime.now() < entry['expires_at']:
                # Update access time
                self.access_times[key] = datetime.now()
                self.stats.hits += 1
                
                # Deserialize value
                return self._deserialize(entry['value'], entry['serialization'])
            else:
                # Remove expired entry
                self._remove_key(key)
        
        self.stats.misses += 1
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with TTL."""
        try:
            # Check if we need to evict entries
            if len(self.cache) >= self.max_size:
                self._evict_lru()
            
            ttl = ttl or self.default_ttl
            expires_at = datetime.now() + timedelta(seconds=ttl)
            
            # Serialize value
            serialized_value, serialization_type = self._serialize(value)
            
            self.cache[key] = {
                'value': serialized_value,
                'serialization': serialization_type,
                'expires_at': expires_at,
                'created_at': datetime.now()
            }
            
            self.access_times[key] = datetime.now()
            self.stats.sets += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to set cache key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if key in self.cache:
            self._remove_key(key)
            self.stats.deletes += 1
            return True
        return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        if key in self.cache:
            entry = self.cache[key]
            if datetime.now() < entry['expires_at']:
                return True
            else:
                self._remove_key(key)
        return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        self.access_times.clear()
        logger.info("Cache cleared")
    
    def cleanup_expired(self) -> int:
        """Remove expired entries and return count."""
        now = datetime.now()
        expired_keys = [
            key for key, entry in self.cache.items()
            if now >= entry['expires_at']
        ]
        
        for key in expired_keys:
            self._remove_key(key)
        
        logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
        return len(expired_keys)
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self.stats
    
    def get_info(self) -> Dict[str, Any]:
        """Get cache information."""
        now = datetime.now()
        active_entries = sum(
            1 for entry in self.cache.values()
            if now < entry['expires_at']
        )
        
        return {
            'total_entries': len(self.cache),
            'active_entries': active_entries,
            'expired_entries': len(self.cache) - active_entries,
            'max_size': self.max_size,
            'memory_usage_estimate': self._estimate_memory_usage(),
            'stats': {
                'hits': self.stats.hits,
                'misses': self.stats.misses,
                'sets': self.stats.sets,
                'deletes': self.stats.deletes,
                'evictions': self.stats.evictions,
                'hit_rate': self.stats.hit_rate
            }
        }
    
    def _serialize(self, value: Any) -> tuple[Any, str]:
        """Serialize value for storage."""
        if isinstance(value, (str, int, float, bool, type(None))):
            return value, 'primitive'
        elif isinstance(value, (list, dict)):
            return json.dumps(value), 'json'
        else:
            # Use pickle for complex objects
            return pickle.dumps(value), 'pickle'
    
    def _deserialize(self, value: Any, serialization_type: str) -> Any:
        """Deserialize value from storage."""
        if serialization_type == 'primitive':
            return value
        elif serialization_type == 'json':
            return json.loads(value)
        elif serialization_type == 'pickle':
            return pickle.loads(value)
        else:
            return value
    
    def _remove_key(self, key: str) -> None:
        """Remove key from cache and access times."""
        if key in self.cache:
            del self.cache[key]
        if key in self.access_times:
            del self.access_times[key]
    
    def _evict_lru(self) -> None:
        """Evict least recently used entries."""
        if not self.access_times:
            return
        
        # Find oldest accessed key
        oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        self._remove_key(oldest_key)
        self.stats.evictions += 1
        
        logger.debug(f"Evicted LRU cache entry: {oldest_key}")
    
    def _estimate_memory_usage(self) -> str:
        """Estimate memory usage of cache."""
        try:
            total_size = 0
            for entry in self.cache.values():
                if isinstance(entry['value'], str):
                    total_size += len(entry['value'].encode('utf-8'))
                elif isinstance(entry['value'], bytes):
                    total_size += len(entry['value'])
                else:
                    total_size += 100  # Rough estimate for other types
            
            # Convert to human readable format
            if total_size < 1024:
                return f"{total_size} B"
            elif total_size < 1024 * 1024:
                return f"{total_size / 1024:.1f} KB"
            else:
                return f"{total_size / (1024 * 1024):.1f} MB"
                
        except Exception:
            return "unknown"


class QueryResultCache:
    """Specialized cache for database query results."""
    
    def __init__(self, cache_service: CacheService):
        """Initialize query result cache."""
        self.cache = cache_service
        self.query_prefix = "query:"
    
    def get_query_result(self, query_hash: str) -> Optional[Any]:
        """Get cached query result."""
        key = f"{self.query_prefix}{query_hash}"
        return self.cache.get(key)
    
    def set_query_result(self, query_hash: str, result: Any, ttl: Optional[int] = None) -> bool:
        """Cache query result."""
        key = f"{self.query_prefix}{query_hash}"
        return self.cache.set(key, result, ttl)
    
    def invalidate_query(self, query_hash: str) -> bool:
        """Invalidate cached query result."""
        key = f"{self.query_prefix}{query_hash}"
        return self.cache.delete(key)
    
    def invalidate_by_pattern(self, pattern: str) -> int:
        """Invalidate cached queries matching pattern."""
        count = 0
        keys_to_delete = []
        
        for key in self.cache.cache.keys():
            if key.startswith(self.query_prefix) and pattern in key:
                keys_to_delete.append(key)
        
        for key in keys_to_delete:
            if self.cache.delete(key):
                count += 1
        
        return count
    
    def get_query_stats(self) -> Dict[str, Any]:
        """Get query cache statistics."""
        query_keys = [
            key for key in self.cache.cache.keys()
            if key.startswith(self.query_prefix)
        ]
        
        return {
            'total_cached_queries': len(query_keys),
            'cache_stats': self.cache.get_stats()
        }


class CacheInvalidationManager:
    """Manages cache invalidation based on data changes."""
    
    def __init__(self, cache_service: CacheService):
        """Initialize cache invalidation manager."""
        self.cache = cache_service
        
        # Define invalidation rules
        self.invalidation_rules = {
            'documents': [
                'query:documents:*',
                'query:project:*',
                'query:system:performance'
            ],
            'tasks': [
                'query:tasks:*',
                'query:project:*',
                'query:quality:*',
                'query:system:performance'
            ],
            'billing_records': [
                'query:billing:*',
                'query:tenant:*',
                'query:user:*',
                'query:system:performance'
            ],
            'quality_issues': [
                'query:quality:*',
                'query:tasks:*',
                'query:project:*',
                'query:system:performance'
            ]
        }
    
    def invalidate_for_table(self, table_name: str) -> int:
        """Invalidate cache entries affected by table changes."""
        if table_name not in self.invalidation_rules:
            return 0
        
        total_invalidated = 0
        patterns = self.invalidation_rules[table_name]
        
        for pattern in patterns:
            # Remove wildcard and use as prefix
            prefix = pattern.replace('*', '')
            count = self._invalidate_by_prefix(prefix)
            total_invalidated += count
        
        logger.info(f"Invalidated {total_invalidated} cache entries for table {table_name}")
        return total_invalidated
    
    def _invalidate_by_prefix(self, prefix: str) -> int:
        """Invalidate cache entries with given prefix."""
        count = 0
        keys_to_delete = []
        
        for key in self.cache.cache.keys():
            if key.startswith(prefix):
                keys_to_delete.append(key)
        
        for key in keys_to_delete:
            if self.cache.delete(key):
                count += 1
        
        return count


# Global cache instances
cache_service = CacheService(max_size=10000, default_ttl=3600)
query_cache = QueryResultCache(cache_service)
cache_invalidation = CacheInvalidationManager(cache_service)


def get_cache_service() -> CacheService:
    """Get global cache service instance."""
    return cache_service


def get_query_cache() -> QueryResultCache:
    """Get global query cache instance."""
    return query_cache


def get_cache_invalidation() -> CacheInvalidationManager:
    """Get global cache invalidation manager."""
    return cache_invalidation