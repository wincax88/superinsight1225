"""
Caching Service for AI Predictions in SuperInsight platform.

Provides intelligent caching for AI prediction results to improve performance.
"""

import json
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from uuid import UUID
import redis
from dataclasses import dataclass
from enum import Enum

from .base import Prediction, ModelConfig
try:
    from models.task import Task
except ImportError:
    from src.models.task import Task


class CacheStrategy(str, Enum):
    """Enumeration of caching strategies."""
    LRU = "lru"  # Least Recently Used
    TTL = "ttl"  # Time To Live
    HYBRID = "hybrid"  # Combination of LRU and TTL


@dataclass
class CacheConfig:
    """Configuration for prediction caching."""
    
    enabled: bool = True
    strategy: CacheStrategy = CacheStrategy.HYBRID
    ttl_hours: int = 24
    max_entries: int = 10000
    key_prefix: str = "superinsight:ai_cache:"
    compression_enabled: bool = True
    cache_hit_threshold: float = 0.8  # Minimum confidence to cache


class PredictionCacheService:
    """Service for caching AI prediction results."""
    
    def __init__(self, redis_client: redis.Redis, config: CacheConfig = None):
        """
        Initialize cache service.
        
        Args:
            redis_client: Redis client for caching
            config: Cache configuration
        """
        self.redis_client = redis_client
        self.config = config or CacheConfig()
        
        # Statistics tracking
        self.stats = {
            "hits": 0,
            "misses": 0,
            "stores": 0,
            "evictions": 0
        }
    
    def _generate_cache_key(
        self,
        task: Task,
        model_config: ModelConfig,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate cache key for a task and model configuration.
        
        Args:
            task: The annotation task
            model_config: Model configuration used
            additional_context: Additional context for key generation
            
        Returns:
            Cache key string
        """
        # Create a deterministic key based on task content and model config
        key_components = {
            "task_id": str(task.id),
            "document_id": str(task.document_id),
            "project_id": task.project_id,
            "model_type": model_config.model_type.value,
            "model_name": model_config.model_name,
            "temperature": model_config.temperature,
            "max_tokens": model_config.max_tokens
        }
        
        if additional_context:
            key_components.update(additional_context)
        
        # Create hash of the key components
        key_string = json.dumps(key_components, sort_keys=True)
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()[:16]
        
        return f"{self.config.key_prefix}pred:{key_hash}"
    
    def _generate_content_key(self, content: str, model_config: ModelConfig) -> str:
        """
        Generate cache key based on content hash and model config.
        
        Args:
            content: Text content to hash
            model_config: Model configuration
            
        Returns:
            Cache key string
        """
        # Hash the content for consistent caching across similar texts
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        model_hash = hashlib.sha256(
            f"{model_config.model_type.value}:{model_config.model_name}".encode()
        ).hexdigest()[:8]
        
        return f"{self.config.key_prefix}content:{content_hash}:{model_hash}"
    
    async def get_cached_prediction(
        self,
        task: Task,
        model_config: ModelConfig,
        content: Optional[str] = None
    ) -> Optional[Prediction]:
        """
        Get cached prediction for a task.
        
        Args:
            task: The annotation task
            model_config: Model configuration
            content: Optional content for content-based caching
            
        Returns:
            Cached prediction or None if not found
        """
        if not self.config.enabled:
            return None
        
        try:
            # Try task-specific cache first
            task_key = self._generate_cache_key(task, model_config)
            cached_data = await self.redis_client.get(task_key)
            
            if cached_data:
                prediction = self._deserialize_prediction(cached_data)
                if prediction:
                    await self._update_access_time(task_key)
                    self.stats["hits"] += 1
                    return prediction
            
            # Try content-based cache if content is provided
            if content:
                content_key = self._generate_content_key(content, model_config)
                cached_data = await self.redis_client.get(content_key)
                
                if cached_data:
                    prediction = self._deserialize_prediction(cached_data)
                    if prediction:
                        await self._update_access_time(content_key)
                        self.stats["hits"] += 1
                        return prediction
            
            self.stats["misses"] += 1
            return None
            
        except Exception as e:
            print(f"Cache get error: {e}")
            self.stats["misses"] += 1
            return None
    
    async def cache_prediction(
        self,
        prediction: Prediction,
        task: Task,
        content: Optional[str] = None
    ) -> bool:
        """
        Cache a prediction result.
        
        Args:
            prediction: The prediction to cache
            task: The associated task
            content: Optional content for content-based caching
            
        Returns:
            True if cached successfully, False otherwise
        """
        if not self.config.enabled:
            return False
        
        # Only cache high-confidence predictions
        if prediction.confidence < self.config.cache_hit_threshold:
            return False
        
        try:
            serialized_data = self._serialize_prediction(prediction)
            ttl_seconds = self.config.ttl_hours * 3600
            
            # Cache by task
            task_key = self._generate_cache_key(task, prediction.model_config)
            await self.redis_client.setex(task_key, ttl_seconds, serialized_data)
            
            # Cache by content if provided
            if content:
                content_key = self._generate_content_key(content, prediction.model_config)
                await self.redis_client.setex(content_key, ttl_seconds, serialized_data)
            
            # Update access time for LRU
            await self._update_access_time(task_key)
            
            # Check and enforce cache size limits
            await self._enforce_cache_limits()
            
            self.stats["stores"] += 1
            return True
            
        except Exception as e:
            print(f"Cache store error: {e}")
            return False
    
    def _serialize_prediction(self, prediction: Prediction) -> str:
        """Serialize prediction for caching."""
        data = prediction.to_dict()
        
        # Add cache metadata
        data["_cached_at"] = datetime.now().isoformat()
        data["_cache_version"] = "1.0"
        
        serialized = json.dumps(data)
        
        # Compress if enabled
        if self.config.compression_enabled:
            import gzip
            serialized = gzip.compress(serialized.encode()).decode('latin1')
        
        return serialized
    
    def _deserialize_prediction(self, cached_data: str) -> Optional[Prediction]:
        """Deserialize cached prediction."""
        try:
            # Decompress if needed
            if self.config.compression_enabled:
                import gzip
                cached_data = gzip.decompress(cached_data.encode('latin1')).decode()
            
            data = json.loads(cached_data)
            
            # Remove cache metadata
            data.pop("_cached_at", None)
            data.pop("_cache_version", None)
            
            # Reconstruct prediction (simplified)
            # In a real implementation, you'd properly reconstruct the Prediction object
            return Prediction(
                id=UUID(data["id"]),
                task_id=UUID(data["task_id"]),
                ai_model_config=ModelConfig(**data["model_config"]),
                prediction_data=data["prediction_data"],
                confidence=data["confidence"],
                processing_time=data["processing_time"],
                created_at=datetime.fromisoformat(data["created_at"])
            )
            
        except Exception as e:
            print(f"Cache deserialize error: {e}")
            return None
    
    async def _update_access_time(self, cache_key: str) -> None:
        """Update access time for LRU tracking."""
        if self.config.strategy in [CacheStrategy.LRU, CacheStrategy.HYBRID]:
            access_key = f"{cache_key}:access"
            await self.redis_client.set(access_key, int(time.time()))
            await self.redis_client.expire(access_key, self.config.ttl_hours * 3600)
    
    async def _enforce_cache_limits(self) -> None:
        """Enforce cache size limits based on strategy."""
        if self.config.strategy == CacheStrategy.TTL:
            return  # TTL handles expiration automatically
        
        try:
            # Get all cache keys
            pattern = f"{self.config.key_prefix}pred:*"
            cache_keys = await self.redis_client.keys(pattern)
            
            if len(cache_keys) <= self.config.max_entries:
                return
            
            # Evict based on strategy
            if self.config.strategy == CacheStrategy.LRU:
                await self._evict_lru_entries(cache_keys)
            elif self.config.strategy == CacheStrategy.HYBRID:
                await self._evict_hybrid_entries(cache_keys)
                
        except Exception as e:
            print(f"Cache limit enforcement error: {e}")
    
    async def _evict_lru_entries(self, cache_keys: List[str]) -> None:
        """Evict least recently used entries."""
        # Get access times for all keys
        access_times = {}
        for key in cache_keys:
            access_key = f"{key}:access"
            access_time = await self.redis_client.get(access_key)
            access_times[key] = int(access_time) if access_time else 0
        
        # Sort by access time and evict oldest
        sorted_keys = sorted(access_times.items(), key=lambda x: x[1])
        evict_count = len(cache_keys) - self.config.max_entries
        
        for key, _ in sorted_keys[:evict_count]:
            await self.redis_client.delete(key)
            await self.redis_client.delete(f"{key}:access")
            self.stats["evictions"] += 1
    
    async def _evict_hybrid_entries(self, cache_keys: List[str]) -> None:
        """Evict entries using hybrid strategy (TTL + LRU)."""
        # First, let TTL handle natural expiration
        # Then apply LRU for remaining entries
        await self._evict_lru_entries(cache_keys)
    
    async def invalidate_cache(
        self,
        task_id: Optional[UUID] = None,
        model_type: Optional[str] = None,
        pattern: Optional[str] = None
    ) -> int:
        """
        Invalidate cache entries based on criteria.
        
        Args:
            task_id: Specific task ID to invalidate
            model_type: Model type to invalidate
            pattern: Custom pattern to match
            
        Returns:
            Number of entries invalidated
        """
        try:
            if pattern:
                keys = await self.redis_client.keys(pattern)
            elif task_id:
                # Find keys containing the task ID
                pattern = f"{self.config.key_prefix}*{task_id}*"
                keys = await self.redis_client.keys(pattern)
            elif model_type:
                # Find keys for specific model type
                pattern = f"{self.config.key_prefix}*{model_type}*"
                keys = await self.redis_client.keys(pattern)
            else:
                # Invalidate all cache
                pattern = f"{self.config.key_prefix}*"
                keys = await self.redis_client.keys(pattern)
            
            if keys:
                await self.redis_client.delete(*keys)
                # Also delete access time keys
                access_keys = [f"{key}:access" for key in keys]
                existing_access_keys = []
                for access_key in access_keys:
                    if await self.redis_client.exists(access_key):
                        existing_access_keys.append(access_key)
                if existing_access_keys:
                    await self.redis_client.delete(*existing_access_keys)
            
            return len(keys)
            
        except Exception as e:
            print(f"Cache invalidation error: {e}")
            return 0
    
    async def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = (self.stats["hits"] / total_requests * 100) if total_requests > 0 else 0
        
        # Get cache size
        pattern = f"{self.config.key_prefix}*"
        cache_keys = await self.redis_client.keys(pattern)
        cache_size = len(cache_keys)
        
        return {
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "stores": self.stats["stores"],
            "evictions": self.stats["evictions"],
            "hit_rate_percent": round(hit_rate, 2),
            "cache_size": cache_size,
            "max_entries": self.config.max_entries,
            "ttl_hours": self.config.ttl_hours,
            "strategy": self.config.strategy.value,
            "enabled": self.config.enabled
        }
    
    async def clear_all_cache(self) -> int:
        """Clear all cached predictions."""
        return await self.invalidate_cache()
    
    async def optimize_cache(self) -> Dict[str, Any]:
        """Optimize cache by removing expired and low-value entries."""
        optimization_stats = {
            "expired_removed": 0,
            "low_confidence_removed": 0,
            "total_removed": 0
        }
        
        try:
            pattern = f"{self.config.key_prefix}pred:*"
            cache_keys = await self.redis_client.keys(pattern)
            
            for key in cache_keys:
                cached_data = await self.redis_client.get(key)
                if not cached_data:
                    continue
                
                prediction = self._deserialize_prediction(cached_data)
                if not prediction:
                    await self.redis_client.delete(key)
                    optimization_stats["expired_removed"] += 1
                    continue
                
                # Remove low confidence predictions
                if prediction.confidence < self.config.cache_hit_threshold:
                    await self.redis_client.delete(key)
                    optimization_stats["low_confidence_removed"] += 1
            
            optimization_stats["total_removed"] = (
                optimization_stats["expired_removed"] + 
                optimization_stats["low_confidence_removed"]
            )
            
        except Exception as e:
            print(f"Cache optimization error: {e}")
        
        return optimization_stats