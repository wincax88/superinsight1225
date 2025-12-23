"""
Model Version Manager for SuperInsight platform.

Manages different versions of AI models and their configurations.
"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set
from uuid import uuid4
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
import redis
from pydantic import BaseModel

from .base import ModelConfig, ModelType, AIAnnotationError


class ModelStatus(str, Enum):
    """Enumeration of model statuses."""
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    EXPERIMENTAL = "experimental"
    DISABLED = "disabled"


@dataclass
class ModelVersion:
    """Represents a specific version of an AI model."""
    
    id: str = field(default_factory=lambda: str(uuid4()))
    model_type: ModelType = ModelType.OLLAMA
    model_name: str = ""
    version: str = "1.0.0"
    config: ModelConfig = field(default_factory=lambda: ModelConfig(
        model_type=ModelType.OLLAMA,
        model_name="default"
    ))
    status: ModelStatus = ModelStatus.EXPERIMENTAL
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "model_type": self.model_type.value,
            "model_name": self.model_name,
            "version": self.version,
            "config": self.config.dict(),
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "performance_metrics": self.performance_metrics,
            "tags": list(self.tags),
            "description": self.description
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelVersion':
        """Create from dictionary."""
        # Convert string dates back to datetime
        if isinstance(data.get('created_at'), str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if isinstance(data.get('updated_at'), str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        
        # Convert config dict to ModelConfig
        if isinstance(data.get('config'), dict):
            data['config'] = ModelConfig(**data['config'])
        
        # Convert enums
        if isinstance(data.get('model_type'), str):
            data['model_type'] = ModelType(data['model_type'])
        if isinstance(data.get('status'), str):
            data['status'] = ModelStatus(data['status'])
        
        # Convert tags list to set
        if isinstance(data.get('tags'), list):
            data['tags'] = set(data['tags'])
        
        return cls(**data)


class ModelVersionManager:
    """Manager for AI model versions and configurations."""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None, storage_path: Optional[str] = None):
        """
        Initialize model version manager.
        
        Args:
            redis_client: Redis client for caching
            storage_path: Path for persistent storage
        """
        self.redis_client = redis_client
        self.storage_path = Path(storage_path) if storage_path else Path("./model_versions")
        self.storage_path.mkdir(exist_ok=True)
        
        # In-memory storage
        self.model_versions: Dict[str, ModelVersion] = {}
        
        # Cache settings
        self.cache_prefix = "superinsight:model_version:"
        self.cache_ttl = 3600  # 1 hour
        
        # Load existing versions
        self._load_versions()
    
    def _load_versions(self) -> None:
        """Load model versions from persistent storage."""
        try:
            versions_file = self.storage_path / "versions.json"
            if versions_file.exists():
                with open(versions_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                for version_data in data.get('versions', []):
                    version = ModelVersion.from_dict(version_data)
                    self.model_versions[version.id] = version
                    
        except Exception as e:
            print(f"Failed to load model versions: {e}")
    
    def _save_versions(self) -> None:
        """Save model versions to persistent storage."""
        try:
            versions_file = self.storage_path / "versions.json"
            data = {
                "versions": [v.to_dict() for v in self.model_versions.values()],
                "updated_at": datetime.now().isoformat()
            }
            
            with open(versions_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"Failed to save model versions: {e}")
    
    async def register_model_version(
        self,
        model_type: ModelType,
        model_name: str,
        version: str,
        config: ModelConfig,
        description: str = "",
        tags: Optional[Set[str]] = None,
        status: ModelStatus = ModelStatus.EXPERIMENTAL
    ) -> str:
        """
        Register a new model version.
        
        Args:
            model_type: Type of the model
            model_name: Name of the model
            version: Version string
            config: Model configuration
            description: Description of the model version
            tags: Optional tags for categorization
            status: Initial status of the model
            
        Returns:
            ID of the registered model version
        """
        # Check if version already exists
        existing_id = self._find_version_id(model_type, model_name, version)
        if existing_id:
            raise ValueError(f"Model version {model_name}:{version} already exists")
        
        # Create new version
        model_version = ModelVersion(
            model_type=model_type,
            model_name=model_name,
            version=version,
            config=config,
            description=description,
            tags=tags or set(),
            status=status
        )
        
        # Store in memory
        self.model_versions[model_version.id] = model_version
        
        # Persist to storage
        self._save_versions()
        
        # Cache in Redis
        if self.redis_client:
            await self._cache_version(model_version)
        
        return model_version.id
    
    def _find_version_id(self, model_type: ModelType, model_name: str, version: str) -> Optional[str]:
        """Find version ID by model type, name, and version."""
        for version_id, model_version in self.model_versions.items():
            if (model_version.model_type == model_type and
                model_version.model_name == model_name and
                model_version.version == version):
                return version_id
        return None
    
    async def get_model_version(self, version_id: str) -> Optional[ModelVersion]:
        """
        Get model version by ID.
        
        Args:
            version_id: ID of the model version
            
        Returns:
            Model version or None if not found
        """
        # Check memory first
        if version_id in self.model_versions:
            return self.model_versions[version_id]
        
        # Check Redis cache
        if self.redis_client:
            cached_version = await self._get_cached_version(version_id)
            if cached_version:
                self.model_versions[version_id] = cached_version
                return cached_version
        
        return None
    
    async def get_latest_version(
        self,
        model_type: ModelType,
        model_name: str,
        status: Optional[ModelStatus] = None
    ) -> Optional[ModelVersion]:
        """
        Get the latest version of a model.
        
        Args:
            model_type: Type of the model
            model_name: Name of the model
            status: Optional status filter
            
        Returns:
            Latest model version or None if not found
        """
        matching_versions = []
        
        for model_version in self.model_versions.values():
            if (model_version.model_type == model_type and
                model_version.model_name == model_name):
                if status is None or model_version.status == status:
                    matching_versions.append(model_version)
        
        if not matching_versions:
            return None
        
        # Sort by creation date and return the latest
        matching_versions.sort(key=lambda v: v.created_at, reverse=True)
        return matching_versions[0]
    
    async def list_model_versions(
        self,
        model_type: Optional[ModelType] = None,
        model_name: Optional[str] = None,
        status: Optional[ModelStatus] = None,
        tags: Optional[Set[str]] = None
    ) -> List[ModelVersion]:
        """
        List model versions with optional filters.
        
        Args:
            model_type: Optional model type filter
            model_name: Optional model name filter
            status: Optional status filter
            tags: Optional tags filter (must have all specified tags)
            
        Returns:
            List of matching model versions
        """
        results = []
        
        for model_version in self.model_versions.values():
            # Apply filters
            if model_type and model_version.model_type != model_type:
                continue
            if model_name and model_version.model_name != model_name:
                continue
            if status and model_version.status != status:
                continue
            if tags and not tags.issubset(model_version.tags):
                continue
            
            results.append(model_version)
        
        # Sort by creation date (newest first)
        results.sort(key=lambda v: v.created_at, reverse=True)
        return results
    
    async def update_model_version(
        self,
        version_id: str,
        status: Optional[ModelStatus] = None,
        performance_metrics: Optional[Dict[str, float]] = None,
        tags: Optional[Set[str]] = None,
        description: Optional[str] = None
    ) -> bool:
        """
        Update model version metadata.
        
        Args:
            version_id: ID of the model version
            status: New status
            performance_metrics: Performance metrics to update
            tags: New tags
            description: New description
            
        Returns:
            True if updated, False if not found
        """
        model_version = await self.get_model_version(version_id)
        if not model_version:
            return False
        
        # Update fields
        if status is not None:
            model_version.status = status
        if performance_metrics is not None:
            model_version.performance_metrics.update(performance_metrics)
        if tags is not None:
            model_version.tags = tags
        if description is not None:
            model_version.description = description
        
        model_version.updated_at = datetime.now()
        
        # Persist changes
        self._save_versions()
        
        # Update cache
        if self.redis_client:
            await self._cache_version(model_version)
        
        return True
    
    async def delete_model_version(self, version_id: str) -> bool:
        """
        Delete a model version.
        
        Args:
            version_id: ID of the model version to delete
            
        Returns:
            True if deleted, False if not found
        """
        if version_id not in self.model_versions:
            return False
        
        # Remove from memory
        del self.model_versions[version_id]
        
        # Persist changes
        self._save_versions()
        
        # Remove from cache
        if self.redis_client:
            await self._remove_cached_version(version_id)
        
        return True
    
    async def get_active_models(self) -> Dict[ModelType, List[ModelVersion]]:
        """
        Get all active models grouped by type.
        
        Returns:
            Dictionary mapping model types to lists of active versions
        """
        active_models = {}
        
        for model_version in self.model_versions.values():
            if model_version.status == ModelStatus.ACTIVE:
                model_type = model_version.model_type
                if model_type not in active_models:
                    active_models[model_type] = []
                active_models[model_type].append(model_version)
        
        return active_models
    
    async def promote_to_active(self, version_id: str) -> bool:
        """
        Promote a model version to active status.
        
        Args:
            version_id: ID of the model version to promote
            
        Returns:
            True if promoted, False if not found
        """
        model_version = await self.get_model_version(version_id)
        if not model_version:
            return False
        
        # Demote other active versions of the same model
        for other_version in self.model_versions.values():
            if (other_version.model_type == model_version.model_type and
                other_version.model_name == model_version.model_name and
                other_version.status == ModelStatus.ACTIVE):
                other_version.status = ModelStatus.DEPRECATED
                other_version.updated_at = datetime.now()
        
        # Promote this version
        model_version.status = ModelStatus.ACTIVE
        model_version.updated_at = datetime.now()
        
        # Persist changes
        self._save_versions()
        
        # Update cache
        if self.redis_client:
            await self._cache_version(model_version)
        
        return True
    
    async def _cache_version(self, model_version: ModelVersion) -> None:
        """Cache model version in Redis."""
        if not self.redis_client:
            return
        
        try:
            cache_key = f"{self.cache_prefix}{model_version.id}"
            cache_data = json.dumps(model_version.to_dict())
            await self.redis_client.setex(cache_key, self.cache_ttl, cache_data)
        except Exception:
            pass  # Cache error, continue without caching
    
    async def _get_cached_version(self, version_id: str) -> Optional[ModelVersion]:
        """Get cached model version from Redis."""
        if not self.redis_client:
            return None
        
        try:
            cache_key = f"{self.cache_prefix}{version_id}"
            cached_data = await self.redis_client.get(cache_key)
            
            if cached_data:
                version_dict = json.loads(cached_data)
                return ModelVersion.from_dict(version_dict)
        except Exception:
            pass  # Cache miss or error
        
        return None
    
    async def _remove_cached_version(self, version_id: str) -> None:
        """Remove cached model version from Redis."""
        if not self.redis_client:
            return
        
        try:
            cache_key = f"{self.cache_prefix}{version_id}"
            await self.redis_client.delete(cache_key)
        except Exception:
            pass  # Cache error, continue
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about model versions."""
        total_versions = len(self.model_versions)
        
        # Count by status
        status_counts = {}
        for model_version in self.model_versions.values():
            status = model_version.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Count by type
        type_counts = {}
        for model_version in self.model_versions.values():
            model_type = model_version.model_type.value
            type_counts[model_type] = type_counts.get(model_type, 0) + 1
        
        return {
            "total_versions": total_versions,
            "status_counts": status_counts,
            "type_counts": type_counts,
            "storage_path": str(self.storage_path)
        }