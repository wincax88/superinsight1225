"""
Dataset Discovery Service for AI-Friendly Data Integration.

This module provides capabilities for discovering, downloading, and integrating
high-quality industry datasets from various sources like HuggingFace, Kaggle, and GitHub.
"""

import asyncio
import hashlib
import json
import logging
import os
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class DatasetSource(str, Enum):
    """Supported dataset sources."""

    HUGGINGFACE = "huggingface"
    KAGGLE = "kaggle"
    GITHUB = "github"
    LOCAL = "local"
    CUSTOM_URL = "custom_url"


class DatasetDomain(str, Enum):
    """Industry domains for datasets."""

    FINANCE = "finance"
    HEALTHCARE = "healthcare"
    LEGAL = "legal"
    GENERAL = "general"
    CUSTOMER_SERVICE = "customer_service"
    TECHNICAL = "technical"
    EDUCATION = "education"
    ECOMMERCE = "ecommerce"


class DatasetFormat(str, Enum):
    """Dataset file formats."""

    JSON = "json"
    JSONL = "jsonl"
    CSV = "csv"
    PARQUET = "parquet"
    TSV = "tsv"
    TXT = "txt"
    ARROW = "arrow"


class DatasetStatus(str, Enum):
    """Dataset lifecycle status."""

    DISCOVERED = "discovered"
    DOWNLOADING = "downloading"
    DOWNLOADED = "downloaded"
    PROCESSING = "processing"
    READY = "ready"
    INTEGRATED = "integrated"
    FAILED = "failed"
    DEPRECATED = "deprecated"


@dataclass
class DatasetQualityMetrics:
    """Quality metrics for a dataset."""

    completeness: float = 0.0  # 0-1: percentage of non-null values
    consistency: float = 0.0  # 0-1: format consistency score
    accuracy: float = 0.0  # 0-1: estimated accuracy
    relevance: float = 0.0  # 0-1: domain relevance score
    freshness: float = 0.0  # 0-1: data recency score
    diversity: float = 0.0  # 0-1: sample diversity score

    @property
    def overall_score(self) -> float:
        weights = {
            "completeness": 0.15,
            "consistency": 0.15,
            "accuracy": 0.25,
            "relevance": 0.25,
            "freshness": 0.10,
            "diversity": 0.10,
        }
        return sum(
            getattr(self, metric) * weight for metric, weight in weights.items()
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "completeness": round(self.completeness, 3),
            "consistency": round(self.consistency, 3),
            "accuracy": round(self.accuracy, 3),
            "relevance": round(self.relevance, 3),
            "freshness": round(self.freshness, 3),
            "diversity": round(self.diversity, 3),
            "overall_score": round(self.overall_score, 3),
        }


class DatasetMetadata(BaseModel):
    """Metadata for a discovered dataset."""

    id: str
    name: str
    description: str = ""
    source: DatasetSource
    source_url: str = ""
    domain: DatasetDomain = DatasetDomain.GENERAL
    format: DatasetFormat = DatasetFormat.JSONL
    languages: List[str] = Field(default_factory=lambda: ["en"])
    size_bytes: int = 0
    sample_count: int = 0
    version: str = "1.0.0"
    license: str = ""
    tags: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    quality_metrics: Optional[Dict[str, float]] = None
    schema_info: Dict[str, Any] = Field(default_factory=dict)
    download_url: str = ""
    local_path: str = ""
    status: DatasetStatus = DatasetStatus.DISCOVERED

    class Config:
        use_enum_values = True


class DatasetSearchQuery(BaseModel):
    """Query parameters for dataset search."""

    keywords: List[str] = Field(default_factory=list)
    domains: List[DatasetDomain] = Field(default_factory=list)
    sources: List[DatasetSource] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=list)
    min_samples: int = 0
    max_samples: int = 0
    min_quality_score: float = 0.0
    formats: List[DatasetFormat] = Field(default_factory=list)
    include_deprecated: bool = False
    limit: int = 50
    offset: int = 0


class DatasetConnector(ABC):
    """Abstract base class for dataset source connectors."""

    @abstractmethod
    async def search(self, query: DatasetSearchQuery) -> List[DatasetMetadata]:
        """Search for datasets matching the query."""
        pass

    @abstractmethod
    async def download(self, dataset_id: str, target_path: Path) -> bool:
        """Download a dataset to the target path."""
        pass

    @abstractmethod
    async def get_metadata(self, dataset_id: str) -> Optional[DatasetMetadata]:
        """Get detailed metadata for a dataset."""
        pass


class HuggingFaceConnector(DatasetConnector):
    """Connector for HuggingFace datasets."""

    # Popular datasets for different domains
    CURATED_DATASETS = {
        DatasetDomain.FINANCE: [
            {
                "id": "financial_phrasebank",
                "name": "Financial PhraseBank",
                "description": "Sentiment analysis dataset for financial news",
                "sample_count": 4840,
                "tags": ["sentiment", "financial", "news"],
            },
            {
                "id": "fiqa",
                "name": "FiQA",
                "description": "Financial opinion mining and question answering",
                "sample_count": 17000,
                "tags": ["qa", "financial", "opinion"],
            },
            {
                "id": "convfinqa",
                "name": "ConvFinQA",
                "description": "Conversational financial QA over tables",
                "sample_count": 14115,
                "tags": ["qa", "financial", "tables", "conversational"],
            },
        ],
        DatasetDomain.HEALTHCARE: [
            {
                "id": "pubmed_qa",
                "name": "PubMedQA",
                "description": "Biomedical question answering from PubMed abstracts",
                "sample_count": 211000,
                "tags": ["qa", "biomedical", "pubmed"],
            },
            {
                "id": "medmcqa",
                "name": "MedMCQA",
                "description": "Medical multiple choice question answering",
                "sample_count": 194000,
                "tags": ["qa", "medical", "multiple-choice"],
            },
        ],
        DatasetDomain.LEGAL: [
            {
                "id": "cuad",
                "name": "CUAD",
                "description": "Contract Understanding Atticus Dataset",
                "sample_count": 510,
                "tags": ["legal", "contracts", "extraction"],
            },
            {
                "id": "ledgar",
                "name": "LEDGAR",
                "description": "Legal document clause classification",
                "sample_count": 60000,
                "tags": ["legal", "classification", "contracts"],
            },
        ],
        DatasetDomain.GENERAL: [
            {
                "id": "squad_v2",
                "name": "SQuAD 2.0",
                "description": "Stanford Question Answering Dataset",
                "sample_count": 130000,
                "tags": ["qa", "reading-comprehension"],
            },
            {
                "id": "natural_questions",
                "name": "Natural Questions",
                "description": "Google Natural Questions for QA",
                "sample_count": 307000,
                "tags": ["qa", "wikipedia"],
            },
            {
                "id": "ms_marco",
                "name": "MS MARCO",
                "description": "Microsoft Machine Reading Comprehension",
                "sample_count": 1010000,
                "tags": ["qa", "passage-ranking"],
            },
        ],
    }

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path("./data/datasets/huggingface")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    async def search(self, query: DatasetSearchQuery) -> List[DatasetMetadata]:
        """Search HuggingFace datasets."""
        results = []

        domains = query.domains or list(DatasetDomain)

        for domain in domains:
            if domain not in self.CURATED_DATASETS:
                continue

            for ds_info in self.CURATED_DATASETS[domain]:
                # Apply filters
                if query.min_samples and ds_info["sample_count"] < query.min_samples:
                    continue
                if query.max_samples and ds_info["sample_count"] > query.max_samples:
                    continue

                # Keyword matching
                if query.keywords:
                    text = f"{ds_info['name']} {ds_info['description']} {' '.join(ds_info['tags'])}"
                    if not any(kw.lower() in text.lower() for kw in query.keywords):
                        continue

                metadata = DatasetMetadata(
                    id=f"huggingface/{ds_info['id']}",
                    name=ds_info["name"],
                    description=ds_info["description"],
                    source=DatasetSource.HUGGINGFACE,
                    source_url=f"https://huggingface.co/datasets/{ds_info['id']}",
                    domain=domain,
                    format=DatasetFormat.JSONL,
                    sample_count=ds_info["sample_count"],
                    tags=ds_info["tags"],
                    license="Various",
                )
                results.append(metadata)

        return results[query.offset : query.offset + query.limit]

    async def download(self, dataset_id: str, target_path: Path) -> bool:
        """Download a HuggingFace dataset."""
        try:
            # In production, use datasets library
            # from datasets import load_dataset
            # dataset = load_dataset(dataset_id.replace("huggingface/", ""))
            # dataset.save_to_disk(str(target_path))

            # For now, create a placeholder
            target_path.mkdir(parents=True, exist_ok=True)
            (target_path / "metadata.json").write_text(
                json.dumps({"id": dataset_id, "downloaded_at": datetime.utcnow().isoformat()})
            )
            logger.info(f"Downloaded dataset {dataset_id} to {target_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to download {dataset_id}: {e}")
            return False

    async def get_metadata(self, dataset_id: str) -> Optional[DatasetMetadata]:
        """Get metadata for a specific dataset."""
        clean_id = dataset_id.replace("huggingface/", "")

        for domain, datasets in self.CURATED_DATASETS.items():
            for ds_info in datasets:
                if ds_info["id"] == clean_id:
                    return DatasetMetadata(
                        id=dataset_id,
                        name=ds_info["name"],
                        description=ds_info["description"],
                        source=DatasetSource.HUGGINGFACE,
                        source_url=f"https://huggingface.co/datasets/{clean_id}",
                        domain=domain,
                        sample_count=ds_info["sample_count"],
                        tags=ds_info["tags"],
                    )
        return None


class LocalConnector(DatasetConnector):
    """Connector for local datasets."""

    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)

    async def search(self, query: DatasetSearchQuery) -> List[DatasetMetadata]:
        """Search local datasets."""
        results = []

        for dataset_dir in self.base_path.iterdir():
            if not dataset_dir.is_dir():
                continue

            metadata_file = dataset_dir / "metadata.json"
            if not metadata_file.exists():
                continue

            try:
                with open(metadata_file) as f:
                    metadata_dict = json.load(f)
                metadata = DatasetMetadata(**metadata_dict)
                metadata.local_path = str(dataset_dir)
                results.append(metadata)
            except Exception as e:
                logger.warning(f"Failed to load metadata from {metadata_file}: {e}")

        return results[query.offset : query.offset + query.limit]

    async def download(self, dataset_id: str, target_path: Path) -> bool:
        """Copy a local dataset."""
        source_path = self.base_path / dataset_id.replace("local/", "")
        if source_path.exists():
            shutil.copytree(source_path, target_path, dirs_exist_ok=True)
            return True
        return False

    async def get_metadata(self, dataset_id: str) -> Optional[DatasetMetadata]:
        """Get metadata for a local dataset."""
        clean_id = dataset_id.replace("local/", "")
        metadata_file = self.base_path / clean_id / "metadata.json"

        if metadata_file.exists():
            with open(metadata_file) as f:
                return DatasetMetadata(**json.load(f))
        return None


class DatasetRegistry:
    """Registry for managing discovered and integrated datasets."""

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("./data/dataset_registry")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._datasets: Dict[str, DatasetMetadata] = {}
        self._load_registry()

    def _load_registry(self) -> None:
        """Load registry from disk."""
        registry_file = self.storage_path / "registry.json"
        if registry_file.exists():
            try:
                with open(registry_file) as f:
                    data = json.load(f)
                for ds_dict in data.get("datasets", []):
                    ds = DatasetMetadata(**ds_dict)
                    self._datasets[ds.id] = ds
                logger.info(f"Loaded {len(self._datasets)} datasets from registry")
            except Exception as e:
                logger.error(f"Failed to load registry: {e}")

    def _save_registry(self) -> None:
        """Save registry to disk."""
        registry_file = self.storage_path / "registry.json"
        data = {
            "datasets": [ds.dict() for ds in self._datasets.values()],
            "updated_at": datetime.utcnow().isoformat(),
        }
        with open(registry_file, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def register(self, dataset: DatasetMetadata) -> None:
        """Register a dataset."""
        self._datasets[dataset.id] = dataset
        self._save_registry()

    def unregister(self, dataset_id: str) -> bool:
        """Unregister a dataset."""
        if dataset_id in self._datasets:
            del self._datasets[dataset_id]
            self._save_registry()
            return True
        return False

    def get(self, dataset_id: str) -> Optional[DatasetMetadata]:
        """Get a dataset by ID."""
        return self._datasets.get(dataset_id)

    def list(
        self,
        domain: Optional[DatasetDomain] = None,
        status: Optional[DatasetStatus] = None,
        source: Optional[DatasetSource] = None,
    ) -> List[DatasetMetadata]:
        """List datasets with optional filters."""
        results = list(self._datasets.values())

        if domain:
            results = [ds for ds in results if ds.domain == domain.value]
        if status:
            results = [ds for ds in results if ds.status == status.value]
        if source:
            results = [ds for ds in results if ds.source == source.value]

        return results

    def update_status(self, dataset_id: str, status: DatasetStatus) -> bool:
        """Update dataset status."""
        if dataset_id in self._datasets:
            self._datasets[dataset_id].status = status
            self._datasets[dataset_id].updated_at = datetime.utcnow()
            self._save_registry()
            return True
        return False

    def update_quality_metrics(
        self, dataset_id: str, metrics: DatasetQualityMetrics
    ) -> bool:
        """Update quality metrics for a dataset."""
        if dataset_id in self._datasets:
            self._datasets[dataset_id].quality_metrics = metrics.to_dict()
            self._datasets[dataset_id].updated_at = datetime.utcnow()
            self._save_registry()
            return True
        return False


class DatasetDiscoveryService:
    """Service for discovering and managing industry datasets."""

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        download_path: Optional[Path] = None,
    ):
        self.storage_path = storage_path or Path("./data/datasets")
        self.download_path = download_path or Path("./data/downloads")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.download_path.mkdir(parents=True, exist_ok=True)

        self.registry = DatasetRegistry(self.storage_path / "registry")
        self.connectors: Dict[DatasetSource, DatasetConnector] = {
            DatasetSource.HUGGINGFACE: HuggingFaceConnector(
                self.download_path / "huggingface"
            ),
            DatasetSource.LOCAL: LocalConnector(self.storage_path / "local"),
        }

    async def discover(
        self,
        query: Optional[DatasetSearchQuery] = None,
        sources: Optional[List[DatasetSource]] = None,
    ) -> List[DatasetMetadata]:
        """Discover datasets from configured sources."""
        query = query or DatasetSearchQuery()
        sources = sources or [DatasetSource.HUGGINGFACE]

        all_results = []

        for source in sources:
            connector = self.connectors.get(source)
            if connector:
                try:
                    results = await connector.search(query)
                    all_results.extend(results)
                    logger.info(f"Discovered {len(results)} datasets from {source.value}")
                except Exception as e:
                    logger.error(f"Failed to search {source.value}: {e}")

        # Register discovered datasets
        for ds in all_results:
            if not self.registry.get(ds.id):
                self.registry.register(ds)

        return all_results

    async def download_dataset(
        self, dataset_id: str, force: bool = False
    ) -> Optional[Path]:
        """Download a dataset by ID."""
        dataset = self.registry.get(dataset_id)
        if not dataset:
            logger.error(f"Dataset {dataset_id} not found in registry")
            return None

        target_path = self.download_path / dataset_id.replace("/", "_")

        if target_path.exists() and not force:
            logger.info(f"Dataset {dataset_id} already downloaded")
            return target_path

        # Update status
        self.registry.update_status(dataset_id, DatasetStatus.DOWNLOADING)

        # Get appropriate connector
        source = DatasetSource(dataset.source)
        connector = self.connectors.get(source)

        if not connector:
            logger.error(f"No connector for source {source.value}")
            self.registry.update_status(dataset_id, DatasetStatus.FAILED)
            return None

        try:
            success = await connector.download(dataset_id, target_path)
            if success:
                self.registry.update_status(dataset_id, DatasetStatus.DOWNLOADED)
                dataset.local_path = str(target_path)
                self.registry.register(dataset)
                return target_path
            else:
                self.registry.update_status(dataset_id, DatasetStatus.FAILED)
                return None
        except Exception as e:
            logger.error(f"Failed to download {dataset_id}: {e}")
            self.registry.update_status(dataset_id, DatasetStatus.FAILED)
            return None

    async def evaluate_quality(
        self, dataset_id: str
    ) -> Optional[DatasetQualityMetrics]:
        """Evaluate quality metrics for a dataset."""
        dataset = self.registry.get(dataset_id)
        if not dataset or not dataset.local_path:
            logger.error(f"Dataset {dataset_id} not found or not downloaded")
            return None

        # Simulate quality evaluation
        # In production, this would analyze the actual data
        metrics = DatasetQualityMetrics(
            completeness=0.95,
            consistency=0.92,
            accuracy=0.88,
            relevance=0.90,
            freshness=0.85,
            diversity=0.80,
        )

        self.registry.update_quality_metrics(dataset_id, metrics)
        return metrics

    async def recommend_datasets(
        self,
        domain: DatasetDomain,
        existing_data_quality: float = 0.7,
        min_improvement: float = 0.1,
        max_results: int = 5,
    ) -> List[DatasetMetadata]:
        """Recommend datasets for improving data quality."""
        # Get datasets for the domain
        available = self.registry.list(domain=domain, status=DatasetStatus.READY)

        if not available:
            # Discover new datasets
            query = DatasetSearchQuery(domains=[domain], limit=20)
            available = await self.discover(query)

        # Filter by quality improvement potential
        recommendations = []
        for ds in available:
            if ds.quality_metrics:
                quality = ds.quality_metrics.get("overall_score", 0)
                if quality > existing_data_quality + min_improvement:
                    recommendations.append(ds)

        # Sort by quality score
        recommendations.sort(
            key=lambda x: x.quality_metrics.get("overall_score", 0) if x.quality_metrics else 0,
            reverse=True,
        )

        return recommendations[:max_results]

    def get_dataset_info(self, dataset_id: str) -> Optional[DatasetMetadata]:
        """Get information about a dataset."""
        return self.registry.get(dataset_id)

    def list_datasets(
        self,
        domain: Optional[DatasetDomain] = None,
        status: Optional[DatasetStatus] = None,
    ) -> List[DatasetMetadata]:
        """List all registered datasets."""
        return self.registry.list(domain=domain, status=status)


# Global instance
_discovery_service: Optional[DatasetDiscoveryService] = None


def get_discovery_service() -> DatasetDiscoveryService:
    """Get or create global discovery service."""
    global _discovery_service
    if _discovery_service is None:
        _discovery_service = DatasetDiscoveryService()
    return _discovery_service
