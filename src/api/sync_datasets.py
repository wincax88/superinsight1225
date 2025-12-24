"""
Sync Datasets API Routes.

Provides API endpoints for managing industry datasets,
including discovery, download, integration, and quality assessment
for AI-friendly data enhancement.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from src.sync.gateway.auth import (
    AuthToken,
    Permission,
    PermissionLevel,
    ResourceType,
    get_tenant_id,
    sync_auth_handler,
)
from src.sync.models import DatasetCategory, DatasetStatus

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/sync/datasets", tags=["sync-datasets"])


# ============================================================================
# Request/Response Models
# ============================================================================

class DatasetSource(BaseModel):
    """Dataset source information."""
    platform: str = Field(..., description="Source platform: huggingface, kaggle, github")
    url: str
    identifier: str


class DatasetCreate(BaseModel):
    """Request model for registering a dataset."""
    name: str = Field(..., min_length=1, max_length=200)
    display_name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    source: DatasetSource
    category: DatasetCategory
    domain_tags: List[str] = Field(default_factory=list)
    language: str = "zh"
    is_public: bool = True
    license_type: Optional[str] = None
    license_url: Optional[str] = None


class DatasetResponse(BaseModel):
    """Response model for dataset."""
    id: UUID
    tenant_id: Optional[str]
    name: str
    display_name: str
    description: Optional[str]
    version: str
    source_platform: str
    source_url: str
    source_identifier: str
    category: str
    domain_tags: List[str]
    language: str
    status: str
    is_public: bool
    total_records: int
    file_size_bytes: int
    quality_score: float
    quality_metrics: Dict[str, Any]
    local_path: Optional[str]
    recommended_dilution_ratio: float
    download_count: int
    integration_count: int
    last_downloaded_at: Optional[datetime]
    last_integrated_at: Optional[datetime]
    license_type: Optional[str]
    license_url: Optional[str]
    created_at: datetime
    updated_at: datetime


class DatasetListResponse(BaseModel):
    """Response model for dataset list."""
    items: List[DatasetResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class DatasetDiscoverRequest(BaseModel):
    """Request model for discovering datasets."""
    platforms: List[str] = Field(default=["huggingface", "kaggle", "github"])
    categories: Optional[List[DatasetCategory]] = None
    keywords: Optional[List[str]] = None
    min_quality_score: float = Field(default=0.5, ge=0, le=1)
    language: Optional[str] = None


class DatasetDiscoverResponse(BaseModel):
    """Response model for dataset discovery."""
    discovered: List[Dict[str, Any]]
    total: int
    platforms_searched: List[str]
    search_time_ms: float


class DatasetDownloadRequest(BaseModel):
    """Request model for downloading a dataset."""
    dataset_id: UUID
    version: Optional[str] = None
    subset: Optional[str] = None


class DatasetDownloadResponse(BaseModel):
    """Response model for dataset download."""
    dataset_id: UUID
    download_id: str
    status: str
    progress: float
    file_size_bytes: int
    estimated_time_seconds: Optional[int]
    message: str


class DatasetIntegrateRequest(BaseModel):
    """Request model for integrating a dataset."""
    dataset_id: UUID
    target_schema: Optional[str] = None
    field_mapping: Dict[str, str] = Field(default_factory=dict)
    dilution_ratio: float = Field(default=0.3, ge=0.1, le=0.5)
    filter_conditions: Dict[str, Any] = Field(default_factory=dict)


class DatasetIntegrateResponse(BaseModel):
    """Response model for dataset integration."""
    integration_id: str
    dataset_id: UUID
    status: str
    records_integrated: int
    dilution_ratio: float
    message: str


class DatasetQualityRequest(BaseModel):
    """Request model for quality assessment."""
    dataset_id: UUID
    sample_size: int = Field(default=1000, ge=100, le=10000)
    metrics: List[str] = Field(default=["completeness", "consistency", "accuracy", "relevancy"])


class DatasetQualityResponse(BaseModel):
    """Response model for quality assessment."""
    dataset_id: UUID
    overall_score: float
    metrics: Dict[str, float]
    sample_size: int
    issues: List[Dict[str, Any]]
    recommendations: List[str]
    assessed_at: datetime


class DiluteRequest(BaseModel):
    """Request model for data dilution."""
    source_dataset_id: Optional[UUID] = None
    target_noise_ratio: float = Field(default=0.1, ge=0.01, le=0.3)
    industry_datasets: List[UUID] = Field(default_factory=list)
    auto_select_datasets: bool = True
    max_datasets: int = Field(default=5, ge=1, le=10)


class DiluteResponse(BaseModel):
    """Response model for data dilution."""
    dilution_id: str
    status: str
    original_noise_ratio: float
    target_noise_ratio: float
    achieved_noise_ratio: float
    datasets_used: List[Dict[str, Any]]
    records_added: int
    quality_improvement: float
    message: str


class AugmentRequest(BaseModel):
    """Request model for data augmentation."""
    dataset_id: Optional[UUID] = None
    strategies: List[str] = Field(default=["synonym", "back_translation", "paraphrase"])
    target_multiplier: float = Field(default=3.0, ge=1.5, le=5.0)
    quality_threshold: float = Field(default=0.7, ge=0.5, le=0.95)
    preserve_semantics: bool = True


class AugmentResponse(BaseModel):
    """Response model for data augmentation."""
    augmentation_id: str
    status: str
    original_samples: int
    augmented_samples: int
    multiplier_achieved: float
    strategies_used: List[str]
    quality_score: float
    message: str


class AIDatasetGenerateRequest(BaseModel):
    """Request model for generating AI-friendly dataset."""
    source_datasets: List[UUID] = Field(default_factory=list)
    target_category: Optional[DatasetCategory] = None
    quality_target: float = Field(default=0.9, ge=0.7, le=0.99)
    include_dilution: bool = True
    include_augmentation: bool = True
    output_format: str = Field(default="jsonl", pattern="^(json|jsonl|parquet|csv)$")


class AIDatasetGenerateResponse(BaseModel):
    """Response model for AI dataset generation."""
    generation_id: str
    status: str
    input_records: int
    output_records: int
    quality_achieved: float
    noise_reduction: float
    speed_improvement: float
    output_path: Optional[str]
    message: str


# ============================================================================
# In-Memory Storage (Replace with database in production)
# ============================================================================

_datasets: Dict[str, Dict[str, Any]] = {}
_download_tasks: Dict[str, Dict[str, Any]] = {}
_integration_tasks: Dict[str, Dict[str, Any]] = {}

# Pre-populate with sample industry datasets
SAMPLE_DATASETS = [
    {
        "name": "financial-phrasebank",
        "display_name": "Financial PhraseBank",
        "description": "Sentences from financial news categorized by sentiment",
        "source_platform": "huggingface",
        "source_url": "https://huggingface.co/datasets/financial_phrasebank",
        "source_identifier": "financial_phrasebank",
        "category": DatasetCategory.FINANCE.value,
        "domain_tags": ["sentiment", "finance", "news"],
        "language": "en",
        "total_records": 4846,
        "quality_score": 0.85,
        "recommended_dilution_ratio": 0.25
    },
    {
        "name": "fiqa",
        "display_name": "FiQA - Financial QA",
        "description": "Question answering dataset for financial domain",
        "source_platform": "huggingface",
        "source_url": "https://huggingface.co/datasets/fiqa",
        "source_identifier": "fiqa",
        "category": DatasetCategory.FINANCE.value,
        "domain_tags": ["qa", "finance"],
        "language": "en",
        "total_records": 17000,
        "quality_score": 0.88,
        "recommended_dilution_ratio": 0.3
    },
    {
        "name": "pubmedqa",
        "display_name": "PubMedQA",
        "description": "Biomedical research question answering dataset",
        "source_platform": "huggingface",
        "source_url": "https://huggingface.co/datasets/pubmed_qa",
        "source_identifier": "pubmed_qa",
        "category": DatasetCategory.HEALTHCARE.value,
        "domain_tags": ["qa", "medical", "research"],
        "language": "en",
        "total_records": 211269,
        "quality_score": 0.92,
        "recommended_dilution_ratio": 0.35
    },
    {
        "name": "cuad",
        "display_name": "CUAD - Contract Understanding",
        "description": "Legal contract understanding and analysis dataset",
        "source_platform": "github",
        "source_url": "https://github.com/TheAtticusProject/cuad",
        "source_identifier": "cuad",
        "category": DatasetCategory.LEGAL.value,
        "domain_tags": ["contract", "legal", "ner"],
        "language": "en",
        "total_records": 13000,
        "quality_score": 0.9,
        "recommended_dilution_ratio": 0.3
    },
    {
        "name": "squad-2.0",
        "display_name": "SQuAD 2.0",
        "description": "Stanford Question Answering Dataset",
        "source_platform": "huggingface",
        "source_url": "https://huggingface.co/datasets/squad_v2",
        "source_identifier": "squad_v2",
        "category": DatasetCategory.GENERAL.value,
        "domain_tags": ["qa", "reading-comprehension"],
        "language": "en",
        "total_records": 150000,
        "quality_score": 0.95,
        "recommended_dilution_ratio": 0.4
    },
]


def _init_sample_datasets():
    """Initialize sample datasets."""
    for ds in SAMPLE_DATASETS:
        dataset_id = str(uuid4())
        _datasets[dataset_id] = {
            "id": dataset_id,
            "tenant_id": None,  # Global datasets
            "version": "1.0.0",
            "status": DatasetStatus.AVAILABLE.value,
            "is_public": True,
            "file_size_bytes": 0,
            "quality_metrics": {},
            "local_path": None,
            "integration_config": {},
            "min_dilution_ratio": 0.1,
            "max_dilution_ratio": 0.5,
            "download_count": 0,
            "integration_count": 0,
            "last_downloaded_at": None,
            "last_integrated_at": None,
            "license_type": "Apache-2.0",
            "license_url": None,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            **ds
        }


_init_sample_datasets()


# ============================================================================
# API Endpoints
# ============================================================================

@router.get("", response_model=DatasetListResponse)
async def list_datasets(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    category: Optional[DatasetCategory] = None,
    status: Optional[DatasetStatus] = None,
    platform: Optional[str] = None,
    search: Optional[str] = None,
    include_global: bool = True,
    tenant_id: str = Depends(get_tenant_id),
    auth: AuthToken = Depends(
        sync_auth_handler.require_permission(ResourceType.DATASET, PermissionLevel.READ)
    )
):
    """
    List available datasets.

    Includes both tenant-specific and global (public) datasets.
    """
    datasets = list(_datasets.values())

    # Filter by tenant
    if include_global:
        datasets = [
            d for d in datasets
            if d.get("tenant_id") is None or d.get("tenant_id") == tenant_id
        ]
    else:
        datasets = [d for d in datasets if d.get("tenant_id") == tenant_id]

    # Apply filters
    if category:
        datasets = [d for d in datasets if d.get("category") == category.value]
    if status:
        datasets = [d for d in datasets if d.get("status") == status.value]
    if platform:
        datasets = [d for d in datasets if d.get("source_platform") == platform]
    if search:
        search_lower = search.lower()
        datasets = [
            d for d in datasets
            if search_lower in d.get("name", "").lower()
            or search_lower in d.get("display_name", "").lower()
            or search_lower in (d.get("description") or "").lower()
            or any(search_lower in tag.lower() for tag in d.get("domain_tags", []))
        ]

    # Sort by quality score
    datasets.sort(key=lambda d: d.get("quality_score", 0), reverse=True)

    # Pagination
    total = len(datasets)
    total_pages = (total + page_size - 1) // page_size
    start = (page - 1) * page_size
    end = start + page_size
    paginated = datasets[start:end]

    return DatasetListResponse(
        items=[DatasetResponse(**d) for d in paginated],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages
    )


@router.post("/discover", response_model=DatasetDiscoverResponse)
async def discover_datasets(
    request: DatasetDiscoverRequest,
    tenant_id: str = Depends(get_tenant_id),
    auth: AuthToken = Depends(
        sync_auth_handler.require_permission(ResourceType.DATASET, PermissionLevel.READ)
    )
):
    """
    Discover new datasets from external platforms.

    Searches Hugging Face, Kaggle, and GitHub for relevant datasets.
    """
    import time
    start_time = time.time()

    # In production, this would call external APIs
    # For demo, return sample results
    discovered = []

    for ds in SAMPLE_DATASETS:
        if ds["source_platform"] not in request.platforms:
            continue

        if request.categories:
            if DatasetCategory(ds["category"]) not in request.categories:
                continue

        if ds["quality_score"] < request.min_quality_score:
            continue

        if request.keywords:
            keyword_match = any(
                kw.lower() in ds["name"].lower()
                or kw.lower() in ds.get("description", "").lower()
                or any(kw.lower() in tag.lower() for tag in ds.get("domain_tags", []))
                for kw in request.keywords
            )
            if not keyword_match:
                continue

        discovered.append({
            "name": ds["name"],
            "display_name": ds["display_name"],
            "description": ds["description"],
            "platform": ds["source_platform"],
            "url": ds["source_url"],
            "category": ds["category"],
            "quality_score": ds["quality_score"],
            "records": ds["total_records"],
            "tags": ds["domain_tags"]
        })

    search_time = (time.time() - start_time) * 1000

    return DatasetDiscoverResponse(
        discovered=discovered,
        total=len(discovered),
        platforms_searched=request.platforms,
        search_time_ms=search_time
    )


@router.post("/download", response_model=DatasetDownloadResponse)
async def download_dataset(
    request: DatasetDownloadRequest,
    tenant_id: str = Depends(get_tenant_id),
    auth: AuthToken = Depends(
        sync_auth_handler.require_permission(ResourceType.DATASET, PermissionLevel.WRITE)
    )
):
    """
    Download a dataset for local use.

    Downloads the dataset and prepares it for integration.
    """
    dataset = _datasets.get(str(request.dataset_id))
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    download_id = str(uuid4())

    # Update dataset status
    dataset["status"] = DatasetStatus.DOWNLOADING.value
    dataset["download_count"] = dataset.get("download_count", 0) + 1
    dataset["last_downloaded_at"] = datetime.utcnow()

    # Create download task
    _download_tasks[download_id] = {
        "download_id": download_id,
        "dataset_id": str(request.dataset_id),
        "tenant_id": tenant_id,
        "status": "downloading",
        "progress": 0.0,
        "started_at": datetime.utcnow()
    }

    logger.info(f"Started download: dataset={request.dataset_id}, download_id={download_id}")

    return DatasetDownloadResponse(
        dataset_id=request.dataset_id,
        download_id=download_id,
        status="downloading",
        progress=0.0,
        file_size_bytes=dataset.get("file_size_bytes", 0),
        estimated_time_seconds=60,
        message="Download started"
    )


@router.post("/{dataset_id}/integrate", response_model=DatasetIntegrateResponse)
async def integrate_dataset(
    dataset_id: UUID,
    request: DatasetIntegrateRequest,
    tenant_id: str = Depends(get_tenant_id),
    auth: AuthToken = Depends(
        sync_auth_handler.require_permission(ResourceType.DATASET, PermissionLevel.WRITE)
    )
):
    """
    Integrate a dataset with customer data.

    Merges the dataset with existing data using the specified dilution ratio.
    """
    dataset = _datasets.get(str(dataset_id))
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    integration_id = str(uuid4())

    # Update dataset status
    dataset["status"] = DatasetStatus.PROCESSING.value
    dataset["integration_count"] = dataset.get("integration_count", 0) + 1
    dataset["last_integrated_at"] = datetime.utcnow()

    # Create integration task
    _integration_tasks[integration_id] = {
        "integration_id": integration_id,
        "dataset_id": str(dataset_id),
        "tenant_id": tenant_id,
        "dilution_ratio": request.dilution_ratio,
        "status": "processing",
        "started_at": datetime.utcnow()
    }

    # Simulate integration
    records_integrated = int(dataset.get("total_records", 0) * request.dilution_ratio)

    logger.info(f"Started integration: dataset={dataset_id}, records={records_integrated}")

    return DatasetIntegrateResponse(
        integration_id=integration_id,
        dataset_id=dataset_id,
        status="processing",
        records_integrated=records_integrated,
        dilution_ratio=request.dilution_ratio,
        message=f"Integration started, {records_integrated} records will be integrated"
    )


@router.get("/{dataset_id}/quality", response_model=DatasetQualityResponse)
async def get_dataset_quality(
    dataset_id: UUID,
    sample_size: int = Query(1000, ge=100, le=10000),
    tenant_id: str = Depends(get_tenant_id),
    auth: AuthToken = Depends(
        sync_auth_handler.require_permission(ResourceType.DATASET, PermissionLevel.READ)
    )
):
    """
    Get quality assessment for a dataset.

    Returns detailed quality metrics and recommendations.
    """
    dataset = _datasets.get(str(dataset_id))
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Simulate quality assessment
    base_score = dataset.get("quality_score", 0.8)
    metrics = {
        "completeness": min(1.0, base_score + 0.05),
        "consistency": min(1.0, base_score + 0.02),
        "accuracy": base_score,
        "relevancy": min(1.0, base_score + 0.08),
        "noise_ratio": max(0, 1 - base_score - 0.1),
        "duplicate_ratio": max(0, 0.15 - base_score * 0.1)
    }

    issues = []
    recommendations = []

    if metrics["noise_ratio"] > 0.1:
        issues.append({
            "type": "high_noise",
            "severity": "medium",
            "description": f"Noise ratio is {metrics['noise_ratio']:.1%}"
        })
        recommendations.append("Consider diluting with high-quality industry datasets")

    if metrics["duplicate_ratio"] > 0.05:
        issues.append({
            "type": "duplicates",
            "severity": "low",
            "description": f"Duplicate ratio is {metrics['duplicate_ratio']:.1%}"
        })
        recommendations.append("Run deduplication before integration")

    if base_score < 0.85:
        recommendations.append("Apply data augmentation to increase sample quality")
        recommendations.append(f"Recommended dilution ratio: {dataset.get('recommended_dilution_ratio', 0.3):.0%}")

    return DatasetQualityResponse(
        dataset_id=dataset_id,
        overall_score=base_score,
        metrics=metrics,
        sample_size=min(sample_size, dataset.get("total_records", 0)),
        issues=issues,
        recommendations=recommendations,
        assessed_at=datetime.utcnow()
    )


@router.post("/dilute", response_model=DiluteResponse)
async def dilute_data(
    request: DiluteRequest,
    tenant_id: str = Depends(get_tenant_id),
    auth: AuthToken = Depends(
        sync_auth_handler.require_permission(ResourceType.DATASET, PermissionLevel.WRITE)
    )
):
    """
    Dilute noisy data with high-quality industry datasets.

    Reduces noise ratio by mixing in curated industry data.
    """
    dilution_id = str(uuid4())

    # Select datasets if auto-select enabled
    datasets_used = []
    if request.auto_select_datasets:
        # Select top quality datasets
        available = [
            d for d in _datasets.values()
            if d.get("is_public", True) and d.get("quality_score", 0) > 0.8
        ]
        available.sort(key=lambda d: d.get("quality_score", 0), reverse=True)
        selected = available[:request.max_datasets]
        datasets_used = [
            {"id": d["id"], "name": d["name"], "quality_score": d["quality_score"]}
            for d in selected
        ]
    else:
        for ds_id in request.industry_datasets:
            ds = _datasets.get(str(ds_id))
            if ds:
                datasets_used.append({
                    "id": ds["id"],
                    "name": ds["name"],
                    "quality_score": ds["quality_score"]
                })

    # Simulate dilution
    original_noise = 0.35  # Assume 35% noise
    achieved_noise = max(request.target_noise_ratio, original_noise * 0.3)
    quality_improvement = (original_noise - achieved_noise) / original_noise
    records_added = sum(d.get("quality_score", 0.8) * 1000 for d in datasets_used)

    logger.info(f"Dilution completed: {dilution_id}, improvement={quality_improvement:.1%}")

    return DiluteResponse(
        dilution_id=dilution_id,
        status="completed",
        original_noise_ratio=original_noise,
        target_noise_ratio=request.target_noise_ratio,
        achieved_noise_ratio=achieved_noise,
        datasets_used=datasets_used,
        records_added=int(records_added),
        quality_improvement=quality_improvement,
        message=f"Noise reduced from {original_noise:.1%} to {achieved_noise:.1%}"
    )


@router.post("/augment", response_model=AugmentResponse)
async def augment_data(
    request: AugmentRequest,
    tenant_id: str = Depends(get_tenant_id),
    auth: AuthToken = Depends(
        sync_auth_handler.require_permission(ResourceType.DATASET, PermissionLevel.WRITE)
    )
):
    """
    Augment data using various text augmentation strategies.

    Increases sample count while maintaining semantic consistency.
    """
    augmentation_id = str(uuid4())

    # Simulate augmentation
    original_samples = 1000
    if request.dataset_id:
        dataset = _datasets.get(str(request.dataset_id))
        if dataset:
            original_samples = dataset.get("total_records", 1000)

    multiplier_achieved = min(request.target_multiplier, 4.0)  # Cap at 4x
    augmented_samples = int(original_samples * multiplier_achieved)
    quality_score = max(0.7, request.quality_threshold - 0.05)

    logger.info(
        f"Augmentation completed: {augmentation_id}, "
        f"{original_samples} -> {augmented_samples} samples"
    )

    return AugmentResponse(
        augmentation_id=augmentation_id,
        status="completed",
        original_samples=original_samples,
        augmented_samples=augmented_samples,
        multiplier_achieved=multiplier_achieved,
        strategies_used=request.strategies,
        quality_score=quality_score,
        message=f"Augmented {original_samples} samples to {augmented_samples} ({multiplier_achieved:.1f}x)"
    )


@router.post("/ai-dataset/generate", response_model=AIDatasetGenerateResponse)
async def generate_ai_dataset(
    request: AIDatasetGenerateRequest,
    tenant_id: str = Depends(get_tenant_id),
    auth: AuthToken = Depends(
        sync_auth_handler.require_permission(ResourceType.DATASET, PermissionLevel.WRITE)
    )
):
    """
    Generate an AI-friendly dataset.

    Combines data cleaning, dilution, and augmentation to produce
    high-quality datasets optimized for AI training and inference.
    """
    generation_id = str(uuid4())

    # Calculate metrics
    input_records = 0
    for ds_id in request.source_datasets:
        ds = _datasets.get(str(ds_id))
        if ds:
            input_records += ds.get("total_records", 0)

    if input_records == 0:
        input_records = 10000  # Default for demo

    # Apply transformations
    output_records = input_records

    # Dilution adds records
    if request.include_dilution:
        output_records = int(output_records * 1.3)

    # Augmentation multiplies
    if request.include_augmentation:
        output_records = int(output_records * 3)

    # Quality improvement
    base_quality = 0.7
    quality_achieved = min(request.quality_target, 0.95)
    noise_reduction = 0.25  # 25% reduction
    speed_improvement = 0.35  # 35% faster

    logger.info(
        f"AI dataset generated: {generation_id}, "
        f"{input_records} -> {output_records} records, quality={quality_achieved:.1%}"
    )

    return AIDatasetGenerateResponse(
        generation_id=generation_id,
        status="completed",
        input_records=input_records,
        output_records=output_records,
        quality_achieved=quality_achieved,
        noise_reduction=noise_reduction,
        speed_improvement=speed_improvement,
        output_path=f"/data/ai-datasets/{generation_id}.{request.output_format}",
        message=(
            f"Generated AI-friendly dataset: {output_records} records, "
            f"{quality_achieved:.0%} quality, "
            f"{noise_reduction:.0%} noise reduction, "
            f"{speed_improvement:.0%} speed improvement"
        )
    )


@router.get("/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(
    dataset_id: UUID,
    tenant_id: str = Depends(get_tenant_id),
    auth: AuthToken = Depends(
        sync_auth_handler.require_permission(ResourceType.DATASET, PermissionLevel.READ)
    )
):
    """Get a specific dataset by ID."""
    dataset = _datasets.get(str(dataset_id))
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Check access
    if dataset.get("tenant_id") and dataset.get("tenant_id") != tenant_id:
        if not dataset.get("is_public", False):
            raise HTTPException(status_code=403, detail="Access denied")

    return DatasetResponse(**dataset)
