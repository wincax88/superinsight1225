"""
Export data models for SuperInsight Platform.

Defines data structures for export operations.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
from uuid import UUID, uuid4
from enum import Enum
from pydantic import BaseModel, Field, field_validator


class ExportFormat(str, Enum):
    """Supported export formats."""
    JSON = "json"
    CSV = "csv"
    COCO = "coco"


class ExportRequest(BaseModel):
    """Request model for data export."""
    
    project_id: Optional[str] = Field(None, description="Filter by project ID")
    task_ids: Optional[List[str]] = Field(None, description="Specific task IDs to export")
    document_ids: Optional[List[str]] = Field(None, description="Specific document IDs to export")
    format: ExportFormat = Field(..., description="Export format")
    include_annotations: bool = Field(True, description="Include annotation data")
    include_ai_predictions: bool = Field(False, description="Include AI predictions")
    include_metadata: bool = Field(True, description="Include document metadata")
    batch_size: int = Field(1000, ge=1, le=10000, description="Batch size for large exports")
    date_from: Optional[datetime] = Field(None, description="Filter from date")
    date_to: Optional[datetime] = Field(None, description="Filter to date")
    
    @field_validator('task_ids', 'document_ids')
    @classmethod
    def validate_ids(cls, v):
        """Validate that IDs are valid UUIDs if provided."""
        if v is not None:
            for id_str in v:
                try:
                    UUID(id_str)
                except ValueError:
                    raise ValueError(f'Invalid UUID format: {id_str}')
        return v


class ExportResult(BaseModel):
    """Result model for export operations."""
    
    export_id: str = Field(..., description="Unique export identifier")
    status: str = Field(..., description="Export status")
    format: ExportFormat = Field(..., description="Export format")
    total_records: int = Field(..., description="Total number of records")
    exported_records: int = Field(0, description="Number of exported records")
    file_path: Optional[str] = Field(None, description="Path to exported file")
    file_size: Optional[int] = Field(None, description="File size in bytes")
    error: Optional[str] = Field(None, description="Error message if failed")
    created_at: datetime = Field(default_factory=datetime.now, description="Export creation time")
    completed_at: Optional[datetime] = Field(None, description="Export completion time")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ExportedDocument(BaseModel):
    """Model for exported document data."""
    
    id: str = Field(..., description="Document ID")
    source_type: str = Field(..., description="Source type")
    content: str = Field(..., description="Document content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    created_at: str = Field(..., description="Creation timestamp")
    tasks: List[Dict[str, Any]] = Field(default_factory=list, description="Associated tasks")


class COCOAnnotation(BaseModel):
    """COCO format annotation model."""
    
    id: int = Field(..., description="Annotation ID")
    image_id: int = Field(..., description="Image/document ID")
    category_id: int = Field(..., description="Category ID")
    bbox: Optional[List[float]] = Field(None, description="Bounding box [x, y, width, height]")
    area: Optional[float] = Field(None, description="Annotation area")
    iscrowd: int = Field(0, description="Is crowd annotation")
    segmentation: Optional[List[List[float]]] = Field(None, description="Segmentation polygons")


class COCOCategory(BaseModel):
    """COCO format category model."""
    
    id: int = Field(..., description="Category ID")
    name: str = Field(..., description="Category name")
    supercategory: str = Field("", description="Super category name")


class COCOImage(BaseModel):
    """COCO format image/document model."""
    
    id: int = Field(..., description="Image/document ID")
    width: int = Field(800, description="Image width")
    height: int = Field(600, description="Image height")
    file_name: str = Field(..., description="File name")
    license: int = Field(1, description="License ID")
    flickr_url: str = Field("", description="Source URL")
    coco_url: str = Field("", description="COCO URL")
    date_captured: str = Field(..., description="Capture date")


class COCODataset(BaseModel):
    """COCO format dataset model."""
    
    info: Dict[str, Any] = Field(default_factory=dict, description="Dataset info")
    licenses: List[Dict[str, Any]] = Field(default_factory=list, description="Licenses")
    images: List[COCOImage] = Field(default_factory=list, description="Images/documents")
    annotations: List[COCOAnnotation] = Field(default_factory=list, description="Annotations")
    categories: List[COCOCategory] = Field(default_factory=list, description="Categories")