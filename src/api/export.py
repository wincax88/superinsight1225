"""
FastAPI endpoints for data export in SuperInsight Platform.

Provides RESTful API for exporting annotation data in multiple formats.
"""

import logging
import os
from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, BackgroundTasks, status, Query
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field

from src.export import ExportService, ExportRequest, ExportResult, ExportFormat

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/export", tags=["Data Export"])

# Global export service instance
export_service = ExportService()


class ExportJobResponse(BaseModel):
    """Response model for export job creation."""
    export_id: str = Field(..., description="Unique export identifier")
    status: str = Field(..., description="Export status")
    message: str = Field(..., description="Status message")
    created_at: datetime = Field(..., description="Job creation time")


class ExportStatusResponse(BaseModel):
    """Response model for export status."""
    export_id: str = Field(..., description="Export identifier")
    status: str = Field(..., description="Export status")
    format: str = Field(..., description="Export format")
    total_records: int = Field(..., description="Total records to export")
    exported_records: int = Field(..., description="Records exported so far")
    progress_percentage: float = Field(..., description="Export progress percentage")
    file_path: Optional[str] = Field(None, description="Path to exported file")
    file_size: Optional[int] = Field(None, description="File size in bytes")
    error: Optional[str] = Field(None, description="Error message if failed")
    created_at: datetime = Field(..., description="Creation time")
    completed_at: Optional[datetime] = Field(None, description="Completion time")


async def perform_export(export_id: str, request: ExportRequest) -> None:
    """Perform export in background task."""
    try:
        logger.info(f"Starting export job {export_id}")
        result = export_service.export_data(export_id, request)
        logger.info(f"Export job {export_id} completed with status: {result.status}")
    except Exception as e:
        logger.error(f"Export job {export_id} failed: {e}")


@router.post("/start", response_model=ExportJobResponse)
async def start_export(
    request: ExportRequest,
    background_tasks: BackgroundTasks
) -> ExportJobResponse:
    """Start a new export job."""
    try:
        # Create export job
        export_id = export_service.start_export(request)
        
        # Start background task
        background_tasks.add_task(perform_export, export_id, request)
        
        return ExportJobResponse(
            export_id=export_id,
            status="pending",
            message=f"Export job started for format {request.format.value}",
            created_at=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Failed to start export: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start export: {str(e)}"
        )


@router.get("/status/{export_id}", response_model=ExportStatusResponse)
async def get_export_status(export_id: str) -> ExportStatusResponse:
    """Get export job status."""
    result = export_service.get_export_status(export_id)
    
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Export job not found"
        )
    
    # Calculate progress percentage
    progress = 0.0
    if result.total_records > 0:
        progress = (result.exported_records / result.total_records) * 100
    
    return ExportStatusResponse(
        export_id=result.export_id,
        status=result.status,
        format=result.format.value,
        total_records=result.total_records,
        exported_records=result.exported_records,
        progress_percentage=progress,
        file_path=result.file_path,
        file_size=result.file_size,
        error=result.error,
        created_at=result.created_at,
        completed_at=result.completed_at
    )


@router.get("/list", response_model=List[ExportStatusResponse])
async def list_exports() -> List[ExportStatusResponse]:
    """List all export jobs."""
    results = export_service.list_exports()
    
    response_list = []
    for result in results:
        progress = 0.0
        if result.total_records > 0:
            progress = (result.exported_records / result.total_records) * 100
        
        response_list.append(ExportStatusResponse(
            export_id=result.export_id,
            status=result.status,
            format=result.format.value,
            total_records=result.total_records,
            exported_records=result.exported_records,
            progress_percentage=progress,
            file_path=result.file_path,
            file_size=result.file_size,
            error=result.error,
            created_at=result.created_at,
            completed_at=result.completed_at
        ))
    
    return response_list


@router.get("/download/{export_id}")
async def download_export(export_id: str) -> FileResponse:
    """Download exported file."""
    result = export_service.get_export_status(export_id)
    
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Export job not found"
        )
    
    if result.status != "completed" or not result.file_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Export not completed or file not available"
        )
    
    if not os.path.exists(result.file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Export file not found"
        )
    
    # Determine media type based on format
    media_type = "application/octet-stream"
    if result.format == ExportFormat.JSON:
        media_type = "application/json"
    elif result.format == ExportFormat.CSV:
        media_type = "text/csv"
    elif result.format == ExportFormat.COCO:
        media_type = "application/json"
    
    filename = f"export_{export_id}.{result.format.value}"
    
    return FileResponse(
        path=result.file_path,
        media_type=media_type,
        filename=filename
    )


@router.delete("/delete/{export_id}")
async def delete_export(export_id: str) -> JSONResponse:
    """Delete export job and associated files."""
    success = export_service.delete_export(export_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Export job not found"
        )
    
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"message": "Export deleted successfully"}
    )


@router.post("/batch", response_model=ExportJobResponse)
async def start_batch_export(
    request: ExportRequest,
    background_tasks: BackgroundTasks
) -> ExportJobResponse:
    """Start a batch export job for large datasets."""
    try:
        # Create export job
        export_id = export_service.start_export(request)
        
        # Start batch export background task
        background_tasks.add_task(perform_batch_export, export_id, request)
        
        return ExportJobResponse(
            export_id=export_id,
            status="pending",
            message=f"Batch export job started for format {request.format.value}",
            created_at=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Failed to start batch export: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start batch export: {str(e)}"
        )


async def perform_batch_export(export_id: str, request: ExportRequest) -> None:
    """Perform batch export in background task."""
    try:
        logger.info(f"Starting batch export job {export_id}")
        
        # Process batches
        for result in export_service.export_batch(export_id, request):
            logger.info(f"Batch export {export_id} progress: {result.exported_records}/{result.total_records}")
        
        logger.info(f"Batch export job {export_id} completed")
        
    except Exception as e:
        logger.error(f"Batch export job {export_id} failed: {e}")


@router.get("/formats")
async def get_supported_formats() -> JSONResponse:
    """Get list of supported export formats."""
    formats = [
        {
            "format": ExportFormat.JSON.value,
            "description": "JSON format with full document and annotation data",
            "mime_type": "application/json",
            "extension": ".json"
        },
        {
            "format": ExportFormat.CSV.value,
            "description": "CSV format for tabular data analysis",
            "mime_type": "text/csv",
            "extension": ".csv"
        },
        {
            "format": ExportFormat.COCO.value,
            "description": "COCO format for computer vision applications",
            "mime_type": "application/json",
            "extension": ".json"
        }
    ]
    
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "supported_formats": formats,
            "total_formats": len(formats)
        }
    )


@router.post("/preview")
async def preview_export(
    request: ExportRequest,
    limit: int = Query(10, ge=1, le=100, description="Number of records to preview")
) -> JSONResponse:
    """Preview export data without creating a file."""
    try:
        # Create a temporary export with limited records
        preview_request = request.copy()
        preview_request.batch_size = limit
        
        # Create temporary export
        export_id = export_service.start_export(preview_request)
        
        # Perform export
        result = export_service.export_data(export_id, preview_request)
        
        if result.status == "completed" and result.file_path:
            # Read and return file content
            with open(result.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Clean up temporary file
            export_service.delete_export(export_id)
            
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "preview": content,
                    "format": request.format.value,
                    "records_count": result.exported_records,
                    "message": f"Preview of {result.exported_records} records"
                }
            )
        else:
            # Clean up on error
            export_service.delete_export(export_id)
            
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Preview failed: {result.error}"
            )
    
    except Exception as e:
        logger.error(f"Export preview failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Preview failed: {str(e)}"
        )