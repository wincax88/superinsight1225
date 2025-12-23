"""
FastAPI endpoints for data extraction in SuperInsight Platform.

Provides RESTful API for managing data extraction operations.
"""

import logging
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from src.extractors import (
    ExtractorFactory,
    ExtractionResult,
    SourceType,
    DatabaseType,
    FileType
)
from src.models.document import Document
from src.database.connection import get_db_session
from src.database.models import DocumentModel
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/extraction", tags=["Data Extraction"])


# Pydantic models for API requests/responses
class DatabaseExtractionRequest(BaseModel):
    """Request model for database extraction."""
    host: str = Field(..., description="Database host")
    port: int = Field(..., ge=1, le=65535, description="Database port")
    database: str = Field(..., description="Database name")
    username: str = Field(..., description="Database username")
    password: str = Field(..., description="Database password")
    database_type: str = Field(..., description="Database type (mysql, postgresql, oracle)")
    table_name: Optional[str] = Field(None, description="Specific table to extract")
    query: Optional[str] = Field(None, description="Custom SQL query")
    limit: Optional[int] = Field(100, ge=1, le=10000, description="Maximum records to extract")
    use_ssl: bool = Field(True, description="Use SSL connection")
    
    @field_validator('database_type')
    @classmethod
    def validate_database_type(cls, v):
        allowed_types = {'mysql', 'postgresql', 'oracle'}
        if v.lower() not in allowed_types:
            raise ValueError(f'database_type must be one of {allowed_types}')
        return v.lower()


class FileExtractionRequest(BaseModel):
    """Request model for file extraction."""
    file_path: str = Field(..., description="File path or URL")
    file_type: Optional[str] = Field(None, description="File type (pdf, docx, txt, html)")
    encoding: str = Field("utf-8", description="File encoding")
    use_ssl: bool = Field(True, description="Use SSL for URLs")
    
    @field_validator('file_type')
    @classmethod
    def validate_file_type(cls, v):
        if v is not None:
            allowed_types = {'pdf', 'docx', 'txt', 'html'}
            if v.lower() not in allowed_types:
                raise ValueError(f'file_type must be one of {allowed_types}')
            return v.lower()
        return v


class WebExtractionRequest(BaseModel):
    """Request model for web extraction."""
    base_url: str = Field(..., description="Base URL to extract")
    max_pages: int = Field(10, ge=1, le=100, description="Maximum pages to crawl")
    max_depth: int = Field(2, ge=1, le=5, description="Maximum crawl depth")


class APIExtractionRequest(BaseModel):
    """Request model for API extraction."""
    base_url: str = Field(..., description="API base URL")
    endpoint: str = Field("", description="API endpoint")
    method: str = Field("GET", description="HTTP method")
    headers: Dict[str, str] = Field(default_factory=dict, description="HTTP headers")
    auth_token: Optional[str] = Field(None, description="Authentication token")
    params: Dict[str, Any] = Field(default_factory=dict, description="Query parameters")
    data: Dict[str, Any] = Field(default_factory=dict, description="Request body data")
    paginate: bool = Field(True, description="Enable pagination")
    use_ssl: bool = Field(True, description="Use SSL")
    
    @field_validator('method')
    @classmethod
    def validate_method(cls, v):
        allowed_methods = {'GET', 'POST'}
        if v.upper() not in allowed_methods:
            raise ValueError(f'method must be one of {allowed_methods}')
        return v.upper()


class ExtractionJobResponse(BaseModel):
    """Response model for extraction job."""
    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Job status")
    created_at: datetime = Field(..., description="Job creation time")
    message: str = Field(..., description="Status message")


class ExtractionResultResponse(BaseModel):
    """Response model for extraction results."""
    job_id: str = Field(..., description="Job identifier")
    status: str = Field(..., description="Job status")
    success: bool = Field(..., description="Extraction success")
    documents_count: int = Field(..., description="Number of extracted documents")
    error: Optional[str] = Field(None, description="Error message if failed")
    documents: List[Dict[str, Any]] = Field(default_factory=list, description="Extracted documents")
    created_at: datetime = Field(..., description="Job creation time")
    completed_at: Optional[datetime] = Field(None, description="Job completion time")


# In-memory job storage (in production, use Redis or database)
extraction_jobs: Dict[str, Dict[str, Any]] = {}


def create_extraction_job(job_type: str, config: Dict[str, Any]) -> str:
    """Create a new extraction job."""
    job_id = str(uuid4())
    job = {
        "id": job_id,
        "type": job_type,
        "config": config,
        "status": "pending",
        "created_at": datetime.now(),
        "completed_at": None,
        "result": None,
        "error": None
    }
    extraction_jobs[job_id] = job
    logger.info(f"Created extraction job {job_id} of type {job_type}")
    return job_id


def update_job_status(job_id: str, status: str, result: Optional[ExtractionResult] = None, 
                     error: Optional[str] = None) -> None:
    """Update job status and result."""
    if job_id in extraction_jobs:
        job = extraction_jobs[job_id]
        job["status"] = status
        if status in ["completed", "failed"]:
            job["completed_at"] = datetime.now()
        if result:
            job["result"] = result
        if error:
            job["error"] = error
        logger.info(f"Updated job {job_id} status to {status}")


async def perform_database_extraction(job_id: str, config: DatabaseExtractionRequest) -> None:
    """Perform database extraction in background."""
    try:
        update_job_status(job_id, "running")
        
        # Create database extractor
        extractor = ExtractorFactory.create_database_extractor(
            host=config.host,
            port=config.port,
            database=config.database,
            username=config.username,
            password=config.password,
            database_type=config.database_type,
            use_ssl=config.use_ssl
        )
        
        # Test connection first
        if not extractor.test_connection():
            raise Exception("Database connection test failed")
        
        # Extract data
        if config.query:
            result = extractor.extract_data(query=config.query, limit=config.limit)
        elif config.table_name:
            result = extractor.extract_data(table_name=config.table_name, limit=config.limit)
        else:
            result = extractor.extract_data(limit=config.limit)
        
        # Close extractor
        extractor.close()
        
        if result.success:
            update_job_status(job_id, "completed", result)
        else:
            update_job_status(job_id, "failed", error=result.error)
            
    except Exception as e:
        logger.error(f"Database extraction job {job_id} failed: {e}")
        update_job_status(job_id, "failed", error=str(e))


async def perform_file_extraction(job_id: str, config: FileExtractionRequest) -> None:
    """Perform file extraction in background."""
    try:
        update_job_status(job_id, "running")
        
        # Create file extractor
        extractor = ExtractorFactory.create_file_extractor(
            file_path=config.file_path,
            file_type=config.file_type,
            encoding=config.encoding,
            use_ssl=config.use_ssl
        )
        
        # Test connection first
        if not extractor.test_connection():
            raise Exception("File access test failed")
        
        # Extract data
        result = extractor.extract_data()
        
        if result.success:
            update_job_status(job_id, "completed", result)
        else:
            update_job_status(job_id, "failed", error=result.error)
            
    except Exception as e:
        logger.error(f"File extraction job {job_id} failed: {e}")
        update_job_status(job_id, "failed", error=str(e))


async def perform_web_extraction(job_id: str, config: WebExtractionRequest) -> None:
    """Perform web extraction in background."""
    try:
        update_job_status(job_id, "running")
        
        # Create web extractor
        extractor = ExtractorFactory.create_web_extractor(
            base_url=config.base_url,
            max_pages=config.max_pages
        )
        
        # Test connection first
        if not extractor.test_connection():
            raise Exception("Web connection test failed")
        
        # Extract data
        result = extractor.extract_data(max_depth=config.max_depth)
        
        if result.success:
            update_job_status(job_id, "completed", result)
        else:
            update_job_status(job_id, "failed", error=result.error)
            
    except Exception as e:
        logger.error(f"Web extraction job {job_id} failed: {e}")
        update_job_status(job_id, "failed", error=str(e))


async def perform_api_extraction(job_id: str, config: APIExtractionRequest) -> None:
    """Perform API extraction in background."""
    try:
        update_job_status(job_id, "running")
        
        # Create API extractor
        extractor = ExtractorFactory.create_api_extractor(
            base_url=config.base_url,
            headers=config.headers,
            auth_token=config.auth_token,
            use_ssl=config.use_ssl
        )
        
        # Test connection first
        if not extractor.test_connection():
            raise Exception("API connection test failed")
        
        # Extract data
        result = extractor.extract_data(
            endpoint=config.endpoint,
            method=config.method,
            params=config.params,
            data=config.data if config.method == "POST" else None,
            paginate=config.paginate
        )
        
        # Close extractor
        extractor.close()
        
        if result.success:
            update_job_status(job_id, "completed", result)
        else:
            update_job_status(job_id, "failed", error=result.error)
            
    except Exception as e:
        logger.error(f"API extraction job {job_id} failed: {e}")
        update_job_status(job_id, "failed", error=str(e))


async def save_documents_to_database(documents: List[Document], db: Session) -> List[str]:
    """Save extracted documents to database."""
    try:
        document_ids = []
        
        for doc in documents:
            # Create database model
            db_document = DocumentModel(
                id=doc.id,
                source_type=doc.source_type,
                source_config=doc.source_config,
                content=doc.content,
                document_metadata=doc.metadata,
                created_at=doc.created_at,
                updated_at=doc.updated_at
            )
            
            db.add(db_document)
            document_ids.append(str(doc.id))
        
        db.commit()
        logger.info(f"Saved {len(documents)} documents to database")
        return document_ids
        
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to save documents to database: {e}")
        raise


# API Endpoints
@router.post("/database", response_model=ExtractionJobResponse)
async def extract_from_database(
    request: DatabaseExtractionRequest,
    background_tasks: BackgroundTasks
) -> ExtractionJobResponse:
    """Start database extraction job."""
    try:
        job_id = create_extraction_job("database", request.dict())
        
        # Start background task
        background_tasks.add_task(perform_database_extraction, job_id, request)
        
        return ExtractionJobResponse(
            job_id=job_id,
            status="pending",
            created_at=datetime.now(),
            message="Database extraction job started"
        )
        
    except Exception as e:
        logger.error(f"Failed to start database extraction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start extraction: {str(e)}"
        )


@router.post("/file", response_model=ExtractionJobResponse)
async def extract_from_file(
    request: FileExtractionRequest,
    background_tasks: BackgroundTasks
) -> ExtractionJobResponse:
    """Start file extraction job."""
    try:
        job_id = create_extraction_job("file", request.dict())
        
        # Start background task
        background_tasks.add_task(perform_file_extraction, job_id, request)
        
        return ExtractionJobResponse(
            job_id=job_id,
            status="pending",
            created_at=datetime.now(),
            message="File extraction job started"
        )
        
    except Exception as e:
        logger.error(f"Failed to start file extraction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start extraction: {str(e)}"
        )


@router.post("/web", response_model=ExtractionJobResponse)
async def extract_from_web(
    request: WebExtractionRequest,
    background_tasks: BackgroundTasks
) -> ExtractionJobResponse:
    """Start web extraction job."""
    try:
        job_id = create_extraction_job("web", request.dict())
        
        # Start background task
        background_tasks.add_task(perform_web_extraction, job_id, request)
        
        return ExtractionJobResponse(
            job_id=job_id,
            status="pending",
            created_at=datetime.now(),
            message="Web extraction job started"
        )
        
    except Exception as e:
        logger.error(f"Failed to start web extraction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start extraction: {str(e)}"
        )


@router.post("/api", response_model=ExtractionJobResponse)
async def extract_from_api(
    request: APIExtractionRequest,
    background_tasks: BackgroundTasks
) -> ExtractionJobResponse:
    """Start API extraction job."""
    try:
        job_id = create_extraction_job("api", request.dict())
        
        # Start background task
        background_tasks.add_task(perform_api_extraction, job_id, request)
        
        return ExtractionJobResponse(
            job_id=job_id,
            status="pending",
            created_at=datetime.now(),
            message="API extraction job started"
        )
        
    except Exception as e:
        logger.error(f"Failed to start API extraction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start extraction: {str(e)}"
        )


@router.get("/jobs/{job_id}", response_model=ExtractionResultResponse)
async def get_extraction_job(job_id: str) -> ExtractionResultResponse:
    """Get extraction job status and results."""
    if job_id not in extraction_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
    
    job = extraction_jobs[job_id]
    
    # Prepare documents for response
    documents = []
    if job["result"] and job["result"].success:
        documents = [doc.to_dict() for doc in job["result"].documents]
    
    return ExtractionResultResponse(
        job_id=job_id,
        status=job["status"],
        success=job["result"].success if job["result"] else False,
        documents_count=len(documents),
        error=job["error"],
        documents=documents,
        created_at=job["created_at"],
        completed_at=job["completed_at"]
    )


@router.get("/jobs", response_model=List[ExtractionResultResponse])
async def list_extraction_jobs() -> List[ExtractionResultResponse]:
    """List all extraction jobs."""
    jobs = []
    
    for job_id, job in extraction_jobs.items():
        documents = []
        if job["result"] and job["result"].success:
            documents = [doc.to_dict() for doc in job["result"].documents]
        
        jobs.append(ExtractionResultResponse(
            job_id=job_id,
            status=job["status"],
            success=job["result"].success if job["result"] else False,
            documents_count=len(documents),
            error=job["error"],
            documents=documents,
            created_at=job["created_at"],
            completed_at=job["completed_at"]
        ))
    
    return jobs


@router.post("/jobs/{job_id}/save")
async def save_extraction_results(
    job_id: str,
    db: Session = Depends(get_db_session)
) -> JSONResponse:
    """Save extraction results to database."""
    if job_id not in extraction_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
    
    job = extraction_jobs[job_id]
    
    if job["status"] != "completed" or not job["result"] or not job["result"].success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Job not completed successfully"
        )
    
    try:
        document_ids = await save_documents_to_database(job["result"].documents, db)
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": f"Saved {len(document_ids)} documents to database",
                "document_ids": document_ids
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to save extraction results: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save results: {str(e)}"
        )


@router.delete("/jobs/{job_id}")
async def delete_extraction_job(job_id: str) -> JSONResponse:
    """Delete extraction job."""
    if job_id not in extraction_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
    
    del extraction_jobs[job_id]
    
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"message": "Job deleted successfully"}
    )


@router.post("/test-connection")
async def test_extraction_connection(
    request: Union[DatabaseExtractionRequest, FileExtractionRequest, APIExtractionRequest]
) -> JSONResponse:
    """Test connection to data source without extracting data."""
    try:
        if isinstance(request, DatabaseExtractionRequest):
            extractor = ExtractorFactory.create_database_extractor(
                host=request.host,
                port=request.port,
                database=request.database,
                username=request.username,
                password=request.password,
                database_type=request.database_type,
                use_ssl=request.use_ssl
            )
        elif isinstance(request, FileExtractionRequest):
            extractor = ExtractorFactory.create_file_extractor(
                file_path=request.file_path,
                file_type=request.file_type,
                encoding=request.encoding,
                use_ssl=request.use_ssl
            )
        elif isinstance(request, APIExtractionRequest):
            extractor = ExtractorFactory.create_api_extractor(
                base_url=request.base_url,
                headers=request.headers,
                auth_token=request.auth_token,
                use_ssl=request.use_ssl
            )
        else:
            raise ValueError("Unsupported request type")
        
        # Test connection
        success = extractor.test_connection()
        
        # Close extractor if it has a close method
        if hasattr(extractor, 'close'):
            extractor.close()
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "success": success,
                "message": "Connection successful" if success else "Connection failed"
            }
        )
        
    except Exception as e:
        logger.error(f"Connection test failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "message": f"Connection test failed: {str(e)}"
            }
        )