"""
AI Annotation API endpoints for SuperInsight platform.

Provides REST API endpoints for AI pre-annotation services.
"""

from typing import Dict, Any, List, Optional
from uuid import UUID
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field
import redis

try:
    from ai import (
        AnnotatorFactory,
        ModelConfig,
        ModelType,
        BatchAnnotationProcessor,
        BatchJobConfig,
        BatchResult,
        PredictionCacheService,
        CacheConfig,
        ModelVersionManager
    )
    from models.task import Task
    from config.settings import settings
except ImportError:
    from src.ai import (
        AnnotatorFactory,
        ModelConfig,
        ModelType,
        BatchAnnotationProcessor,
        BatchJobConfig,
        BatchResult,
        PredictionCacheService,
        CacheConfig,
        ModelVersionManager
    )
    from src.models.task import Task
    from src.config.settings import settings


# Request/Response models
class PredictionRequest(BaseModel):
    """Request model for single prediction."""
    task_id: UUID
    ai_model_config: ModelConfig
    content: Optional[str] = None


class BatchPredictionRequest(BaseModel):
    """Request model for batch prediction."""
    task_ids: List[UUID]
    ai_model_configs: List[ModelConfig]
    batch_config: Optional[BatchJobConfig] = None


class PredictionResponse(BaseModel):
    """Response model for prediction."""
    task_id: UUID
    prediction_data: Dict[str, Any]
    confidence: float
    processing_time: float
    model_info: Dict[str, Any]
    cached: bool = False


class BatchJobResponse(BaseModel):
    """Response model for batch job submission."""
    job_id: str
    status: str
    message: str


class ModelListResponse(BaseModel):
    """Response model for available models."""
    models: List[Dict[str, Any]]


# Dependencies
def get_redis_client() -> redis.Redis:
    """Get Redis client for caching."""
    return redis.Redis.from_url(settings.redis.redis_url)


def get_batch_processor(redis_client: redis.Redis = Depends(get_redis_client)) -> BatchAnnotationProcessor:
    """Get batch annotation processor."""
    return BatchAnnotationProcessor(redis_client)


def get_cache_service(redis_client: redis.Redis = Depends(get_redis_client)) -> PredictionCacheService:
    """Get prediction cache service."""
    cache_config = CacheConfig(
        ttl_hours=24,
        max_entries=10000,
        cache_hit_threshold=0.7
    )
    return PredictionCacheService(redis_client, cache_config)


def get_model_manager(redis_client: redis.Redis = Depends(get_redis_client)) -> ModelVersionManager:
    """Get model version manager."""
    return ModelVersionManager(redis_client, "./model_versions")


# Router
router = APIRouter(prefix="/api/v1/ai", tags=["AI Annotation"])


@router.post("/predict", response_model=PredictionResponse)
async def predict_single(
    request: PredictionRequest,
    cache_service: PredictionCacheService = Depends(get_cache_service)
):
    """
    Generate AI prediction for a single task.
    
    Args:
        request: Prediction request with task and model configuration
        cache_service: Cache service for prediction caching
        
    Returns:
        Prediction response with results and metadata
    """
    try:
        # Create a dummy task for demonstration
        # In real implementation, fetch from database
        task = Task(
            id=request.task_id,
            document_id=request.task_id,  # Simplified
            project_id="demo_project"
        )
        
        # Check cache first
        cached_prediction = await cache_service.get_cached_prediction(
            task, request.ai_model_config, request.content
        )
        
        if cached_prediction:
            return PredictionResponse(
                task_id=cached_prediction.task_id,
                prediction_data=cached_prediction.prediction_data,
                confidence=cached_prediction.confidence,
                processing_time=cached_prediction.processing_time,
                model_info=cached_prediction.ai_model_config.dict(),
                cached=True
            )
        
        # Create annotator and generate prediction
        annotator = AnnotatorFactory.create_annotator(request.ai_model_config)
        prediction = await annotator.predict(task)
        
        # Cache the result
        await cache_service.cache_prediction(prediction, task, request.content)
        
        return PredictionResponse(
            task_id=prediction.task_id,
            prediction_data=prediction.prediction_data,
            confidence=prediction.confidence,
            processing_time=prediction.processing_time,
            model_info=annotator.get_model_info(),
            cached=False
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/predict/batch", response_model=BatchJobResponse)
async def predict_batch(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    batch_processor: BatchAnnotationProcessor = Depends(get_batch_processor)
):
    """
    Submit batch prediction job.
    
    Args:
        request: Batch prediction request
        background_tasks: FastAPI background tasks
        batch_processor: Batch annotation processor
        
    Returns:
        Batch job response with job ID and status
    """
    try:
        # Create dummy tasks for demonstration
        tasks = []
        for task_id in request.task_ids:
            task = Task(
                id=task_id,
                document_id=task_id,
                project_id="demo_project"
            )
            tasks.append(task)
        
        # Use provided batch config or create default
        batch_config = request.batch_config or BatchJobConfig(
            ai_model_configs=request.ai_model_configs,
            max_concurrent_tasks=5,
            retry_attempts=2
        )
        
        # Submit batch job
        job_id = await batch_processor.submit_batch_job(tasks, batch_config)
        
        return BatchJobResponse(
            job_id=job_id,
            status="submitted",
            message=f"Batch job submitted with {len(tasks)} tasks"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch submission failed: {str(e)}")


@router.get("/batch/{job_id}", response_model=Dict[str, Any])
async def get_batch_status(
    job_id: str,
    batch_processor: BatchAnnotationProcessor = Depends(get_batch_processor)
):
    """
    Get status of a batch prediction job.
    
    Args:
        job_id: ID of the batch job
        batch_processor: Batch annotation processor
        
    Returns:
        Batch job status and results
    """
    try:
        batch_result = await batch_processor.get_job_status(job_id)
        
        if not batch_result:
            raise HTTPException(status_code=404, detail="Batch job not found")
        
        return batch_result.to_dict()
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get batch status: {str(e)}")


@router.delete("/batch/{job_id}")
async def cancel_batch_job(
    job_id: str,
    batch_processor: BatchAnnotationProcessor = Depends(get_batch_processor)
):
    """
    Cancel a batch prediction job.
    
    Args:
        job_id: ID of the batch job to cancel
        batch_processor: Batch annotation processor
        
    Returns:
        Cancellation status
    """
    try:
        cancelled = await batch_processor.cancel_job(job_id)
        
        if not cancelled:
            raise HTTPException(status_code=404, detail="Batch job not found or already completed")
        
        return {"message": f"Batch job {job_id} cancelled successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cancel batch job: {str(e)}")


@router.get("/models", response_model=ModelListResponse)
async def list_available_models():
    """
    List available AI models.
    
    Returns:
        List of available models with their configurations
    """
    try:
        supported_types = AnnotatorFactory.get_supported_model_types()
        
        models = []
        for model_type in supported_types:
            model_info = {
                "model_type": model_type.value,
                "description": f"{model_type.value.replace('_', ' ').title()} models",
                "supported": True
            }
            
            # Add specific model examples
            if model_type == ModelType.OLLAMA:
                model_info["examples"] = ["llama2", "codellama", "mistral"]
            elif model_type == ModelType.HUGGINGFACE:
                model_info["examples"] = ["bert-base-chinese", "distilbert-base-uncased"]
            elif model_type == ModelType.ZHIPU_GLM:
                model_info["examples"] = ["glm-4", "glm-3-turbo", "chatglm3-6b"]
            elif model_type == ModelType.BAIDU_WENXIN:
                model_info["examples"] = ["ernie-bot", "ernie-bot-turbo", "ernie-4.0"]
            
            models.append(model_info)
        
        return ModelListResponse(models=models)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")


@router.get("/models/{model_type}/health")
async def check_model_health(model_type: str):
    """
    Check health status of a specific model type.
    
    Args:
        model_type: Type of model to check
        
    Returns:
        Health status information
    """
    try:
        # Validate model type
        try:
            model_type_enum = ModelType(model_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid model type: {model_type}")
        
        # Create a test configuration
        test_config = ModelConfig(
            model_type=model_type_enum,
            model_name="test",
            api_key=getattr(settings.ai, f"{model_type}_api_key", None),
            base_url=getattr(settings.ai, f"{model_type}_base_url", None)
        )
        
        # Try to create annotator
        try:
            annotator = AnnotatorFactory.create_annotator(test_config)
            
            # Check availability if method exists
            if hasattr(annotator, 'check_model_availability'):
                available = await annotator.check_model_availability()
            else:
                available = True
            
            return {
                "model_type": model_type,
                "status": "healthy" if available else "unavailable",
                "available": available,
                "message": "Model is accessible" if available else "Model is not accessible"
            }
            
        except Exception as e:
            return {
                "model_type": model_type,
                "status": "error",
                "available": False,
                "message": f"Model check failed: {str(e)}"
            }
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.get("/cache/stats")
async def get_cache_statistics(
    cache_service: PredictionCacheService = Depends(get_cache_service)
):
    """
    Get cache performance statistics.
    
    Args:
        cache_service: Prediction cache service
        
    Returns:
        Cache statistics
    """
    try:
        stats = await cache_service.get_cache_statistics()
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get cache stats: {str(e)}")


@router.delete("/cache")
async def clear_cache(
    cache_service: PredictionCacheService = Depends(get_cache_service)
):
    """
    Clear all cached predictions.
    
    Args:
        cache_service: Prediction cache service
        
    Returns:
        Number of entries cleared
    """
    try:
        cleared_count = await cache_service.clear_all_cache()
        return {"message": f"Cleared {cleared_count} cache entries"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")


@router.get("/batch/stats")
async def get_batch_statistics(
    batch_processor: BatchAnnotationProcessor = Depends(get_batch_processor)
):
    """
    Get batch processing statistics.
    
    Args:
        batch_processor: Batch annotation processor
        
    Returns:
        Batch processing statistics
    """
    try:
        stats = await batch_processor.get_job_statistics()
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get batch stats: {str(e)}")


@router.post("/models/register")
async def register_model_version(
    model_type: ModelType,
    model_name: str,
    version: str,
    config: ModelConfig,
    description: str = "",
    model_manager: ModelVersionManager = Depends(get_model_manager)
):
    """
    Register a new model version.
    
    Args:
        model_type: Type of the model
        model_name: Name of the model
        version: Version string
        config: Model configuration
        description: Description of the model
        model_manager: Model version manager
        
    Returns:
        Registration result
    """
    try:
        version_id = await model_manager.register_model_version(
            model_type=model_type,
            model_name=model_name,
            version=version,
            config=config,
            description=description
        )
        
        return {
            "version_id": version_id,
            "message": f"Model version {model_name}:{version} registered successfully"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to register model: {str(e)}")


@router.get("/models/versions")
async def list_model_versions(
    model_type: Optional[ModelType] = None,
    model_name: Optional[str] = None,
    model_manager: ModelVersionManager = Depends(get_model_manager)
):
    """
    List registered model versions.
    
    Args:
        model_type: Optional model type filter
        model_name: Optional model name filter
        model_manager: Model version manager
        
    Returns:
        List of model versions
    """
    try:
        versions = await model_manager.list_model_versions(
            model_type=model_type,
            model_name=model_name
        )
        
        return {
            "versions": [version.to_dict() for version in versions],
            "count": len(versions)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list model versions: {str(e)}")