"""
AI Models API endpoints for SuperInsight platform.

Provides REST API for managing AI models, performance analysis, auto-selection,
and comprehensive model integration services.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

from ..ai import (
    EnhancedModelManager, ModelConfig, ModelType, PerformanceMetric,
    ModelPerformanceAnalyzer, ModelAutoSelector
)
from ..ai.integration_service import get_integration_service, AIModelIntegrationService
from ..database.connection import get_db_session


# Pydantic models for API
class ModelRegistrationRequest(BaseModel):
    """Request model for registering a model with credentials."""
    model_type: ModelType
    model_name: str
    api_key: str
    secret_key: Optional[str] = None  # For Tencent models
    base_url: Optional[str] = None
    max_tokens: int = Field(default=1000, ge=1, le=8000)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    timeout: int = Field(default=30, ge=1, le=300)


class ModelComparisonRequest(BaseModel):
    """Request model for comparing models."""
    task_type: str
    model_names: Optional[List[str]] = None  # If None, compare all models


class OptimizationRequest(BaseModel):
    """Request model for model optimization."""
    task_type: str
    optimization_target: PerformanceMetric = PerformanceMetric.ACCURACY


class ModelSelectionRequest(BaseModel):
    """Request model for auto model selection."""
    task_type: str
    max_response_time: Optional[float] = None
    min_accuracy: Optional[float] = None
    preferred_models: Optional[List[ModelType]] = None
    budget_constraint: Optional[float] = None
    min_samples: int = Field(default=5, ge=1)


class BenchmarkRequest(BaseModel):
    """Request model for model benchmarking."""
    test_cases: List[Dict[str, Any]]
    task_type: str = "classification"
    models: Optional[List[str]] = None  # If None, benchmark all models


class PredictionRequest(BaseModel):
    """Request model for making predictions with performance tracking."""
    annotator_name: str
    task_content: str
    task_type: str = "classification"
    project_id: str = "default"
    ground_truth: Optional[Dict[str, Any]] = None


# Global integration service instance
integration_service: Optional[AIModelIntegrationService] = None


def get_integration_service_instance() -> AIModelIntegrationService:
    """Get or create integration service instance."""
    global integration_service
    if integration_service is None:
        integration_service = get_integration_service()
    return integration_service


# API Router
router = APIRouter(prefix="/api/ai-models", tags=["AI Models"])


@router.post("/register-with-credentials", response_model=Dict[str, Any])
async def register_model_with_credentials(
    request: ModelRegistrationRequest,
    service: AIModelIntegrationService = Depends(get_integration_service_instance)
):
    """Register a model with API credentials."""
    try:
        version_id = await service.register_model_with_credentials(
            model_type=request.model_type,
            model_name=request.model_name,
            api_key=request.api_key,
            secret_key=request.secret_key,
            base_url=request.base_url,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            timeout=request.timeout
        )
        
        return {
            "success": True,
            "version_id": version_id,
            "model_type": request.model_type.value,
            "model_name": request.model_name,
            "message": "Model registered successfully with credentials"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/predict-with-best", response_model=Dict[str, Any])
async def predict_with_best_model(
    request: PredictionRequest,
    service: AIModelIntegrationService = Depends(get_integration_service_instance)
):
    """Make prediction using the best available model for the task."""
    try:
        # Create mock task object
        from uuid import uuid4
        mock_task = type('Task', (), {
            'id': uuid4(),
            'project_id': request.project_id,
            'content': request.task_content
        })()
        
        # Parse requirements if provided
        requirements = {}
        if hasattr(request, 'max_response_time') and request.max_response_time:
            requirements['max_response_time'] = request.max_response_time
        if hasattr(request, 'min_accuracy') and request.min_accuracy:
            requirements['min_accuracy'] = request.min_accuracy
        
        prediction = await service.predict_with_best_model(
            task=mock_task,
            task_type=request.task_type,
            requirements=requirements if requirements else None
        )
        
        return {
            "success": True,
            "prediction": prediction.to_dict(),
            "model_info": {
                "type": prediction.ai_model_config.model_type.value,
                "name": prediction.ai_model_config.model_name,
                "processing_time": prediction.processing_time,
                "confidence": prediction.confidence
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/compare-models", response_model=Dict[str, Any])
async def compare_models_for_task(
    request: ModelComparisonRequest,
    service: AIModelIntegrationService = Depends(get_integration_service_instance)
):
    """Compare models for a specific task type."""
    try:
        comparison_result = await service.compare_models_for_task(
            task_type=request.task_type,
            model_names=request.model_names
        )
        
        return comparison_result
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/performance-report", response_model=Dict[str, Any])
async def get_comprehensive_performance_report(
    task_type: Optional[str] = None,
    format: str = "json",
    service: AIModelIntegrationService = Depends(get_integration_service_instance)
):
    """Get comprehensive performance report."""
    try:
        if format == "text":
            report = await service.get_model_performance_report(task_type)
            return {
                "format": "text",
                "report": report,
                "task_type": task_type
            }
        else:
            # Get structured data for JSON format
            status = await service.get_integration_status()
            return {
                "format": "json",
                "integration_status": status,
                "task_type": task_type
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimize-selection", response_model=Dict[str, Any])
async def optimize_model_selection_enhanced(
    request: OptimizationRequest,
    service: AIModelIntegrationService = Depends(get_integration_service_instance)
):
    """Optimize model selection with enhanced analysis."""
    try:
        optimization_result = await service.optimize_model_selection(request.task_type)
        return optimization_result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/integration-status", response_model=Dict[str, Any])
async def get_integration_service_status(
    service: AIModelIntegrationService = Depends(get_integration_service_instance)
):
    """Get overall integration service status."""
    try:
        status = await service.get_integration_status()
        return status
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health-comprehensive", response_model=Dict[str, Any])
async def comprehensive_health_check(
    service: AIModelIntegrationService = Depends(get_integration_service_instance)
):
    """Perform comprehensive health check on all models."""
    try:
        health_status = await service.health_check_all_models()
        return health_status
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cleanup-enhanced", response_model=Dict[str, Any])
async def cleanup_old_data(
    days_threshold: int = 30,
    service: AIModelIntegrationService = Depends(get_integration_service_instance)
):
    """Clean up old performance data and benchmarks."""
    try:
        cleanup_result = await service.cleanup_old_data(days_threshold)
        return cleanup_result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list", response_model=Dict[str, Any])
async def list_models(
    service: AIModelIntegrationService = Depends(get_integration_service_instance)
):
    """List all registered models with their status."""
    try:
        status = await service.get_integration_status()
        health_status = await service.health_check_all_models()
        
        return {
            "models": status["model_statistics"],
            "health_status": health_status,
            "service_status": status["service_info"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict", response_model=Dict[str, Any])
async def predict_with_tracking(
    request: PredictionRequest,
    manager: EnhancedModelManager = Depends(get_enhanced_manager)
):
    """Make prediction with performance tracking."""
    try:
        # Create mock task object
        from uuid import uuid4
        mock_task = type('Task', (), {
            'id': uuid4(),
            'project_id': request.project_id,
            'content': request.task_content
        })()
        
        prediction = await manager.predict_with_performance_tracking(
            annotator_name=request.annotator_name,
            task=mock_task,
            task_type=request.task_type,
            ground_truth=request.ground_truth
        )
        
        return {
            "success": True,
            "prediction": prediction.to_dict(),
            "model_info": {
                "name": request.annotator_name,
                "type": prediction.ai_model_config.model_type.value,
                "processing_time": prediction.processing_time
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/auto-select", response_model=Dict[str, Any])
async def auto_select_model(
    request: ModelSelectionRequest,
    manager: EnhancedModelManager = Depends(get_enhanced_manager)
):
    """Automatically select the best model for a task."""
    try:
        requirements = {
            "max_response_time": request.max_response_time,
            "min_accuracy": request.min_accuracy,
            "preferred_models": request.preferred_models,
            "budget_constraint": request.budget_constraint,
            "min_samples": request.min_samples
        }
        
        # Remove None values
        requirements = {k: v for k, v in requirements.items() if v is not None}
        
        selected_model = await manager.auto_select_model(request.task_type, requirements)
        
        if selected_model:
            # Get recommendations for context
            recommendations = await manager.get_model_recommendations(request.task_type)
            
            return {
                "success": True,
                "selected_model": selected_model,
                "task_type": request.task_type,
                "requirements": requirements,
                "recommendations": recommendations
            }
        else:
            return {
                "success": False,
                "message": "No suitable model found for the given requirements",
                "recommendations": await manager.get_model_recommendations(request.task_type)
            }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/recommendations/{task_type}", response_model=Dict[str, Any])
async def get_model_recommendations(
    task_type: str,
    top_n: int = 3,
    manager: EnhancedModelManager = Depends(get_enhanced_manager)
):
    """Get model recommendations for a specific task type."""
    try:
        recommendations = await manager.get_model_recommendations(task_type, top_n)
        
        return {
            "task_type": task_type,
            "recommendations": recommendations,
            "total_available": len(recommendations)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance/compare/{task_type}", response_model=Dict[str, Any])
async def compare_models(
    task_type: str,
    metric: PerformanceMetric = PerformanceMetric.ACCURACY,
    manager: EnhancedModelManager = Depends(get_enhanced_manager)
):
    """Compare models for a specific task type and metric."""
    try:
        comparison = await manager.compare_models(task_type, metric)
        
        return {
            "task_type": task_type,
            "metric": metric.value,
            "comparison": [
                {"model": model, "score": score}
                for model, score in comparison
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/benchmark", response_model=Dict[str, Any])
async def benchmark_models(
    request: BenchmarkRequest,
    background_tasks: BackgroundTasks,
    manager: EnhancedModelManager = Depends(get_enhanced_manager)
):
    """Benchmark models against test cases."""
    try:
        # Run benchmark in background for large test sets
        if len(request.test_cases) > 10:
            background_tasks.add_task(
                manager.benchmark_all_models,
                request.test_cases,
                request.task_type
            )
            
            return {
                "success": True,
                "message": "Benchmark started in background",
                "task_type": request.task_type,
                "test_cases_count": len(request.test_cases)
            }
        else:
            # Run synchronously for small test sets
            results = await manager.benchmark_all_models(request.test_cases, request.task_type)
            
            return {
                "success": True,
                "results": results,
                "task_type": request.task_type,
                "test_cases_count": len(request.test_cases)
            }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/performance/report", response_model=Dict[str, Any])
async def get_performance_report(
    task_type: Optional[str] = None,
    format: str = "json",
    manager: EnhancedModelManager = Depends(get_enhanced_manager)
):
    """Get performance report for models."""
    try:
        if format == "text":
            report = await manager.get_performance_report(task_type)
            return {
                "format": "text",
                "report": report
            }
        else:
            # JSON format
            summary = manager.performance_analyzer.get_performance_summary(task_type)
            return {
                "format": "json",
                "summary": summary,
                "task_type": task_type
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimize/{task_type}", response_model=Dict[str, Any])
async def optimize_model_selection(
    task_type: str,
    optimization_target: PerformanceMetric = PerformanceMetric.ACCURACY,
    manager: EnhancedModelManager = Depends(get_enhanced_manager)
):
    """Optimize model selection for a specific task type."""
    try:
        optimization_results = await manager.optimize_model_selection(
            task_type, optimization_target
        )
        
        return optimization_results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=Dict[str, Any])
async def health_check_models(
    manager: EnhancedModelManager = Depends(get_enhanced_manager)
):
    """Perform health check on all models."""
    try:
        health_status = await manager.health_check_all_models()
        
        healthy_count = sum(1 for status in health_status.values() if status.get("healthy", False))
        total_count = len(health_status)
        
        return {
            "overall_health": "healthy" if healthy_count == total_count else "degraded",
            "healthy_models": healthy_count,
            "total_models": total_count,
            "health_percentage": (healthy_count / total_count * 100) if total_count > 0 else 0,
            "models": health_status,
            "last_checked": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics", response_model=Dict[str, Any])
async def get_model_statistics(
    manager: EnhancedModelManager = Depends(get_enhanced_manager)
):
    """Get comprehensive statistics about all models."""
    try:
        statistics = await manager.get_model_statistics()
        return statistics
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cleanup", response_model=Dict[str, Any])
async def cleanup_inactive_models(
    days_threshold: int = 30,
    manager: EnhancedModelManager = Depends(get_enhanced_manager)
):
    """Clean up models that haven't been used recently."""
    try:
        cleanup_results = await manager.cleanup_inactive_models(days_threshold)
        return cleanup_results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/supported-types", response_model=Dict[str, Any])
async def get_supported_model_types():
    """Get list of supported model types and their capabilities."""
    try:
        from ..ai.factory import AnnotatorFactory
        
        supported_types = AnnotatorFactory.get_supported_model_types()
        
        # Model type capabilities
        capabilities = {
            ModelType.OLLAMA: {
                "description": "Local open-source models via Ollama",
                "deployment": "local",
                "cost": "free",
                "languages": ["en", "zh", "multilingual"]
            },
            ModelType.HUGGINGFACE: {
                "description": "HuggingFace Transformers models",
                "deployment": "local/cloud",
                "cost": "free/paid",
                "languages": ["en", "zh", "multilingual"]
            },
            ModelType.ZHIPU_GLM: {
                "description": "Zhipu GLM (ChatGLM) API",
                "deployment": "cloud",
                "cost": "paid",
                "languages": ["zh", "en"]
            },
            ModelType.BAIDU_WENXIN: {
                "description": "Baidu Wenxin (ERNIE) API",
                "deployment": "cloud",
                "cost": "paid",
                "languages": ["zh", "en"]
            },
            ModelType.ALIBABA_TONGYI: {
                "description": "Alibaba Tongyi Qianwen API",
                "deployment": "cloud",
                "cost": "paid",
                "languages": ["zh", "en"]
            },
            ModelType.TENCENT_HUNYUAN: {
                "description": "Tencent Hunyuan API",
                "deployment": "cloud",
                "cost": "paid",
                "languages": ["zh", "en"]
            }
        }
        
        return {
            "supported_types": [t.value for t in supported_types],
            "capabilities": {t.value: capabilities.get(t, {}) for t in supported_types},
            "total_supported": len(supported_types)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))