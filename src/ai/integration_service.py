"""
AI Model Integration Service for SuperInsight platform.

Provides unified interface for AI model management, performance tracking,
and intelligent model selection across all supported providers.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import logging

from .base import AIAnnotator, ModelConfig, ModelType, Prediction
from .factory import AnnotatorFactory
from .enhanced_model_manager import EnhancedModelManager
from .model_performance import ModelPerformanceAnalyzer, PerformanceMetric
from .model_comparison import ModelBenchmarkSuite, ModelAutoSelector
from .alibaba_annotator import AlibabaAnnotator
from .hunyuan_annotator import HunyuanAnnotator

logger = logging.getLogger(__name__)


class AIModelIntegrationService:
    """
    Unified service for AI model integration and management.
    
    Provides high-level interface for:
    - Model registration and management
    - Performance tracking and analysis
    - Automatic model selection
    - Benchmarking and comparison
    - Health monitoring
    """
    
    def __init__(self, storage_path: Optional[str] = None, redis_client=None):
        """Initialize the integration service."""
        self.storage_path = Path(storage_path) if storage_path else Path("./ai_integration")
        self.storage_path.mkdir(exist_ok=True)
        
        # Initialize core components
        self.model_manager = EnhancedModelManager(
            str(self.storage_path / "models"), 
            redis_client
        )
        self.performance_analyzer = ModelPerformanceAnalyzer(
            str(self.storage_path / "performance")
        )
        self.benchmark_suite = ModelBenchmarkSuite(
            str(self.storage_path / "benchmarks")
        )
        self.auto_selector = ModelAutoSelector(
            self.benchmark_suite, 
            self.performance_analyzer
        )
        
        # Service configuration
        self.config = self._load_service_config()
        
        # Initialize default models
        asyncio.create_task(self._initialize_default_models())
    
    def _load_service_config(self) -> Dict[str, Any]:
        """Load service configuration."""
        try:
            config_file = self.storage_path / "service_config.json"
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load service config: {e}")
        
        # Default configuration
        return {
            "auto_benchmark_enabled": True,
            "benchmark_interval_hours": 24,
            "performance_tracking_enabled": True,
            "health_check_interval_minutes": 30,
            "default_task_types": ["sentiment", "classification", "ner"],
            "model_selection_strategy": "performance_based"
        }
    
    def _save_service_config(self) -> None:
        """Save service configuration."""
        try:
            config_file = self.storage_path / "service_config.json"
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save service config: {e}")
    
    async def _initialize_default_models(self) -> None:
        """Initialize default model configurations."""
        try:
            # Alibaba Tongyi models
            alibaba_configs = [
                ModelConfig(
                    model_type=ModelType.ALIBABA_TONGYI,
                    model_name="qwen-turbo",
                    max_tokens=1000,
                    temperature=0.7,
                    timeout=30
                ),
                ModelConfig(
                    model_type=ModelType.ALIBABA_TONGYI,
                    model_name="qwen-plus",
                    max_tokens=1500,
                    temperature=0.7,
                    timeout=45
                ),
                ModelConfig(
                    model_type=ModelType.ALIBABA_TONGYI,
                    model_name="qwen-max",
                    max_tokens=2000,
                    temperature=0.7,
                    timeout=60
                )
            ]
            
            # Tencent Hunyuan models
            hunyuan_configs = [
                ModelConfig(
                    model_type=ModelType.TENCENT_HUNYUAN,
                    model_name="hunyuan-lite",
                    max_tokens=1000,
                    temperature=0.7,
                    timeout=30
                ),
                ModelConfig(
                    model_type=ModelType.TENCENT_HUNYUAN,
                    model_name="hunyuan-standard",
                    max_tokens=1500,
                    temperature=0.7,
                    timeout=45
                ),
                ModelConfig(
                    model_type=ModelType.TENCENT_HUNYUAN,
                    model_name="hunyuan-pro",
                    max_tokens=2000,
                    temperature=0.7,
                    timeout=60
                )
            ]
            
            # Register models
            all_configs = alibaba_configs + hunyuan_configs
            
            for config in all_configs:
                model_name = f"{config.model_type.value}_{config.model_name}"
                try:
                    await self.model_manager.register_model(
                        name=model_name,
                        config=config,
                        description=f"Default {config.model_type.value} model: {config.model_name}",
                        tags={"provider", config.model_type.value, "default"},
                        auto_activate=False  # Don't auto-activate without API keys
                    )
                    logger.info(f"Registered model: {model_name}")
                except Exception as e:
                    logger.warning(f"Failed to register {model_name}: {e}")
            
        except Exception as e:
            logger.error(f"Failed to initialize default models: {e}")
    
    async def register_model_with_credentials(self, model_type: ModelType, 
                                            model_name: str,
                                            api_key: str,
                                            secret_key: Optional[str] = None,
                                            base_url: Optional[str] = None,
                                            **kwargs) -> str:
        """
        Register a model with API credentials.
        
        Args:
            model_type: Type of model
            model_name: Name of the model
            api_key: API key for the service
            secret_key: Secret key (for Tencent models)
            base_url: Custom base URL
            **kwargs: Additional configuration parameters
            
        Returns:
            Version ID of the registered model
        """
        # Create model configuration
        config_dict = {
            "model_type": model_type,
            "model_name": model_name,
            "api_key": api_key,
            "max_tokens": kwargs.get("max_tokens", 1000),
            "temperature": kwargs.get("temperature", 0.7),
            "timeout": kwargs.get("timeout", 30)
        }
        
        if base_url:
            config_dict["base_url"] = base_url
        
        # Add secret key for Tencent models
        if model_type == ModelType.TENCENT_HUNYUAN and secret_key:
            config_dict["secret_key"] = secret_key
        
        config = ModelConfig(**config_dict)
        
        # Register with model manager
        service_name = f"{model_type.value}_{model_name}_configured"
        version_id = await self.model_manager.register_model(
            name=service_name,
            config=config,
            description=f"Configured {model_type.value} model: {model_name}",
            tags={"provider", model_type.value, "configured"},
            auto_activate=True
        )
        
        # Test model availability
        try:
            annotator = await self.model_manager.get_annotator(service_name)
            if annotator and hasattr(annotator, 'check_model_availability'):
                is_available = await annotator.check_model_availability()
                if not is_available:
                    logger.warning(f"Model {service_name} registered but not available")
            
        except Exception as e:
            logger.warning(f"Failed to test model availability for {service_name}: {e}")
        
        return version_id
    
    async def predict_with_best_model(self, task, task_type: str = "classification",
                                    requirements: Optional[Dict[str, Any]] = None) -> Prediction:
        """
        Make prediction using the best available model for the task.
        
        Args:
            task: The annotation task
            task_type: Type of annotation task
            requirements: Optional requirements for model selection
            
        Returns:
            Prediction from the selected model
        """
        # Get available models
        available_configs = list(self.model_manager.model_configs.values())
        
        if not available_configs:
            raise ValueError("No models available for prediction")
        
        # Select optimal model
        selected_config = await self.auto_selector.select_optimal_model(
            task_type, available_configs, requirements
        )
        
        if not selected_config:
            # Fallback to first available model
            selected_config = available_configs[0]
            logger.warning(f"No optimal model found, using fallback: {selected_config.model_name}")
        
        # Get annotator and make prediction
        model_name = f"{selected_config.model_type.value}_{selected_config.model_name}"
        annotator = await self.model_manager.get_annotator(model_name)
        
        if not annotator:
            raise ValueError(f"Failed to get annotator for {model_name}")
        
        # Make prediction with performance tracking
        prediction = await self.model_manager.predict_with_performance_tracking(
            model_name, task, task_type
        )
        
        return prediction
    
    async def compare_models_for_task(self, task_type: str, 
                                    model_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compare models for a specific task type.
        
        Args:
            task_type: Type of task to compare models for
            model_names: Optional list of specific models to compare
            
        Returns:
            Comparison results with rankings and recommendations
        """
        # Get models to compare
        if model_names:
            configs = []
            for name in model_names:
                if name in self.model_manager.model_configs:
                    configs.append(self.model_manager.model_configs[name])
        else:
            configs = list(self.model_manager.model_configs.values())
        
        if not configs:
            return {"error": "No models available for comparison"}
        
        # Run benchmark comparison
        try:
            comparison_result = await self.benchmark_suite.benchmark_models(
                configs, task_type
            )
            
            return {
                "success": True,
                "task_type": task_type,
                "comparison_result": comparison_result.to_dict(),
                "recommendations": comparison_result.recommendations
            }
            
        except Exception as e:
            logger.error(f"Model comparison failed: {e}")
            return {"error": f"Comparison failed: {str(e)}"}
    
    async def get_model_performance_report(self, task_type: Optional[str] = None) -> str:
        """Get comprehensive performance report."""
        try:
            # Get performance summary
            perf_summary = self.performance_analyzer.get_performance_summary(task_type)
            
            # Get comparison history
            comparison_history = self.benchmark_suite.get_comparison_history(task_type)
            
            # Get model statistics
            model_stats = await self.model_manager.get_model_statistics()
            
            # Generate report
            report = []
            report.append("=" * 80)
            report.append("AI Model Integration Service - Performance Report")
            report.append("=" * 80)
            report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            if task_type:
                report.append(f"Task Type: {task_type}")
            
            report.append("")
            report.append("Service Statistics:")
            report.append(f"- Total Registered Models: {model_stats['total_registered_models']}")
            report.append(f"- Healthy Models: {model_stats['healthy_models']}")
            report.append(f"- Model Type Distribution: {model_stats['model_type_distribution']}")
            report.append("")
            
            report.append("Performance Summary:")
            report.append(f"- Total Models Analyzed: {perf_summary['total_models']}")
            report.append(f"- Total Predictions: {perf_summary['total_predictions']}")
            report.append(f"- Average Accuracy: {perf_summary['average_accuracy']:.3f}")
            report.append(f"- Average Response Time: {perf_summary['average_response_time']:.3f}s")
            report.append("")
            
            if comparison_history:
                report.append("Recent Comparisons:")
                for i, comp in enumerate(comparison_history[:3], 1):
                    report.append(f"{i}. {comp.task_type} - {comp.benchmark_date.strftime('%Y-%m-%d')}")
                    if comp.overall_ranking:
                        best_model, best_score = comp.overall_ranking[0]
                        report.append(f"   Best Model: {best_model} (Score: {best_score:.3f})")
                report.append("")
            
            if perf_summary['models']:
                report.append("Top Performing Models:")
                for i, model in enumerate(perf_summary['models'][:5], 1):
                    report.append(f"{i}. {model['model_type']}:{model['model_name']}")
                    report.append(f"   Overall Score: {model['overall_score']:.3f}")
                    report.append(f"   Sample Count: {model['sample_count']}")
            
            return "\n".join(report)
            
        except Exception as e:
            logger.error(f"Failed to generate performance report: {e}")
            return f"Error generating report: {str(e)}"
    
    async def health_check_all_models(self) -> Dict[str, Any]:
        """Perform comprehensive health check on all models."""
        try:
            # Get model health status
            health_status = await self.model_manager.health_check_all_models()
            
            # Calculate summary statistics
            total_models = len(health_status)
            healthy_models = sum(1 for status in health_status.values() if status.get("healthy", False))
            unhealthy_models = total_models - healthy_models
            
            # Group by model type
            type_health = {}
            for model_name, status in health_status.items():
                model_type = model_name.split('_')[0] if '_' in model_name else 'unknown'
                if model_type not in type_health:
                    type_health[model_type] = {"healthy": 0, "unhealthy": 0}
                
                if status.get("healthy", False):
                    type_health[model_type]["healthy"] += 1
                else:
                    type_health[model_type]["unhealthy"] += 1
            
            return {
                "summary": {
                    "total_models": total_models,
                    "healthy_models": healthy_models,
                    "unhealthy_models": unhealthy_models,
                    "health_percentage": (healthy_models / total_models * 100) if total_models > 0 else 0
                },
                "by_type": type_health,
                "detailed_status": health_status,
                "check_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"error": f"Health check failed: {str(e)}"}
    
    async def optimize_model_selection(self, task_type: str) -> Dict[str, Any]:
        """
        Optimize model selection for a specific task type.
        
        Args:
            task_type: Type of task to optimize for
            
        Returns:
            Optimization results and recommendations
        """
        try:
            # Run optimization analysis
            optimization_result = await self.model_manager.optimize_model_selection(
                task_type, PerformanceMetric.ACCURACY
            )
            
            # Get current best model
            available_configs = list(self.model_manager.model_configs.values())
            best_config = await self.auto_selector.select_optimal_model(
                task_type, available_configs
            )
            
            # Get selection explanation
            explanation = {}
            if best_config:
                explanation = self.auto_selector.get_selection_explanation(
                    best_config, task_type
                )
            
            return {
                "success": True,
                "task_type": task_type,
                "optimization_result": optimization_result,
                "current_best_model": explanation,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Model optimization failed: {e}")
            return {"error": f"Optimization failed: {str(e)}"}
    
    async def get_integration_status(self) -> Dict[str, Any]:
        """Get overall integration service status."""
        try:
            # Get component status
            model_stats = await self.model_manager.get_model_statistics()
            health_status = await self.health_check_all_models()
            perf_summary = self.performance_analyzer.get_performance_summary()
            
            # Service uptime and configuration
            service_info = {
                "service_name": "AI Model Integration Service",
                "version": "1.0.0",
                "storage_path": str(self.storage_path),
                "configuration": self.config,
                "components": {
                    "model_manager": "active",
                    "performance_analyzer": "active",
                    "benchmark_suite": "active",
                    "auto_selector": "active"
                }
            }
            
            return {
                "service_info": service_info,
                "model_statistics": model_stats,
                "health_summary": health_status["summary"],
                "performance_summary": {
                    "total_models": perf_summary["total_models"],
                    "total_predictions": perf_summary["total_predictions"],
                    "average_accuracy": perf_summary["average_accuracy"]
                },
                "status": "operational",
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get integration status: {e}")
            return {
                "status": "error",
                "error": str(e),
                "last_updated": datetime.now().isoformat()
            }
    
    def update_service_config(self, new_config: Dict[str, Any]) -> None:
        """Update service configuration."""
        self.config.update(new_config)
        self._save_service_config()
        logger.info("Service configuration updated")
    
    async def cleanup_old_data(self, days_threshold: int = 30) -> Dict[str, Any]:
        """Clean up old performance data and benchmarks."""
        try:
            # Clean up inactive models
            cleanup_result = await self.model_manager.cleanup_inactive_models(days_threshold)
            
            # TODO: Add cleanup for old benchmark results and performance data
            
            return {
                "success": True,
                "cleanup_result": cleanup_result,
                "cleanup_date": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return {"error": f"Cleanup failed: {str(e)}"}


# Global service instance
_integration_service: Optional[AIModelIntegrationService] = None


def get_integration_service(storage_path: Optional[str] = None, 
                          redis_client=None) -> AIModelIntegrationService:
    """Get or create the global integration service instance."""
    global _integration_service
    
    if _integration_service is None:
        _integration_service = AIModelIntegrationService(storage_path, redis_client)
    
    return _integration_service


async def initialize_integration_service(storage_path: Optional[str] = None,
                                       redis_client=None) -> AIModelIntegrationService:
    """Initialize and return the integration service."""
    service = get_integration_service(storage_path, redis_client)
    
    # Ensure initialization is complete
    await service._initialize_default_models()
    
    return service