"""
Enhanced Model Manager with Performance Analysis and Auto-Selection.

Integrates model management with performance tracking and intelligent model selection.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from .base import AIAnnotator, ModelConfig, ModelType, Prediction
from .factory import AnnotatorFactory, ModelManager
from .model_performance import ModelPerformanceAnalyzer, ModelAutoSelector, PerformanceMetric
from .model_manager import ModelVersionManager, ModelVersion, ModelStatus


class EnhancedModelManager:
    """Enhanced model manager with performance analysis and auto-selection capabilities."""
    
    def __init__(self, storage_path: Optional[str] = None, redis_client=None):
        """Initialize enhanced model manager."""
        self.storage_path = Path(storage_path) if storage_path else Path("./enhanced_models")
        self.storage_path.mkdir(exist_ok=True)
        
        # Initialize components
        self.base_manager = ModelManager()
        self.version_manager = ModelVersionManager(redis_client, str(self.storage_path / "versions"))
        self.performance_analyzer = ModelPerformanceAnalyzer(str(self.storage_path / "performance"))
        self.auto_selector = ModelAutoSelector(self.performance_analyzer)
        
        # Model configurations and instances
        self.model_configs: Dict[str, ModelConfig] = {}
        self.active_annotators: Dict[str, AIAnnotator] = {}
        
        # Load configurations
        self._load_model_configs()
    
    def _load_model_configs(self) -> None:
        """Load model configurations from storage."""
        try:
            config_file = self.storage_path / "model_configs.json"
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for name, config_data in data.items():
                    config = ModelConfig(**config_data)
                    self.model_configs[name] = config
                    
        except Exception as e:
            print(f"Failed to load model configurations: {e}")
    
    def _save_model_configs(self) -> None:
        """Save model configurations to storage."""
        try:
            config_file = self.storage_path / "model_configs.json"
            data = {name: config.dict() for name, config in self.model_configs.items()}
            
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"Failed to save model configurations: {e}")
    
    async def register_model(self, name: str, config: ModelConfig, 
                           description: str = "", tags: Optional[set] = None,
                           auto_activate: bool = False) -> str:
        """
        Register a new model with version management.
        
        Args:
            name: Unique name for the model
            config: Model configuration
            description: Description of the model
            tags: Optional tags for categorization
            auto_activate: Whether to automatically activate the model
            
        Returns:
            Version ID of the registered model
        """
        # Store configuration
        self.model_configs[name] = config
        self._save_model_configs()
        
        # Register version
        version_id = await self.version_manager.register_model_version(
            model_type=config.model_type,
            model_name=config.model_name,
            version="1.0.0",
            config=config,
            description=description,
            tags=tags or set(),
            status=ModelStatus.ACTIVE if auto_activate else ModelStatus.EXPERIMENTAL
        )
        
        # Add to base manager
        self.base_manager.add_annotator(name, config)
        
        return version_id
    
    async def get_annotator(self, name: str, auto_create: bool = True) -> Optional[AIAnnotator]:
        """
        Get an annotator instance, creating it if necessary.
        
        Args:
            name: Name of the annotator
            auto_create: Whether to create the annotator if not cached
            
        Returns:
            AI annotator instance or None if not found
        """
        # Check cache first
        if name in self.active_annotators:
            return self.active_annotators[name]
        
        # Get from base manager
        annotator = self.base_manager.get_annotator(name)
        if annotator:
            self.active_annotators[name] = annotator
            return annotator
        
        # Auto-create if configuration exists
        if auto_create and name in self.model_configs:
            config = self.model_configs[name]
            try:
                annotator = AnnotatorFactory.create_annotator(config)
                self.active_annotators[name] = annotator
                self.base_manager.add_annotator(name, config)
                return annotator
            except Exception as e:
                print(f"Failed to create annotator {name}: {e}")
        
        return None
    
    async def predict_with_performance_tracking(self, annotator_name: str, task, 
                                              task_type: str = "classification",
                                              ground_truth: Optional[Dict[str, Any]] = None) -> Prediction:
        """
        Make prediction and track performance metrics.
        
        Args:
            annotator_name: Name of the annotator to use
            task: The annotation task
            task_type: Type of annotation task
            ground_truth: Optional ground truth for accuracy calculation
            
        Returns:
            Prediction result
        """
        annotator = await self.get_annotator(annotator_name)
        if not annotator:
            raise ValueError(f"Annotator {annotator_name} not found")
        
        # Make prediction
        prediction = await annotator.predict(task)
        
        # Evaluate correctness if ground truth provided
        is_correct = None
        if ground_truth:
            is_correct = self._evaluate_prediction(prediction.prediction_data, ground_truth, task_type)
        
        # Record performance
        self.performance_analyzer.record_prediction(prediction, task_type, is_correct, ground_truth)
        
        return prediction
    
    def _evaluate_prediction(self, prediction: Dict[str, Any], ground_truth: Dict[str, Any],
                           task_type: str) -> bool:
        """Evaluate prediction correctness against ground truth."""
        if task_type == "sentiment":
            return prediction.get("sentiment") == ground_truth.get("sentiment")
        elif task_type == "classification":
            return prediction.get("category") == ground_truth.get("category")
        elif task_type == "ner":
            pred_entities = {e.get("text", "") for e in prediction.get("entities", [])}
            true_entities = {e.get("text", "") for e in ground_truth.get("entities", [])}
            return pred_entities == true_entities
        else:
            return prediction.get("result") == ground_truth.get("result")
    
    async def auto_select_model(self, task_type: str, requirements: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Automatically select the best model for a task.
        
        Args:
            task_type: Type of annotation task
            requirements: Optional requirements for model selection
            
        Returns:
            Name of the selected model or None if no suitable model found
        """
        selected_config = self.auto_selector.select_best_model(task_type, requirements)
        if not selected_config:
            return None
        
        # Find model name by configuration
        for name, config in self.model_configs.items():
            if (config.model_type == selected_config.model_type and
                config.model_name == selected_config.model_name):
                return name
        
        # If not found, register the selected model
        model_name = f"{selected_config.model_type.value}_{selected_config.model_name}"
        await self.register_model(model_name, selected_config, auto_activate=True)
        return model_name
    
    async def benchmark_all_models(self, test_cases: List[Dict[str, Any]], 
                                 task_type: str = "classification") -> Dict[str, Any]:
        """
        Benchmark all registered models against test cases.
        
        Args:
            test_cases: List of test cases with input and expected output
            task_type: Type of annotation task
            
        Returns:
            Benchmark results
        """
        configs = list(self.model_configs.values())
        results = await self.performance_analyzer.benchmark_models(configs, test_cases, task_type)
        
        # Generate summary report
        summary = self.performance_analyzer.get_performance_summary(task_type)
        
        return {
            "benchmark_results": results,
            "summary": summary,
            "recommendations": self.auto_selector.get_model_recommendations(task_type)
        }
    
    async def get_model_recommendations(self, task_type: str, top_n: int = 3) -> List[Dict[str, Any]]:
        """Get model recommendations for a specific task type."""
        return self.auto_selector.get_model_recommendations(task_type, top_n)
    
    async def compare_models(self, task_type: str, 
                           metric: PerformanceMetric = PerformanceMetric.ACCURACY) -> List[Tuple[str, float]]:
        """Compare models for a specific task type and metric."""
        return self.performance_analyzer.compare_models(task_type, metric)
    
    async def get_performance_report(self, task_type: Optional[str] = None) -> str:
        """Get formatted performance report."""
        return self.performance_analyzer.export_performance_report(task_type)
    
    async def health_check_all_models(self) -> Dict[str, Dict[str, Any]]:
        """Perform health check on all registered models."""
        results = {}
        
        for name in self.model_configs.keys():
            try:
                annotator = await self.get_annotator(name)
                if annotator:
                    # Check if model has health check method
                    if hasattr(annotator, 'check_model_availability'):
                        is_healthy = await annotator.check_model_availability()
                    else:
                        is_healthy = True  # Assume healthy if no check method
                    
                    model_info = annotator.get_model_info()
                    results[name] = {
                        "healthy": is_healthy,
                        "model_info": model_info,
                        "last_checked": datetime.now().isoformat()
                    }
                else:
                    results[name] = {
                        "healthy": False,
                        "error": "Failed to create annotator",
                        "last_checked": datetime.now().isoformat()
                    }
            except Exception as e:
                results[name] = {
                    "healthy": False,
                    "error": str(e),
                    "last_checked": datetime.now().isoformat()
                }
        
        return results
    
    async def optimize_model_selection(self, task_type: str, 
                                     optimization_target: PerformanceMetric = PerformanceMetric.ACCURACY) -> Dict[str, Any]:
        """
        Optimize model selection for a specific task type and target metric.
        
        Args:
            task_type: Type of annotation task
            optimization_target: Metric to optimize for
            
        Returns:
            Optimization results and recommendations
        """
        # Get current performance data
        comparison = await self.compare_models(task_type, optimization_target)
        
        if not comparison:
            return {
                "status": "no_data",
                "message": "No performance data available for optimization",
                "recommendations": []
            }
        
        # Analyze performance gaps
        best_score = comparison[0][1] if comparison else 0
        recommendations = []
        
        for model_name, score in comparison[:5]:  # Top 5 models
            gap = best_score - score if optimization_target not in [
                PerformanceMetric.RESPONSE_TIME, PerformanceMetric.ERROR_RATE
            ] else score - best_score
            
            if gap > 0.1:  # Significant gap
                recommendations.append({
                    "model": model_name,
                    "current_score": score,
                    "gap_to_best": gap,
                    "recommendation": f"Consider tuning parameters or upgrading model"
                })
        
        # Get auto-selection recommendation
        auto_selected = await self.auto_select_model(task_type)
        
        return {
            "status": "success",
            "optimization_target": optimization_target.value,
            "best_model": comparison[0][0] if comparison else None,
            "best_score": best_score,
            "auto_selected_model": auto_selected,
            "performance_gaps": recommendations,
            "total_models_analyzed": len(comparison)
        }
    
    async def get_model_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about all models."""
        # Get version manager statistics
        version_stats = await self.version_manager.get_statistics()
        
        # Get performance statistics
        perf_summary = self.performance_analyzer.get_performance_summary()
        
        # Get health status
        health_status = await self.health_check_all_models()
        healthy_count = sum(1 for status in health_status.values() if status.get("healthy", False))
        
        # Model type distribution
        type_distribution = {}
        for config in self.model_configs.values():
            model_type = config.model_type.value
            type_distribution[model_type] = type_distribution.get(model_type, 0) + 1
        
        return {
            "total_registered_models": len(self.model_configs),
            "healthy_models": healthy_count,
            "model_type_distribution": type_distribution,
            "version_statistics": version_stats,
            "performance_summary": perf_summary,
            "last_updated": datetime.now().isoformat()
        }
    
    async def cleanup_inactive_models(self, days_threshold: int = 30) -> Dict[str, Any]:
        """Clean up models that haven't been used recently."""
        cutoff_date = datetime.now() - timedelta(days=days_threshold)
        
        inactive_models = []
        cleaned_count = 0
        
        # Check performance data for last usage
        for name, config in list(self.model_configs.items()):
            model_key = f"{config.model_type.value}:{config.model_name}:*"
            
            # Check if model has recent activity
            has_recent_activity = False
            for perf_data in self.performance_analyzer.performance_data.values():
                if (perf_data.model_name == config.model_name and
                    perf_data.model_type == config.model_type and
                    perf_data.last_updated > cutoff_date):
                    has_recent_activity = True
                    break
            
            if not has_recent_activity:
                inactive_models.append(name)
                
                # Remove from active annotators
                if name in self.active_annotators:
                    del self.active_annotators[name]
                
                # Remove from base manager
                self.base_manager.remove_annotator(name)
                
                cleaned_count += 1
        
        return {
            "cleaned_models": inactive_models,
            "cleaned_count": cleaned_count,
            "threshold_days": days_threshold,
            "cleanup_date": datetime.now().isoformat()
        }