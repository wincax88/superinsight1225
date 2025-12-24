"""
Model Performance Analysis and Comparison for SuperInsight platform.

Provides tools for comparing AI model performance and selecting optimal models.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from uuid import uuid4
from dataclasses import dataclass, field
from enum import Enum
import statistics
from pathlib import Path

from .base import AIAnnotator, ModelConfig, ModelType, Prediction
from .factory import AnnotatorFactory


class PerformanceMetric(str, Enum):
    """Performance metrics for model evaluation."""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    CONFIDENCE = "confidence"
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    COST_EFFICIENCY = "cost_efficiency"


@dataclass
class ModelPerformanceData:
    """Performance data for a specific model."""
    
    model_name: str
    model_type: ModelType
    task_type: str
    metrics: Dict[PerformanceMetric, float] = field(default_factory=dict)
    sample_count: int = 0
    total_processing_time: float = 0.0
    error_count: int = 0
    last_updated: datetime = field(default_factory=datetime.now)
    
    def add_prediction_result(self, prediction: Prediction, is_correct: Optional[bool] = None, 
                            ground_truth: Optional[Dict[str, Any]] = None) -> None:
        """Add a prediction result to update performance metrics."""
        self.sample_count += 1
        self.total_processing_time += prediction.processing_time
        
        # Update confidence metrics
        if PerformanceMetric.CONFIDENCE not in self.metrics:
            self.metrics[PerformanceMetric.CONFIDENCE] = 0.0
        
        # Running average of confidence
        current_confidence = self.metrics[PerformanceMetric.CONFIDENCE]
        self.metrics[PerformanceMetric.CONFIDENCE] = (
            (current_confidence * (self.sample_count - 1) + prediction.confidence) / self.sample_count
        )
        
        # Update response time
        self.metrics[PerformanceMetric.RESPONSE_TIME] = self.total_processing_time / self.sample_count
        
        # Update throughput (predictions per second)
        if self.total_processing_time > 0:
            self.metrics[PerformanceMetric.THROUGHPUT] = self.sample_count / self.total_processing_time
        
        # Update error rate
        if "error" in prediction.prediction_data:
            self.error_count += 1
        self.metrics[PerformanceMetric.ERROR_RATE] = self.error_count / self.sample_count
        
        # Update accuracy if ground truth is provided
        if is_correct is not None:
            if PerformanceMetric.ACCURACY not in self.metrics:
                self.metrics[PerformanceMetric.ACCURACY] = 0.0
            
            current_accuracy = self.metrics[PerformanceMetric.ACCURACY]
            correct_count = current_accuracy * (self.sample_count - 1) + (1 if is_correct else 0)
            self.metrics[PerformanceMetric.ACCURACY] = correct_count / self.sample_count
        
        self.last_updated = datetime.now()
    
    def get_overall_score(self, weights: Optional[Dict[PerformanceMetric, float]] = None) -> float:
        """Calculate overall performance score based on weighted metrics."""
        if not weights:
            weights = {
                PerformanceMetric.ACCURACY: 0.3,
                PerformanceMetric.CONFIDENCE: 0.2,
                PerformanceMetric.RESPONSE_TIME: 0.2,  # Lower is better
                PerformanceMetric.ERROR_RATE: 0.15,    # Lower is better
                PerformanceMetric.THROUGHPUT: 0.15
            }
        
        score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in self.metrics:
                value = self.metrics[metric]
                
                # Normalize metrics (higher is better for score)
                if metric in [PerformanceMetric.RESPONSE_TIME, PerformanceMetric.ERROR_RATE]:
                    # For metrics where lower is better, invert the score
                    normalized_value = max(0, 1 - value) if value <= 1 else 1 / (1 + value)
                else:
                    # For metrics where higher is better
                    normalized_value = min(1.0, value)
                
                score += normalized_value * weight
                total_weight += weight
        
        return score / total_weight if total_weight > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "model_type": self.model_type.value,
            "task_type": self.task_type,
            "metrics": {k.value: v for k, v in self.metrics.items()},
            "sample_count": self.sample_count,
            "total_processing_time": self.total_processing_time,
            "error_count": self.error_count,
            "last_updated": self.last_updated.isoformat(),
            "overall_score": self.get_overall_score()
        }


class ModelPerformanceAnalyzer:
    """Analyzer for comparing and evaluating AI model performance."""
    
    def __init__(self, storage_path: Optional[str] = None):
        """Initialize performance analyzer."""
        self.storage_path = Path(storage_path) if storage_path else Path("./model_performance")
        self.storage_path.mkdir(exist_ok=True)
        
        # Performance data storage
        self.performance_data: Dict[str, ModelPerformanceData] = {}
        
        # Load existing data
        self._load_performance_data()
    
    def _get_model_key(self, model_name: str, model_type: ModelType, task_type: str) -> str:
        """Generate unique key for model performance data."""
        return f"{model_type.value}:{model_name}:{task_type}"
    
    def _load_performance_data(self) -> None:
        """Load performance data from storage."""
        try:
            data_file = self.storage_path / "performance_data.json"
            if data_file.exists():
                with open(data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for key, perf_data in data.items():
                    # Convert back to ModelPerformanceData
                    perf_obj = ModelPerformanceData(
                        model_name=perf_data["model_name"],
                        model_type=ModelType(perf_data["model_type"]),
                        task_type=perf_data["task_type"],
                        sample_count=perf_data["sample_count"],
                        total_processing_time=perf_data["total_processing_time"],
                        error_count=perf_data["error_count"],
                        last_updated=datetime.fromisoformat(perf_data["last_updated"])
                    )
                    
                    # Convert metrics back
                    for metric_name, value in perf_data["metrics"].items():
                        perf_obj.metrics[PerformanceMetric(metric_name)] = value
                    
                    self.performance_data[key] = perf_obj
                    
        except Exception as e:
            print(f"Failed to load performance data: {e}")
    
    def _save_performance_data(self) -> None:
        """Save performance data to storage."""
        try:
            data_file = self.storage_path / "performance_data.json"
            data = {key: perf.to_dict() for key, perf in self.performance_data.items()}
            
            with open(data_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"Failed to save performance data: {e}")
    
    def record_prediction(self, prediction: Prediction, task_type: str, 
                         is_correct: Optional[bool] = None,
                         ground_truth: Optional[Dict[str, Any]] = None) -> None:
        """Record a prediction result for performance analysis."""
        model_key = self._get_model_key(
            prediction.ai_model_config.model_name,
            prediction.ai_model_config.model_type,
            task_type
        )
        
        # Get or create performance data
        if model_key not in self.performance_data:
            self.performance_data[model_key] = ModelPerformanceData(
                model_name=prediction.ai_model_config.model_name,
                model_type=prediction.ai_model_config.model_type,
                task_type=task_type
            )
        
        # Update performance data
        self.performance_data[model_key].add_prediction_result(
            prediction, is_correct, ground_truth
        )
        
        # Save updated data
        self._save_performance_data()
    
    def get_model_performance(self, model_name: str, model_type: ModelType, 
                            task_type: str) -> Optional[ModelPerformanceData]:
        """Get performance data for a specific model."""
        model_key = self._get_model_key(model_name, model_type, task_type)
        return self.performance_data.get(model_key)
    
    def compare_models(self, task_type: str, 
                      metric: PerformanceMetric = PerformanceMetric.ACCURACY) -> List[Tuple[str, float]]:
        """Compare models for a specific task type and metric."""
        results = []
        
        for key, perf_data in self.performance_data.items():
            if perf_data.task_type == task_type and metric in perf_data.metrics:
                model_identifier = f"{perf_data.model_type.value}:{perf_data.model_name}"
                results.append((model_identifier, perf_data.metrics[metric]))
        
        # Sort by metric value (descending for most metrics, ascending for time/error)
        reverse_sort = metric not in [PerformanceMetric.RESPONSE_TIME, PerformanceMetric.ERROR_RATE]
        results.sort(key=lambda x: x[1], reverse=reverse_sort)
        
        return results
    
    def get_best_model(self, task_type: str, 
                      metric: PerformanceMetric = PerformanceMetric.ACCURACY,
                      min_samples: int = 10) -> Optional[Tuple[str, ModelType, float]]:
        """Get the best performing model for a task type."""
        candidates = []
        
        for perf_data in self.performance_data.values():
            if (perf_data.task_type == task_type and 
                perf_data.sample_count >= min_samples and
                metric in perf_data.metrics):
                
                candidates.append((
                    perf_data.model_name,
                    perf_data.model_type,
                    perf_data.metrics[metric]
                ))
        
        if not candidates:
            return None
        
        # Sort and return best
        reverse_sort = metric not in [PerformanceMetric.RESPONSE_TIME, PerformanceMetric.ERROR_RATE]
        candidates.sort(key=lambda x: x[2], reverse=reverse_sort)
        
        return candidates[0]
    
    def get_performance_summary(self, task_type: Optional[str] = None) -> Dict[str, Any]:
        """Get performance summary for all models or specific task type."""
        summary = {
            "total_models": 0,
            "total_predictions": 0,
            "average_accuracy": 0.0,
            "average_response_time": 0.0,
            "models": []
        }
        
        filtered_data = []
        for perf_data in self.performance_data.values():
            if task_type is None or perf_data.task_type == task_type:
                filtered_data.append(perf_data)
        
        if not filtered_data:
            return summary
        
        summary["total_models"] = len(filtered_data)
        summary["total_predictions"] = sum(p.sample_count for p in filtered_data)
        
        # Calculate averages
        accuracies = [p.metrics.get(PerformanceMetric.ACCURACY, 0) for p in filtered_data 
                     if PerformanceMetric.ACCURACY in p.metrics]
        if accuracies:
            summary["average_accuracy"] = statistics.mean(accuracies)
        
        response_times = [p.metrics.get(PerformanceMetric.RESPONSE_TIME, 0) for p in filtered_data
                         if PerformanceMetric.RESPONSE_TIME in p.metrics]
        if response_times:
            summary["average_response_time"] = statistics.mean(response_times)
        
        # Add individual model summaries
        for perf_data in filtered_data:
            summary["models"].append({
                "model_name": perf_data.model_name,
                "model_type": perf_data.model_type.value,
                "task_type": perf_data.task_type,
                "sample_count": perf_data.sample_count,
                "overall_score": perf_data.get_overall_score(),
                "metrics": {k.value: v for k, v in perf_data.metrics.items()}
            })
        
        # Sort models by overall score
        summary["models"].sort(key=lambda x: x["overall_score"], reverse=True)
        
        return summary
    
    async def benchmark_models(self, models: List[ModelConfig], test_cases: List[Dict[str, Any]],
                              task_type: str = "classification") -> Dict[str, ModelPerformanceData]:
        """Benchmark multiple models against test cases."""
        results = {}
        
        for model_config in models:
            try:
                # Create annotator
                annotator = AnnotatorFactory.create_annotator(model_config)
                
                model_key = self._get_model_key(
                    model_config.model_name,
                    model_config.model_type,
                    task_type
                )
                
                # Initialize performance data
                if model_key not in self.performance_data:
                    self.performance_data[model_key] = ModelPerformanceData(
                        model_name=model_config.model_name,
                        model_type=model_config.model_type,
                        task_type=task_type
                    )
                
                perf_data = self.performance_data[model_key]
                
                # Run test cases
                for test_case in test_cases:
                    try:
                        # Create mock task
                        from uuid import uuid4
                        mock_task = type('Task', (), {
                            'id': uuid4(),
                            'project_id': test_case.get('project_id', 'benchmark')
                        })()
                        
                        # Get prediction
                        prediction = await annotator.predict(mock_task)
                        
                        # Check correctness if ground truth provided
                        is_correct = None
                        if 'expected' in test_case:
                            is_correct = self._evaluate_prediction(
                                prediction.prediction_data,
                                test_case['expected'],
                                task_type
                            )
                        
                        # Record result
                        perf_data.add_prediction_result(prediction, is_correct)
                        
                    except Exception as e:
                        print(f"Test case failed for {model_config.model_name}: {e}")
                        continue
                
                results[model_key] = perf_data
                
            except Exception as e:
                print(f"Failed to benchmark {model_config.model_name}: {e}")
                continue
        
        # Save results
        self._save_performance_data()
        
        return results
    
    def _evaluate_prediction(self, prediction: Dict[str, Any], expected: Dict[str, Any],
                           task_type: str) -> bool:
        """Evaluate if prediction matches expected result."""
        if task_type == "sentiment":
            return prediction.get("sentiment") == expected.get("sentiment")
        elif task_type == "classification":
            return prediction.get("category") == expected.get("category")
        elif task_type == "ner":
            # Simple entity matching
            pred_entities = set(e.get("text", "") for e in prediction.get("entities", []))
            exp_entities = set(e.get("text", "") for e in expected.get("entities", []))
            return pred_entities == exp_entities
        else:
            # Generic comparison
            return prediction.get("result") == expected.get("result")
    
    def export_performance_report(self, task_type: Optional[str] = None) -> str:
        """Export performance report as formatted text."""
        summary = self.get_performance_summary(task_type)
        
        report = []
        report.append("=" * 60)
        report.append("AI Model Performance Report")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if task_type:
            report.append(f"Task Type: {task_type}")
        
        report.append(f"Total Models: {summary['total_models']}")
        report.append(f"Total Predictions: {summary['total_predictions']}")
        report.append(f"Average Accuracy: {summary['average_accuracy']:.3f}")
        report.append(f"Average Response Time: {summary['average_response_time']:.3f}s")
        report.append("")
        
        report.append("Model Rankings:")
        report.append("-" * 40)
        
        for i, model in enumerate(summary['models'], 1):
            report.append(f"{i}. {model['model_type']}:{model['model_name']}")
            report.append(f"   Overall Score: {model['overall_score']:.3f}")
            report.append(f"   Samples: {model['sample_count']}")
            
            for metric, value in model['metrics'].items():
                report.append(f"   {metric}: {value:.3f}")
            report.append("")
        
        return "\n".join(report)


class ModelAutoSelector:
    """Automatic model selection based on performance and requirements."""
    
    def __init__(self, performance_analyzer: ModelPerformanceAnalyzer):
        """Initialize auto selector with performance analyzer."""
        self.analyzer = performance_analyzer
        
        # Default selection criteria weights
        self.default_weights = {
            PerformanceMetric.ACCURACY: 0.4,
            PerformanceMetric.CONFIDENCE: 0.2,
            PerformanceMetric.RESPONSE_TIME: 0.2,
            PerformanceMetric.ERROR_RATE: 0.1,
            PerformanceMetric.COST_EFFICIENCY: 0.1
        }
    
    def select_best_model(self, task_type: str, requirements: Optional[Dict[str, Any]] = None) -> Optional[ModelConfig]:
        """
        Select the best model for a task based on performance and requirements.
        
        Args:
            task_type: Type of annotation task
            requirements: Optional requirements dict with keys like:
                - max_response_time: Maximum acceptable response time
                - min_accuracy: Minimum required accuracy
                - preferred_models: List of preferred model types
                - budget_constraint: Cost constraint
        
        Returns:
            ModelConfig for the selected model or None if no suitable model found
        """
        if not requirements:
            requirements = {}
        
        candidates = []
        
        # Filter models based on requirements
        for perf_data in self.analyzer.performance_data.values():
            if perf_data.task_type != task_type:
                continue
            
            # Check minimum sample requirement
            if perf_data.sample_count < requirements.get('min_samples', 5):
                continue
            
            # Check response time requirement
            max_response_time = requirements.get('max_response_time')
            if (max_response_time and 
                PerformanceMetric.RESPONSE_TIME in perf_data.metrics and
                perf_data.metrics[PerformanceMetric.RESPONSE_TIME] > max_response_time):
                continue
            
            # Check accuracy requirement
            min_accuracy = requirements.get('min_accuracy')
            if (min_accuracy and 
                PerformanceMetric.ACCURACY in perf_data.metrics and
                perf_data.metrics[PerformanceMetric.ACCURACY] < min_accuracy):
                continue
            
            # Check preferred models
            preferred_models = requirements.get('preferred_models', [])
            if preferred_models and perf_data.model_type not in preferred_models:
                continue
            
            # Calculate weighted score
            weights = requirements.get('weights', self.default_weights)
            score = perf_data.get_overall_score(weights)
            
            candidates.append((perf_data, score))
        
        if not candidates:
            return None
        
        # Sort by score and select best
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_perf_data = candidates[0][0]
        
        # Create ModelConfig for the best model
        return ModelConfig(
            model_type=best_perf_data.model_type,
            model_name=best_perf_data.model_name,
            # Use default values for other parameters
            max_tokens=1000,
            temperature=0.7,
            timeout=30
        )
    
    def get_model_recommendations(self, task_type: str, top_n: int = 3) -> List[Dict[str, Any]]:
        """Get top N model recommendations for a task type."""
        recommendations = []
        
        for perf_data in self.analyzer.performance_data.values():
            if perf_data.task_type == task_type and perf_data.sample_count >= 5:
                score = perf_data.get_overall_score(self.default_weights)
                
                recommendations.append({
                    "model_name": perf_data.model_name,
                    "model_type": perf_data.model_type.value,
                    "overall_score": score,
                    "accuracy": perf_data.metrics.get(PerformanceMetric.ACCURACY, 0),
                    "response_time": perf_data.metrics.get(PerformanceMetric.RESPONSE_TIME, 0),
                    "confidence": perf_data.metrics.get(PerformanceMetric.CONFIDENCE, 0),
                    "sample_count": perf_data.sample_count,
                    "recommendation_reason": self._get_recommendation_reason(perf_data)
                })
        
        # Sort by overall score
        recommendations.sort(key=lambda x: x["overall_score"], reverse=True)
        
        return recommendations[:top_n]
    
    def _get_recommendation_reason(self, perf_data: ModelPerformanceData) -> str:
        """Generate recommendation reason based on model strengths."""
        reasons = []
        
        accuracy = perf_data.metrics.get(PerformanceMetric.ACCURACY, 0)
        response_time = perf_data.metrics.get(PerformanceMetric.RESPONSE_TIME, 0)
        confidence = perf_data.metrics.get(PerformanceMetric.CONFIDENCE, 0)
        
        if accuracy > 0.9:
            reasons.append("高准确率")
        elif accuracy > 0.8:
            reasons.append("良好准确率")
        
        if response_time < 1.0:
            reasons.append("快速响应")
        elif response_time < 3.0:
            reasons.append("适中响应时间")
        
        if confidence > 0.8:
            reasons.append("高置信度")
        
        if perf_data.sample_count > 100:
            reasons.append("充分测试")
        
        return "、".join(reasons) if reasons else "基础性能"