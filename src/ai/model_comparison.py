"""
Advanced Model Comparison and Benchmarking System for SuperInsight platform.

Provides comprehensive tools for comparing AI model performance across different metrics
and automatically selecting optimal models for specific tasks.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Set
from uuid import uuid4
from dataclasses import dataclass, field
from enum import Enum
import statistics
from pathlib import Path
import numpy as np
from collections import defaultdict

from .base import AIAnnotator, ModelConfig, ModelType, Prediction
from .factory import AnnotatorFactory
from .model_performance import ModelPerformanceAnalyzer, PerformanceMetric, ModelPerformanceData
from .enhanced_model_manager import EnhancedModelManager


class ComparisonMetric(str, Enum):
    """Metrics for model comparison."""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    CONFIDENCE = "confidence"
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    COST_EFFICIENCY = "cost_efficiency"
    CONSISTENCY = "consistency"
    ROBUSTNESS = "robustness"


@dataclass
class BenchmarkTask:
    """A single benchmark task with input and expected output."""
    
    id: str
    task_type: str
    input_text: str
    expected_output: Dict[str, Any]
    difficulty_level: str = "medium"  # easy, medium, hard
    domain: str = "general"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelComparisonResult:
    """Result of comparing multiple models."""
    
    task_type: str
    models_compared: List[str]
    metrics: Dict[str, Dict[str, float]]  # metric -> model -> score
    rankings: Dict[str, List[Tuple[str, float]]]  # metric -> [(model, score), ...]
    overall_ranking: List[Tuple[str, float]]
    recommendations: List[Dict[str, Any]]
    benchmark_date: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "task_type": self.task_type,
            "models_compared": self.models_compared,
            "metrics": self.metrics,
            "rankings": self.rankings,
            "overall_ranking": self.overall_ranking,
            "recommendations": self.recommendations,
            "benchmark_date": self.benchmark_date.isoformat()
        }


class ModelBenchmarkSuite:
    """Comprehensive benchmarking suite for AI models."""
    
    def __init__(self, storage_path: Optional[str] = None):
        """Initialize benchmark suite."""
        self.storage_path = Path(storage_path) if storage_path else Path("./model_benchmarks")
        self.storage_path.mkdir(exist_ok=True)
        
        # Load benchmark tasks
        self.benchmark_tasks: Dict[str, List[BenchmarkTask]] = {}
        self._load_benchmark_tasks()
        
        # Results storage
        self.comparison_results: Dict[str, ModelComparisonResult] = {}
        self._load_comparison_results()
    
    def _load_benchmark_tasks(self) -> None:
        """Load benchmark tasks from storage."""
        try:
            tasks_file = self.storage_path / "benchmark_tasks.json"
            if tasks_file.exists():
                with open(tasks_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for task_type, tasks_data in data.items():
                    self.benchmark_tasks[task_type] = [
                        BenchmarkTask(**task_data) for task_data in tasks_data
                    ]
            else:
                # Create default tasks if file doesn't exist
                self._create_default_benchmark_tasks()
        except Exception as e:
            print(f"Failed to load benchmark tasks: {e}")
            self._create_default_benchmark_tasks()
    
    def _create_default_benchmark_tasks(self) -> None:
        """Create default benchmark tasks for testing."""
        # Sentiment analysis tasks
        sentiment_tasks = [
            BenchmarkTask(
                id="sentiment_1",
                task_type="sentiment",
                input_text="这个产品真的很棒，我非常满意！",
                expected_output={"sentiment": "positive", "confidence": 0.9},
                difficulty_level="easy",
                domain="product_review"
            ),
            BenchmarkTask(
                id="sentiment_2",
                task_type="sentiment",
                input_text="服务态度一般，还有改进空间。",
                expected_output={"sentiment": "neutral", "confidence": 0.7},
                difficulty_level="medium",
                domain="service_review"
            ),
            BenchmarkTask(
                id="sentiment_3",
                task_type="sentiment",
                input_text="虽然价格有点贵，但是质量确实不错，总体来说还是值得的。",
                expected_output={"sentiment": "positive", "confidence": 0.6},
                difficulty_level="hard",
                domain="product_review"
            )
        ]
        
        # Classification tasks
        classification_tasks = [
            BenchmarkTask(
                id="class_1",
                task_type="classification",
                input_text="请问如何申请退款？",
                expected_output={"category": "customer_service", "confidence": 0.8},
                difficulty_level="easy",
                domain="customer_support"
            ),
            BenchmarkTask(
                id="class_2",
                task_type="classification",
                input_text="系统登录时出现错误代码500",
                expected_output={"category": "technical_issue", "confidence": 0.9},
                difficulty_level="medium",
                domain="technical_support"
            )
        ]
        
        # NER tasks
        ner_tasks = [
            BenchmarkTask(
                id="ner_1",
                task_type="ner",
                input_text="张三在北京大学学习计算机科学。",
                expected_output={
                    "entities": [
                        {"text": "张三", "label": "PERSON", "start": 0, "end": 2},
                        {"text": "北京大学", "label": "ORG", "start": 3, "end": 7}
                    ],
                    "confidence": 0.8
                },
                difficulty_level="medium",
                domain="education"
            )
        ]
        
        self.benchmark_tasks = {
            "sentiment": sentiment_tasks,
            "classification": classification_tasks,
            "ner": ner_tasks
        }
        
        self._save_benchmark_tasks()
    
    def _save_benchmark_tasks(self) -> None:
        """Save benchmark tasks to storage."""
        try:
            tasks_file = self.storage_path / "benchmark_tasks.json"
            data = {}
            
            for task_type, tasks in self.benchmark_tasks.items():
                data[task_type] = [
                    {
                        "id": task.id,
                        "task_type": task.task_type,
                        "input_text": task.input_text,
                        "expected_output": task.expected_output,
                        "difficulty_level": task.difficulty_level,
                        "domain": task.domain,
                        "metadata": task.metadata
                    }
                    for task in tasks
                ]
            
            with open(tasks_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"Failed to save benchmark tasks: {e}")
    
    def _load_comparison_results(self) -> None:
        """Load comparison results from storage."""
        try:
            results_file = self.storage_path / "comparison_results.json"
            if results_file.exists():
                with open(results_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for key, result_data in data.items():
                    result = ModelComparisonResult(
                        task_type=result_data["task_type"],
                        models_compared=result_data["models_compared"],
                        metrics=result_data["metrics"],
                        rankings=result_data["rankings"],
                        overall_ranking=result_data["overall_ranking"],
                        recommendations=result_data["recommendations"],
                        benchmark_date=datetime.fromisoformat(result_data["benchmark_date"])
                    )
                    self.comparison_results[key] = result
                    
        except Exception as e:
            print(f"Failed to load comparison results: {e}")
    
    def _save_comparison_results(self) -> None:
        """Save comparison results to storage."""
        try:
            results_file = self.storage_path / "comparison_results.json"
            data = {key: result.to_dict() for key, result in self.comparison_results.items()}
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"Failed to save comparison results: {e}")
    
    async def benchmark_models(self, model_configs: List[ModelConfig], 
                             task_type: str = "sentiment") -> ModelComparisonResult:
        """
        Benchmark multiple models against test cases.
        
        Args:
            model_configs: List of model configurations to benchmark
            task_type: Type of task to benchmark
            
        Returns:
            Comparison result with rankings and recommendations
        """
        if task_type not in self.benchmark_tasks:
            raise ValueError(f"No benchmark tasks available for task type: {task_type}")
        
        tasks = self.benchmark_tasks[task_type]
        model_results = {}
        
        # Run benchmarks for each model
        for config in model_configs:
            try:
                annotator = AnnotatorFactory.create_annotator(config)
                model_name = f"{config.model_type.value}:{config.model_name}"
                
                results = await self._run_model_benchmark(annotator, tasks, task_type)
                model_results[model_name] = results
                
            except Exception as e:
                print(f"Failed to benchmark {config.model_name}: {e}")
                continue
        
        # Calculate comparison metrics
        comparison_result = self._calculate_comparison_metrics(
            model_results, task_type, list(model_results.keys())
        )
        
        # Store results
        result_key = f"{task_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.comparison_results[result_key] = comparison_result
        self._save_comparison_results()
        
        return comparison_result
    
    async def _run_model_benchmark(self, annotator: AIAnnotator, 
                                 tasks: List[BenchmarkTask],
                                 task_type: str) -> Dict[str, Any]:
        """Run benchmark for a single model."""
        results = {
            "predictions": [],
            "response_times": [],
            "accuracies": [],
            "confidences": [],
            "errors": 0
        }
        
        for task in tasks:
            try:
                # Create mock task object
                mock_task = type('Task', (), {
                    'id': uuid4(),
                    'project_id': f'{task_type}_benchmark'
                })()
                
                start_time = time.time()
                prediction = await annotator.predict(mock_task)
                response_time = time.time() - start_time
                
                # Evaluate accuracy
                accuracy = self._evaluate_prediction_accuracy(
                    prediction.prediction_data, 
                    task.expected_output, 
                    task_type
                )
                
                results["predictions"].append(prediction.prediction_data)
                results["response_times"].append(response_time)
                results["accuracies"].append(accuracy)
                results["confidences"].append(prediction.confidence)
                
            except Exception as e:
                print(f"Benchmark task {task.id} failed: {e}")
                results["errors"] += 1
                continue
        
        return results
    
    def _evaluate_prediction_accuracy(self, prediction: Dict[str, Any], 
                                    expected: Dict[str, Any], 
                                    task_type: str) -> float:
        """Evaluate prediction accuracy against expected output."""
        if task_type == "sentiment":
            pred_sentiment = prediction.get("sentiment", "").lower()
            exp_sentiment = expected.get("sentiment", "").lower()
            return 1.0 if pred_sentiment == exp_sentiment else 0.0
            
        elif task_type == "classification":
            pred_category = prediction.get("category", "").lower()
            exp_category = expected.get("category", "").lower()
            return 1.0 if pred_category == exp_category else 0.0
            
        elif task_type == "ner":
            pred_entities = set(
                e.get("text", "") for e in prediction.get("entities", [])
            )
            exp_entities = set(
                e.get("text", "") for e in expected.get("entities", [])
            )
            
            if not exp_entities:
                return 1.0 if not pred_entities else 0.0
            
            # Calculate F1 score for entity matching
            intersection = pred_entities & exp_entities
            precision = len(intersection) / len(pred_entities) if pred_entities else 0
            recall = len(intersection) / len(exp_entities) if exp_entities else 0
            
            if precision + recall == 0:
                return 0.0
            
            return 2 * (precision * recall) / (precision + recall)
        
        else:
            # Generic comparison
            return 1.0 if prediction.get("result") == expected.get("result") else 0.0
    
    def _calculate_comparison_metrics(self, model_results: Dict[str, Dict[str, Any]], 
                                    task_type: str, 
                                    model_names: List[str]) -> ModelComparisonResult:
        """Calculate comparison metrics from benchmark results."""
        metrics = {}
        rankings = {}
        
        # Calculate metrics for each model
        for metric in ComparisonMetric:
            metrics[metric.value] = {}
            
            for model_name in model_names:
                results = model_results.get(model_name, {})
                
                if metric == ComparisonMetric.ACCURACY:
                    accuracies = results.get("accuracies", [])
                    metrics[metric.value][model_name] = statistics.mean(accuracies) if accuracies else 0.0
                    
                elif metric == ComparisonMetric.CONFIDENCE:
                    confidences = results.get("confidences", [])
                    metrics[metric.value][model_name] = statistics.mean(confidences) if confidences else 0.0
                    
                elif metric == ComparisonMetric.RESPONSE_TIME:
                    times = results.get("response_times", [])
                    metrics[metric.value][model_name] = statistics.mean(times) if times else float('inf')
                    
                elif metric == ComparisonMetric.ERROR_RATE:
                    total_tasks = len(self.benchmark_tasks.get(task_type, []))
                    errors = results.get("errors", 0)
                    metrics[metric.value][model_name] = errors / total_tasks if total_tasks > 0 else 1.0
                    
                elif metric == ComparisonMetric.THROUGHPUT:
                    times = results.get("response_times", [])
                    avg_time = statistics.mean(times) if times else float('inf')
                    metrics[metric.value][model_name] = 1.0 / avg_time if avg_time > 0 else 0.0
                    
                elif metric == ComparisonMetric.CONSISTENCY:
                    confidences = results.get("confidences", [])
                    if len(confidences) > 1:
                        metrics[metric.value][model_name] = 1.0 - statistics.stdev(confidences)
                    else:
                        metrics[metric.value][model_name] = 1.0
        
        # Create rankings for each metric
        for metric, model_scores in metrics.items():
            # Sort by score (descending for most metrics, ascending for time/error)
            reverse_sort = metric not in ["response_time", "error_rate"]
            sorted_models = sorted(
                model_scores.items(), 
                key=lambda x: x[1], 
                reverse=reverse_sort
            )
            rankings[metric] = sorted_models
        
        # Calculate overall ranking (weighted average)
        overall_scores = {}
        weights = {
            "accuracy": 0.3,
            "confidence": 0.2,
            "response_time": 0.2,
            "error_rate": 0.15,
            "consistency": 0.15
        }
        
        for model_name in model_names:
            score = 0.0
            total_weight = 0.0
            
            for metric, weight in weights.items():
                if metric in metrics and model_name in metrics[metric]:
                    value = metrics[metric][model_name]
                    
                    # Normalize metrics (higher is better for overall score)
                    if metric in ["response_time", "error_rate"]:
                        normalized_value = max(0, 1 - value) if value <= 1 else 1 / (1 + value)
                    else:
                        normalized_value = min(1.0, value)
                    
                    score += normalized_value * weight
                    total_weight += weight
            
            overall_scores[model_name] = score / total_weight if total_weight > 0 else 0.0
        
        overall_ranking = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            rankings, overall_ranking, task_type
        )
        
        return ModelComparisonResult(
            task_type=task_type,
            models_compared=model_names,
            metrics=metrics,
            rankings=rankings,
            overall_ranking=overall_ranking,
            recommendations=recommendations
        )
    
    def _generate_recommendations(self, rankings: Dict[str, List[Tuple[str, float]]], 
                                overall_ranking: List[Tuple[str, float]],
                                task_type: str) -> List[Dict[str, Any]]:
        """Generate recommendations based on benchmark results."""
        recommendations = []
        
        if not overall_ranking:
            return recommendations
        
        # Best overall model
        best_model, best_score = overall_ranking[0]
        recommendations.append({
            "type": "best_overall",
            "model": best_model,
            "score": best_score,
            "reason": f"在{task_type}任务中综合表现最佳",
            "use_case": "推荐用于生产环境的主要模型"
        })
        
        # Best for accuracy
        if "accuracy" in rankings and rankings["accuracy"]:
            acc_model, acc_score = rankings["accuracy"][0]
            if acc_model != best_model:
                recommendations.append({
                    "type": "best_accuracy",
                    "model": acc_model,
                    "score": acc_score,
                    "reason": f"在{task_type}任务中准确率最高",
                    "use_case": "推荐用于对准确性要求极高的场景"
                })
        
        # Best for speed
        if "response_time" in rankings and rankings["response_time"]:
            speed_model, speed_time = rankings["response_time"][0]
            if speed_model != best_model:
                recommendations.append({
                    "type": "best_speed",
                    "model": speed_model,
                    "score": speed_time,
                    "reason": f"在{task_type}任务中响应速度最快",
                    "use_case": "推荐用于对响应时间要求严格的实时场景"
                })
        
        # Most consistent
        if "consistency" in rankings and rankings["consistency"]:
            consistent_model, consistency_score = rankings["consistency"][0]
            if consistent_model != best_model:
                recommendations.append({
                    "type": "most_consistent",
                    "model": consistent_model,
                    "score": consistency_score,
                    "reason": f"在{task_type}任务中表现最稳定",
                    "use_case": "推荐用于需要稳定输出的批量处理场景"
                })
        
        return recommendations
    
    def get_comparison_history(self, task_type: Optional[str] = None) -> List[ModelComparisonResult]:
        """Get historical comparison results."""
        results = list(self.comparison_results.values())
        
        if task_type:
            results = [r for r in results if r.task_type == task_type]
        
        # Sort by benchmark date (newest first)
        results.sort(key=lambda x: x.benchmark_date, reverse=True)
        
        return results
    
    def export_comparison_report(self, task_type: Optional[str] = None) -> str:
        """Export comparison report as formatted text."""
        results = self.get_comparison_history(task_type)
        
        if not results:
            return "No comparison results available."
        
        report = []
        report.append("=" * 80)
        report.append("AI Model Comparison Report")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if task_type:
            report.append(f"Task Type: {task_type}")
        
        report.append(f"Total Comparisons: {len(results)}")
        report.append("")
        
        # Latest comparison details
        if results:
            latest = results[0]
            report.append("Latest Comparison Results:")
            report.append("-" * 40)
            report.append(f"Task Type: {latest.task_type}")
            report.append(f"Date: {latest.benchmark_date.strftime('%Y-%m-%d %H:%M:%S')}")
            report.append(f"Models Compared: {len(latest.models_compared)}")
            report.append("")
            
            report.append("Overall Rankings:")
            for i, (model, score) in enumerate(latest.overall_ranking, 1):
                report.append(f"{i}. {model}: {score:.3f}")
            report.append("")
            
            report.append("Recommendations:")
            for rec in latest.recommendations:
                report.append(f"- {rec['type']}: {rec['model']}")
                report.append(f"  Reason: {rec['reason']}")
                report.append(f"  Use Case: {rec['use_case']}")
                report.append("")
        
        return "\n".join(report)


class ModelAutoSelector:
    """Enhanced automatic model selection with comparison-based intelligence."""
    
    def __init__(self, benchmark_suite: ModelBenchmarkSuite, 
                 performance_analyzer: ModelPerformanceAnalyzer):
        """Initialize auto selector."""
        self.benchmark_suite = benchmark_suite
        self.performance_analyzer = performance_analyzer
    
    async def select_optimal_model(self, task_type: str, 
                                 available_models: List[ModelConfig],
                                 requirements: Optional[Dict[str, Any]] = None) -> Optional[ModelConfig]:
        """
        Select optimal model based on benchmarks and requirements.
        
        Args:
            task_type: Type of task
            available_models: List of available model configurations
            requirements: Optional requirements (max_response_time, min_accuracy, etc.)
            
        Returns:
            Selected model configuration or None if no suitable model found
        """
        if not available_models:
            return None
        
        # Run fresh benchmark if needed
        comparison_results = self.benchmark_suite.get_comparison_history(task_type)
        
        if not comparison_results or self._needs_fresh_benchmark(comparison_results[0]):
            print(f"Running fresh benchmark for {task_type}...")
            comparison_result = await self.benchmark_suite.benchmark_models(
                available_models, task_type
            )
        else:
            comparison_result = comparison_results[0]
        
        # Apply requirements filtering
        if requirements:
            filtered_models = self._filter_models_by_requirements(
                comparison_result, requirements
            )
        else:
            filtered_models = comparison_result.overall_ranking
        
        if not filtered_models:
            return None
        
        # Return best model configuration
        best_model_name = filtered_models[0][0]
        
        # Find corresponding configuration
        for config in available_models:
            model_identifier = f"{config.model_type.value}:{config.model_name}"
            if model_identifier == best_model_name:
                return config
        
        return None
    
    def _needs_fresh_benchmark(self, latest_result: ModelComparisonResult) -> bool:
        """Check if a fresh benchmark is needed."""
        # Run fresh benchmark if latest is older than 24 hours
        age = datetime.now() - latest_result.benchmark_date
        return age > timedelta(hours=24)
    
    def _filter_models_by_requirements(self, comparison_result: ModelComparisonResult,
                                     requirements: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Filter models based on requirements."""
        filtered = []
        
        for model_name, overall_score in comparison_result.overall_ranking:
            meets_requirements = True
            
            # Check response time requirement
            if "max_response_time" in requirements:
                max_time = requirements["max_response_time"]
                if ("response_time" in comparison_result.metrics and
                    model_name in comparison_result.metrics["response_time"]):
                    model_time = comparison_result.metrics["response_time"][model_name]
                    if model_time > max_time:
                        meets_requirements = False
            
            # Check accuracy requirement
            if "min_accuracy" in requirements:
                min_acc = requirements["min_accuracy"]
                if ("accuracy" in comparison_result.metrics and
                    model_name in comparison_result.metrics["accuracy"]):
                    model_acc = comparison_result.metrics["accuracy"][model_name]
                    if model_acc < min_acc:
                        meets_requirements = False
            
            # Check error rate requirement
            if "max_error_rate" in requirements:
                max_error = requirements["max_error_rate"]
                if ("error_rate" in comparison_result.metrics and
                    model_name in comparison_result.metrics["error_rate"]):
                    model_error = comparison_result.metrics["error_rate"][model_name]
                    if model_error > max_error:
                        meets_requirements = False
            
            if meets_requirements:
                filtered.append((model_name, overall_score))
        
        return filtered
    
    def get_selection_explanation(self, selected_config: ModelConfig, 
                                task_type: str) -> Dict[str, Any]:
        """Get explanation for model selection."""
        comparison_results = self.benchmark_suite.get_comparison_history(task_type)
        
        if not comparison_results:
            return {"explanation": "No benchmark data available"}
        
        latest_result = comparison_results[0]
        model_name = f"{selected_config.model_type.value}:{selected_config.model_name}"
        
        # Find model in rankings
        model_rank = None
        model_score = None
        
        for i, (name, score) in enumerate(latest_result.overall_ranking):
            if name == model_name:
                model_rank = i + 1
                model_score = score
                break
        
        # Get model metrics
        model_metrics = {}
        for metric, model_scores in latest_result.metrics.items():
            if model_name in model_scores:
                model_metrics[metric] = model_scores[model_name]
        
        return {
            "selected_model": model_name,
            "overall_rank": model_rank,
            "overall_score": model_score,
            "metrics": model_metrics,
            "benchmark_date": latest_result.benchmark_date.isoformat(),
            "explanation": f"Selected based on comprehensive benchmark ranking #{model_rank} with score {model_score:.3f}"
        }