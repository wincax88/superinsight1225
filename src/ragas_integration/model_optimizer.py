"""
Model Comparison and Optimization Engine for Ragas Integration.

Implements comprehensive model comparison, performance benchmarking,
automatic model selection, and optimization recommendation generation.
"""

import asyncio
import json
import logging
import statistics
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Set
from uuid import uuid4
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import numpy as np
from collections import defaultdict

# Try to import Ragas, fall back gracefully if not available
try:
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        faithfulness,
        context_recall,
        context_precision,
        answer_correctness,
        answer_similarity
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Ragas not available: {e}. Model optimization will use basic metrics only.")
    RAGAS_AVAILABLE = False

from src.ai.base import AIAnnotator, ModelConfig, ModelType
from src.ai.factory import AnnotatorFactory
from src.ai.model_comparison import ModelBenchmarkSuite, ComparisonMetric


logger = logging.getLogger(__name__)


class OptimizationMetric(str, Enum):
    """Metrics for model optimization analysis."""
    RAGAS_FAITHFULNESS = "ragas_faithfulness"
    RAGAS_RELEVANCY = "ragas_relevancy"
    RAGAS_CONTEXT_PRECISION = "ragas_context_precision"
    RAGAS_CONTEXT_RECALL = "ragas_context_recall"
    RAGAS_CORRECTNESS = "ragas_correctness"
    RAGAS_SIMILARITY = "ragas_similarity"
    COMPOSITE_QUALITY = "composite_quality"
    IMPROVEMENT_POTENTIAL = "improvement_potential"
    CONSISTENCY_SCORE = "consistency_score"
    ROBUSTNESS_SCORE = "robustness_score"


@dataclass
class ModelPerformanceProfile:
    """Comprehensive performance profile for a model."""
    
    model_name: str
    model_type: str
    ragas_metrics: Dict[str, float] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    quality_trends: List[Dict[str, Any]] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    optimization_potential: float = 0.0
    benchmark_date: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "ragas_metrics": self.ragas_metrics,
            "performance_metrics": self.performance_metrics,
            "quality_trends": self.quality_trends,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "optimization_potential": self.optimization_potential,
            "benchmark_date": self.benchmark_date.isoformat()
        }


@dataclass
class OptimizationRecommendation:
    """Optimization recommendation for a model."""
    
    model_name: str
    recommendation_type: str
    priority: str  # high, medium, low
    description: str
    expected_improvement: float
    implementation_effort: str  # easy, medium, hard
    specific_actions: List[str] = field(default_factory=list)
    metrics_to_improve: List[str] = field(default_factory=list)
    estimated_timeline: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "recommendation_type": self.recommendation_type,
            "priority": self.priority,
            "description": self.description,
            "expected_improvement": self.expected_improvement,
            "implementation_effort": self.implementation_effort,
            "specific_actions": self.specific_actions,
            "metrics_to_improve": self.metrics_to_improve,
            "estimated_timeline": self.estimated_timeline
        }


@dataclass
class ModelComparisonReport:
    """Comprehensive model comparison report with Ragas integration."""
    
    comparison_id: str
    models_compared: List[str]
    ragas_comparison: Dict[str, Dict[str, float]] = field(default_factory=dict)
    performance_comparison: Dict[str, Dict[str, float]] = field(default_factory=dict)
    quality_rankings: Dict[str, List[Tuple[str, float]]] = field(default_factory=dict)
    overall_ranking: List[Tuple[str, float]] = field(default_factory=list)
    model_profiles: Dict[str, ModelPerformanceProfile] = field(default_factory=dict)
    optimization_recommendations: List[OptimizationRecommendation] = field(default_factory=list)
    selection_recommendation: Optional[str] = None
    comparison_date: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "comparison_id": self.comparison_id,
            "models_compared": self.models_compared,
            "ragas_comparison": self.ragas_comparison,
            "performance_comparison": self.performance_comparison,
            "quality_rankings": self.quality_rankings,
            "overall_ranking": self.overall_ranking,
            "model_profiles": {k: v.to_dict() for k, v in self.model_profiles.items()},
            "optimization_recommendations": [r.to_dict() for r in self.optimization_recommendations],
            "selection_recommendation": self.selection_recommendation,
            "comparison_date": self.comparison_date.isoformat()
        }


class ModelComparisonEngine:
    """Engine for comprehensive model comparison with Ragas integration."""
    
    def __init__(self, storage_path: Optional[str] = None):
        """Initialize comparison engine."""
        self.storage_path = Path(storage_path) if storage_path else Path("./ragas_model_comparisons")
        self.storage_path.mkdir(exist_ok=True)
        
        # Ragas metrics configuration
        self.ragas_metrics = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall,
            "answer_correctness": answer_correctness,
            "answer_similarity": answer_similarity
        }
        
        # Metric weights for composite scoring
        self.metric_weights = {
            "ragas_faithfulness": 0.25,
            "ragas_relevancy": 0.25,
            "ragas_context_precision": 0.15,
            "ragas_context_recall": 0.15,
            "ragas_correctness": 0.10,
            "ragas_similarity": 0.10
        }
        
        # Comparison history
        self.comparison_history: List[ModelComparisonReport] = []
        self._load_comparison_history()
    
    def _load_comparison_history(self) -> None:
        """Load comparison history from storage."""
        try:
            history_file = self.storage_path / "comparison_history.json"
            if history_file.exists():
                with open(history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for report_data in data:
                    # Reconstruct model profiles
                    model_profiles = {}
                    for model_name, profile_data in report_data.get("model_profiles", {}).items():
                        profile = ModelPerformanceProfile(
                            model_name=profile_data["model_name"],
                            model_type=profile_data["model_type"],
                            ragas_metrics=profile_data.get("ragas_metrics", {}),
                            performance_metrics=profile_data.get("performance_metrics", {}),
                            quality_trends=profile_data.get("quality_trends", []),
                            strengths=profile_data.get("strengths", []),
                            weaknesses=profile_data.get("weaknesses", []),
                            optimization_potential=profile_data.get("optimization_potential", 0.0),
                            benchmark_date=datetime.fromisoformat(profile_data["benchmark_date"])
                        )
                        model_profiles[model_name] = profile
                    
                    # Reconstruct optimization recommendations
                    recommendations = []
                    for rec_data in report_data.get("optimization_recommendations", []):
                        rec = OptimizationRecommendation(
                            model_name=rec_data["model_name"],
                            recommendation_type=rec_data["recommendation_type"],
                            priority=rec_data["priority"],
                            description=rec_data["description"],
                            expected_improvement=rec_data["expected_improvement"],
                            implementation_effort=rec_data["implementation_effort"],
                            specific_actions=rec_data.get("specific_actions", []),
                            metrics_to_improve=rec_data.get("metrics_to_improve", []),
                            estimated_timeline=rec_data.get("estimated_timeline", "")
                        )
                        recommendations.append(rec)
                    
                    # Create comparison report
                    report = ModelComparisonReport(
                        comparison_id=report_data["comparison_id"],
                        models_compared=report_data["models_compared"],
                        ragas_comparison=report_data.get("ragas_comparison", {}),
                        performance_comparison=report_data.get("performance_comparison", {}),
                        quality_rankings=report_data.get("quality_rankings", {}),
                        overall_ranking=report_data.get("overall_ranking", []),
                        model_profiles=model_profiles,
                        optimization_recommendations=recommendations,
                        selection_recommendation=report_data.get("selection_recommendation"),
                        comparison_date=datetime.fromisoformat(report_data["comparison_date"])
                    )
                    
                    self.comparison_history.append(report)
                    
        except Exception as e:
            logger.error(f"Failed to load comparison history: {e}")
    
    def _save_comparison_history(self) -> None:
        """Save comparison history to storage."""
        try:
            history_file = self.storage_path / "comparison_history.json"
            data = [report.to_dict() for report in self.comparison_history]
            
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Failed to save comparison history: {e}")
    
    async def compare_models_with_ragas(self, 
                                      model_configs: List[ModelConfig],
                                      evaluation_dataset: List[Dict[str, Any]],
                                      task_type: str = "rag_qa") -> ModelComparisonReport:
        """
        Compare multiple models using Ragas metrics.
        
        Args:
            model_configs: List of model configurations to compare
            evaluation_dataset: Dataset for evaluation (questions, answers, contexts, ground_truths)
            task_type: Type of task being evaluated
            
        Returns:
            Comprehensive comparison report
        """
        if not RAGAS_AVAILABLE:
            logger.warning("Ragas not available, falling back to basic comparison")
            return await self._basic_model_comparison(model_configs, evaluation_dataset, task_type)
        
        comparison_id = f"ragas_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Starting Ragas model comparison: {comparison_id}")
        
        # Initialize comparison report
        report = ModelComparisonReport(
            comparison_id=comparison_id,
            models_compared=[f"{config.model_type.value}:{config.model_name}" for config in model_configs]
        )
        
        # Evaluate each model with Ragas
        for config in model_configs:
            try:
                model_name = f"{config.model_type.value}:{config.model_name}"
                logger.info(f"Evaluating model: {model_name}")
                
                # Generate predictions for evaluation dataset
                predictions = await self._generate_model_predictions(config, evaluation_dataset)
                
                # Run Ragas evaluation
                ragas_results = await self._run_ragas_evaluation(predictions, evaluation_dataset)
                
                # Create performance profile
                profile = await self._create_performance_profile(
                    model_name, config.model_type.value, ragas_results, predictions
                )
                
                # Store results
                report.ragas_comparison[model_name] = ragas_results
                report.model_profiles[model_name] = profile
                
            except Exception as e:
                logger.error(f"Failed to evaluate model {config.model_name}: {e}")
                continue
        
        # Calculate rankings and recommendations
        report.quality_rankings = self._calculate_quality_rankings(report.ragas_comparison)
        report.overall_ranking = self._calculate_overall_ranking(report.ragas_comparison)
        report.optimization_recommendations = self._generate_optimization_recommendations(report.model_profiles)
        report.selection_recommendation = self._generate_selection_recommendation(report.overall_ranking)
        
        # Save to history
        self.comparison_history.append(report)
        self._save_comparison_history()
        
        logger.info(f"Completed Ragas model comparison: {comparison_id}")
        return report
    
    async def _generate_model_predictions(self, 
                                        config: ModelConfig, 
                                        evaluation_dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate predictions from a model for the evaluation dataset."""
        try:
            annotator = AnnotatorFactory.create_annotator(config)
            predictions = []
            
            for item in evaluation_dataset:
                try:
                    # Create mock task object
                    mock_task = type('Task', (), {
                        'id': uuid4(),
                        'project_id': 'ragas_evaluation',
                        'data': item
                    })()
                    
                    # Generate prediction
                    prediction = await annotator.predict(mock_task)
                    
                    # Extract answer from prediction
                    answer = prediction.prediction_data.get('answer', 
                            prediction.prediction_data.get('output',
                            prediction.prediction_data.get('result', '')))
                    
                    predictions.append({
                        'question': item.get('question', ''),
                        'answer': str(answer),
                        'contexts': item.get('contexts', []),
                        'ground_truth': item.get('ground_truth', ''),
                        'confidence': prediction.confidence,
                        'metadata': prediction.prediction_data
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to generate prediction for item: {e}")
                    continue
            
            return predictions
            
        except Exception as e:
            logger.error(f"Failed to create annotator for {config.model_name}: {e}")
            return []
    
    async def _run_ragas_evaluation(self, 
                                  predictions: List[Dict[str, Any]], 
                                  evaluation_dataset: List[Dict[str, Any]]) -> Dict[str, float]:
        """Run Ragas evaluation on model predictions."""
        if not predictions:
            return {}
        
        try:
            # Prepare dataset for Ragas
            dataset_dict = {
                "question": [p["question"] for p in predictions],
                "answer": [p["answer"] for p in predictions],
                "contexts": [[ctx] if isinstance(ctx, str) else ctx for ctx in [p.get("contexts", []) for p in predictions]],
                "ground_truth": [p["ground_truth"] for p in predictions]
            }
            
            # Filter out empty contexts if needed
            if not any(dataset_dict["contexts"]):
                dataset_dict.pop("contexts")
            
            # Create Ragas dataset
            dataset = Dataset.from_dict(dataset_dict)
            
            # Select appropriate metrics based on available data
            available_metrics = []
            if "contexts" in dataset_dict:
                available_metrics.extend([faithfulness, context_precision, context_recall])
            if dataset_dict["answer"] and dataset_dict["question"]:
                available_metrics.append(answer_relevancy)
            if dataset_dict["ground_truth"]:
                available_metrics.extend([answer_correctness, answer_similarity])
            
            if not available_metrics:
                logger.warning("No suitable Ragas metrics available for evaluation")
                return {}
            
            # Run evaluation
            result = evaluate(dataset, metrics=available_metrics)
            
            # Extract scores
            ragas_scores = {}
            for metric in available_metrics:
                metric_name = metric.name
                if metric_name in result:
                    ragas_scores[f"ragas_{metric_name}"] = float(result[metric_name])
            
            return ragas_scores
            
        except Exception as e:
            logger.error(f"Ragas evaluation failed: {e}")
            return {}
    
    async def _create_performance_profile(self, 
                                        model_name: str, 
                                        model_type: str,
                                        ragas_results: Dict[str, float],
                                        predictions: List[Dict[str, Any]]) -> ModelPerformanceProfile:
        """Create comprehensive performance profile for a model."""
        
        # Calculate additional performance metrics
        performance_metrics = {}
        
        if predictions:
            # Confidence statistics
            confidences = [p.get("confidence", 0.0) for p in predictions]
            performance_metrics["avg_confidence"] = statistics.mean(confidences)
            performance_metrics["confidence_std"] = statistics.stdev(confidences) if len(confidences) > 1 else 0.0
            
            # Answer length statistics
            answer_lengths = [len(str(p.get("answer", ""))) for p in predictions]
            performance_metrics["avg_answer_length"] = statistics.mean(answer_lengths)
            performance_metrics["answer_length_std"] = statistics.stdev(answer_lengths) if len(answer_lengths) > 1 else 0.0
        
        # Analyze strengths and weaknesses
        strengths, weaknesses = self._analyze_model_strengths_weaknesses(ragas_results, performance_metrics)
        
        # Calculate optimization potential
        optimization_potential = self._calculate_optimization_potential(ragas_results, performance_metrics)
        
        return ModelPerformanceProfile(
            model_name=model_name,
            model_type=model_type,
            ragas_metrics=ragas_results,
            performance_metrics=performance_metrics,
            strengths=strengths,
            weaknesses=weaknesses,
            optimization_potential=optimization_potential
        )
    
    def _analyze_model_strengths_weaknesses(self, 
                                          ragas_results: Dict[str, float],
                                          performance_metrics: Dict[str, float]) -> Tuple[List[str], List[str]]:
        """Analyze model strengths and weaknesses based on metrics."""
        strengths = []
        weaknesses = []
        
        # Analyze Ragas metrics
        for metric, score in ragas_results.items():
            if score >= 0.8:
                if "faithfulness" in metric:
                    strengths.append("高忠实度 - 生成内容严格基于提供的上下文")
                elif "relevancy" in metric:
                    strengths.append("高相关性 - 回答与问题高度相关")
                elif "precision" in metric:
                    strengths.append("高精确度 - 检索到的上下文精确相关")
                elif "recall" in metric:
                    strengths.append("高召回率 - 能够检索到所有相关信息")
                elif "correctness" in metric:
                    strengths.append("高正确性 - 回答内容准确无误")
                elif "similarity" in metric:
                    strengths.append("高相似性 - 回答与标准答案高度相似")
            elif score < 0.6:
                if "faithfulness" in metric:
                    weaknesses.append("忠实度不足 - 可能生成不基于上下文的内容")
                elif "relevancy" in metric:
                    weaknesses.append("相关性不足 - 回答可能偏离问题主题")
                elif "precision" in metric:
                    weaknesses.append("精确度不足 - 检索到的上下文可能包含无关信息")
                elif "recall" in metric:
                    weaknesses.append("召回率不足 - 可能遗漏重要的相关信息")
                elif "correctness" in metric:
                    weaknesses.append("正确性不足 - 回答可能包含错误信息")
                elif "similarity" in metric:
                    weaknesses.append("相似性不足 - 回答风格或内容与期望差异较大")
        
        # Analyze performance metrics
        avg_confidence = performance_metrics.get("avg_confidence", 0.0)
        confidence_std = performance_metrics.get("confidence_std", 0.0)
        
        if avg_confidence >= 0.8:
            strengths.append("高置信度 - 模型对预测结果有较高信心")
        elif avg_confidence < 0.6:
            weaknesses.append("置信度不足 - 模型对预测结果信心不足")
        
        if confidence_std < 0.1:
            strengths.append("置信度稳定 - 预测结果一致性好")
        elif confidence_std > 0.3:
            weaknesses.append("置信度不稳定 - 预测结果一致性差")
        
        return strengths, weaknesses
    
    def _calculate_optimization_potential(self, 
                                        ragas_results: Dict[str, float],
                                        performance_metrics: Dict[str, float]) -> float:
        """Calculate optimization potential score (0-1, higher means more potential for improvement)."""
        
        # Calculate current performance score
        current_scores = []
        
        # Add Ragas scores
        for metric, score in ragas_results.items():
            current_scores.append(score)
        
        # Add performance scores (normalized)
        avg_confidence = performance_metrics.get("avg_confidence", 0.0)
        current_scores.append(avg_confidence)
        
        # Stability score (inverse of std deviation)
        confidence_std = performance_metrics.get("confidence_std", 0.0)
        stability_score = max(0, 1 - confidence_std)
        current_scores.append(stability_score)
        
        if not current_scores:
            return 0.5  # Default moderate potential
        
        # Calculate average current performance
        avg_performance = statistics.mean(current_scores)
        
        # Optimization potential is inverse of current performance
        # But also consider variance - high variance means more optimization potential
        variance_factor = statistics.stdev(current_scores) if len(current_scores) > 1 else 0
        
        # Combine factors: lower performance + higher variance = higher optimization potential
        optimization_potential = (1 - avg_performance) * 0.7 + variance_factor * 0.3
        
        return min(1.0, max(0.0, optimization_potential))
    
    def _calculate_quality_rankings(self, ragas_comparison: Dict[str, Dict[str, float]]) -> Dict[str, List[Tuple[str, float]]]:
        """Calculate quality rankings for each metric."""
        rankings = {}
        
        # Get all metrics
        all_metrics = set()
        for model_results in ragas_comparison.values():
            all_metrics.update(model_results.keys())
        
        # Rank models for each metric
        for metric in all_metrics:
            model_scores = []
            for model_name, results in ragas_comparison.items():
                if metric in results:
                    model_scores.append((model_name, results[metric]))
            
            # Sort by score (descending)
            model_scores.sort(key=lambda x: x[1], reverse=True)
            rankings[metric] = model_scores
        
        return rankings
    
    def _calculate_overall_ranking(self, ragas_comparison: Dict[str, Dict[str, float]]) -> List[Tuple[str, float]]:
        """Calculate overall ranking using weighted combination of metrics."""
        overall_scores = {}
        
        for model_name, results in ragas_comparison.items():
            weighted_score = 0.0
            total_weight = 0.0
            
            for metric, score in results.items():
                weight = self.metric_weights.get(metric, 0.1)  # Default weight for unknown metrics
                weighted_score += score * weight
                total_weight += weight
            
            if total_weight > 0:
                overall_scores[model_name] = weighted_score / total_weight
            else:
                overall_scores[model_name] = 0.0
        
        # Sort by overall score (descending)
        return sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
    
    def _generate_optimization_recommendations(self, 
                                             model_profiles: Dict[str, ModelPerformanceProfile]) -> List[OptimizationRecommendation]:
        """Generate optimization recommendations for each model."""
        recommendations = []
        
        for model_name, profile in model_profiles.items():
            model_recommendations = self._generate_model_specific_recommendations(profile)
            recommendations.extend(model_recommendations)
        
        # Sort by priority and expected improvement
        priority_order = {"high": 3, "medium": 2, "low": 1}
        recommendations.sort(
            key=lambda x: (priority_order.get(x.priority, 0), x.expected_improvement),
            reverse=True
        )
        
        return recommendations
    
    def _generate_model_specific_recommendations(self, 
                                               profile: ModelPerformanceProfile) -> List[OptimizationRecommendation]:
        """Generate specific recommendations for a model based on its profile."""
        recommendations = []
        
        # Analyze Ragas metrics for specific recommendations
        ragas_metrics = profile.ragas_metrics
        
        # Faithfulness recommendations
        faithfulness_score = ragas_metrics.get("ragas_faithfulness", 0.0)
        if faithfulness_score < 0.7:
            recommendations.append(OptimizationRecommendation(
                model_name=profile.model_name,
                recommendation_type="faithfulness_improvement",
                priority="high",
                description="提高模型忠实度，确保生成内容严格基于提供的上下文",
                expected_improvement=0.8 - faithfulness_score,
                implementation_effort="medium",
                specific_actions=[
                    "优化提示词模板，强调基于事实回答",
                    "增加上下文相关性训练数据",
                    "调整生成参数，降低创造性设置",
                    "实施后处理验证机制"
                ],
                metrics_to_improve=["ragas_faithfulness"],
                estimated_timeline="2-4周"
            ))
        
        # Relevancy recommendations
        relevancy_score = ragas_metrics.get("ragas_answer_relevancy", 0.0)
        if relevancy_score < 0.7:
            recommendations.append(OptimizationRecommendation(
                model_name=profile.model_name,
                recommendation_type="relevancy_improvement",
                priority="high",
                description="提高回答相关性，确保回答直接针对问题",
                expected_improvement=0.8 - relevancy_score,
                implementation_effort="medium",
                specific_actions=[
                    "优化问题理解模块",
                    "改进检索算法，提高检索精度",
                    "增加问答对训练数据",
                    "实施相关性评分机制"
                ],
                metrics_to_improve=["ragas_answer_relevancy"],
                estimated_timeline="3-5周"
            ))
        
        # Context precision recommendations
        precision_score = ragas_metrics.get("ragas_context_precision", 0.0)
        if precision_score < 0.7:
            recommendations.append(OptimizationRecommendation(
                model_name=profile.model_name,
                recommendation_type="precision_improvement",
                priority="medium",
                description="提高上下文精确度，减少无关信息干扰",
                expected_improvement=0.75 - precision_score,
                implementation_effort="hard",
                specific_actions=[
                    "优化检索排序算法",
                    "增加负样本训练",
                    "实施上下文过滤机制",
                    "调整检索阈值参数"
                ],
                metrics_to_improve=["ragas_context_precision"],
                estimated_timeline="4-6周"
            ))
        
        # Context recall recommendations
        recall_score = ragas_metrics.get("ragas_context_recall", 0.0)
        if recall_score < 0.7:
            recommendations.append(OptimizationRecommendation(
                model_name=profile.model_name,
                recommendation_type="recall_improvement",
                priority="medium",
                description="提高上下文召回率，确保检索到所有相关信息",
                expected_improvement=0.75 - recall_score,
                implementation_effort="medium",
                specific_actions=[
                    "扩大检索范围",
                    "优化向量化策略",
                    "增加知识库内容",
                    "调整检索策略参数"
                ],
                metrics_to_improve=["ragas_context_recall"],
                estimated_timeline="2-4周"
            ))
        
        # Confidence-based recommendations
        avg_confidence = profile.performance_metrics.get("avg_confidence", 0.0)
        confidence_std = profile.performance_metrics.get("confidence_std", 0.0)
        
        if avg_confidence < 0.7:
            recommendations.append(OptimizationRecommendation(
                model_name=profile.model_name,
                recommendation_type="confidence_improvement",
                priority="low",
                description="提高模型置信度，增强预测可靠性",
                expected_improvement=0.8 - avg_confidence,
                implementation_effort="easy",
                specific_actions=[
                    "校准置信度计算方法",
                    "增加不确定性量化",
                    "优化模型参数",
                    "实施置信度后处理"
                ],
                metrics_to_improve=["avg_confidence"],
                estimated_timeline="1-2周"
            ))
        
        if confidence_std > 0.3:
            recommendations.append(OptimizationRecommendation(
                model_name=profile.model_name,
                recommendation_type="consistency_improvement",
                priority="medium",
                description="提高预测一致性，减少结果波动",
                expected_improvement=0.3 - confidence_std,
                implementation_effort="medium",
                specific_actions=[
                    "标准化输入预处理",
                    "优化模型稳定性",
                    "实施结果平滑机制",
                    "增加一致性训练"
                ],
                metrics_to_improve=["confidence_std"],
                estimated_timeline="3-4周"
            ))
        
        # High optimization potential recommendations
        if profile.optimization_potential > 0.7:
            recommendations.append(OptimizationRecommendation(
                model_name=profile.model_name,
                recommendation_type="comprehensive_optimization",
                priority="high",
                description="模型具有高优化潜力，建议进行全面优化",
                expected_improvement=profile.optimization_potential * 0.6,
                implementation_effort="hard",
                specific_actions=[
                    "进行全面性能分析",
                    "重新评估模型架构",
                    "考虑模型微调或替换",
                    "实施A/B测试验证改进效果"
                ],
                metrics_to_improve=list(profile.ragas_metrics.keys()),
                estimated_timeline="6-8周"
            ))
        
        return recommendations
    
    def _generate_selection_recommendation(self, overall_ranking: List[Tuple[str, float]]) -> Optional[str]:
        """Generate model selection recommendation based on overall ranking."""
        if not overall_ranking:
            return None
        
        best_model, best_score = overall_ranking[0]
        
        if best_score >= 0.8:
            return f"强烈推荐使用 {best_model}，综合评分 {best_score:.3f}，在所有评估指标中表现优秀"
        elif best_score >= 0.7:
            return f"推荐使用 {best_model}，综合评分 {best_score:.3f}，整体表现良好，但仍有优化空间"
        elif best_score >= 0.6:
            return f"可以使用 {best_model}，综合评分 {best_score:.3f}，表现中等，建议优先进行优化改进"
        else:
            return f"不推荐直接使用 {best_model}，综合评分 {best_score:.3f}，建议先进行大幅优化或考虑其他模型"
    
    async def _basic_model_comparison(self, 
                                    model_configs: List[ModelConfig],
                                    evaluation_dataset: List[Dict[str, Any]],
                                    task_type: str) -> ModelComparisonReport:
        """Fallback basic model comparison when Ragas is not available."""
        comparison_id = f"basic_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Running basic model comparison (Ragas not available): {comparison_id}")
        
        report = ModelComparisonReport(
            comparison_id=comparison_id,
            models_compared=[f"{config.model_type.value}:{config.model_name}" for config in model_configs]
        )
        
        # Basic evaluation without Ragas
        for config in model_configs:
            try:
                model_name = f"{config.model_type.value}:{config.model_name}"
                
                # Generate predictions
                predictions = await self._generate_model_predictions(config, evaluation_dataset)
                
                # Calculate basic metrics
                basic_metrics = self._calculate_basic_metrics(predictions, evaluation_dataset)
                
                # Create basic profile
                profile = ModelPerformanceProfile(
                    model_name=model_name,
                    model_type=config.model_type.value,
                    performance_metrics=basic_metrics,
                    strengths=["基础功能正常"],
                    weaknesses=["需要Ragas进行详细评估"],
                    optimization_potential=0.5
                )
                
                report.performance_comparison[model_name] = basic_metrics
                report.model_profiles[model_name] = profile
                
            except Exception as e:
                logger.error(f"Failed to evaluate model {config.model_name}: {e}")
                continue
        
        # Basic ranking based on confidence
        if report.performance_comparison:
            ranking = sorted(
                report.performance_comparison.items(),
                key=lambda x: x[1].get("avg_confidence", 0.0),
                reverse=True
            )
            report.overall_ranking = [(model, metrics.get("avg_confidence", 0.0)) for model, metrics in ranking]
        
        report.selection_recommendation = "建议安装Ragas进行详细的质量评估和模型对比"
        
        return report
    
    def _calculate_basic_metrics(self, 
                               predictions: List[Dict[str, Any]], 
                               evaluation_dataset: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate basic metrics without Ragas."""
        if not predictions:
            return {}
        
        metrics = {}
        
        # Confidence statistics
        confidences = [p.get("confidence", 0.0) for p in predictions]
        if confidences:
            metrics["avg_confidence"] = statistics.mean(confidences)
            metrics["confidence_std"] = statistics.stdev(confidences) if len(confidences) > 1 else 0.0
        
        # Response completeness
        complete_responses = sum(1 for p in predictions if p.get("answer", "").strip())
        metrics["response_completeness"] = complete_responses / len(predictions) if predictions else 0.0
        
        # Average response length
        response_lengths = [len(str(p.get("answer", ""))) for p in predictions]
        if response_lengths:
            metrics["avg_response_length"] = statistics.mean(response_lengths)
        
        return metrics
    
    def get_comparison_history(self, limit: Optional[int] = None) -> List[ModelComparisonReport]:
        """Get comparison history, optionally limited to recent comparisons."""
        history = sorted(self.comparison_history, key=lambda x: x.comparison_date, reverse=True)
        
        if limit:
            history = history[:limit]
        
        return history
    
    def get_model_optimization_status(self, model_name: str) -> Dict[str, Any]:
        """Get optimization status and recommendations for a specific model."""
        # Find latest comparison containing this model
        for report in reversed(self.comparison_history):
            if model_name in report.model_profiles:
                profile = report.model_profiles[model_name]
                model_recommendations = [
                    rec for rec in report.optimization_recommendations 
                    if rec.model_name == model_name
                ]
                
                return {
                    "model_name": model_name,
                    "last_evaluation": report.comparison_date.isoformat(),
                    "performance_profile": profile.to_dict(),
                    "optimization_recommendations": [rec.to_dict() for rec in model_recommendations],
                    "optimization_potential": profile.optimization_potential,
                    "current_strengths": profile.strengths,
                    "current_weaknesses": profile.weaknesses
                }
        
        return {"model_name": model_name, "status": "not_evaluated"}


class ModelOptimizer:
    """Main optimizer class that coordinates model comparison and optimization."""
    
    def __init__(self, storage_path: Optional[str] = None):
        """Initialize model optimizer."""
        self.comparison_engine = ModelComparisonEngine(storage_path)
        self.benchmark_suite = ModelBenchmarkSuite(storage_path)
        
        # Optimization tracking
        self.optimization_history: List[Dict[str, Any]] = []
        self._load_optimization_history()
    
    def _load_optimization_history(self) -> None:
        """Load optimization history from storage."""
        try:
            history_file = self.comparison_engine.storage_path / "optimization_history.json"
            if history_file.exists():
                with open(history_file, 'r', encoding='utf-8') as f:
                    self.optimization_history = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load optimization history: {e}")
    
    def _save_optimization_history(self) -> None:
        """Save optimization history to storage."""
        try:
            history_file = self.comparison_engine.storage_path / "optimization_history.json"
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(self.optimization_history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save optimization history: {e}")
    
    async def optimize_models(self, 
                            model_configs: List[ModelConfig],
                            evaluation_dataset: List[Dict[str, Any]],
                            optimization_goals: Optional[Dict[str, float]] = None) -> ModelComparisonReport:
        """
        Comprehensive model optimization with comparison and recommendations.
        
        Args:
            model_configs: List of model configurations to optimize
            evaluation_dataset: Dataset for evaluation
            optimization_goals: Optional optimization targets (e.g., {"ragas_faithfulness": 0.8})
            
        Returns:
            Comprehensive comparison report with optimization recommendations
        """
        logger.info(f"Starting model optimization for {len(model_configs)} models")
        
        # Run comprehensive comparison
        comparison_report = await self.comparison_engine.compare_models_with_ragas(
            model_configs, evaluation_dataset
        )
        
        # Apply optimization goals if provided
        if optimization_goals:
            comparison_report = self._apply_optimization_goals(comparison_report, optimization_goals)
        
        # Record optimization session
        optimization_session = {
            "session_id": comparison_report.comparison_id,
            "timestamp": datetime.now().isoformat(),
            "models_optimized": comparison_report.models_compared,
            "optimization_goals": optimization_goals or {},
            "recommendations_generated": len(comparison_report.optimization_recommendations),
            "best_model": comparison_report.overall_ranking[0][0] if comparison_report.overall_ranking else None
        }
        
        self.optimization_history.append(optimization_session)
        self._save_optimization_history()
        
        logger.info(f"Completed model optimization session: {comparison_report.comparison_id}")
        return comparison_report
    
    def _apply_optimization_goals(self, 
                                report: ModelComparisonReport, 
                                goals: Dict[str, float]) -> ModelComparisonReport:
        """Apply optimization goals to prioritize recommendations."""
        
        # Re-prioritize recommendations based on goals
        for recommendation in report.optimization_recommendations:
            for metric in recommendation.metrics_to_improve:
                if metric in goals:
                    target_value = goals[metric]
                    
                    # Get current value from model profile
                    model_profile = report.model_profiles.get(recommendation.model_name)
                    if model_profile:
                        current_value = (
                            model_profile.ragas_metrics.get(metric, 0.0) or
                            model_profile.performance_metrics.get(metric, 0.0)
                        )
                        
                        # Calculate gap to goal
                        gap_to_goal = max(0, target_value - current_value)
                        
                        # Adjust priority based on gap
                        if gap_to_goal > 0.2:
                            recommendation.priority = "high"
                        elif gap_to_goal > 0.1:
                            recommendation.priority = "medium"
                        
                        # Update expected improvement
                        recommendation.expected_improvement = max(
                            recommendation.expected_improvement, gap_to_goal
                        )
        
        # Re-sort recommendations
        priority_order = {"high": 3, "medium": 2, "low": 1}
        report.optimization_recommendations.sort(
            key=lambda x: (priority_order.get(x.priority, 0), x.expected_improvement),
            reverse=True
        )
        
        return report
    
    async def auto_select_optimal_model(self, 
                                      model_configs: List[ModelConfig],
                                      evaluation_dataset: List[Dict[str, Any]],
                                      selection_criteria: Optional[Dict[str, Any]] = None) -> Optional[ModelConfig]:
        """
        Automatically select the optimal model based on comprehensive evaluation.
        
        Args:
            model_configs: Available model configurations
            evaluation_dataset: Evaluation dataset
            selection_criteria: Selection criteria (e.g., {"min_faithfulness": 0.8, "max_response_time": 2.0})
            
        Returns:
            Selected optimal model configuration
        """
        logger.info(f"Auto-selecting optimal model from {len(model_configs)} candidates")
        
        # Run comparison
        comparison_report = await self.comparison_engine.compare_models_with_ragas(
            model_configs, evaluation_dataset
        )
        
        # Apply selection criteria if provided
        if selection_criteria:
            filtered_models = self._filter_models_by_criteria(comparison_report, selection_criteria)
        else:
            filtered_models = comparison_report.overall_ranking
        
        if not filtered_models:
            logger.warning("No models meet the selection criteria")
            return None
        
        # Select best model
        best_model_name = filtered_models[0][0]
        
        # Find corresponding configuration
        for config in model_configs:
            model_identifier = f"{config.model_type.value}:{config.model_name}"
            if model_identifier == best_model_name:
                logger.info(f"Selected optimal model: {best_model_name}")
                return config
        
        logger.error(f"Could not find configuration for selected model: {best_model_name}")
        return None
    
    def _filter_models_by_criteria(self, 
                                 report: ModelComparisonReport, 
                                 criteria: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Filter models based on selection criteria."""
        filtered = []
        
        for model_name, overall_score in report.overall_ranking:
            meets_criteria = True
            
            if model_name in report.model_profiles:
                profile = report.model_profiles[model_name]
                
                # Check Ragas criteria
                for criterion, threshold in criteria.items():
                    if criterion.startswith("min_"):
                        metric_name = criterion[4:]  # Remove "min_" prefix
                        
                        # Check in Ragas metrics
                        if f"ragas_{metric_name}" in profile.ragas_metrics:
                            if profile.ragas_metrics[f"ragas_{metric_name}"] < threshold:
                                meets_criteria = False
                                break
                        
                        # Check in performance metrics
                        elif metric_name in profile.performance_metrics:
                            if profile.performance_metrics[metric_name] < threshold:
                                meets_criteria = False
                                break
                    
                    elif criterion.startswith("max_"):
                        metric_name = criterion[4:]  # Remove "max_" prefix
                        
                        # Check in performance metrics
                        if metric_name in profile.performance_metrics:
                            if profile.performance_metrics[metric_name] > threshold:
                                meets_criteria = False
                                break
            
            if meets_criteria:
                filtered.append((model_name, overall_score))
        
        return filtered
    
    def generate_optimization_report(self, 
                                   comparison_report: ModelComparisonReport,
                                   include_detailed_analysis: bool = True) -> str:
        """Generate comprehensive optimization report."""
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("模型优化分析报告")
        report_lines.append("=" * 80)
        report_lines.append(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"对比ID: {comparison_report.comparison_id}")
        report_lines.append(f"参与对比的模型数量: {len(comparison_report.models_compared)}")
        report_lines.append("")
        
        # Overall ranking
        report_lines.append("整体排名:")
        report_lines.append("-" * 40)
        for i, (model_name, score) in enumerate(comparison_report.overall_ranking, 1):
            report_lines.append(f"{i}. {model_name}: {score:.3f}")
        report_lines.append("")
        
        # Selection recommendation
        if comparison_report.selection_recommendation:
            report_lines.append("选择建议:")
            report_lines.append("-" * 40)
            report_lines.append(comparison_report.selection_recommendation)
            report_lines.append("")
        
        # Optimization recommendations
        if comparison_report.optimization_recommendations:
            report_lines.append("优化建议:")
            report_lines.append("-" * 40)
            
            high_priority = [r for r in comparison_report.optimization_recommendations if r.priority == "high"]
            medium_priority = [r for r in comparison_report.optimization_recommendations if r.priority == "medium"]
            low_priority = [r for r in comparison_report.optimization_recommendations if r.priority == "low"]
            
            if high_priority:
                report_lines.append("高优先级建议:")
                for rec in high_priority:
                    report_lines.append(f"  • {rec.model_name}: {rec.description}")
                    report_lines.append(f"    预期改进: {rec.expected_improvement:.3f}, 实施难度: {rec.implementation_effort}")
                    if include_detailed_analysis and rec.specific_actions:
                        report_lines.append("    具体行动:")
                        for action in rec.specific_actions:
                            report_lines.append(f"      - {action}")
                    report_lines.append("")
            
            if medium_priority:
                report_lines.append("中优先级建议:")
                for rec in medium_priority:
                    report_lines.append(f"  • {rec.model_name}: {rec.description}")
                    report_lines.append(f"    预期改进: {rec.expected_improvement:.3f}, 实施难度: {rec.implementation_effort}")
                    report_lines.append("")
            
            if low_priority:
                report_lines.append("低优先级建议:")
                for rec in low_priority:
                    report_lines.append(f"  • {rec.model_name}: {rec.description}")
                    report_lines.append("")
        
        # Detailed model analysis
        if include_detailed_analysis:
            report_lines.append("详细模型分析:")
            report_lines.append("-" * 40)
            
            for model_name, profile in comparison_report.model_profiles.items():
                report_lines.append(f"模型: {model_name}")
                report_lines.append(f"类型: {profile.model_type}")
                report_lines.append(f"优化潜力: {profile.optimization_potential:.3f}")
                
                if profile.ragas_metrics:
                    report_lines.append("Ragas指标:")
                    for metric, score in profile.ragas_metrics.items():
                        report_lines.append(f"  {metric}: {score:.3f}")
                
                if profile.strengths:
                    report_lines.append("优势:")
                    for strength in profile.strengths:
                        report_lines.append(f"  + {strength}")
                
                if profile.weaknesses:
                    report_lines.append("劣势:")
                    for weakness in profile.weaknesses:
                        report_lines.append(f"  - {weakness}")
                
                report_lines.append("")
        
        return "\n".join(report_lines)
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get optimization history."""
        return sorted(self.optimization_history, key=lambda x: x["timestamp"], reverse=True)