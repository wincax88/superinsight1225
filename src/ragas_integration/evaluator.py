"""
Ragas Evaluator for Quality Assessment Integration.

Provides comprehensive Ragas-based evaluation capabilities for the
quality-billing loop system.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from uuid import UUID
from dataclasses import dataclass, field

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
    logging.warning(f"Ragas not available: {e}. Evaluation will use basic metrics only.")
    RAGAS_AVAILABLE = False

from src.models.annotation import Annotation


logger = logging.getLogger(__name__)


@dataclass
class RagasEvaluationResult:
    """Result of Ragas evaluation."""
    
    evaluation_id: str
    task_id: Optional[UUID] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    individual_scores: List[Dict[str, Any]] = field(default_factory=list)
    overall_score: float = 0.0
    evaluation_date: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "evaluation_id": self.evaluation_id,
            "task_id": str(self.task_id) if self.task_id else None,
            "metrics": self.metrics,
            "individual_scores": self.individual_scores,
            "overall_score": self.overall_score,
            "evaluation_date": self.evaluation_date.isoformat(),
            "metadata": self.metadata
        }


class RagasEvaluator:
    """Ragas-based quality evaluator for annotations."""
    
    def __init__(self):
        """Initialize Ragas evaluator."""
        self.available_metrics = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall,
            "answer_correctness": answer_correctness,
            "answer_similarity": answer_similarity
        } if RAGAS_AVAILABLE else {}
        
        # Default metric weights for overall score calculation
        self.metric_weights = {
            "faithfulness": 0.25,
            "answer_relevancy": 0.25,
            "context_precision": 0.15,
            "context_recall": 0.15,
            "answer_correctness": 0.10,
            "answer_similarity": 0.10
        }
    
    def is_available(self) -> bool:
        """Check if Ragas is available for evaluation."""
        return RAGAS_AVAILABLE
    
    async def evaluate_annotations(self, 
                                 annotations: List[Annotation],
                                 metrics: Optional[List[str]] = None,
                                 task_id: Optional[UUID] = None) -> RagasEvaluationResult:
        """
        Evaluate annotations using Ragas metrics.
        
        Args:
            annotations: List of annotations to evaluate
            metrics: Optional list of specific metrics to use
            task_id: Optional task ID for tracking
            
        Returns:
            Ragas evaluation result
        """
        if not RAGAS_AVAILABLE:
            logger.warning("Ragas not available, returning basic evaluation")
            return self._basic_evaluation(annotations, task_id)
        
        if not annotations:
            raise ValueError("No annotations provided for evaluation")
        
        evaluation_id = f"ragas_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Starting Ragas evaluation: {evaluation_id}")
        
        try:
            # Prepare dataset for Ragas
            dataset = self._prepare_ragas_dataset(annotations)
            
            if not dataset:
                logger.warning("Could not prepare dataset for Ragas evaluation")
                return self._basic_evaluation(annotations, task_id)
            
            # Select metrics to use
            selected_metrics = self._select_metrics(dataset, metrics)
            
            if not selected_metrics:
                logger.warning("No suitable metrics available for evaluation")
                return self._basic_evaluation(annotations, task_id)
            
            # Run Ragas evaluation
            result = evaluate(dataset, metrics=selected_metrics)
            
            # Process results
            evaluation_result = self._process_ragas_results(
                result, evaluation_id, task_id, selected_metrics, annotations
            )
            
            logger.info(f"Completed Ragas evaluation: {evaluation_id}")
            return evaluation_result
            
        except Exception as e:
            logger.error(f"Ragas evaluation failed: {e}")
            return self._basic_evaluation(annotations, task_id)
    
    def _prepare_ragas_dataset(self, annotations: List[Annotation]) -> Optional[Dataset]:
        """Prepare dataset for Ragas evaluation."""
        try:
            questions = []
            answers = []
            contexts = []
            ground_truths = []
            
            for annotation in annotations:
                data = annotation.annotation_data
                
                # Extract required fields
                question = self._extract_question(data)
                answer = self._extract_answer(data)
                context = self._extract_context(data)
                ground_truth = self._extract_ground_truth(data)
                
                if question and answer:
                    questions.append(str(question))
                    answers.append(str(answer))
                    contexts.append([str(context)] if context else [""])
                    ground_truths.append(str(ground_truth) if ground_truth else str(answer))
            
            if not questions or not answers:
                logger.warning("Insufficient data for Ragas evaluation")
                return None
            
            # Create dataset dictionary
            dataset_dict = {
                "question": questions,
                "answer": answers
            }
            
            # Add contexts if available
            if any(ctx[0] for ctx in contexts):
                dataset_dict["contexts"] = contexts
            
            # Add ground truths if available
            if any(gt != ans for gt, ans in zip(ground_truths, answers)):
                dataset_dict["ground_truth"] = ground_truths
            
            return Dataset.from_dict(dataset_dict)
            
        except Exception as e:
            logger.error(f"Failed to prepare Ragas dataset: {e}")
            return None
    
    def _extract_question(self, data: Dict[str, Any]) -> Optional[str]:
        """Extract question from annotation data."""
        return (
            data.get("question") or
            data.get("input") or
            data.get("query") or
            data.get("prompt")
        )
    
    def _extract_answer(self, data: Dict[str, Any]) -> Optional[str]:
        """Extract answer from annotation data."""
        return (
            data.get("answer") or
            data.get("output") or
            data.get("response") or
            data.get("result") or
            data.get("label")
        )
    
    def _extract_context(self, data: Dict[str, Any]) -> Optional[str]:
        """Extract context from annotation data."""
        context = (
            data.get("context") or
            data.get("contexts") or
            data.get("reference") or
            data.get("background")
        )
        
        # Handle different context formats
        if isinstance(context, list):
            return " ".join(str(c) for c in context)
        elif isinstance(context, str):
            return context
        else:
            return None
    
    def _extract_ground_truth(self, data: Dict[str, Any]) -> Optional[str]:
        """Extract ground truth from annotation data."""
        return (
            data.get("ground_truth") or
            data.get("expected") or
            data.get("target") or
            data.get("reference_answer")
        )
    
    def _select_metrics(self, dataset: Dataset, requested_metrics: Optional[List[str]] = None) -> List[Any]:
        """Select appropriate metrics based on available data and requests."""
        available_columns = set(dataset.column_names)
        selected = []
        
        # Define metric requirements
        metric_requirements = {
            "faithfulness": {"contexts"},
            "answer_relevancy": {"question", "answer"},
            "context_precision": {"contexts", "ground_truth"},
            "context_recall": {"contexts", "ground_truth"},
            "answer_correctness": {"ground_truth"},
            "answer_similarity": {"ground_truth"}
        }
        
        # Select metrics based on availability and requirements
        for metric_name, metric_obj in self.available_metrics.items():
            # Skip if specific metrics requested and this isn't one of them
            if requested_metrics and metric_name not in requested_metrics:
                continue
            
            # Check if required columns are available
            required_columns = metric_requirements.get(metric_name, set())
            if required_columns.issubset(available_columns):
                selected.append(metric_obj)
            else:
                logger.debug(f"Skipping {metric_name}: missing required columns {required_columns - available_columns}")
        
        return selected
    
    def _process_ragas_results(self, 
                             result: Dict[str, Any],
                             evaluation_id: str,
                             task_id: Optional[UUID],
                             metrics: List[Any],
                             annotations: List[Annotation]) -> RagasEvaluationResult:
        """Process Ragas evaluation results."""
        
        # Extract metric scores
        metric_scores = {}
        for metric in metrics:
            metric_name = metric.name
            if metric_name in result:
                metric_scores[metric_name] = float(result[metric_name])
        
        # Calculate overall score using weights
        overall_score = 0.0
        total_weight = 0.0
        
        for metric_name, score in metric_scores.items():
            weight = self.metric_weights.get(metric_name, 0.1)
            overall_score += score * weight
            total_weight += weight
        
        if total_weight > 0:
            overall_score /= total_weight
        
        # Create individual scores (if available in result)
        individual_scores = []
        try:
            # Some Ragas versions provide per-sample scores
            for i, annotation in enumerate(annotations):
                sample_scores = {}
                for metric_name in metric_scores.keys():
                    # Try to extract individual scores if available
                    if f"{metric_name}_scores" in result:
                        scores_list = result[f"{metric_name}_scores"]
                        if i < len(scores_list):
                            sample_scores[metric_name] = float(scores_list[i])
                
                if sample_scores:
                    individual_scores.append({
                        "annotation_id": str(annotation.id) if hasattr(annotation, 'id') else f"annotation_{i}",
                        "scores": sample_scores
                    })
        except Exception as e:
            logger.debug(f"Could not extract individual scores: {e}")
        
        return RagasEvaluationResult(
            evaluation_id=evaluation_id,
            task_id=task_id,
            metrics=metric_scores,
            individual_scores=individual_scores,
            overall_score=overall_score,
            metadata={
                "metrics_used": [m.name for m in metrics],
                "annotations_count": len(annotations),
                "ragas_version": "available" if RAGAS_AVAILABLE else "not_available"
            }
        )
    
    def _basic_evaluation(self, 
                        annotations: List[Annotation], 
                        task_id: Optional[UUID]) -> RagasEvaluationResult:
        """Fallback basic evaluation when Ragas is not available."""
        evaluation_id = f"basic_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Calculate basic metrics
        basic_metrics = {}
        
        if annotations:
            # Average confidence
            confidences = [ann.confidence for ann in annotations]
            basic_metrics["avg_confidence"] = sum(confidences) / len(confidences)
            
            # Completeness (non-empty answers)
            complete_answers = sum(
                1 for ann in annotations 
                if ann.annotation_data.get("answer", "").strip()
            )
            basic_metrics["completeness"] = complete_answers / len(annotations)
            
            # Response length consistency
            answer_lengths = [
                len(str(ann.annotation_data.get("answer", ""))) 
                for ann in annotations
            ]
            if answer_lengths:
                avg_length = sum(answer_lengths) / len(answer_lengths)
                basic_metrics["avg_answer_length"] = avg_length
        
        # Calculate basic overall score
        overall_score = basic_metrics.get("avg_confidence", 0.0) * 0.7 + basic_metrics.get("completeness", 0.0) * 0.3
        
        return RagasEvaluationResult(
            evaluation_id=evaluation_id,
            task_id=task_id,
            metrics=basic_metrics,
            overall_score=overall_score,
            metadata={
                "evaluation_type": "basic",
                "reason": "Ragas not available",
                "annotations_count": len(annotations)
            }
        )
    
    async def evaluate_single_annotation(self, 
                                       annotation: Annotation,
                                       reference_data: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Evaluate a single annotation with Ragas.
        
        Args:
            annotation: Single annotation to evaluate
            reference_data: Optional reference data for comparison
            
        Returns:
            Dictionary of metric scores
        """
        if not RAGAS_AVAILABLE:
            logger.warning("Ragas not available for single annotation evaluation")
            return {"confidence": annotation.confidence}
        
        try:
            # Create single-item dataset
            data = annotation.annotation_data
            
            question = self._extract_question(data)
            answer = self._extract_answer(data)
            context = self._extract_context(data)
            ground_truth = reference_data.get("ground_truth") if reference_data else None
            
            if not question or not answer:
                logger.warning("Insufficient data for single annotation evaluation")
                return {"confidence": annotation.confidence}
            
            # Create dataset
            dataset_dict = {
                "question": [question],
                "answer": [answer]
            }
            
            if context:
                dataset_dict["contexts"] = [[context]]
            
            if ground_truth:
                dataset_dict["ground_truth"] = [ground_truth]
            
            dataset = Dataset.from_dict(dataset_dict)
            
            # Select and run metrics
            metrics = self._select_metrics(dataset)
            if not metrics:
                return {"confidence": annotation.confidence}
            
            result = evaluate(dataset, metrics=metrics)
            
            # Extract scores
            scores = {}
            for metric in metrics:
                metric_name = metric.name
                if metric_name in result:
                    scores[metric_name] = float(result[metric_name])
            
            return scores
            
        except Exception as e:
            logger.error(f"Single annotation evaluation failed: {e}")
            return {"confidence": annotation.confidence}
    
    def get_metric_descriptions(self) -> Dict[str, str]:
        """Get descriptions of available Ragas metrics."""
        descriptions = {
            "faithfulness": "衡量生成答案是否忠实于给定的上下文，避免幻觉内容",
            "answer_relevancy": "评估答案与问题的相关性程度",
            "context_precision": "衡量检索到的上下文中相关信息的精确度",
            "context_recall": "评估检索到的上下文是否包含回答问题所需的所有信息",
            "answer_correctness": "通过与标准答案对比评估答案的正确性",
            "answer_similarity": "衡量生成答案与标准答案的语义相似度"
        }
        
        if not RAGAS_AVAILABLE:
            descriptions = {k: f"{v} (需要安装Ragas)" for k, v in descriptions.items()}
        
        return descriptions
    
    def configure_metric_weights(self, weights: Dict[str, float]) -> None:
        """Configure metric weights for overall score calculation."""
        # Validate weights
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Metric weights sum to {total_weight}, not 1.0. Normalizing...")
            weights = {k: v / total_weight for k, v in weights.items()}
        
        # Update weights
        self.metric_weights.update(weights)
        logger.info(f"Updated metric weights: {self.metric_weights}")
    
    async def batch_evaluate(self, 
                           annotation_batches: List[List[Annotation]],
                           batch_size: int = 10) -> List[RagasEvaluationResult]:
        """
        Evaluate multiple batches of annotations efficiently.
        
        Args:
            annotation_batches: List of annotation batches
            batch_size: Maximum batch size for processing
            
        Returns:
            List of evaluation results for each batch
        """
        results = []
        
        for i, batch in enumerate(annotation_batches):
            try:
                logger.info(f"Evaluating batch {i+1}/{len(annotation_batches)}")
                
                # Split large batches
                if len(batch) > batch_size:
                    sub_batches = [batch[j:j+batch_size] for j in range(0, len(batch), batch_size)]
                    
                    for sub_batch in sub_batches:
                        result = await self.evaluate_annotations(sub_batch)
                        results.append(result)
                else:
                    result = await self.evaluate_annotations(batch)
                    results.append(result)
                    
            except Exception as e:
                logger.error(f"Failed to evaluate batch {i+1}: {e}")
                continue
        
        return results
    
    def compare_evaluations(self, 
                          results: List[RagasEvaluationResult]) -> Dict[str, Any]:
        """
        Compare multiple evaluation results.
        
        Args:
            results: List of evaluation results to compare
            
        Returns:
            Comparison analysis
        """
        if not results:
            return {"error": "No results to compare"}
        
        comparison = {
            "total_evaluations": len(results),
            "metric_comparison": {},
            "overall_scores": [],
            "best_evaluation": None,
            "worst_evaluation": None,
            "average_scores": {}
        }
        
        # Collect all metrics
        all_metrics = set()
        for result in results:
            all_metrics.update(result.metrics.keys())
        
        # Compare metrics
        for metric in all_metrics:
            metric_scores = []
            for result in results:
                if metric in result.metrics:
                    metric_scores.append(result.metrics[metric])
            
            if metric_scores:
                comparison["metric_comparison"][metric] = {
                    "scores": metric_scores,
                    "average": sum(metric_scores) / len(metric_scores),
                    "min": min(metric_scores),
                    "max": max(metric_scores),
                    "std": (sum((x - sum(metric_scores)/len(metric_scores))**2 for x in metric_scores) / len(metric_scores))**0.5
                }
        
        # Overall score comparison
        overall_scores = [result.overall_score for result in results]
        comparison["overall_scores"] = overall_scores
        
        if overall_scores:
            best_idx = overall_scores.index(max(overall_scores))
            worst_idx = overall_scores.index(min(overall_scores))
            
            comparison["best_evaluation"] = {
                "evaluation_id": results[best_idx].evaluation_id,
                "score": overall_scores[best_idx]
            }
            
            comparison["worst_evaluation"] = {
                "evaluation_id": results[worst_idx].evaluation_id,
                "score": overall_scores[worst_idx]
            }
            
            comparison["average_scores"] = {
                "overall_average": sum(overall_scores) / len(overall_scores),
                "overall_std": (sum((x - sum(overall_scores)/len(overall_scores))**2 for x in overall_scores) / len(overall_scores))**0.5
            }
        
        return comparison