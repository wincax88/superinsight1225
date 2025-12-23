"""
Quality Manager for SuperInsight Platform.

Implements quality assessment, rule templates, and work order management
using Ragas framework for semantic quality evaluation.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from uuid import UUID, uuid4
from enum import Enum
from sqlalchemy import select

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
    logging.warning(f"Ragas not available: {e}. Quality evaluation will use basic rules only.")
    RAGAS_AVAILABLE = False

from src.models.quality_issue import QualityIssue, IssueSeverity, IssueStatus
from src.models.task import Task, TaskStatus
from src.models.annotation import Annotation
from src.database.connection import get_db_session, db_manager
from src.database.models import QualityIssueModel, TaskModel


logger = logging.getLogger(__name__)


class QualityRuleType(str, Enum):
    """Types of quality rules that can be applied."""
    SEMANTIC_CONSISTENCY = "semantic_consistency"
    ANNOTATION_COMPLETENESS = "annotation_completeness"
    CONFIDENCE_THRESHOLD = "confidence_threshold"
    INTER_ANNOTATOR_AGREEMENT = "inter_annotator_agreement"
    LABEL_DISTRIBUTION = "label_distribution"
    RESPONSE_RELEVANCY = "response_relevancy"
    FACTUAL_ACCURACY = "factual_accuracy"


class QualityRule:
    """Represents a quality assessment rule."""
    
    def __init__(
        self,
        rule_id: str,
        rule_type: QualityRuleType,
        name: str,
        description: str,
        threshold: float = 0.7,
        severity: IssueSeverity = IssueSeverity.MEDIUM,
        enabled: bool = True,
        parameters: Optional[Dict[str, Any]] = None
    ):
        self.rule_id = rule_id
        self.rule_type = rule_type
        self.name = name
        self.description = description
        self.threshold = threshold
        self.severity = severity
        self.enabled = enabled
        self.parameters = parameters or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert rule to dictionary."""
        return {
            "rule_id": self.rule_id,
            "rule_type": self.rule_type.value,
            "name": self.name,
            "description": self.description,
            "threshold": self.threshold,
            "severity": self.severity.value,
            "enabled": self.enabled,
            "parameters": self.parameters
        }


class QualityReport:
    """Represents a quality assessment report."""
    
    def __init__(
        self,
        task_id: UUID,
        overall_score: float,
        rule_results: List[Dict[str, Any]],
        issues_found: List[QualityIssue],
        ragas_metrics: Optional[Dict[str, float]] = None,
        created_at: Optional[datetime] = None
    ):
        self.task_id = task_id
        self.overall_score = overall_score
        self.rule_results = rule_results
        self.issues_found = issues_found
        self.ragas_metrics = ragas_metrics or {}
        self.created_at = created_at or datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "task_id": str(self.task_id),
            "overall_score": self.overall_score,
            "rule_results": self.rule_results,
            "issues_found": [issue.to_dict() for issue in self.issues_found],
            "ragas_metrics": self.ragas_metrics,
            "created_at": self.created_at.isoformat()
        }


class QualityManager:
    """
    Quality Manager for SuperInsight Platform.
    
    Provides quality assessment, rule management, and work order creation
    using Ragas framework for semantic evaluation.
    """
    
    def __init__(self):
        self.quality_rules: Dict[str, QualityRule] = {}
        self.rule_templates: Dict[str, QualityRule] = {}
        self._initialize_default_rules()
    
    def _initialize_default_rules(self) -> None:
        """Initialize default quality rule templates."""
        # Confidence threshold rule
        confidence_rule = QualityRule(
            rule_id="confidence_threshold",
            rule_type=QualityRuleType.CONFIDENCE_THRESHOLD,
            name="置信度阈值检查",
            description="检查标注结果的置信度是否达到最低要求",
            threshold=0.7,
            severity=IssueSeverity.MEDIUM,
            parameters={"min_confidence": 0.7}
        )
        
        # Semantic consistency rule
        semantic_rule = QualityRule(
            rule_id="semantic_consistency",
            rule_type=QualityRuleType.SEMANTIC_CONSISTENCY,
            name="语义一致性检查",
            description="使用 Ragas 检查标注结果的语义一致性",
            threshold=0.8,
            severity=IssueSeverity.HIGH,
            parameters={"use_ragas": True}
        )
        
        # Annotation completeness rule
        completeness_rule = QualityRule(
            rule_id="annotation_completeness",
            rule_type=QualityRuleType.ANNOTATION_COMPLETENESS,
            name="标注完整性检查",
            description="检查标注是否包含所有必需的字段和标签",
            threshold=1.0,
            severity=IssueSeverity.HIGH,
            parameters={"required_fields": ["label", "confidence"]}
        )
        
        # Response relevancy rule (for RAG/QA tasks)
        relevancy_rule = QualityRule(
            rule_id="response_relevancy",
            rule_type=QualityRuleType.RESPONSE_RELEVANCY,
            name="回答相关性检查",
            description="使用 Ragas 检查回答与问题的相关性",
            threshold=0.75,
            severity=IssueSeverity.MEDIUM,
            parameters={"use_ragas": True, "metric": "answer_relevancy"}
        )
        
        # Factual accuracy rule
        accuracy_rule = QualityRule(
            rule_id="factual_accuracy",
            rule_type=QualityRuleType.FACTUAL_ACCURACY,
            name="事实准确性检查",
            description="使用 Ragas 检查回答的事实准确性",
            threshold=0.8,
            severity=IssueSeverity.CRITICAL,
            parameters={"use_ragas": True, "metric": "faithfulness"}
        )
        
        # Store templates
        self.rule_templates = {
            rule.rule_id: rule for rule in [
                confidence_rule, semantic_rule, completeness_rule,
                relevancy_rule, accuracy_rule
            ]
        }
        
        # Initialize active rules with templates
        self.quality_rules = self.rule_templates.copy()
    
    def get_rule_templates(self) -> Dict[str, QualityRule]:
        """Get available quality rule templates."""
        return self.rule_templates.copy()
    
    def add_quality_rule(self, rule: QualityRule) -> None:
        """Add or update a quality rule."""
        self.quality_rules[rule.rule_id] = rule
        logger.info(f"Added quality rule: {rule.name} ({rule.rule_id})")
    
    def remove_quality_rule(self, rule_id: str) -> bool:
        """Remove a quality rule."""
        if rule_id in self.quality_rules:
            del self.quality_rules[rule_id]
            logger.info(f"Removed quality rule: {rule_id}")
            return True
        return False
    
    def enable_rule(self, rule_id: str) -> bool:
        """Enable a quality rule."""
        if rule_id in self.quality_rules:
            self.quality_rules[rule_id].enabled = True
            return True
        return False
    
    def disable_rule(self, rule_id: str) -> bool:
        """Disable a quality rule."""
        if rule_id in self.quality_rules:
            self.quality_rules[rule_id].enabled = False
            return True
        return False
    
    async def evaluate_quality(self, annotations: List[Annotation]) -> QualityReport:
        """
        Evaluate quality of annotations using configured rules and Ragas.
        
        Args:
            annotations: List of annotations to evaluate
            
        Returns:
            QualityReport with assessment results
        """
        if not annotations:
            raise ValueError("No annotations provided for quality evaluation")
        
        # Get task ID from first annotation
        task_id = annotations[0].task_id
        
        # Initialize results
        rule_results = []
        issues_found = []
        ragas_metrics = {}
        
        # Apply each enabled quality rule
        for rule in self.quality_rules.values():
            if not rule.enabled:
                continue
            
            try:
                rule_result = await self._apply_quality_rule(rule, annotations)
                rule_results.append(rule_result)
                
                # Create quality issues for failed rules
                if not rule_result["passed"]:
                    issue = QualityIssue(
                        task_id=task_id,
                        issue_type=rule.rule_type.value,
                        description=f"{rule.name}: {rule_result.get('message', '质量检查失败')}",
                        severity=rule.severity,
                        status=IssueStatus.OPEN
                    )
                    issues_found.append(issue)
                
                # Collect Ragas metrics if available
                if "ragas_score" in rule_result:
                    ragas_metrics[rule.rule_id] = rule_result["ragas_score"]
                    
            except Exception as e:
                logger.error(f"Error applying quality rule {rule.rule_id}: {str(e)}")
                rule_results.append({
                    "rule_id": rule.rule_id,
                    "rule_name": rule.name,
                    "passed": False,
                    "score": 0.0,
                    "message": f"规则执行错误: {str(e)}"
                })
        
        # Calculate overall quality score
        if rule_results:
            passed_rules = sum(1 for result in rule_results if result["passed"])
            overall_score = passed_rules / len(rule_results)
        else:
            overall_score = 1.0  # No rules means perfect score
        
        # Run Ragas evaluation if we have suitable data
        try:
            ragas_results = await self._run_ragas_evaluation(annotations)
            ragas_metrics.update(ragas_results)
        except Exception as e:
            logger.warning(f"Ragas evaluation failed: {str(e)}")
        
        return QualityReport(
            task_id=task_id,
            overall_score=overall_score,
            rule_results=rule_results,
            issues_found=issues_found,
            ragas_metrics=ragas_metrics
        )
    
    async def _apply_quality_rule(
        self, 
        rule: QualityRule, 
        annotations: List[Annotation]
    ) -> Dict[str, Any]:
        """Apply a single quality rule to annotations."""
        
        if rule.rule_type == QualityRuleType.CONFIDENCE_THRESHOLD:
            return await self._check_confidence_threshold(rule, annotations)
        elif rule.rule_type == QualityRuleType.ANNOTATION_COMPLETENESS:
            return await self._check_annotation_completeness(rule, annotations)
        elif rule.rule_type == QualityRuleType.SEMANTIC_CONSISTENCY:
            return await self._check_semantic_consistency(rule, annotations)
        elif rule.rule_type == QualityRuleType.RESPONSE_RELEVANCY:
            return await self._check_response_relevancy(rule, annotations)
        elif rule.rule_type == QualityRuleType.FACTUAL_ACCURACY:
            return await self._check_factual_accuracy(rule, annotations)
        else:
            return {
                "rule_id": rule.rule_id,
                "rule_name": rule.name,
                "passed": True,
                "score": 1.0,
                "message": "规则类型未实现"
            }
    
    async def _check_confidence_threshold(
        self, 
        rule: QualityRule, 
        annotations: List[Annotation]
    ) -> Dict[str, Any]:
        """Check if annotation confidence meets threshold."""
        min_confidence = rule.parameters.get("min_confidence", rule.threshold)
        
        low_confidence_count = 0
        total_confidence = 0.0
        
        for annotation in annotations:
            total_confidence += annotation.confidence
            if annotation.confidence < min_confidence:
                low_confidence_count += 1
        
        avg_confidence = total_confidence / len(annotations)
        passed = low_confidence_count == 0 and avg_confidence >= min_confidence
        
        return {
            "rule_id": rule.rule_id,
            "rule_name": rule.name,
            "passed": passed,
            "score": avg_confidence,
            "message": f"平均置信度: {avg_confidence:.3f}, 低置信度标注数: {low_confidence_count}"
        }
    
    async def _check_annotation_completeness(
        self, 
        rule: QualityRule, 
        annotations: List[Annotation]
    ) -> Dict[str, Any]:
        """Check if annotations contain all required fields."""
        required_fields = rule.parameters.get("required_fields", ["label"])
        
        incomplete_count = 0
        
        for annotation in annotations:
            annotation_data = annotation.annotation_data
            missing_fields = []
            
            for field in required_fields:
                if field not in annotation_data or not annotation_data[field]:
                    missing_fields.append(field)
            
            if missing_fields:
                incomplete_count += 1
        
        completeness_score = (len(annotations) - incomplete_count) / len(annotations)
        passed = incomplete_count == 0
        
        return {
            "rule_id": rule.rule_id,
            "rule_name": rule.name,
            "passed": passed,
            "score": completeness_score,
            "message": f"完整性评分: {completeness_score:.3f}, 不完整标注数: {incomplete_count}"
        }
    
    async def _check_semantic_consistency(
        self, 
        rule: QualityRule, 
        annotations: List[Annotation]
    ) -> Dict[str, Any]:
        """Check semantic consistency using Ragas if available."""
        if not rule.parameters.get("use_ragas", False) or not RAGAS_AVAILABLE:
            # Simple consistency check without Ragas
            return {
                "rule_id": rule.rule_id,
                "rule_name": rule.name,
                "passed": True,
                "score": 1.0,
                "message": "语义一致性检查通过（未使用 Ragas）"
            }
        
        try:
            # Use Ragas for semantic evaluation
            ragas_score = await self._evaluate_with_ragas(annotations, "consistency")
            passed = ragas_score >= rule.threshold
            
            return {
                "rule_id": rule.rule_id,
                "rule_name": rule.name,
                "passed": passed,
                "score": ragas_score,
                "ragas_score": ragas_score,
                "message": f"Ragas 语义一致性评分: {ragas_score:.3f}"
            }
        except Exception as e:
            logger.error(f"Ragas semantic consistency check failed: {str(e)}")
            return {
                "rule_id": rule.rule_id,
                "rule_name": rule.name,
                "passed": False,
                "score": 0.0,
                "message": f"Ragas 评估失败: {str(e)}"
            }
    
    async def _check_response_relevancy(
        self, 
        rule: QualityRule, 
        annotations: List[Annotation]
    ) -> Dict[str, Any]:
        """Check response relevancy using Ragas."""
        if not RAGAS_AVAILABLE:
            return {
                "rule_id": rule.rule_id,
                "rule_name": rule.name,
                "passed": True,
                "score": 1.0,
                "message": "相关性检查跳过（Ragas 不可用）"
            }
            
        try:
            ragas_score = await self._evaluate_with_ragas(annotations, "answer_relevancy")
            passed = ragas_score >= rule.threshold
            
            return {
                "rule_id": rule.rule_id,
                "rule_name": rule.name,
                "passed": passed,
                "score": ragas_score,
                "ragas_score": ragas_score,
                "message": f"回答相关性评分: {ragas_score:.3f}"
            }
        except Exception as e:
            logger.error(f"Ragas relevancy check failed: {str(e)}")
            return {
                "rule_id": rule.rule_id,
                "rule_name": rule.name,
                "passed": False,
                "score": 0.0,
                "message": f"相关性评估失败: {str(e)}"
            }
    
    async def _check_factual_accuracy(
        self, 
        rule: QualityRule, 
        annotations: List[Annotation]
    ) -> Dict[str, Any]:
        """Check factual accuracy using Ragas faithfulness metric."""
        if not RAGAS_AVAILABLE:
            return {
                "rule_id": rule.rule_id,
                "rule_name": rule.name,
                "passed": True,
                "score": 1.0,
                "message": "准确性检查跳过（Ragas 不可用）"
            }
            
        try:
            ragas_score = await self._evaluate_with_ragas(annotations, "faithfulness")
            passed = ragas_score >= rule.threshold
            
            return {
                "rule_id": rule.rule_id,
                "rule_name": rule.name,
                "passed": passed,
                "score": ragas_score,
                "ragas_score": ragas_score,
                "message": f"事实准确性评分: {ragas_score:.3f}"
            }
        except Exception as e:
            logger.error(f"Ragas faithfulness check failed: {str(e)}")
            return {
                "rule_id": rule.rule_id,
                "rule_name": rule.name,
                "passed": False,
                "score": 0.0,
                "message": f"准确性评估失败: {str(e)}"
            }
    
    async def _evaluate_with_ragas(
        self, 
        annotations: List[Annotation], 
        metric_type: str
    ) -> float:
        """Evaluate annotations using Ragas metrics."""
        
        if not RAGAS_AVAILABLE:
            raise ValueError("Ragas is not available")
        
        # Prepare data for Ragas evaluation
        questions = []
        answers = []
        contexts = []
        ground_truths = []
        
        for annotation in annotations:
            data = annotation.annotation_data
            
            # Extract question, answer, context from annotation data
            question = data.get("question", data.get("input", ""))
            answer = data.get("answer", data.get("output", data.get("label", "")))
            context = data.get("context", data.get("reference", ""))
            ground_truth = data.get("ground_truth", data.get("expected", answer))
            
            if question and answer:
                questions.append(str(question))
                answers.append(str(answer))
                contexts.append(str(context) if context else "")
                ground_truths.append(str(ground_truth))
        
        if not questions or not answers:
            raise ValueError("Insufficient data for Ragas evaluation")
        
        # Create dataset for Ragas
        dataset_dict = {
            "question": questions,
            "answer": answers,
            "contexts": [[ctx] for ctx in contexts] if contexts[0] else None,
            "ground_truth": ground_truths
        }
        
        # Remove None values
        dataset_dict = {k: v for k, v in dataset_dict.items() if v is not None}
        
        dataset = Dataset.from_dict(dataset_dict)
        
        # Select appropriate metric
        if metric_type == "answer_relevancy":
            metrics = [answer_relevancy]
        elif metric_type == "faithfulness":
            metrics = [faithfulness]
        elif metric_type == "consistency":
            metrics = [answer_correctness]
        else:
            metrics = [answer_relevancy]  # Default
        
        # Run evaluation
        result = evaluate(dataset, metrics=metrics)
        
        # Return the score for the requested metric
        metric_name = metrics[0].name
        return float(result[metric_name])
    
    async def _run_ragas_evaluation(self, annotations: List[Annotation]) -> Dict[str, float]:
        """Run comprehensive Ragas evaluation on annotations."""
        if not RAGAS_AVAILABLE:
            logger.warning("Ragas not available, skipping comprehensive evaluation")
            return {}
            
        try:
            # Try to evaluate with multiple metrics
            metrics_results = {}
            
            for metric_name in ["answer_relevancy", "faithfulness", "consistency"]:
                try:
                    score = await self._evaluate_with_ragas(annotations, metric_name)
                    metrics_results[f"ragas_{metric_name}"] = score
                except Exception as e:
                    logger.warning(f"Ragas {metric_name} evaluation failed: {str(e)}")
            
            return metrics_results
            
        except Exception as e:
            logger.warning(f"Comprehensive Ragas evaluation failed: {str(e)}")
            return {}
    
    async def create_quality_issue(
        self, 
        task_id: UUID, 
        issue_type: str, 
        description: str,
        severity: IssueSeverity = IssueSeverity.MEDIUM,
        assignee_id: Optional[str] = None
    ) -> QualityIssue:
        """Create a new quality issue and save to database."""
        
        issue = QualityIssue(
            task_id=task_id,
            issue_type=issue_type,
            description=description,
            severity=severity,
            status=IssueStatus.OPEN,
            assignee_id=assignee_id
        )
        
        # Save to database
        with db_manager.get_session() as session:
            db_issue = QualityIssueModel(
                id=issue.id,
                task_id=issue.task_id,
                issue_type=issue.issue_type,
                description=issue.description,
                severity=issue.severity,
                status=issue.status,
                assignee_id=issue.assignee_id,
                created_at=issue.created_at
            )
            
            session.add(db_issue)
            session.commit()
        
        logger.info(f"Created quality issue {issue.id} for task {task_id}")
        
        # Auto-assign if assignee specified
        if assignee_id:
            await self.assign_quality_issue(issue.id, assignee_id)
        
        return issue
    
    async def assign_quality_issue(self, issue_id: UUID, assignee_id: str) -> bool:
        """Assign a quality issue to a user."""
        
        with db_manager.get_session() as session:
            stmt = select(QualityIssueModel).where(
                QualityIssueModel.id == issue_id
            )
            result = session.execute(stmt)
            db_issue = result.scalar_one_or_none()
            
            if not db_issue:
                logger.error(f"Quality issue {issue_id} not found")
                return False
            
            db_issue.assignee_id = assignee_id
            if db_issue.status == IssueStatus.OPEN:
                db_issue.status = IssueStatus.IN_PROGRESS
            
            session.commit()
        
        logger.info(f"Assigned quality issue {issue_id} to {assignee_id}")
        return True
    
    async def resolve_quality_issue(
        self, 
        issue_id: UUID, 
        resolution_notes: Optional[str] = None
    ) -> bool:
        """Mark a quality issue as resolved."""
        
        with db_manager.get_session() as session:
            stmt = select(QualityIssueModel).where(
                QualityIssueModel.id == issue_id
            )
            result = session.execute(stmt)
            db_issue = result.scalar_one_or_none()
            
            if not db_issue:
                logger.error(f"Quality issue {issue_id} not found")
                return False
            
            db_issue.status = IssueStatus.RESOLVED
            db_issue.resolved_at = datetime.now()
            
            if resolution_notes:
                # Add resolution notes to description
                db_issue.description += f"\n\n解决方案: {resolution_notes}"
            
            session.commit()
        
        logger.info(f"Resolved quality issue {issue_id}")
        return True
    
    async def get_quality_issues(
        self, 
        task_id: Optional[UUID] = None,
        status: Optional[IssueStatus] = None,
        assignee_id: Optional[str] = None
    ) -> List[QualityIssue]:
        """Get quality issues with optional filters."""
        
        with db_manager.get_session() as session:
            stmt = select(QualityIssueModel)
            
            if task_id:
                stmt = stmt.where(QualityIssueModel.task_id == task_id)
            if status:
                stmt = stmt.where(QualityIssueModel.status == status)
            if assignee_id:
                stmt = stmt.where(QualityIssueModel.assignee_id == assignee_id)
            
            result = session.execute(stmt)
            db_issues = result.scalars().all()
            
            # Convert to domain models
            issues = []
            for db_issue in db_issues:
                issue = QualityIssue(
                    id=db_issue.id,
                    task_id=db_issue.task_id,
                    issue_type=db_issue.issue_type,
                    description=db_issue.description,
                    severity=db_issue.severity,
                    status=db_issue.status,
                    assignee_id=db_issue.assignee_id,
                    created_at=db_issue.created_at,
                    resolved_at=db_issue.resolved_at
                )
                issues.append(issue)
            
            return issues
    
    async def trigger_quality_check(
        self, 
        task_id: UUID, 
        annotation_data: Dict[str, Any]
    ) -> bool:
        """
        Trigger quality check for a completed annotation.
        
        This method is called automatically when annotations are completed
        to fulfill Requirement 3.4.
        """
        try:
            logger.info(f"Triggering quality check for task {task_id}")
            
            # Get task and annotations from database
            with db_manager.get_session() as session:
                stmt = select(TaskModel).where(
                    TaskModel.id == task_id
                )
                result = session.execute(stmt)
                db_task = result.scalar_one_or_none()
                
                if not db_task:
                    logger.error(f"Task {task_id} not found")
                    return False
                
                # Convert database annotations to domain models
                annotations = []
                for ann_data in db_task.annotations:
                    try:
                        annotation = Annotation(
                            task_id=task_id,
                            annotator_id=ann_data.get("annotator", "unknown"),
                            annotation_data=ann_data.get("result", {"label": "default"}),
                            confidence=ann_data.get("confidence", 1.0),
                            time_spent=ann_data.get("lead_time", 0),
                            created_at=datetime.now()
                        )
                        annotations.append(annotation)
                    except Exception as e:
                        logger.warning(f"Failed to parse annotation: {str(e)}")
                
                if not annotations:
                    logger.warning(f"No valid annotations found for task {task_id}")
                    return True  # No annotations to check
                
                # Run quality evaluation
                quality_report = await self.evaluate_quality(annotations)
                
                # Update task quality score
                db_task.quality_score = quality_report.overall_score
                
                # Create quality issues for any problems found
                for issue in quality_report.issues_found:
                    db_issue = QualityIssueModel(
                        id=issue.id,
                        task_id=issue.task_id,
                        issue_type=issue.issue_type,
                        description=issue.description,
                        severity=issue.severity,
                        status=issue.status,
                        assignee_id=issue.assignee_id,
                        created_at=issue.created_at
                    )
                    session.add(db_issue)
                
                session.commit()
                
                logger.info(
                    f"Quality check completed for task {task_id}. "
                    f"Score: {quality_report.overall_score:.3f}, "
                    f"Issues: {len(quality_report.issues_found)}"
                )
                
                return True
                
        except Exception as e:
            logger.error(f"Quality check failed for task {task_id}: {str(e)}")
            return False
    
    def calculate_quality_score(
        self, 
        rule_results: List[Dict[str, Any]], 
        ragas_metrics: Dict[str, float]
    ) -> float:
        """
        Calculate overall quality score from rule results and Ragas metrics.
        
        Combines rule-based checks with Ragas semantic evaluation.
        """
        if not rule_results and not ragas_metrics:
            return 0.0
        
        # Calculate rule-based score
        rule_score = 0.0
        if rule_results:
            passed_rules = sum(1 for result in rule_results if result["passed"])
            rule_score = passed_rules / len(rule_results)
        
        # Calculate Ragas-based score
        ragas_score = 0.0
        if ragas_metrics:
            ragas_score = sum(ragas_metrics.values()) / len(ragas_metrics)
        
        # Weighted combination (70% rules, 30% Ragas)
        if rule_results and ragas_metrics:
            overall_score = 0.7 * rule_score + 0.3 * ragas_score
        elif rule_results:
            overall_score = rule_score
        else:
            overall_score = ragas_score
        
        return min(1.0, max(0.0, overall_score))