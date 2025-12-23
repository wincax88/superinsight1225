#!/usr/bin/env python3
"""
Demonstration script for SuperInsight Platform data models.

Shows how to create, serialize, and deserialize the core data models.
"""

import json
from uuid import uuid4
from datetime import datetime

from src.models import Document, Task, Annotation, QualityIssue
from src.models.task import TaskStatus
from src.models.quality_issue import IssueSeverity, IssueStatus


def demo_document_model():
    """Demonstrate Document model usage."""
    print("=== Document Model Demo ===")
    
    # Create a document
    doc = Document(
        source_type="database",
        source_config={
            "host": "localhost",
            "database": "customer_db",
            "table": "feedback"
        },
        content="这个产品很好用，推荐给大家！",
        metadata={
            "language": "zh-CN",
            "category": "customer_feedback",
            "sentiment": "positive"
        }
    )
    
    print(f"Created document with ID: {doc.id}")
    print(f"Source type: {doc.source_type}")
    print(f"Content: {doc.content}")
    
    # Serialize to JSON
    doc_json = json.dumps(doc.to_dict(), ensure_ascii=False, indent=2)
    print(f"Serialized to JSON:\n{doc_json}")
    
    # Deserialize from JSON
    doc_dict = json.loads(doc_json)
    doc_restored = Document.from_dict(doc_dict)
    print(f"Restored document ID: {doc_restored.id}")
    print(f"Content matches: {doc.content == doc_restored.content}")
    print()


def demo_task_model():
    """Demonstrate Task model usage."""
    print("=== Task Model Demo ===")
    
    # Create a task
    doc_id = uuid4()
    task = Task(
        document_id=doc_id,
        project_id="sentiment_analysis_project",
        ai_predictions=[
            {
                "model": "sentiment_classifier_v1",
                "prediction": "positive",
                "confidence": 0.85
            }
        ]
    )
    
    print(f"Created task with ID: {task.id}")
    print(f"Document ID: {task.document_id}")
    print(f"Status: {task.status}")
    print(f"AI predictions: {task.ai_predictions}")
    
    # Add annotation and update status
    task.add_annotation({
        "label": "positive",
        "confidence": 0.95,
        "annotator": "expert_user_123"
    })
    
    print(f"Status after annotation: {task.status}")
    print(f"Annotations: {task.annotations}")
    
    # Serialize and deserialize
    task_dict = task.to_dict()
    task_restored = Task.from_dict(task_dict)
    print(f"Serialization successful: {task.id == task_restored.id}")
    print()


def demo_annotation_model():
    """Demonstrate Annotation model usage."""
    print("=== Annotation Model Demo ===")
    
    # Create an annotation
    task_id = uuid4()
    annotation = Annotation(
        task_id=task_id,
        annotator_id="expert_annotator_456",
        annotation_data={
            "sentiment": "positive",
            "entities": [
                {
                    "text": "产品",
                    "label": "PRODUCT",
                    "start": 2,
                    "end": 4
                }
            ],
            "confidence_score": 0.95
        },
        confidence=0.95,
        time_spent=180  # 3 minutes
    )
    
    print(f"Created annotation with ID: {annotation.id}")
    print(f"Annotator: {annotation.annotator_id}")
    print(f"Confidence: {annotation.confidence}")
    print(f"Time spent: {annotation.time_spent} seconds")
    print(f"Annotation data: {annotation.annotation_data}")
    
    # Update confidence and time
    annotation.update_confidence(0.98)
    annotation.add_time_spent(30)
    
    print(f"Updated confidence: {annotation.confidence}")
    print(f"Updated time spent: {annotation.time_spent} seconds")
    
    # Serialize and deserialize
    ann_dict = annotation.to_dict()
    ann_restored = Annotation.from_dict(ann_dict)
    print(f"Serialization successful: {annotation.id == ann_restored.id}")
    print()


def demo_quality_issue_model():
    """Demonstrate QualityIssue model usage."""
    print("=== Quality Issue Model Demo ===")
    
    # Create a quality issue
    task_id = uuid4()
    issue = QualityIssue(
        task_id=task_id,
        issue_type="annotation_inconsistency",
        description="标注结果与预期不符，需要重新审核实体识别部分",
        severity=IssueSeverity.HIGH
    )
    
    print(f"Created quality issue with ID: {issue.id}")
    print(f"Issue type: {issue.issue_type}")
    print(f"Severity: {issue.severity}")
    print(f"Status: {issue.status}")
    print(f"Description: {issue.description}")
    
    # Assign and resolve issue
    issue.assign_to("quality_expert_789")
    print(f"Assigned to: {issue.assignee_id}")
    print(f"Status after assignment: {issue.status}")
    
    issue.resolve()
    print(f"Status after resolution: {issue.status}")
    print(f"Resolved at: {issue.resolved_at}")
    print(f"Is resolved: {issue.is_resolved()}")
    
    # Serialize and deserialize
    issue_dict = issue.to_dict()
    issue_restored = QualityIssue.from_dict(issue_dict)
    print(f"Serialization successful: {issue.id == issue_restored.id}")
    print()


def demo_complete_workflow():
    """Demonstrate a complete annotation workflow."""
    print("=== Complete Workflow Demo ===")
    
    # 1. Create a document
    doc = Document(
        source_type="file",
        source_config={"path": "/data/customer_reviews.csv"},
        content="这款手机的摄像头效果非常好，电池续航也很棒！",
        metadata={"source": "customer_review", "product": "smartphone"}
    )
    print(f"1. Created document: {doc.id}")
    
    # 2. Create annotation task
    task = Task(
        document_id=doc.id,
        project_id="product_review_analysis"
    )
    print(f"2. Created task: {task.id}")
    
    # 3. Add AI prediction
    task.add_ai_prediction({
        "model": "product_sentiment_v2",
        "sentiment": "positive",
        "entities": ["手机", "摄像头", "电池"],
        "confidence": 0.88
    })
    print(f"3. Added AI prediction, status: {task.status}")
    
    # 4. Create human annotation
    annotation = Annotation(
        task_id=task.id,
        annotator_id="human_expert_001",
        annotation_data={
            "sentiment": "positive",
            "entities": [
                {"text": "手机", "label": "PRODUCT", "start": 2, "end": 4},
                {"text": "摄像头", "label": "FEATURE", "start": 5, "end": 8},
                {"text": "电池", "label": "FEATURE", "start": 15, "end": 17}
            ],
            "overall_rating": 5
        },
        confidence=0.95,
        time_spent=240
    )
    print(f"4. Created annotation: {annotation.id}")
    
    # 5. Add annotation to task
    task.add_annotation(annotation.annotation_data)
    task.update_quality_score(0.92)
    task.mark_completed()
    print(f"5. Task completed with quality score: {task.quality_score}")
    
    # 6. Create quality issue (if needed)
    if task.quality_score < 0.95:
        issue = QualityIssue(
            task_id=task.id,
            issue_type="quality_threshold_not_met",
            description=f"质量评分 {task.quality_score} 低于阈值 0.95",
            severity=IssueSeverity.MEDIUM
        )
        print(f"6. Created quality issue: {issue.id}")
    else:
        print("6. No quality issues - task meets quality standards")
    
    print("Workflow completed successfully!")


if __name__ == "__main__":
    demo_document_model()
    demo_task_model()
    demo_annotation_model()
    demo_quality_issue_model()
    demo_complete_workflow()