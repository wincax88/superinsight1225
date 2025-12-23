"""
Basic tests for core data models.

Tests JSON serialization/deserialization and data validation.
"""

import pytest
from datetime import datetime
from uuid import UUID, uuid4
from hypothesis import given, strategies as st
from typing import Dict, Any

from src.models import Document, Task, Annotation, QualityIssue
from src.models.task import TaskStatus
from src.models.quality_issue import IssueSeverity, IssueStatus


class TestDocument:
    """Tests for Document model."""
    
    def test_document_creation(self):
        """Test creating a valid document."""
        doc = Document(
            source_type="database",
            source_config={"host": "localhost", "database": "test_db"},
            content="Test content"
        )
        assert doc.id is not None
        assert doc.source_type == "database"
        assert doc.content == "Test content"
        assert isinstance(doc.created_at, datetime)
    
    def test_document_serialization(self):
        """Test document to_dict and from_dict methods."""
        doc = Document(
            source_type="file",
            source_config={"path": "/data/test.pdf"},
            content="Sample content",
            metadata={"language": "zh-CN"}
        )
        
        # Serialize to dict
        doc_dict = doc.to_dict()
        assert isinstance(doc_dict["id"], str)
        assert doc_dict["source_type"] == "file"
        assert doc_dict["metadata"]["language"] == "zh-CN"
        
        # Deserialize from dict
        doc_restored = Document.from_dict(doc_dict)
        assert doc_restored.id == doc.id
        assert doc_restored.source_type == doc.source_type
        assert doc_restored.content == doc.content
    
    def test_document_invalid_source_type(self):
        """Test that invalid source_type raises error."""
        with pytest.raises(ValueError, match="source_type must be one of"):
            Document(
                source_type="invalid_type",
                source_config={},
                content="Test"
            )
    
    def test_document_empty_content(self):
        """Test that empty content raises error."""
        with pytest.raises(ValueError, match="content cannot be empty"):
            Document(
                source_type="database",
                source_config={},
                content=""
            )
    
    def test_document_whitespace_only_content(self):
        """Test that whitespace-only content raises error."""
        with pytest.raises(ValueError, match="content cannot be empty"):
            Document(
                source_type="database",
                source_config={},
                content="   \n\t  "
            )
    
    def test_document_update_content(self):
        """Test updating document content."""
        doc = Document(
            source_type="file",
            source_config={"path": "/test.txt"},
            content="Original content"
        )
        original_updated_at = doc.updated_at
        
        # Update content
        doc.update_content("New content")
        assert doc.content == "New content"
        assert doc.updated_at > original_updated_at
    
    def test_document_add_metadata(self):
        """Test adding metadata to document."""
        doc = Document(
            source_type="api",
            source_config={"url": "https://api.example.com"},
            content="API response content"
        )
        original_updated_at = doc.updated_at
        
        # Add metadata
        doc.add_metadata("category", "customer_feedback")
        assert doc.metadata["category"] == "customer_feedback"
        assert doc.updated_at > original_updated_at
    
    def test_document_complex_source_config(self):
        """Test document with complex source configuration."""
        complex_config = {
            "host": "db.example.com",
            "port": 5432,
            "database": "production",
            "table": "user_feedback",
            "query": "SELECT * FROM feedback WHERE created_at > ?",
            "parameters": ["2024-01-01"],
            "connection_options": {
                "ssl": True,
                "timeout": 30
            }
        }
        
        doc = Document(
            source_type="database",
            source_config=complex_config,
            content="Database query result"
        )
        
        assert doc.source_config == complex_config
        
        # Test serialization with complex config
        doc_dict = doc.to_dict()
        restored_doc = Document.from_dict(doc_dict)
        assert restored_doc.source_config == complex_config
    
    def test_document_serialization_with_special_characters(self):
        """Test serialization with special characters and unicode."""
        doc = Document(
            source_type="file",
            source_config={"path": "/数据/测试文件.pdf"},
            content="包含中文和特殊字符的内容：！@#￥%……&*（）",
            metadata={"语言": "中文", "类别": "测试数据"}
        )
        
        # Test serialization preserves unicode
        doc_dict = doc.to_dict()
        restored_doc = Document.from_dict(doc_dict)
        
        assert restored_doc.content == doc.content
        assert restored_doc.metadata == doc.metadata
        assert restored_doc.source_config == doc.source_config


class TestTask:
    """Tests for Task model."""
    
    def test_task_creation(self):
        """Test creating a valid task."""
        doc_id = uuid4()
        task = Task(
            document_id=doc_id,
            project_id="test_project"
        )
        assert task.id is not None
        assert task.document_id == doc_id
        assert task.status == TaskStatus.PENDING
        assert task.quality_score == 0.0
    
    def test_task_serialization(self):
        """Test task to_dict and from_dict methods."""
        doc_id = uuid4()
        task = Task(
            document_id=doc_id,
            project_id="test_project",
            annotations=[{"label": "positive"}],
            quality_score=0.85
        )
        
        # Serialize to dict
        task_dict = task.to_dict()
        assert isinstance(task_dict["id"], str)
        assert task_dict["status"] == "pending"
        assert task_dict["quality_score"] == 0.85
        
        # Deserialize from dict
        task_restored = Task.from_dict(task_dict)
        assert task_restored.id == task.id
        assert task_restored.document_id == task.document_id
        assert task_restored.quality_score == task.quality_score
    
    def test_task_quality_score_validation(self):
        """Test that quality_score must be between 0.0 and 1.0."""
        doc_id = uuid4()
        with pytest.raises(ValueError, match="quality_score must be between"):
            Task(
                document_id=doc_id,
                project_id="test",
                quality_score=1.5
            )
    
    def test_task_empty_project_id(self):
        """Test that empty project_id raises error."""
        doc_id = uuid4()
        with pytest.raises(ValueError, match="project_id cannot be empty"):
            Task(
                document_id=doc_id,
                project_id=""
            )
    
    def test_task_whitespace_project_id(self):
        """Test that whitespace-only project_id raises error."""
        doc_id = uuid4()
        with pytest.raises(ValueError, match="project_id cannot be empty"):
            Task(
                document_id=doc_id,
                project_id="   \n\t  "
            )
    
    def test_task_negative_quality_score(self):
        """Test that negative quality_score raises error."""
        doc_id = uuid4()
        with pytest.raises(ValueError, match="quality_score must be between"):
            Task(
                document_id=doc_id,
                project_id="test_project",
                quality_score=-0.1
            )
    
    def test_task_boundary_quality_scores(self):
        """Test boundary values for quality_score."""
        doc_id = uuid4()
        
        # Test minimum boundary (0.0)
        task_min = Task(
            document_id=doc_id,
            project_id="test_project",
            quality_score=0.0
        )
        assert task_min.quality_score == 0.0
        
        # Test maximum boundary (1.0)
        task_max = Task(
            document_id=doc_id,
            project_id="test_project",
            quality_score=1.0
        )
        assert task_max.quality_score == 1.0
    
    def test_task_add_annotation(self):
        """Test adding annotation to task."""
        doc_id = uuid4()
        task = Task(
            document_id=doc_id,
            project_id="test_project"
        )
        
        assert task.status == TaskStatus.PENDING
        assert len(task.annotations) == 0
        
        # Add annotation
        annotation_data = {"label": "positive", "confidence": 0.9}
        task.add_annotation(annotation_data)
        
        assert len(task.annotations) == 1
        assert task.annotations[0] == annotation_data
        assert task.status == TaskStatus.IN_PROGRESS
    
    def test_task_add_ai_prediction(self):
        """Test adding AI prediction to task."""
        doc_id = uuid4()
        task = Task(
            document_id=doc_id,
            project_id="test_project"
        )
        
        assert len(task.ai_predictions) == 0
        
        # Add AI prediction
        prediction_data = {"model": "bert", "prediction": "positive", "confidence": 0.85}
        task.add_ai_prediction(prediction_data)
        
        assert len(task.ai_predictions) == 1
        assert task.ai_predictions[0] == prediction_data
    
    def test_task_update_quality_score(self):
        """Test updating task quality score."""
        doc_id = uuid4()
        task = Task(
            document_id=doc_id,
            project_id="test_project"
        )
        
        # Valid update
        task.update_quality_score(0.75)
        assert task.quality_score == 0.75
        
        # Invalid update should raise error
        with pytest.raises(ValueError, match="Quality score must be between"):
            task.update_quality_score(1.5)
    
    def test_task_status_transitions(self):
        """Test task status transitions."""
        doc_id = uuid4()
        task = Task(
            document_id=doc_id,
            project_id="test_project"
        )
        
        assert task.status == TaskStatus.PENDING
        
        # Mark completed
        task.mark_completed()
        assert task.status == TaskStatus.COMPLETED
        
        # Mark reviewed
        task.mark_reviewed()
        assert task.status == TaskStatus.REVIEWED
    
    def test_task_serialization_with_complex_data(self):
        """Test serialization with complex annotation and prediction data."""
        doc_id = uuid4()
        complex_annotations = [
            {
                "label": "sentiment",
                "value": "positive",
                "entities": [
                    {"text": "产品", "label": "PRODUCT", "start": 0, "end": 2},
                    {"text": "很好", "label": "OPINION", "start": 3, "end": 5}
                ],
                "confidence": 0.95
            }
        ]
        
        complex_predictions = [
            {
                "model_name": "chinese_sentiment_v2",
                "model_version": "1.2.3",
                "predictions": {
                    "sentiment": {"positive": 0.85, "negative": 0.15},
                    "entities": [{"text": "产品", "label": "PRODUCT", "confidence": 0.92}]
                }
            }
        ]
        
        task = Task(
            document_id=doc_id,
            project_id="complex_nlp_project",
            annotations=complex_annotations,
            ai_predictions=complex_predictions,
            quality_score=0.88
        )
        
        # Test serialization
        task_dict = task.to_dict()
        restored_task = Task.from_dict(task_dict)
        
        assert restored_task.annotations == complex_annotations
        assert restored_task.ai_predictions == complex_predictions
        assert restored_task.quality_score == 0.88


class TestAnnotation:
    """Tests for Annotation model."""
    
    def test_annotation_creation(self):
        """Test creating a valid annotation."""
        task_id = uuid4()
        annotation = Annotation(
            task_id=task_id,
            annotator_id="user123",
            annotation_data={"label": "positive", "score": 0.9}
        )
        assert annotation.id is not None
        assert annotation.task_id == task_id
        assert annotation.confidence == 1.0
        assert annotation.time_spent == 0
    
    def test_annotation_serialization(self):
        """Test annotation to_dict and from_dict methods."""
        task_id = uuid4()
        annotation = Annotation(
            task_id=task_id,
            annotator_id="user123",
            annotation_data={"label": "positive"},
            confidence=0.95,
            time_spent=120
        )
        
        # Serialize to dict
        ann_dict = annotation.to_dict()
        assert isinstance(ann_dict["id"], str)
        assert ann_dict["confidence"] == 0.95
        assert ann_dict["time_spent"] == 120
        
        # Deserialize from dict
        ann_restored = Annotation.from_dict(ann_dict)
        assert ann_restored.id == annotation.id
        assert ann_restored.task_id == annotation.task_id
        assert ann_restored.confidence == annotation.confidence
    
    def test_annotation_confidence_validation(self):
        """Test that confidence must be between 0.0 and 1.0."""
        task_id = uuid4()
        with pytest.raises(ValueError, match="confidence must be between"):
            Annotation(
                task_id=task_id,
                annotator_id="user123",
                annotation_data={"label": "test"},
                confidence=1.5
            )
    
    def test_annotation_empty_data(self):
        """Test that annotation_data cannot be empty."""
        task_id = uuid4()
        with pytest.raises(ValueError, match="annotation_data cannot be empty"):
            Annotation(
                task_id=task_id,
                annotator_id="user123",
                annotation_data={}
            )
    
    def test_annotation_empty_annotator_id(self):
        """Test that annotator_id cannot be empty."""
        task_id = uuid4()
        with pytest.raises(ValueError, match="annotator_id cannot be empty"):
            Annotation(
                task_id=task_id,
                annotator_id="",
                annotation_data={"label": "test"}
            )
    
    def test_annotation_whitespace_annotator_id(self):
        """Test that whitespace-only annotator_id raises error."""
        task_id = uuid4()
        with pytest.raises(ValueError, match="annotator_id cannot be empty"):
            Annotation(
                task_id=task_id,
                annotator_id="   \n\t  ",
                annotation_data={"label": "test"}
            )
    
    def test_annotation_negative_confidence(self):
        """Test that negative confidence raises error."""
        task_id = uuid4()
        with pytest.raises(ValueError, match="confidence must be between"):
            Annotation(
                task_id=task_id,
                annotator_id="user123",
                annotation_data={"label": "test"},
                confidence=-0.1
            )
    
    def test_annotation_negative_time_spent(self):
        """Test that negative time_spent raises error."""
        task_id = uuid4()
        with pytest.raises(ValueError, match="time_spent must be non-negative"):
            Annotation(
                task_id=task_id,
                annotator_id="user123",
                annotation_data={"label": "test"},
                time_spent=-1
            )
    
    def test_annotation_boundary_values(self):
        """Test boundary values for confidence and time_spent."""
        task_id = uuid4()
        
        # Test minimum confidence (0.0)
        ann_min_conf = Annotation(
            task_id=task_id,
            annotator_id="user123",
            annotation_data={"label": "test"},
            confidence=0.0
        )
        assert ann_min_conf.confidence == 0.0
        
        # Test maximum confidence (1.0)
        ann_max_conf = Annotation(
            task_id=task_id,
            annotator_id="user123",
            annotation_data={"label": "test"},
            confidence=1.0
        )
        assert ann_max_conf.confidence == 1.0
        
        # Test minimum time_spent (0)
        ann_min_time = Annotation(
            task_id=task_id,
            annotator_id="user123",
            annotation_data={"label": "test"},
            time_spent=0
        )
        assert ann_min_time.time_spent == 0
    
    def test_annotation_update_confidence(self):
        """Test updating annotation confidence."""
        task_id = uuid4()
        annotation = Annotation(
            task_id=task_id,
            annotator_id="user123",
            annotation_data={"label": "positive"}
        )
        
        # Valid update
        annotation.update_confidence(0.85)
        assert annotation.confidence == 0.85
        
        # Invalid update should raise error
        with pytest.raises(ValueError, match="Confidence must be between"):
            annotation.update_confidence(1.5)
    
    def test_annotation_add_time_spent(self):
        """Test adding time spent to annotation."""
        task_id = uuid4()
        annotation = Annotation(
            task_id=task_id,
            annotator_id="user123",
            annotation_data={"label": "positive"},
            time_spent=60
        )
        
        # Valid addition
        annotation.add_time_spent(30)
        assert annotation.time_spent == 90
        
        # Invalid addition should raise error
        with pytest.raises(ValueError, match="Additional time must be non-negative"):
            annotation.add_time_spent(-10)
    
    def test_annotation_update_annotation_data(self):
        """Test updating annotation data."""
        task_id = uuid4()
        annotation = Annotation(
            task_id=task_id,
            annotator_id="user123",
            annotation_data={"label": "positive"}
        )
        
        # Valid update
        new_data = {"label": "negative", "confidence": 0.8, "entities": []}
        annotation.update_annotation_data(new_data)
        assert annotation.annotation_data == new_data
        
        # Invalid update should raise error
        with pytest.raises(ValueError, match="Annotation data cannot be empty"):
            annotation.update_annotation_data({})
    
    def test_annotation_complex_data_serialization(self):
        """Test serialization with complex annotation data."""
        task_id = uuid4()
        complex_data = {
            "sentiment": {
                "label": "positive",
                "confidence": 0.92,
                "reasoning": "积极的情感词汇较多"
            },
            "entities": [
                {
                    "text": "iPhone",
                    "label": "PRODUCT",
                    "start": 5,
                    "end": 11,
                    "confidence": 0.98
                },
                {
                    "text": "苹果公司",
                    "label": "ORGANIZATION",
                    "start": 15,
                    "end": 19,
                    "confidence": 0.95
                }
            ],
            "keywords": ["手机", "科技", "评价"],
            "metadata": {
                "language": "zh-CN",
                "domain": "technology",
                "annotator_notes": "清晰的产品评价文本"
            }
        }
        
        annotation = Annotation(
            task_id=task_id,
            annotator_id="expert_annotator_001",
            annotation_data=complex_data,
            confidence=0.94,
            time_spent=180
        )
        
        # Test serialization
        ann_dict = annotation.to_dict()
        restored_annotation = Annotation.from_dict(ann_dict)
        
        assert restored_annotation.annotation_data == complex_data
        assert restored_annotation.confidence == 0.94
        assert restored_annotation.time_spent == 180


class TestQualityIssue:
    """Tests for QualityIssue model."""
    
    def test_quality_issue_creation(self):
        """Test creating a valid quality issue."""
        task_id = uuid4()
        issue = QualityIssue(
            task_id=task_id,
            issue_type="annotation_inconsistency",
            description="标注不一致"
        )
        assert issue.id is not None
        assert issue.task_id == task_id
        assert issue.severity == IssueSeverity.MEDIUM
        assert issue.status == IssueStatus.OPEN
        assert issue.resolved_at is None
    
    def test_quality_issue_serialization(self):
        """Test quality issue to_dict and from_dict methods."""
        task_id = uuid4()
        issue = QualityIssue(
            task_id=task_id,
            issue_type="quality_check_failed",
            description="质量检查失败",
            severity=IssueSeverity.HIGH,
            assignee_id="expert123"
        )
        
        # Serialize to dict
        issue_dict = issue.to_dict()
        assert isinstance(issue_dict["id"], str)
        assert issue_dict["severity"] == "high"
        assert issue_dict["assignee_id"] == "expert123"
        
        # Deserialize from dict
        issue_restored = QualityIssue.from_dict(issue_dict)
        assert issue_restored.id == issue.id
        assert issue_restored.task_id == issue.task_id
        assert issue_restored.severity == issue.severity
    
    def test_quality_issue_resolve(self):
        """Test resolving a quality issue."""
        task_id = uuid4()
        issue = QualityIssue(
            task_id=task_id,
            issue_type="test_issue"
        )
        
        assert issue.resolved_at is None
        issue.resolve()
        assert issue.status == IssueStatus.RESOLVED
        assert issue.resolved_at is not None
        assert issue.is_resolved()
    
    def test_quality_issue_empty_issue_type(self):
        """Test that empty issue_type raises error."""
        task_id = uuid4()
        with pytest.raises(ValueError, match="issue_type cannot be empty"):
            QualityIssue(
                task_id=task_id,
                issue_type=""
            )
    
    def test_quality_issue_whitespace_issue_type(self):
        """Test that whitespace-only issue_type raises error."""
        task_id = uuid4()
        with pytest.raises(ValueError, match="issue_type cannot be empty"):
            QualityIssue(
                task_id=task_id,
                issue_type="   \n\t  "
            )
    
    def test_quality_issue_empty_assignee_id(self):
        """Test that empty string assignee_id raises error."""
        task_id = uuid4()
        with pytest.raises(ValueError, match="assignee_id cannot be empty string"):
            QualityIssue(
                task_id=task_id,
                issue_type="test_issue",
                assignee_id=""
            )
    
    def test_quality_issue_none_assignee_id(self):
        """Test that None assignee_id is allowed."""
        task_id = uuid4()
        issue = QualityIssue(
            task_id=task_id,
            issue_type="test_issue",
            assignee_id=None
        )
        assert issue.assignee_id is None
    
    def test_quality_issue_assign_to(self):
        """Test assigning issue to user."""
        task_id = uuid4()
        issue = QualityIssue(
            task_id=task_id,
            issue_type="test_issue"
        )
        
        assert issue.status == IssueStatus.OPEN
        assert issue.assignee_id is None
        
        # Assign to user
        issue.assign_to("expert123")
        assert issue.assignee_id == "expert123"
        assert issue.status == IssueStatus.IN_PROGRESS
        
        # Test invalid assignment
        with pytest.raises(ValueError, match="assignee_id cannot be empty"):
            issue.assign_to("")
    
    def test_quality_issue_close(self):
        """Test closing a quality issue."""
        task_id = uuid4()
        issue = QualityIssue(
            task_id=task_id,
            issue_type="test_issue"
        )
        
        assert issue.resolved_at is None
        issue.close()
        assert issue.status == IssueStatus.CLOSED
        assert issue.resolved_at is not None
        assert issue.is_resolved()
    
    def test_quality_issue_reopen(self):
        """Test reopening a resolved issue."""
        task_id = uuid4()
        issue = QualityIssue(
            task_id=task_id,
            issue_type="test_issue"
        )
        
        # First resolve it
        issue.resolve()
        assert issue.status == IssueStatus.RESOLVED
        assert issue.resolved_at is not None
        
        # Then reopen it
        issue.reopen()
        assert issue.status == IssueStatus.OPEN
        assert issue.resolved_at is None
        assert not issue.is_resolved()
    
    def test_quality_issue_update_severity(self):
        """Test updating issue severity."""
        task_id = uuid4()
        issue = QualityIssue(
            task_id=task_id,
            issue_type="test_issue",
            severity=IssueSeverity.LOW
        )
        
        assert issue.severity == IssueSeverity.LOW
        
        # Update severity
        issue.update_severity(IssueSeverity.CRITICAL)
        assert issue.severity == IssueSeverity.CRITICAL
    
    def test_quality_issue_status_transitions(self):
        """Test various status transitions."""
        task_id = uuid4()
        issue = QualityIssue(
            task_id=task_id,
            issue_type="complex_issue"
        )
        
        # Initial state
        assert issue.status == IssueStatus.OPEN
        assert not issue.is_resolved()
        
        # Assign (moves to IN_PROGRESS)
        issue.assign_to("expert123")
        assert issue.status == IssueStatus.IN_PROGRESS
        
        # Resolve
        issue.resolve()
        assert issue.status == IssueStatus.RESOLVED
        assert issue.is_resolved()
        
        # Reopen
        issue.reopen()
        assert issue.status == IssueStatus.OPEN
        assert not issue.is_resolved()
        
        # Close directly
        issue.close()
        assert issue.status == IssueStatus.CLOSED
        assert issue.is_resolved()
    
    def test_quality_issue_serialization_with_resolved_at(self):
        """Test serialization when resolved_at is set."""
        task_id = uuid4()
        issue = QualityIssue(
            task_id=task_id,
            issue_type="resolved_issue",
            description="This issue was resolved",
            severity=IssueSeverity.HIGH
        )
        
        # Resolve the issue
        issue.resolve()
        
        # Test serialization
        issue_dict = issue.to_dict()
        assert issue_dict["resolved_at"] is not None
        assert isinstance(issue_dict["resolved_at"], str)
        
        # Test deserialization
        restored_issue = QualityIssue.from_dict(issue_dict)
        assert restored_issue.resolved_at is not None
        assert restored_issue.status == IssueStatus.RESOLVED
        assert restored_issue.is_resolved()
    
    def test_quality_issue_all_severity_levels(self):
        """Test creating issues with all severity levels."""
        task_id = uuid4()
        
        for severity in IssueSeverity:
            issue = QualityIssue(
                task_id=task_id,
                issue_type=f"test_issue_{severity.value}",
                severity=severity
            )
            assert issue.severity == severity
            
            # Test serialization preserves severity
            issue_dict = issue.to_dict()
            restored_issue = QualityIssue.from_dict(issue_dict)
            assert restored_issue.severity == severity
    
    def test_quality_issue_all_status_values(self):
        """Test creating issues with all status values."""
        task_id = uuid4()
        
        for status in IssueStatus:
            issue = QualityIssue(
                task_id=task_id,
                issue_type=f"test_issue_{status.value}",
                status=status
            )
            assert issue.status == status
            
            # Test serialization preserves status
            issue_dict = issue.to_dict()
            restored_issue = QualityIssue.from_dict(issue_dict)
            assert restored_issue.status == status


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# Property-Based Tests for JSONB Storage Consistency

# Hypothesis strategies for generating test data

def valid_source_types():
    """Strategy for valid source types."""
    return st.sampled_from(['database', 'file', 'api'])

def json_compatible_dict():
    """Strategy for JSON-compatible dictionaries."""
    return st.dictionaries(
        keys=st.text(min_size=1, max_size=50),
        values=st.one_of(
            st.text(max_size=100),
            st.integers(),
            st.floats(allow_nan=False, allow_infinity=False),
            st.booleans(),
            st.none(),
            st.lists(st.text(max_size=50), max_size=5),
            st.dictionaries(
                keys=st.text(min_size=1, max_size=20),
                values=st.one_of(st.text(max_size=50), st.integers(), st.booleans()),
                max_size=3
            )
        ),
        min_size=0,
        max_size=10
    )

def document_strategy():
    """Strategy for generating Document instances."""
    return st.builds(
        Document,
        source_type=valid_source_types(),
        source_config=json_compatible_dict(),
        content=st.text(min_size=1, max_size=1000).filter(lambda x: x.strip()),  # Ensure content is not just whitespace
        metadata=json_compatible_dict()
    )

def task_strategy():
    """Strategy for generating Task instances."""
    return st.builds(
        Task,
        document_id=st.builds(uuid4),
        project_id=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
        status=st.sampled_from(list(TaskStatus)),
        annotations=st.lists(json_compatible_dict(), max_size=5),
        ai_predictions=st.lists(json_compatible_dict(), max_size=5),
        quality_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
    )

def annotation_strategy():
    """Strategy for generating Annotation instances."""
    return st.builds(
        Annotation,
        task_id=st.builds(uuid4),
        annotator_id=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
        annotation_data=json_compatible_dict().filter(lambda x: len(x) > 0),
        confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        time_spent=st.integers(min_value=0, max_value=86400)  # Max 24 hours
    )

def quality_issue_strategy():
    """Strategy for generating QualityIssue instances."""
    return st.builds(
        QualityIssue,
        task_id=st.builds(uuid4),
        issue_type=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
        description=st.one_of(st.none(), st.text(max_size=500)),
        severity=st.sampled_from(list(IssueSeverity)),
        status=st.sampled_from(list(IssueStatus)),
        assignee_id=st.one_of(st.none(), st.text(min_size=1, max_size=100).filter(lambda x: x.strip()))
    )


class TestJSONBStorageConsistency:
    """
    Property-based tests for JSONB storage consistency.
    
    Validates Requirements 2.1, 2.2, 2.3:
    - PostgreSQL JSONB format storage of original data
    - PostgreSQL JSONB format storage of annotation results and labels  
    - PostgreSQL JSONB format storage of enhanced quality data
    """
    
    @given(document_strategy())
    def test_document_jsonb_storage_consistency(self, document: Document):
        """
        Property 3: JSONB Storage Consistency for Documents
        
        For any Document instance, serializing to dict (simulating JSONB storage)
        and then deserializing should produce an equivalent Document.
        
        **Validates: Requirements 2.1, 2.2, 2.3**
        """
        # Simulate JSONB storage by converting to dict (JSON serialization)
        stored_dict = document.to_dict()
        
        # Simulate reading from JSONB storage by converting back to Document
        restored_document = Document.from_dict(stored_dict)
        
        # Verify all fields are preserved
        assert restored_document.id == document.id
        assert restored_document.source_type == document.source_type
        assert restored_document.source_config == document.source_config
        assert restored_document.content == document.content
        assert restored_document.metadata == document.metadata
        
        # Verify timestamps are preserved (within reasonable precision)
        assert abs((restored_document.created_at - document.created_at).total_seconds()) < 1
        assert abs((restored_document.updated_at - document.updated_at).total_seconds()) < 1
    
    @given(task_strategy())
    def test_task_jsonb_storage_consistency(self, task: Task):
        """
        Property 3: JSONB Storage Consistency for Tasks
        
        For any Task instance, serializing to dict (simulating JSONB storage)
        and then deserializing should produce an equivalent Task.
        
        **Validates: Requirements 2.1, 2.2, 2.3**
        """
        # Simulate JSONB storage by converting to dict (JSON serialization)
        stored_dict = task.to_dict()
        
        # Simulate reading from JSONB storage by converting back to Task
        restored_task = Task.from_dict(stored_dict)
        
        # Verify all fields are preserved
        assert restored_task.id == task.id
        assert restored_task.document_id == task.document_id
        assert restored_task.project_id == task.project_id
        assert restored_task.status == task.status
        assert restored_task.annotations == task.annotations
        assert restored_task.ai_predictions == task.ai_predictions
        assert restored_task.quality_score == task.quality_score
        
        # Verify timestamp is preserved (within reasonable precision)
        assert abs((restored_task.created_at - task.created_at).total_seconds()) < 1
    
    @given(annotation_strategy())
    def test_annotation_jsonb_storage_consistency(self, annotation: Annotation):
        """
        Property 3: JSONB Storage Consistency for Annotations
        
        For any Annotation instance, serializing to dict (simulating JSONB storage)
        and then deserializing should produce an equivalent Annotation.
        
        **Validates: Requirements 2.1, 2.2, 2.3**
        """
        # Simulate JSONB storage by converting to dict (JSON serialization)
        stored_dict = annotation.to_dict()
        
        # Simulate reading from JSONB storage by converting back to Annotation
        restored_annotation = Annotation.from_dict(stored_dict)
        
        # Verify all fields are preserved
        assert restored_annotation.id == annotation.id
        assert restored_annotation.task_id == annotation.task_id
        assert restored_annotation.annotator_id == annotation.annotator_id
        assert restored_annotation.annotation_data == annotation.annotation_data
        assert restored_annotation.confidence == annotation.confidence
        assert restored_annotation.time_spent == annotation.time_spent
        
        # Verify timestamp is preserved (within reasonable precision)
        assert abs((restored_annotation.created_at - annotation.created_at).total_seconds()) < 1
    
    @given(quality_issue_strategy())
    def test_quality_issue_jsonb_storage_consistency(self, quality_issue: QualityIssue):
        """
        Property 3: JSONB Storage Consistency for Quality Issues
        
        For any QualityIssue instance, serializing to dict (simulating JSONB storage)
        and then deserializing should produce an equivalent QualityIssue.
        
        **Validates: Requirements 2.1, 2.2, 2.3**
        """
        # Simulate JSONB storage by converting to dict (JSON serialization)
        stored_dict = quality_issue.to_dict()
        
        # Simulate reading from JSONB storage by converting back to QualityIssue
        restored_issue = QualityIssue.from_dict(stored_dict)
        
        # Verify all fields are preserved
        assert restored_issue.id == quality_issue.id
        assert restored_issue.task_id == quality_issue.task_id
        assert restored_issue.issue_type == quality_issue.issue_type
        assert restored_issue.description == quality_issue.description
        assert restored_issue.severity == quality_issue.severity
        assert restored_issue.status == quality_issue.status
        assert restored_issue.assignee_id == quality_issue.assignee_id
        
        # Verify timestamps are preserved (within reasonable precision)
        assert abs((restored_issue.created_at - quality_issue.created_at).total_seconds()) < 1
        
        # Handle optional resolved_at timestamp
        if quality_issue.resolved_at is not None:
            assert restored_issue.resolved_at is not None
            assert abs((restored_issue.resolved_at - quality_issue.resolved_at).total_seconds()) < 1
        else:
            assert restored_issue.resolved_at is None