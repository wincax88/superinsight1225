"""
Basic unit tests for Quality Management functionality without Ragas.

Tests the core QualityManager and DataRepairService implementations.
"""

import pytest
import asyncio
from datetime import datetime
from uuid import uuid4
from unittest.mock import Mock, patch, AsyncMock

from src.models.annotation import Annotation
from src.models.quality_issue import QualityIssue, IssueSeverity, IssueStatus


class TestQualityManagerBasic:
    """Basic test cases for QualityManager without Ragas dependencies."""
    
    def test_quality_issue_model(self):
        """Test QualityIssue model functionality."""
        task_id = uuid4()
        
        issue = QualityIssue(
            task_id=task_id,
            issue_type="test_issue",
            description="Test issue description",
            severity=IssueSeverity.HIGH
        )
        
        assert issue.task_id == task_id
        assert issue.issue_type == "test_issue"
        assert issue.severity == IssueSeverity.HIGH
        assert issue.status == IssueStatus.OPEN
        
        # Test assignment
        issue.assign_to("test_user")
        assert issue.assignee_id == "test_user"
        assert issue.status == IssueStatus.IN_PROGRESS
        
        # Test resolution
        issue.resolve()
        assert issue.status == IssueStatus.RESOLVED
        assert issue.resolved_at is not None
    
    def test_annotation_model(self):
        """Test Annotation model functionality."""
        task_id = uuid4()
        
        annotation = Annotation(
            task_id=task_id,
            annotator_id="test_user",
            annotation_data={"label": "positive", "confidence": 0.9},
            confidence=0.9,
            time_spent=120
        )
        
        assert annotation.task_id == task_id
        assert annotation.annotator_id == "test_user"
        assert annotation.confidence == 0.9
        assert annotation.time_spent == 120
        
        # Test serialization
        data_dict = annotation.to_dict()
        assert "id" in data_dict
        assert "task_id" in data_dict
        assert data_dict["annotator_id"] == "test_user"
        
        # Test deserialization
        restored = Annotation.from_dict(data_dict)
        assert restored.annotator_id == annotation.annotator_id
        assert restored.confidence == annotation.confidence
    
    def test_quality_issue_serialization(self):
        """Test QualityIssue serialization and deserialization."""
        task_id = uuid4()
        
        issue = QualityIssue(
            task_id=task_id,
            issue_type="annotation_error",
            description="标注错误需要修复",
            severity=IssueSeverity.MEDIUM
        )
        
        # Test to_dict
        data_dict = issue.to_dict()
        assert data_dict["task_id"] == str(task_id)
        assert data_dict["issue_type"] == "annotation_error"
        assert data_dict["severity"] == "medium"
        
        # Test from_dict
        restored = QualityIssue.from_dict(data_dict)
        assert restored.task_id == issue.task_id
        assert restored.issue_type == issue.issue_type
        assert restored.severity == issue.severity
    
    def test_quality_issue_workflow(self):
        """Test quality issue workflow operations."""
        issue = QualityIssue(
            task_id=uuid4(),
            issue_type="test_issue",
            description="Test workflow"
        )
        
        # Initial state
        assert issue.status == IssueStatus.OPEN
        assert not issue.is_resolved()
        
        # Assign
        issue.assign_to("expert_user")
        assert issue.status == IssueStatus.IN_PROGRESS
        assert issue.assignee_id == "expert_user"
        
        # Resolve
        issue.resolve()
        assert issue.status == IssueStatus.RESOLVED
        assert issue.is_resolved()
        assert issue.resolved_at is not None
        
        # Reopen
        issue.reopen()
        assert issue.status == IssueStatus.OPEN
        assert not issue.is_resolved()
        assert issue.resolved_at is None
        
        # Close
        issue.close()
        assert issue.status == IssueStatus.CLOSED
        assert issue.is_resolved()
    
    def test_annotation_validation(self):
        """Test annotation data validation."""
        task_id = uuid4()
        
        # Valid annotation
        annotation = Annotation(
            task_id=task_id,
            annotator_id="test_user",
            annotation_data={"label": "test"},
            confidence=0.8
        )
        assert annotation.confidence == 0.8
        
        # Test confidence validation
        with pytest.raises(ValueError):
            Annotation(
                task_id=task_id,
                annotator_id="test_user",
                annotation_data={"label": "test"},
                confidence=1.5  # Invalid confidence > 1.0
            )
        
        with pytest.raises(ValueError):
            Annotation(
                task_id=task_id,
                annotator_id="test_user",
                annotation_data={"label": "test"},
                confidence=-0.1  # Invalid confidence < 0.0
            )
        
        # Test empty annotator_id
        with pytest.raises(ValueError):
            Annotation(
                task_id=task_id,
                annotator_id="",  # Empty annotator_id
                annotation_data={"label": "test"}
            )
        
        # Test empty annotation_data
        with pytest.raises(ValueError):
            Annotation(
                task_id=task_id,
                annotator_id="test_user",
                annotation_data={}  # Empty annotation_data
            )
    
    def test_quality_issue_validation(self):
        """Test quality issue validation."""
        task_id = uuid4()
        
        # Valid issue
        issue = QualityIssue(
            task_id=task_id,
            issue_type="test_issue",
            description="Test description"
        )
        assert issue.issue_type == "test_issue"
        
        # Test empty issue_type
        with pytest.raises(ValueError):
            QualityIssue(
                task_id=task_id,
                issue_type="",  # Empty issue_type
                description="Test description"
            )
        
        # Test whitespace-only issue_type
        with pytest.raises(ValueError):
            QualityIssue(
                task_id=task_id,
                issue_type="   ",  # Whitespace-only issue_type
                description="Test description"
            )
        
        # Test invalid assignee_id
        with pytest.raises(ValueError):
            issue = QualityIssue(
                task_id=task_id,
                issue_type="test_issue",
                description="Test description"
            )
            issue.assign_to("")  # Empty assignee_id
    
    def test_annotation_operations(self):
        """Test annotation update operations."""
        annotation = Annotation(
            task_id=uuid4(),
            annotator_id="test_user",
            annotation_data={"label": "initial"},
            confidence=0.7,
            time_spent=60
        )
        
        # Update confidence
        annotation.update_confidence(0.9)
        assert annotation.confidence == 0.9
        
        # Invalid confidence update
        with pytest.raises(ValueError):
            annotation.update_confidence(1.5)
        
        # Add time spent
        annotation.add_time_spent(30)
        assert annotation.time_spent == 90
        
        # Invalid time addition
        with pytest.raises(ValueError):
            annotation.add_time_spent(-10)
        
        # Update annotation data
        new_data = {"label": "updated", "confidence": 0.95}
        annotation.update_annotation_data(new_data)
        assert annotation.annotation_data == new_data
        
        # Invalid annotation data update
        with pytest.raises(ValueError):
            annotation.update_annotation_data({})


class TestDataRepairBasic:
    """Basic test cases for data repair functionality."""
    
    def test_repair_status_enum(self):
        """Test repair status enumeration."""
        # Import directly to avoid Ragas import issues
        from src.quality.repair import RepairStatus, RepairType
        
        # Test all status values
        assert RepairStatus.PENDING == "pending"
        assert RepairStatus.IN_PROGRESS == "in_progress"
        assert RepairStatus.COMPLETED == "completed"
        assert RepairStatus.FAILED == "failed"
        assert RepairStatus.REJECTED == "rejected"
        
        # Test all repair types
        assert RepairType.ANNOTATION_CORRECTION == "annotation_correction"
        assert RepairType.SOURCE_DATA_UPDATE == "source_data_update"
        assert RepairType.METADATA_REPAIR == "metadata_repair"
        assert RepairType.QUALITY_SCORE_ADJUSTMENT == "quality_score_adjustment"
        assert RepairType.LABEL_STANDARDIZATION == "label_standardization"
    
    def test_repair_record_creation(self):
        """Test repair record creation and serialization."""
        from src.quality.repair import RepairRecord, RepairType, RepairStatus
        
        repair_id = uuid4()
        quality_issue_id = uuid4()
        
        record = RepairRecord(
            repair_id=repair_id,
            quality_issue_id=quality_issue_id,
            repair_type=RepairType.ANNOTATION_CORRECTION,
            description="Fix annotation error",
            original_data={"label": "wrong"},
            proposed_data={"label": "correct"},
            requested_by="test_user"
        )
        
        assert record.repair_id == repair_id
        assert record.quality_issue_id == quality_issue_id
        assert record.repair_type == RepairType.ANNOTATION_CORRECTION
        assert record.status == RepairStatus.PENDING
        assert record.requested_by == "test_user"
        
        # Test serialization
        data_dict = record.to_dict()
        assert data_dict["repair_id"] == str(repair_id)
        assert data_dict["repair_type"] == "annotation_correction"
        assert data_dict["status"] == "pending"
    
    def test_approval_workflow_basic(self):
        """Test basic approval workflow functionality."""
        from src.quality.repair import RepairApprovalWorkflow, RepairType
        
        workflow = RepairApprovalWorkflow()
        
        # Test approval requirements
        assert workflow.requires_approval(RepairType.SOURCE_DATA_UPDATE, 0.9) is True
        assert workflow.requires_approval(RepairType.METADATA_REPAIR, 0.9) is False
        
        # Test auto-approval thresholds
        assert workflow.requires_approval(RepairType.ANNOTATION_CORRECTION, 0.95) is False
        assert workflow.requires_approval(RepairType.ANNOTATION_CORRECTION, 0.5) is True
        
        # Test required approver roles
        roles = workflow.get_required_approver_roles(RepairType.SOURCE_DATA_UPDATE)
        assert isinstance(roles, list)
        assert len(roles) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])