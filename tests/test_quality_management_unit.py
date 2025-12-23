"""
Unit tests for Quality Management functionality.

Tests the QualityManager and DataRepairService implementations.
"""

import pytest
import asyncio
from datetime import datetime
from uuid import uuid4
from unittest.mock import Mock, patch, AsyncMock

from src.quality.manager import QualityManager, QualityRule, QualityRuleType, QualityReport
from src.quality.repair import DataRepairService, RepairType, RepairStatus
from src.models.annotation import Annotation
from src.models.quality_issue import QualityIssue, IssueSeverity, IssueStatus


class TestQualityManager:
    """Test cases for QualityManager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.quality_manager = QualityManager()
    
    def test_quality_manager_initialization(self):
        """Test QualityManager initializes with default rules."""
        assert len(self.quality_manager.quality_rules) > 0
        assert len(self.quality_manager.rule_templates) > 0
        
        # Check that default rules are present
        assert "confidence_threshold" in self.quality_manager.quality_rules
        assert "semantic_consistency" in self.quality_manager.quality_rules
        assert "annotation_completeness" in self.quality_manager.quality_rules
    
    def test_add_quality_rule(self):
        """Test adding a new quality rule."""
        rule = QualityRule(
            rule_id="test_rule",
            rule_type=QualityRuleType.CONFIDENCE_THRESHOLD,
            name="Test Rule",
            description="Test rule description",
            threshold=0.8
        )
        
        self.quality_manager.add_quality_rule(rule)
        
        assert "test_rule" in self.quality_manager.quality_rules
        assert self.quality_manager.quality_rules["test_rule"].name == "Test Rule"
    
    def test_enable_disable_rule(self):
        """Test enabling and disabling quality rules."""
        rule_id = "confidence_threshold"
        
        # Test disable
        success = self.quality_manager.disable_rule(rule_id)
        assert success is True
        assert self.quality_manager.quality_rules[rule_id].enabled is False
        
        # Test enable
        success = self.quality_manager.enable_rule(rule_id)
        assert success is True
        assert self.quality_manager.quality_rules[rule_id].enabled is True
        
        # Test non-existent rule
        success = self.quality_manager.enable_rule("non_existent")
        assert success is False
    
    def test_remove_quality_rule(self):
        """Test removing a quality rule."""
        # Add a test rule first
        rule = QualityRule(
            rule_id="temp_rule",
            rule_type=QualityRuleType.CONFIDENCE_THRESHOLD,
            name="Temporary Rule",
            description="Temporary rule for testing"
        )
        self.quality_manager.add_quality_rule(rule)
        
        # Remove the rule
        success = self.quality_manager.remove_quality_rule("temp_rule")
        assert success is True
        assert "temp_rule" not in self.quality_manager.quality_rules
        
        # Try to remove non-existent rule
        success = self.quality_manager.remove_quality_rule("non_existent")
        assert success is False
    
    @pytest.mark.asyncio
    async def test_evaluate_quality_basic(self):
        """Test basic quality evaluation."""
        # Create test annotations
        annotations = [
            Annotation(
                task_id=uuid4(),
                annotator_id="test_user",
                annotation_data={"label": "positive", "confidence": 0.9},
                confidence=0.9,
                time_spent=120
            ),
            Annotation(
                task_id=uuid4(),
                annotator_id="test_user",
                annotation_data={"label": "negative", "confidence": 0.8},
                confidence=0.8,
                time_spent=90
            )
        ]
        
        # Mock Ragas evaluation to avoid external dependencies
        with patch.object(self.quality_manager, '_run_ragas_evaluation', return_value={}):
            quality_report = await self.quality_manager.evaluate_quality(annotations)
        
        assert isinstance(quality_report, QualityReport)
        assert quality_report.task_id == annotations[0].task_id
        assert 0.0 <= quality_report.overall_score <= 1.0
        assert isinstance(quality_report.rule_results, list)
        assert isinstance(quality_report.issues_found, list)
    
    @pytest.mark.asyncio
    async def test_evaluate_quality_with_low_confidence(self):
        """Test quality evaluation with low confidence annotations."""
        # Create annotations with low confidence
        annotations = [
            Annotation(
                task_id=uuid4(),
                annotator_id="test_user",
                annotation_data={"label": "uncertain"},
                confidence=0.3,  # Below default threshold
                time_spent=60
            )
        ]
        
        with patch.object(self.quality_manager, '_run_ragas_evaluation', return_value={}):
            quality_report = await self.quality_manager.evaluate_quality(annotations)
        
        # Should have quality issues due to low confidence
        assert len(quality_report.issues_found) > 0
        
        # Check that confidence threshold issue was created
        confidence_issues = [
            issue for issue in quality_report.issues_found 
            if issue.issue_type == QualityRuleType.CONFIDENCE_THRESHOLD.value
        ]
        assert len(confidence_issues) > 0
    
    @pytest.mark.asyncio
    async def test_evaluate_quality_incomplete_annotations(self):
        """Test quality evaluation with incomplete annotations."""
        # Create annotations missing required fields (has label but missing confidence field)
        annotations = [
            Annotation(
                task_id=uuid4(),
                annotator_id="test_user",
                annotation_data={"label": "test"},  # Missing required confidence field
                confidence=0.9,
                time_spent=120
            )
        ]
        
        with patch.object(self.quality_manager, '_run_ragas_evaluation', return_value={}):
            quality_report = await self.quality_manager.evaluate_quality(annotations)
        
        # Should have completeness issues
        completeness_issues = [
            issue for issue in quality_report.issues_found 
            if issue.issue_type == QualityRuleType.ANNOTATION_COMPLETENESS.value
        ]
        assert len(completeness_issues) > 0
    
    @pytest.mark.asyncio
    async def test_create_quality_issue(self):
        """Test creating a quality issue."""
        task_id = uuid4()
        
        with patch('src.quality.manager.db_manager') as mock_db_manager:
            mock_session = Mock()
            mock_db_manager.get_session.return_value.__enter__.return_value = mock_session
            
            issue = await self.quality_manager.create_quality_issue(
                task_id=task_id,
                issue_type="test_issue",
                description="Test issue description",
                severity=IssueSeverity.HIGH
            )
        
        assert isinstance(issue, QualityIssue)
        assert issue.task_id == task_id
        assert issue.issue_type == "test_issue"
        assert issue.severity == IssueSeverity.HIGH
        assert issue.status == IssueStatus.OPEN
    
    @pytest.mark.asyncio
    async def test_trigger_quality_check(self):
        """Test triggering quality check for a task."""
        task_id = uuid4()
        annotation_data = {
            "id": 123,
            "annotations": [{"result": {"label": "test"}}],
            "completed_by": {"id": "test_user"}
        }
        
        with patch('src.quality.manager.db_manager') as mock_db_manager:
            mock_session = Mock()
            mock_db_manager.get_session.return_value.__enter__.return_value = mock_session
            
            # Mock task in database
            mock_task = Mock()
            mock_task.id = task_id
            mock_task.annotations = [
                {
                    "annotator": "test_user",
                    "result": {"label": "test"},
                    "confidence": 0.8,
                    "lead_time": 120
                }
            ]
            mock_task.quality_score = 0.0
            
            # Mock SQLAlchemy 2.0 style execute/select
            mock_result = Mock()
            mock_result.scalar_one_or_none.return_value = mock_task
            mock_session.execute.return_value = mock_result
            
            # Mock quality evaluation
            with patch.object(self.quality_manager, 'evaluate_quality') as mock_eval:
                mock_report = Mock()
                mock_report.overall_score = 0.85
                mock_report.issues_found = []
                mock_eval.return_value = mock_report
                
                success = await self.quality_manager.trigger_quality_check(task_id, annotation_data)
        
        assert success is True
        assert mock_task.quality_score == 0.85
    
    def test_calculate_quality_score(self):
        """Test quality score calculation."""
        # Test with only rule results
        rule_results = [
            {"passed": True, "score": 0.9},
            {"passed": True, "score": 0.8},
            {"passed": False, "score": 0.5}
        ]
        
        score = self.quality_manager.calculate_quality_score(rule_results, {})
        expected = 2 / 3  # 2 out of 3 rules passed
        assert abs(score - expected) < 0.01
        
        # Test with Ragas metrics
        ragas_metrics = {"answer_relevancy": 0.8, "faithfulness": 0.9}
        
        score = self.quality_manager.calculate_quality_score(rule_results, ragas_metrics)
        rule_score = 2 / 3
        ragas_score = (0.8 + 0.9) / 2
        expected = 0.7 * rule_score + 0.3 * ragas_score
        assert abs(score - expected) < 0.01


class TestDataRepairService:
    """Test cases for DataRepairService."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.repair_service = DataRepairService()
    
    @pytest.mark.asyncio
    async def test_create_repair_request(self):
        """Test creating a repair request."""
        quality_issue_id = uuid4()
        
        repair_record = await self.repair_service.create_repair_request(
            quality_issue_id=quality_issue_id,
            repair_type=RepairType.ANNOTATION_CORRECTION,
            description="Fix annotation error",
            original_data={"label": "wrong"},
            proposed_data={"label": "correct"},
            requested_by="test_user",
            confidence=0.9
        )
        
        assert repair_record.quality_issue_id == quality_issue_id
        assert repair_record.repair_type == RepairType.ANNOTATION_CORRECTION
        assert repair_record.requested_by == "test_user"
        assert repair_record.status == RepairStatus.IN_PROGRESS  # High confidence, auto-approved
    
    @pytest.mark.asyncio
    async def test_create_repair_request_requires_approval(self):
        """Test creating a repair request that requires approval."""
        quality_issue_id = uuid4()
        
        repair_record = await self.repair_service.create_repair_request(
            quality_issue_id=quality_issue_id,
            repair_type=RepairType.SOURCE_DATA_UPDATE,  # Always requires approval
            description="Update source data",
            original_data={"content": "old"},
            proposed_data={"content": "new"},
            requested_by="test_user",
            confidence=0.9
        )
        
        assert repair_record.status == RepairStatus.PENDING  # Requires approval
    
    @pytest.mark.asyncio
    async def test_approve_repair(self):
        """Test approving a repair request."""
        # Create a repair request first
        quality_issue_id = uuid4()
        repair_record = await self.repair_service.create_repair_request(
            quality_issue_id=quality_issue_id,
            repair_type=RepairType.SOURCE_DATA_UPDATE,
            description="Update source data",
            original_data={"content": "old"},
            proposed_data={"content": "new"},
            requested_by="test_user"
        )
        
        repair_id = repair_record.repair_id
        
        # Mock the execute_repair method to avoid database operations
        with patch.object(self.repair_service, 'execute_repair', return_value=True):
            success = await self.repair_service.approve_repair(
                repair_id=repair_id,
                approved_by="manager",
                approval_notes="Approved for testing"
            )
        
        assert success is True
        assert repair_record.approved_by == "manager"
        assert repair_record.approved_at is not None
    
    @pytest.mark.asyncio
    async def test_reject_repair(self):
        """Test rejecting a repair request."""
        # Create a repair request first
        quality_issue_id = uuid4()
        repair_record = await self.repair_service.create_repair_request(
            quality_issue_id=quality_issue_id,
            repair_type=RepairType.SOURCE_DATA_UPDATE,
            description="Update source data",
            original_data={"content": "old"},
            proposed_data={"content": "new"},
            requested_by="test_user"
        )
        
        repair_id = repair_record.repair_id
        
        success = await self.repair_service.reject_repair(
            repair_id=repair_id,
            rejected_by="manager",
            rejection_reason="Not necessary"
        )
        
        assert success is True
        assert repair_record.status == RepairStatus.REJECTED
        assert "Not necessary" in repair_record.description
    
    @pytest.mark.asyncio
    async def test_verify_repair_result(self):
        """Test verifying repair results."""
        # Create a repair record
        quality_issue_id = uuid4()
        repair_record = await self.repair_service.create_repair_request(
            quality_issue_id=quality_issue_id,
            repair_type=RepairType.ANNOTATION_CORRECTION,
            description="Fix annotation",
            original_data={"label": "wrong"},
            proposed_data={"label": "correct"},
            requested_by="test_user"
        )
        
        repair_id = repair_record.repair_id
        
        verification_result = await self.repair_service.verify_repair_result(repair_id)
        
        assert verification_result["verified"] is True
        assert verification_result["repair_id"] == str(repair_id)
        assert "checks_performed" in verification_result
    
    def test_get_repair_history(self):
        """Test getting repair history."""
        # The repair_records dict should be empty initially
        history = self.repair_service.get_repair_history()
        assert isinstance(history, list)
        
        # Test filtering (with empty data, should return empty list)
        filtered = self.repair_service.get_repair_history(
            repair_type=RepairType.ANNOTATION_CORRECTION
        )
        assert isinstance(filtered, list)
    
    def test_get_repair_statistics(self):
        """Test getting repair statistics."""
        stats = self.repair_service.get_repair_statistics()
        
        assert "total_repairs" in stats
        assert "by_status" in stats
        assert "by_type" in stats
        assert "success_rate" in stats
        
        # With no repairs, should have zero stats
        assert stats["total_repairs"] == 0
        assert stats["success_rate"] == 0.0
    
    def test_approval_workflow(self):
        """Test repair approval workflow rules."""
        workflow = self.repair_service.approval_workflow
        
        # Test different repair types
        assert workflow.requires_approval(RepairType.SOURCE_DATA_UPDATE, 0.9) is True
        assert workflow.requires_approval(RepairType.METADATA_REPAIR, 0.9) is False
        
        # Test auto-approval thresholds
        assert workflow.requires_approval(RepairType.ANNOTATION_CORRECTION, 0.95) is False
        assert workflow.requires_approval(RepairType.ANNOTATION_CORRECTION, 0.5) is True
        
        # Test required approver roles
        roles = workflow.get_required_approver_roles(RepairType.SOURCE_DATA_UPDATE)
        assert "data_manager" in roles or "system_admin" in roles


class TestQualityAssessmentAlgorithm:
    """Test cases for quality assessment algorithm accuracy."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.quality_manager = QualityManager()
    
    @pytest.mark.asyncio
    async def test_confidence_threshold_algorithm_accuracy(self):
        """Test confidence threshold algorithm accuracy."""
        # Test case 1: All annotations above threshold
        high_confidence_annotations = [
            Annotation(
                task_id=uuid4(),
                annotator_id="user1",
                annotation_data={"label": "positive"},
                confidence=0.9,
                time_spent=120
            ),
            Annotation(
                task_id=uuid4(),
                annotator_id="user2", 
                annotation_data={"label": "positive"},
                confidence=0.85,
                time_spent=100
            )
        ]
        
        rule = self.quality_manager.quality_rules["confidence_threshold"]
        result = await self.quality_manager._check_confidence_threshold(rule, high_confidence_annotations)
        
        assert result["passed"] is True
        assert result["score"] == 0.875  # Average of 0.9 and 0.85
        
        # Test case 2: Some annotations below threshold
        mixed_confidence_annotations = [
            Annotation(
                task_id=uuid4(),
                annotator_id="user1",
                annotation_data={"label": "positive"},
                confidence=0.9,
                time_spent=120
            ),
            Annotation(
                task_id=uuid4(),
                annotator_id="user2",
                annotation_data={"label": "negative"},
                confidence=0.5,  # Below default threshold of 0.7
                time_spent=80
            )
        ]
        
        result = await self.quality_manager._check_confidence_threshold(rule, mixed_confidence_annotations)
        
        assert result["passed"] is False
        assert result["score"] == 0.7  # Average of 0.9 and 0.5
        assert "低置信度标注数: 1" in result["message"]
    
    @pytest.mark.asyncio
    async def test_annotation_completeness_algorithm_accuracy(self):
        """Test annotation completeness algorithm accuracy."""
        rule = self.quality_manager.quality_rules["annotation_completeness"]
        
        # Test case 1: Complete annotations
        complete_annotations = [
            Annotation(
                task_id=uuid4(),
                annotator_id="user1",
                annotation_data={"label": "positive", "confidence": 0.9},
                confidence=0.9,
                time_spent=120
            ),
            Annotation(
                task_id=uuid4(),
                annotator_id="user2",
                annotation_data={"label": "negative", "confidence": 0.8},
                confidence=0.8,
                time_spent=100
            )
        ]
        
        result = await self.quality_manager._check_annotation_completeness(rule, complete_annotations)
        
        assert result["passed"] is True
        assert result["score"] == 1.0
        assert "不完整标注数: 0" in result["message"]
        
        # Test case 2: Incomplete annotations
        incomplete_annotations = [
            Annotation(
                task_id=uuid4(),
                annotator_id="user1",
                annotation_data={"label": "positive"},  # Missing confidence
                confidence=0.9,
                time_spent=120
            ),
            Annotation(
                task_id=uuid4(),
                annotator_id="user2",
                annotation_data={"partial": "data"},  # Missing both label and confidence
                confidence=0.8,
                time_spent=100
            )
        ]
        
        result = await self.quality_manager._check_annotation_completeness(rule, incomplete_annotations)
        
        assert result["passed"] is False
        assert result["score"] == 0.0  # Both annotations are incomplete
        assert "不完整标注数: 2" in result["message"]
    
    @pytest.mark.asyncio
    async def test_quality_score_calculation_accuracy(self):
        """Test overall quality score calculation accuracy."""
        # Test case 1: All rules pass
        all_pass_results = [
            {"passed": True, "score": 0.9},
            {"passed": True, "score": 0.8},
            {"passed": True, "score": 0.85}
        ]
        
        score = self.quality_manager.calculate_quality_score(all_pass_results, {})
        assert score == 1.0  # All rules passed
        
        # Test case 2: Mixed results
        mixed_results = [
            {"passed": True, "score": 0.9},
            {"passed": False, "score": 0.4},
            {"passed": True, "score": 0.8}
        ]
        
        score = self.quality_manager.calculate_quality_score(mixed_results, {})
        expected = 2 / 3  # 2 out of 3 rules passed
        assert abs(score - expected) < 0.01
        
        # Test case 3: With Ragas metrics
        ragas_metrics = {"answer_relevancy": 0.8, "faithfulness": 0.9}
        
        score = self.quality_manager.calculate_quality_score(mixed_results, ragas_metrics)
        rule_score = 2 / 3
        ragas_score = (0.8 + 0.9) / 2
        expected = 0.7 * rule_score + 0.3 * ragas_score
        assert abs(score - expected) < 0.01
    
    @pytest.mark.asyncio
    async def test_quality_evaluation_edge_cases(self):
        """Test quality evaluation with edge cases."""
        # Test case 1: Empty annotations list
        with pytest.raises(ValueError, match="No annotations provided"):
            await self.quality_manager.evaluate_quality([])
        
        # Test case 2: Single annotation
        single_annotation = [
            Annotation(
                task_id=uuid4(),
                annotator_id="user1",
                annotation_data={"label": "test"},
                confidence=0.8,
                time_spent=60
            )
        ]
        
        with patch.object(self.quality_manager, '_run_ragas_evaluation', return_value={}):
            report = await self.quality_manager.evaluate_quality(single_annotation)
        
        assert isinstance(report, QualityReport)
        assert 0.0 <= report.overall_score <= 1.0
        
        # Test case 3: Annotations with extreme confidence values
        extreme_annotations = [
            Annotation(
                task_id=uuid4(),
                annotator_id="user1",
                annotation_data={"label": "high_confidence"},
                confidence=1.0,  # Maximum confidence
                time_spent=120
            ),
            Annotation(
                task_id=uuid4(),
                annotator_id="user2",
                annotation_data={"label": "low_confidence"},
                confidence=0.0,  # Minimum confidence
                time_spent=60
            )
        ]
        
        with patch.object(self.quality_manager, '_run_ragas_evaluation', return_value={}):
            report = await self.quality_manager.evaluate_quality(extreme_annotations)
        
        # Should handle extreme values gracefully
        assert isinstance(report, QualityReport)
        assert len(report.issues_found) > 0  # Low confidence should create issues


class TestWorkOrderCreationAndDispatch:
    """Test cases for work order creation and dispatch logic."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.quality_manager = QualityManager()
    
    @pytest.mark.asyncio
    async def test_quality_issue_creation_logic(self):
        """Test quality issue creation logic."""
        task_id = uuid4()
        
        with patch('src.quality.manager.get_db_session') as mock_db:
            mock_session = AsyncMock()
            mock_db.return_value.__aenter__.return_value = mock_session
            
            # Test creating issue with different severities
            high_severity_issue = await self.quality_manager.create_quality_issue(
                task_id=task_id,
                issue_type="critical_error",
                description="Critical annotation error",
                severity=IssueSeverity.CRITICAL
            )
            
            assert high_severity_issue.severity == IssueSeverity.CRITICAL
            assert high_severity_issue.status == IssueStatus.OPEN
            assert high_severity_issue.task_id == task_id
            
            # Test creating issue with assignee
            assigned_issue = await self.quality_manager.create_quality_issue(
                task_id=task_id,
                issue_type="review_needed",
                description="Needs expert review",
                severity=IssueSeverity.MEDIUM,
                assignee_id="expert_user"
            )
            
            assert assigned_issue.assignee_id == "expert_user"
    
    @pytest.mark.asyncio
    async def test_automatic_issue_creation_from_evaluation(self):
        """Test automatic quality issue creation from evaluation results."""
        task_id = uuid4()
        
        # Create annotations that will fail quality checks
        failing_annotations = [
            Annotation(
                task_id=task_id,
                annotator_id="user1",
                annotation_data={"incomplete": "data"},  # Missing required fields
                confidence=0.3,  # Below threshold
                time_spent=30
            )
        ]
        
        with patch.object(self.quality_manager, '_run_ragas_evaluation', return_value={}):
            report = await self.quality_manager.evaluate_quality(failing_annotations)
        
        # Should create multiple issues for different rule failures
        assert len(report.issues_found) >= 2  # Confidence and completeness issues
        
        # Check issue types
        issue_types = [issue.issue_type for issue in report.issues_found]
        assert "confidence_threshold" in issue_types
        assert "annotation_completeness" in issue_types
        
        # Check issue severities match rule severities
        for issue in report.issues_found:
            if issue.issue_type == "confidence_threshold":
                assert issue.severity == IssueSeverity.MEDIUM
            elif issue.issue_type == "annotation_completeness":
                assert issue.severity == IssueSeverity.HIGH
    
    @pytest.mark.asyncio
    async def test_issue_assignment_logic(self):
        """Test quality issue assignment logic."""
        task_id = uuid4()
        
        with patch('src.quality.manager.get_db_session') as mock_db:
            mock_session = Mock()
            mock_db.return_value.__aenter__.return_value = mock_session
            
            # Mock database issue
            mock_issue = Mock()
            mock_issue.id = uuid4()
            mock_issue.status = IssueStatus.OPEN
            mock_session.query.return_value.filter.return_value.first.return_value = mock_issue
            
            # Test successful assignment
            success = await self.quality_manager.assign_quality_issue(mock_issue.id, "expert_user")
            
            assert success is True
            assert mock_issue.assignee_id == "expert_user"
            assert mock_issue.status == IssueStatus.IN_PROGRESS
            
            # Test assignment to non-existent issue
            mock_session.query.return_value.filter.return_value.first.return_value = None
            success = await self.quality_manager.assign_quality_issue(uuid4(), "expert_user")
            
            assert success is False
    
    @pytest.mark.asyncio
    async def test_issue_resolution_logic(self):
        """Test quality issue resolution logic."""
        issue_id = uuid4()
        
        with patch('src.quality.manager.get_db_session') as mock_db:
            mock_session = Mock()
            mock_db.return_value.__aenter__.return_value = mock_session
            
            # Mock database issue
            mock_issue = Mock()
            mock_issue.id = issue_id
            mock_issue.status = IssueStatus.IN_PROGRESS
            mock_issue.description = "Original description"
            mock_session.query.return_value.filter.return_value.first.return_value = mock_issue
            
            # Test successful resolution
            success = await self.quality_manager.resolve_quality_issue(
                issue_id, 
                "Fixed by updating annotation data"
            )
            
            assert success is True
            assert mock_issue.status == IssueStatus.RESOLVED
            assert mock_issue.resolved_at is not None
            assert "解决方案: Fixed by updating annotation data" in mock_issue.description
    
    @pytest.mark.asyncio
    async def test_issue_filtering_and_retrieval(self):
        """Test quality issue filtering and retrieval logic."""
        with patch('src.quality.manager.get_db_session') as mock_db:
            mock_session = Mock()
            mock_db.return_value.__aenter__.return_value = mock_session
            
            # Mock database issues
            task_id = uuid4()
            mock_issues = [
                Mock(
                    id=uuid4(),
                    task_id=task_id,
                    issue_type="confidence_threshold",
                    status=IssueStatus.OPEN,
                    assignee_id="user1",
                    severity=IssueSeverity.MEDIUM,
                    description="Low confidence",
                    created_at=datetime.now(),
                    resolved_at=None
                ),
                Mock(
                    id=uuid4(),
                    task_id=task_id,
                    issue_type="annotation_completeness",
                    status=IssueStatus.RESOLVED,
                    assignee_id="user2",
                    severity=IssueSeverity.HIGH,
                    description="Missing fields",
                    created_at=datetime.now(),
                    resolved_at=datetime.now()
                )
            ]
            
            mock_session.query.return_value.all.return_value = mock_issues
            
            # Test retrieval without filters
            issues = await self.quality_manager.get_quality_issues()
            assert len(issues) == 2
            
            # Test filtering by task_id (mock query filtering)
            mock_session.query.return_value.filter.return_value.all.return_value = mock_issues
            issues = await self.quality_manager.get_quality_issues(task_id=task_id)
            assert len(issues) == 2
            
            # Test filtering by status
            open_issues = [mock_issues[0]]  # Only the first issue is open
            mock_session.query.return_value.filter.return_value.all.return_value = open_issues
            issues = await self.quality_manager.get_quality_issues(status=IssueStatus.OPEN)
            assert len(issues) == 1
            assert issues[0].status == IssueStatus.OPEN


class TestDataRepairFunctionality:
    """Test cases for data repair functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.repair_service = DataRepairService()
    
    @pytest.mark.asyncio
    async def test_annotation_correction_repair(self):
        """Test annotation correction repair functionality."""
        quality_issue_id = uuid4()
        
        # Create repair request for annotation correction
        repair_record = await self.repair_service.create_repair_request(
            quality_issue_id=quality_issue_id,
            repair_type=RepairType.ANNOTATION_CORRECTION,
            description="Fix incorrect label",
            original_data={"id": "ann_123", "result": {"label": "wrong_label"}},
            proposed_data={"id": "ann_123", "result": {"label": "correct_label"}},
            requested_by="annotator",
            confidence=0.95
        )
        
        assert repair_record.repair_type == RepairType.ANNOTATION_CORRECTION
        assert repair_record.status == RepairStatus.IN_PROGRESS  # Auto-approved due to high confidence
        
        # Mock database operations for execution
        with patch('src.quality.repair.get_db_session') as mock_db:
            mock_session = Mock()
            mock_db.return_value.__aenter__.return_value = mock_session
            
            # Mock quality issue and task
            mock_issue = Mock()
            mock_issue.task_id = uuid4()
            mock_task = Mock()
            mock_task.annotations = [
                {"id": "ann_123", "result": {"label": "wrong_label"}}
            ]
            
            mock_session.query.return_value.filter.return_value.first.side_effect = [
                mock_issue, mock_task
            ]
            
            # Execute repair
            success = await self.repair_service.execute_repair(
                repair_record.repair_id, 
                "system"
            )
            
            assert success is True
            assert repair_record.status == RepairStatus.COMPLETED
            assert repair_record.executed_by == "system"
    
    @pytest.mark.asyncio
    async def test_source_data_update_repair(self):
        """Test source data update repair functionality."""
        quality_issue_id = uuid4()
        
        # Create repair request for source data update
        repair_record = await self.repair_service.create_repair_request(
            quality_issue_id=quality_issue_id,
            repair_type=RepairType.SOURCE_DATA_UPDATE,
            description="Fix source document content",
            original_data={"content": "incorrect content"},
            proposed_data={"content": "corrected content"},
            requested_by="data_manager",
            confidence=0.8
        )
        
        assert repair_record.repair_type == RepairType.SOURCE_DATA_UPDATE
        assert repair_record.status == RepairStatus.PENDING  # Requires approval
        
        # Approve the repair
        with patch.object(self.repair_service, 'execute_repair', return_value=True):
            success = await self.repair_service.approve_repair(
                repair_record.repair_id,
                "supervisor",
                "Approved after review"
            )
        
        assert success is True
        assert repair_record.approved_by == "supervisor"
        assert "Approved after review" in repair_record.description
    
    @pytest.mark.asyncio
    async def test_quality_score_adjustment_repair(self):
        """Test quality score adjustment repair functionality."""
        quality_issue_id = uuid4()
        
        repair_record = await self.repair_service.create_repair_request(
            quality_issue_id=quality_issue_id,
            repair_type=RepairType.QUALITY_SCORE_ADJUSTMENT,
            description="Adjust quality score after manual review",
            original_data={"quality_score": 0.6},
            proposed_data={"quality_score": 0.8},
            requested_by="quality_manager",
            confidence=0.9
        )
        
        # Mock database operations
        with patch('src.quality.repair.get_db_session') as mock_db:
            mock_session = Mock()
            mock_db.return_value.__aenter__.return_value = mock_session
            
            mock_issue = Mock()
            mock_issue.task_id = uuid4()
            mock_task = Mock()
            mock_task.quality_score = 0.6
            
            mock_session.query.return_value.filter.return_value.first.side_effect = [
                mock_issue, mock_task
            ]
            
            # Execute repair
            success = await self.repair_service._repair_quality_score(repair_record)
            
            assert success is True
            assert mock_task.quality_score == 0.8
    
    @pytest.mark.asyncio
    async def test_label_standardization_repair(self):
        """Test label standardization repair functionality."""
        quality_issue_id = uuid4()
        
        repair_record = await self.repair_service.create_repair_request(
            quality_issue_id=quality_issue_id,
            repair_type=RepairType.LABEL_STANDARDIZATION,
            description="Standardize inconsistent labels",
            original_data={"labels": ["pos", "positive", "good"]},
            proposed_data={"label_mapping": {"pos": "positive", "good": "positive"}},
            requested_by="data_curator",
            confidence=0.95
        )
        
        # Mock database operations
        with patch('src.quality.repair.get_db_session') as mock_db:
            mock_session = Mock()
            mock_db.return_value.__aenter__.return_value = mock_session
            
            mock_issue = Mock()
            mock_issue.task_id = uuid4()
            mock_task = Mock()
            mock_task.annotations = [
                {
                    "result": [
                        {"value": {"labels": ["pos", "good"]}}
                    ]
                }
            ]
            
            mock_session.query.return_value.filter.return_value.first.side_effect = [
                mock_issue, mock_task
            ]
            
            # Execute repair
            success = await self.repair_service._repair_label_standardization(repair_record)
            
            assert success is True
            # Check that labels were standardized
            standardized_labels = mock_task.annotations[0]["result"][0]["value"]["labels"]
            assert standardized_labels == ["positive", "positive"]
    
    @pytest.mark.asyncio
    async def test_repair_verification_functionality(self):
        """Test repair result verification functionality."""
        quality_issue_id = uuid4()
        
        repair_record = await self.repair_service.create_repair_request(
            quality_issue_id=quality_issue_id,
            repair_type=RepairType.ANNOTATION_CORRECTION,
            description="Test repair for verification",
            original_data={"label": "old"},
            proposed_data={"label": "new"},
            requested_by="tester",
            confidence=0.9
        )
        
        # Test verification
        verification_result = await self.repair_service.verify_repair_result(repair_record.repair_id)
        
        assert verification_result["verified"] is True
        assert verification_result["repair_id"] == str(repair_record.repair_id)
        assert verification_result["repair_type"] == RepairType.ANNOTATION_CORRECTION.value
        assert "checks_performed" in verification_result
        assert "data_integrity" in verification_result
        assert verification_result["data_integrity"] == "passed"
    
    @pytest.mark.asyncio
    async def test_repair_failure_handling(self):
        """Test repair failure handling."""
        quality_issue_id = uuid4()
        
        repair_record = await self.repair_service.create_repair_request(
            quality_issue_id=quality_issue_id,
            repair_type=RepairType.SOURCE_DATA_UPDATE,
            description="Test repair that will fail",
            original_data={"content": "old"},
            proposed_data={"content": "new"},
            requested_by="tester",
            confidence=0.8
        )
        
        # Mock database failure
        with patch('src.quality.repair.get_db_session') as mock_db:
            mock_db.side_effect = Exception("Database connection failed")
            
            # Execute repair (should fail)
            success = await self.repair_service.execute_repair(
                repair_record.repair_id,
                "executor"
            )
            
            assert success is False
            assert repair_record.status == RepairStatus.FAILED
    
    def test_repair_history_and_statistics(self):
        """Test repair history and statistics functionality."""
        # Initially empty
        history = self.repair_service.get_repair_history()
        assert len(history) == 0
        
        stats = self.repair_service.get_repair_statistics()
        assert stats["total_repairs"] == 0
        assert stats["success_rate"] == 0.0
        
        # Add some mock repair records for testing
        repair_id1 = uuid4()
        repair_id2 = uuid4()
        
        from src.quality.repair import RepairRecord
        
        self.repair_service.repair_records[repair_id1] = RepairRecord(
            repair_id=repair_id1,
            quality_issue_id=uuid4(),
            repair_type=RepairType.ANNOTATION_CORRECTION,
            description="Test repair 1",
            original_data={},
            proposed_data={},
            status=RepairStatus.COMPLETED
        )
        
        self.repair_service.repair_records[repair_id2] = RepairRecord(
            repair_id=repair_id2,
            quality_issue_id=uuid4(),
            repair_type=RepairType.SOURCE_DATA_UPDATE,
            description="Test repair 2",
            original_data={},
            proposed_data={},
            status=RepairStatus.FAILED
        )
        
        # Test history retrieval
        history = self.repair_service.get_repair_history()
        assert len(history) == 2
        
        # Test filtering by type
        annotation_repairs = self.repair_service.get_repair_history(
            repair_type=RepairType.ANNOTATION_CORRECTION
        )
        assert len(annotation_repairs) == 1
        assert annotation_repairs[0].repair_type == RepairType.ANNOTATION_CORRECTION
        
        # Test filtering by status
        completed_repairs = self.repair_service.get_repair_history(
            status=RepairStatus.COMPLETED
        )
        assert len(completed_repairs) == 1
        assert completed_repairs[0].status == RepairStatus.COMPLETED
        
        # Test statistics
        stats = self.repair_service.get_repair_statistics()
        assert stats["total_repairs"] == 2
        assert stats["success_rate"] == 0.5  # 1 completed out of 2 total


if __name__ == "__main__":
    pytest.main([__file__, "-v"])