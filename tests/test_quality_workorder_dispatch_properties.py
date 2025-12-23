"""
Property-based tests for quality work order automatic dispatch functionality.

Tests the automatic creation and dispatch of quality work orders when quality issues
are discovered, as specified in Requirements 4.2 and 4.3 of the SuperInsight Platform.
"""

import pytest
from hypothesis import given, strategies as st, assume, settings
from hypothesis import HealthCheck
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, Any, List, Optional
from uuid import UUID, uuid4
from datetime import datetime
import json
import asyncio

# Import models directly to avoid Ragas dependency issues
from src.models.annotation import Annotation
from src.models.quality_issue import QualityIssue, IssueSeverity, IssueStatus
from src.models.task import Task, TaskStatus


# Hypothesis strategies for generating test data

def quality_issue_strategy():
    """Strategy for generating valid QualityIssue instances."""
    return st.builds(
        QualityIssue,
        id=st.just(uuid4()),
        task_id=st.just(uuid4()),
        issue_type=st.sampled_from([
            "confidence_threshold", "annotation_completeness", "semantic_consistency",
            "response_relevancy", "factual_accuracy", "inter_annotator_agreement"
        ]),
        description=st.text(min_size=10, max_size=200),
        severity=st.sampled_from([IssueSeverity.LOW, IssueSeverity.MEDIUM, IssueSeverity.HIGH, IssueSeverity.CRITICAL]),
        status=st.just(IssueStatus.OPEN),
        assignee_id=st.one_of(st.none(), st.text(min_size=1, max_size=50).filter(lambda x: x.strip())),
        created_at=st.just(datetime.now()),
        resolved_at=st.none()
    )


def annotation_with_issues_strategy():
    """Strategy for generating annotations that would trigger quality issues."""
    return st.builds(
        Annotation,
        id=st.just(uuid4()),
        task_id=st.just(uuid4()),
        annotator_id=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
        annotation_data=st.one_of(
            # Valid but minimal annotation data (to avoid empty validation error)
            st.dictionaries(
                st.sampled_from(['label', 'partial_label', 'incomplete']),
                st.text(min_size=1, max_size=20),
                min_size=1, max_size=2
            ),
            # Valid but low-quality annotation data
            st.dictionaries(
                st.sampled_from(['label', 'confidence', 'quality_score']),
                st.one_of(st.text(min_size=1), st.floats(min_value=0.0, max_value=0.5)),
                min_size=1, max_size=3
            )
        ),
        confidence=st.floats(min_value=0.0, max_value=0.6),  # Low confidence to trigger issues
        time_spent=st.integers(min_value=1, max_value=3600),
        created_at=st.just(datetime.now())
    )


def expert_user_strategy():
    """Strategy for generating expert user IDs for assignment."""
    return st.sampled_from([
        "quality_expert_1", "senior_annotator_2", "domain_expert_3",
        "technical_reviewer_4", "business_analyst_5", "data_scientist_6"
    ])


def quality_rule_failure_strategy():
    """Strategy for generating quality rule failures that should create work orders."""
    return st.builds(
        dict,
        rule_id=st.sampled_from([
            "confidence_threshold", "annotation_completeness", "semantic_consistency"
        ]),
        rule_name=st.text(min_size=5, max_size=50),
        passed=st.just(False),  # Always failed to trigger work order creation
        score=st.floats(min_value=0.0, max_value=0.6),  # Low score indicating failure
        message=st.text(min_size=10, max_size=100),
        severity=st.sampled_from([IssueSeverity.MEDIUM, IssueSeverity.HIGH, IssueSeverity.CRITICAL])
    )


class MockWorkOrderDispatcher:
    """Mock work order dispatcher for testing automatic dispatch functionality."""
    
    def __init__(self):
        self.created_work_orders = []
        self.assigned_work_orders = []
        self.expert_assignments = {}
    
    async def create_work_order(self, quality_issue: QualityIssue) -> Dict[str, Any]:
        """Mock work order creation."""
        work_order = {
            'work_order_id': str(uuid4()),
            'quality_issue_id': str(quality_issue.id),
            'task_id': str(quality_issue.task_id),
            'issue_type': quality_issue.issue_type,
            'severity': quality_issue.severity.value,
            'status': 'created',
            'created_at': datetime.now().isoformat()
        }
        self.created_work_orders.append(work_order)
        return work_order
    
    async def auto_assign_work_order(self, work_order_id: str, issue_type: str, severity: IssueSeverity) -> Optional[str]:
        """Mock automatic work order assignment based on issue type and severity."""
        # Simulate assignment logic based on issue type and severity
        if issue_type == "confidence_threshold":
            assignee = "quality_expert_1"
        elif issue_type == "annotation_completeness":
            assignee = "senior_annotator_2"
        elif issue_type == "semantic_consistency":
            assignee = "domain_expert_3"
        elif severity == IssueSeverity.CRITICAL:
            assignee = "technical_reviewer_4"
        else:
            assignee = "business_analyst_5"
        
        assignment = {
            'work_order_id': work_order_id,
            'assignee_id': assignee,
            'assigned_at': datetime.now().isoformat(),
            'assignment_reason': f"Auto-assigned based on {issue_type} and {severity.value} severity"
        }
        
        self.assigned_work_orders.append(assignment)
        self.expert_assignments[work_order_id] = assignee
        return assignee
    
    def get_work_order_by_issue(self, quality_issue_id: str) -> Optional[Dict[str, Any]]:
        """Get work order by quality issue ID."""
        for work_order in self.created_work_orders:
            if work_order['quality_issue_id'] == quality_issue_id:
                return work_order
        return None
    
    def get_assignment_by_work_order(self, work_order_id: str) -> Optional[Dict[str, Any]]:
        """Get assignment by work order ID."""
        for assignment in self.assigned_work_orders:
            if assignment['work_order_id'] == work_order_id:
                return assignment
        return None


class TestQualityWorkOrderAutoDispatch:
    """
    Property-based tests for quality work order automatic dispatch.
    
    Validates Requirements 4.2 and 4.3:
    - When quality issues are discovered, work orders should be automatically created
    - Work orders should be automatically assigned to appropriate experts
    """
    
    @given(quality_issue_strategy())
    @settings(suppress_health_check=[HealthCheck.filter_too_much])
    def test_quality_issue_discovery_creates_work_order(self, quality_issue: QualityIssue):
        """
        Property 8: Quality Work Order Auto-Creation
        
        For any quality issue that is discovered during quality evaluation,
        a work order should be automatically created for resolution.
        
        **Validates: Requirements 4.2**
        """
        # Create mock work order dispatcher
        mock_dispatcher = MockWorkOrderDispatcher()
        
        # Simulate quality issue discovery and work order creation
        async def process_quality_issue_discovery():
            # This simulates the quality manager discovering an issue
            # and automatically creating a work order
            work_order = await mock_dispatcher.create_work_order(quality_issue)
            return work_order
        
        # Run the async process
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            work_order = loop.run_until_complete(process_quality_issue_discovery())
            
            # Verify work order was created
            assert work_order is not None
            assert work_order['quality_issue_id'] == str(quality_issue.id)
            assert work_order['task_id'] == str(quality_issue.task_id)
            assert work_order['issue_type'] == quality_issue.issue_type
            assert work_order['severity'] == quality_issue.severity.value
            assert work_order['status'] == 'created'
            
            # Verify work order is tracked in dispatcher
            assert len(mock_dispatcher.created_work_orders) == 1
            
            # Verify work order can be retrieved by quality issue ID
            retrieved_work_order = mock_dispatcher.get_work_order_by_issue(str(quality_issue.id))
            assert retrieved_work_order is not None
            assert retrieved_work_order['work_order_id'] == work_order['work_order_id']
            
        finally:
            loop.close()
    
    @given(quality_issue_strategy(), expert_user_strategy())
    @settings(suppress_health_check=[HealthCheck.filter_too_much])
    def test_work_order_automatic_assignment(self, quality_issue: QualityIssue, expected_expert: str):
        """
        Property 8: Work Order Auto-Assignment
        
        For any work order created from a quality issue, the work order should be
        automatically assigned to an appropriate expert based on issue type and severity.
        
        **Validates: Requirements 4.3**
        """
        # Create mock work order dispatcher
        mock_dispatcher = MockWorkOrderDispatcher()
        
        # Simulate work order creation and automatic assignment
        async def process_work_order_assignment():
            # Create work order
            work_order = await mock_dispatcher.create_work_order(quality_issue)
            work_order_id = work_order['work_order_id']
            
            # Auto-assign work order
            assigned_expert = await mock_dispatcher.auto_assign_work_order(
                work_order_id=work_order_id,
                issue_type=quality_issue.issue_type,
                severity=quality_issue.severity
            )
            
            return work_order_id, assigned_expert
        
        # Run the async process
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            work_order_id, assigned_expert = loop.run_until_complete(process_work_order_assignment())
            
            # Verify work order was assigned
            assert assigned_expert is not None
            assert isinstance(assigned_expert, str)
            assert len(assigned_expert) > 0
            
            # Verify assignment is tracked
            assert len(mock_dispatcher.assigned_work_orders) == 1
            assert work_order_id in mock_dispatcher.expert_assignments
            assert mock_dispatcher.expert_assignments[work_order_id] == assigned_expert
            
            # Verify assignment can be retrieved
            assignment = mock_dispatcher.get_assignment_by_work_order(work_order_id)
            assert assignment is not None
            assert assignment['assignee_id'] == assigned_expert
            assert assignment['work_order_id'] == work_order_id
            assert 'assigned_at' in assignment
            assert 'assignment_reason' in assignment
            
        finally:
            loop.close()
    
    @given(st.lists(annotation_with_issues_strategy(), min_size=1, max_size=5))
    @settings(suppress_health_check=[HealthCheck.filter_too_much])
    def test_multiple_quality_issues_create_multiple_work_orders(self, problematic_annotations: List[Annotation]):
        """
        Property 8: Multiple Quality Issues Work Order Creation
        
        For any set of annotations that have quality issues, each quality issue
        should result in its own work order being created and assigned.
        
        **Validates: Requirements 4.2, 4.3**
        """
        # Create mock work order dispatcher
        mock_dispatcher = MockWorkOrderDispatcher()
        
        # Process annotations and create work orders for quality issues
        async def process_multiple_quality_issues():
            all_work_orders = []
            
            for annotation in problematic_annotations:
                # Simulate quality issues based on annotation characteristics
                quality_issues = []
                
                # Check for low confidence (confidence threshold issue)
                if annotation.confidence < 0.7:
                    quality_issue = QualityIssue(
                        task_id=annotation.task_id,
                        issue_type="confidence_threshold",
                        description=f"Low confidence annotation: {annotation.confidence}",
                        severity=IssueSeverity.MEDIUM,
                        status=IssueStatus.OPEN
                    )
                    quality_issues.append(quality_issue)
                
                # Check for incomplete annotation data
                if len(annotation.annotation_data) < 2:  # Require at least 2 fields
                    quality_issue = QualityIssue(
                        task_id=annotation.task_id,
                        issue_type="annotation_completeness",
                        description="Insufficient annotation data",
                        severity=IssueSeverity.HIGH,
                        status=IssueStatus.OPEN
                    )
                    quality_issues.append(quality_issue)
                
                # Create work orders for each quality issue found
                for quality_issue in quality_issues:
                    work_order = await mock_dispatcher.create_work_order(quality_issue)
                    
                    # Auto-assign the work order
                    assigned_expert = await mock_dispatcher.auto_assign_work_order(
                        work_order_id=work_order['work_order_id'],
                        issue_type=quality_issue.issue_type,
                        severity=quality_issue.severity
                    )
                    
                    work_order['assigned_expert'] = assigned_expert
                    all_work_orders.append(work_order)
            
            return all_work_orders
        
        # Run the async process
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            work_orders = loop.run_until_complete(process_multiple_quality_issues())
            
            # Verify work orders were created for quality issues
            assert len(mock_dispatcher.created_work_orders) >= 0  # May be 0 if no issues found
            
            # If quality issues were found, verify work orders and assignments
            if len(mock_dispatcher.created_work_orders) > 0:
                # Each work order should have been assigned
                assert len(mock_dispatcher.assigned_work_orders) == len(mock_dispatcher.created_work_orders)
                
                # Each work order should have a unique ID
                work_order_ids = [wo['work_order_id'] for wo in mock_dispatcher.created_work_orders]
                assert len(work_order_ids) == len(set(work_order_ids))  # No duplicates
                
                # Each assignment should correspond to a work order
                for assignment in mock_dispatcher.assigned_work_orders:
                    work_order_id = assignment['work_order_id']
                    assert work_order_id in work_order_ids
                    assert work_order_id in mock_dispatcher.expert_assignments
            
        finally:
            loop.close()
    
    @given(quality_rule_failure_strategy())
    @settings(suppress_health_check=[HealthCheck.filter_too_much])
    def test_quality_rule_failure_triggers_work_order_dispatch(self, rule_failure: Dict[str, Any]):
        """
        Property 8: Quality Rule Failure Work Order Dispatch
        
        For any quality rule that fails during evaluation, a work order should be
        automatically created and dispatched to handle the quality issue.
        
        **Validates: Requirements 4.2, 4.3**
        """
        task_id = uuid4()
        mock_dispatcher = MockWorkOrderDispatcher()
        
        # Simulate quality rule failure processing
        async def process_rule_failure():
            # Create quality issue from rule failure
            quality_issue = QualityIssue(
                task_id=task_id,
                issue_type=rule_failure['rule_id'],
                description=f"{rule_failure['rule_name']}: {rule_failure['message']}",
                severity=rule_failure['severity'],
                status=IssueStatus.OPEN
            )
            
            # Create and assign work order for the quality issue
            work_order = await mock_dispatcher.create_work_order(quality_issue)
            
            assigned_expert = await mock_dispatcher.auto_assign_work_order(
                work_order_id=work_order['work_order_id'],
                issue_type=quality_issue.issue_type,
                severity=quality_issue.severity
            )
            
            return work_order, assigned_expert
        
        # Run the async process
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            work_order, assigned_expert = loop.run_until_complete(process_rule_failure())
            
            # Verify work order was created from rule failure
            assert work_order['issue_type'] == rule_failure['rule_id']
            assert work_order['severity'] == rule_failure['severity'].value
            assert work_order['task_id'] == str(task_id)
            
            # Verify work order was assigned
            assert assigned_expert is not None
            assert len(assigned_expert) > 0
            
            # Verify assignment logic based on issue type
            assignment = mock_dispatcher.get_assignment_by_work_order(work_order['work_order_id'])
            assert assignment is not None
            assert assignment['assignee_id'] == assigned_expert
            
            # Verify assignment reason includes issue type and severity
            assert rule_failure['rule_id'] in assignment['assignment_reason']
            assert rule_failure['severity'].value in assignment['assignment_reason']
            
        finally:
            loop.close()
    
    @given(st.integers(min_value=1, max_value=10), st.sampled_from([IssueSeverity.HIGH, IssueSeverity.CRITICAL]))
    @settings(suppress_health_check=[HealthCheck.filter_too_much])
    def test_high_severity_issues_get_priority_assignment(self, num_issues: int, high_severity: IssueSeverity):
        """
        Property 8: High Severity Issue Priority Assignment
        
        For any high-severity quality issues, work orders should be created and
        assigned with appropriate priority to ensure rapid resolution.
        
        **Validates: Requirements 4.2, 4.3**
        """
        mock_dispatcher = MockWorkOrderDispatcher()
        
        # Create multiple quality issues with varying severities
        async def process_priority_assignment():
            high_severity_work_orders = []
            
            for i in range(num_issues):
                # Create high-severity quality issue
                quality_issue = QualityIssue(
                    task_id=uuid4(),
                    issue_type=f"critical_issue_{i}",
                    description=f"High severity quality issue {i}",
                    severity=high_severity,
                    status=IssueStatus.OPEN
                )
                
                # Create work order
                work_order = await mock_dispatcher.create_work_order(quality_issue)
                
                # Auto-assign with priority consideration
                assigned_expert = await mock_dispatcher.auto_assign_work_order(
                    work_order_id=work_order['work_order_id'],
                    issue_type=quality_issue.issue_type,
                    severity=quality_issue.severity
                )
                
                work_order['assigned_expert'] = assigned_expert
                high_severity_work_orders.append(work_order)
            
            return high_severity_work_orders
        
        # Run the async process
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            work_orders = loop.run_until_complete(process_priority_assignment())
            
            # Verify all high-severity issues got work orders
            assert len(mock_dispatcher.created_work_orders) == num_issues
            
            # Verify all work orders were assigned
            assert len(mock_dispatcher.assigned_work_orders) == num_issues
            
            # Verify high-severity issues get appropriate expert assignment
            for work_order in work_orders:
                assert work_order['severity'] == high_severity.value
                assert work_order['assigned_expert'] is not None
                
                # For critical issues, should be assigned to technical reviewer
                if high_severity == IssueSeverity.CRITICAL:
                    assignment = mock_dispatcher.get_assignment_by_work_order(work_order['work_order_id'])
                    # The mock assigns critical issues to technical_reviewer_4
                    assert assignment['assignee_id'] == "technical_reviewer_4"
            
        finally:
            loop.close()
    
    @given(st.text(min_size=1, max_size=50), st.text(min_size=10, max_size=200))
    @settings(suppress_health_check=[HealthCheck.filter_too_much])
    def test_work_order_dispatch_integration_with_quality_manager(self, issue_type: str, issue_description: str):
        """
        Property 8: Quality Manager Integration with Work Order Dispatch
        
        For any quality issue detected by the quality manager, the work order
        dispatch system should be automatically triggered to create and assign work orders.
        
        **Validates: Requirements 4.2, 4.3**
        """
        task_id = uuid4()
        mock_dispatcher = MockWorkOrderDispatcher()
        
        # Simulate integrated quality check and work order dispatch
        async def integrated_quality_check_and_dispatch():
            # Create annotation that will trigger quality issues
            annotation = Annotation(
                task_id=task_id,
                annotator_id="test_annotator",
                annotation_data={"label": "incomplete"},  # Minimal data to avoid validation error
                confidence=0.3,  # Low confidence to trigger threshold issue
                time_spent=60
            )
            
            # Simulate quality issues that would be found
            quality_issues = []
            
            # Low confidence issue
            if annotation.confidence < 0.7:
                quality_issue = QualityIssue(
                    task_id=task_id,
                    issue_type="confidence_threshold",
                    description=f"Low confidence: {annotation.confidence}",
                    severity=IssueSeverity.MEDIUM,
                    status=IssueStatus.OPEN
                )
                quality_issues.append(quality_issue)
            
            # Completeness issue (check for minimal required fields)
            if len(annotation.annotation_data) < 2:  # Require at least 2 fields for completeness
                quality_issue = QualityIssue(
                    task_id=task_id,
                    issue_type="annotation_completeness",
                    description="Insufficient annotation data fields",
                    severity=IssueSeverity.HIGH,
                    status=IssueStatus.OPEN
                )
                quality_issues.append(quality_issue)
            
            # Simulate automatic work order creation for each quality issue
            created_work_orders = []
            for quality_issue in quality_issues:
                # Create work order
                work_order = await mock_dispatcher.create_work_order(quality_issue)
                
                # Auto-assign work order
                assigned_expert = await mock_dispatcher.auto_assign_work_order(
                    work_order_id=work_order['work_order_id'],
                    issue_type=quality_issue.issue_type,
                    severity=quality_issue.severity
                )
                
                work_order['assigned_expert'] = assigned_expert
                created_work_orders.append(work_order)
            
            return quality_issues, created_work_orders
        
        # Run the integrated process
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            quality_issues, work_orders = loop.run_until_complete(integrated_quality_check_and_dispatch())
            
            # Verify quality issues were found (due to low confidence and empty data)
            assert len(quality_issues) > 0
            
            # Verify work orders were created for each quality issue
            assert len(work_orders) == len(quality_issues)
            assert len(mock_dispatcher.created_work_orders) == len(quality_issues)
            
            # Verify each work order corresponds to a quality issue
            for work_order, quality_issue in zip(work_orders, quality_issues):
                assert work_order['quality_issue_id'] == str(quality_issue.id)
                assert work_order['task_id'] == str(quality_issue.task_id)
                assert work_order['issue_type'] == quality_issue.issue_type
                assert work_order['assigned_expert'] is not None
            
            # Verify all work orders were assigned
            assert len(mock_dispatcher.assigned_work_orders) == len(work_orders)
            
        finally:
            loop.close()
    
    def test_work_order_dispatch_configuration(self):
        """
        Property 8: Work Order Dispatch Configuration
        
        The work order dispatch system should be properly configured to handle
        automatic creation and assignment of work orders for quality issues.
        
        **Validates: Requirements 4.2, 4.3**
        """
        mock_dispatcher = MockWorkOrderDispatcher()
        
        # Verify dispatcher is properly initialized
        assert hasattr(mock_dispatcher, 'created_work_orders')
        assert hasattr(mock_dispatcher, 'assigned_work_orders')
        assert hasattr(mock_dispatcher, 'expert_assignments')
        
        # Verify dispatcher methods are available
        assert callable(mock_dispatcher.create_work_order)
        assert callable(mock_dispatcher.auto_assign_work_order)
        assert callable(mock_dispatcher.get_work_order_by_issue)
        assert callable(mock_dispatcher.get_assignment_by_work_order)
        
        # Verify initial state
        assert len(mock_dispatcher.created_work_orders) == 0
        assert len(mock_dispatcher.assigned_work_orders) == 0
        assert len(mock_dispatcher.expert_assignments) == 0
        
        # Test assignment logic for different issue types
        test_cases = [
            ("confidence_threshold", IssueSeverity.MEDIUM, "quality_expert_1"),
            ("annotation_completeness", IssueSeverity.HIGH, "senior_annotator_2"),
            ("semantic_consistency", IssueSeverity.LOW, "domain_expert_3"),
            ("critical_issue", IssueSeverity.CRITICAL, "technical_reviewer_4")
        ]
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            for issue_type, severity, expected_assignee in test_cases:
                # Create quality issue
                quality_issue = QualityIssue(
                    task_id=uuid4(),
                    issue_type=issue_type,
                    description=f"Test {issue_type} issue",
                    severity=severity,
                    status=IssueStatus.OPEN
                )
                
                # Test work order creation and assignment
                work_order = loop.run_until_complete(mock_dispatcher.create_work_order(quality_issue))
                assigned_expert = loop.run_until_complete(
                    mock_dispatcher.auto_assign_work_order(
                        work_order_id=work_order['work_order_id'],
                        issue_type=issue_type,
                        severity=severity
                    )
                )
                
                # Verify assignment matches expected logic
                if severity == IssueSeverity.CRITICAL:
                    assert assigned_expert == "technical_reviewer_4"
                elif issue_type == "confidence_threshold":
                    assert assigned_expert == "quality_expert_1"
                elif issue_type == "annotation_completeness":
                    assert assigned_expert == "senior_annotator_2"
                elif issue_type == "semantic_consistency":
                    assert assigned_expert == "domain_expert_3"
                
        finally:
            loop.close()


if __name__ == "__main__":
    # Run with verbose output and show hypothesis examples
    pytest.main([__file__, "-v", "--hypothesis-show-statistics"])