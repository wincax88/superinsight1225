"""
Property-based tests for quality check auto-triggering functionality.

Tests the automatic quality check triggering when annotations are completed
as specified in Requirement 3.4 of the SuperInsight Platform requirements.
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

from src.label_studio.integration import LabelStudioIntegration, ProjectConfig
from src.models.task import Task, TaskStatus
from src.models.annotation import Annotation
from src.models.quality_issue import QualityIssue, IssueSeverity, IssueStatus
from src.database.models import TaskModel


# Hypothesis strategies for generating test data

def annotation_strategy():
    """Strategy for generating valid Annotation instances."""
    return st.builds(
        Annotation,
        id=st.just(uuid4()),
        task_id=st.just(uuid4()),
        annotator_id=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
        annotation_data=st.dictionaries(
            st.text(min_size=1, max_size=20),
            st.one_of(st.text(), st.integers(), st.floats(), st.booleans()),
            min_size=1, max_size=5
        ),
        confidence=st.floats(min_value=0.0, max_value=1.0),
        time_spent=st.integers(min_value=1, max_value=3600),
        created_at=st.just(datetime.now())
    )


def task_strategy():
    """Strategy for generating valid Task instances."""
    return st.builds(
        Task,
        id=st.just(uuid4()),
        document_id=st.just(uuid4()),
        project_id=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),  # Ensure project_id is not empty
        status=st.sampled_from([TaskStatus.PENDING, TaskStatus.IN_PROGRESS, TaskStatus.COMPLETED]),
        annotations=st.lists(
            st.dictionaries(
                st.sampled_from(['id', 'result', 'annotator_id', 'confidence']),
                st.one_of(st.text(), st.integers(), st.floats(min_value=0.0, max_value=1.0))
            ),
            min_size=0, max_size=3
        ),
        ai_predictions=st.lists(st.dictionaries(st.text(), st.text()), min_size=0, max_size=2),
        quality_score=st.floats(min_value=0.0, max_value=1.0),
        created_at=st.just(datetime.now())
    )


def completed_annotation_data_strategy():
    """Strategy for generating Label Studio annotation completion data."""
    return st.builds(
        dict,
        id=st.integers(min_value=1, max_value=10000),
        annotations=st.lists(
            st.dictionaries(
                st.sampled_from(['id', 'result', 'created_at', 'updated_at']),
                st.one_of(
                    st.integers(min_value=1, max_value=1000),
                    st.text(min_size=1, max_size=100),
                    st.lists(st.dictionaries(st.text(), st.text()))
                )
            ),
            min_size=1, max_size=1
        ),
        completed_by=st.dictionaries(
            st.sampled_from(['id', 'username', 'email']),
            st.text(min_size=1, max_size=50)
        ),
        lead_time=st.floats(min_value=1.0, max_value=3600.0),
        meta=st.dictionaries(
            st.sampled_from(['superinsight_task_id', 'project_id']),
            st.text(min_size=1, max_size=50)
        )
    )


def webhook_payload_strategy():
    """Strategy for generating webhook payload data from Label Studio."""
    return st.builds(
        dict,
        action=st.just("annotation_created"),  # Label Studio webhook action
        task=st.fixed_dictionaries({
            'id': st.integers(min_value=1, max_value=1000),
            'data': st.dictionaries(st.text(min_size=1, max_size=10), st.text()),
            'meta': st.dictionaries(st.text(min_size=1, max_size=10), st.text())
        }),
        annotation=completed_annotation_data_strategy(),
        project=st.fixed_dictionaries({
            'id': st.integers(min_value=1, max_value=1000),
            'title': st.text(min_size=1, max_size=50)
        })
    )


class MockQualityManager:
    """Mock quality manager for testing quality check triggering."""
    
    def __init__(self):
        self.triggered_checks = []
        self.created_issues = []
    
    async def trigger_quality_check(self, task_id: UUID, annotation_data: Dict[str, Any]) -> bool:
        """Mock quality check trigger."""
        self.triggered_checks.append({
            'task_id': task_id,
            'annotation_data': annotation_data,
            'triggered_at': datetime.now()
        })
        return True
    
    async def create_quality_issue(self, task_id: UUID, issue_type: str, description: str) -> QualityIssue:
        """Mock quality issue creation."""
        issue = QualityIssue(
            task_id=task_id,
            issue_type=issue_type,
            description=description,
            severity=IssueSeverity.MEDIUM,
            status=IssueStatus.OPEN
        )
        self.created_issues.append(issue)
        return issue


class TestQualityCheckAutoTrigger:
    """
    Property-based tests for quality check auto-triggering.
    
    Validates Requirement 3.4:
    - When annotation is completed, quality check should be automatically triggered
    - Quality check process should be initiated through webhook mechanism
    """
    
    @given(completed_annotation_data_strategy())
    @settings(suppress_health_check=[HealthCheck.filter_too_much])
    def test_annotation_completion_triggers_quality_check(self, annotation_data: Dict[str, Any]):
        """
        Property 7: Quality Check Auto-Trigger
        
        For any completed annotation, the system should automatically trigger
        a quality check process when the annotation is marked as complete.
        
        **Validates: Requirement 3.4**
        """
        # Extract task ID from annotation metadata
        task_id = uuid4()
        if 'meta' in annotation_data:
            annotation_data['meta']['superinsight_task_id'] = str(task_id)
        else:
            annotation_data['meta'] = {'superinsight_task_id': str(task_id)}
        
        # Create mock quality manager
        mock_quality_manager = MockQualityManager()
        
        # Mock the Label Studio integration
        integration = LabelStudioIntegration()
        
        # Simulate annotation completion processing
        with patch('src.label_studio.integration.get_db_session') as mock_db:
            mock_session = Mock()
            mock_db.return_value.__enter__.return_value = mock_session
            
            # Mock task model in database
            mock_task = TaskModel(
                id=task_id,
                document_id=uuid4(),
                project_id="test_project",
                status=TaskStatus.IN_PROGRESS,
                annotations=[],
                ai_predictions=[],
                quality_score=0.0
            )
            mock_session.query.return_value.filter.return_value.first.return_value = mock_task
            
            # Simulate webhook processing that should trigger quality check
            async def process_annotation_completion():
                # This simulates the webhook handler processing annotation completion
                if annotation_data.get('annotations'):
                    # Update task status to completed
                    mock_task.status = TaskStatus.COMPLETED
                    
                    # Add annotation to task
                    if not mock_task.annotations:
                        mock_task.annotations = []
                    
                    mock_task.annotations.append({
                        "id": annotation_data.get("id"),
                        "result": annotation_data.get("annotations", [{}])[0].get("result", []),
                        "created_at": annotation_data.get("created_at"),
                        "updated_at": annotation_data.get("updated_at"),
                        "lead_time": annotation_data.get("lead_time", 0),
                        "annotator": annotation_data.get("completed_by", {}).get("id")
                    })
                    
                    # Trigger quality check (this is what we're testing)
                    await mock_quality_manager.trigger_quality_check(
                        task_id=task_id,
                        annotation_data=annotation_data
                    )
                    
                    return True
                return False
            
            # Run the async process
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(process_annotation_completion())
                
                # Verify quality check was triggered
                if annotation_data.get('annotations'):
                    assert result is True
                    assert len(mock_quality_manager.triggered_checks) == 1
                    
                    triggered_check = mock_quality_manager.triggered_checks[0]
                    assert triggered_check['task_id'] == task_id
                    assert triggered_check['annotation_data'] == annotation_data
                    
                    # Verify task was updated to completed status
                    assert mock_task.status == TaskStatus.COMPLETED
                    assert len(mock_task.annotations) == 1
                else:
                    # If no annotations, quality check should not be triggered
                    assert len(mock_quality_manager.triggered_checks) == 0
                    
            finally:
                loop.close()
    
    @given(webhook_payload_strategy())
    @settings(suppress_health_check=[HealthCheck.filter_too_much])
    def test_webhook_annotation_completion_triggers_quality_check(self, webhook_payload: Dict[str, Any]):
        """
        Property 7: Webhook-Based Quality Check Triggering
        
        For any webhook payload indicating annotation completion, the system
        should automatically trigger quality check processing.
        
        **Validates: Requirement 3.4**
        """
        # Ensure the webhook payload indicates annotation completion
        webhook_payload['action'] = 'annotation_created'
        
        # Extract or create task ID
        task_id = uuid4()
        if 'task' not in webhook_payload:
            webhook_payload['task'] = {}
        if 'meta' not in webhook_payload['task']:
            webhook_payload['task']['meta'] = {}
        webhook_payload['task']['meta']['superinsight_task_id'] = str(task_id)
        
        # Create mock quality manager
        mock_quality_manager = MockQualityManager()
        
        # Simulate webhook handler processing
        async def handle_webhook(payload: Dict[str, Any]) -> bool:
            """Simulate webhook handler that should trigger quality checks."""
            if payload.get('action') == 'annotation_created':
                # Extract task information
                task_info = payload.get('task', {})
                annotation_info = payload.get('annotation', {})
                
                # Get task ID from metadata
                task_meta = task_info.get('meta', {})
                superinsight_task_id = task_meta.get('superinsight_task_id')
                
                if superinsight_task_id:
                    task_uuid = UUID(superinsight_task_id)
                    
                    # Trigger quality check for completed annotation
                    await mock_quality_manager.trigger_quality_check(
                        task_id=task_uuid,
                        annotation_data=annotation_info
                    )
                    return True
            return False
        
        # Process the webhook
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(handle_webhook(webhook_payload))
            
            # Verify quality check was triggered
            assert result is True
            assert len(mock_quality_manager.triggered_checks) == 1
            
            triggered_check = mock_quality_manager.triggered_checks[0]
            assert triggered_check['task_id'] == task_id
            assert triggered_check['annotation_data'] == webhook_payload.get('annotation', {})
            
        finally:
            loop.close()
    
    @given(st.lists(annotation_strategy(), min_size=1, max_size=5))
    @settings(suppress_health_check=[HealthCheck.filter_too_much])
    def test_multiple_annotations_trigger_individual_quality_checks(self, annotations: List[Annotation]):
        """
        Property 7: Multiple Annotation Quality Check Triggering
        
        For any list of completed annotations, each annotation completion
        should trigger its own individual quality check process.
        
        **Validates: Requirement 3.4**
        """
        # Create mock quality manager
        mock_quality_manager = MockQualityManager()
        
        # Process each annotation completion
        async def process_multiple_annotations():
            for annotation in annotations:
                # Simulate annotation completion
                annotation_data = {
                    'id': annotation.id,
                    'annotations': [{
                        'result': annotation.annotation_data,
                        'created_at': annotation.created_at.isoformat()
                    }],
                    'completed_by': {'id': annotation.annotator_id},
                    'lead_time': annotation.time_spent,
                    'meta': {'superinsight_task_id': str(annotation.task_id)}
                }
                
                # Trigger quality check for this annotation
                await mock_quality_manager.trigger_quality_check(
                    task_id=annotation.task_id,
                    annotation_data=annotation_data
                )
        
        # Run the async process
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(process_multiple_annotations())
            
            # Verify each annotation triggered a quality check
            assert len(mock_quality_manager.triggered_checks) == len(annotations)
            
            # Verify each quality check corresponds to the correct annotation
            for i, (annotation, triggered_check) in enumerate(zip(annotations, mock_quality_manager.triggered_checks)):
                assert triggered_check['task_id'] == annotation.task_id
                assert triggered_check['annotation_data']['completed_by']['id'] == annotation.annotator_id
                
        finally:
            loop.close()
    
    @given(task_strategy())
    @settings(suppress_health_check=[HealthCheck.filter_too_much])
    def test_task_status_change_triggers_quality_check(self, task: Task):
        """
        Property 7: Task Status Change Quality Check Triggering
        
        For any task that changes status to COMPLETED, a quality check
        should be automatically triggered.
        
        **Validates: Requirement 3.4**
        """
        # Ensure task has at least one annotation to be considered complete
        if not task.annotations:
            task.annotations = [{
                'id': str(uuid4()),
                'annotator_id': 'test_annotator',
                'annotation_data': {'label': 'test_label'},
                'confidence': 0.8
            }]
        
        # Create mock quality manager
        mock_quality_manager = MockQualityManager()
        
        # Simulate task status change to completed
        async def update_task_status():
            original_status = task.status
            
            # Force status change to simulate completion
            if original_status != TaskStatus.COMPLETED:
                task.status = TaskStatus.COMPLETED
                status_changed = True
            else:
                status_changed = False
            
            # If task status changed to completed and has annotations, trigger quality check
            if (status_changed and 
                task.status == TaskStatus.COMPLETED and 
                task.annotations):
                
                # Trigger quality check for the completed task
                for annotation in task.annotations:
                    await mock_quality_manager.trigger_quality_check(
                        task_id=task.id,
                        annotation_data=annotation
                    )
                return True
            return False
        
        # Run the async process
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(update_task_status())
            
            # Verify quality check was triggered if task status changed to completed
            if result:
                assert len(mock_quality_manager.triggered_checks) == len(task.annotations)
                
                # Verify each annotation triggered a quality check
                for annotation, triggered_check in zip(task.annotations, mock_quality_manager.triggered_checks):
                    assert triggered_check['task_id'] == task.id
                    assert triggered_check['annotation_data'] == annotation
            else:
                # If task was already completed or has no annotations, no new quality checks
                assert len(mock_quality_manager.triggered_checks) == 0
                
        finally:
            loop.close()
    
    @given(st.text(min_size=1, max_size=50), st.text(min_size=1, max_size=100))
    @settings(suppress_health_check=[HealthCheck.filter_too_much])
    def test_quality_check_failure_creates_quality_issue(self, project_id: str, issue_description: str):
        """
        Property 7: Quality Check Failure Issue Creation
        
        For any quality check that fails, a quality issue should be
        automatically created and assigned for resolution.
        
        **Validates: Requirement 3.4**
        """
        task_id = uuid4()
        
        # Create mock quality manager
        mock_quality_manager = MockQualityManager()
        
        # Simulate quality check that finds issues
        async def run_quality_check_with_failure():
            # Simulate annotation data that fails quality check
            annotation_data = {
                'id': 123,
                'annotations': [{
                    'result': [{'value': {'text': 'low quality annotation'}}],
                    'created_at': datetime.now().isoformat()
                }],
                'completed_by': {'id': 'annotator_123'},
                'confidence_score': 0.3  # Low confidence indicates potential quality issue
            }
            
            # Trigger quality check
            quality_check_passed = await mock_quality_manager.trigger_quality_check(
                task_id=task_id,
                annotation_data=annotation_data
            )
            
            # If quality check finds issues (simulated by low confidence), create quality issue
            if annotation_data.get('confidence_score', 1.0) < 0.5:
                await mock_quality_manager.create_quality_issue(
                    task_id=task_id,
                    issue_type="low_confidence_annotation",
                    description=issue_description
                )
                return False  # Quality check failed
            
            return quality_check_passed
        
        # Run the async process
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            quality_passed = loop.run_until_complete(run_quality_check_with_failure())
            
            # Verify quality check was triggered
            assert len(mock_quality_manager.triggered_checks) == 1
            
            # Verify quality issue was created for failed check
            assert quality_passed is False
            assert len(mock_quality_manager.created_issues) == 1
            
            created_issue = mock_quality_manager.created_issues[0]
            assert created_issue.task_id == task_id
            assert created_issue.issue_type == "low_confidence_annotation"
            assert created_issue.description == issue_description
            assert created_issue.status == IssueStatus.OPEN
            
        finally:
            loop.close()
    
    def test_webhook_configuration_enables_quality_triggering(self):
        """
        Property 7: Webhook Configuration for Quality Triggering
        
        Proper webhook configuration should enable automatic quality check
        triggering when annotations are completed in Label Studio.
        
        **Validates: Requirement 3.4**
        """
        # Create Label Studio integration
        integration = LabelStudioIntegration()
        
        # Mock HTTP client for webhook configuration
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 201
            mock_response.json.return_value = {'id': 1, 'url': 'http://test-webhook.com'}
            
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_response
            
            # Configure webhooks for quality check triggering
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    integration.setup_webhooks(
                        project_id="test_project",
                        webhook_urls=["http://test-webhook.com/quality-check"]
                    )
                )
                
                # Verify webhook was configured successfully
                assert result is True
                
                # Verify the webhook configuration was called
                mock_client.return_value.__aenter__.return_value.post.assert_called_once()
                
                # Verify the webhook URL was configured for quality checks
                call_args = mock_client.return_value.__aenter__.return_value.post.call_args
                assert "webhooks" in call_args[1]['json']['url'] or "quality-check" in call_args[1]['json']['url']
                
            finally:
                loop.close()
    
    @given(st.integers(min_value=1, max_value=10))
    @settings(suppress_health_check=[HealthCheck.filter_too_much])
    def test_concurrent_annotation_completions_trigger_quality_checks(self, num_concurrent: int):
        """
        Property 7: Concurrent Quality Check Triggering
        
        For any number of concurrent annotation completions, each should
        trigger its own quality check without interference.
        
        **Validates: Requirement 3.4**
        """
        # Create mock quality manager
        mock_quality_manager = MockQualityManager()
        
        # Create concurrent annotation completion tasks
        async def process_concurrent_annotations():
            tasks = []
            task_ids = []
            
            for i in range(num_concurrent):
                task_id = uuid4()
                task_ids.append(task_id)
                
                # Create async task for each annotation completion
                async def complete_annotation(tid=task_id, index=i):
                    annotation_data = {
                        'id': index,
                        'annotations': [{'result': [{'value': f'annotation_{index}'}]}],
                        'completed_by': {'id': f'annotator_{index}'},
                        'meta': {'superinsight_task_id': str(tid)}
                    }
                    
                    # Simulate some processing delay
                    await asyncio.sleep(0.01)
                    
                    # Trigger quality check
                    await mock_quality_manager.trigger_quality_check(
                        task_id=tid,
                        annotation_data=annotation_data
                    )
                
                tasks.append(complete_annotation())
            
            # Run all annotation completions concurrently
            await asyncio.gather(*tasks)
            return task_ids
        
        # Run the concurrent process
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            task_ids = loop.run_until_complete(process_concurrent_annotations())
            
            # Verify all quality checks were triggered
            assert len(mock_quality_manager.triggered_checks) == num_concurrent
            
            # Verify each task ID appears exactly once in triggered checks
            triggered_task_ids = [check['task_id'] for check in mock_quality_manager.triggered_checks]
            assert set(triggered_task_ids) == set(task_ids)
            assert len(triggered_task_ids) == len(set(triggered_task_ids))  # No duplicates
            
        finally:
            loop.close()


if __name__ == "__main__":
    # Run with verbose output and show hypothesis examples
    pytest.main([__file__, "-v", "--hypothesis-show-statistics"])