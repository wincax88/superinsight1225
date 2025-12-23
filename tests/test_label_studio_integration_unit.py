"""
Unit tests for Label Studio integration functionality.

Tests project creation and configuration, task import/export, and webhook trigger mechanisms
as specified in Requirements 3.1, 3.4, 3.5.
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from uuid import uuid4, UUID
from datetime import datetime
from typing import Dict, Any, List
import httpx
from httpx import Response

from src.label_studio.integration import (
    LabelStudioIntegration,
    LabelStudioIntegrationError,
    ProjectConfig,
    ImportResult,
    ExportResult
)
from src.label_studio.config import LabelStudioConfig, LabelStudioProject
from src.models.task import Task, TaskStatus
from src.models.document import Document


class TestLabelStudioIntegrationProjectCreation:
    """Unit tests for Label Studio project creation and configuration."""
    
    @pytest.fixture
    def mock_config(self):
        """Mock Label Studio configuration."""
        config = Mock(spec=LabelStudioConfig)
        config.base_url = "https://labelstudio.example.com"
        config.api_token = "test_token_123"
        config.validate_config.return_value = True
        config.get_default_label_config.return_value = "<View><Text name='text' value='$text'/></View>"
        return config
    
    @pytest.fixture
    def integration(self, mock_config):
        """Label Studio integration instance with mocked config."""
        return LabelStudioIntegration(mock_config)
    
    @pytest.fixture
    def project_config(self):
        """Sample project configuration."""
        return ProjectConfig(
            title="Test Annotation Project",
            description="Test project for unit testing",
            annotation_type="text_classification"
        )
    
    @pytest.mark.asyncio
    async def test_create_project_success(self, integration, project_config):
        """Test successful project creation."""
        # Mock successful API response
        mock_response_data = {
            "id": 123,
            "title": "Test Annotation Project",
            "description": "Test project for unit testing",
            "label_config": "<View><Text name='text' value='$text'/></View>",
            "created_at": "2024-01-01T12:00:00Z",
            "is_published": False
        }
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 201
            mock_response.json.return_value = mock_response_data
            
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
            
            result = await integration.create_project(project_config)
            
            # Verify result
            assert isinstance(result, LabelStudioProject)
            assert result.id == 123
            assert result.title == "Test Annotation Project"
            assert result.description == "Test project for unit testing"
            
            # Verify API call was made correctly
            mock_client.return_value.__aenter__.return_value.post.assert_called_once()
            call_args = mock_client.return_value.__aenter__.return_value.post.call_args
            assert "projects" in call_args[0][0]  # URL contains 'projects'
            
            # Verify request data
            request_data = call_args[1]['json']
            assert request_data['title'] == "Test Annotation Project"
            assert request_data['description'] == "Test project for unit testing"
    
    @pytest.mark.asyncio
    async def test_create_project_api_error(self, integration, project_config):
        """Test project creation with API error."""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 400
            mock_response.text = "Bad Request: Invalid project configuration"
            
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
            
            with pytest.raises(LabelStudioIntegrationError, match="Failed to create project"):
                await integration.create_project(project_config)
    
    @pytest.mark.asyncio
    async def test_create_project_network_error(self, integration, project_config):
        """Test project creation with network error."""
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=httpx.RequestError("Network error")
            )
            
            with pytest.raises(LabelStudioIntegrationError, match="Network error creating project"):
                await integration.create_project(project_config)
    
    @pytest.mark.asyncio
    async def test_create_project_with_custom_label_config(self, integration):
        """Test project creation with custom label configuration."""
        custom_config = ProjectConfig(
            title="Custom Config Project",
            description="Project with custom label config",
            annotation_type="named_entity_recognition",
            label_config="<View><Text name='text' value='$text'/><Labels name='label' toName='text'><Label value='PERSON'/></Labels></View>"
        )
        
        mock_response_data = {
            "id": 456,
            "title": "Custom Config Project",
            "label_config": custom_config.label_config,
            "created_at": "2024-01-01T12:00:00Z"
        }
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 201
            mock_response.json.return_value = mock_response_data
            
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
            
            result = await integration.create_project(custom_config)
            
            assert result.id == 456
            assert result.title == "Custom Config Project"
            
            # Verify custom label config was used
            call_args = mock_client.return_value.__aenter__.return_value.post.call_args
            request_data = call_args[1]['json']
            assert request_data['label_config'] == custom_config.label_config
    
    @pytest.mark.asyncio
    async def test_get_project_info_success(self, integration):
        """Test successful project info retrieval."""
        project_id = "123"
        mock_response_data = {
            "id": 123,
            "title": "Existing Project",
            "description": "Project description",
            "label_config": "<View><Text name='text' value='$text'/></View>",
            "created_at": "2024-01-01T12:00:00Z"
        }
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_response_data
            
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)
            
            result = await integration.get_project_info(project_id)
            
            assert isinstance(result, LabelStudioProject)
            assert result.id == 123
            assert result.title == "Existing Project"
    
    @pytest.mark.asyncio
    async def test_get_project_info_not_found(self, integration):
        """Test project info retrieval for non-existent project."""
        project_id = "999"
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 404
            
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)
            
            result = await integration.get_project_info(project_id)
            
            assert result is None
    
    @pytest.mark.asyncio
    async def test_delete_project_success(self, integration):
        """Test successful project deletion."""
        project_id = "123"
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 204
            
            mock_client.return_value.__aenter__.return_value.delete = AsyncMock(return_value=mock_response)
            
            result = await integration.delete_project(project_id)
            
            assert result is True
    
    @pytest.mark.asyncio
    async def test_delete_project_failure(self, integration):
        """Test project deletion failure."""
        project_id = "123"
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 404
            
            mock_client.return_value.__aenter__.return_value.delete = AsyncMock(return_value=mock_response)
            
            result = await integration.delete_project(project_id)
            
            assert result is False


class TestLabelStudioIntegrationTaskImportExport:
    """Unit tests for task import and export functionality."""
    
    @pytest.fixture
    def integration(self):
        """Label Studio integration instance."""
        config = Mock(spec=LabelStudioConfig)
        config.base_url = "https://labelstudio.example.com"
        config.api_token = "test_token_123"
        config.validate_config.return_value = True
        return LabelStudioIntegration(config)
    
    @pytest.fixture
    def sample_tasks(self):
        """Sample tasks for testing."""
        doc_id1 = uuid4()
        doc_id2 = uuid4()
        
        tasks = [
            Task(
                id=uuid4(),
                document_id=doc_id1,
                project_id="test_project",
                status=TaskStatus.PENDING,
                ai_predictions=[{
                    "model": "test_model",
                    "result": [{"value": {"choices": ["positive"]}}],
                    "confidence": 0.85
                }]
            ),
            Task(
                id=uuid4(),
                document_id=doc_id2,
                project_id="test_project",
                status=TaskStatus.PENDING
            )
        ]
        return tasks
    
    @pytest.fixture
    def sample_documents(self, sample_tasks):
        """Sample documents corresponding to tasks."""
        return [
            Document(
                id=sample_tasks[0].document_id,
                source_type="database",
                source_config={"table": "reviews"},
                content="This is a positive review of the product."
            ),
            Document(
                id=sample_tasks[1].document_id,
                source_type="file",
                source_config={"path": "/data/review2.txt"},
                content="This is another review to be annotated."
            )
        ]
    
    @pytest.mark.asyncio
    async def test_import_tasks_success(self, integration, sample_tasks, sample_documents):
        """Test successful task import."""
        project_id = "test_project_123"
        
        # Mock database session and document retrieval
        with patch('src.label_studio.integration.get_db_session') as mock_db_session:
            mock_db = Mock()
            mock_db_session.return_value.__enter__.return_value = mock_db
            
            # Mock document queries
            mock_db.query.return_value.filter.return_value.first.side_effect = [
                Mock(id=sample_tasks[0].document_id, content=sample_documents[0].content),
                Mock(id=sample_tasks[1].document_id, content=sample_documents[1].content)
            ]
            
            # Mock successful API response
            with patch('httpx.AsyncClient') as mock_client:
                mock_response = Mock()
                mock_response.status_code = 201
                mock_response.json.return_value = {"task_count": 2}
                
                mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
                
                # Mock the sync method
                with patch.object(integration, '_sync_tasks_to_db', new_callable=AsyncMock) as mock_sync:
                    result = await integration.import_tasks(project_id, sample_tasks)
                    
                    # Verify result
                    assert isinstance(result, ImportResult)
                    assert result.success is True
                    assert result.imported_count == 2
                    assert result.failed_count == 0
                    
                    # Verify sync was called
                    mock_sync.assert_called_once_with(project_id, sample_tasks)
    
    @pytest.mark.asyncio
    async def test_import_tasks_document_not_found(self, integration, sample_tasks):
        """Test task import when document is not found."""
        project_id = "test_project_123"
        
        # Mock database session with no documents found
        with patch('src.label_studio.integration.get_db_session') as mock_db_session:
            mock_db = Mock()
            mock_db_session.return_value.__enter__.return_value = mock_db
            mock_db.query.return_value.filter.return_value.first.return_value = None
            
            result = await integration.import_tasks(project_id, sample_tasks)
            
            # Should fail because documents not found
            assert result.success is False
            assert result.failed_count == len(sample_tasks)
            assert len(result.errors) == len(sample_tasks)
            assert all("Document not found" in error for error in result.errors)
    
    @pytest.mark.asyncio
    async def test_import_tasks_api_error(self, integration, sample_tasks, sample_documents):
        """Test task import with API error."""
        project_id = "test_project_123"
        
        # Mock database session
        with patch('src.label_studio.integration.get_db_session') as mock_db_session:
            mock_db = Mock()
            mock_db_session.return_value.__enter__.return_value = mock_db
            mock_db.query.return_value.filter.return_value.first.side_effect = [
                Mock(id=sample_tasks[0].document_id, content=sample_documents[0].content),
                Mock(id=sample_tasks[1].document_id, content=sample_documents[1].content)
            ]
            
            # Mock API error response
            with patch('httpx.AsyncClient') as mock_client:
                mock_response = Mock()
                mock_response.status_code = 400
                mock_response.text = "Bad Request"
                
                mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
                
                result = await integration.import_tasks(project_id, sample_tasks)
                
                # Should handle API error gracefully
                assert result.success is False
                assert result.failed_count == len(sample_tasks)
                assert len(result.errors) > 0
    
    @pytest.mark.asyncio
    async def test_export_annotations_success(self, integration):
        """Test successful annotation export."""
        project_id = "test_project_123"
        
        # Mock annotation data from Label Studio
        mock_annotations = [
            {
                "id": 1,
                "data": {"text": "Sample text", "task_id": str(uuid4())},
                "annotations": [{
                    "id": 101,
                    "result": [{"value": {"choices": ["positive"]}}],
                    "created_at": "2024-01-01T12:00:00Z",
                    "updated_at": "2024-01-01T12:05:00Z",
                    "lead_time": 300,
                    "completed_by": {"id": 1}
                }],
                "meta": {"superinsight_task_id": str(uuid4())},
                "created_at": "2024-01-01T12:00:00Z",
                "updated_at": "2024-01-01T12:05:00Z"
            },
            {
                "id": 2,
                "data": {"text": "Another text", "task_id": str(uuid4())},
                "annotations": [{
                    "id": 102,
                    "result": [{"value": {"choices": ["negative"]}}],
                    "created_at": "2024-01-01T12:10:00Z",
                    "updated_at": "2024-01-01T12:15:00Z",
                    "lead_time": 250,
                    "completed_by": {"id": 2}
                }],
                "meta": {"superinsight_task_id": str(uuid4())},
                "created_at": "2024-01-01T12:10:00Z",
                "updated_at": "2024-01-01T12:15:00Z"
            }
        ]
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_annotations
            
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)
            
            # Mock the sync method
            with patch.object(integration, '_sync_annotations_to_db', new_callable=AsyncMock) as mock_sync:
                result = await integration.export_annotations(project_id, "JSON")
                
                # Verify result
                assert isinstance(result, ExportResult)
                assert result.success is True
                assert result.exported_count == 2
                assert len(result.data) == 2
                assert result.data == mock_annotations
                
                # Verify sync was called
                mock_sync.assert_called_once_with(project_id, mock_annotations)
    
    @pytest.mark.asyncio
    async def test_export_annotations_api_error(self, integration):
        """Test annotation export with API error."""
        project_id = "test_project_123"
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 404
            mock_response.text = "Project not found"
            
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)
            
            result = await integration.export_annotations(project_id, "JSON")
            
            # Should handle API error gracefully
            assert result.success is False
            assert result.exported_count == 0
            assert len(result.errors) > 0
            assert "Failed to export annotations" in result.errors[0]
    
    @pytest.mark.asyncio
    async def test_export_annotations_different_formats(self, integration):
        """Test annotation export with different formats."""
        project_id = "test_project_123"
        formats = ["JSON", "CSV", "COCO"]
        
        for export_format in formats:
            with patch('httpx.AsyncClient') as mock_client:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = []
                
                mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)
                
                with patch.object(integration, '_sync_annotations_to_db', new_callable=AsyncMock):
                    result = await integration.export_annotations(project_id, export_format)
                    
                    assert result.success is True
                    
                    # Verify correct format parameter was passed
                    call_args = mock_client.return_value.__aenter__.return_value.get.call_args
                    assert call_args[1]['params']['exportType'] == export_format


class TestLabelStudioIntegrationWebhooks:
    """Unit tests for webhook configuration and trigger mechanisms."""
    
    @pytest.fixture
    def integration(self):
        """Label Studio integration instance."""
        config = Mock(spec=LabelStudioConfig)
        config.base_url = "https://labelstudio.example.com"
        config.api_token = "test_token_123"
        config.validate_config.return_value = True
        config.get_webhook_config.return_value = {
            "url": "https://superinsight.example.com/webhook",
            "send_payload": True,
            "send_for_all_actions": False,
            "headers": {"Content-Type": "application/json"},
            "actions": ["ANNOTATION_CREATED", "ANNOTATION_UPDATED", "TASK_COMPLETED"]
        }
        return LabelStudioIntegration(config)
    
    @pytest.mark.asyncio
    async def test_setup_webhooks_success(self, integration):
        """Test successful webhook configuration."""
        project_id = "test_project_123"
        webhook_urls = [
            "https://superinsight.example.com/webhook/quality-check",
            "https://superinsight.example.com/webhook/progress-tracking"
        ]
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 201
            mock_response.json.return_value = {"id": 1, "url": webhook_urls[0]}
            
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
            
            result = await integration.setup_webhooks(project_id, webhook_urls)
            
            # Verify result
            assert result is True
            
            # Verify API calls were made for each webhook
            assert mock_client.return_value.__aenter__.return_value.post.call_count == len(webhook_urls)
    
    @pytest.mark.asyncio
    async def test_setup_webhooks_partial_failure(self, integration):
        """Test webhook configuration with partial failure."""
        project_id = "test_project_123"
        webhook_urls = [
            "https://superinsight.example.com/webhook/quality-check",
            "https://invalid-url.example.com/webhook"
        ]
        
        with patch('httpx.AsyncClient') as mock_client:
            # First webhook succeeds, second fails
            responses = [
                Mock(status_code=201, json=lambda: {"id": 1}),
                Mock(status_code=400, text="Invalid webhook URL")
            ]
            
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(side_effect=responses)
            
            result = await integration.setup_webhooks(project_id, webhook_urls)
            
            # Should return False if any webhook fails
            assert result is False
    
    @pytest.mark.asyncio
    async def test_setup_webhooks_network_error(self, integration):
        """Test webhook configuration with network error."""
        project_id = "test_project_123"
        webhook_urls = ["https://superinsight.example.com/webhook"]
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=httpx.RequestError("Network error")
            )
            
            result = await integration.setup_webhooks(project_id, webhook_urls)
            
            assert result is False
    
    @pytest.mark.asyncio
    async def test_configure_ml_backend_success(self, integration):
        """Test successful ML backend configuration."""
        project_id = "test_project_123"
        ml_backend_url = "https://ai.superinsight.example.com/predict"
        
        # Mock ML backend config
        integration.config.get_ml_backend_config.return_value = {
            "url": ml_backend_url,
            "title": "SuperInsight AI Backend",
            "description": "AI prediction service",
            "model_version": "1.0.0",
            "is_interactive": True
        }
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 201
            mock_response.json.return_value = {"id": 1, "url": ml_backend_url}
            
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
            
            result = await integration.configure_ml_backend(project_id, ml_backend_url)
            
            # Verify result
            assert result is True
            
            # Verify API call was made correctly
            call_args = mock_client.return_value.__aenter__.return_value.post.call_args
            assert "ml" in call_args[0][0]  # URL contains 'ml'
            
            # Verify project ID was included in request
            request_data = call_args[1]['json']
            assert request_data['project'] == project_id
    
    @pytest.mark.asyncio
    async def test_configure_ml_backend_failure(self, integration):
        """Test ML backend configuration failure."""
        project_id = "test_project_123"
        ml_backend_url = "https://invalid-ai.example.com/predict"
        
        integration.config.get_ml_backend_config.return_value = {
            "url": ml_backend_url,
            "title": "Invalid Backend"
        }
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 400
            mock_response.text = "Invalid ML backend configuration"
            
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
            
            result = await integration.configure_ml_backend(project_id, ml_backend_url)
            
            assert result is False


class TestLabelStudioIntegrationDataSync:
    """Unit tests for data synchronization between Label Studio and PostgreSQL."""
    
    @pytest.fixture
    def integration(self):
        """Label Studio integration instance."""
        config = Mock(spec=LabelStudioConfig)
        config.base_url = "https://labelstudio.example.com"
        config.api_token = "test_token_123"
        config.validate_config.return_value = True
        return LabelStudioIntegration(config)
    
    @pytest.mark.asyncio
    async def test_sync_tasks_to_db_success(self, integration):
        """Test successful task synchronization to database."""
        project_id = "test_project_123"
        tasks = [
            Task(
                id=uuid4(),
                document_id=uuid4(),
                project_id=project_id,
                status=TaskStatus.PENDING
            )
        ]
        
        with patch('src.label_studio.integration.get_db_session') as mock_db_session:
            mock_db = Mock()
            mock_db_session.return_value.__enter__.return_value = mock_db
            
            # Mock existing task query
            mock_db.query.return_value.filter.return_value.first.return_value = None
            
            await integration._sync_tasks_to_db(project_id, tasks)
            
            # Verify database operations
            mock_db.add.assert_called_once()
            mock_db.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_sync_tasks_to_db_update_existing(self, integration):
        """Test updating existing tasks in database."""
        project_id = "test_project_123"
        task_id = uuid4()
        tasks = [
            Task(
                id=task_id,
                document_id=uuid4(),
                project_id=project_id,
                status=TaskStatus.PENDING
            )
        ]
        
        with patch('src.label_studio.integration.get_db_session') as mock_db_session:
            mock_db = Mock()
            mock_db_session.return_value.__enter__.return_value = mock_db
            
            # Mock existing task
            existing_task = Mock()
            existing_task.id = task_id
            mock_db.query.return_value.filter.return_value.first.return_value = existing_task
            
            await integration._sync_tasks_to_db(project_id, tasks)
            
            # Verify task was updated
            assert existing_task.project_id == project_id
            assert existing_task.status == TaskStatus.PENDING
            mock_db.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_sync_annotations_to_db_success(self, integration):
        """Test successful annotation synchronization to database."""
        project_id = "test_project_123"
        task_id = uuid4()
        
        annotations_data = [
            {
                "id": 1,
                "data": {"task_id": str(task_id)},
                "annotations": [{
                    "id": 101,
                    "result": [{"value": {"choices": ["positive"]}}],
                    "created_at": "2024-01-01T12:00:00Z",
                    "updated_at": "2024-01-01T12:05:00Z",
                    "lead_time": 300,
                    "completed_by": {"id": 1}
                }],
                "created_at": "2024-01-01T12:00:00Z",
                "updated_at": "2024-01-01T12:05:00Z"
            }
        ]
        
        with patch('src.label_studio.integration.get_db_session') as mock_db_session:
            mock_db = Mock()
            mock_db_session.return_value.__enter__.return_value = mock_db
            
            # Mock existing task
            mock_task = Mock()
            mock_task.id = task_id
            mock_task.annotations = []
            mock_db.query.return_value.filter.return_value.first.return_value = mock_task
            
            await integration._sync_annotations_to_db(project_id, annotations_data)
            
            # Verify annotation was added to task
            assert len(mock_task.annotations) == 1
            assert mock_task.annotations[0]["id"] == 1  # This should be the annotation ID from the data
            assert mock_task.status == TaskStatus.COMPLETED
            mock_db.add.assert_called_once_with(mock_task)
            mock_db.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_sync_annotations_to_db_task_not_found(self, integration):
        """Test annotation sync when task is not found."""
        project_id = "test_project_123"
        
        annotations_data = [
            {
                "id": 1,
                "data": {"task_id": str(uuid4())},
                "annotations": [{"id": 101}]
            }
        ]
        
        with patch('src.label_studio.integration.get_db_session') as mock_db_session:
            mock_db = Mock()
            mock_db_session.return_value.__enter__.return_value = mock_db
            
            # Mock task not found
            mock_db.query.return_value.filter.return_value.first.return_value = None
            
            # Should not raise exception, just log warning
            await integration._sync_annotations_to_db(project_id, annotations_data)
            
            # Verify no database operations were performed
            mock_db.add.assert_not_called()
            mock_db.commit.assert_called_once()  # Commit is still called
    
    @pytest.mark.asyncio
    async def test_sync_annotations_to_db_missing_task_id(self, integration):
        """Test annotation sync with missing task ID."""
        project_id = "test_project_123"
        
        annotations_data = [
            {
                "id": 1,
                "data": {"text": "Some text"},  # No task_id
                "annotations": [{"id": 101}]
            }
        ]
        
        with patch('src.label_studio.integration.get_db_session') as mock_db_session:
            mock_db = Mock()
            mock_db_session.return_value.__enter__.return_value = mock_db
            
            # Should not raise exception, just skip the annotation
            await integration._sync_annotations_to_db(project_id, annotations_data)
            
            # Verify no database operations were performed
            mock_db.query.assert_not_called()
            mock_db.commit.assert_called_once()


class TestLabelStudioIntegrationErrorHandling:
    """Unit tests for error handling and edge cases."""
    
    @pytest.fixture
    def integration(self):
        """Label Studio integration instance."""
        config = Mock(spec=LabelStudioConfig)
        config.base_url = "https://labelstudio.example.com"
        config.api_token = "test_token_123"
        config.validate_config.return_value = True
        return LabelStudioIntegration(config)
    
    def test_integration_invalid_config(self):
        """Test integration initialization with invalid config."""
        config = Mock(spec=LabelStudioConfig)
        config.base_url = "https://labelstudio.example.com"
        config.api_token = "test_token_123"
        config.validate_config.return_value = False
        
        with pytest.raises(LabelStudioIntegrationError, match="Invalid Label Studio configuration"):
            LabelStudioIntegration(config)
    
    @pytest.mark.asyncio
    async def test_import_tasks_empty_list(self, integration):
        """Test importing empty task list."""
        project_id = "test_project_123"
        
        # Mock the sync method to avoid database issues
        with patch.object(integration, '_sync_tasks_to_db', new_callable=AsyncMock) as mock_sync:
            result = await integration.import_tasks(project_id, [])
            
            assert result.success is True
            assert result.imported_count == 0
            assert result.failed_count == 0
            
            # Verify sync was called with empty list
            mock_sync.assert_called_once_with(project_id, [])
    
    @pytest.mark.asyncio
    async def test_export_annotations_empty_response(self, integration):
        """Test exporting annotations with empty response."""
        project_id = "test_project_123"
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = []
            
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)
            
            with patch.object(integration, '_sync_annotations_to_db', new_callable=AsyncMock):
                result = await integration.export_annotations(project_id, "JSON")
                
                assert result.success is True
                assert result.exported_count == 0
                assert len(result.data) == 0
    
    @pytest.mark.asyncio
    async def test_setup_webhooks_empty_list(self, integration):
        """Test setting up webhooks with empty URL list."""
        project_id = "test_project_123"
        
        result = await integration.setup_webhooks(project_id, [])
        
        assert result is True  # Should succeed with no operations
    
    @pytest.mark.asyncio
    async def test_database_sync_error_handling(self, integration):
        """Test database synchronization error handling."""
        project_id = "test_project_123"
        tasks = [Task(id=uuid4(), document_id=uuid4(), project_id=project_id)]
        
        with patch('src.label_studio.integration.get_db_session') as mock_db_session:
            mock_db_session.side_effect = Exception("Database connection failed")
            
            with pytest.raises(Exception, match="Database connection failed"):
                await integration._sync_tasks_to_db(project_id, tasks)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])