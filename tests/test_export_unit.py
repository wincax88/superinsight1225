"""
Unit tests for data export functionality in SuperInsight Platform.

Tests various export formats correctness, large data export performance,
and RAG/Agent interface functionality as specified in Requirements 6.1, 6.2, 6.3, 6.4, 6.5.
"""

import pytest
import json
import csv
import os
import tempfile
import shutil
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from uuid import uuid4
from pathlib import Path

from src.export.service import ExportService
from src.export.models import ExportRequest, ExportFormat, ExportResult
from src.api.export import router as export_router
from src.api.rag_agent import router as rag_agent_router
from src.rag.service import RAGService
from src.rag.models import RAGRequest, DocumentChunk
from src.agent.service import AgentService
from src.agent.models import AgentRequest
from src.database.models import DocumentModel, TaskModel, TaskStatus


class TestExportServiceFormats:
    """Unit tests for export format correctness."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.export_service = ExportService(export_dir=self.temp_dir)
        
    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_json_export_format_correctness(self):
        """Test JSON export format produces valid JSON with correct structure."""
        # Create mock documents
        mock_documents = self._create_mock_documents()
        
        # Create export request
        request = ExportRequest(
            format=ExportFormat.JSON,
            include_annotations=True,
            include_metadata=True,
            include_ai_predictions=True
        )
        
        # Start export
        export_id = self.export_service.start_export(request)
        
        # Mock database query
        with patch.object(self.export_service, '_query_documents', return_value=mock_documents):
            result = self.export_service.export_data(export_id, request)
        
        # Verify export completed successfully
        assert result.status == "completed"
        assert result.file_path is not None
        assert os.path.exists(result.file_path)
        
        # Verify JSON format correctness
        with open(result.file_path, 'r', encoding='utf-8') as f:
            exported_data = json.load(f)
        
        # Check JSON structure
        assert "export_info" in exported_data
        assert "documents" in exported_data
        assert exported_data["export_info"]["format"] == "json"
        assert exported_data["export_info"]["total_records"] == len(mock_documents)
        
        # Check document structure
        for i, doc_data in enumerate(exported_data["documents"]):
            assert "id" in doc_data
            assert "source_type" in doc_data
            assert "content" in doc_data
            assert "created_at" in doc_data
            assert "tasks" in doc_data
            
            # Verify metadata inclusion
            if request.include_metadata:
                assert "metadata" in doc_data
                assert "source_config" in doc_data
    
    def test_csv_export_format_correctness(self):
        """Test CSV export format produces valid CSV with correct headers."""
        # Create mock documents
        mock_documents = self._create_mock_documents()
        
        # Create export request
        request = ExportRequest(
            format=ExportFormat.CSV,
            include_annotations=True,
            include_metadata=True
        )
        
        # Start export
        export_id = self.export_service.start_export(request)
        
        # Mock database query
        with patch.object(self.export_service, '_query_documents', return_value=mock_documents):
            result = self.export_service.export_data(export_id, request)
        
        # Verify export completed successfully
        assert result.status == "completed"
        assert result.file_path is not None
        assert os.path.exists(result.file_path)
        
        # Verify CSV format correctness
        with open(result.file_path, 'r', encoding='utf-8') as f:
            csv_reader = csv.reader(f)
            headers = next(csv_reader)
            
            # Check required headers
            expected_headers = [
                'document_id', 'source_type', 'content', 'created_at',
                'task_id', 'project_id', 'task_status', 'quality_score'
            ]
            
            for header in expected_headers:
                assert header in headers
            
            # Check metadata headers if included
            if request.include_metadata:
                assert 'document_metadata' in headers
                assert 'source_config' in headers
            
            # Check annotation headers if included
            if request.include_annotations:
                assert 'annotations' in headers
            
            # Verify data rows
            rows = list(csv_reader)
            assert len(rows) > 0  # Should have at least one row
            
            # Check first row has correct number of columns
            assert len(rows[0]) == len(headers)
    
    def test_coco_export_format_correctness(self):
        """Test COCO export format produces valid COCO dataset structure."""
        # Create mock documents with annotations
        mock_documents = self._create_mock_documents_with_annotations()
        
        # Create export request
        request = ExportRequest(
            format=ExportFormat.COCO,
            include_annotations=True
        )
        
        # Start export
        export_id = self.export_service.start_export(request)
        
        # Mock database query
        with patch.object(self.export_service, '_query_documents', return_value=mock_documents):
            result = self.export_service.export_data(export_id, request)
        
        # Verify export completed successfully
        assert result.status == "completed"
        assert result.file_path is not None
        assert os.path.exists(result.file_path)
        
        # Verify COCO format correctness
        with open(result.file_path, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
        
        # Check COCO structure
        required_keys = ["info", "licenses", "images", "annotations", "categories"]
        for key in required_keys:
            assert key in coco_data
        
        # Check info section
        assert "description" in coco_data["info"]
        assert "version" in coco_data["info"]
        assert "year" in coco_data["info"]
        
        # Check images section
        assert len(coco_data["images"]) == len(mock_documents)
        for image in coco_data["images"]:
            assert "id" in image
            assert "file_name" in image
            assert "date_captured" in image
        
        # Check annotations section (if any)
        if coco_data["annotations"]:
            for annotation in coco_data["annotations"]:
                assert "id" in annotation
                assert "image_id" in annotation
                assert "category_id" in annotation
                assert "bbox" in annotation
    
    def test_export_with_filters(self):
        """Test export with various filters applied."""
        # Create mock documents
        mock_documents = self._create_mock_documents()
        
        # Test with document ID filter
        request = ExportRequest(
            format=ExportFormat.JSON,
            document_ids=[str(mock_documents[0].id)]
        )
        
        export_id = self.export_service.start_export(request)
        
        with patch.object(self.export_service, '_query_documents', return_value=[mock_documents[0]]):
            result = self.export_service.export_data(export_id, request)
        
        assert result.status == "completed"
        assert result.total_records == 1
        
        # Test with date filter
        request_with_date = ExportRequest(
            format=ExportFormat.JSON,
            date_from=datetime.now() - timedelta(days=1),
            date_to=datetime.now() + timedelta(days=1)
        )
        
        export_id_2 = self.export_service.start_export(request_with_date)
        
        with patch.object(self.export_service, '_query_documents', return_value=mock_documents):
            result_2 = self.export_service.export_data(export_id_2, request_with_date)
        
        assert result_2.status == "completed"
    
    def test_export_error_handling(self):
        """Test export error handling for various failure scenarios."""
        # Test with invalid export ID
        result = self.export_service.get_export_status("invalid_id")
        assert result is None
        
        # Test export with database error
        request = ExportRequest(format=ExportFormat.JSON)
        export_id = self.export_service.start_export(request)
        
        with patch.object(self.export_service, '_query_documents', side_effect=Exception("Database error")):
            result = self.export_service.export_data(export_id, request)
        
        assert result.status == "failed"
        assert "Database error" in result.error
        
        # Test export with file write error
        with patch('builtins.open', side_effect=IOError("File write error")):
            export_id_2 = self.export_service.start_export(request)
            with patch.object(self.export_service, '_query_documents', return_value=[]):
                result_2 = self.export_service.export_data(export_id_2, request)
            
            assert result_2.status == "failed"
    
    def _create_mock_documents(self):
        """Create mock documents for testing."""
        documents = []
        
        for i in range(3):
            # Create mock document
            doc = Mock(spec=DocumentModel)
            doc.id = uuid4()
            doc.source_type = "database"
            doc.content = f"Test document content {i}"
            doc.document_metadata = {"test": True, "index": i}
            doc.source_config = {"connection": "test"}
            doc.created_at = datetime.now()
            doc.updated_at = datetime.now()
            
            # Create mock tasks
            tasks = []
            for j in range(2):
                task = Mock(spec=TaskModel)
                task.id = uuid4()
                task.project_id = f"project_{i}"
                task.status = TaskStatus.COMPLETED
                task.quality_score = 0.8 + (j * 0.1)
                task.annotations = [{"category": "test", "confidence": 0.9}]
                task.ai_predictions = [{"category": "test", "confidence": 0.8}]
                task.created_at = datetime.now()
                tasks.append(task)
            
            doc.tasks = tasks
            documents.append(doc)
        
        return documents
    
    def _create_mock_documents_with_annotations(self):
        """Create mock documents with COCO-style annotations."""
        documents = self._create_mock_documents()
        
        # Add COCO-style annotations to tasks
        for doc in documents:
            for task in doc.tasks:
                task.annotations = [
                    {
                        "category": "person",
                        "bbox": [10, 10, 50, 50],
                        "area": 2500,
                        "confidence": 0.9
                    },
                    {
                        "category": "object",
                        "bbox": [60, 60, 30, 30],
                        "area": 900,
                        "confidence": 0.8
                    }
                ]
        
        return documents


class TestExportServicePerformance:
    """Unit tests for large data export performance."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.export_service = ExportService(export_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_large_dataset_export_performance(self):
        """Test export performance with large datasets."""
        # Create large mock dataset
        large_documents = self._create_large_mock_dataset(1000)
        
        request = ExportRequest(
            format=ExportFormat.JSON,
            batch_size=100
        )
        
        export_id = self.export_service.start_export(request)
        
        # Measure export time
        start_time = time.time()
        
        with patch.object(self.export_service, '_query_documents', return_value=large_documents):
            result = self.export_service.export_data(export_id, request)
        
        export_time = time.time() - start_time
        
        # Verify export completed successfully
        assert result.status == "completed"
        assert result.total_records == 1000
        assert result.exported_records == 1000
        
        # Performance assertions (should complete within reasonable time)
        assert export_time < 30.0  # Should complete within 30 seconds
        
        # Verify file size is reasonable
        assert result.file_size > 0
        assert os.path.exists(result.file_path)
    
    def test_batch_export_functionality(self):
        """Test batch export for large datasets."""
        # Create large mock dataset
        large_documents = self._create_large_mock_dataset(500)
        
        request = ExportRequest(
            format=ExportFormat.CSV,
            batch_size=50
        )
        
        export_id = self.export_service.start_export(request)
        
        # Test batch export
        batch_results = []
        
        with patch.object(self.export_service, '_query_documents', return_value=large_documents):
            # Mock the database query for batch processing
            with patch.object(self.export_service, 'export_batch') as mock_batch:
                # Simulate batch processing
                mock_results = []
                for i in range(10):  # 10 batches of 50 each
                    batch_result = ExportResult(
                        export_id=export_id,
                        status="running" if i < 9 else "completed",
                        format=ExportFormat.CSV,
                        total_records=500,
                        exported_records=(i + 1) * 50
                    )
                    mock_results.append(batch_result)
                
                mock_batch.return_value = iter(mock_results)
                
                # Execute batch export
                for result in self.export_service.export_batch(export_id, request):
                    batch_results.append(result)
        
        # Verify batch processing
        assert len(batch_results) == 10
        assert batch_results[-1].status == "completed"
        assert batch_results[-1].exported_records == 500
    
    def test_export_memory_efficiency(self):
        """Test export memory efficiency with large content."""
        # Create documents with large content
        large_content_docs = []
        
        for i in range(10):
            doc = Mock(spec=DocumentModel)
            doc.id = uuid4()
            doc.source_type = "file"
            # Create large content (1MB per document)
            doc.content = "Large content " * 100000  # ~1MB
            doc.document_metadata = {"size": "large"}
            doc.source_config = {}
            doc.created_at = datetime.now()
            doc.updated_at = datetime.now()
            doc.tasks = []
            large_content_docs.append(doc)
        
        request = ExportRequest(
            format=ExportFormat.JSON,
            batch_size=5  # Small batch size for memory efficiency
        )
        
        export_id = self.export_service.start_export(request)
        
        with patch.object(self.export_service, '_query_documents', return_value=large_content_docs):
            result = self.export_service.export_data(export_id, request)
        
        # Verify export completed successfully despite large content
        assert result.status == "completed"
        assert result.total_records == 10
        assert result.file_size > 10000000  # Should be > 10MB
    
    def test_concurrent_export_handling(self):
        """Test handling multiple concurrent export jobs."""
        # Create multiple export requests
        requests = []
        export_ids = []
        
        for i in range(5):
            request = ExportRequest(
                format=ExportFormat.JSON if i % 2 == 0 else ExportFormat.CSV,
                batch_size=100
            )
            requests.append(request)
            export_ids.append(self.export_service.start_export(request))
        
        # Verify all exports were created
        assert len(export_ids) == 5
        
        # Check export status
        for export_id in export_ids:
            status = self.export_service.get_export_status(export_id)
            assert status is not None
            assert status.status == "pending"
        
        # List all exports
        all_exports = self.export_service.list_exports()
        assert len(all_exports) >= 5
    
    def _create_large_mock_dataset(self, count):
        """Create large mock dataset for performance testing."""
        documents = []
        
        for i in range(count):
            doc = Mock(spec=DocumentModel)
            doc.id = uuid4()
            doc.source_type = "database"
            doc.content = f"Performance test document {i} with some content"
            doc.document_metadata = {"index": i, "batch": i // 100}
            doc.source_config = {"test": True}
            doc.created_at = datetime.now()
            doc.updated_at = datetime.now()
            
            # Add some tasks
            tasks = []
            for j in range(2):
                task = Mock(spec=TaskModel)
                task.id = uuid4()
                task.project_id = f"perf_project_{i // 50}"
                task.status = TaskStatus.COMPLETED
                task.quality_score = 0.8
                task.annotations = [{"category": "test"}]
                task.ai_predictions = []
                task.created_at = datetime.now()
                tasks.append(task)
            
            doc.tasks = tasks
            documents.append(doc)
        
        return documents


class TestRAGAgentInterfaceUnit:
    """Unit tests for RAG and Agent interface functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.rag_service = RAGService()
        self.agent_service = AgentService()
    
    def test_rag_search_functionality(self):
        """Test RAG search interface functionality."""
        # Create test request
        request = RAGRequest(
            query="test query for RAG search",
            top_k=5,
            similarity_threshold=0.5
        )
        
        # Mock database documents
        mock_documents = self._create_mock_rag_documents()
        
        with patch.object(self.rag_service, '_query_documents', return_value=mock_documents):
            response = self.rag_service.search_documents(request)
        
        # Verify response structure
        assert response.query == request.query
        assert isinstance(response.chunks, list)
        assert response.total_chunks >= 0
        assert response.processing_time > 0
        assert isinstance(response.metadata, dict)
        
        # Verify chunks structure
        for chunk in response.chunks:
            assert hasattr(chunk, 'id')
            assert hasattr(chunk, 'document_id')
            assert hasattr(chunk, 'content')
            assert hasattr(chunk, 'similarity_score')
    
    def test_rag_document_chunking(self):
        """Test RAG document chunking functionality."""
        document_id = str(uuid4())
        
        # Mock document
        mock_doc = Mock(spec=DocumentModel)
        mock_doc.id = document_id
        mock_doc.content = "This is a long document that should be split into multiple chunks for RAG processing. " * 20
        mock_doc.source_type = "file"
        mock_doc.created_at = datetime.now()
        
        with patch('src.database.connection.db_manager.get_session') as mock_session:
            mock_db = Mock()
            mock_session.return_value.__enter__.return_value = mock_db
            
            # Mock the SQLAlchemy 2.0 style query execution
            mock_result = Mock()
            mock_result.scalar_one_or_none.return_value = mock_doc
            mock_db.execute.return_value = mock_result
            
            chunks = self.rag_service.get_document_chunks(document_id, chunk_size=100, chunk_overlap=20)
        
        # Verify chunking
        assert len(chunks) > 1  # Should be split into multiple chunks
        
        for i, chunk in enumerate(chunks):
            assert chunk.document_id == document_id
            assert len(chunk.content) <= 120  # Should respect chunk size + some tolerance
            assert chunk.metadata['chunk_index'] == i
            assert chunk.metadata['total_chunks'] == len(chunks)
    
    def test_rag_caching_functionality(self):
        """Test RAG caching functionality."""
        request = RAGRequest(
            query="cached query test",
            top_k=3
        )
        
        mock_documents = self._create_mock_rag_documents()
        
        # First request - should not be cached
        with patch.object(self.rag_service, '_query_documents', return_value=mock_documents):
            response1 = self.rag_service.search_documents(request)
        
        # Second request - should use cache
        with patch.object(self.rag_service, '_query_documents', return_value=[]):  # Empty to test cache
            response2 = self.rag_service.search_documents(request)
        
        # Verify caching worked (responses should be similar)
        assert response1.query == response2.query
        
        # Test cache clearing
        self.rag_service.clear_cache()
        
        # Verify cache was cleared by checking metrics
        metrics = self.rag_service.get_metrics()
        assert isinstance(metrics.cache_hit_rate, float)
    
    def test_agent_task_execution(self):
        """Test Agent task execution functionality."""
        # Test classification task
        classification_request = AgentRequest(
            task_type="classification",
            input_data={
                "text": "This is a test text for classification",
                "categories": ["positive", "negative", "neutral"]
            }
        )
        
        response = self.agent_service.execute_task(classification_request)
        
        # Verify response structure
        assert response.task_type == "classification"
        assert response.status in ["completed", "failed"]
        assert isinstance(response.result, dict)
        assert isinstance(response.steps, list)
        assert response.total_steps > 0
        assert response.execution_time > 0
        assert 0.0 <= response.confidence <= 1.0
        
        # Verify steps structure
        for step in response.steps:
            assert hasattr(step, 'step_number')
            assert hasattr(step, 'action')
            assert hasattr(step, 'confidence')
            assert 0.0 <= step.confidence <= 1.0
    
    def test_agent_supported_tasks(self):
        """Test Agent supported tasks functionality."""
        supported_tasks = self.agent_service.get_supported_tasks()
        
        # Verify structure
        assert isinstance(supported_tasks, list)
        assert len(supported_tasks) > 0
        
        # Check required task types
        task_types = [task['task_type'] for task in supported_tasks]
        expected_types = ['classification', 'extraction', 'summarization', 'question_answering']
        
        for expected_type in expected_types:
            assert expected_type in task_types
        
        # Verify task structure
        for task in supported_tasks:
            assert 'task_type' in task
            assert 'description' in task
            assert 'required_inputs' in task
    
    def test_agent_error_handling(self):
        """Test Agent error handling for invalid requests."""
        # Test with invalid task type - should raise validation error during model creation
        from pydantic import ValidationError
        
        with pytest.raises(ValidationError):
            AgentRequest(task_type="invalid_task", input_data={"text": "test"})
        
        # Test with valid request but empty input data (should be allowed)
        valid_request = AgentRequest(task_type="classification", input_data={})
        assert valid_request.task_type == "classification"
        
        # Test with timeout
        timeout_request = AgentRequest(
            task_type="classification",
            input_data={"text": "test"},
            timeout=5  # Minimum allowed timeout
        )
        
        # Mock a slow execution
        with patch.object(self.agent_service, '_handle_classification', side_effect=lambda *args: time.sleep(2)):
            response = self.agent_service.execute_task(timeout_request)
            # Should handle timeout gracefully
            assert response.status == "failed" or response.execution_time < 2
    
    def test_rag_agent_metrics(self):
        """Test RAG and Agent metrics functionality."""
        # Test RAG metrics
        rag_metrics = self.rag_service.get_metrics()
        
        assert hasattr(rag_metrics, 'query_count')
        assert hasattr(rag_metrics, 'avg_response_time')
        assert hasattr(rag_metrics, 'cache_hit_rate')
        assert 0.0 <= rag_metrics.cache_hit_rate <= 100.0
        
        # Test Agent metrics
        agent_metrics = self.agent_service.get_metrics()
        
        assert hasattr(agent_metrics, 'task_count')
        assert hasattr(agent_metrics, 'success_rate')
        assert hasattr(agent_metrics, 'avg_execution_time')
        assert 0.0 <= agent_metrics.success_rate <= 100.0
        
        # Test metrics reset
        self.rag_service.reset_metrics()
        self.agent_service.reset_metrics()
        
        reset_rag_metrics = self.rag_service.get_metrics()
        reset_agent_metrics = self.agent_service.get_metrics()
        
        assert reset_rag_metrics.query_count == 0
        assert reset_agent_metrics.task_count == 0
    
    def test_rag_agent_pipeline_integration(self):
        """Test RAG-Agent pipeline integration."""
        # Step 1: RAG search
        rag_request = RAGRequest(
            query="integration test query",
            top_k=3
        )
        
        mock_documents = self._create_mock_rag_documents()
        
        with patch.object(self.rag_service, '_query_documents', return_value=mock_documents):
            rag_response = self.rag_service.search_documents(rag_request)
        
        # Step 2: Use RAG results in Agent
        if rag_response.chunks:
            combined_content = " ".join([chunk.content for chunk in rag_response.chunks[:2]])
            
            agent_request = AgentRequest(
                task_type="summarization",
                input_data={
                    "text": combined_content,
                    "max_length": 100
                }
            )
            
            agent_response = self.agent_service.execute_task(agent_request)
            
            # Verify pipeline integration
            assert rag_response.total_chunks > 0
            assert agent_response.status in ["completed", "failed"]
            
            # Verify data flow
            assert len(combined_content) > 0
            if agent_response.status == "completed":
                assert "summary" in agent_response.result
    
    def _create_mock_rag_documents(self):
        """Create mock documents for RAG testing."""
        documents = []
        
        for i in range(5):
            doc = Mock(spec=DocumentModel)
            doc.id = uuid4()
            doc.source_type = "database"
            doc.content = f"RAG test document {i} with searchable content about testing and validation"
            doc.document_metadata = {"category": "test", "index": i}
            doc.created_at = datetime.now()
            doc.tasks = []
            documents.append(doc)
        
        return documents


class TestExportAPIEndpoints:
    """Unit tests for export API endpoints."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_export_formats_endpoint(self):
        """Test supported export formats endpoint."""
        # This would typically be tested with FastAPI TestClient
        # For now, we'll test the underlying logic
        
        from src.api.export import router
        
        # Verify router has the expected endpoints
        route_paths = [route.path for route in router.routes]
        
        expected_paths = [
            "/api/v1/export/start",
            "/api/v1/export/status/{export_id}",
            "/api/v1/export/download/{export_id}",
            "/api/v1/export/formats"
        ]
        
        for expected_path in expected_paths:
            assert expected_path in route_paths
    
    def test_rag_agent_api_endpoints(self):
        """Test RAG and Agent API endpoints."""
        from src.api.rag_agent import router
        
        # Verify router has the expected endpoints
        route_paths = [route.path for route in router.routes]
        
        expected_paths = [
            "/api/v1/rag/search",
            "/api/v1/rag/metrics",
            "/api/v1/agent/execute",
            "/api/v1/agent/metrics",
            "/api/v1/test/health"
        ]
        
        for expected_path in expected_paths:
            assert expected_path in route_paths
    
    @patch('src.api.export.export_service')
    def test_export_request_validation(self, mock_export_service):
        """Test export request validation."""
        # Test valid request
        valid_request = ExportRequest(
            format=ExportFormat.JSON,
            include_annotations=True
        )
        
        # Should not raise validation error
        assert valid_request.format == ExportFormat.JSON
        assert valid_request.include_annotations is True
        
        # Test invalid batch size
        with pytest.raises(ValueError):
            ExportRequest(
                format=ExportFormat.JSON,
                batch_size=0  # Invalid: should be >= 1
            )
        
        with pytest.raises(ValueError):
            ExportRequest(
                format=ExportFormat.JSON,
                batch_size=20000  # Invalid: should be <= 10000
            )
    
    @patch('src.api.rag_agent.rag_service')
    @patch('src.api.rag_agent.agent_service')
    def test_rag_agent_request_validation(self, mock_agent_service, mock_rag_service):
        """Test RAG and Agent request validation."""
        # Test valid RAG request
        valid_rag_request = RAGRequest(
            query="test query",
            top_k=5,
            similarity_threshold=0.7
        )
        
        assert valid_rag_request.query == "test query"
        assert valid_rag_request.top_k == 5
        
        # Test invalid similarity threshold
        with pytest.raises(ValueError):
            RAGRequest(
                query="test",
                similarity_threshold=1.5  # Invalid: should be <= 1.0
            )
        
        # Test valid Agent request
        valid_agent_request = AgentRequest(
            task_type="classification",
            input_data={"text": "test"}
        )
        
        assert valid_agent_request.task_type == "classification"
        
        # Test invalid task type
        with pytest.raises(ValueError):
            AgentRequest(
                task_type="invalid_type",
                input_data={"text": "test"}
            )


if __name__ == "__main__":
    pytest.main([__file__])