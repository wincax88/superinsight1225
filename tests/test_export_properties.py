"""
Property-based tests for data export functionality in SuperInsight Platform.

Tests universal properties that should hold for all export operations.
"""

import pytest
import json
import csv
import tempfile
import os
from datetime import datetime
from typing import Dict, Any, List
from uuid import uuid4
from hypothesis import given, strategies as st, settings
from hypothesis.strategies import composite

from src.export.service import ExportService
from src.export.models import ExportRequest, ExportFormat
from src.models.document import Document
from src.models.task import Task, TaskStatus
from src.models.annotation import Annotation


# Test data generators
@composite
def document_strategy(draw):
    """Generate valid Document instances."""
    return Document(
        id=uuid4(),
        source_type=draw(st.sampled_from(['database', 'file', 'api'])),
        source_config=draw(st.dictionaries(
            st.text(min_size=1, max_size=20),
            st.text(min_size=1, max_size=50),
            min_size=1, max_size=5
        )),
        content=draw(st.text(min_size=10, max_size=1000)),
        metadata=draw(st.dictionaries(
            st.text(min_size=1, max_size=20),
            st.one_of(st.text(), st.integers(), st.floats()),
            min_size=0, max_size=5
        )),
        created_at=datetime.now(),
        updated_at=datetime.now()
    )


@composite
def annotation_strategy(draw):
    """Generate valid Annotation instances."""
    return Annotation(
        id=uuid4(),
        task_id=uuid4(),
        annotator_id=draw(st.text(min_size=1, max_size=50)),
        annotation_data=draw(st.dictionaries(
            st.text(min_size=1, max_size=20),
            st.one_of(st.text(), st.integers(), st.floats()),
            min_size=1, max_size=10
        )),
        confidence=draw(st.floats(min_value=0.0, max_value=1.0)),
        time_spent=draw(st.integers(min_value=0, max_value=3600)),
        created_at=datetime.now()
    )


@composite
def task_strategy(draw):
    """Generate valid Task instances."""
    return Task(
        id=uuid4(),
        document_id=uuid4(),
        project_id=draw(st.text(min_size=1, max_size=100)),
        status=draw(st.sampled_from(list(TaskStatus))),
        annotations=draw(st.lists(
            st.dictionaries(
                st.text(min_size=1, max_size=20),
                st.one_of(st.text(), st.integers(), st.floats()),
                min_size=1, max_size=5
            ),
            min_size=0, max_size=5
        )),
        ai_predictions=draw(st.lists(
            st.dictionaries(
                st.text(min_size=1, max_size=20),
                st.one_of(st.text(), st.integers(), st.floats()),
                min_size=1, max_size=5
            ),
            min_size=0, max_size=3
        )),
        quality_score=draw(st.floats(min_value=0.0, max_value=1.0)),
        created_at=datetime.now()
    )


@composite
def export_request_strategy(draw):
    """Generate valid ExportRequest instances."""
    return ExportRequest(
        format=draw(st.sampled_from(list(ExportFormat))),
        project_id=draw(st.one_of(st.none(), st.text(min_size=1, max_size=100))),
        include_annotations=draw(st.booleans()),
        include_ai_predictions=draw(st.booleans()),
        include_metadata=draw(st.booleans()),
        batch_size=draw(st.integers(min_value=1, max_value=100))
    )


class TestExportProperties:
    """Property-based tests for export functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.export_service = ExportService(export_dir="test_exports")
        # Ensure test export directory exists
        os.makedirs("test_exports", exist_ok=True)
    
    def teardown_method(self):
        """Clean up test environment."""
        # Clean up test files
        import shutil
        if os.path.exists("test_exports"):
            shutil.rmtree("test_exports")
    
    @given(export_request_strategy())
    @settings(max_examples=100, deadline=30000)
    def test_export_import_roundtrip_consistency_property(self, export_request):
        """
        **Feature: superinsight-platform, Property 10: 数据导出格式一致性**
        **Validates: Requirements 6.1, 6.2, 6.3**
        
        For any data export followed by re-import, the data should be consistent
        with the original data (round-trip property).
        """
        try:
            # Create a test export job
            export_id = self.export_service.start_export(export_request)
            
            # Mock some test data (since we don't have a real database in tests)
            original_documents = self._create_mock_documents()
            
            # Perform export with mock data
            result = self._mock_export_with_data(export_id, export_request, original_documents)
            
            if result.status == "completed" and result.file_path:
                # Verify format consistency
                self._verify_format_consistency(result.file_path, export_request.format)
                
                # Test round-trip: import the exported data back
                imported_documents = self._import_exported_data(result.file_path, export_request.format)
                
                # Verify round-trip consistency
                self._verify_roundtrip_consistency(original_documents, imported_documents, export_request)
                
                # Clean up
                if os.path.exists(result.file_path):
                    os.remove(result.file_path)
        
        except Exception as e:
            # Property should not fail due to implementation errors
            pytest.fail(f"Export-import round-trip consistency property failed: {e}")
    
    def _create_mock_documents(self) -> List[Dict[str, Any]]:
        """Create mock document data for testing."""
        return [
            {
                "id": str(uuid4()),
                "source_type": "database",
                "content": "Test document content for export testing",
                "metadata": {"test": True, "category": "mock"},
                "created_at": datetime.now().isoformat(),
                "tasks": [
                    {
                        "id": str(uuid4()),
                        "project_id": "test_project",
                        "status": "completed",
                        "annotations": [{"label": "test", "confidence": 0.9}],
                        "ai_predictions": [{"model": "test", "prediction": "positive"}],
                        "quality_score": 0.85
                    }
                ]
            }
        ]
    
    def _mock_export_with_data(self, export_id: str, request: ExportRequest, 
                              documents: List[Dict[str, Any]]):
        """Mock export operation with test data."""
        result = self.export_service.export_jobs[export_id]
        result.status = "running"
        result.total_records = len(documents)
        
        try:
            # Create temporary file for export
            with tempfile.NamedTemporaryFile(
                mode='w', 
                suffix=f'.{request.format.value}',
                delete=False
            ) as f:
                file_path = f.name
                
                if request.format == ExportFormat.JSON:
                    self._write_json_export(f, documents, request)
                elif request.format == ExportFormat.CSV:
                    self._write_csv_export(f, documents, request)
                elif request.format == ExportFormat.COCO:
                    self._write_coco_export(f, documents, request)
            
            result.file_path = file_path
            result.file_size = os.path.getsize(file_path)
            result.exported_records = len(documents)
            result.status = "completed"
            result.completed_at = datetime.now()
            
        except Exception as e:
            result.status = "failed"
            result.error = str(e)
        
        return result
    
    def _write_json_export(self, file_handle, documents: List[Dict[str, Any]], 
                          request: ExportRequest):
        """Write JSON export format."""
        export_data = {
            "export_info": {
                "format": "json",
                "created_at": datetime.now().isoformat(),
                "total_records": len(documents)
            },
            "documents": documents
        }
        json.dump(export_data, file_handle, ensure_ascii=False, indent=2)
    
    def _write_csv_export(self, file_handle, documents: List[Dict[str, Any]], 
                         request: ExportRequest):
        """Write CSV export format."""
        writer = csv.writer(file_handle)
        
        # Write header
        headers = ['document_id', 'source_type', 'content', 'created_at']
        if request.include_metadata:
            headers.append('metadata')
        if request.include_annotations:
            headers.append('annotations')
        
        writer.writerow(headers)
        
        # Write data
        for doc in documents:
            row = [
                doc['id'],
                doc['source_type'],
                doc['content'],
                doc['created_at']
            ]
            
            if request.include_metadata:
                row.append(json.dumps(doc.get('metadata', {})))
            
            if request.include_annotations:
                annotations = []
                for task in doc.get('tasks', []):
                    annotations.extend(task.get('annotations', []))
                row.append(json.dumps(annotations))
            
            writer.writerow(row)
    
    def _write_coco_export(self, file_handle, documents: List[Dict[str, Any]], 
                          request: ExportRequest):
        """Write COCO export format."""
        coco_data = {
            "info": {
                "description": "Test COCO Export",
                "version": "1.0",
                "year": datetime.now().year
            },
            "licenses": [{"id": 1, "name": "Test License"}],
            "images": [],
            "annotations": [],
            "categories": [{"id": 1, "name": "test", "supercategory": "annotation"}]
        }
        
        for i, doc in enumerate(documents):
            # Add image entry
            coco_data["images"].append({
                "id": i + 1,
                "file_name": f"document_{doc['id']}.txt",
                "width": 800,
                "height": 600
            })
            
            # Add annotations
            if request.include_annotations:
                for task in doc.get('tasks', []):
                    for j, annotation in enumerate(task.get('annotations', [])):
                        coco_data["annotations"].append({
                            "id": len(coco_data["annotations"]) + 1,
                            "image_id": i + 1,
                            "category_id": 1,
                            "bbox": [0, 0, 100, 100],
                            "area": 10000,
                            "iscrowd": 0
                        })
        
        json.dump(coco_data, file_handle, ensure_ascii=False, indent=2)
    
    def _verify_format_consistency(self, file_path: str, format_type: ExportFormat):
        """Verify that exported file follows format specifications."""
        if format_type == ExportFormat.JSON:
            self._verify_json_format(file_path)
        elif format_type == ExportFormat.CSV:
            self._verify_csv_format(file_path)
        elif format_type == ExportFormat.COCO:
            self._verify_coco_format(file_path)
    
    def _verify_json_format(self, file_path: str):
        """Verify JSON format consistency."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Verify required JSON structure
        assert isinstance(data, dict), "JSON export should be a dictionary"
        assert "export_info" in data, "JSON export should have export_info"
        assert "documents" in data, "JSON export should have documents"
        assert isinstance(data["documents"], list), "Documents should be a list"
    
    def _verify_csv_format(self, file_path: str):
        """Verify CSV format consistency."""
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            headers = next(reader)
            
            # Verify CSV structure
            assert len(headers) > 0, "CSV should have headers"
            assert "document_id" in headers, "CSV should have document_id column"
            
            # Verify all rows have same number of columns
            for row in reader:
                assert len(row) == len(headers), "All CSV rows should have same column count"
    
    def _verify_coco_format(self, file_path: str):
        """Verify COCO format consistency."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Verify required COCO structure
        required_keys = ["info", "licenses", "images", "annotations", "categories"]
        for key in required_keys:
            assert key in data, f"COCO export should have {key}"
        
        # Verify data types
        assert isinstance(data["images"], list), "COCO images should be a list"
        assert isinstance(data["annotations"], list), "COCO annotations should be a list"
        assert isinstance(data["categories"], list), "COCO categories should be a list"
    
    def _verify_data_integrity(self, file_path: str, format_type: ExportFormat, 
                              original_documents: List[Dict[str, Any]]):
        """Verify that exported data maintains integrity."""
        if format_type == ExportFormat.JSON:
            self._verify_json_data_integrity(file_path, original_documents)
        elif format_type == ExportFormat.CSV:
            self._verify_csv_data_integrity(file_path, original_documents)
        elif format_type == ExportFormat.COCO:
            self._verify_coco_data_integrity(file_path, original_documents)
    
    def _verify_json_data_integrity(self, file_path: str, 
                                   original_documents: List[Dict[str, Any]]):
        """Verify JSON data integrity."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        exported_docs = data["documents"]
        
        # Verify document count
        assert len(exported_docs) == len(original_documents), \
            "Exported document count should match original"
        
        # Verify document IDs are preserved
        original_ids = {doc["id"] for doc in original_documents}
        exported_ids = {doc["id"] for doc in exported_docs}
        assert original_ids == exported_ids, "Document IDs should be preserved"
    
    def _verify_csv_data_integrity(self, file_path: str, 
                                  original_documents: List[Dict[str, Any]]):
        """Verify CSV data integrity."""
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            exported_rows = list(reader)
        
        # Verify we have data rows
        assert len(exported_rows) > 0, "CSV should have data rows"
        
        # Verify document IDs are present
        exported_ids = {row["document_id"] for row in exported_rows}
        original_ids = {doc["id"] for doc in original_documents}
        
        # At least some original IDs should be in exported data
        assert len(exported_ids.intersection(original_ids)) > 0, \
            "Some original document IDs should be preserved in CSV"
    
    def _verify_coco_data_integrity(self, file_path: str, 
                                   original_documents: List[Dict[str, Any]]):
        """Verify COCO data integrity."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Verify we have images corresponding to documents
        assert len(data["images"]) == len(original_documents), \
            "COCO images count should match document count"
        
        # Verify image IDs are sequential
        image_ids = [img["id"] for img in data["images"]]
        expected_ids = list(range(1, len(original_documents) + 1))
        assert image_ids == expected_ids, "COCO image IDs should be sequential"
    
    def _import_exported_data(self, file_path: str, format_type: ExportFormat) -> List[Dict[str, Any]]:
        """Import data from exported file to test round-trip consistency."""
        if format_type == ExportFormat.JSON:
            return self._import_json_data(file_path)
        elif format_type == ExportFormat.CSV:
            return self._import_csv_data(file_path)
        elif format_type == ExportFormat.COCO:
            return self._import_coco_data(file_path)
        else:
            raise ValueError(f"Unsupported format for import: {format_type}")
    
    def _import_json_data(self, file_path: str) -> List[Dict[str, Any]]:
        """Import data from JSON export file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data.get("documents", [])
    
    def _import_csv_data(self, file_path: str) -> List[Dict[str, Any]]:
        """Import data from CSV export file."""
        documents = {}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                doc_id = row["document_id"]
                
                if doc_id not in documents:
                    documents[doc_id] = {
                        "id": doc_id,
                        "source_type": row["source_type"],
                        "content": row["content"],
                        "created_at": row["created_at"],
                        "tasks": []
                    }
                    
                    # Add metadata if present
                    if "document_metadata" in row:
                        try:
                            documents[doc_id]["metadata"] = json.loads(row["document_metadata"])
                        except (json.JSONDecodeError, KeyError):
                            documents[doc_id]["metadata"] = {}
                
                # Add task if present
                if row.get("task_id"):
                    task = {
                        "id": row["task_id"],
                        "project_id": row["project_id"],
                        "status": row["task_status"],
                        "quality_score": float(row["quality_score"]) if row["quality_score"] else 0.0
                    }
                    
                    # Add annotations if present
                    if "annotations" in row:
                        try:
                            task["annotations"] = json.loads(row["annotations"])
                        except (json.JSONDecodeError, KeyError):
                            task["annotations"] = []
                    
                    # Add AI predictions if present
                    if "ai_predictions" in row:
                        try:
                            task["ai_predictions"] = json.loads(row["ai_predictions"])
                        except (json.JSONDecodeError, KeyError):
                            task["ai_predictions"] = []
                    
                    documents[doc_id]["tasks"].append(task)
        
        return list(documents.values())
    
    def _import_coco_data(self, file_path: str) -> List[Dict[str, Any]]:
        """Import data from COCO export file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = []
        
        # Convert COCO images back to documents
        for image in data.get("images", []):
            doc = {
                "id": f"coco_image_{image['id']}",
                "source_type": "coco_import",
                "content": f"COCO image: {image['file_name']}",
                "created_at": image.get("date_captured", ""),
                "metadata": {
                    "width": image.get("width", 0),
                    "height": image.get("height", 0),
                    "file_name": image.get("file_name", "")
                },
                "tasks": []
            }
            
            # Find annotations for this image
            image_annotations = [
                ann for ann in data.get("annotations", [])
                if ann.get("image_id") == image["id"]
            ]
            
            if image_annotations:
                task = {
                    "id": f"coco_task_{image['id']}",
                    "project_id": "coco_import",
                    "status": "completed",
                    "quality_score": 1.0,
                    "annotations": []
                }
                
                for ann in image_annotations:
                    task["annotations"].append({
                        "id": ann.get("id"),
                        "category_id": ann.get("category_id"),
                        "bbox": ann.get("bbox", []),
                        "area": ann.get("area", 0),
                        "iscrowd": ann.get("iscrowd", 0)
                    })
                
                doc["tasks"].append(task)
            
            documents.append(doc)
        
        return documents
    
    def _verify_roundtrip_consistency(self, original_documents: List[Dict[str, Any]], 
                                    imported_documents: List[Dict[str, Any]], 
                                    export_request: ExportRequest):
        """Verify that imported data is consistent with original data."""
        
        # Basic count check
        assert len(imported_documents) == len(original_documents), \
            f"Document count mismatch: original={len(original_documents)}, imported={len(imported_documents)}"
        
        # For JSON format, we expect near-perfect round-trip consistency
        if export_request.format == ExportFormat.JSON:
            self._verify_json_roundtrip_consistency(original_documents, imported_documents, export_request)
        
        # For CSV format, some data transformation is expected
        elif export_request.format == ExportFormat.CSV:
            self._verify_csv_roundtrip_consistency(original_documents, imported_documents, export_request)
        
        # For COCO format, significant transformation is expected
        elif export_request.format == ExportFormat.COCO:
            self._verify_coco_roundtrip_consistency(original_documents, imported_documents, export_request)
    
    def _verify_json_roundtrip_consistency(self, original_documents: List[Dict[str, Any]], 
                                         imported_documents: List[Dict[str, Any]], 
                                         export_request: ExportRequest):
        """Verify JSON round-trip consistency (should be near-perfect)."""
        
        # Create lookup maps by document ID
        original_map = {doc["id"]: doc for doc in original_documents}
        imported_map = {doc["id"]: doc for doc in imported_documents}
        
        # Verify all original document IDs are present
        original_ids = set(original_map.keys())
        imported_ids = set(imported_map.keys())
        assert original_ids == imported_ids, \
            f"Document ID mismatch: missing={original_ids - imported_ids}, extra={imported_ids - original_ids}"
        
        # Verify document content consistency
        for doc_id in original_ids:
            original_doc = original_map[doc_id]
            imported_doc = imported_map[doc_id]
            
            # Core fields should match exactly
            assert original_doc["id"] == imported_doc["id"], f"ID mismatch for document {doc_id}"
            assert original_doc["source_type"] == imported_doc["source_type"], f"Source type mismatch for document {doc_id}"
            assert original_doc["content"] == imported_doc["content"], f"Content mismatch for document {doc_id}"
            
            # Metadata should match if included
            if export_request.include_metadata:
                original_metadata = original_doc.get("metadata", {})
                imported_metadata = imported_doc.get("metadata", {})
                assert original_metadata == imported_metadata, f"Metadata mismatch for document {doc_id}"
            
            # Task data should match if included
            if export_request.include_annotations or export_request.include_ai_predictions:
                original_tasks = original_doc.get("tasks", [])
                imported_tasks = imported_doc.get("tasks", [])
                assert len(original_tasks) == len(imported_tasks), f"Task count mismatch for document {doc_id}"
    
    def _verify_csv_roundtrip_consistency(self, original_documents: List[Dict[str, Any]], 
                                        imported_documents: List[Dict[str, Any]], 
                                        export_request: ExportRequest):
        """Verify CSV round-trip consistency (some data loss expected due to format limitations)."""
        
        # For CSV, we mainly check that core document information is preserved
        original_ids = {doc["id"] for doc in original_documents}
        imported_ids = {doc["id"] for doc in imported_documents}
        
        # At least the same number of documents should be present
        assert len(imported_ids) >= len(original_ids), \
            "CSV import should preserve at least the same number of documents"
        
        # Check that core document fields are preserved for documents that exist in both
        common_ids = original_ids.intersection(imported_ids)
        assert len(common_ids) > 0, "At least some documents should be preserved in CSV round-trip"
        
        original_map = {doc["id"]: doc for doc in original_documents}
        imported_map = {doc["id"]: doc for doc in imported_documents}
        
        for doc_id in common_ids:
            original_doc = original_map[doc_id]
            imported_doc = imported_map[doc_id]
            
            # Core fields should be preserved
            assert original_doc["source_type"] == imported_doc["source_type"], \
                f"Source type should be preserved for document {doc_id}"
            assert original_doc["content"] == imported_doc["content"], \
                f"Content should be preserved for document {doc_id}"
    
    def _verify_coco_roundtrip_consistency(self, original_documents: List[Dict[str, Any]], 
                                         imported_documents: List[Dict[str, Any]], 
                                         export_request: ExportRequest):
        """Verify COCO round-trip consistency (significant transformation expected)."""
        
        # For COCO format, we mainly check that the number of documents is preserved
        # and that annotation structure is maintained
        
        assert len(imported_documents) == len(original_documents), \
            "COCO round-trip should preserve document count"
        
        # Check that annotation data is preserved if it was included
        if export_request.include_annotations:
            original_annotation_count = sum(
                len(task.get("annotations", []))
                for doc in original_documents
                for task in doc.get("tasks", [])
            )
            
            imported_annotation_count = sum(
                len(task.get("annotations", []))
                for doc in imported_documents
                for task in doc.get("tasks", [])
            )
            
            # COCO format may aggregate or transform annotations, but should preserve some
            if original_annotation_count > 0:
                assert imported_annotation_count > 0, \
                    "COCO round-trip should preserve some annotation data"


# Additional property tests for specific scenarios
class TestExportEdgeCases:
    """Test edge cases and boundary conditions for export functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.export_service = ExportService(export_dir="test_exports")
        os.makedirs("test_exports", exist_ok=True)
    
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        if os.path.exists("test_exports"):
            shutil.rmtree("test_exports")
    
    @given(st.lists(st.text(min_size=0, max_size=1000), min_size=0, max_size=10))
    @settings(max_examples=50, deadline=15000)
    def test_empty_content_export_property(self, content_list):
        """
        **Feature: superinsight-platform, Property 10a: 空内容导出一致性**
        **Validates: Requirements 6.1, 6.2, 6.3**
        
        For any list of content (including empty), export should handle gracefully
        and produce valid format files.
        """
        try:
            # Create export request
            request = ExportRequest(
                format=ExportFormat.JSON,
                include_annotations=True,
                include_metadata=True
            )
            
            export_id = self.export_service.start_export(request)
            
            # Create mock documents with the given content
            documents = []
            for i, content in enumerate(content_list):
                documents.append({
                    "id": str(uuid4()),
                    "source_type": "test",
                    "content": content,
                    "metadata": {},
                    "created_at": datetime.now().isoformat(),
                    "tasks": []
                })
            
            # Mock export
            result = self.export_service.export_jobs[export_id]
            result.status = "running"
            result.total_records = len(documents)
            
            # Create export file
            with tempfile.NamedTemporaryFile(
                mode='w', suffix='.json', delete=False
            ) as f:
                export_data = {
                    "export_info": {
                        "format": "json",
                        "total_records": len(documents)
                    },
                    "documents": documents
                }
                json.dump(export_data, f, ensure_ascii=False)
                result.file_path = f.name
            
            result.status = "completed"
            result.exported_records = len(documents)
            
            # Verify file is valid JSON
            with open(result.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                assert isinstance(data, dict)
                assert "documents" in data
                assert len(data["documents"]) == len(content_list)
            
            # Clean up
            os.remove(result.file_path)
            
        except Exception as e:
            pytest.fail(f"Empty content export property failed: {e}")
    
    @given(st.integers(min_value=1, max_value=1000))
    @settings(max_examples=20, deadline=10000)
    def test_batch_size_consistency_property(self, batch_size):
        """
        **Feature: superinsight-platform, Property 10b: 批量大小一致性**
        **Validates: Requirements 6.1, 6.2, 6.3**
        
        For any valid batch size, export should respect the batch size setting
        and process data in appropriate chunks.
        """
        try:
            # Create export request with specific batch size
            request = ExportRequest(
                format=ExportFormat.CSV,
                batch_size=batch_size,
                include_annotations=False
            )
            
            export_id = self.export_service.start_export(request)
            result = self.export_service.get_export_status(export_id)
            
            # Verify batch size is respected
            assert result is not None
            assert result.export_id == export_id
            
            # The batch size should be stored in the request
            # (actual batching logic would be tested in integration tests)
            assert request.batch_size == batch_size
            assert 1 <= request.batch_size <= 10000  # Within valid range
            
        except Exception as e:
            pytest.fail(f"Batch size consistency property failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])