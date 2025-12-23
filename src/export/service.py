"""
Export service for SuperInsight Platform.

Provides data export functionality in multiple formats (JSON, CSV, COCO).
"""

import json
import csv
import os
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Iterator
from uuid import uuid4
from pathlib import Path
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, select, func

from .models import (
    ExportRequest, ExportResult, ExportFormat, ExportedDocument,
    COCODataset, COCOImage, COCOAnnotation, COCOCategory
)
from src.database.models import DocumentModel, TaskModel
from src.database.connection import db_manager
from src.database.batch_processor import BatchProcessor, BatchConfig, optimized_batch_processing
from src.database.pagination import pagination_service, PaginationStrategy

logger = logging.getLogger(__name__)


class ExportService:
    """Service for exporting annotation data in multiple formats."""
    
    def __init__(self, export_dir: str = "exports"):
        """Initialize export service."""
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(exist_ok=True)
        
        # In-memory storage for export jobs (in production, use Redis or database)
        self.export_jobs: Dict[str, ExportResult] = {}
    
    def start_export(self, request: ExportRequest) -> str:
        """Start an export job and return job ID."""
        export_id = str(uuid4())
        
        # Create export result
        result = ExportResult(
            export_id=export_id,
            status="pending",
            format=request.format,
            total_records=0,
            exported_records=0
        )
        
        self.export_jobs[export_id] = result
        logger.info(f"Started export job {export_id} with format {request.format}")
        
        return export_id
    
    def get_export_status(self, export_id: str) -> Optional[ExportResult]:
        """Get export job status."""
        return self.export_jobs.get(export_id)
    
    def list_exports(self) -> List[ExportResult]:
        """List all export jobs."""
        return list(self.export_jobs.values())
    
    def delete_export(self, export_id: str) -> bool:
        """Delete export job and associated files."""
        if export_id not in self.export_jobs:
            return False
        
        result = self.export_jobs[export_id]
        
        # Delete file if exists
        if result.file_path and os.path.exists(result.file_path):
            try:
                os.remove(result.file_path)
                logger.info(f"Deleted export file: {result.file_path}")
            except Exception as e:
                logger.error(f"Failed to delete export file: {e}")
        
        # Remove from jobs
        del self.export_jobs[export_id]
        return True
    
    def export_data_optimized(self, export_id: str, request: ExportRequest) -> ExportResult:
        """Perform optimized data export using batch processing."""
        if export_id not in self.export_jobs:
            raise ValueError(f"Export job {export_id} not found")
        
        result = self.export_jobs[export_id]
        
        try:
            result.status = "running"
            
            # Estimate dataset size for optimization
            with db_manager.get_session() as db:
                count_query = self._build_count_query(request)
                total_count = db.execute(count_query).scalar() or 0
                result.total_records = total_count
            
            # Use optimized batch processing for large datasets
            if total_count > 10000:
                return self._export_large_dataset(export_id, request, result)
            else:
                return self._export_small_dataset(export_id, request, result)
                
        except Exception as e:
            result.status = "failed"
            result.error = str(e)
            result.completed_at = datetime.now()
            logger.error(f"Optimized export {export_id} failed: {e}")
        
        return result
    
    def _export_large_dataset(self, export_id: str, request: ExportRequest, result: ExportResult) -> ExportResult:
        """Export large dataset using optimized batch processing."""
        
        def export_batch(documents: List[DocumentModel]) -> List[Dict[str, Any]]:
            """Process a batch of documents for export."""
            batch_data = []
            
            for doc in documents:
                doc_data = {
                    "id": str(doc.id),
                    "source_type": doc.source_type,
                    "content": doc.content,
                    "created_at": doc.created_at.isoformat(),
                    "updated_at": doc.updated_at.isoformat()
                }
                
                if request.include_metadata:
                    doc_data["metadata"] = doc.document_metadata or {}
                    doc_data["source_config"] = doc.source_config or {}
                
                # Add task information
                tasks_data = []
                for task in doc.tasks:
                    task_data = {
                        "id": str(task.id),
                        "project_id": task.project_id,
                        "status": task.status.value,
                        "quality_score": task.quality_score,
                        "created_at": task.created_at.isoformat()
                    }
                    
                    if request.include_annotations:
                        task_data["annotations"] = task.annotations or []
                    
                    if request.include_ai_predictions:
                        task_data["ai_predictions"] = task.ai_predictions or []
                    
                    tasks_data.append(task_data)
                
                doc_data["tasks"] = tasks_data
                batch_data.append(doc_data)
            
            return batch_data
        
        def progress_callback(stats):
            """Update export progress."""
            result.exported_records = stats.processed_items
            logger.info(f"Export progress: {stats.processed_items}/{stats.total_items} "
                       f"({stats.success_rate:.1f}%)")
        
        # Use optimized batch processing
        with optimized_batch_processing(result.total_records) as processor:
            query = self._build_export_query(request)
            
            stats = processor.process_query_batches(
                query, export_batch, progress_callback
            )
            
            # Write results to file
            file_path = self._write_export_file(export_id, request, stats)
            
            # Update result
            result.file_path = file_path
            result.file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
            result.exported_records = stats.processed_items
            result.status = "completed"
            result.completed_at = datetime.now()
            
            logger.info(f"Large dataset export {export_id} completed: "
                       f"{result.exported_records} records in {stats.processing_time:.2f}s")
        
        return result
    
    def _export_small_dataset(self, export_id: str, request: ExportRequest, result: ExportResult) -> ExportResult:
        """Export small dataset using traditional method."""
        with db_manager.get_session() as db:
            documents = self._query_documents(db, request)
            result.total_records = len(documents)
            
            # Export based on format
            if request.format == ExportFormat.JSON:
                file_path = self._export_json(export_id, documents, request)
            elif request.format == ExportFormat.CSV:
                file_path = self._export_csv(export_id, documents, request)
            elif request.format == ExportFormat.COCO:
                file_path = self._export_coco(export_id, documents, request)
            else:
                raise ValueError(f"Unsupported export format: {request.format}")
            
            # Update result
            result.file_path = file_path
            result.file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
            result.exported_records = result.total_records
            result.status = "completed"
            result.completed_at = datetime.now()
            
            logger.info(f"Small dataset export {export_id} completed: {result.exported_records} records")
        
        return result
    
    def _build_count_query(self, request: ExportRequest):
        """Build count query for export request."""
        stmt = select(func.count(DocumentModel.id))
        
        # Apply same filters as main query
        filters = []
        
        if request.document_ids:
            filters.append(DocumentModel.id.in_(request.document_ids))
        
        if request.project_id or request.task_ids:
            stmt = stmt.join(TaskModel)
            
            if request.project_id:
                filters.append(TaskModel.project_id == request.project_id)
            
            if request.task_ids:
                filters.append(TaskModel.id.in_(request.task_ids))
        
        if request.date_from:
            filters.append(DocumentModel.created_at >= request.date_from)
        
        if request.date_to:
            filters.append(DocumentModel.created_at <= request.date_to)
        
        if filters:
            stmt = stmt.where(and_(*filters))
        
        return stmt
    
    def _build_export_query(self, request: ExportRequest):
        """Build main export query."""
        stmt = select(DocumentModel)
        
        # Apply filters (same as _query_documents)
        filters = []
        
        if request.document_ids:
            filters.append(DocumentModel.id.in_(request.document_ids))
        
        if request.project_id or request.task_ids:
            stmt = stmt.join(TaskModel)
            
            if request.project_id:
                filters.append(TaskModel.project_id == request.project_id)
            
            if request.task_ids:
                filters.append(TaskModel.id.in_(request.task_ids))
        
        if request.date_from:
            filters.append(DocumentModel.created_at >= request.date_from)
        
        if request.date_to:
            filters.append(DocumentModel.created_at <= request.date_to)
        
        if filters:
            stmt = stmt.where(and_(*filters))
        
        return stmt
    
    def _write_export_file(self, export_id: str, request: ExportRequest, stats) -> str:
        """Write export results to file."""
        # This would need to be implemented based on the specific format
        # For now, return a placeholder path
        file_path = self.export_dir / f"{export_id}_optimized.{request.format.value}"
        
        # Write placeholder file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"Optimized export completed: {stats.processed_items} items processed")
        
        return str(file_path)
        """Perform the actual data export."""
        if export_id not in self.export_jobs:
            raise ValueError(f"Export job {export_id} not found")
        
        result = self.export_jobs[export_id]
        
        try:
            result.status = "running"
            
            # Query data from database
            with db_manager.get_session() as db:
                documents = self._query_documents(db, request)
                result.total_records = len(documents)
                
                # Export based on format
                if request.format == ExportFormat.JSON:
                    file_path = self._export_json(export_id, documents, request)
                elif request.format == ExportFormat.CSV:
                    file_path = self._export_csv(export_id, documents, request)
                elif request.format == ExportFormat.COCO:
                    file_path = self._export_coco(export_id, documents, request)
                else:
                    raise ValueError(f"Unsupported export format: {request.format}")
                
                # Update result
                result.file_path = file_path
                result.file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
                result.exported_records = result.total_records
                result.status = "completed"
                result.completed_at = datetime.now()
                
                logger.info(f"Export {export_id} completed: {result.exported_records} records")
                
        except Exception as e:
            result.status = "failed"
            result.error = str(e)
            result.completed_at = datetime.now()
            logger.error(f"Export {export_id} failed: {e}")
        
        return result
    
    def _query_documents(self, db: Session, request: ExportRequest) -> List[DocumentModel]:
        """Query documents from database based on request filters."""
        stmt = select(DocumentModel)
        
        # Apply filters
        filters = []
        
        if request.document_ids:
            filters.append(DocumentModel.id.in_(request.document_ids))
        
        if request.project_id or request.task_ids:
            # Join with tasks table for project/task filtering
            stmt = stmt.join(TaskModel)
            
            if request.project_id:
                filters.append(TaskModel.project_id == request.project_id)
            
            if request.task_ids:
                filters.append(TaskModel.id.in_(request.task_ids))
        
        if request.date_from:
            filters.append(DocumentModel.created_at >= request.date_from)
        
        if request.date_to:
            filters.append(DocumentModel.created_at <= request.date_to)
        
        if filters:
            stmt = stmt.where(and_(*filters))
        
        # Execute query with batching for large datasets
        stmt = stmt.limit(request.batch_size * 10)  # Reasonable limit
        documents = db.execute(stmt).scalars().all()
        
        return documents
    
    def _export_json(self, export_id: str, documents: List[DocumentModel], request: ExportRequest) -> str:
        """Export data in JSON format."""
        file_path = self.export_dir / f"{export_id}.json"
        
        exported_data = {
            "export_info": {
                "export_id": export_id,
                "format": "json",
                "created_at": datetime.now().isoformat(),
                "total_records": len(documents),
                "include_annotations": request.include_annotations,
                "include_ai_predictions": request.include_ai_predictions,
                "include_metadata": request.include_metadata
            },
            "documents": []
        }
        
        for doc in documents:
            doc_data = {
                "id": str(doc.id),
                "source_type": doc.source_type,
                "content": doc.content,
                "created_at": doc.created_at.isoformat(),
                "updated_at": doc.updated_at.isoformat()
            }
            
            if request.include_metadata:
                doc_data["metadata"] = doc.document_metadata or {}
                doc_data["source_config"] = doc.source_config or {}
            
            # Add task information
            tasks_data = []
            for task in doc.tasks:
                task_data = {
                    "id": str(task.id),
                    "project_id": task.project_id,
                    "status": task.status.value,
                    "quality_score": task.quality_score,
                    "created_at": task.created_at.isoformat()
                }
                
                if request.include_annotations:
                    task_data["annotations"] = task.annotations or []
                
                if request.include_ai_predictions:
                    task_data["ai_predictions"] = task.ai_predictions or []
                
                tasks_data.append(task_data)
            
            doc_data["tasks"] = tasks_data
            exported_data["documents"].append(doc_data)
        
        # Write to file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(exported_data, f, ensure_ascii=False, indent=2)
        
        return str(file_path)
    
    def _export_csv(self, export_id: str, documents: List[DocumentModel], request: ExportRequest) -> str:
        """Export data in CSV format."""
        file_path = self.export_dir / f"{export_id}.csv"
        
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header
            headers = [
                'document_id', 'source_type', 'content', 'created_at',
                'task_id', 'project_id', 'task_status', 'quality_score'
            ]
            
            if request.include_metadata:
                headers.extend(['document_metadata', 'source_config'])
            
            if request.include_annotations:
                headers.append('annotations')
            
            if request.include_ai_predictions:
                headers.append('ai_predictions')
            
            writer.writerow(headers)
            
            # Write data rows
            for doc in documents:
                base_row = [
                    str(doc.id),
                    doc.source_type,
                    doc.content,
                    doc.created_at.isoformat()
                ]
                
                if doc.tasks:
                    for task in doc.tasks:
                        row = base_row + [
                            str(task.id),
                            task.project_id,
                            task.status.value,
                            task.quality_score
                        ]
                        
                        if request.include_metadata:
                            row.extend([
                                json.dumps(doc.document_metadata or {}),
                                json.dumps(doc.source_config or {})
                            ])
                        
                        if request.include_annotations:
                            row.append(json.dumps(task.annotations or []))
                        
                        if request.include_ai_predictions:
                            row.append(json.dumps(task.ai_predictions or []))
                        
                        writer.writerow(row)
                else:
                    # Document without tasks
                    row = base_row + ['', '', '', '']
                    
                    if request.include_metadata:
                        row.extend([
                            json.dumps(doc.document_metadata or {}),
                            json.dumps(doc.source_config or {})
                        ])
                    
                    if request.include_annotations:
                        row.append('[]')
                    
                    if request.include_ai_predictions:
                        row.append('[]')
                    
                    writer.writerow(row)
        
        return str(file_path)
    
    def _export_coco(self, export_id: str, documents: List[DocumentModel], request: ExportRequest) -> str:
        """Export data in COCO format."""
        file_path = self.export_dir / f"{export_id}.json"
        
        # Create COCO dataset structure
        coco_dataset = COCODataset(
            info={
                "description": "SuperInsight Platform Export",
                "url": "https://superinsight.ai",
                "version": "1.0",
                "year": datetime.now().year,
                "contributor": "SuperInsight Platform",
                "date_created": datetime.now().isoformat()
            },
            licenses=[
                {
                    "id": 1,
                    "name": "Custom License",
                    "url": ""
                }
            ]
        )
        
        # Track categories and their IDs
        category_map = {}
        category_id_counter = 1
        annotation_id_counter = 1
        
        # Process documents and tasks
        for i, doc in enumerate(documents):
            # Create COCO image entry
            coco_image = COCOImage(
                id=i + 1,
                file_name=f"document_{doc.id}.txt",
                date_captured=doc.created_at.isoformat()
            )
            coco_dataset.images.append(coco_image)
            
            # Process annotations from tasks
            if request.include_annotations:
                for task in doc.tasks:
                    for annotation in (task.annotations or []):
                        # Extract category information
                        category_name = annotation.get('category', 'default')
                        
                        if category_name not in category_map:
                            category_map[category_name] = category_id_counter
                            coco_category = COCOCategory(
                                id=category_id_counter,
                                name=category_name,
                                supercategory="annotation"
                            )
                            coco_dataset.categories.append(coco_category)
                            category_id_counter += 1
                        
                        # Create COCO annotation
                        coco_annotation = COCOAnnotation(
                            id=annotation_id_counter,
                            image_id=i + 1,
                            category_id=category_map[category_name],
                            bbox=annotation.get('bbox', [0, 0, 100, 100]),
                            area=annotation.get('area', 10000),
                            iscrowd=0
                        )
                        
                        coco_dataset.annotations.append(coco_annotation)
                        annotation_id_counter += 1
        
        # Write COCO format to file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(coco_dataset.dict(), f, ensure_ascii=False, indent=2)
        
        return str(file_path)
    
    def export_batch(self, export_id: str, request: ExportRequest, 
                    batch_callback: Optional[callable] = None) -> Iterator[ExportResult]:
        """Export data in batches for large datasets."""
        if export_id not in self.export_jobs:
            raise ValueError(f"Export job {export_id} not found")
        
        result = self.export_jobs[export_id]
        
        try:
            result.status = "running"
            
            with db_manager.get_session() as db:
                # Get total count first
                total_stmt = select(DocumentModel)
                
                # Apply same filters as _query_documents
                filters = []
                
                if request.document_ids:
                    filters.append(DocumentModel.id.in_(request.document_ids))
                
                if request.project_id or request.task_ids:
                    total_stmt = total_stmt.join(TaskModel)
                    
                    if request.project_id:
                        filters.append(TaskModel.project_id == request.project_id)
                    
                    if request.task_ids:
                        filters.append(TaskModel.id.in_(request.task_ids))
                
                if request.date_from:
                    filters.append(DocumentModel.created_at >= request.date_from)
                
                if request.date_to:
                    filters.append(DocumentModel.created_at <= request.date_to)
                
                if filters:
                    total_stmt = total_stmt.where(and_(*filters))
                
                result.total_records = len(db.execute(total_stmt).scalars().all())
                
                # Process in batches
                offset = 0
                batch_num = 1
                
                while offset < result.total_records:
                    # Get batch of documents
                    batch_stmt = total_stmt.offset(offset).limit(request.batch_size)
                    documents = db.execute(batch_stmt).scalars().all()
                    
                    if not documents:
                        break
                    
                    # Export batch
                    batch_file_path = self._export_batch_file(
                        export_id, documents, request, batch_num
                    )
                    
                    # Update progress
                    result.exported_records = min(
                        offset + len(documents), 
                        result.total_records
                    )
                    
                    # Call callback if provided
                    if batch_callback:
                        batch_callback(result, batch_file_path)
                    
                    yield result
                    
                    offset += request.batch_size
                    batch_num += 1
                
                result.status = "completed"
                result.completed_at = datetime.now()
                
        except Exception as e:
            result.status = "failed"
            result.error = str(e)
            result.completed_at = datetime.now()
            logger.error(f"Batch export {export_id} failed: {e}")
        
        yield result
    
    def _export_batch_file(self, export_id: str, documents: List[DocumentModel], 
                          request: ExportRequest, batch_num: int) -> str:
        """Export a single batch to file."""
        file_path = self.export_dir / f"{export_id}_batch_{batch_num}.{request.format.value}"
        
        if request.format == ExportFormat.JSON:
            return self._export_json_batch(file_path, documents, request, batch_num)
        elif request.format == ExportFormat.CSV:
            return self._export_csv_batch(file_path, documents, request, batch_num)
        elif request.format == ExportFormat.COCO:
            return self._export_coco_batch(file_path, documents, request, batch_num)
        else:
            raise ValueError(f"Unsupported export format: {request.format}")
    
    def _export_json_batch(self, file_path: Path, documents: List[DocumentModel], 
                          request: ExportRequest, batch_num: int) -> str:
        """Export JSON batch file."""
        batch_data = {
            "batch_info": {
                "batch_number": batch_num,
                "records_count": len(documents),
                "created_at": datetime.now().isoformat()
            },
            "documents": []
        }
        
        for doc in documents:
            doc_data = {
                "id": str(doc.id),
                "source_type": doc.source_type,
                "content": doc.content,
                "created_at": doc.created_at.isoformat()
            }
            
            if request.include_metadata:
                doc_data["metadata"] = doc.document_metadata or {}
            
            # Add tasks
            tasks_data = []
            for task in doc.tasks:
                task_data = {
                    "id": str(task.id),
                    "project_id": task.project_id,
                    "status": task.status.value,
                    "quality_score": task.quality_score
                }
                
                if request.include_annotations:
                    task_data["annotations"] = task.annotations or []
                
                if request.include_ai_predictions:
                    task_data["ai_predictions"] = task.ai_predictions or []
                
                tasks_data.append(task_data)
            
            doc_data["tasks"] = tasks_data
            batch_data["documents"].append(doc_data)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(batch_data, f, ensure_ascii=False, indent=2)
        
        return str(file_path)
    
    def _export_csv_batch(self, file_path: Path, documents: List[DocumentModel], 
                         request: ExportRequest, batch_num: int) -> str:
        """Export CSV batch file."""
        # Similar to _export_csv but for batch
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header only for first batch
            if batch_num == 1:
                headers = [
                    'document_id', 'source_type', 'content', 'created_at',
                    'task_id', 'project_id', 'task_status', 'quality_score'
                ]
                
                if request.include_metadata:
                    headers.append('document_metadata')
                
                if request.include_annotations:
                    headers.append('annotations')
                
                if request.include_ai_predictions:
                    headers.append('ai_predictions')
                
                writer.writerow(headers)
            
            # Write data
            for doc in documents:
                base_row = [
                    str(doc.id),
                    doc.source_type,
                    doc.content,
                    doc.created_at.isoformat()
                ]
                
                if doc.tasks:
                    for task in doc.tasks:
                        row = base_row + [
                            str(task.id),
                            task.project_id,
                            task.status.value,
                            task.quality_score
                        ]
                        
                        if request.include_metadata:
                            row.append(json.dumps(doc.document_metadata or {}))
                        
                        if request.include_annotations:
                            row.append(json.dumps(task.annotations or []))
                        
                        if request.include_ai_predictions:
                            row.append(json.dumps(task.ai_predictions or []))
                        
                        writer.writerow(row)
                else:
                    row = base_row + ['', '', '', '']
                    
                    if request.include_metadata:
                        row.append(json.dumps(doc.document_metadata or {}))
                    
                    if request.include_annotations:
                        row.append('[]')
                    
                    if request.include_ai_predictions:
                        row.append('[]')
                    
                    writer.writerow(row)
        
        return str(file_path)
    
    def _export_coco_batch(self, file_path: Path, documents: List[DocumentModel], 
                          request: ExportRequest, batch_num: int) -> str:
        """Export COCO batch file."""
        # For COCO, each batch is a separate valid COCO dataset
        return self._export_coco(str(file_path).replace('.coco', ''), documents, request)
    
    def export_data(self, export_id: str, request: ExportRequest) -> ExportResult:
        """
        Main export method that delegates to the appropriate export strategy.
        
        This method provides a unified interface for data export and automatically
        chooses between optimized batch processing for large datasets and 
        traditional processing for smaller datasets.
        """
        return self.export_data_optimized(export_id, request)