"""
Label Studio Integration Module

Provides comprehensive integration with Label Studio for project management,
task import/export, webhook configuration, and PostgreSQL synchronization.
"""

import logging
import json
import asyncio
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from uuid import UUID
import httpx
from sqlalchemy.orm import Session
from sqlalchemy import select

from src.label_studio.config import LabelStudioConfig, LabelStudioProject
from src.database.connection import get_db_session
from src.database.models import TaskModel, DocumentModel
from src.models.task import Task, TaskStatus
from src.models.document import Document
from src.config.settings import settings

logger = logging.getLogger(__name__)


class LabelStudioIntegrationError(Exception):
    """Custom exception for Label Studio integration errors"""
    pass


class ProjectConfig:
    """Project configuration for Label Studio"""
    
    def __init__(self, 
                 title: str,
                 description: str = "",
                 annotation_type: str = "text_classification",
                 label_config: Optional[str] = None):
        self.title = title
        self.description = description
        self.annotation_type = annotation_type
        self.label_config = label_config


class ImportResult:
    """Result of task import operation"""
    
    def __init__(self, 
                 success: bool,
                 imported_count: int = 0,
                 failed_count: int = 0,
                 errors: List[str] = None):
        self.success = success
        self.imported_count = imported_count
        self.failed_count = failed_count
        self.errors = errors or []


class ExportResult:
    """Result of annotation export operation"""
    
    def __init__(self,
                 success: bool,
                 exported_count: int = 0,
                 data: List[Dict[str, Any]] = None,
                 errors: List[str] = None):
        self.success = success
        self.exported_count = exported_count
        self.data = data or []
        self.errors = errors or []


class LabelStudioIntegration:
    """
    Main integration class for Label Studio operations.
    
    Handles project creation, task management, webhook configuration,
    and data synchronization with PostgreSQL.
    """
    
    def __init__(self, config: Optional[LabelStudioConfig] = None):
        """Initialize Label Studio integration with configuration"""
        self.config = config or LabelStudioConfig()
        self.base_url = self.config.base_url.rstrip('/')
        self.api_token = self.config.api_token
        self.headers = {
            'Authorization': f'Token {self.api_token}' if self.api_token else '',
            'Content-Type': 'application/json'
        }
        
        # Validate configuration
        if not self.config.validate_config():
            raise LabelStudioIntegrationError("Invalid Label Studio configuration")
    
    async def create_project(self, project_config: ProjectConfig) -> LabelStudioProject:
        """
        Create a new Label Studio project.
        
        Args:
            project_config: Project configuration settings
            
        Returns:
            LabelStudioProject: Created project information
            
        Raises:
            LabelStudioIntegrationError: If project creation fails
        """
        try:
            # Prepare project data
            label_config = (project_config.label_config or 
                          self.config.get_default_label_config(project_config.annotation_type))
            
            project_data = {
                "title": project_config.title,
                "description": project_config.description,
                "label_config": label_config,
                "expert_instruction": "请根据标注指南进行数据标注。",
                "show_instruction": True,
                "show_skip_button": True,
                "enable_empty_annotation": False,
                "show_annotation_history": True,
                "color": "#1f77b4",
                "maximum_annotations": 1,
                "is_published": False,
                "is_draft": False,
                "sampling": "Sequential sampling",
                "show_collab_predictions": True,
                "reveal_preannotations_interactively": True
            }
            
            # Make API request
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/projects/",
                    headers=self.headers,
                    json=project_data
                )
                
                if response.status_code == 201:
                    project_info = response.json()
                    logger.info(f"Created Label Studio project: {project_info['id']}")
                    
                    # Convert to LabelStudioProject - avoid duplicate parameters
                    project_kwargs = {k: v for k, v in project_info.items() 
                                    if k in LabelStudioProject.__dataclass_fields__}
                    
                    return LabelStudioProject(**project_kwargs)
                else:
                    error_msg = f"Failed to create project: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    raise LabelStudioIntegrationError(error_msg)
                    
        except httpx.RequestError as e:
            error_msg = f"Network error creating project: {str(e)}"
            logger.error(error_msg)
            raise LabelStudioIntegrationError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error creating project: {str(e)}"
            logger.error(error_msg)
            raise LabelStudioIntegrationError(error_msg)
    
    async def import_tasks(self, project_id: str, tasks: List[Task]) -> ImportResult:
        """
        Import tasks into a Label Studio project.
        
        Args:
            project_id: Label Studio project ID
            tasks: List of tasks to import
            
        Returns:
            ImportResult: Import operation results
        """
        try:
            imported_count = 0
            failed_count = 0
            errors = []
            
            # Convert tasks to Label Studio format
            ls_tasks = []
            for task in tasks:
                try:
                    # Get document content
                    with get_db_session() as db:
                        stmt = select(DocumentModel).where(
                            DocumentModel.id == task.document_id
                        )
                        doc_model = db.execute(stmt).scalar_one_or_none()
                        
                        if not doc_model:
                            errors.append(f"Document not found for task {task.id}")
                            failed_count += 1
                            continue
                    
                    # Prepare task data for Label Studio
                    ls_task = {
                        "data": {
                            "text": doc_model.content,
                            "document_id": str(task.document_id),
                            "task_id": str(task.id)
                        },
                        "meta": {
                            "superinsight_task_id": str(task.id),
                            "document_id": str(task.document_id),
                            "created_at": task.created_at.isoformat()
                        }
                    }
                    
                    # Add AI predictions if available
                    if task.ai_predictions:
                        ls_task["predictions"] = []
                        for pred in task.ai_predictions:
                            ls_task["predictions"].append({
                                "model_version": pred.get("model", "unknown"),
                                "result": pred.get("result", []),
                                "score": pred.get("confidence", 0.0)
                            })
                    
                    ls_tasks.append(ls_task)
                    
                except Exception as e:
                    errors.append(f"Error processing task {task.id}: {str(e)}")
                    failed_count += 1
            
            # Import tasks in batches
            batch_size = 100
            for i in range(0, len(ls_tasks), batch_size):
                batch = ls_tasks[i:i + batch_size]
                
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(
                        f"{self.base_url}/api/projects/{project_id}/import",
                        headers=self.headers,
                        json=batch
                    )
                    
                    if response.status_code == 201:
                        result = response.json()
                        imported_count += len(batch)
                        logger.info(f"Imported batch of {len(batch)} tasks to project {project_id}")
                    else:
                        error_msg = f"Failed to import batch: {response.status_code} - {response.text}"
                        errors.append(error_msg)
                        failed_count += len(batch)
                        logger.error(error_msg)
            
            # Update task status in database
            await self._sync_tasks_to_db(project_id, tasks)
            
            return ImportResult(
                success=failed_count == 0,
                imported_count=imported_count,
                failed_count=failed_count,
                errors=errors
            )
            
        except Exception as e:
            error_msg = f"Error importing tasks: {str(e)}"
            logger.error(error_msg)
            return ImportResult(
                success=False,
                failed_count=len(tasks),
                errors=[error_msg]
            )
    
    async def export_annotations(self, project_id: str, export_format: str = "JSON") -> ExportResult:
        """
        Export annotations from a Label Studio project.
        
        Args:
            project_id: Label Studio project ID
            export_format: Export format (JSON, CSV, etc.)
            
        Returns:
            ExportResult: Export operation results
        """
        try:
            # Get annotations from Label Studio
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.get(
                    f"{self.base_url}/api/projects/{project_id}/export",
                    headers=self.headers,
                    params={"exportType": export_format}
                )
                
                if response.status_code == 200:
                    annotations_data = response.json()
                    
                    # Sync annotations back to PostgreSQL
                    await self._sync_annotations_to_db(project_id, annotations_data)
                    
                    logger.info(f"Exported {len(annotations_data)} annotations from project {project_id}")
                    
                    return ExportResult(
                        success=True,
                        exported_count=len(annotations_data),
                        data=annotations_data
                    )
                else:
                    error_msg = f"Failed to export annotations: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    return ExportResult(
                        success=False,
                        errors=[error_msg]
                    )
                    
        except Exception as e:
            error_msg = f"Error exporting annotations: {str(e)}"
            logger.error(error_msg)
            return ExportResult(
                success=False,
                errors=[error_msg]
            )
    
    async def setup_webhooks(self, project_id: str, webhook_urls: List[str]) -> bool:
        """
        Configure webhooks for quality check triggers.
        
        Args:
            project_id: Label Studio project ID
            webhook_urls: List of webhook URLs to configure
            
        Returns:
            bool: True if webhooks were configured successfully
        """
        try:
            for webhook_url in webhook_urls:
                webhook_config = self.config.get_webhook_config(webhook_url)
                
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        f"{self.base_url}/api/projects/{project_id}/webhooks/",
                        headers=self.headers,
                        json=webhook_config
                    )
                    
                    if response.status_code == 201:
                        logger.info(f"Configured webhook for project {project_id}: {webhook_url}")
                    else:
                        logger.error(f"Failed to configure webhook: {response.status_code} - {response.text}")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error configuring webhooks: {str(e)}")
            return False
    
    async def configure_ml_backend(self, project_id: str, ml_backend_url: str) -> bool:
        """
        Configure ML backend for AI predictions.
        
        Args:
            project_id: Label Studio project ID
            ml_backend_url: ML backend service URL
            
        Returns:
            bool: True if ML backend was configured successfully
        """
        try:
            ml_config = self.config.get_ml_backend_config(ml_backend_url)
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/ml/",
                    headers=self.headers,
                    json={**ml_config, "project": project_id}
                )
                
                if response.status_code == 201:
                    logger.info(f"Configured ML backend for project {project_id}: {ml_backend_url}")
                    return True
                else:
                    logger.error(f"Failed to configure ML backend: {response.status_code} - {response.text}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error configuring ML backend: {str(e)}")
            return False
    
    async def _sync_tasks_to_db(self, project_id: str, tasks: List[Task]) -> None:
        """Synchronize tasks to PostgreSQL database"""
        try:
            with get_db_session() as db:
                for task in tasks:
                    # Update or create task in database
                    stmt = select(TaskModel).where(TaskModel.id == task.id)
                    task_model = db.execute(stmt).scalar_one_or_none()
                    
                    if task_model:
                        task_model.project_id = project_id
                        task_model.status = TaskStatus.PENDING
                    else:
                        task_model = TaskModel(
                            id=task.id,
                            document_id=task.document_id,
                            project_id=project_id,
                            status=TaskStatus.PENDING,
                            annotations=task.annotations,
                            ai_predictions=task.ai_predictions,
                            quality_score=task.quality_score
                        )
                        db.add(task_model)
                
                db.commit()
                logger.info(f"Synchronized {len(tasks)} tasks to database")
                
        except Exception as e:
            logger.error(f"Error syncing tasks to database: {str(e)}")
            raise
    
    async def _sync_annotations_to_db(self, project_id: str, annotations_data: List[Dict[str, Any]]) -> None:
        """Synchronize annotations from Label Studio to PostgreSQL"""
        try:
            with get_db_session() as db:
                for annotation in annotations_data:
                    # Extract task ID from annotation metadata
                    task_id = None
                    if 'meta' in annotation and 'superinsight_task_id' in annotation['meta']:
                        task_id = UUID(annotation['meta']['superinsight_task_id'])
                    elif 'data' in annotation and 'task_id' in annotation['data']:
                        task_id = UUID(annotation['data']['task_id'])
                    
                    if not task_id:
                        logger.warning(f"Could not find task ID for annotation: {annotation.get('id')}")
                        continue
                    
                    # Update task with annotation data
                    stmt = select(TaskModel).where(TaskModel.id == task_id)
                    task_model = db.execute(stmt).scalar_one_or_none()
                    if task_model:
                        # Add annotation to task
                        if not task_model.annotations:
                            task_model.annotations = []
                        
                        task_model.annotations.append({
                            "id": annotation.get("id"),
                            "result": annotation.get("annotations", [{}])[0].get("result", []),
                            "created_at": annotation.get("created_at"),
                            "updated_at": annotation.get("updated_at"),
                            "lead_time": annotation.get("lead_time", 0),
                            "annotator": annotation.get("completed_by", {}).get("id")
                        })
                        
                        # Update task status
                        if annotation.get("annotations"):
                            task_model.status = TaskStatus.COMPLETED
                        
                        db.add(task_model)
                
                db.commit()
                logger.info(f"Synchronized {len(annotations_data)} annotations to database")
                
        except Exception as e:
            logger.error(f"Error syncing annotations to database: {str(e)}")
            raise
    
    async def test_connection(self, timeout: float = 10.0) -> bool:
        """
        Test Label Studio API connectivity.

        Args:
            timeout: Connection timeout in seconds

        Returns:
            bool: True if connection is successful, False otherwise
        """
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                # Try to access the API health endpoint or user info
                response = await client.get(
                    f"{self.base_url}/api/current-user/whoami/",
                    headers=self.headers
                )

                if response.status_code == 200:
                    logger.info("Label Studio connection test successful")
                    return True
                elif response.status_code == 401:
                    logger.warning("Label Studio authentication failed")
                    return False
                else:
                    logger.warning(f"Label Studio returned status code: {response.status_code}")
                    return False

        except httpx.TimeoutException:
            logger.error("Label Studio connection timed out")
            return False
        except httpx.RequestError as e:
            logger.error(f"Label Studio connection error: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error testing Label Studio connection: {str(e)}")
            return False

    async def get_project_info(self, project_id: str) -> Optional[LabelStudioProject]:
        """Get project information from Label Studio"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.base_url}/api/projects/{project_id}/",
                    headers=self.headers
                )
                
                if response.status_code == 200:
                    project_data = response.json()
                    return LabelStudioProject(**project_data)
                else:
                    logger.error(f"Failed to get project info: {response.status_code}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting project info: {str(e)}")
            return None
    
    async def delete_project(self, project_id: str) -> bool:
        """Delete a Label Studio project"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.delete(
                    f"{self.base_url}/api/projects/{project_id}/",
                    headers=self.headers
                )
                
                if response.status_code == 204:
                    logger.info(f"Deleted Label Studio project: {project_id}")
                    return True
                else:
                    logger.error(f"Failed to delete project: {response.status_code}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error deleting project: {str(e)}")
            return False


# Global integration instance
label_studio_integration = LabelStudioIntegration()