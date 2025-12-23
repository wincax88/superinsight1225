"""
Data reconstruction service implementation.

Provides functionality for data structure transformation and reconstruction.
"""

import time
import logging
from typing import List, Dict, Any, Optional, Union
from uuid import UUID, uuid4
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, field_validator, ConfigDict

from ..models.document import Document
from ..models.task import Task
from ..models.annotation import Annotation


logger = logging.getLogger(__name__)


class ReconstructionType(str, Enum):
    """Types of data reconstruction operations."""
    STRUCTURE_TRANSFORM = "structure_transform"
    FORMAT_CONVERSION = "format_conversion"
    SCHEMA_MIGRATION = "schema_migration"
    DATA_NORMALIZATION = "data_normalization"


class ReconstructionConfig(BaseModel):
    """Configuration for data reconstruction operations."""
    
    reconstruction_type: ReconstructionType = Field(..., description="Type of reconstruction")
    source_format: str = Field(..., description="Source data format")
    target_format: str = Field(..., description="Target data format")
    transformation_rules: Dict[str, Any] = Field(default_factory=dict, description="Transformation rules")
    preserve_metadata: bool = Field(default=True, description="Whether to preserve metadata")
    validate_output: bool = Field(default=True, description="Whether to validate output")
    
    @field_validator('source_format', 'target_format')
    @classmethod
    def validate_formats(cls, v):
        """Validate that formats are not empty."""
        if not v or not v.strip():
            raise ValueError('format cannot be empty')
        return v


class ReconstructionRecord(BaseModel):
    """Record of a data reconstruction operation."""
    
    id: UUID = Field(default_factory=uuid4, description="Unique record identifier")
    source_id: UUID = Field(..., description="Source data identifier")
    target_id: UUID = Field(..., description="Target data identifier")
    reconstruction_type: ReconstructionType = Field(..., description="Type of reconstruction")
    config: ReconstructionConfig = Field(..., description="Reconstruction configuration")
    status: str = Field(default="pending", description="Reconstruction status")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary for JSON serialization."""
        return {
            "id": str(self.id),
            "source_id": str(self.source_id),
            "target_id": str(self.target_id),
            "reconstruction_type": self.reconstruction_type.value,
            "config": self.config.dict(),
            "status": self.status,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }


class ReconstructionResult(BaseModel):
    """Result of a data reconstruction operation."""
    
    success: bool = Field(..., description="Whether reconstruction succeeded")
    reconstructed_data: Optional[Union[Document, Task, Dict[str, Any]]] = Field(None, description="Reconstructed data")
    record: ReconstructionRecord = Field(..., description="Reconstruction record")
    validation_errors: List[str] = Field(default_factory=list, description="Validation errors")
    processing_time: float = Field(..., description="Processing time in seconds")
    
    model_config = ConfigDict(arbitrary_types_allowed=True)


class DataReconstructionService:
    """
    Service for data reconstruction and transformation.
    
    Provides functionality for:
    - Data structure transformation
    - Format conversion
    - Schema migration
    - Data normalization
    """
    
    def __init__(self, db_manager=None):
        """Initialize the data reconstruction service."""
        self.db_manager = db_manager
        self.reconstruction_history: List[ReconstructionRecord] = []
    
    async def reconstruct_document(
        self, 
        document: Document, 
        config: ReconstructionConfig
    ) -> ReconstructionResult:
        """
        Reconstruct a document according to the specified configuration.
        
        Args:
            document: Source document to reconstruct
            config: Reconstruction configuration
            
        Returns:
            Reconstruction result with transformed data
        """
        start_time = time.time()
        logger.info(f"Starting document reconstruction: {config.reconstruction_type}")
        
        # Create reconstruction record
        record = ReconstructionRecord(
            source_id=document.id,
            target_id=uuid4(),
            reconstruction_type=config.reconstruction_type,
            config=config,
            status="in_progress"
        )
        
        try:
            # Perform reconstruction based on type
            if config.reconstruction_type == ReconstructionType.STRUCTURE_TRANSFORM:
                reconstructed_doc = await self._transform_document_structure(document, config)
            elif config.reconstruction_type == ReconstructionType.FORMAT_CONVERSION:
                reconstructed_doc = await self._convert_document_format(document, config)
            elif config.reconstruction_type == ReconstructionType.SCHEMA_MIGRATION:
                reconstructed_doc = await self._migrate_document_schema(document, config)
            elif config.reconstruction_type == ReconstructionType.DATA_NORMALIZATION:
                reconstructed_doc = await self._normalize_document_data(document, config)
            else:
                raise ValueError(f"Unsupported reconstruction type: {config.reconstruction_type}")
            
            # Update record with target ID
            record.target_id = reconstructed_doc.id
            record.status = "completed"
            record.completed_at = datetime.now()
            
            # Validate output if requested
            validation_errors = []
            if config.validate_output:
                validation_errors = await self._validate_reconstructed_document(reconstructed_doc, config)
            
            processing_time = time.time() - start_time
            
            # Store reconstruction record
            self.reconstruction_history.append(record)
            
            result = ReconstructionResult(
                success=True,
                reconstructed_data=reconstructed_doc,
                record=record,
                validation_errors=validation_errors,
                processing_time=processing_time
            )
            
            logger.info(f"Document reconstruction completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            # Update record with error
            record.status = "failed"
            record.error_message = str(e)
            record.completed_at = datetime.now()
            self.reconstruction_history.append(record)
            
            processing_time = time.time() - start_time
            
            result = ReconstructionResult(
                success=False,
                record=record,
                validation_errors=[str(e)],
                processing_time=processing_time
            )
            
            logger.error(f"Document reconstruction failed: {str(e)}")
            return result
    
    async def reconstruct_task(
        self, 
        task: Task, 
        config: ReconstructionConfig
    ) -> ReconstructionResult:
        """
        Reconstruct a task according to the specified configuration.
        
        Args:
            task: Source task to reconstruct
            config: Reconstruction configuration
            
        Returns:
            Reconstruction result with transformed task
        """
        start_time = time.time()
        logger.info(f"Starting task reconstruction: {config.reconstruction_type}")
        
        # Create reconstruction record
        record = ReconstructionRecord(
            source_id=task.id,
            target_id=uuid4(),
            reconstruction_type=config.reconstruction_type,
            config=config,
            status="in_progress"
        )
        
        try:
            # Perform task reconstruction
            if config.reconstruction_type == ReconstructionType.STRUCTURE_TRANSFORM:
                reconstructed_task = await self._transform_task_structure(task, config)
            elif config.reconstruction_type == ReconstructionType.FORMAT_CONVERSION:
                reconstructed_task = await self._convert_task_format(task, config)
            else:
                # For other types, use generic transformation
                reconstructed_task = await self._generic_task_reconstruction(task, config)
            
            # Update record
            record.target_id = reconstructed_task.id
            record.status = "completed"
            record.completed_at = datetime.now()
            
            # Validate output
            validation_errors = []
            if config.validate_output:
                validation_errors = await self._validate_reconstructed_task(reconstructed_task, config)
            
            processing_time = time.time() - start_time
            self.reconstruction_history.append(record)
            
            result = ReconstructionResult(
                success=True,
                reconstructed_data=reconstructed_task,
                record=record,
                validation_errors=validation_errors,
                processing_time=processing_time
            )
            
            logger.info(f"Task reconstruction completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            record.status = "failed"
            record.error_message = str(e)
            record.completed_at = datetime.now()
            self.reconstruction_history.append(record)
            
            processing_time = time.time() - start_time
            
            result = ReconstructionResult(
                success=False,
                record=record,
                validation_errors=[str(e)],
                processing_time=processing_time
            )
            
            logger.error(f"Task reconstruction failed: {str(e)}")
            return result
    
    async def batch_reconstruct(
        self, 
        data_items: List[Union[Document, Task]], 
        config: ReconstructionConfig
    ) -> List[ReconstructionResult]:
        """
        Perform batch reconstruction of multiple data items.
        
        Args:
            data_items: List of documents or tasks to reconstruct
            config: Reconstruction configuration
            
        Returns:
            List of reconstruction results
        """
        logger.info(f"Starting batch reconstruction for {len(data_items)} items")
        
        results = []
        for item in data_items:
            if isinstance(item, Document):
                result = await self.reconstruct_document(item, config)
            elif isinstance(item, Task):
                result = await self.reconstruct_task(item, config)
            else:
                # Create error result for unsupported type
                record = ReconstructionRecord(
                    source_id=uuid4(),
                    target_id=uuid4(),
                    reconstruction_type=config.reconstruction_type,
                    config=config,
                    status="failed",
                    error_message=f"Unsupported data type: {type(item)}"
                )
                result = ReconstructionResult(
                    success=False,
                    record=record,
                    validation_errors=[f"Unsupported data type: {type(item)}"],
                    processing_time=0.0
                )
            
            results.append(result)
        
        logger.info(f"Batch reconstruction completed: {len(results)} results")
        return results
    
    def get_reconstruction_history(
        self, 
        source_id: Optional[UUID] = None,
        reconstruction_type: Optional[ReconstructionType] = None
    ) -> List[ReconstructionRecord]:
        """
        Get reconstruction history with optional filtering.
        
        Args:
            source_id: Filter by source data ID
            reconstruction_type: Filter by reconstruction type
            
        Returns:
            List of reconstruction records
        """
        history = self.reconstruction_history
        
        if source_id:
            history = [r for r in history if r.source_id == source_id]
        
        if reconstruction_type:
            history = [r for r in history if r.reconstruction_type == reconstruction_type]
        
        return history
    
    async def verify_reconstruction(
        self, 
        record: ReconstructionRecord
    ) -> Dict[str, Any]:
        """
        Verify a completed reconstruction operation.
        
        Args:
            record: Reconstruction record to verify
            
        Returns:
            Verification result
        """
        logger.info(f"Verifying reconstruction: {record.id}")
        
        if record.status != "completed":
            return {
                "verified": False,
                "reason": f"Reconstruction not completed (status: {record.status})"
            }
        
        # Mock verification - in real implementation, would check data integrity
        verification_result = {
            "verified": True,
            "source_id": str(record.source_id),
            "target_id": str(record.target_id),
            "reconstruction_type": record.reconstruction_type.value,
            "verification_time": datetime.now().isoformat()
        }
        
        logger.info(f"Reconstruction verification completed: {record.id}")
        return verification_result
    
    # Private helper methods for different reconstruction types
    
    async def _transform_document_structure(
        self, 
        document: Document, 
        config: ReconstructionConfig
    ) -> Document:
        """Transform document structure according to rules."""
        # Apply transformation rules
        transformed_content = document.content
        transformed_metadata = document.metadata.copy() if config.preserve_metadata else {}
        
        # Apply transformation rules from config
        for rule_name, rule_value in config.transformation_rules.items():
            if rule_name == "add_prefix":
                transformed_content = f"{rule_value}{transformed_content}"
            elif rule_name == "add_suffix":
                transformed_content = f"{transformed_content}{rule_value}"
            elif rule_name == "add_metadata":
                transformed_metadata.update(rule_value)
        
        # Create new document with transformed data
        return Document(
            source_type=document.source_type,
            source_config=document.source_config,
            content=transformed_content,
            metadata={
                **transformed_metadata,
                "reconstructed": True,
                "original_id": str(document.id),
                "reconstruction_type": config.reconstruction_type.value
            }
        )
    
    async def _convert_document_format(
        self, 
        document: Document, 
        config: ReconstructionConfig
    ) -> Document:
        """Convert document format."""
        # Mock format conversion
        converted_content = document.content
        
        if config.source_format == "text" and config.target_format == "json":
            # Convert text to JSON structure
            converted_content = f'{{"content": "{document.content}", "format": "json"}}'
        elif config.source_format == "json" and config.target_format == "text":
            # Extract text from JSON (simplified)
            converted_content = document.content.replace('{"content": "', '').replace('", "format": "json"}', '')
        
        return Document(
            source_type=document.source_type,
            source_config={**document.source_config, "target_format": config.target_format},
            content=converted_content,
            metadata={
                **(document.metadata if config.preserve_metadata else {}),
                "format_converted": True,
                "source_format": config.source_format,
                "target_format": config.target_format
            }
        )
    
    async def _migrate_document_schema(
        self, 
        document: Document, 
        config: ReconstructionConfig
    ) -> Document:
        """Migrate document to new schema."""
        # Mock schema migration
        migrated_metadata = document.metadata.copy() if config.preserve_metadata else {}
        
        # Add schema migration metadata
        migrated_metadata.update({
            "schema_version": "2.0",
            "migrated_from": "1.0",
            "migration_date": datetime.now().isoformat()
        })
        
        return Document(
            source_type=document.source_type,
            source_config=document.source_config,
            content=document.content,
            metadata=migrated_metadata
        )
    
    async def _normalize_document_data(
        self, 
        document: Document, 
        config: ReconstructionConfig
    ) -> Document:
        """Normalize document data."""
        # Mock data normalization
        normalized_content = document.content.strip().lower()
        
        return Document(
            source_type=document.source_type,
            source_config=document.source_config,
            content=normalized_content,
            metadata={
                **(document.metadata if config.preserve_metadata else {}),
                "normalized": True,
                "normalization_rules": ["strip", "lowercase"]
            }
        )
    
    async def _transform_task_structure(
        self, 
        task: Task, 
        config: ReconstructionConfig
    ) -> Task:
        """Transform task structure."""
        # Create new task with transformed structure
        return Task(
            document_id=task.document_id,
            project_id=task.project_id,
            status=task.status,
            annotations=task.annotations.copy(),
            ai_predictions=task.ai_predictions.copy(),
            quality_score=task.quality_score
        )
    
    async def _convert_task_format(
        self, 
        task: Task, 
        config: ReconstructionConfig
    ) -> Task:
        """Convert task format."""
        # Mock task format conversion
        return Task(
            document_id=task.document_id,
            project_id=f"{task.project_id}_{config.target_format}",
            status=task.status,
            annotations=task.annotations.copy(),
            ai_predictions=task.ai_predictions.copy(),
            quality_score=task.quality_score
        )
    
    async def _generic_task_reconstruction(
        self, 
        task: Task, 
        config: ReconstructionConfig
    ) -> Task:
        """Generic task reconstruction."""
        return Task(
            document_id=task.document_id,
            project_id=task.project_id,
            status=task.status,
            annotations=task.annotations.copy(),
            ai_predictions=task.ai_predictions.copy(),
            quality_score=task.quality_score
        )
    
    async def _validate_reconstructed_document(
        self, 
        document: Document, 
        config: ReconstructionConfig
    ) -> List[str]:
        """Validate reconstructed document."""
        errors = []
        
        # Basic validation
        if not document.content:
            errors.append("Document content is empty")
        
        if not document.source_type:
            errors.append("Document source_type is missing")
        
        return errors
    
    async def _validate_reconstructed_task(
        self, 
        task: Task, 
        config: ReconstructionConfig
    ) -> List[str]:
        """Validate reconstructed task."""
        errors = []
        
        # Basic validation
        if not task.project_id:
            errors.append("Task project_id is missing")
        
        if not task.document_id:
            errors.append("Task document_id is missing")
        
        return errors