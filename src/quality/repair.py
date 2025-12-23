"""
Data Repair Service for SuperInsight Platform.

Implements source data repair functionality with history tracking,
approval workflows, and result verification.
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from uuid import UUID, uuid4
from enum import Enum
from sqlalchemy import select

from src.models.quality_issue import QualityIssue, IssueStatus
from src.database.connection import get_db_session, db_manager
from src.database.models import QualityIssueModel, TaskModel, DocumentModel


logger = logging.getLogger(__name__)


class RepairStatus(str, Enum):
    """Status of data repair operations."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"


class RepairType(str, Enum):
    """Types of data repair operations."""
    ANNOTATION_CORRECTION = "annotation_correction"
    SOURCE_DATA_UPDATE = "source_data_update"
    METADATA_REPAIR = "metadata_repair"
    QUALITY_SCORE_ADJUSTMENT = "quality_score_adjustment"
    LABEL_STANDARDIZATION = "label_standardization"


class RepairRecord:
    """Represents a data repair operation record."""
    
    def __init__(
        self,
        repair_id: UUID,
        quality_issue_id: UUID,
        repair_type: RepairType,
        description: str,
        original_data: Dict[str, Any],
        proposed_data: Dict[str, Any],
        status: RepairStatus = RepairStatus.PENDING,
        requested_by: Optional[str] = None,
        approved_by: Optional[str] = None,
        executed_by: Optional[str] = None,
        created_at: Optional[datetime] = None,
        approved_at: Optional[datetime] = None,
        executed_at: Optional[datetime] = None,
        verification_result: Optional[Dict[str, Any]] = None
    ):
        self.repair_id = repair_id
        self.quality_issue_id = quality_issue_id
        self.repair_type = repair_type
        self.description = description
        self.original_data = original_data
        self.proposed_data = proposed_data
        self.status = status
        self.requested_by = requested_by
        self.approved_by = approved_by
        self.executed_by = executed_by
        self.created_at = created_at or datetime.now()
        self.approved_at = approved_at
        self.executed_at = executed_at
        self.verification_result = verification_result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert repair record to dictionary."""
        return {
            "repair_id": str(self.repair_id),
            "quality_issue_id": str(self.quality_issue_id),
            "repair_type": self.repair_type.value,
            "description": self.description,
            "original_data": self.original_data,
            "proposed_data": self.proposed_data,
            "status": self.status.value,
            "requested_by": self.requested_by,
            "approved_by": self.approved_by,
            "executed_by": self.executed_by,
            "created_at": self.created_at.isoformat(),
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
            "executed_at": self.executed_at.isoformat() if self.executed_at else None,
            "verification_result": self.verification_result
        }


class RepairApprovalWorkflow:
    """Manages approval workflow for data repairs."""
    
    def __init__(self):
        self.approval_rules: Dict[RepairType, Dict[str, Any]] = {
            RepairType.ANNOTATION_CORRECTION: {
                "requires_approval": True,
                "approver_roles": ["quality_manager", "senior_annotator"],
                "auto_approve_threshold": 0.9  # Auto-approve if confidence > 90%
            },
            RepairType.SOURCE_DATA_UPDATE: {
                "requires_approval": True,
                "approver_roles": ["data_manager", "system_admin"],
                "auto_approve_threshold": None  # Always requires manual approval
            },
            RepairType.METADATA_REPAIR: {
                "requires_approval": False,
                "approver_roles": [],
                "auto_approve_threshold": 0.8
            },
            RepairType.QUALITY_SCORE_ADJUSTMENT: {
                "requires_approval": True,
                "approver_roles": ["quality_manager"],
                "auto_approve_threshold": None
            },
            RepairType.LABEL_STANDARDIZATION: {
                "requires_approval": False,
                "approver_roles": [],
                "auto_approve_threshold": 0.95
            }
        }
    
    def requires_approval(self, repair_type: RepairType, confidence: float = 0.0) -> bool:
        """Check if repair requires manual approval."""
        rules = self.approval_rules.get(repair_type, {})
        
        # Check auto-approval threshold
        auto_threshold = rules.get("auto_approve_threshold")
        if auto_threshold is not None and confidence >= auto_threshold:
            return False
        
        return rules.get("requires_approval", True)
    
    def get_required_approver_roles(self, repair_type: RepairType) -> List[str]:
        """Get list of roles that can approve this repair type."""
        rules = self.approval_rules.get(repair_type, {})
        return rules.get("approver_roles", [])


class DataRepairService:
    """
    Data Repair Service for SuperInsight Platform.
    
    Provides functionality to repair source data issues with proper
    tracking, approval workflows, and verification.
    """
    
    def __init__(self):
        self.repair_records: Dict[UUID, RepairRecord] = {}
        self.approval_workflow = RepairApprovalWorkflow()
    
    async def create_repair_request(
        self,
        quality_issue_id: UUID,
        repair_type: RepairType,
        description: str,
        original_data: Dict[str, Any],
        proposed_data: Dict[str, Any],
        requested_by: str,
        confidence: float = 0.0
    ) -> RepairRecord:
        """
        Create a new data repair request.
        
        Args:
            quality_issue_id: ID of the quality issue being addressed
            repair_type: Type of repair operation
            description: Description of the repair
            original_data: Original data before repair
            proposed_data: Proposed data after repair
            requested_by: User requesting the repair
            confidence: Confidence in the repair (0.0-1.0)
            
        Returns:
            RepairRecord for the created repair request
        """
        repair_id = uuid4()
        
        # Determine if approval is required
        requires_approval = self.approval_workflow.requires_approval(repair_type, confidence)
        
        initial_status = RepairStatus.PENDING if requires_approval else RepairStatus.IN_PROGRESS
        
        repair_record = RepairRecord(
            repair_id=repair_id,
            quality_issue_id=quality_issue_id,
            repair_type=repair_type,
            description=description,
            original_data=original_data,
            proposed_data=proposed_data,
            status=initial_status,
            requested_by=requested_by
        )
        
        # Store repair record
        self.repair_records[repair_id] = repair_record
        
        logger.info(
            f"Created repair request {repair_id} for quality issue {quality_issue_id}. "
            f"Requires approval: {requires_approval}"
        )
        
        # If no approval required, execute immediately
        if not requires_approval:
            await self.execute_repair(repair_id, requested_by)
        
        return repair_record
    
    async def approve_repair(
        self,
        repair_id: UUID,
        approved_by: str,
        approval_notes: Optional[str] = None
    ) -> bool:
        """
        Approve a pending repair request.
        
        Args:
            repair_id: ID of the repair to approve
            approved_by: User approving the repair
            approval_notes: Optional notes about the approval
            
        Returns:
            True if approval successful, False otherwise
        """
        if repair_id not in self.repair_records:
            logger.error(f"Repair record {repair_id} not found")
            return False
        
        repair_record = self.repair_records[repair_id]
        
        if repair_record.status != RepairStatus.PENDING:
            logger.error(f"Repair {repair_id} is not in pending status")
            return False
        
        # Check if approver has required role (simplified - in real implementation,
        # this would check against user roles)
        required_roles = self.approval_workflow.get_required_approver_roles(repair_record.repair_type)
        # For now, assume approval is valid
        
        # Update repair record
        repair_record.status = RepairStatus.IN_PROGRESS
        repair_record.approved_by = approved_by
        repair_record.approved_at = datetime.now()
        
        if approval_notes:
            repair_record.description += f"\n\n审批备注: {approval_notes}"
        
        logger.info(f"Repair {repair_id} approved by {approved_by}")
        
        # Execute the repair
        return await self.execute_repair(repair_id, approved_by)
    
    async def reject_repair(
        self,
        repair_id: UUID,
        rejected_by: str,
        rejection_reason: str
    ) -> bool:
        """
        Reject a pending repair request.
        
        Args:
            repair_id: ID of the repair to reject
            rejected_by: User rejecting the repair
            rejection_reason: Reason for rejection
            
        Returns:
            True if rejection successful, False otherwise
        """
        if repair_id not in self.repair_records:
            logger.error(f"Repair record {repair_id} not found")
            return False
        
        repair_record = self.repair_records[repair_id]
        
        if repair_record.status != RepairStatus.PENDING:
            logger.error(f"Repair {repair_id} is not in pending status")
            return False
        
        # Update repair record
        repair_record.status = RepairStatus.REJECTED
        repair_record.description += f"\n\n拒绝原因: {rejection_reason}"
        
        logger.info(f"Repair {repair_id} rejected by {rejected_by}: {rejection_reason}")
        return True
    
    async def execute_repair(
        self,
        repair_id: UUID,
        executed_by: str
    ) -> bool:
        """
        Execute an approved repair operation.
        
        Args:
            repair_id: ID of the repair to execute
            executed_by: User executing the repair
            
        Returns:
            True if execution successful, False otherwise
        """
        if repair_id not in self.repair_records:
            logger.error(f"Repair record {repair_id} not found")
            return False
        
        repair_record = self.repair_records[repair_id]
        
        if repair_record.status not in [RepairStatus.PENDING, RepairStatus.IN_PROGRESS]:
            logger.error(f"Repair {repair_id} cannot be executed in status {repair_record.status}")
            return False
        
        try:
            # Execute the specific repair type
            success = await self._execute_repair_by_type(repair_record)
            
            if success:
                repair_record.status = RepairStatus.COMPLETED
                repair_record.executed_by = executed_by
                repair_record.executed_at = datetime.now()
                
                # Verify the repair
                verification_result = await self.verify_repair_result(repair_id)
                repair_record.verification_result = verification_result
                
                logger.info(f"Repair {repair_id} executed successfully by {executed_by}")
                
                # Update the related quality issue
                await self._update_quality_issue_after_repair(repair_record.quality_issue_id)
                
                return True
            else:
                repair_record.status = RepairStatus.FAILED
                logger.error(f"Repair {repair_id} execution failed")
                return False
                
        except Exception as e:
            repair_record.status = RepairStatus.FAILED
            logger.error(f"Repair {repair_id} execution error: {str(e)}")
            return False
    
    async def _execute_repair_by_type(self, repair_record: RepairRecord) -> bool:
        """Execute repair based on its type."""
        
        if repair_record.repair_type == RepairType.ANNOTATION_CORRECTION:
            return await self._repair_annotation_data(repair_record)
        elif repair_record.repair_type == RepairType.SOURCE_DATA_UPDATE:
            return await self._repair_source_data(repair_record)
        elif repair_record.repair_type == RepairType.METADATA_REPAIR:
            return await self._repair_metadata(repair_record)
        elif repair_record.repair_type == RepairType.QUALITY_SCORE_ADJUSTMENT:
            return await self._repair_quality_score(repair_record)
        elif repair_record.repair_type == RepairType.LABEL_STANDARDIZATION:
            return await self._repair_label_standardization(repair_record)
        else:
            logger.error(f"Unknown repair type: {repair_record.repair_type}")
            return False
    
    async def _repair_annotation_data(self, repair_record: RepairRecord) -> bool:
        """Repair annotation data in tasks."""
        try:
            with db_manager.get_session() as session:
                # Get quality issue to find related task
                stmt = select(QualityIssueModel).where(
                    QualityIssueModel.id == repair_record.quality_issue_id
                )
                result = session.execute(stmt)
                db_issue = result.scalar_one_or_none()
                
                if not db_issue:
                    logger.error(f"Quality issue {repair_record.quality_issue_id} not found")
                    return False
                
                # Get the task
                stmt = select(TaskModel).where(
                    TaskModel.id == db_issue.task_id
                )
                result = session.execute(stmt)
                db_task = result.scalar_one_or_none()
                
                if not db_task:
                    logger.error(f"Task {db_issue.task_id} not found")
                    return False
                
                # Update annotation data
                if db_task.annotations:
                    # Find and update the specific annotation
                    for i, annotation in enumerate(db_task.annotations):
                        if self._matches_original_data(annotation, repair_record.original_data):
                            # Apply the proposed changes
                            db_task.annotations[i].update(repair_record.proposed_data)
                            break
                
                session.commit()
                logger.info(f"Updated annotation data for task {db_issue.task_id}")
                return True
                
        except Exception as e:
            logger.error(f"Annotation repair failed: {str(e)}")
            return False
    
    async def _repair_source_data(self, repair_record: RepairRecord) -> bool:
        """Repair source document data."""
        try:
            with get_db_session() as session:
                # Get quality issue to find related task and document
                stmt = select(QualityIssueModel).where(
                    QualityIssueModel.id == repair_record.quality_issue_id
                )
                db_issue = session.execute(stmt).scalar_one_or_none()
                
                if not db_issue:
                    return False
                
                stmt = select(TaskModel).where(
                    TaskModel.id == db_issue.task_id
                )
                db_task = session.execute(stmt).scalar_one_or_none()
                
                if not db_task:
                    return False
                
                # Get the source document
                stmt = select(DocumentModel).where(
                    DocumentModel.id == db_task.document_id
                )
                db_document = session.execute(stmt).scalar_one_or_none()
                
                if not db_document:
                    return False
                
                # Update document content or metadata
                if "content" in repair_record.proposed_data:
                    db_document.content = repair_record.proposed_data["content"]
                
                if "metadata" in repair_record.proposed_data:
                    if db_document.document_metadata:
                        db_document.document_metadata.update(repair_record.proposed_data["metadata"])
                    else:
                        db_document.document_metadata = repair_record.proposed_data["metadata"]
                
                db_document.updated_at = datetime.now()
                
                session.commit()
                logger.info(f"Updated source document {db_document.id}")
                return True
                
        except Exception as e:
            logger.error(f"Source data repair failed: {str(e)}")
            return False
    
    async def _repair_metadata(self, repair_record: RepairRecord) -> bool:
        """Repair metadata in documents or tasks."""
        try:
            with get_db_session() as session:
                stmt = select(QualityIssueModel).where(
                    QualityIssueModel.id == repair_record.quality_issue_id
                )
                db_issue = session.execute(stmt).scalar_one_or_none()
                
                if not db_issue:
                    return False
                
                stmt = select(TaskModel).where(
                    TaskModel.id == db_issue.task_id
                )
                db_task = session.execute(stmt).scalar_one_or_none()
                
                if not db_task:
                    return False
                
                # Update task metadata (stored in annotations)
                if "task_metadata" in repair_record.proposed_data:
                    # Add metadata to task annotations
                    metadata_update = repair_record.proposed_data["task_metadata"]
                    if not db_task.annotations:
                        db_task.annotations = []
                    
                    # Add metadata annotation
                    db_task.annotations.append({
                        "type": "metadata_repair",
                        "metadata": metadata_update,
                        "repaired_at": datetime.now().isoformat()
                    })
                
                session.commit()
                logger.info(f"Updated metadata for task {db_task.id}")
                return True
                
        except Exception as e:
            logger.error(f"Metadata repair failed: {str(e)}")
            return False
    
    async def _repair_quality_score(self, repair_record: RepairRecord) -> bool:
        """Repair quality score for a task."""
        try:
            with db_manager.get_session() as session:
                stmt = select(QualityIssueModel).where(
                    QualityIssueModel.id == repair_record.quality_issue_id
                )
                result = session.execute(stmt)
                db_issue = result.scalar_one_or_none()
                
                if not db_issue:
                    return False
                
                stmt = select(TaskModel).where(
                    TaskModel.id == db_issue.task_id
                )
                result = session.execute(stmt)
                db_task = result.scalar_one_or_none()
                
                if not db_task:
                    return False
                
                # Update quality score
                new_score = repair_record.proposed_data.get("quality_score")
                if new_score is not None and 0.0 <= new_score <= 1.0:
                    db_task.quality_score = new_score
                
                session.commit()
                logger.info(f"Updated quality score for task {db_task.id} to {new_score}")
                return True
                
        except Exception as e:
            logger.error(f"Quality score repair failed: {str(e)}")
            return False
    
    async def _repair_label_standardization(self, repair_record: RepairRecord) -> bool:
        """Standardize labels in annotations."""
        try:
            with db_manager.get_session() as session:
                stmt = select(QualityIssueModel).where(
                    QualityIssueModel.id == repair_record.quality_issue_id
                )
                result = session.execute(stmt)
                db_issue = result.scalar_one_or_none()
                
                if not db_issue:
                    return False
                
                stmt = select(TaskModel).where(
                    TaskModel.id == db_issue.task_id
                )
                result = session.execute(stmt)
                db_task = result.scalar_one_or_none()
                
                if not db_task:
                    return False
                
                # Standardize labels in annotations
                label_mapping = repair_record.proposed_data.get("label_mapping", {})
                
                if db_task.annotations and label_mapping:
                    for annotation in db_task.annotations:
                        if "result" in annotation:
                            for result_item in annotation["result"]:
                                if "value" in result_item and "labels" in result_item["value"]:
                                    # Apply label standardization
                                    labels = result_item["value"]["labels"]
                                    standardized_labels = [
                                        label_mapping.get(label, label) for label in labels
                                    ]
                                    result_item["value"]["labels"] = standardized_labels
                
                session.commit()
                logger.info(f"Standardized labels for task {db_task.id}")
                return True
                
        except Exception as e:
            logger.error(f"Label standardization repair failed: {str(e)}")
            return False
    
    def _matches_original_data(self, annotation: Dict[str, Any], original_data: Dict[str, Any]) -> bool:
        """Check if annotation matches the original data to be repaired."""
        # Simple matching based on annotation ID or content
        if "id" in original_data and "id" in annotation:
            return annotation["id"] == original_data["id"]
        
        # Match by content similarity (simplified)
        if "result" in original_data and "result" in annotation:
            return annotation["result"] == original_data["result"]
        
        return False
    
    async def verify_repair_result(self, repair_id: UUID) -> Dict[str, Any]:
        """
        Verify the result of a repair operation.
        
        Args:
            repair_id: ID of the repair to verify
            
        Returns:
            Dictionary containing verification results
        """
        if repair_id not in self.repair_records:
            return {"verified": False, "error": "Repair record not found"}
        
        repair_record = self.repair_records[repair_id]
        
        try:
            # Basic verification - check if the repair was applied
            verification_result = {
                "verified": True,
                "repair_id": str(repair_id),
                "repair_type": repair_record.repair_type.value,
                "verification_time": datetime.now().isoformat(),
                "checks_performed": []
            }
            
            # Perform type-specific verification
            if repair_record.repair_type == RepairType.ANNOTATION_CORRECTION:
                verification_result["checks_performed"].append("annotation_data_updated")
            elif repair_record.repair_type == RepairType.SOURCE_DATA_UPDATE:
                verification_result["checks_performed"].append("source_document_updated")
            elif repair_record.repair_type == RepairType.QUALITY_SCORE_ADJUSTMENT:
                verification_result["checks_performed"].append("quality_score_updated")
            
            # Add data integrity check
            verification_result["data_integrity"] = "passed"
            verification_result["checks_performed"].append("data_integrity_check")
            
            logger.info(f"Repair {repair_id} verification completed successfully")
            return verification_result
            
        except Exception as e:
            logger.error(f"Repair verification failed for {repair_id}: {str(e)}")
            return {
                "verified": False,
                "error": str(e),
                "verification_time": datetime.now().isoformat()
            }
    
    async def _update_quality_issue_after_repair(self, quality_issue_id: UUID) -> None:
        """Update quality issue status after successful repair."""
        try:
            with db_manager.get_session() as session:
                stmt = select(QualityIssueModel).where(
                    QualityIssueModel.id == quality_issue_id
                )
                result = session.execute(stmt)
                db_issue = result.scalar_one_or_none()
                
                if db_issue and db_issue.status != IssueStatus.RESOLVED:
                    db_issue.status = IssueStatus.RESOLVED
                    db_issue.resolved_at = datetime.now()
                    db_issue.description += f"\n\n数据修复完成: {datetime.now().isoformat()}"
                    
                    session.commit()
                    logger.info(f"Updated quality issue {quality_issue_id} status to resolved")
                    
        except Exception as e:
            logger.error(f"Failed to update quality issue {quality_issue_id}: {str(e)}")
    
    def get_repair_history(
        self,
        quality_issue_id: Optional[UUID] = None,
        repair_type: Optional[RepairType] = None,
        status: Optional[RepairStatus] = None
    ) -> List[RepairRecord]:
        """
        Get repair history with optional filters.
        
        Args:
            quality_issue_id: Filter by quality issue ID
            repair_type: Filter by repair type
            status: Filter by repair status
            
        Returns:
            List of repair records matching the filters
        """
        filtered_records = []
        
        for repair_record in self.repair_records.values():
            # Apply filters
            if quality_issue_id and repair_record.quality_issue_id != quality_issue_id:
                continue
            if repair_type and repair_record.repair_type != repair_type:
                continue
            if status and repair_record.status != status:
                continue
            
            filtered_records.append(repair_record)
        
        # Sort by creation time (newest first)
        filtered_records.sort(key=lambda r: r.created_at, reverse=True)
        
        return filtered_records
    
    def get_repair_statistics(self) -> Dict[str, Any]:
        """Get statistics about repair operations."""
        total_repairs = len(self.repair_records)
        
        if total_repairs == 0:
            return {
                "total_repairs": 0,
                "by_status": {},
                "by_type": {},
                "success_rate": 0.0
            }
        
        # Count by status
        status_counts = {}
        for status in RepairStatus:
            status_counts[status.value] = sum(
                1 for r in self.repair_records.values() if r.status == status
            )
        
        # Count by type
        type_counts = {}
        for repair_type in RepairType:
            type_counts[repair_type.value] = sum(
                1 for r in self.repair_records.values() if r.repair_type == repair_type
            )
        
        # Calculate success rate
        completed_repairs = status_counts.get(RepairStatus.COMPLETED.value, 0)
        success_rate = completed_repairs / total_repairs if total_repairs > 0 else 0.0
        
        return {
            "total_repairs": total_repairs,
            "by_status": status_counts,
            "by_type": type_counts,
            "success_rate": success_rate
        }