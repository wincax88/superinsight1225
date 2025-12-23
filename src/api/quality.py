"""
Quality Management API for SuperInsight Platform.

Provides REST API endpoints for quality assessment, issue management,
and data repair operations.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field

from src.quality.repair import DataRepairService, RepairRecord, RepairType, RepairStatus

# Lazy import for quality manager to avoid Ragas dependency issues
def get_quality_manager():
    """Get QualityManager instance with lazy import."""
    from src.quality.manager import QualityManager
    return QualityManager()

def get_quality_rule():
    """Get QualityRule class with lazy import."""
    from src.quality.manager import QualityRule
    return QualityRule

def get_quality_rule_type():
    """Get QualityRuleType enum with lazy import."""
    from src.quality.manager import QualityRuleType
    return QualityRuleType

def get_quality_report():
    """Get QualityReport class with lazy import."""
    from src.quality.manager import QualityReport
    return QualityReport


router = APIRouter(prefix="/api/quality", tags=["quality"])

# Global instances - use lazy initialization
_quality_manager = None
_repair_service = None

def get_quality_manager_instance():
    """Get or create quality manager instance."""
    global _quality_manager
    if _quality_manager is None:
        _quality_manager = get_quality_manager()
    return _quality_manager

def get_repair_service_instance():
    """Get or create repair service instance."""
    global _repair_service
    if _repair_service is None:
        _repair_service = DataRepairService()
    return _repair_service


# Request/Response Models

class QualityRuleRequest(BaseModel):
    """Request model for creating/updating quality rules."""
    rule_id: str = Field(..., description="Unique rule identifier")
    rule_type: str = Field(..., description="Type of quality rule")  # Use string instead of enum
    name: str = Field(..., description="Human-readable rule name")
    description: str = Field(..., description="Rule description")
    threshold: float = Field(0.7, ge=0.0, le=1.0, description="Quality threshold")
    severity: str = Field("medium", description="Issue severity")  # Use string instead of enum
    enabled: bool = Field(True, description="Whether rule is enabled")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Rule parameters")


class QualityEvaluationRequest(BaseModel):
    """Request model for quality evaluation."""
    task_id: UUID = Field(..., description="Task ID to evaluate")
    annotations: List[Dict[str, Any]] = Field(..., description="Annotations to evaluate")


class QualityIssueRequest(BaseModel):
    """Request model for creating quality issues."""
    task_id: UUID = Field(..., description="Related task ID")
    issue_type: str = Field(..., description="Type of quality issue")
    description: str = Field(..., description="Issue description")
    severity: str = Field("medium", description="Issue severity")  # Use string instead of enum
    assignee_id: Optional[str] = Field(None, description="Assigned user ID")


class RepairRequest(BaseModel):
    """Request model for creating repair requests."""
    quality_issue_id: UUID = Field(..., description="Quality issue ID")
    repair_type: str = Field(..., description="Type of repair")  # Use string instead of enum
    description: str = Field(..., description="Repair description")
    original_data: Dict[str, Any] = Field(..., description="Original data")
    proposed_data: Dict[str, Any] = Field(..., description="Proposed data")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Repair confidence")


class RepairApprovalRequest(BaseModel):
    """Request model for repair approval."""
    approved: bool = Field(..., description="Whether to approve or reject")
    notes: Optional[str] = Field(None, description="Approval/rejection notes")


# Quality Rule Management Endpoints

@router.get("/rules/templates")
async def get_rule_templates() -> Dict[str, Dict[str, Any]]:
    """Get available quality rule templates."""
    quality_manager = get_quality_manager_instance()
    templates = quality_manager.get_rule_templates()
    return {rule_id: rule.to_dict() for rule_id, rule in templates.items()}


@router.get("/rules")
async def get_quality_rules() -> Dict[str, Dict[str, Any]]:
    """Get all configured quality rules."""
    quality_manager = get_quality_manager_instance()
    return {rule_id: rule.to_dict() for rule_id, rule in quality_manager.quality_rules.items()}


@router.post("/rules")
async def create_quality_rule(rule_request: QualityRuleRequest) -> Dict[str, str]:
    """Create or update a quality rule."""
    quality_manager = get_quality_manager_instance()
    QualityRule = get_quality_rule()
    QualityRuleType = get_quality_rule_type()
    
    # Convert string to enum
    rule_type = QualityRuleType(rule_request.rule_type)
    
    # Import IssueSeverity for severity conversion
    from src.models.quality_issue import IssueSeverity
    severity = IssueSeverity(rule_request.severity)
    
    rule = QualityRule(
        rule_id=rule_request.rule_id,
        rule_type=rule_type,
        name=rule_request.name,
        description=rule_request.description,
        threshold=rule_request.threshold,
        severity=severity,
        enabled=rule_request.enabled,
        parameters=rule_request.parameters
    )
    
    quality_manager.add_quality_rule(rule)
    
    return {"message": f"Quality rule '{rule_request.rule_id}' created successfully"}


@router.put("/rules/{rule_id}/enable")
async def enable_quality_rule(rule_id: str) -> Dict[str, str]:
    """Enable a quality rule."""
    quality_manager = get_quality_manager_instance()
    success = quality_manager.enable_rule(rule_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Quality rule '{rule_id}' not found")
    
    return {"message": f"Quality rule '{rule_id}' enabled"}


@router.put("/rules/{rule_id}/disable")
async def disable_quality_rule(rule_id: str) -> Dict[str, str]:
    """Disable a quality rule."""
    quality_manager = get_quality_manager_instance()
    success = quality_manager.disable_rule(rule_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Quality rule '{rule_id}' not found")
    
    return {"message": f"Quality rule '{rule_id}' disabled"}


@router.delete("/rules/{rule_id}")
async def delete_quality_rule(rule_id: str) -> Dict[str, str]:
    """Delete a quality rule."""
    quality_manager = get_quality_manager_instance()
    success = quality_manager.remove_quality_rule(rule_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Quality rule '{rule_id}' not found")
    
    return {"message": f"Quality rule '{rule_id}' deleted"}


# Quality Evaluation Endpoints

@router.post("/evaluate")
async def evaluate_quality(request: QualityEvaluationRequest) -> Dict[str, Any]:
    """Evaluate quality of annotations."""
    try:
        quality_manager = get_quality_manager_instance()
        
        # Import Annotation class
        from src.models.annotation import Annotation
        
        # Convert annotation data to Annotation objects
        annotations = []
        for ann_data in request.annotations:
            annotation = Annotation(
                task_id=request.task_id,
                annotator_id=ann_data.get("annotator_id", "unknown"),
                annotation_data=ann_data.get("annotation_data", {}),
                confidence=ann_data.get("confidence", 1.0),
                time_spent=ann_data.get("time_spent", 0)
            )
            annotations.append(annotation)
        
        # Run quality evaluation
        quality_report = await quality_manager.evaluate_quality(annotations)
        
        return quality_report.to_dict()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quality evaluation failed: {str(e)}")


@router.post("/trigger-check/{task_id}")
async def trigger_quality_check(task_id: UUID, annotation_data: Dict[str, Any]) -> Dict[str, Any]:
    """Trigger quality check for a completed annotation."""
    try:
        quality_manager = get_quality_manager_instance()
        success = await quality_manager.trigger_quality_check(task_id, annotation_data)
        
        if success:
            return {"message": f"Quality check triggered for task {task_id}", "success": True}
        else:
            raise HTTPException(status_code=500, detail="Quality check failed")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quality check trigger failed: {str(e)}")


# Quality Issue Management Endpoints

@router.post("/issues")
async def create_quality_issue(request: QualityIssueRequest) -> Dict[str, Any]:
    """Create a new quality issue."""
    try:
        quality_manager = get_quality_manager_instance()
        
        # Import IssueSeverity for conversion
        from src.models.quality_issue import IssueSeverity
        severity = IssueSeverity(request.severity)
        
        issue = await quality_manager.create_quality_issue(
            task_id=request.task_id,
            issue_type=request.issue_type,
            description=request.description,
            severity=severity,
            assignee_id=request.assignee_id
        )
        
        return issue.to_dict()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create quality issue: {str(e)}")


@router.get("/issues")
async def get_quality_issues(
    task_id: Optional[UUID] = Query(None, description="Filter by task ID"),
    status: Optional[str] = Query(None, description="Filter by status"),
    assignee_id: Optional[str] = Query(None, description="Filter by assignee")
) -> List[Dict[str, Any]]:
    """Get quality issues with optional filters."""
    try:
        quality_manager = get_quality_manager_instance()
        
        # Convert string status to enum if provided
        status_enum = None
        if status:
            from src.models.quality_issue import IssueStatus
            status_enum = IssueStatus(status)
        
        issues = await quality_manager.get_quality_issues(
            task_id=task_id,
            status=status_enum,
            assignee_id=assignee_id
        )
        
        return [issue.to_dict() for issue in issues]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get quality issues: {str(e)}")


@router.put("/issues/{issue_id}/assign")
async def assign_quality_issue(issue_id: UUID, assignee_id: str) -> Dict[str, str]:
    """Assign a quality issue to a user."""
    try:
        quality_manager = get_quality_manager_instance()
        success = await quality_manager.assign_quality_issue(issue_id, assignee_id)
        
        if success:
            return {"message": f"Quality issue {issue_id} assigned to {assignee_id}"}
        else:
            raise HTTPException(status_code=404, detail="Quality issue not found")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to assign quality issue: {str(e)}")


@router.put("/issues/{issue_id}/resolve")
async def resolve_quality_issue(
    issue_id: UUID, 
    resolution_notes: Optional[str] = None
) -> Dict[str, str]:
    """Mark a quality issue as resolved."""
    try:
        quality_manager = get_quality_manager_instance()
        success = await quality_manager.resolve_quality_issue(issue_id, resolution_notes)
        
        if success:
            return {"message": f"Quality issue {issue_id} resolved"}
        else:
            raise HTTPException(status_code=404, detail="Quality issue not found")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to resolve quality issue: {str(e)}")


# Data Repair Endpoints

@router.post("/repairs")
async def create_repair_request(request: RepairRequest, requested_by: str) -> Dict[str, Any]:
    """Create a new data repair request."""
    try:
        repair_service = get_repair_service_instance()
        
        # Convert string to enum
        repair_type = RepairType(request.repair_type)
        
        repair_record = await repair_service.create_repair_request(
            quality_issue_id=request.quality_issue_id,
            repair_type=repair_type,
            description=request.description,
            original_data=request.original_data,
            proposed_data=request.proposed_data,
            requested_by=requested_by,
            confidence=request.confidence
        )
        
        return repair_record.to_dict()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create repair request: {str(e)}")


@router.get("/repairs")
async def get_repair_history(
    quality_issue_id: Optional[UUID] = Query(None, description="Filter by quality issue ID"),
    repair_type: Optional[str] = Query(None, description="Filter by repair type"),
    status: Optional[str] = Query(None, description="Filter by status")
) -> List[Dict[str, Any]]:
    """Get repair history with optional filters."""
    try:
        repair_service = get_repair_service_instance()
        
        # Convert strings to enums if provided
        repair_type_enum = RepairType(repair_type) if repair_type else None
        status_enum = RepairStatus(status) if status else None
        
        repairs = repair_service.get_repair_history(
            quality_issue_id=quality_issue_id,
            repair_type=repair_type_enum,
            status=status_enum
        )
        
        return [repair.to_dict() for repair in repairs]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get repair history: {str(e)}")


@router.put("/repairs/{repair_id}/approve")
async def approve_repair(
    repair_id: UUID, 
    request: RepairApprovalRequest,
    approved_by: str
) -> Dict[str, str]:
    """Approve or reject a repair request."""
    try:
        repair_service = get_repair_service_instance()
        
        if request.approved:
            success = await repair_service.approve_repair(repair_id, approved_by, request.notes)
            action = "approved"
        else:
            success = await repair_service.reject_repair(
                repair_id, 
                approved_by, 
                request.notes or "No reason provided"
            )
            action = "rejected"
        
        if success:
            return {"message": f"Repair {repair_id} {action} successfully"}
        else:
            raise HTTPException(status_code=404, detail="Repair request not found")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process repair approval: {str(e)}")


@router.post("/repairs/{repair_id}/execute")
async def execute_repair(repair_id: UUID, executed_by: str) -> Dict[str, Any]:
    """Execute an approved repair."""
    try:
        repair_service = get_repair_service_instance()
        success = await repair_service.execute_repair(repair_id, executed_by)
        
        if success:
            # Get verification result
            verification = await repair_service.verify_repair_result(repair_id)
            
            return {
                "message": f"Repair {repair_id} executed successfully",
                "verification": verification
            }
        else:
            raise HTTPException(status_code=500, detail="Repair execution failed")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to execute repair: {str(e)}")


@router.get("/repairs/{repair_id}/verify")
async def verify_repair(repair_id: UUID) -> Dict[str, Any]:
    """Verify the result of a repair operation."""
    try:
        repair_service = get_repair_service_instance()
        verification_result = await repair_service.verify_repair_result(repair_id)
        return verification_result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to verify repair: {str(e)}")


@router.get("/repairs/statistics")
async def get_repair_statistics() -> Dict[str, Any]:
    """Get statistics about repair operations."""
    try:
        repair_service = get_repair_service_instance()
        stats = repair_service.get_repair_statistics()
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get repair statistics: {str(e)}")


# Quality Dashboard Endpoints

@router.get("/dashboard/summary")
async def get_quality_dashboard_summary() -> Dict[str, Any]:
    """Get quality management dashboard summary."""
    try:
        quality_manager = get_quality_manager_instance()
        repair_service = get_repair_service_instance()
        
        # Import IssueStatus for filtering
        from src.models.quality_issue import IssueStatus
        
        # Get quality issues summary
        all_issues = await quality_manager.get_quality_issues()
        
        issue_summary = {
            "total_issues": len(all_issues),
            "open_issues": len([i for i in all_issues if i.status == IssueStatus.OPEN]),
            "in_progress_issues": len([i for i in all_issues if i.status == IssueStatus.IN_PROGRESS]),
            "resolved_issues": len([i for i in all_issues if i.status == IssueStatus.RESOLVED])
        }
        
        # Get repair statistics
        repair_stats = repair_service.get_repair_statistics()
        
        # Get active quality rules
        active_rules = len([r for r in quality_manager.quality_rules.values() if r.enabled])
        total_rules = len(quality_manager.quality_rules)
        
        return {
            "quality_issues": issue_summary,
            "repairs": repair_stats,
            "quality_rules": {
                "active_rules": active_rules,
                "total_rules": total_rules
            },
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dashboard summary: {str(e)}")


@router.get("/health")
async def quality_service_health() -> Dict[str, Any]:
    """Health check for quality management service."""
    return {
        "status": "healthy",
        "service": "quality_management",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "quality_manager": "operational",
            "repair_service": "operational",
            "ragas_integration": "operational"
        }
    }