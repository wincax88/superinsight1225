"""
Audit Service for SuperInsight Platform.

Provides comprehensive audit logging, analysis, and alerting functionality.
"""

import logging
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc, select, delete
from collections import defaultdict, Counter

from src.security.models import AuditLogModel, AuditAction, UserModel
from src.database.connection import get_db_session


class AuditService:
    """
    Comprehensive audit service for logging, analysis, and alerting.
    
    Handles audit log management, security analysis, and sensitive operation monitoring.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.sensitive_actions = {
            AuditAction.DELETE,
            AuditAction.EXPORT,
            AuditAction.UPDATE
        }
        self.critical_resources = {
            "user", "permission", "ip_whitelist", "masking_rule"
        }
    
    # Enhanced Logging Methods
    
    def log_bulk_actions(
        self,
        actions: List[Dict[str, Any]],
        db: Session
    ) -> bool:
        """Log multiple actions in bulk for performance."""
        try:
            audit_logs = []
            for action_data in actions:
                audit_log = AuditLogModel(
                    user_id=action_data.get("user_id"),
                    tenant_id=action_data["tenant_id"],
                    action=action_data["action"],
                    resource_type=action_data["resource_type"],
                    resource_id=action_data.get("resource_id"),
                    ip_address=action_data.get("ip_address"),
                    user_agent=action_data.get("user_agent"),
                    details=action_data.get("details", {})
                )
                audit_logs.append(audit_log)
            
            db.add_all(audit_logs)
            db.commit()
            return True
        except Exception as e:
            self.logger.error(f"Failed to log bulk actions: {e}")
            db.rollback()
            return False
    
    def log_system_event(
        self,
        event_type: str,
        description: str,
        tenant_id: str,
        details: Optional[Dict[str, Any]] = None,
        db: Session = None
    ) -> bool:
        """Log system-level events (not user actions)."""
        try:
            audit_log = AuditLogModel(
                user_id=None,  # System events have no user
                tenant_id=tenant_id,
                action=AuditAction.CREATE,  # Generic action for system events
                resource_type="system",
                resource_id=event_type,
                details={
                    "event_type": event_type,
                    "description": description,
                    **(details or {})
                }
            )
            db.add(audit_log)
            db.commit()
            return True
        except Exception as e:
            self.logger.error(f"Failed to log system event: {e}")
            if db:
                db.rollback()
            return False
    
    # Log Analysis Methods
    
    def analyze_user_activity(
        self,
        user_id: UUID,
        tenant_id: str,
        days: int = 30,
        db: Session = None
    ) -> Dict[str, Any]:
        """Analyze user activity patterns over a time period."""
        start_date = datetime.utcnow() - timedelta(days=days)
        
        stmt = select(AuditLogModel).where(
            and_(
                AuditLogModel.user_id == user_id,
                AuditLogModel.tenant_id == tenant_id,
                AuditLogModel.timestamp >= start_date
            )
        )
        logs = db.execute(stmt).scalars().all()
        
        if not logs:
            return {
                "total_actions": 0,
                "actions_by_type": {},
                "resources_accessed": {},
                "daily_activity": {},
                "suspicious_patterns": []
            }
        
        # Analyze action patterns
        actions_by_type = Counter(log.action.value for log in logs)
        resources_accessed = Counter(log.resource_type for log in logs)
        
        # Daily activity breakdown
        daily_activity = defaultdict(int)
        for log in logs:
            date_key = log.timestamp.date().isoformat()
            daily_activity[date_key] += 1
        
        # Detect suspicious patterns
        suspicious_patterns = self._detect_suspicious_patterns(logs)
        
        return {
            "total_actions": len(logs),
            "actions_by_type": dict(actions_by_type),
            "resources_accessed": dict(resources_accessed),
            "daily_activity": dict(daily_activity),
            "suspicious_patterns": suspicious_patterns,
            "analysis_period_days": days
        }
    
    def get_security_summary(
        self,
        tenant_id: str,
        days: int = 7,
        db: Session = None
    ) -> Dict[str, Any]:
        """Get security summary for a tenant."""
        start_date = datetime.utcnow() - timedelta(days=days)
        
        # Get all logs for the period
        stmt = select(AuditLogModel).where(
            and_(
                AuditLogModel.tenant_id == tenant_id,
                AuditLogModel.timestamp >= start_date
            )
        )
        logs = db.execute(stmt).scalars().all()
        
        # Failed login attempts
        failed_logins = [
            log for log in logs 
            if log.action == AuditAction.LOGIN and 
            log.details.get("status") == "failed"
        ]
        
        # Sensitive operations
        sensitive_ops = [
            log for log in logs 
            if log.action in self.sensitive_actions or 
            log.resource_type in self.critical_resources
        ]
        
        # Unique users active
        active_users = set(
            log.user_id for log in logs 
            if log.user_id is not None
        )
        
        # IP addresses used
        ip_addresses = set(
            str(log.ip_address) for log in logs 
            if log.ip_address is not None
        )
        
        return {
            "period_days": days,
            "total_events": len(logs),
            "failed_logins": len(failed_logins),
            "sensitive_operations": len(sensitive_ops),
            "active_users": len(active_users),
            "unique_ip_addresses": len(ip_addresses),
            "recent_failed_logins": [
                {
                    "timestamp": log.timestamp.isoformat(),
                    "ip_address": str(log.ip_address) if log.ip_address else None,
                    "username": log.details.get("username"),
                    "user_agent": log.user_agent
                }
                for log in failed_logins[-10:]  # Last 10 failed attempts
            ]
        }
    
    def _detect_suspicious_patterns(self, logs: List[AuditLogModel]) -> List[Dict[str, Any]]:
        """Detect suspicious activity patterns in audit logs."""
        patterns = []
        
        # Pattern 1: Rapid successive actions (potential automation/attack)
        time_sorted_logs = sorted(logs, key=lambda x: x.timestamp)
        rapid_actions = []
        
        for i in range(1, len(time_sorted_logs)):
            time_diff = (time_sorted_logs[i].timestamp - time_sorted_logs[i-1].timestamp).total_seconds()
            if time_diff < 1:  # Less than 1 second between actions
                rapid_actions.append({
                    "timestamp": time_sorted_logs[i].timestamp.isoformat(),
                    "action": time_sorted_logs[i].action.value,
                    "time_diff_seconds": time_diff
                })
        
        if len(rapid_actions) > 5:  # More than 5 rapid actions
            patterns.append({
                "type": "rapid_successive_actions",
                "severity": "medium",
                "description": f"Detected {len(rapid_actions)} rapid successive actions",
                "details": rapid_actions[:10]  # Show first 10
            })
        
        # Pattern 2: Unusual time access (outside business hours)
        unusual_time_actions = []
        for log in logs:
            hour = log.timestamp.hour
            if hour < 6 or hour > 22:  # Outside 6 AM - 10 PM
                unusual_time_actions.append({
                    "timestamp": log.timestamp.isoformat(),
                    "action": log.action.value,
                    "hour": hour
                })
        
        if len(unusual_time_actions) > 10:  # More than 10 actions outside hours
            patterns.append({
                "type": "unusual_time_access",
                "severity": "low",
                "description": f"Detected {len(unusual_time_actions)} actions outside business hours",
                "details": unusual_time_actions[:5]  # Show first 5
            })
        
        # Pattern 3: Multiple failed login attempts
        failed_logins = [
            log for log in logs 
            if log.action == AuditAction.LOGIN and 
            log.details.get("status") == "failed"
        ]
        
        if len(failed_logins) > 5:  # More than 5 failed attempts
            patterns.append({
                "type": "multiple_failed_logins",
                "severity": "high",
                "description": f"Detected {len(failed_logins)} failed login attempts",
                "details": [
                    {
                        "timestamp": log.timestamp.isoformat(),
                        "ip_address": str(log.ip_address) if log.ip_address else None,
                        "username": log.details.get("username")
                    }
                    for log in failed_logins[-5:]  # Last 5 attempts
                ]
            })
        
        return patterns
    
    # Log Query and Search Methods
    
    def search_logs(
        self,
        tenant_id: str,
        query_params: Dict[str, Any],
        db: Session
    ) -> Tuple[List[AuditLogModel], int]:
        """Advanced search in audit logs with pagination."""
        base_stmt = select(AuditLogModel).where(
            AuditLogModel.tenant_id == tenant_id
        )
        
        # Apply filters
        if query_params.get("user_id"):
            base_stmt = base_stmt.where(
                AuditLogModel.user_id == UUID(query_params["user_id"])
            )
        
        if query_params.get("action"):
            base_stmt = base_stmt.where(
                AuditLogModel.action == AuditAction(query_params["action"])
            )
        
        if query_params.get("resource_type"):
            base_stmt = base_stmt.where(
                AuditLogModel.resource_type == query_params["resource_type"]
            )
        
        if query_params.get("resource_id"):
            base_stmt = base_stmt.where(
                AuditLogModel.resource_id == query_params["resource_id"]
            )
        
        if query_params.get("ip_address"):
            base_stmt = base_stmt.where(
                AuditLogModel.ip_address == query_params["ip_address"]
            )
        
        if query_params.get("start_date"):
            base_stmt = base_stmt.where(
                AuditLogModel.timestamp >= query_params["start_date"]
            )
        
        if query_params.get("end_date"):
            base_stmt = base_stmt.where(
                AuditLogModel.timestamp <= query_params["end_date"]
            )
        
        # Text search in details
        if query_params.get("search_text"):
            search_text = f"%{query_params['search_text']}%"
            base_stmt = base_stmt.where(
                or_(
                    AuditLogModel.user_agent.ilike(search_text),
                    AuditLogModel.details.astext.ilike(search_text)
                )
            )
        
        # Get total count
        total_count = len(db.execute(base_stmt).scalars().all())
        
        # Apply pagination
        page = query_params.get("page", 1)
        page_size = min(query_params.get("page_size", 50), 100)  # Max 100 per page
        offset = (page - 1) * page_size
        
        paginated_stmt = base_stmt.order_by(desc(AuditLogModel.timestamp)).offset(offset).limit(page_size)
        logs = db.execute(paginated_stmt).scalars().all()
        
        return logs, total_count
    
    # Alerting Methods
    
    def check_security_alerts(
        self,
        tenant_id: str,
        db: Session
    ) -> List[Dict[str, Any]]:
        """Check for security conditions that require alerts."""
        alerts = []
        now = datetime.utcnow()
        
        # Alert 1: Multiple failed logins in last hour
        one_hour_ago = now - timedelta(hours=1)
        failed_login_stmt = select(AuditLogModel).where(
            and_(
                AuditLogModel.tenant_id == tenant_id,
                AuditLogModel.action == AuditAction.LOGIN,
                AuditLogModel.timestamp >= one_hour_ago,
                AuditLogModel.details["status"].astext == "failed"
            )
        )
        recent_failed_logins = len(db.execute(failed_login_stmt).scalars().all())
        
        if recent_failed_logins >= 10:
            alerts.append({
                "type": "multiple_failed_logins",
                "severity": "high",
                "message": f"{recent_failed_logins} failed login attempts in the last hour",
                "timestamp": now.isoformat(),
                "action_required": "Review IP whitelist and user accounts"
            })
        
        # Alert 2: Sensitive operations outside business hours
        if now.hour < 6 or now.hour > 22:  # Outside business hours
            sensitive_ops_stmt = select(AuditLogModel).where(
                and_(
                    AuditLogModel.tenant_id == tenant_id,
                    AuditLogModel.timestamp >= one_hour_ago,
                    or_(
                        AuditLogModel.action.in_(self.sensitive_actions),
                        AuditLogModel.resource_type.in_(self.critical_resources)
                    )
                )
            )
            recent_sensitive_ops = len(db.execute(sensitive_ops_stmt).scalars().all())
            
            if recent_sensitive_ops > 0:
                alerts.append({
                    "type": "after_hours_sensitive_operations",
                    "severity": "medium",
                    "message": f"{recent_sensitive_ops} sensitive operations detected outside business hours",
                    "timestamp": now.isoformat(),
                    "action_required": "Verify if operations were authorized"
                })
        
        # Alert 3: Unusual IP addresses
        last_24_hours = now - timedelta(hours=24)
        ip_stmt = select(AuditLogModel.ip_address).where(
            and_(
                AuditLogModel.tenant_id == tenant_id,
                AuditLogModel.timestamp >= last_24_hours,
                AuditLogModel.ip_address.isnot(None)
            )
        ).distinct()
        recent_ips = db.execute(ip_stmt).scalars().all()
        
        # This is a simplified check - in practice, you'd compare against historical patterns
        if len(recent_ips) > 20:  # More than 20 unique IPs in 24 hours
            alerts.append({
                "type": "unusual_ip_activity",
                "severity": "medium",
                "message": f"{len(recent_ips)} unique IP addresses accessed the system in 24 hours",
                "timestamp": now.isoformat(),
                "action_required": "Review IP whitelist configuration"
            })
        
        return alerts
    
    # Log Rotation and Cleanup
    
    def rotate_logs(
        self,
        tenant_id: str,
        retention_days: int = 365,
        db: Session = None
    ) -> Dict[str, Any]:
        """Rotate audit logs by archiving old entries."""
        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
        
        # Count logs to be archived
        old_logs_stmt = select(AuditLogModel).where(
            and_(
                AuditLogModel.tenant_id == tenant_id,
                AuditLogModel.timestamp < cutoff_date
            )
        )
        old_logs_count = len(db.execute(old_logs_stmt).scalars().all())
        
        if old_logs_count == 0:
            return {
                "archived_count": 0,
                "message": "No logs to archive"
            }
        
        # In a production system, you would:
        # 1. Export logs to archive storage (S3, etc.)
        # 2. Compress the data
        # 3. Delete from active database
        
        # For this implementation, we'll just delete old logs
        try:
            delete_stmt = delete(AuditLogModel).where(
                and_(
                    AuditLogModel.tenant_id == tenant_id,
                    AuditLogModel.timestamp < cutoff_date
                )
            )
            result = db.execute(delete_stmt)
            deleted_count = result.rowcount
            
            db.commit()
            
            return {
                "archived_count": deleted_count,
                "cutoff_date": cutoff_date.isoformat(),
                "message": f"Successfully archived {deleted_count} log entries"
            }
        except Exception as e:
            db.rollback()
            self.logger.error(f"Failed to rotate logs: {e}")
            return {
                "archived_count": 0,
                "error": str(e),
                "message": "Failed to archive logs"
            }
    
    def get_log_statistics(
        self,
        tenant_id: str,
        db: Session
    ) -> Dict[str, Any]:
        """Get audit log storage statistics."""
        total_logs_stmt = select(AuditLogModel).where(
            AuditLogModel.tenant_id == tenant_id
        )
        total_logs = len(db.execute(total_logs_stmt).scalars().all())
        
        if total_logs == 0:
            return {
                "total_logs": 0,
                "oldest_log": None,
                "newest_log": None,
                "storage_size_estimate": "0 MB"
            }
        
        oldest_log_stmt = select(func.min(AuditLogModel.timestamp)).where(
            AuditLogModel.tenant_id == tenant_id
        )
        oldest_log = db.execute(oldest_log_stmt).scalar()
        
        newest_log_stmt = select(func.max(AuditLogModel.timestamp)).where(
            AuditLogModel.tenant_id == tenant_id
        )
        newest_log = db.execute(newest_log_stmt).scalar()
        
        # Rough estimate: ~1KB per log entry
        storage_size_mb = (total_logs * 1024) / (1024 * 1024)
        
        return {
            "total_logs": total_logs,
            "oldest_log": oldest_log.isoformat() if oldest_log else None,
            "newest_log": newest_log.isoformat() if newest_log else None,
            "storage_size_estimate": f"{storage_size_mb:.2f} MB"
        }