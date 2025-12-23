"""
Database management utilities for SuperInsight Platform.

Provides high-level database operations and management functions with query optimization.
"""

import logging
from typing import List, Dict, Any, Optional
from uuid import UUID
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import text, func, select

from src.database.connection import db_manager
from src.database.models import DocumentModel, TaskModel, BillingRecordModel, QualityIssueModel
from src.database.query_optimizer import query_optimizer, optimized_queries, index_manager
from src.database.cache_service import get_cache_service, get_query_cache, get_cache_invalidation

logger = logging.getLogger(__name__)


class DatabaseManager:
    """High-level database management class with query optimization."""
    
    def __init__(self):
        self.db_manager = db_manager
        self.cache_service = get_cache_service()
        self.query_cache = get_query_cache()
        self.cache_invalidation = get_cache_invalidation()
        self.optimizer = query_optimizer
        self.optimized_queries = optimized_queries
        self.index_manager = index_manager
    
    def get_session(self) -> Session:
        """Get a database session."""
        return self.db_manager.get_session()
    
    def test_connection(self) -> bool:
        """Test database connectivity."""
        return self.db_manager.test_connection()
    
    # Document operations
    def create_document(self, source_type: str, source_config: Dict[str, Any], 
                       content: str, metadata: Optional[Dict[str, Any]] = None) -> DocumentModel:
        """
        Create a new document record.
        
        Args:
            source_type: Type of data source
            source_config: Source configuration
            content: Document content
            metadata: Optional metadata
            
        Returns:
            Created document model
        """
        with self.get_session() as session:
            document = DocumentModel(
                source_type=source_type,
                source_config=source_config,
                content=content,
                document_metadata=metadata or {}
            )
            session.add(document)
            session.flush()
            session.refresh(document)
            return document
    
    def get_document(self, document_id: UUID) -> Optional[DocumentModel]:
        """Get a document by ID."""
        with self.get_session() as session:
            stmt = select(DocumentModel).where(DocumentModel.id == document_id)
            return session.execute(stmt).scalar_one_or_none()
    
    def list_documents(self, limit: int = 100, offset: int = 0) -> List[DocumentModel]:
        """List documents with pagination."""
        with self.get_session() as session:
            stmt = select(DocumentModel).offset(offset).limit(limit)
            return list(session.execute(stmt).scalars().all())
    
    def search_documents_by_metadata(self, metadata_query: Dict[str, Any]) -> List[DocumentModel]:
        """
        Search documents by metadata using optimized JSONB queries.
        
        Args:
            metadata_query: Dictionary of metadata key-value pairs to search for
            
        Returns:
            List of matching documents
        """
        # Generate cache key for this query
        cache_key = f"documents:metadata:{hash(str(sorted(metadata_query.items())))}"
        
        # Try to get from cache first
        cached_result = self.query_cache.get_query_result(cache_key)
        if cached_result is not None:
            return cached_result
        
        with self.optimizer.monitor_query("search_documents_by_metadata"):
            with self.get_session() as session:
                # Use optimized query with proper indexing
                stmt = select(DocumentModel)
                
                for key, value in metadata_query.items():
                    # Use JSONB containment operator for better performance
                    if isinstance(value, dict):
                        stmt = stmt.where(DocumentModel.document_metadata[key].contains(value))
                    else:
                        stmt = stmt.where(DocumentModel.document_metadata[key].astext == str(value))
                
                results = session.execute(stmt).scalars().all()
                
                # Cache the results for 30 minutes
                self.query_cache.set_query_result(cache_key, results, ttl=1800)
                
                return results
    
    # Task operations
    def create_task(self, document_id: UUID, project_id: str) -> TaskModel:
        """Create a new annotation task."""
        with self.get_session() as session:
            task = TaskModel(
                document_id=document_id,
                project_id=project_id
            )
            session.add(task)
            session.flush()
            session.refresh(task)
            return task
    
    def get_task(self, task_id: UUID) -> Optional[TaskModel]:
        """Get a task by ID."""
        with self.get_session() as session:
            stmt = select(TaskModel).where(TaskModel.id == task_id)
            return session.execute(stmt).scalar_one_or_none()
    
    def list_tasks_by_project(self, project_id: str, limit: int = 100, offset: int = 0) -> List[TaskModel]:
        """List all tasks for a project with pagination and caching."""
        cache_key = f"tasks:project:{project_id}:limit:{limit}:offset:{offset}"
        
        # Try cache first
        cached_result = self.query_cache.get_query_result(cache_key)
        if cached_result is not None:
            return cached_result
        
        with self.optimizer.monitor_query("list_tasks_by_project"):
            with self.get_session() as session:
                stmt = (
                    select(TaskModel)
                    .where(TaskModel.project_id == project_id)
                    .order_by(TaskModel.created_at.desc())
                    .offset(offset)
                    .limit(limit)
                )
                
                results = session.execute(stmt).scalars().all()
                
                # Cache for 10 minutes
                self.query_cache.set_query_result(cache_key, results, ttl=600)
                
                return results
    
    def update_task_annotations(self, task_id: UUID, annotations: List[Dict[str, Any]]) -> bool:
        """Update task annotations and invalidate related cache."""
        try:
            with self.get_session() as session:
                stmt = select(TaskModel).where(TaskModel.id == task_id)
                task = session.execute(stmt).scalar_one_or_none()
                if task:
                    task.annotations = annotations
                    
                    # Invalidate related cache entries
                    self.cache_invalidation.invalidate_for_table('tasks')
                    
                    return True
                return False
        except Exception as e:
            logger.error(f"Failed to update task annotations: {e}")
            return False
    
    def update_task_quality_score(self, task_id: UUID, quality_score: float) -> bool:
        """Update task quality score and invalidate related cache."""
        try:
            with self.get_session() as session:
                stmt = select(TaskModel).where(TaskModel.id == task_id)
                task = session.execute(stmt).scalar_one_or_none()
                if task:
                    task.quality_score = quality_score
                    
                    # Invalidate related cache entries
                    self.cache_invalidation.invalidate_for_table('tasks')
                    
                    return True
                return False
        except Exception as e:
            logger.error(f"Failed to update task quality score: {e}")
            return False
    
    def get_tasks_by_document_id(self, document_id: UUID) -> List[TaskModel]:
        """Get all tasks for a specific document."""
        with self.get_session() as session:
            stmt = select(TaskModel).where(TaskModel.document_id == document_id)
            return list(session.execute(stmt).scalars().all())
    
    def get_task_by_id(self, task_id: UUID) -> Optional[TaskModel]:
        """Get a task by ID (alias for get_task)."""
        return self.get_task(task_id)
    
    # Billing operations
    def create_billing_record(self, tenant_id: str, user_id: str, task_id: Optional[UUID] = None,
                            annotation_count: int = 0, time_spent: int = 0, cost: float = 0.0) -> BillingRecordModel:
        """Create a billing record."""
        with self.get_session() as session:
            billing_record = BillingRecordModel(
                tenant_id=tenant_id,
                user_id=user_id,
                task_id=task_id,
                annotation_count=annotation_count,
                time_spent=time_spent,
                cost=cost
            )
            session.add(billing_record)
            session.flush()
            session.refresh(billing_record)
            return billing_record
    
    def get_billing_records_by_tenant(self, tenant_id: str, 
                                    start_date: Optional[datetime] = None,
                                    end_date: Optional[datetime] = None,
                                    limit: int = 1000) -> List[BillingRecordModel]:
        """Get billing records for a tenant within a date range with caching."""
        cache_key = f"billing:tenant:{tenant_id}:start:{start_date}:end:{end_date}:limit:{limit}"
        
        # Try cache first
        cached_result = self.query_cache.get_query_result(cache_key)
        if cached_result is not None:
            return cached_result
        
        with self.optimizer.monitor_query("get_billing_records_by_tenant"):
            with self.get_session() as session:
                stmt = select(BillingRecordModel).where(BillingRecordModel.tenant_id == tenant_id)
                
                if start_date:
                    stmt = stmt.where(BillingRecordModel.billing_date >= start_date)
                if end_date:
                    stmt = stmt.where(BillingRecordModel.billing_date <= end_date)
                
                stmt = stmt.order_by(BillingRecordModel.billing_date.desc()).limit(limit)
                
                results = session.execute(stmt).scalars().all()
                
                # Cache for 30 minutes
                self.query_cache.set_query_result(cache_key, results, ttl=1800)
                
                return results
    
    def calculate_tenant_costs(self, tenant_id: str, 
                             start_date: Optional[datetime] = None,
                             end_date: Optional[datetime] = None) -> Dict[str, float]:
        """Calculate total costs for a tenant."""
        with self.get_session() as session:
            query = session.query(
                func.sum(BillingRecordModel.cost).label('total_cost'),
                func.sum(BillingRecordModel.annotation_count).label('total_annotations'),
                func.sum(BillingRecordModel.time_spent).label('total_time')
            ).filter(BillingRecordModel.tenant_id == tenant_id)
            
            if start_date:
                query = query.filter(BillingRecordModel.billing_date >= start_date)
            if end_date:
                query = query.filter(BillingRecordModel.billing_date <= end_date)
            
            result = query.first()
            
            return {
                'total_cost': float(result.total_cost or 0),
                'total_annotations': int(result.total_annotations or 0),
                'total_time': int(result.total_time or 0)
            }
    
    # Quality issue operations
    def create_quality_issue(self, task_id: UUID, issue_type: str, 
                           description: Optional[str] = None) -> QualityIssueModel:
        """Create a quality issue."""
        with self.get_session() as session:
            quality_issue = QualityIssueModel(
                task_id=task_id,
                issue_type=issue_type,
                description=description
            )
            session.add(quality_issue)
            session.flush()
            session.refresh(quality_issue)
            return quality_issue
    
    def get_quality_issues_by_task(self, task_id: UUID) -> List[QualityIssueModel]:
        """Get all quality issues for a task."""
        with self.get_session() as session:
            stmt = select(QualityIssueModel).where(QualityIssueModel.task_id == task_id)
            return list(session.execute(stmt).scalars().all())
    
    def assign_quality_issue(self, issue_id: UUID, assignee_id: str) -> bool:
        """Assign a quality issue to a user."""
        try:
            with self.get_session() as session:
                stmt = select(QualityIssueModel).where(QualityIssueModel.id == issue_id)
                issue = session.execute(stmt).scalar_one_or_none()
                if issue:
                    issue.assignee_id = assignee_id
                    return True
                return False
        except Exception as e:
            logger.error(f"Failed to assign quality issue: {e}")
            return False
    
    # Database statistics and health checks
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics with caching."""
        cache_key = "system:database_stats"
        
        # Try cache first (5 minute cache)
        cached_result = self.query_cache.get_query_result(cache_key)
        if cached_result is not None:
            return cached_result
        
        with self.optimizer.monitor_query("get_database_stats"):
            with self.get_session() as session:
                stats = {}
                
                # Count records in each table
                stats['documents_count'] = session.execute(select(func.count(DocumentModel.id))).scalar()
                stats['tasks_count'] = session.execute(select(func.count(TaskModel.id))).scalar()
                stats['billing_records_count'] = session.execute(select(func.count(BillingRecordModel.id))).scalar()
                stats['quality_issues_count'] = session.execute(select(func.count(QualityIssueModel.id))).scalar()
                
                # Get recent activity (count of recent records)
                from datetime import datetime, timedelta
                one_day_ago = datetime.now() - timedelta(days=1)
                
                recent_docs_stmt = select(func.count(DocumentModel.id)).where(
                    DocumentModel.created_at > one_day_ago
                )
                stats['recent_documents'] = session.execute(recent_docs_stmt).scalar()
                
                recent_tasks_stmt = select(func.count(TaskModel.id)).where(
                    TaskModel.created_at > one_day_ago
                )
                stats['recent_tasks'] = session.execute(recent_tasks_stmt).scalar()
                
                # Cache for 5 minutes
                self.query_cache.set_query_result(cache_key, stats, ttl=300)
                
                return stats
    
    # New optimized query methods
    
    def get_project_summary(self, project_id: str) -> Dict[str, Any]:
        """Get comprehensive project summary with caching."""
        return self.optimized_queries.get_project_quality_metrics(project_id)
    
    def get_tenant_billing_summary(self, tenant_id: str, days: int = 30) -> Dict[str, Any]:
        """Get tenant billing summary with caching."""
        return self.optimized_queries.get_tenant_billing_summary(tenant_id, days)
    
    def get_user_productivity_metrics(self, tenant_id: str, days: int = 7) -> List[Dict[str, Any]]:
        """Get user productivity metrics with caching."""
        return self.optimized_queries.get_user_productivity_metrics(tenant_id, days)
    
    def get_system_performance_metrics(self) -> Dict[str, Any]:
        """Get system-wide performance metrics with caching."""
        return self.optimized_queries.get_system_performance_metrics()
    
    def get_documents_by_project_cached(self, project_id: str, limit: int = 100) -> List[DocumentModel]:
        """Get documents by project with caching."""
        return self.optimized_queries.get_documents_by_project(project_id, limit)
    
    def get_task_status_summary(self, project_id: Optional[str] = None) -> Dict[str, int]:
        """Get task status summary with caching."""
        return self.optimized_queries.get_task_status_summary(project_id)
    
    def get_recent_documents_cached(self, hours: int = 24, limit: int = 50) -> List[DocumentModel]:
        """Get recent documents with caching."""
        return self.optimized_queries.get_recent_documents(hours, limit)
    
    # Cache management methods
    
    def clear_cache(self) -> Dict[str, Any]:
        """Clear all database caches."""
        self.cache_service.clear()
        self.optimizer.clear_cache()
        
        return {
            "status": "success",
            "message": "All database caches cleared",
            "timestamp": datetime.now().isoformat()
        }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_service": self.cache_service.get_info(),
            "query_optimizer": self.optimizer.get_query_stats(),
            "query_cache": self.query_cache.get_query_stats()
        }
    
    def cleanup_expired_cache(self) -> Dict[str, Any]:
        """Clean up expired cache entries."""
        expired_count = self.cache_service.cleanup_expired()
        
        return {
            "expired_entries_removed": expired_count,
            "timestamp": datetime.now().isoformat()
        }
    
    # Index management methods
    
    def create_performance_indexes(self) -> Dict[str, Any]:
        """Create recommended performance indexes."""
        results = self.index_manager.create_recommended_indexes()
        
        return {
            "indexes_created": results,
            "total_indexes": len(results),
            "successful": sum(1 for success in results.values() if success),
            "failed": sum(1 for success in results.values() if not success)
        }
    
    def analyze_query_performance(self) -> List[Dict[str, Any]]:
        """Analyze query performance and get optimization suggestions."""
        return self.index_manager.analyze_query_performance()
    
    def check_database_health(self) -> Dict[str, Any]:
        """Perform comprehensive database health checks."""
        health_status = {
            'connection': False,
            'tables_exist': False,
            'indexes_exist': False,
            'cache_status': {},
            'query_performance': {},
            'errors': []
        }
        
        try:
            # Test connection
            health_status['connection'] = self.test_connection()
            
            # Check if tables exist
            with self.get_session() as session:
                try:
                    session.execute(select(DocumentModel).limit(1))
                    session.execute(select(TaskModel).limit(1))
                    session.execute(select(BillingRecordModel).limit(1))
                    session.execute(select(QualityIssueModel).limit(1))
                    health_status['tables_exist'] = True
                except Exception as e:
                    health_status['errors'].append(f"Tables check failed: {e}")
                
                # Check if GIN indexes exist
                try:
                    result = session.execute(text("""
                        SELECT COUNT(*) FROM pg_indexes 
                        WHERE indexname LIKE 'idx_%_gin'
                    """))
                    gin_count = result.scalar()
                    health_status['indexes_exist'] = gin_count > 0
                    health_status['gin_indexes_count'] = gin_count
                    
                    # Check for recommended indexes
                    recommended_indexes = self.index_manager.recommended_indexes
                    existing_indexes = []
                    for index_info in recommended_indexes:
                        check_sql = text(f"""
                            SELECT 1 FROM pg_indexes 
                            WHERE indexname = '{index_info['name']}'
                        """)
                        exists = session.execute(check_sql).scalar()
                        if exists:
                            existing_indexes.append(index_info['name'])
                    
                    health_status['recommended_indexes_count'] = len(existing_indexes)
                    health_status['missing_indexes'] = len(recommended_indexes) - len(existing_indexes)
                    
                except Exception as e:
                    health_status['errors'].append(f"Index check failed: {e}")
            
            # Check cache status
            try:
                health_status['cache_status'] = self.cache_service.get_info()
            except Exception as e:
                health_status['errors'].append(f"Cache check failed: {e}")
            
            # Check query performance
            try:
                health_status['query_performance'] = self.optimizer.get_query_stats()
            except Exception as e:
                health_status['errors'].append(f"Query performance check failed: {e}")
        
        except Exception as e:
            health_status['errors'].append(f"Health check failed: {e}")
        
        return health_status


# Global database manager instance
database_manager = DatabaseManager()