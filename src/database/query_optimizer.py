"""
Database query optimization and caching utilities.

This module provides query optimization, caching, and index management
for improved database performance.
"""

import logging
import time
from datetime import datetime, timedelta
from functools import wraps
from typing import Dict, List, Optional, Any, Callable
from sqlalchemy import select, func, text
from sqlalchemy.orm import Session

from src.database.models import DocumentModel, TaskModel, QualityIssueModel, BillingRecordModel
from src.database.connection import db_manager

logger = logging.getLogger(__name__)


class QueryCache:
    """Simple in-memory query cache."""
    
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.timestamps = {}
        self.max_size = max_size
    
    def get(self, key: str, ttl: int = 3600) -> Optional[Any]:
        """Get cached value if not expired."""
        if key not in self.cache:
            return None
        
        timestamp = self.timestamps.get(key, 0)
        if time.time() - timestamp > ttl:
            self.cache.pop(key, None)
            self.timestamps.pop(key, None)
            return None
        
        return self.cache[key]
    
    def set(self, key: str, value: Any) -> None:
        """Set cached value."""
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(self.timestamps.keys(), key=lambda k: self.timestamps[k])
            self.cache.pop(oldest_key, None)
            self.timestamps.pop(oldest_key, None)
        
        self.cache[key] = value
        self.timestamps[key] = time.time()
    
    def clear(self) -> None:
        """Clear all cached values."""
        self.cache.clear()
        self.timestamps.clear()


class QueryOptimizer:
    """Database query optimizer with caching and performance monitoring."""
    
    def __init__(self):
        self.cache = QueryCache()
        self.query_stats = {}
        self.slow_query_threshold = 1.0  # seconds
    
    def cached_query(self, ttl: int = 3600):
        """Decorator for caching query results."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Create cache key from function name and arguments
                cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
                
                # Try to get from cache
                cached_result = self.cache.get(cache_key, ttl)
                if cached_result is not None:
                    return cached_result
                
                # Execute query and cache result
                start_time = time.time()
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Track query performance
                self._track_query_performance(func.__name__, execution_time)
                
                # Cache the result
                self.cache.set(cache_key, result)
                
                return result
            return wrapper
        return decorator
    
    def _track_query_performance(self, query_name: str, execution_time: float):
        """Track query performance statistics."""
        if query_name not in self.query_stats:
            self.query_stats[query_name] = {
                'count': 0,
                'total_time': 0.0,
                'avg_time': 0.0,
                'max_time': 0.0,
                'slow_queries': 0
            }
        
        stats = self.query_stats[query_name]
        stats['count'] += 1
        stats['total_time'] += execution_time
        stats['avg_time'] = stats['total_time'] / stats['count']
        stats['max_time'] = max(stats['max_time'], execution_time)
        
        if execution_time > self.slow_query_threshold:
            stats['slow_queries'] += 1
            logger.warning(f"Slow query detected: {query_name} took {execution_time:.2f}s")
    
    def get_query_stats(self) -> Dict[str, Any]:
        """Get query performance statistics."""
        return self.query_stats.copy()
    
    def monitor_query(self, query_name: str):
        """Context manager for monitoring query performance."""
        class QueryMonitor:
            def __init__(self, optimizer, name):
                self.optimizer = optimizer
                self.name = name
                self.start_time = None
            
            def __enter__(self):
                self.start_time = time.time()
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.start_time:
                    execution_time = time.time() - self.start_time
                    self.optimizer._track_query_performance(self.name, execution_time)
        
        return QueryMonitor(self, query_name)
    
    def clear_cache(self):
        """Clear query cache."""
        self.cache.clear()


class OptimizedQueries:
    """Collection of optimized database queries."""
    
    def __init__(self, optimizer: QueryOptimizer):
        self.optimizer = optimizer
    
    # Document queries
    
    def get_documents_by_project(self, project_id: str, limit: int = 100) -> List[DocumentModel]:
        """Get documents by project ID."""
        with db_manager.get_session() as db:
            stmt = (
                select(DocumentModel)
                .join(TaskModel)
                .where(TaskModel.project_id == project_id)
                .limit(limit)
            )
            return db.execute(stmt).scalars().all()
    
    def get_document_count_by_source_type(self) -> Dict[str, int]:
        """Get document count grouped by source type."""
        with db_manager.get_session() as db:
            stmt = (
                select(DocumentModel.source_type, func.count(DocumentModel.id))
                .group_by(DocumentModel.source_type)
            )
            results = db.execute(stmt).all()
            return {source_type: count for source_type, count in results}
    
    def get_recent_documents(self, hours: int = 24, limit: int = 50) -> List[DocumentModel]:
        """Get recently created documents."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with db_manager.get_session() as db:
            stmt = (
                select(DocumentModel)
                .where(DocumentModel.created_at >= cutoff_time)
                .order_by(DocumentModel.created_at.desc())
                .limit(limit)
            )
            return db.execute(stmt).scalars().all()
    
    # Task queries
    
    def get_task_status_summary(self, project_id: Optional[str] = None) -> Dict[str, int]:
        """Get task count by status."""
        with db_manager.get_session() as db:
            stmt = (
                select(TaskModel.status, func.count(TaskModel.id))
                .group_by(TaskModel.status)
            )
            
            if project_id:
                stmt = stmt.where(TaskModel.project_id == project_id)
            
            results = db.execute(stmt).all()
            return {status.value if hasattr(status, 'value') else str(status): count 
                    for status, count in results}
    
    def get_tasks_with_quality_issues(self, project_id: Optional[str] = None) -> List[TaskModel]:
        """Get tasks that have quality issues."""
        with db_manager.get_session() as db:
            stmt = (
                select(TaskModel)
                .join(QualityIssueModel)
                .where(QualityIssueModel.status == 'open')
            )
            
            if project_id:
                stmt = stmt.where(TaskModel.project_id == project_id)
            
            return db.execute(stmt).scalars().all()
    
    def get_project_quality_metrics(self, project_id: str) -> Dict[str, Any]:
        """Get quality metrics for a project."""
        with db_manager.get_session() as db:
            # Get task counts
            task_count_stmt = (
                select(func.count(TaskModel.id))
                .where(TaskModel.project_id == project_id)
            )
            total_tasks = db.execute(task_count_stmt).scalar() or 0
            
            # Get average quality score
            avg_quality_stmt = (
                select(func.avg(TaskModel.quality_score))
                .where(TaskModel.project_id == project_id)
            )
            avg_quality = db.execute(avg_quality_stmt).scalar() or 0.0
            
            # Get quality issues count
            issues_count_stmt = (
                select(func.count(QualityIssueModel.id))
                .join(TaskModel)
                .where(TaskModel.project_id == project_id)
                .where(QualityIssueModel.status == 'open')
            )
            open_issues = db.execute(issues_count_stmt).scalar() or 0
            
            return {
                'total_tasks': total_tasks,
                'average_quality_score': float(avg_quality),
                'open_quality_issues': open_issues,
                'quality_ratio': (total_tasks - open_issues) / max(total_tasks, 1)
            }
    
    # Billing queries
    
    def get_tenant_billing_summary(self, tenant_id: str, days: int = 30) -> Dict[str, Any]:
        """Get billing summary for tenant."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        with db_manager.get_session() as db:
            stmt = (
                select(
                    func.sum(BillingRecordModel.annotation_count),
                    func.sum(BillingRecordModel.time_spent),
                    func.sum(BillingRecordModel.cost)
                )
                .where(BillingRecordModel.tenant_id == tenant_id)
                .where(BillingRecordModel.billing_date >= cutoff_date.date())
            )
            
            result = db.execute(stmt).first()
            total_annotations, total_time, total_cost = result if result else (0, 0, 0.0)
            
            return {
                'tenant_id': tenant_id,
                'period_days': days,
                'total_annotations': total_annotations or 0,
                'total_time_spent': total_time or 0,
                'total_cost': float(total_cost or 0.0),
                'average_cost_per_annotation': float(total_cost or 0.0) / max(total_annotations or 1, 1)
            }
    
    def get_user_productivity_metrics(self, tenant_id: str, days: int = 7) -> List[Dict[str, Any]]:
        """Get user productivity metrics."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        with db_manager.get_session() as db:
            stmt = (
                select(
                    BillingRecordModel.user_id,
                    func.sum(BillingRecordModel.annotation_count),
                    func.sum(BillingRecordModel.time_spent),
                    func.avg(BillingRecordModel.annotation_count / func.greatest(BillingRecordModel.time_spent, 1))
                )
                .where(BillingRecordModel.tenant_id == tenant_id)
                .where(BillingRecordModel.billing_date >= cutoff_date.date())
                .group_by(BillingRecordModel.user_id)
            )
            
            results = db.execute(stmt).all()
            
            return [
                {
                    'user_id': user_id,
                    'total_annotations': total_annotations or 0,
                    'total_time_spent': total_time or 0,
                    'annotations_per_hour': float(avg_rate or 0.0) * 3600  # Convert to per hour
                }
                for user_id, total_annotations, total_time, avg_rate in results
            ]
    
    # Complex analytical queries
    
    def get_system_performance_metrics(self) -> Dict[str, Any]:
        """Get overall system performance metrics."""
        with db_manager.get_session() as db:
            # Get total counts
            total_documents = db.execute(select(func.count(DocumentModel.id))).scalar() or 0
            total_tasks = db.execute(select(func.count(TaskModel.id))).scalar() or 0
            
            # Get recent activity (last 24 hours)
            yesterday = datetime.now() - timedelta(days=1)
            recent_documents = db.execute(
                select(func.count(DocumentModel.id))
                .where(DocumentModel.created_at >= yesterday)
            ).scalar() or 0
            
            recent_tasks = db.execute(
                select(func.count(TaskModel.id))
                .where(TaskModel.created_at >= yesterday)
            ).scalar() or 0
            
            return {
                'total_documents': total_documents,
                'total_tasks': total_tasks,
                'recent_documents_24h': recent_documents,
                'recent_tasks_24h': recent_tasks,
                'system_utilization': {
                    'documents_per_day': recent_documents,
                    'tasks_per_day': recent_tasks
                }
            }


class IndexManager:
    """Database index management for performance optimization."""
    
    def __init__(self):
        self.recommended_indexes = [
            # Document indexes
            "CREATE INDEX IF NOT EXISTS idx_documents_source_type ON documents(source_type)",
            "CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_documents_metadata_gin ON documents USING GIN (metadata)",
            
            # Task indexes
            "CREATE INDEX IF NOT EXISTS idx_tasks_project_id ON tasks(project_id)",
            "CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)",
            "CREATE INDEX IF NOT EXISTS idx_tasks_created_at ON tasks(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_tasks_quality_score ON tasks(quality_score)",
            
            # Billing indexes
            "CREATE INDEX IF NOT EXISTS idx_billing_tenant_id ON billing_records(tenant_id)",
            "CREATE INDEX IF NOT EXISTS idx_billing_user_id ON billing_records(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_billing_date ON billing_records(billing_date)",
            
            # Quality issue indexes
            "CREATE INDEX IF NOT EXISTS idx_quality_issues_task_id ON quality_issues(task_id)",
            "CREATE INDEX IF NOT EXISTS idx_quality_issues_status ON quality_issues(status)",
        ]
    
    def create_recommended_indexes(self):
        """Create all recommended indexes."""
        with db_manager.get_session() as db:
            for index_sql in self.recommended_indexes:
                try:
                    db.execute(text(index_sql))
                    logger.info(f"Created index: {index_sql}")
                except Exception as e:
                    logger.warning(f"Failed to create index: {index_sql}, error: {e}")
            
            db.commit()
    
    def analyze_query_performance(self) -> Dict[str, Any]:
        """Analyze query performance and suggest optimizations."""
        with db_manager.get_session() as db:
            # Get slow queries from PostgreSQL stats
            slow_queries_sql = """
            SELECT query, calls, total_time, mean_time
            FROM pg_stat_statements
            WHERE mean_time > 1000  -- queries taking more than 1 second on average
            ORDER BY mean_time DESC
            LIMIT 10
            """
            
            try:
                result = db.execute(text(slow_queries_sql)).fetchall()
                return {
                    'slow_queries': [
                        {
                            'query': row[0][:100] + '...' if len(row[0]) > 100 else row[0],
                            'calls': row[1],
                            'total_time': row[2],
                            'mean_time': row[3]
                        }
                        for row in result
                    ]
                }
            except Exception as e:
                logger.warning(f"Could not analyze query performance: {e}")
                return {'slow_queries': []}


# Global instances
query_optimizer = QueryOptimizer()
optimized_queries = OptimizedQueries(query_optimizer)
index_manager = IndexManager()