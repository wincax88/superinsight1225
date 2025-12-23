"""
Optimized batch processing module for SuperInsight Platform.

Provides efficient batch processing with memory optimization, pagination, and parallel processing.
"""

import asyncio
import logging
import gc
import psutil
from typing import List, Dict, Any, Optional, Iterator, Callable, TypeVar, Generic, Union
from dataclasses import dataclass
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from sqlalchemy.orm import Session
from sqlalchemy import select, func, text
from sqlalchemy.sql import Select

from src.database.connection import db_manager
from src.database.models import DocumentModel, TaskModel, BillingRecordModel, QualityIssueModel

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class BatchConfig:
    """Configuration for batch processing operations."""
    batch_size: int = 1000
    max_memory_mb: int = 512
    max_workers: int = 4
    use_multiprocessing: bool = False
    enable_gc: bool = True
    gc_threshold: int = 100
    prefetch_size: int = 2
    chunk_size: int = 10000
    timeout_seconds: Optional[int] = None


@dataclass
class BatchStats:
    """Statistics for batch processing operations."""
    total_items: int = 0
    processed_items: int = 0
    failed_items: int = 0
    batches_processed: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    peak_memory_mb: float = 0.0
    avg_batch_time: float = 0.0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_items == 0:
            return 0.0
        return (self.processed_items / self.total_items) * 100
    
    @property
    def processing_time(self) -> float:
        """Calculate total processing time in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0


class MemoryMonitor:
    """Memory usage monitoring for batch operations."""
    
    def __init__(self, max_memory_mb: int = 512):
        """Initialize memory monitor."""
        self.max_memory_mb = max_memory_mb
        self.peak_memory_mb = 0.0
        self.process = psutil.Process()
    
    def get_current_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            self.peak_memory_mb = max(self.peak_memory_mb, memory_mb)
            return memory_mb
        except Exception:
            return 0.0
    
    def is_memory_limit_exceeded(self) -> bool:
        """Check if memory limit is exceeded."""
        current_memory = self.get_current_memory_mb()
        return current_memory > self.max_memory_mb
    
    def force_gc(self) -> None:
        """Force garbage collection."""
        gc.collect()
        logger.debug(f"Forced garbage collection. Current memory: {self.get_current_memory_mb():.1f} MB")


class OptimizedPaginator(Generic[T]):
    """Optimized paginator with cursor-based pagination for large datasets."""
    
    def __init__(self, session: Session, query: Select, config: BatchConfig):
        """Initialize paginator."""
        self.session = session
        self.query = query
        self.config = config
        self.last_id = None
        self.total_count = None
    
    def get_total_count(self) -> int:
        """Get total count of items (cached)."""
        if self.total_count is None:
            # Use a more efficient count query
            count_query = select(func.count()).select_from(self.query.subquery())
            self.total_count = self.session.execute(count_query).scalar() or 0
        return self.total_count
    
    def get_batches(self) -> Iterator[List[T]]:
        """Get batches using cursor-based pagination."""
        offset = 0
        
        while True:
            # Use cursor-based pagination for better performance on large datasets
            if hasattr(self.query.column_descriptions[0]['type'], 'id'):
                # If the model has an ID field, use cursor-based pagination
                batch_query = self.query
                if self.last_id:
                    batch_query = batch_query.where(
                        self.query.column_descriptions[0]['type'].id > self.last_id
                    )
                batch_query = batch_query.order_by(
                    self.query.column_descriptions[0]['type'].id
                ).limit(self.config.batch_size)
            else:
                # Fall back to offset-based pagination
                batch_query = self.query.offset(offset).limit(self.config.batch_size)
            
            batch = self.session.execute(batch_query).scalars().all()
            
            if not batch:
                break
            
            # Update cursor for next batch
            if hasattr(batch[-1], 'id'):
                self.last_id = batch[-1].id
            
            offset += len(batch)
            yield batch
            
            # Break if we got less than batch_size (last batch)
            if len(batch) < self.config.batch_size:
                break


class BatchProcessor:
    """Optimized batch processor with memory management and parallel processing."""
    
    def __init__(self, config: BatchConfig = None):
        """Initialize batch processor."""
        self.config = config or BatchConfig()
        self.memory_monitor = MemoryMonitor(self.config.max_memory_mb)
        self.stats = BatchStats()
        
        # Initialize executor based on configuration
        if self.config.use_multiprocessing:
            self.executor = ProcessPoolExecutor(max_workers=self.config.max_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
    
    def __enter__(self):
        """Context manager entry."""
        self.stats.start_time = datetime.now()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stats.end_time = datetime.now()
        self.stats.peak_memory_mb = self.memory_monitor.peak_memory_mb
        self.executor.shutdown(wait=True)
        
        if self.config.enable_gc:
            self.memory_monitor.force_gc()
    
    def process_query_batches(
        self,
        query: Select,
        processor_func: Callable[[List[T]], List[Any]],
        progress_callback: Optional[Callable[[BatchStats], None]] = None
    ) -> BatchStats:
        """Process database query results in optimized batches."""
        
        with db_manager.get_session() as session:
            paginator = OptimizedPaginator(session, query, self.config)
            self.stats.total_items = paginator.get_total_count()
            
            logger.info(f"Starting batch processing of {self.stats.total_items} items")
            
            batch_times = []
            
            for batch in paginator.get_batches():
                batch_start = datetime.now()
                
                try:
                    # Check memory usage
                    if self.memory_monitor.is_memory_limit_exceeded():
                        logger.warning("Memory limit exceeded, forcing garbage collection")
                        self.memory_monitor.force_gc()
                    
                    # Process batch
                    results = processor_func(batch)
                    
                    # Update statistics
                    self.stats.processed_items += len(results)
                    self.stats.batches_processed += 1
                    
                    # Calculate batch processing time
                    batch_time = (datetime.now() - batch_start).total_seconds()
                    batch_times.append(batch_time)
                    
                    # Periodic garbage collection
                    if (self.config.enable_gc and 
                        self.stats.batches_processed % self.config.gc_threshold == 0):
                        self.memory_monitor.force_gc()
                    
                    # Progress callback
                    if progress_callback:
                        progress_callback(self.stats)
                    
                    logger.debug(f"Processed batch {self.stats.batches_processed}: "
                               f"{len(batch)} items in {batch_time:.2f}s")
                
                except Exception as e:
                    logger.error(f"Error processing batch {self.stats.batches_processed}: {e}")
                    self.stats.failed_items += len(batch)
            
            # Calculate average batch time
            if batch_times:
                self.stats.avg_batch_time = sum(batch_times) / len(batch_times)
            
            logger.info(f"Batch processing completed: {self.stats.processed_items} processed, "
                       f"{self.stats.failed_items} failed")
            
            return self.stats
    
    async def process_async_batches(
        self,
        items: List[T],
        async_processor_func: Callable[[List[T]], Any],
        progress_callback: Optional[Callable[[BatchStats], None]] = None
    ) -> BatchStats:
        """Process items in async batches with concurrency control."""
        
        self.stats.total_items = len(items)
        logger.info(f"Starting async batch processing of {self.stats.total_items} items")
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.config.max_workers)
        
        async def process_batch_with_semaphore(batch: List[T]) -> List[Any]:
            async with semaphore:
                return await async_processor_func(batch)
        
        # Split items into batches
        batches = [
            items[i:i + self.config.batch_size]
            for i in range(0, len(items), self.config.batch_size)
        ]
        
        # Process batches concurrently
        tasks = [process_batch_with_semaphore(batch) for batch in batches]
        
        batch_times = []
        
        for i, task in enumerate(asyncio.as_completed(tasks)):
            batch_start = datetime.now()
            
            try:
                results = await task
                
                # Update statistics
                self.stats.processed_items += len(results) if results else len(batches[i])
                self.stats.batches_processed += 1
                
                # Calculate batch processing time
                batch_time = (datetime.now() - batch_start).total_seconds()
                batch_times.append(batch_time)
                
                # Memory management
                if self.memory_monitor.is_memory_limit_exceeded():
                    logger.warning("Memory limit exceeded during async processing")
                    self.memory_monitor.force_gc()
                
                # Progress callback
                if progress_callback:
                    progress_callback(self.stats)
                
                logger.debug(f"Completed async batch {self.stats.batches_processed}: "
                           f"{len(batches[i])} items in {batch_time:.2f}s")
            
            except Exception as e:
                logger.error(f"Error in async batch {i}: {e}")
                self.stats.failed_items += len(batches[i])
        
        # Calculate average batch time
        if batch_times:
            self.stats.avg_batch_time = sum(batch_times) / len(batch_times)
        
        logger.info(f"Async batch processing completed: {self.stats.processed_items} processed, "
                   f"{self.stats.failed_items} failed")
        
        return self.stats
    
    def process_parallel_batches(
        self,
        items: List[T],
        processor_func: Callable[[List[T]], List[Any]],
        progress_callback: Optional[Callable[[BatchStats], None]] = None
    ) -> BatchStats:
        """Process items in parallel batches using thread/process pool."""
        
        self.stats.total_items = len(items)
        logger.info(f"Starting parallel batch processing of {self.stats.total_items} items")
        
        # Split items into batches
        batches = [
            items[i:i + self.config.batch_size]
            for i in range(0, len(items), self.config.batch_size)
        ]
        
        # Submit all batches to executor
        future_to_batch = {
            self.executor.submit(processor_func, batch): batch
            for batch in batches
        }
        
        batch_times = []
        
        for future in as_completed(future_to_batch, timeout=self.config.timeout_seconds):
            batch_start = datetime.now()
            batch = future_to_batch[future]
            
            try:
                results = future.result()
                
                # Update statistics
                self.stats.processed_items += len(results) if results else len(batch)
                self.stats.batches_processed += 1
                
                # Calculate batch processing time
                batch_time = (datetime.now() - batch_start).total_seconds()
                batch_times.append(batch_time)
                
                # Memory management
                if self.memory_monitor.is_memory_limit_exceeded():
                    logger.warning("Memory limit exceeded during parallel processing")
                    self.memory_monitor.force_gc()
                
                # Progress callback
                if progress_callback:
                    progress_callback(self.stats)
                
                logger.debug(f"Completed parallel batch {self.stats.batches_processed}: "
                           f"{len(batch)} items in {batch_time:.2f}s")
            
            except Exception as e:
                logger.error(f"Error in parallel batch processing: {e}")
                self.stats.failed_items += len(batch)
        
        # Calculate average batch time
        if batch_times:
            self.stats.avg_batch_time = sum(batch_times) / len(batch_times)
        
        logger.info(f"Parallel batch processing completed: {self.stats.processed_items} processed, "
                   f"{self.stats.failed_items} failed")
        
        return self.stats


class StreamingProcessor:
    """Streaming processor for very large datasets that don't fit in memory."""
    
    def __init__(self, config: BatchConfig = None):
        """Initialize streaming processor."""
        self.config = config or BatchConfig()
        self.memory_monitor = MemoryMonitor(self.config.max_memory_mb)
    
    def stream_process_query(
        self,
        query: Select,
        processor_func: Callable[[T], Any],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Iterator[Any]:
        """Stream process query results one item at a time."""
        
        with db_manager.get_session() as session:
            # Use server-side cursor for streaming
            result = session.execute(query.execution_options(stream_results=True))
            
            processed_count = 0
            
            for row in result:
                try:
                    # Check memory usage periodically
                    if processed_count % 1000 == 0:
                        if self.memory_monitor.is_memory_limit_exceeded():
                            logger.warning("Memory limit exceeded during streaming")
                            self.memory_monitor.force_gc()
                    
                    # Process single item
                    result_item = processor_func(row)
                    processed_count += 1
                    
                    # Progress callback
                    if progress_callback and processed_count % 100 == 0:
                        progress_callback(processed_count, -1)  # -1 indicates unknown total
                    
                    yield result_item
                
                except Exception as e:
                    logger.error(f"Error processing stream item {processed_count}: {e}")
                    continue
            
            logger.info(f"Stream processing completed: {processed_count} items processed")


class OptimizedBatchQueries:
    """Collection of optimized batch query operations."""
    
    def __init__(self, processor: BatchProcessor):
        """Initialize optimized batch queries."""
        self.processor = processor
    
    def batch_update_quality_scores(
        self,
        task_ids: List[str],
        score_calculator: Callable[[TaskModel], float],
        progress_callback: Optional[Callable[[BatchStats], None]] = None
    ) -> BatchStats:
        """Batch update quality scores for tasks."""
        
        def update_batch(tasks: List[TaskModel]) -> List[TaskModel]:
            updated_tasks = []
            
            with db_manager.get_session() as session:
                for task in tasks:
                    try:
                        # Calculate new quality score
                        new_score = score_calculator(task)
                        
                        # Update in database using SQLAlchemy 2.0 style
                        from sqlalchemy import update
                        stmt = update(TaskModel).where(
                            TaskModel.id == task.id
                        ).values(quality_score=new_score)
                        session.execute(stmt)
                        
                        task.quality_score = new_score
                        updated_tasks.append(task)
                        
                    except Exception as e:
                        logger.error(f"Error updating quality score for task {task.id}: {e}")
                
                session.commit()
            
            return updated_tasks
        
        # Create query for tasks
        query = select(TaskModel).where(TaskModel.id.in_(task_ids))
        
        return self.processor.process_query_batches(
            query, update_batch, progress_callback
        )
    
    def batch_export_documents(
        self,
        project_id: str,
        export_processor: Callable[[List[DocumentModel]], List[Dict[str, Any]]],
        progress_callback: Optional[Callable[[BatchStats], None]] = None
    ) -> BatchStats:
        """Batch export documents for a project."""
        
        # Create optimized query with joins
        query = (
            select(DocumentModel)
            .join(TaskModel)
            .where(TaskModel.project_id == project_id)
            .options(
                # Eager load related tasks to avoid N+1 queries
                # Note: This would need proper SQLAlchemy relationship setup
            )
        )
        
        return self.processor.process_query_batches(
            query, export_processor, progress_callback
        )
    
    def batch_calculate_billing(
        self,
        tenant_id: str,
        billing_calculator: Callable[[List[BillingRecordModel]], List[Dict[str, Any]]],
        progress_callback: Optional[Callable[[BatchStats], None]] = None
    ) -> BatchStats:
        """Batch calculate billing for a tenant."""
        
        query = (
            select(BillingRecordModel)
            .where(BillingRecordModel.tenant_id == tenant_id)
            .order_by(BillingRecordModel.billing_date.desc())
        )
        
        return self.processor.process_query_batches(
            query, billing_calculator, progress_callback
        )


# Utility functions for common batch operations

def create_optimized_batch_config(
    dataset_size: int,
    available_memory_mb: int = 512,
    cpu_cores: int = None
) -> BatchConfig:
    """Create optimized batch configuration based on dataset size and resources."""
    
    if cpu_cores is None:
        cpu_cores = psutil.cpu_count()
    
    # Calculate optimal batch size based on dataset size and memory
    if dataset_size < 1000:
        batch_size = min(100, dataset_size)
        max_workers = 2
    elif dataset_size < 10000:
        batch_size = 500
        max_workers = min(4, cpu_cores)
    elif dataset_size < 100000:
        batch_size = 1000
        max_workers = min(8, cpu_cores)
    else:
        batch_size = 2000
        max_workers = min(16, cpu_cores)
        
    # Use multiprocessing for CPU-intensive tasks on large datasets
    use_multiprocessing = dataset_size > 50000 and cpu_cores > 4
    
    return BatchConfig(
        batch_size=batch_size,
        max_memory_mb=available_memory_mb,
        max_workers=max_workers,
        use_multiprocessing=use_multiprocessing,
        enable_gc=True,
        gc_threshold=max(10, batch_size // 100)
    )


@contextmanager
def optimized_batch_processing(
    dataset_size: int,
    available_memory_mb: int = 512,
    cpu_cores: int = None
):
    """Context manager for optimized batch processing."""
    
    config = create_optimized_batch_config(dataset_size, available_memory_mb, cpu_cores)
    
    with BatchProcessor(config) as processor:
        yield processor