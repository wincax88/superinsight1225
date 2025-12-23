"""
Batch AI Annotation Processor for SuperInsight platform.

Provides batch processing capabilities for AI pre-annotation with async task queues.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Union
from uuid import UUID, uuid4
from enum import Enum
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import redis
from pydantic import BaseModel, ConfigDict

from .base import AIAnnotator, ModelConfig, Prediction, AIAnnotationError
from .factory import AnnotatorFactory, ConfidenceScorer
try:
    from models.task import Task
except ImportError:
    from src.models.task import Task


class BatchStatus(str, Enum):
    """Enumeration of batch processing statuses."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BatchJobConfig(BaseModel):
    """Configuration for batch annotation jobs."""
    
    job_id: str = field(default_factory=lambda: str(uuid4()))
    model_configs: List[ModelConfig]
    max_concurrent_tasks: int = 10
    retry_attempts: int = 3
    timeout_seconds: int = 300
    enable_caching: bool = True
    cache_ttl_hours: int = 24
    confidence_threshold: float = 0.5
    ensemble_method: str = "average"  # average, max, min, weighted_average
    
    model_config = ConfigDict(
        json_encoders={
            UUID: str,
            datetime: lambda v: v.isoformat()
        }
    )


@dataclass
class BatchResult:
    """Result of batch annotation processing."""
    
    job_id: str
    status: BatchStatus
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    predictions: List[Prediction] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    processing_time: float = 0.0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_tasks == 0:
            return 0.0
        return (self.completed_tasks / self.total_tasks) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "total_tasks": self.total_tasks,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "success_rate": self.success_rate,
            "predictions": [p.to_dict() for p in self.predictions],
            "errors": self.errors,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "processing_time": self.processing_time
        }


class BatchAnnotationProcessor:
    """Batch processor for AI annotations with async task queue support."""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        """
        Initialize batch processor.
        
        Args:
            redis_client: Redis client for caching and job tracking
        """
        self.redis_client = redis_client
        self.active_jobs: Dict[str, BatchResult] = {}
        self.executor = ThreadPoolExecutor(max_workers=20)
        
        # Cache key prefixes
        self.cache_prefix = "superinsight:ai_cache:"
        self.job_prefix = "superinsight:batch_job:"
    
    async def submit_batch_job(
        self,
        tasks: List[Task],
        config: BatchJobConfig,
        progress_callback: Optional[Callable[[BatchResult], None]] = None
    ) -> str:
        """
        Submit a batch annotation job.
        
        Args:
            tasks: List of tasks to annotate
            config: Batch job configuration
            progress_callback: Optional callback for progress updates
            
        Returns:
            Job ID for tracking the batch job
        """
        job_id = config.job_id
        
        # Initialize batch result
        batch_result = BatchResult(
            job_id=job_id,
            status=BatchStatus.PENDING,
            total_tasks=len(tasks),
            completed_tasks=0,
            failed_tasks=0,
            started_at=datetime.now()
        )
        
        self.active_jobs[job_id] = batch_result
        
        # Store job in Redis if available
        if self.redis_client:
            await self._store_job_status(batch_result)
        
        # Start processing asynchronously
        asyncio.create_task(
            self._process_batch_job(tasks, config, batch_result, progress_callback)
        )
        
        return job_id
    
    async def _process_batch_job(
        self,
        tasks: List[Task],
        config: BatchJobConfig,
        batch_result: BatchResult,
        progress_callback: Optional[Callable[[BatchResult], None]] = None
    ) -> None:
        """Process batch annotation job."""
        try:
            batch_result.status = BatchStatus.PROCESSING
            
            # Create annotators from configs
            annotators = []
            for model_config in config.model_configs:
                try:
                    annotator = AnnotatorFactory.create_annotator(model_config)
                    annotators.append(annotator)
                except Exception as e:
                    batch_result.errors.append(f"Failed to create annotator: {str(e)}")
            
            if not annotators:
                batch_result.status = BatchStatus.FAILED
                batch_result.errors.append("No valid annotators available")
                return
            
            # Process tasks with concurrency control
            semaphore = asyncio.Semaphore(config.max_concurrent_tasks)
            
            async def process_single_task(task: Task) -> Optional[Prediction]:
                async with semaphore:
                    return await self._process_task_with_ensemble(
                        task, annotators, config
                    )
            
            # Create tasks for concurrent processing
            task_coroutines = [process_single_task(task) for task in tasks]
            
            # Process with progress tracking
            for i, coro in enumerate(asyncio.as_completed(task_coroutines)):
                try:
                    prediction = await coro
                    if prediction:
                        batch_result.predictions.append(prediction)
                        batch_result.completed_tasks += 1
                    else:
                        batch_result.failed_tasks += 1
                        
                except Exception as e:
                    batch_result.failed_tasks += 1
                    batch_result.errors.append(f"Task processing failed: {str(e)}")
                
                # Update progress
                if progress_callback:
                    progress_callback(batch_result)
                
                # Store progress in Redis
                if self.redis_client:
                    await self._store_job_status(batch_result)
            
            # Mark as completed
            batch_result.status = BatchStatus.COMPLETED
            batch_result.completed_at = datetime.now()
            batch_result.processing_time = (
                batch_result.completed_at - batch_result.started_at
            ).total_seconds()
            
        except Exception as e:
            batch_result.status = BatchStatus.FAILED
            batch_result.errors.append(f"Batch processing failed: {str(e)}")
            batch_result.completed_at = datetime.now()
        
        finally:
            # Final status update
            if self.redis_client:
                await self._store_job_status(batch_result)
            
            if progress_callback:
                progress_callback(batch_result)
    
    async def _process_task_with_ensemble(
        self,
        task: Task,
        annotators: List[AIAnnotator],
        config: BatchJobConfig
    ) -> Optional[Prediction]:
        """Process a single task with ensemble of annotators."""
        try:
            # Check cache first
            if config.enable_caching and self.redis_client:
                cached_result = await self._get_cached_prediction(task, config)
                if cached_result:
                    return cached_result
            
            # Get predictions from all annotators
            predictions = []
            for annotator in annotators:
                for attempt in range(config.retry_attempts):
                    try:
                        prediction = await asyncio.wait_for(
                            annotator.predict(task),
                            timeout=config.timeout_seconds
                        )
                        predictions.append(prediction)
                        break
                    except asyncio.TimeoutError:
                        if attempt == config.retry_attempts - 1:
                            raise AIAnnotationError(
                                f"Timeout after {config.timeout_seconds}s",
                                model_type=str(annotator.config.model_type),
                                task_id=task.id
                            )
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    except Exception as e:
                        if attempt == config.retry_attempts - 1:
                            raise e
                        await asyncio.sleep(2 ** attempt)
            
            if not predictions:
                return None
            
            # Create ensemble prediction
            ensemble_prediction = await self._create_ensemble_prediction(
                predictions, config
            )
            
            # Cache result if enabled
            if config.enable_caching and self.redis_client:
                await self._cache_prediction(ensemble_prediction, config)
            
            return ensemble_prediction
            
        except Exception as e:
            print(f"Failed to process task {task.id}: {e}")
            return None
    
    async def _create_ensemble_prediction(
        self,
        predictions: List[Prediction],
        config: BatchJobConfig
    ) -> Prediction:
        """Create ensemble prediction from multiple model predictions."""
        if len(predictions) == 1:
            return predictions[0]
        
        # Extract confidence scores
        confidences = [p.confidence for p in predictions]
        
        # Calculate ensemble confidence
        ensemble_confidence = ConfidenceScorer.calculate_ensemble_confidence(
            confidences, config.ensemble_method
        )
        
        # Combine prediction data
        ensemble_data = {
            "ensemble_method": config.ensemble_method,
            "individual_predictions": [p.prediction_data for p in predictions],
            "individual_confidences": confidences,
            "model_configs": [p.model_config.dict() for p in predictions]
        }
        
        # Use the first prediction as base and enhance with ensemble data
        base_prediction = predictions[0]
        
        return Prediction(
            id=uuid4(),
            task_id=base_prediction.task_id,
            ai_model_config=base_prediction.ai_model_config,  # Use first model's config
            prediction_data=ensemble_data,
            confidence=ensemble_confidence,
            processing_time=sum(p.processing_time for p in predictions)
        )
    
    async def _get_cached_prediction(
        self,
        task: Task,
        config: BatchJobConfig
    ) -> Optional[Prediction]:
        """Get cached prediction if available."""
        if not self.redis_client:
            return None
        
        try:
            cache_key = self._get_cache_key(task, config)
            cached_data = await self.redis_client.get(cache_key)
            
            if cached_data:
                prediction_dict = json.loads(cached_data)
                # Reconstruct prediction object (simplified)
                return Prediction(**prediction_dict)
                
        except Exception:
            pass  # Cache miss or error, continue without cache
        
        return None
    
    async def _cache_prediction(
        self,
        prediction: Prediction,
        config: BatchJobConfig
    ) -> None:
        """Cache prediction result."""
        if not self.redis_client:
            return
        
        try:
            cache_key = self._get_cache_key_from_prediction(prediction, config)
            cache_data = json.dumps(prediction.to_dict())
            ttl_seconds = config.cache_ttl_hours * 3600
            
            await self.redis_client.setex(cache_key, ttl_seconds, cache_data)
            
        except Exception:
            pass  # Cache error, continue without caching
    
    def _get_cache_key(self, task: Task, config: BatchJobConfig) -> str:
        """Generate cache key for task and config."""
        model_names = [mc.model_name for mc in config.model_configs]
        model_key = "_".join(sorted(model_names))
        return f"{self.cache_prefix}{task.id}_{model_key}"
    
    def _get_cache_key_from_prediction(
        self,
        prediction: Prediction,
        config: BatchJobConfig
    ) -> str:
        """Generate cache key from prediction."""
        model_names = [mc.model_name for mc in config.model_configs]
        model_key = "_".join(sorted(model_names))
        return f"{self.cache_prefix}{prediction.task_id}_{model_key}"
    
    async def _store_job_status(self, batch_result: BatchResult) -> None:
        """Store job status in Redis."""
        if not self.redis_client:
            return
        
        try:
            job_key = f"{self.job_prefix}{batch_result.job_id}"
            job_data = json.dumps(batch_result.to_dict())
            
            # Store with 7 days TTL
            await self.redis_client.setex(job_key, 7 * 24 * 3600, job_data)
            
        except Exception:
            pass  # Storage error, continue without persistence
    
    async def get_job_status(self, job_id: str) -> Optional[BatchResult]:
        """
        Get status of a batch job.
        
        Args:
            job_id: ID of the batch job
            
        Returns:
            Batch result or None if not found
        """
        # Check active jobs first
        if job_id in self.active_jobs:
            return self.active_jobs[job_id]
        
        # Check Redis storage
        if self.redis_client:
            try:
                job_key = f"{self.job_prefix}{job_id}"
                job_data = await self.redis_client.get(job_key)
                
                if job_data:
                    result_dict = json.loads(job_data)
                    # Reconstruct BatchResult (simplified)
                    return BatchResult(**result_dict)
                    
            except Exception:
                pass
        
        return None
    
    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a batch job.
        
        Args:
            job_id: ID of the batch job to cancel
            
        Returns:
            True if cancelled, False if not found or already completed
        """
        if job_id in self.active_jobs:
            batch_result = self.active_jobs[job_id]
            if batch_result.status in [BatchStatus.PENDING, BatchStatus.PROCESSING]:
                batch_result.status = BatchStatus.CANCELLED
                batch_result.completed_at = datetime.now()
                
                if self.redis_client:
                    await self._store_job_status(batch_result)
                
                return True
        
        return False
    
    async def list_active_jobs(self) -> List[str]:
        """Get list of active job IDs."""
        return list(self.active_jobs.keys())
    
    async def cleanup_completed_jobs(self, max_age_hours: int = 24) -> int:
        """
        Clean up completed jobs older than specified age.
        
        Args:
            max_age_hours: Maximum age in hours for completed jobs
            
        Returns:
            Number of jobs cleaned up
        """
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        cleaned_count = 0
        
        jobs_to_remove = []
        for job_id, batch_result in self.active_jobs.items():
            if (batch_result.status in [BatchStatus.COMPLETED, BatchStatus.FAILED, BatchStatus.CANCELLED] and
                batch_result.completed_at and batch_result.completed_at < cutoff_time):
                jobs_to_remove.append(job_id)
        
        for job_id in jobs_to_remove:
            del self.active_jobs[job_id]
            cleaned_count += 1
        
        return cleaned_count
    
    async def get_job_statistics(self) -> Dict[str, Any]:
        """Get statistics about batch jobs."""
        total_jobs = len(self.active_jobs)
        status_counts = {}
        
        for batch_result in self.active_jobs.values():
            status = batch_result.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "total_jobs": total_jobs,
            "status_counts": status_counts,
            "active_jobs": list(self.active_jobs.keys())
        }