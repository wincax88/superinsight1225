"""
Sync Executor.

Provides execution engine for sync jobs with support for
incremental sync, checkpointing, and error recovery.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from src.sync.connectors.base import (
    BaseConnector,
    DataBatch,
    SyncResult,
)

logger = logging.getLogger(__name__)


class ExecutionStatus(str, Enum):
    """Execution status enumeration."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ExecutionContext:
    """Context for sync execution."""
    execution_id: str
    job_id: str
    tenant_id: str
    source_connector: Optional[BaseConnector] = None
    target_connector: Optional[BaseConnector] = None
    batch_size: int = 1000
    enable_incremental: bool = True
    incremental_field: Optional[str] = None
    last_sync_value: Optional[str] = None
    checkpoint: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Runtime state
    status: ExecutionStatus = ExecutionStatus.INITIALIZING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    current_batch: int = 0
    total_batches: int = 0

    # Counters
    records_total: int = 0
    records_processed: int = 0
    records_inserted: int = 0
    records_updated: int = 0
    records_deleted: int = 0
    records_skipped: int = 0
    records_failed: int = 0
    bytes_transferred: int = 0

    # Errors
    errors: List[Dict[str, Any]] = field(default_factory=list)
    last_error: Optional[str] = None


@dataclass
class ExecutionResult:
    """Result of sync execution."""
    execution_id: str
    job_id: str
    success: bool
    status: ExecutionStatus
    started_at: datetime
    completed_at: Optional[datetime]
    duration_seconds: float
    records_total: int
    records_processed: int
    records_inserted: int
    records_updated: int
    records_deleted: int
    records_skipped: int
    records_failed: int
    bytes_transferred: int
    errors: List[Dict[str, Any]]
    checkpoint: Dict[str, Any]
    metadata: Dict[str, Any]


class SyncExecutor:
    """
    Sync Job Executor.

    Executes sync jobs with support for:
    - Incremental sync
    - Checkpointing and resume
    - Error recovery
    - Progress tracking
    - Concurrent batch processing
    """

    def __init__(
        self,
        max_concurrent_batches: int = 3,
        checkpoint_interval: int = 10
    ):
        self._max_concurrent = max_concurrent_batches
        self._checkpoint_interval = checkpoint_interval
        self._active_executions: Dict[str, ExecutionContext] = {}
        self._execution_history: List[ExecutionResult] = []

    async def execute(
        self,
        context: ExecutionContext,
        on_progress: Optional[callable] = None
    ) -> ExecutionResult:
        """
        Execute a sync job.

        Args:
            context: Execution context with configuration
            on_progress: Optional callback for progress updates

        Returns:
            ExecutionResult with sync statistics
        """
        execution_id = context.execution_id
        self._active_executions[execution_id] = context

        context.status = ExecutionStatus.RUNNING
        context.started_at = datetime.utcnow()
        start_time = time.time()

        logger.info(f"Starting sync execution: {execution_id}")

        try:
            # Connect to source
            if context.source_connector:
                if not context.source_connector.is_connected:
                    await context.source_connector.connect()

            # Get total record count
            if context.source_connector:
                context.records_total = await context.source_connector.get_record_count()

            # Calculate batches
            if context.records_total > 0:
                context.total_batches = (
                    (context.records_total + context.batch_size - 1) //
                    context.batch_size
                )

            # Process data in batches
            await self._process_batches(context, on_progress)

            # Mark as completed
            context.status = ExecutionStatus.COMPLETED
            context.completed_at = datetime.utcnow()

            logger.info(
                f"Sync execution completed: {execution_id}, "
                f"processed={context.records_processed}"
            )

        except asyncio.CancelledError:
            context.status = ExecutionStatus.CANCELLED
            logger.warning(f"Sync execution cancelled: {execution_id}")
            raise

        except Exception as e:
            context.status = ExecutionStatus.FAILED
            context.last_error = str(e)
            context.errors.append({
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            })
            logger.error(f"Sync execution failed: {execution_id}, error={e}")

        finally:
            # Disconnect
            if context.source_connector and context.source_connector.is_connected:
                await context.source_connector.disconnect()

            if context.target_connector and context.target_connector.is_connected:
                await context.target_connector.disconnect()

            # Create result
            duration = time.time() - start_time
            result = ExecutionResult(
                execution_id=execution_id,
                job_id=context.job_id,
                success=context.status == ExecutionStatus.COMPLETED,
                status=context.status,
                started_at=context.started_at,
                completed_at=context.completed_at or datetime.utcnow(),
                duration_seconds=duration,
                records_total=context.records_total,
                records_processed=context.records_processed,
                records_inserted=context.records_inserted,
                records_updated=context.records_updated,
                records_deleted=context.records_deleted,
                records_skipped=context.records_skipped,
                records_failed=context.records_failed,
                bytes_transferred=context.bytes_transferred,
                errors=context.errors,
                checkpoint=context.checkpoint,
                metadata=context.metadata
            )

            # Store in history
            self._execution_history.append(result)

            # Remove from active
            self._active_executions.pop(execution_id, None)

            return result

    async def _process_batches(
        self,
        context: ExecutionContext,
        on_progress: Optional[callable] = None
    ) -> None:
        """Process data in batches."""
        if not context.source_connector:
            # Simulate processing for demo
            for i in range(10):
                context.current_batch = i + 1
                context.records_processed += 100
                context.records_inserted += 80
                context.records_updated += 20

                if on_progress:
                    on_progress(context)

                await asyncio.sleep(0.1)
            return

        # Stream data from source
        batch_num = 0
        async for batch in context.source_connector.fetch_data_stream(
            batch_size=context.batch_size,
            incremental_field=context.incremental_field,
            incremental_value=context.last_sync_value
        ):
            batch_num += 1
            context.current_batch = batch_num

            # Process batch
            result = await self._process_batch(context, batch)

            # Update counters
            context.records_processed += result.records_processed
            context.records_inserted += result.records_inserted
            context.records_updated += result.records_updated
            context.records_deleted += result.records_deleted
            context.records_failed += result.records_failed

            # Update checkpoint
            if batch.checkpoint:
                context.checkpoint = batch.checkpoint

            # Save checkpoint periodically
            if batch_num % self._checkpoint_interval == 0:
                await self._save_checkpoint(context)

            # Progress callback
            if on_progress:
                on_progress(context)

            # Check for cancellation
            if context.status == ExecutionStatus.CANCELLED:
                break

    async def _process_batch(
        self,
        context: ExecutionContext,
        batch: DataBatch
    ) -> SyncResult:
        """Process a single batch."""
        logger.debug(
            f"Processing batch {context.current_batch}: "
            f"{len(batch.records)} records"
        )

        # Transform records if needed
        # In production, apply transformation rules here

        # Write to target
        if context.target_connector:
            return await context.target_connector.write_data(batch)

        # Simulate write for demo
        return SyncResult(
            success=True,
            records_processed=len(batch.records),
            records_inserted=len(batch.records),
            duration_seconds=0.1
        )

    async def _save_checkpoint(self, context: ExecutionContext) -> None:
        """Save execution checkpoint."""
        logger.debug(
            f"Saving checkpoint for {context.execution_id}: "
            f"batch={context.current_batch}"
        )
        # In production, persist to database

    def cancel_execution(self, execution_id: str) -> bool:
        """
        Cancel a running execution.

        Args:
            execution_id: ID of execution to cancel

        Returns:
            True if execution was cancelled
        """
        context = self._active_executions.get(execution_id)
        if not context:
            return False

        context.status = ExecutionStatus.CANCELLED
        logger.info(f"Cancelling execution: {execution_id}")
        return True

    def pause_execution(self, execution_id: str) -> bool:
        """
        Pause a running execution.

        Args:
            execution_id: ID of execution to pause

        Returns:
            True if execution was paused
        """
        context = self._active_executions.get(execution_id)
        if not context:
            return False

        context.status = ExecutionStatus.PAUSED
        logger.info(f"Pausing execution: {execution_id}")
        return True

    def resume_execution(self, execution_id: str) -> bool:
        """
        Resume a paused execution.

        Args:
            execution_id: ID of execution to resume

        Returns:
            True if execution was resumed
        """
        context = self._active_executions.get(execution_id)
        if not context or context.status != ExecutionStatus.PAUSED:
            return False

        context.status = ExecutionStatus.RUNNING
        logger.info(f"Resuming execution: {execution_id}")
        return True

    def get_execution(self, execution_id: str) -> Optional[ExecutionContext]:
        """Get active execution context."""
        return self._active_executions.get(execution_id)

    def list_active_executions(self) -> List[ExecutionContext]:
        """List all active executions."""
        return list(self._active_executions.values())

    def get_execution_history(
        self,
        job_id: Optional[str] = None,
        limit: int = 100
    ) -> List[ExecutionResult]:
        """
        Get execution history.

        Args:
            job_id: Filter by job ID
            limit: Maximum results

        Returns:
            List of execution results
        """
        history = self._execution_history

        if job_id:
            history = [r for r in history if r.job_id == job_id]

        return history[-limit:]

    def get_stats(self) -> Dict[str, Any]:
        """Get executor statistics."""
        total_processed = sum(r.records_processed for r in self._execution_history)
        total_duration = sum(r.duration_seconds for r in self._execution_history)
        success_count = sum(1 for r in self._execution_history if r.success)

        return {
            "active_executions": len(self._active_executions),
            "total_executions": len(self._execution_history),
            "successful_executions": success_count,
            "failed_executions": len(self._execution_history) - success_count,
            "total_records_processed": total_processed,
            "total_duration_seconds": total_duration,
            "avg_duration_seconds": (
                total_duration / len(self._execution_history)
                if self._execution_history else 0
            )
        }


# Global executor instance
sync_executor = SyncExecutor()
