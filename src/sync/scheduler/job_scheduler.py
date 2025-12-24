"""
Job Scheduler.

Provides scheduling capabilities for sync jobs with support for
cron expressions, intervals, and one-time executions.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import UUID, uuid4

logger = logging.getLogger(__name__)


class JobState(str, Enum):
    """Job state enumeration."""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class TriggerType(str, Enum):
    """Trigger type enumeration."""
    CRON = "cron"
    INTERVAL = "interval"
    ONE_TIME = "one_time"
    MANUAL = "manual"
    EVENT = "event"


@dataclass
class JobTrigger(ABC):
    """Abstract base for job triggers."""
    trigger_type: TriggerType
    enabled: bool = True

    @abstractmethod
    def get_next_run_time(self, after: datetime) -> Optional[datetime]:
        """Get next run time after given datetime."""
        pass


@dataclass
class CronTrigger(JobTrigger):
    """Cron-based job trigger."""
    cron_expression: str
    timezone: str = "UTC"

    def __post_init__(self):
        self.trigger_type = TriggerType.CRON

    def get_next_run_time(self, after: datetime) -> Optional[datetime]:
        """
        Calculate next run time based on cron expression.

        Simple implementation for common cron patterns.
        In production, use croniter library.
        """
        if not self.enabled:
            return None

        # Simple implementation: add 1 hour as default
        # In production, parse cron expression properly
        return after + timedelta(hours=1)


@dataclass
class IntervalTrigger(JobTrigger):
    """Interval-based job trigger."""
    interval_seconds: int
    start_time: Optional[datetime] = None

    def __post_init__(self):
        self.trigger_type = TriggerType.INTERVAL

    def get_next_run_time(self, after: datetime) -> Optional[datetime]:
        """Calculate next run time based on interval."""
        if not self.enabled:
            return None

        if self.start_time and after < self.start_time:
            return self.start_time

        return after + timedelta(seconds=self.interval_seconds)


@dataclass
class OneTimeTrigger(JobTrigger):
    """One-time job trigger."""
    run_at: datetime
    executed: bool = False

    def __post_init__(self):
        self.trigger_type = TriggerType.ONE_TIME

    def get_next_run_time(self, after: datetime) -> Optional[datetime]:
        """Return scheduled time if not yet executed."""
        if not self.enabled or self.executed:
            return None

        if self.run_at > after:
            return self.run_at
        return None


@dataclass
class ScheduledJob:
    """Represents a scheduled sync job."""
    id: str
    job_id: str  # Reference to SyncJobModel
    tenant_id: str
    name: str
    trigger: JobTrigger
    handler: Optional[Callable] = None
    state: JobState = JobState.PENDING
    priority: int = 5  # 1-10, higher is more urgent
    max_retries: int = 3
    retry_count: int = 0
    timeout_seconds: int = 3600
    created_at: datetime = field(default_factory=datetime.utcnow)
    next_run_at: Optional[datetime] = None
    last_run_at: Optional[datetime] = None
    last_result: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def should_run(self, now: datetime) -> bool:
        """Check if job should run now."""
        if self.state not in [JobState.PENDING, JobState.SCHEDULED]:
            return False
        if not self.next_run_at:
            return False
        return now >= self.next_run_at

    def schedule_next(self, after: datetime) -> None:
        """Schedule next execution."""
        next_time = self.trigger.get_next_run_time(after)
        self.next_run_at = next_time
        if next_time:
            self.state = JobState.SCHEDULED


class SyncScheduler:
    """
    Sync Job Scheduler.

    Manages scheduling and execution of sync jobs with support for:
    - Cron-based scheduling
    - Interval-based scheduling
    - One-time execution
    - Priority-based execution
    - Retry handling
    - Concurrent execution limits
    """

    def __init__(
        self,
        max_concurrent_jobs: int = 10,
        check_interval: float = 1.0
    ):
        self._jobs: Dict[str, ScheduledJob] = {}
        self._running_jobs: Set[str] = set()
        self._max_concurrent = max_concurrent_jobs
        self._check_interval = check_interval
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._handlers: Dict[str, Callable] = {}
        self._listeners: List[Callable] = []

    @property
    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._running

    @property
    def job_count(self) -> int:
        """Get total job count."""
        return len(self._jobs)

    @property
    def running_job_count(self) -> int:
        """Get running job count."""
        return len(self._running_jobs)

    def add_job(self, job: ScheduledJob) -> None:
        """
        Add a job to the scheduler.

        Args:
            job: ScheduledJob to add
        """
        if job.id in self._jobs:
            raise ValueError(f"Job {job.id} already exists")

        # Schedule first run
        job.schedule_next(datetime.utcnow())

        self._jobs[job.id] = job
        logger.info(f"Added job: {job.id}, next run: {job.next_run_at}")

        self._notify_listeners("job_added", job)

    def remove_job(self, job_id: str) -> bool:
        """
        Remove a job from the scheduler.

        Args:
            job_id: ID of job to remove

        Returns:
            True if job was removed
        """
        if job_id not in self._jobs:
            return False

        job = self._jobs.pop(job_id)
        logger.info(f"Removed job: {job_id}")

        self._notify_listeners("job_removed", job)
        return True

    def get_job(self, job_id: str) -> Optional[ScheduledJob]:
        """Get a job by ID."""
        return self._jobs.get(job_id)

    def list_jobs(
        self,
        tenant_id: Optional[str] = None,
        state: Optional[JobState] = None
    ) -> List[ScheduledJob]:
        """
        List scheduled jobs.

        Args:
            tenant_id: Filter by tenant
            state: Filter by state

        Returns:
            List of matching jobs
        """
        jobs = list(self._jobs.values())

        if tenant_id:
            jobs = [j for j in jobs if j.tenant_id == tenant_id]

        if state:
            jobs = [j for j in jobs if j.state == state]

        return sorted(jobs, key=lambda j: (j.priority, j.next_run_at or datetime.max), reverse=True)

    def pause_job(self, job_id: str) -> bool:
        """Pause a job."""
        job = self._jobs.get(job_id)
        if not job:
            return False

        job.state = JobState.PAUSED
        logger.info(f"Paused job: {job_id}")
        return True

    def resume_job(self, job_id: str) -> bool:
        """Resume a paused job."""
        job = self._jobs.get(job_id)
        if not job or job.state != JobState.PAUSED:
            return False

        job.schedule_next(datetime.utcnow())
        logger.info(f"Resumed job: {job_id}")
        return True

    def trigger_job(self, job_id: str) -> bool:
        """
        Manually trigger a job execution.

        Args:
            job_id: ID of job to trigger

        Returns:
            True if job was triggered
        """
        job = self._jobs.get(job_id)
        if not job:
            return False

        job.next_run_at = datetime.utcnow()
        job.state = JobState.SCHEDULED
        logger.info(f"Manually triggered job: {job_id}")
        return True

    def register_handler(
        self,
        job_type: str,
        handler: Callable
    ) -> None:
        """
        Register a job execution handler.

        Args:
            job_type: Type of job to handle
            handler: Async function to execute
        """
        self._handlers[job_type] = handler
        logger.info(f"Registered handler for: {job_type}")

    def add_listener(self, listener: Callable) -> None:
        """Add an event listener."""
        self._listeners.append(listener)

    def _notify_listeners(self, event: str, job: ScheduledJob) -> None:
        """Notify all listeners of an event."""
        for listener in self._listeners:
            try:
                listener(event, job)
            except Exception as e:
                logger.error(f"Listener error: {e}")

    async def start(self) -> None:
        """Start the scheduler."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Scheduler started")

    async def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Scheduler stopped")

    async def _run_loop(self) -> None:
        """Main scheduler loop."""
        while self._running:
            try:
                await self._check_and_execute()
                await asyncio.sleep(self._check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                await asyncio.sleep(5)

    async def _check_and_execute(self) -> None:
        """Check for due jobs and execute them."""
        now = datetime.utcnow()

        # Get jobs that should run
        due_jobs = [
            job for job in self._jobs.values()
            if job.should_run(now) and job.id not in self._running_jobs
        ]

        # Sort by priority
        due_jobs.sort(key=lambda j: j.priority, reverse=True)

        # Execute up to max concurrent
        available_slots = self._max_concurrent - len(self._running_jobs)
        for job in due_jobs[:available_slots]:
            asyncio.create_task(self._execute_job(job))

    async def _execute_job(self, job: ScheduledJob) -> None:
        """Execute a single job."""
        job.state = JobState.RUNNING
        job.last_run_at = datetime.utcnow()
        self._running_jobs.add(job.id)

        logger.info(f"Executing job: {job.id}")
        self._notify_listeners("job_started", job)

        try:
            # Get handler
            handler = job.handler or self._handlers.get("default")

            if handler:
                # Execute with timeout
                result = await asyncio.wait_for(
                    handler(job),
                    timeout=job.timeout_seconds
                )
                job.last_result = {"success": True, "result": result}
                job.state = JobState.COMPLETED
                job.retry_count = 0

            else:
                raise ValueError(f"No handler for job: {job.id}")

        except asyncio.TimeoutError:
            job.last_result = {"success": False, "error": "Timeout"}
            job.state = JobState.FAILED
            logger.error(f"Job timed out: {job.id}")

        except Exception as e:
            job.last_result = {"success": False, "error": str(e)}
            job.retry_count += 1

            if job.retry_count < job.max_retries:
                job.state = JobState.PENDING
                logger.warning(
                    f"Job failed, will retry: {job.id} "
                    f"(attempt {job.retry_count}/{job.max_retries})"
                )
            else:
                job.state = JobState.FAILED
                logger.error(f"Job failed permanently: {job.id}")

        finally:
            self._running_jobs.discard(job.id)

            # Schedule next run if not failed
            if job.state != JobState.FAILED:
                job.schedule_next(datetime.utcnow())

            self._notify_listeners("job_completed", job)

    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        states = {}
        for job in self._jobs.values():
            states[job.state.value] = states.get(job.state.value, 0) + 1

        return {
            "total_jobs": len(self._jobs),
            "running_jobs": len(self._running_jobs),
            "max_concurrent": self._max_concurrent,
            "is_running": self._running,
            "jobs_by_state": states
        }


# Global scheduler instance
sync_scheduler = SyncScheduler()
