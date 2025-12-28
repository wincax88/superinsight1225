"""
Real-time Data Stream Processor for WebSocket Sync

This module provides stream processing capabilities for real-time data synchronization,
including data filtering, transformation, backpressure control, and flow management.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, Generic, List, Optional, Set, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T")


class StreamState(str, Enum):
    """Stream processing states."""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    BACKPRESSURE = "backpressure"
    ERROR = "error"
    STOPPED = "stopped"


class BackpressureStrategy(str, Enum):
    """Strategies for handling backpressure."""

    DROP_OLDEST = "drop_oldest"
    DROP_NEWEST = "drop_newest"
    BLOCK = "block"
    SAMPLE = "sample"


@dataclass
class StreamMetrics:
    """Metrics for stream processing."""

    messages_received: int = 0
    messages_processed: int = 0
    messages_dropped: int = 0
    messages_filtered: int = 0
    processing_errors: int = 0
    backpressure_events: int = 0
    total_processing_time_ms: float = 0.0
    last_message_time: Optional[datetime] = None
    stream_start_time: Optional[datetime] = None

    @property
    def average_processing_time_ms(self) -> float:
        if self.messages_processed == 0:
            return 0.0
        return self.total_processing_time_ms / self.messages_processed

    @property
    def throughput_per_second(self) -> float:
        if not self.stream_start_time:
            return 0.0
        elapsed = (datetime.utcnow() - self.stream_start_time).total_seconds()
        if elapsed == 0:
            return 0.0
        return self.messages_processed / elapsed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "messages_received": self.messages_received,
            "messages_processed": self.messages_processed,
            "messages_dropped": self.messages_dropped,
            "messages_filtered": self.messages_filtered,
            "processing_errors": self.processing_errors,
            "backpressure_events": self.backpressure_events,
            "average_processing_time_ms": round(self.average_processing_time_ms, 2),
            "throughput_per_second": round(self.throughput_per_second, 2),
            "last_message_time": self.last_message_time.isoformat()
            if self.last_message_time
            else None,
        }


class StreamMessage(BaseModel):
    """Container for stream messages."""

    id: str
    source: str
    type: str
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = {}
    retry_count: int = 0

    class Config:
        arbitrary_types_allowed = True


@dataclass
class FilterRule:
    """Rule for filtering stream messages."""

    field: str
    operator: str  # eq, ne, gt, lt, gte, lte, contains, regex, in
    value: Any
    negate: bool = False

    def matches(self, data: Dict[str, Any]) -> bool:
        """Check if data matches this filter rule."""
        field_value = data.get(self.field)

        if field_value is None:
            return self.negate

        result = False

        if self.operator == "eq":
            result = field_value == self.value
        elif self.operator == "ne":
            result = field_value != self.value
        elif self.operator == "gt":
            result = field_value > self.value
        elif self.operator == "lt":
            result = field_value < self.value
        elif self.operator == "gte":
            result = field_value >= self.value
        elif self.operator == "lte":
            result = field_value <= self.value
        elif self.operator == "contains":
            result = self.value in str(field_value)
        elif self.operator == "in":
            result = field_value in self.value
        elif self.operator == "regex":
            import re

            result = bool(re.match(self.value, str(field_value)))

        return not result if self.negate else result


class StreamFilter:
    """Composite filter for stream messages."""

    def __init__(
        self, rules: Optional[List[FilterRule]] = None, match_all: bool = True
    ):
        self.rules = rules or []
        self.match_all = match_all

    def add_rule(self, rule: FilterRule) -> None:
        self.rules.append(rule)

    def matches(self, message: StreamMessage) -> bool:
        """Check if message matches filter criteria."""
        if not self.rules:
            return True

        data = {"type": message.type, "source": message.source, **message.data}

        if self.match_all:
            return all(rule.matches(data) for rule in self.rules)
        else:
            return any(rule.matches(data) for rule in self.rules)


class DataTransformer(ABC, Generic[T]):
    """Abstract base class for data transformers."""

    @abstractmethod
    async def transform(self, message: StreamMessage) -> T:
        """Transform a stream message."""
        pass


class IdentityTransformer(DataTransformer[StreamMessage]):
    """Pass-through transformer that returns the message unchanged."""

    async def transform(self, message: StreamMessage) -> StreamMessage:
        return message


class FieldMappingTransformer(DataTransformer[StreamMessage]):
    """Transform messages by mapping field names."""

    def __init__(self, field_mappings: Dict[str, str]):
        self.field_mappings = field_mappings

    async def transform(self, message: StreamMessage) -> StreamMessage:
        new_data = {}
        for old_key, new_key in self.field_mappings.items():
            if old_key in message.data:
                new_data[new_key] = message.data[old_key]

        # Include unmapped fields
        for key, value in message.data.items():
            if key not in self.field_mappings:
                new_data[key] = value

        return StreamMessage(
            id=message.id,
            source=message.source,
            type=message.type,
            data=new_data,
            timestamp=message.timestamp,
            metadata=message.metadata,
        )


class AggregatingTransformer(DataTransformer[StreamMessage]):
    """Transform messages by aggregating values over a time window."""

    def __init__(
        self,
        aggregation_field: str,
        aggregation_func: str = "sum",  # sum, avg, min, max, count
        window_size_seconds: float = 60.0,
    ):
        self.aggregation_field = aggregation_field
        self.aggregation_func = aggregation_func
        self.window_size_seconds = window_size_seconds
        self.window_values: List[tuple] = []  # (timestamp, value)

    async def transform(self, message: StreamMessage) -> StreamMessage:
        current_time = time.time()
        value = message.data.get(self.aggregation_field, 0)

        # Add current value to window
        self.window_values.append((current_time, value))

        # Remove old values outside window
        cutoff_time = current_time - self.window_size_seconds
        self.window_values = [
            (ts, val) for ts, val in self.window_values if ts >= cutoff_time
        ]

        # Calculate aggregation
        values = [val for _, val in self.window_values]
        aggregated = self._aggregate(values)

        new_data = {
            **message.data,
            f"{self.aggregation_field}_{self.aggregation_func}": aggregated,
            "window_count": len(values),
        }

        return StreamMessage(
            id=message.id,
            source=message.source,
            type=message.type,
            data=new_data,
            timestamp=message.timestamp,
            metadata=message.metadata,
        )

    def _aggregate(self, values: List[Any]) -> Any:
        if not values:
            return 0

        if self.aggregation_func == "sum":
            return sum(values)
        elif self.aggregation_func == "avg":
            return sum(values) / len(values)
        elif self.aggregation_func == "min":
            return min(values)
        elif self.aggregation_func == "max":
            return max(values)
        elif self.aggregation_func == "count":
            return len(values)
        return values[-1]


class BackpressureController:
    """Controls flow rate and manages backpressure."""

    def __init__(
        self,
        max_buffer_size: int = 1000,
        high_watermark: float = 0.8,
        low_watermark: float = 0.3,
        strategy: BackpressureStrategy = BackpressureStrategy.DROP_OLDEST,
        sample_rate: float = 0.1,
    ):
        self.max_buffer_size = max_buffer_size
        self.high_watermark = high_watermark
        self.low_watermark = low_watermark
        self.strategy = strategy
        self.sample_rate = sample_rate
        self.buffer: deque = deque(maxlen=max_buffer_size)
        self.is_backpressure_active = False
        self._sample_counter = 0
        self._lock = asyncio.Lock()

    @property
    def buffer_utilization(self) -> float:
        return len(self.buffer) / self.max_buffer_size if self.max_buffer_size > 0 else 0

    async def push(self, message: StreamMessage) -> bool:
        """Push a message to the buffer. Returns True if message was accepted."""
        async with self._lock:
            # Check if we're in backpressure mode
            if self.buffer_utilization >= self.high_watermark:
                self.is_backpressure_active = True

            if self.is_backpressure_active:
                return await self._handle_backpressure(message)

            self.buffer.append(message)
            return True

    async def _handle_backpressure(self, message: StreamMessage) -> bool:
        """Handle message during backpressure."""
        if self.strategy == BackpressureStrategy.DROP_OLDEST:
            if len(self.buffer) >= self.max_buffer_size:
                self.buffer.popleft()
            self.buffer.append(message)
            return True

        elif self.strategy == BackpressureStrategy.DROP_NEWEST:
            if len(self.buffer) >= self.max_buffer_size:
                return False
            self.buffer.append(message)
            return True

        elif self.strategy == BackpressureStrategy.BLOCK:
            while len(self.buffer) >= self.max_buffer_size:
                await asyncio.sleep(0.01)
            self.buffer.append(message)
            return True

        elif self.strategy == BackpressureStrategy.SAMPLE:
            self._sample_counter += 1
            if self._sample_counter >= int(1 / self.sample_rate):
                self._sample_counter = 0
                self.buffer.append(message)
                return True
            return False

        return False

    async def pop(self) -> Optional[StreamMessage]:
        """Pop a message from the buffer."""
        async with self._lock:
            if not self.buffer:
                return None

            # Check if we can exit backpressure mode
            if self.is_backpressure_active and self.buffer_utilization <= self.low_watermark:
                self.is_backpressure_active = False

            return self.buffer.popleft()

    async def clear(self) -> int:
        """Clear the buffer. Returns number of messages cleared."""
        async with self._lock:
            count = len(self.buffer)
            self.buffer.clear()
            self.is_backpressure_active = False
            return count


class RetryPolicy:
    """Policy for retrying failed message processing."""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay_seconds: float = 1.0,
        max_delay_seconds: float = 60.0,
        exponential_base: float = 2.0,
    ):
        self.max_retries = max_retries
        self.base_delay_seconds = base_delay_seconds
        self.max_delay_seconds = max_delay_seconds
        self.exponential_base = exponential_base

    def should_retry(self, retry_count: int) -> bool:
        return retry_count < self.max_retries

    def get_delay(self, retry_count: int) -> float:
        delay = self.base_delay_seconds * (self.exponential_base**retry_count)
        return min(delay, self.max_delay_seconds)


class StreamProcessor:
    """
    Main stream processor for handling real-time data streams.

    Features:
    - Message filtering and transformation
    - Backpressure control
    - Error handling with retry policy
    - Metrics collection
    """

    def __init__(
        self,
        processor_id: str,
        handler: Callable[[StreamMessage], Coroutine[Any, Any, None]],
        filter: Optional[StreamFilter] = None,
        transformer: Optional[DataTransformer] = None,
        backpressure_controller: Optional[BackpressureController] = None,
        retry_policy: Optional[RetryPolicy] = None,
        batch_size: int = 1,
        batch_timeout_seconds: float = 1.0,
    ):
        self.processor_id = processor_id
        self.handler = handler
        self.filter = filter or StreamFilter()
        self.transformer = transformer or IdentityTransformer()
        self.backpressure = backpressure_controller or BackpressureController()
        self.retry_policy = retry_policy or RetryPolicy()
        self.batch_size = batch_size
        self.batch_timeout_seconds = batch_timeout_seconds

        self.state = StreamState.IDLE
        self.metrics = StreamMetrics()
        self._processing_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self._pause_event = asyncio.Event()
        self._pause_event.set()  # Not paused initially
        self._dead_letter_queue: deque = deque(maxlen=1000)

    async def start(self) -> None:
        """Start the stream processor."""
        if self.state == StreamState.RUNNING:
            logger.warning(f"Processor {self.processor_id} is already running")
            return

        self._stop_event.clear()
        self.state = StreamState.RUNNING
        self.metrics.stream_start_time = datetime.utcnow()

        self._processing_task = asyncio.create_task(self._processing_loop())
        logger.info(f"Stream processor {self.processor_id} started")

    async def stop(self) -> None:
        """Stop the stream processor."""
        self._stop_event.set()
        self.state = StreamState.STOPPED

        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass

        logger.info(f"Stream processor {self.processor_id} stopped")

    async def pause(self) -> None:
        """Pause the stream processor."""
        self._pause_event.clear()
        self.state = StreamState.PAUSED
        logger.info(f"Stream processor {self.processor_id} paused")

    async def resume(self) -> None:
        """Resume the stream processor."""
        self._pause_event.set()
        self.state = StreamState.RUNNING
        logger.info(f"Stream processor {self.processor_id} resumed")

    async def submit(self, message: StreamMessage) -> bool:
        """Submit a message to the processor."""
        self.metrics.messages_received += 1
        self.metrics.last_message_time = datetime.utcnow()

        # Apply filter
        if not self.filter.matches(message):
            self.metrics.messages_filtered += 1
            return True

        # Push to backpressure controller
        accepted = await self.backpressure.push(message)
        if not accepted:
            self.metrics.messages_dropped += 1
            logger.debug(f"Message {message.id} dropped due to backpressure")
            return False

        if self.backpressure.is_backpressure_active:
            self.state = StreamState.BACKPRESSURE
            self.metrics.backpressure_events += 1

        return True

    async def _processing_loop(self) -> None:
        """Main processing loop."""
        batch: List[StreamMessage] = []
        last_batch_time = time.time()

        while not self._stop_event.is_set():
            try:
                # Wait if paused
                await self._pause_event.wait()

                # Try to get a message
                message = await self.backpressure.pop()

                if message:
                    batch.append(message)

                # Process batch if full or timeout reached
                current_time = time.time()
                should_process = (
                    len(batch) >= self.batch_size
                    or (
                        batch
                        and (current_time - last_batch_time) >= self.batch_timeout_seconds
                    )
                )

                if should_process and batch:
                    await self._process_batch(batch)
                    batch = []
                    last_batch_time = current_time

                if not message:
                    await asyncio.sleep(0.01)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                self.state = StreamState.ERROR
                await asyncio.sleep(1)

        # Process remaining messages
        if batch:
            await self._process_batch(batch)

    async def _process_batch(self, batch: List[StreamMessage]) -> None:
        """Process a batch of messages."""
        for message in batch:
            await self._process_message(message)

    async def _process_message(self, message: StreamMessage) -> None:
        """Process a single message with retry support."""
        start_time = time.time()

        try:
            # Transform message
            transformed = await self.transformer.transform(message)

            # Handle message
            await self.handler(transformed)

            self.metrics.messages_processed += 1
            processing_time = (time.time() - start_time) * 1000
            self.metrics.total_processing_time_ms += processing_time

        except Exception as e:
            logger.error(f"Error processing message {message.id}: {e}")
            self.metrics.processing_errors += 1

            # Check if we should retry
            if self.retry_policy.should_retry(message.retry_count):
                message.retry_count += 1
                delay = self.retry_policy.get_delay(message.retry_count)
                logger.info(
                    f"Retrying message {message.id} in {delay}s (attempt {message.retry_count})"
                )
                await asyncio.sleep(delay)
                await self.backpressure.push(message)
            else:
                # Move to dead letter queue
                self._dead_letter_queue.append(
                    {
                        "message": message.dict(),
                        "error": str(e),
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )
                logger.warning(f"Message {message.id} moved to dead letter queue")

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return {
            **self.metrics.to_dict(),
            "state": self.state.value,
            "buffer_utilization": round(self.backpressure.buffer_utilization, 2),
            "is_backpressure_active": self.backpressure.is_backpressure_active,
            "dead_letter_queue_size": len(self._dead_letter_queue),
        }

    def get_dead_letter_messages(self, limit: int = 100) -> List[Dict]:
        """Get messages from dead letter queue."""
        return list(self._dead_letter_queue)[:limit]


class StreamProcessorManager:
    """Manages multiple stream processors."""

    def __init__(self):
        self.processors: Dict[str, StreamProcessor] = {}
        self._lock = asyncio.Lock()

    async def register_processor(self, processor: StreamProcessor) -> None:
        """Register a stream processor."""
        async with self._lock:
            if processor.processor_id in self.processors:
                raise ValueError(
                    f"Processor {processor.processor_id} already registered"
                )
            self.processors[processor.processor_id] = processor
            logger.info(f"Registered processor {processor.processor_id}")

    async def unregister_processor(self, processor_id: str) -> None:
        """Unregister and stop a processor."""
        async with self._lock:
            if processor_id not in self.processors:
                return

            processor = self.processors[processor_id]
            await processor.stop()
            del self.processors[processor_id]
            logger.info(f"Unregistered processor {processor_id}")

    async def start_all(self) -> None:
        """Start all registered processors."""
        async with self._lock:
            for processor in self.processors.values():
                await processor.start()

    async def stop_all(self) -> None:
        """Stop all registered processors."""
        async with self._lock:
            for processor in self.processors.values():
                await processor.stop()

    async def submit_to_processor(
        self, processor_id: str, message: StreamMessage
    ) -> bool:
        """Submit a message to a specific processor."""
        processor = self.processors.get(processor_id)
        if not processor:
            raise ValueError(f"Processor {processor_id} not found")
        return await processor.submit(message)

    async def broadcast(self, message: StreamMessage) -> Dict[str, bool]:
        """Broadcast a message to all processors."""
        results = {}
        for processor_id, processor in self.processors.items():
            try:
                results[processor_id] = await processor.submit(message)
            except Exception as e:
                logger.error(f"Error broadcasting to {processor_id}: {e}")
                results[processor_id] = False
        return results

    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics from all processors."""
        return {
            processor_id: processor.get_metrics()
            for processor_id, processor in self.processors.items()
        }

    def get_processor(self, processor_id: str) -> Optional[StreamProcessor]:
        """Get a specific processor."""
        return self.processors.get(processor_id)


# Convenience functions for common use cases
def create_sync_data_processor(
    handler: Callable[[StreamMessage], Coroutine[Any, Any, None]],
    source_types: Optional[List[str]] = None,
    max_buffer_size: int = 1000,
) -> StreamProcessor:
    """Create a processor optimized for sync data streams."""
    filter_rules = []
    if source_types:
        filter_rules.append(FilterRule(field="type", operator="in", value=source_types))

    return StreamProcessor(
        processor_id=f"sync_data_{int(time.time() * 1000)}",
        handler=handler,
        filter=StreamFilter(rules=filter_rules),
        backpressure_controller=BackpressureController(
            max_buffer_size=max_buffer_size,
            strategy=BackpressureStrategy.DROP_OLDEST,
        ),
        retry_policy=RetryPolicy(max_retries=3),
        batch_size=10,
        batch_timeout_seconds=1.0,
    )


def create_event_processor(
    handler: Callable[[StreamMessage], Coroutine[Any, Any, None]],
    event_types: Optional[List[str]] = None,
) -> StreamProcessor:
    """Create a processor optimized for event streams."""
    filter_rules = []
    if event_types:
        filter_rules.append(FilterRule(field="type", operator="in", value=event_types))

    return StreamProcessor(
        processor_id=f"event_{int(time.time() * 1000)}",
        handler=handler,
        filter=StreamFilter(rules=filter_rules),
        backpressure_controller=BackpressureController(
            max_buffer_size=500,
            strategy=BackpressureStrategy.BLOCK,
        ),
        retry_policy=RetryPolicy(max_retries=5, base_delay_seconds=0.5),
        batch_size=1,
        batch_timeout_seconds=0.1,
    )
