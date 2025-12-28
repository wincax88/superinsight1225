"""
Event Manager Module.

Provides event-driven architecture for sync orchestration,
including event publishing, subscription, and persistence.
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Standard sync event types."""
    # Workflow events
    WORKFLOW_STARTED = "workflow.started"
    WORKFLOW_COMPLETED = "workflow.completed"
    WORKFLOW_FAILED = "workflow.failed"
    WORKFLOW_PAUSED = "workflow.paused"
    WORKFLOW_RESUMED = "workflow.resumed"
    WORKFLOW_CANCELLED = "workflow.cancelled"

    # Step events
    STEP_STARTED = "step.started"
    STEP_COMPLETED = "step.completed"
    STEP_FAILED = "step.failed"
    STEP_RETRYING = "step.retrying"
    STEP_SKIPPED = "step.skipped"

    # Data events
    DATA_EXTRACTED = "data.extracted"
    DATA_TRANSFORMED = "data.transformed"
    DATA_LOADED = "data.loaded"
    DATA_VALIDATED = "data.validated"
    DATA_CONFLICT = "data.conflict"

    # System events
    CONNECTOR_CONNECTED = "connector.connected"
    CONNECTOR_DISCONNECTED = "connector.disconnected"
    CONNECTOR_ERROR = "connector.error"

    # Alert events
    ALERT_WARNING = "alert.warning"
    ALERT_ERROR = "alert.error"
    ALERT_CRITICAL = "alert.critical"

    # Custom
    CUSTOM = "custom"


class EventPriority(str, Enum):
    """Event priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Event:
    """Represents a sync event."""
    id: str
    type: EventType
    source: str
    timestamp: datetime
    data: Dict[str, Any] = field(default_factory=dict)
    priority: EventPriority = EventPriority.NORMAL
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "id": self.id,
            "type": self.type.value,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "priority": self.priority.value,
            "correlation_id": self.correlation_id,
            "causation_id": self.causation_id,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        """Create event from dictionary."""
        return cls(
            id=data["id"],
            type=EventType(data["type"]),
            source=data["source"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            data=data.get("data", {}),
            priority=EventPriority(data.get("priority", "normal")),
            correlation_id=data.get("correlation_id"),
            causation_id=data.get("causation_id"),
            metadata=data.get("metadata", {})
        )


@dataclass
class Subscription:
    """Event subscription configuration."""
    id: str
    event_types: Set[EventType]
    handler: Callable
    filter_fn: Optional[Callable[[Event], bool]] = None
    priority: int = 0  # Higher priority handlers execute first
    async_handler: bool = True
    max_retries: int = 3


class EventStore:
    """In-memory event store with optional persistence."""

    def __init__(self, max_events: int = 10000):
        self._events: List[Event] = []
        self._max_events = max_events
        self._event_index: Dict[str, int] = {}

    async def store(self, event: Event) -> None:
        """Store an event."""
        if len(self._events) >= self._max_events:
            # Remove oldest events
            removed = self._events[:1000]
            self._events = self._events[1000:]
            for e in removed:
                self._event_index.pop(e.id, None)

        self._events.append(event)
        self._event_index[event.id] = len(self._events) - 1

    async def get(self, event_id: str) -> Optional[Event]:
        """Get event by ID."""
        idx = self._event_index.get(event_id)
        if idx is not None and idx < len(self._events):
            return self._events[idx]
        return None

    async def query(
        self,
        event_types: Optional[List[EventType]] = None,
        source: Optional[str] = None,
        correlation_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Event]:
        """Query events with filters."""
        results = []

        for event in reversed(self._events):
            if len(results) >= limit:
                break

            if event_types and event.type not in event_types:
                continue
            if source and event.source != source:
                continue
            if correlation_id and event.correlation_id != correlation_id:
                continue
            if start_time and event.timestamp < start_time:
                continue
            if end_time and event.timestamp > end_time:
                continue

            results.append(event)

        return results

    async def replay(
        self,
        handler: Callable,
        event_types: Optional[List[EventType]] = None,
        start_time: Optional[datetime] = None
    ) -> int:
        """Replay events to a handler."""
        count = 0

        for event in self._events:
            if event_types and event.type not in event_types:
                continue
            if start_time and event.timestamp < start_time:
                continue

            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
                count += 1
            except Exception as e:
                logger.error(f"Replay handler error: {e}")

        return count


class EventManager:
    """
    Event Manager for pub/sub event handling.

    Features:
    - Type-safe event publishing
    - Subscription with filters
    - Priority-based handler execution
    - Event persistence and replay
    - Dead letter queue for failed events
    """

    def __init__(self, store: Optional[EventStore] = None):
        """
        Initialize event manager.

        Args:
            store: Event store for persistence (optional)
        """
        self._subscriptions: Dict[str, Subscription] = {}
        self._type_subscriptions: Dict[EventType, Set[str]] = {}
        self._store = store or EventStore()
        self._dead_letter: List[tuple] = []
        self._running = False
        self._queue: asyncio.Queue = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the event manager."""
        if self._running:
            return

        self._running = True
        self._worker_task = asyncio.create_task(self._process_events())
        logger.info("Event manager started")

    async def stop(self) -> None:
        """Stop the event manager."""
        self._running = False

        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

        logger.info("Event manager stopped")

    def subscribe(
        self,
        event_types: List[EventType],
        handler: Callable,
        filter_fn: Optional[Callable[[Event], bool]] = None,
        priority: int = 0,
        subscription_id: Optional[str] = None
    ) -> str:
        """
        Subscribe to events.

        Args:
            event_types: List of event types to subscribe to
            handler: Handler function (sync or async)
            filter_fn: Optional filter function
            priority: Handler priority (higher = first)
            subscription_id: Optional subscription ID

        Returns:
            Subscription ID
        """
        sub_id = subscription_id or f"sub_{uuid.uuid4().hex[:8]}"

        subscription = Subscription(
            id=sub_id,
            event_types=set(event_types),
            handler=handler,
            filter_fn=filter_fn,
            priority=priority,
            async_handler=asyncio.iscoroutinefunction(handler)
        )

        self._subscriptions[sub_id] = subscription

        for event_type in event_types:
            if event_type not in self._type_subscriptions:
                self._type_subscriptions[event_type] = set()
            self._type_subscriptions[event_type].add(sub_id)

        logger.debug(f"Subscription created: {sub_id}")
        return sub_id

    def unsubscribe(self, subscription_id: str) -> bool:
        """
        Unsubscribe from events.

        Args:
            subscription_id: ID of subscription to remove

        Returns:
            True if subscription was removed
        """
        subscription = self._subscriptions.pop(subscription_id, None)

        if subscription:
            for event_type in subscription.event_types:
                if event_type in self._type_subscriptions:
                    self._type_subscriptions[event_type].discard(subscription_id)
            logger.debug(f"Subscription removed: {subscription_id}")
            return True

        return False

    async def publish(
        self,
        event_type: EventType,
        source: str,
        data: Dict[str, Any] = None,
        priority: EventPriority = EventPriority.NORMAL,
        correlation_id: Optional[str] = None,
        causation_id: Optional[str] = None,
        wait: bool = False
    ) -> str:
        """
        Publish an event.

        Args:
            event_type: Type of event
            source: Source of event
            data: Event data
            priority: Event priority
            correlation_id: Correlation ID for tracking
            causation_id: ID of causing event
            wait: Wait for handlers to complete

        Returns:
            Event ID
        """
        event = Event(
            id=f"evt_{uuid.uuid4().hex}",
            type=event_type,
            source=source,
            timestamp=datetime.utcnow(),
            data=data or {},
            priority=priority,
            correlation_id=correlation_id,
            causation_id=causation_id
        )

        # Store event
        await self._store.store(event)

        if wait:
            # Process synchronously
            await self._dispatch_event(event)
        else:
            # Queue for async processing
            await self._queue.put(event)

        logger.debug(f"Published event: {event.type.value} ({event.id})")
        return event.id

    async def publish_batch(
        self,
        events: List[Dict[str, Any]],
        source: str
    ) -> List[str]:
        """Publish multiple events."""
        event_ids = []

        for event_data in events:
            event_id = await self.publish(
                event_type=EventType(event_data["type"]),
                source=source,
                data=event_data.get("data", {}),
                priority=EventPriority(event_data.get("priority", "normal"))
            )
            event_ids.append(event_id)

        return event_ids

    async def get_event(self, event_id: str) -> Optional[Event]:
        """Get event by ID."""
        return await self._store.get(event_id)

    async def query_events(
        self,
        event_types: Optional[List[EventType]] = None,
        source: Optional[str] = None,
        correlation_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Event]:
        """Query events with filters."""
        return await self._store.query(
            event_types=event_types,
            source=source,
            correlation_id=correlation_id,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )

    async def replay_events(
        self,
        handler: Callable,
        event_types: Optional[List[EventType]] = None,
        start_time: Optional[datetime] = None
    ) -> int:
        """Replay events to a handler."""
        return await self._store.replay(
            handler=handler,
            event_types=event_types,
            start_time=start_time
        )

    def get_dead_letter_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get failed events from dead letter queue."""
        return [
            {
                "event": event.to_dict(),
                "error": error,
                "failed_at": failed_at.isoformat()
            }
            for event, error, failed_at in self._dead_letter[-limit:]
        ]

    async def _process_events(self) -> None:
        """Process events from queue."""
        while self._running:
            try:
                event = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=1.0
                )
                await self._dispatch_event(event)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Event processing error: {e}")

    async def _dispatch_event(self, event: Event) -> None:
        """Dispatch event to subscribed handlers."""
        subscription_ids = self._type_subscriptions.get(event.type, set())

        if not subscription_ids:
            return

        # Get subscriptions and sort by priority
        subscriptions = [
            self._subscriptions[sid]
            for sid in subscription_ids
            if sid in self._subscriptions
        ]
        subscriptions.sort(key=lambda s: s.priority, reverse=True)

        for subscription in subscriptions:
            # Apply filter
            if subscription.filter_fn:
                try:
                    if not subscription.filter_fn(event):
                        continue
                except Exception as e:
                    logger.warning(f"Filter error: {e}")
                    continue

            # Execute handler
            try:
                if subscription.async_handler:
                    await subscription.handler(event)
                else:
                    subscription.handler(event)

            except Exception as e:
                logger.error(
                    f"Handler error for {event.type.value}: {e}"
                )
                self._dead_letter.append((event, str(e), datetime.utcnow()))

                # Limit dead letter size
                if len(self._dead_letter) > 1000:
                    self._dead_letter = self._dead_letter[-500:]


class EventBuilder:
    """Builder for creating events."""

    def __init__(self, event_type: EventType, source: str):
        self._type = event_type
        self._source = source
        self._data: Dict[str, Any] = {}
        self._priority = EventPriority.NORMAL
        self._correlation_id: Optional[str] = None
        self._causation_id: Optional[str] = None
        self._metadata: Dict[str, Any] = {}

    def data(self, key: str, value: Any) -> "EventBuilder":
        self._data[key] = value
        return self

    def with_data(self, data: Dict[str, Any]) -> "EventBuilder":
        self._data.update(data)
        return self

    def priority(self, priority: EventPriority) -> "EventBuilder":
        self._priority = priority
        return self

    def correlation(self, correlation_id: str) -> "EventBuilder":
        self._correlation_id = correlation_id
        return self

    def caused_by(self, event_id: str) -> "EventBuilder":
        self._causation_id = event_id
        return self

    def metadata(self, key: str, value: Any) -> "EventBuilder":
        self._metadata[key] = value
        return self

    def build(self) -> Event:
        return Event(
            id=f"evt_{uuid.uuid4().hex}",
            type=self._type,
            source=self._source,
            timestamp=datetime.utcnow(),
            data=self._data,
            priority=self._priority,
            correlation_id=self._correlation_id,
            causation_id=self._causation_id,
            metadata=self._metadata
        )


__all__ = [
    "EventManager",
    "EventStore",
    "EventBuilder",
    "Event",
    "EventType",
    "EventPriority",
    "Subscription",
]
