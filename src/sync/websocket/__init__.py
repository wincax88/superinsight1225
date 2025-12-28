"""
WebSocket Module.

Provides WebSocket support for real-time data push and stream processing.
"""

from src.sync.websocket.ws_server import (
    WebSocketConnectionManager,
    ConnectionInfo,
    ConnectionState,
    MessageType,
    SubscriptionType,
    WebSocketMessage,
    AuthPayload,
    SubscriptionPayload,
)

from src.sync.websocket.stream_processor import (
    StreamProcessor,
    StreamProcessorManager,
    StreamState,
    StreamMessage,
    StreamFilter,
    FilterRule,
    BackpressureController,
    BackpressureStrategy,
    RetryPolicy,
    DataTransformer,
    IdentityTransformer,
    FieldMappingTransformer,
    AggregatingTransformer,
    StreamMetrics,
    create_sync_data_processor,
    create_event_processor,
)

__all__ = [
    # WebSocket Server
    "WebSocketConnectionManager",
    "ConnectionInfo",
    "ConnectionState",
    "MessageType",
    "SubscriptionType",
    "WebSocketMessage",
    "AuthPayload",
    "SubscriptionPayload",
    # Stream Processor
    "StreamProcessor",
    "StreamProcessorManager",
    "StreamState",
    "StreamMessage",
    "StreamFilter",
    "FilterRule",
    "BackpressureController",
    "BackpressureStrategy",
    "RetryPolicy",
    "DataTransformer",
    "IdentityTransformer",
    "FieldMappingTransformer",
    "AggregatingTransformer",
    "StreamMetrics",
    "create_sync_data_processor",
    "create_event_processor",
]
