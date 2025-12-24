"""
Enhanced Notification System for SuperInsight Platform.

Provides intelligent error notification with rate limiting,
deduplication, and multiple delivery channels to reduce noise
and improve reliability.
"""

import asyncio
import logging
import smtplib
import time
from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import requests
from collections import defaultdict
import threading

try:
    from email.mime.text import MimeText
    from email.mime.multipart import MimeMultipart
    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False
    MimeText = None
    MimeMultipart = None

from src.config.settings import settings

logger = logging.getLogger(__name__)


class NotificationChannel(Enum):
    """Notification delivery channels."""
    EMAIL = "email"
    WEBHOOK = "webhook"
    SLACK = "slack"
    LOG = "log"
    SMS = "sms"


class NotificationPriority(Enum):
    """Notification priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class NotificationConfig:
    """Configuration for notification behavior."""
    enabled: bool = True
    channels: List[NotificationChannel] = field(default_factory=list)
    rate_limit_window: int = 300  # 5 minutes
    max_notifications_per_window: int = 10
    deduplication_window: int = 600  # 10 minutes
    retry_attempts: int = 3
    retry_delay: float = 5.0
    batch_size: int = 50
    batch_timeout: float = 30.0


@dataclass
class NotificationMessage:
    """Notification message structure."""
    id: str
    title: str
    message: str
    priority: NotificationPriority
    channels: List[NotificationChannel]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    retry_count: int = 0
    deduplication_key: Optional[str] = None


@dataclass
class DeliveryResult:
    """Result of notification delivery attempt."""
    success: bool
    channel: NotificationChannel
    message_id: str
    error: Optional[str] = None
    delivery_time: float = field(default_factory=time.time)


class NotificationRateLimiter:
    """Rate limiter for notifications to prevent spam."""
    
    def __init__(self, window_size: int = 300, max_count: int = 10):
        self.window_size = window_size
        self.max_count = max_count
        self.notifications: Dict[str, List[float]] = defaultdict(list)
        self.lock = threading.Lock()
    
    def is_allowed(self, key: str) -> bool:
        """Check if notification is allowed based on rate limits."""
        with self.lock:
            current_time = time.time()
            
            # Clean old entries
            self.notifications[key] = [
                timestamp for timestamp in self.notifications[key]
                if current_time - timestamp < self.window_size
            ]
            
            # Check if under limit
            if len(self.notifications[key]) < self.max_count:
                self.notifications[key].append(current_time)
                return True
            
            return False
    
    def get_remaining_quota(self, key: str) -> int:
        """Get remaining notification quota for a key."""
        with self.lock:
            current_time = time.time()
            
            # Clean old entries
            self.notifications[key] = [
                timestamp for timestamp in self.notifications[key]
                if current_time - timestamp < self.window_size
            ]
            
            return max(0, self.max_count - len(self.notifications[key]))


class NotificationDeduplicator:
    """Deduplicates notifications to reduce noise."""
    
    def __init__(self, window_size: int = 600):
        self.window_size = window_size
        self.seen_notifications: Dict[str, float] = {}
        self.lock = threading.Lock()
    
    def is_duplicate(self, deduplication_key: str) -> bool:
        """Check if notification is a duplicate."""
        if not deduplication_key:
            return False
        
        with self.lock:
            current_time = time.time()
            
            # Clean old entries
            self.seen_notifications = {
                key: timestamp for key, timestamp in self.seen_notifications.items()
                if current_time - timestamp < self.window_size
            }
            
            # Check if already seen
            if deduplication_key in self.seen_notifications:
                return True
            
            # Mark as seen
            self.seen_notifications[deduplication_key] = current_time
            return False


class EmailNotificationHandler:
    """Email notification handler with SMTP support."""
    
    def __init__(self, smtp_config: Optional[Dict[str, Any]] = None):
        self.smtp_config = smtp_config or {
            "host": getattr(settings, "SMTP_HOST", "localhost"),
            "port": getattr(settings, "SMTP_PORT", 587),
            "username": getattr(settings, "SMTP_USERNAME", ""),
            "password": getattr(settings, "SMTP_PASSWORD", ""),
            "use_tls": getattr(settings, "SMTP_USE_TLS", True),
            "from_email": getattr(settings, "SMTP_FROM_EMAIL", "noreply@superinsight.ai")
        }
        self.recipients = getattr(settings, "NOTIFICATION_EMAIL_RECIPIENTS", [])
    
    def send(self, message: NotificationMessage) -> DeliveryResult:
        """Send email notification."""
        try:
            if not EMAIL_AVAILABLE:
                return DeliveryResult(
                    success=False,
                    channel=NotificationChannel.EMAIL,
                    message_id=message.id,
                    error="Email functionality not available"
                )
            
            if not self.recipients:
                return DeliveryResult(
                    success=False,
                    channel=NotificationChannel.EMAIL,
                    message_id=message.id,
                    error="No email recipients configured"
                )
            
            # Create email message
            msg = MimeMultipart()
            msg['From'] = self.smtp_config['from_email']
            msg['To'] = ', '.join(self.recipients)
            msg['Subject'] = f"[SuperInsight] {message.title}"
            
            # Email body
            body = f"""
{message.message}

Priority: {message.priority.value.upper()}
Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(message.timestamp))}

Metadata:
{json.dumps(message.metadata, indent=2)}
"""
            msg.attach(MimeText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(self.smtp_config['host'], self.smtp_config['port']) as server:
                if self.smtp_config['use_tls']:
                    server.starttls()
                
                if self.smtp_config['username']:
                    server.login(self.smtp_config['username'], self.smtp_config['password'])
                
                server.send_message(msg)
            
            logger.info(f"Email notification sent successfully: {message.id}")
            return DeliveryResult(
                success=True,
                channel=NotificationChannel.EMAIL,
                message_id=message.id
            )
            
        except Exception as e:
            logger.error(f"Failed to send email notification {message.id}: {e}")
            return DeliveryResult(
                success=False,
                channel=NotificationChannel.EMAIL,
                message_id=message.id,
                error=str(e)
            )


class WebhookNotificationHandler:
    """Webhook notification handler for external integrations."""
    
    def __init__(self, webhook_urls: Optional[List[str]] = None):
        self.webhook_urls = webhook_urls or getattr(settings, "NOTIFICATION_WEBHOOK_URLS", [])
        self.timeout = 30
    
    def send(self, message: NotificationMessage) -> DeliveryResult:
        """Send webhook notification."""
        if not self.webhook_urls:
            return DeliveryResult(
                success=False,
                channel=NotificationChannel.WEBHOOK,
                message_id=message.id,
                error="No webhook URLs configured"
            )
        
        payload = {
            "id": message.id,
            "title": message.title,
            "message": message.message,
            "priority": message.priority.value,
            "timestamp": message.timestamp,
            "metadata": message.metadata
        }
        
        success_count = 0
        errors = []
        
        for url in self.webhook_urls:
            try:
                response = requests.post(
                    url,
                    json=payload,
                    timeout=self.timeout,
                    headers={'Content-Type': 'application/json'}
                )
                response.raise_for_status()
                success_count += 1
                logger.debug(f"Webhook notification sent to {url}: {message.id}")
                
            except Exception as e:
                error_msg = f"Failed to send to {url}: {e}"
                errors.append(error_msg)
                logger.error(error_msg)
        
        if success_count > 0:
            return DeliveryResult(
                success=True,
                channel=NotificationChannel.WEBHOOK,
                message_id=message.id
            )
        else:
            return DeliveryResult(
                success=False,
                channel=NotificationChannel.WEBHOOK,
                message_id=message.id,
                error="; ".join(errors)
            )


class SlackNotificationHandler:
    """Slack notification handler."""
    
    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url or getattr(settings, "SLACK_WEBHOOK_URL", "")
    
    def send(self, message: NotificationMessage) -> DeliveryResult:
        """Send Slack notification."""
        if not self.webhook_url:
            return DeliveryResult(
                success=False,
                channel=NotificationChannel.SLACK,
                message_id=message.id,
                error="No Slack webhook URL configured"
            )
        
        # Format message for Slack
        color = {
            NotificationPriority.LOW: "good",
            NotificationPriority.NORMAL: "warning",
            NotificationPriority.HIGH: "danger",
            NotificationPriority.CRITICAL: "danger"
        }.get(message.priority, "warning")
        
        payload = {
            "attachments": [{
                "color": color,
                "title": message.title,
                "text": message.message,
                "fields": [
                    {
                        "title": "Priority",
                        "value": message.priority.value.upper(),
                        "short": True
                    },
                    {
                        "title": "Timestamp",
                        "value": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(message.timestamp)),
                        "short": True
                    }
                ],
                "footer": "SuperInsight Platform",
                "ts": int(message.timestamp)
            }]
        }
        
        try:
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            logger.info(f"Slack notification sent successfully: {message.id}")
            return DeliveryResult(
                success=True,
                channel=NotificationChannel.SLACK,
                message_id=message.id
            )
            
        except Exception as e:
            logger.error(f"Failed to send Slack notification {message.id}: {e}")
            return DeliveryResult(
                success=False,
                channel=NotificationChannel.SLACK,
                message_id=message.id,
                error=str(e)
            )


class EnhancedNotificationSystem:
    """
    Enhanced notification system with intelligent filtering and delivery.
    
    Features:
    - Rate limiting to prevent spam
    - Deduplication to reduce noise
    - Multiple delivery channels
    - Retry mechanisms
    - Batch processing
    - Priority-based routing
    """
    
    def __init__(self, config: Optional[NotificationConfig] = None):
        self.config = config or NotificationConfig()
        self.rate_limiter = NotificationRateLimiter(
            window_size=self.config.rate_limit_window,
            max_count=self.config.max_notifications_per_window
        )
        self.deduplicator = NotificationDeduplicator(
            window_size=self.config.deduplication_window
        )
        
        # Initialize handlers
        self.handlers = {
            NotificationChannel.EMAIL: EmailNotificationHandler(),
            NotificationChannel.WEBHOOK: WebhookNotificationHandler(),
            NotificationChannel.SLACK: SlackNotificationHandler(),
            NotificationChannel.LOG: self._log_handler
        }
        
        # Message queue for batch processing
        self.message_queue: List[NotificationMessage] = []
        self.queue_lock = threading.Lock()
        self.delivery_results: List[DeliveryResult] = []
        
        # Start background processor
        self._start_background_processor()
    
    def send_notification(
        self,
        title: str,
        message: str,
        priority: NotificationPriority = NotificationPriority.NORMAL,
        channels: Optional[List[NotificationChannel]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        deduplication_key: Optional[str] = None
    ) -> bool:
        """Send a notification with intelligent filtering."""
        if not self.config.enabled:
            return False
        
        # Use default channels if none specified
        if channels is None:
            channels = self.config.channels or [NotificationChannel.LOG]
        
        # Create notification message
        notification = NotificationMessage(
            id=f"notif_{int(time.time() * 1000)}",
            title=title,
            message=message,
            priority=priority,
            channels=channels,
            metadata=metadata or {},
            deduplication_key=deduplication_key
        )
        
        # Check deduplication
        if deduplication_key and self.deduplicator.is_duplicate(deduplication_key):
            logger.debug(f"Notification deduplicated: {deduplication_key}")
            return False
        
        # Check rate limits
        rate_limit_key = f"{priority.value}:{':'.join(sorted([c.value for c in channels]))}"
        if not self.rate_limiter.is_allowed(rate_limit_key):
            logger.warning(f"Notification rate limited: {rate_limit_key}")
            return False
        
        # Add to queue for processing
        with self.queue_lock:
            self.message_queue.append(notification)
        
        logger.info(f"Notification queued: {notification.id}")
        return True
    
    def send_error_notification(
        self,
        error_context,  # ErrorContext from error_handler
        include_details: bool = True
    ) -> bool:
        """Send notification for error context with smart formatting."""
        # Determine priority based on error severity
        priority_mapping = {
            "low": NotificationPriority.LOW,
            "medium": NotificationPriority.NORMAL,
            "high": NotificationPriority.HIGH,
            "critical": NotificationPriority.CRITICAL
        }
        priority = priority_mapping.get(error_context.severity.value, NotificationPriority.NORMAL)
        
        # Determine channels based on priority
        channels = []
        if priority in [NotificationPriority.HIGH, NotificationPriority.CRITICAL]:
            channels = [NotificationChannel.EMAIL, NotificationChannel.SLACK, NotificationChannel.LOG]
        elif priority == NotificationPriority.NORMAL:
            channels = [NotificationChannel.SLACK, NotificationChannel.LOG]
        else:
            channels = [NotificationChannel.LOG]
        
        # Create title and message
        title = f"{error_context.category.value.upper()} Error - {error_context.severity.value.upper()}"
        
        message_parts = [f"Error ID: {error_context.error_id}"]
        if error_context.service_name:
            message_parts.append(f"Service: {error_context.service_name}")
        message_parts.append(f"Message: {error_context.message}")
        
        if include_details and error_context.traceback_str:
            message_parts.append(f"Traceback:\n{error_context.traceback_str[:500]}...")
        
        message = "\n".join(message_parts)
        
        # Create deduplication key
        dedup_key = f"error:{error_context.category.value}:{hash(error_context.message)}"
        
        # Prepare metadata
        metadata = {
            "error_id": error_context.error_id,
            "category": error_context.category.value,
            "severity": error_context.severity.value,
            "service_name": error_context.service_name,
            "user_id": error_context.user_id,
            "request_id": error_context.request_id,
            **error_context.metadata
        }
        
        return self.send_notification(
            title=title,
            message=message,
            priority=priority,
            channels=channels,
            metadata=metadata,
            deduplication_key=dedup_key
        )
    
    def _log_handler(self, message: NotificationMessage) -> DeliveryResult:
        """Log notification handler."""
        log_level = {
            NotificationPriority.LOW: logging.INFO,
            NotificationPriority.NORMAL: logging.WARNING,
            NotificationPriority.HIGH: logging.ERROR,
            NotificationPriority.CRITICAL: logging.CRITICAL
        }.get(message.priority, logging.INFO)
        
        logger.log(log_level, f"[NOTIFICATION] {message.title}: {message.message}")
        
        return DeliveryResult(
            success=True,
            channel=NotificationChannel.LOG,
            message_id=message.id
        )
    
    def _start_background_processor(self):
        """Start background thread for processing notifications."""
        def processor():
            while True:
                try:
                    self._process_notification_batch()
                    time.sleep(1)  # Check every second
                except Exception as e:
                    logger.error(f"Notification processor error: {e}")
                    time.sleep(5)  # Wait longer on error
        
        thread = threading.Thread(target=processor, daemon=True)
        thread.start()
        logger.info("Notification background processor started")
    
    def _process_notification_batch(self):
        """Process a batch of notifications."""
        batch = []
        
        # Get batch of messages
        with self.queue_lock:
            if len(self.message_queue) >= self.config.batch_size:
                batch = self.message_queue[:self.config.batch_size]
                self.message_queue = self.message_queue[self.config.batch_size:]
            elif self.message_queue:
                # Process smaller batch if timeout reached
                oldest_message = self.message_queue[0]
                if time.time() - oldest_message.timestamp > self.config.batch_timeout:
                    batch = self.message_queue[:]
                    self.message_queue.clear()
        
        # Process batch
        for message in batch:
            self._deliver_message(message)
    
    def _deliver_message(self, message: NotificationMessage):
        """Deliver a single message to all its channels."""
        for channel in message.channels:
            if channel not in self.handlers:
                logger.warning(f"No handler for channel: {channel.value}")
                continue
            
            handler = self.handlers[channel]
            
            # Retry delivery
            for attempt in range(self.config.retry_attempts):
                try:
                    if callable(handler):
                        result = handler(message)
                    else:
                        result = handler.send(message)
                    
                    self.delivery_results.append(result)
                    
                    if result.success:
                        logger.debug(f"Message {message.id} delivered via {channel.value}")
                        break
                    else:
                        logger.warning(f"Delivery failed for {message.id} via {channel.value}: {result.error}")
                        if attempt < self.config.retry_attempts - 1:
                            time.sleep(self.config.retry_delay * (attempt + 1))
                
                except Exception as e:
                    logger.error(f"Handler error for {message.id} via {channel.value}: {e}")
                    if attempt < self.config.retry_attempts - 1:
                        time.sleep(self.config.retry_delay * (attempt + 1))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get notification system statistics."""
        # Count deliveries by channel and status
        successful_deliveries = defaultdict(int)
        failed_deliveries = defaultdict(int)
        
        for result in self.delivery_results:
            if result.success:
                successful_deliveries[result.channel.value] += 1
            else:
                failed_deliveries[result.channel.value] += 1
        
        return {
            "config": {
                "enabled": self.config.enabled,
                "channels": [c.value for c in self.config.channels],
                "rate_limit_window": self.config.rate_limit_window,
                "max_notifications_per_window": self.config.max_notifications_per_window
            },
            "queue": {
                "pending_messages": len(self.message_queue)
            },
            "deliveries": {
                "successful": dict(successful_deliveries),
                "failed": dict(failed_deliveries),
                "total_attempts": len(self.delivery_results)
            },
            "rate_limits": {
                "active_keys": len(self.rate_limiter.notifications)
            },
            "deduplication": {
                "seen_keys": len(self.deduplicator.seen_notifications)
            }
        }
    
    def clear_statistics(self):
        """Clear delivery statistics."""
        self.delivery_results.clear()
        logger.info("Notification statistics cleared")


# Global notification system instance
notification_system = EnhancedNotificationSystem()


# Convenience functions
def send_notification(
    title: str,
    message: str,
    priority: NotificationPriority = NotificationPriority.NORMAL,
    channels: Optional[List[NotificationChannel]] = None,
    **kwargs
) -> bool:
    """Send a notification using the global notification system."""
    return notification_system.send_notification(
        title=title,
        message=message,
        priority=priority,
        channels=channels,
        **kwargs
    )


def send_error_notification(error_context, include_details: bool = True) -> bool:
    """Send error notification using the global notification system."""
    return notification_system.send_error_notification(error_context, include_details)


def configure_notifications(
    enabled: bool = True,
    channels: Optional[List[NotificationChannel]] = None,
    rate_limit_window: int = 300,
    max_notifications_per_window: int = 10
):
    """Configure the global notification system."""
    global notification_system
    
    config = NotificationConfig(
        enabled=enabled,
        channels=channels or [NotificationChannel.LOG],
        rate_limit_window=rate_limit_window,
        max_notifications_per_window=max_notifications_per_window
    )
    
    notification_system = EnhancedNotificationSystem(config)
    logger.info("Notification system reconfigured")