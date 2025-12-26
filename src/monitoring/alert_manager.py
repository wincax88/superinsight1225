"""
Alert management for SuperInsight Platform.

Provides:
- Alert creation and tracking
- Notification dispatch (WeChat Work, Email)
- Alert escalation
- Alert statistics
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from uuid import UUID, uuid4
from enum import Enum
from dataclasses import dataclass, field
import asyncio
import json

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(str, Enum):
    """Alert status."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SILENCED = "silenced"


class NotificationChannel(str, Enum):
    """Notification channels."""
    WECHAT_WORK = "wechat_work"  # 企业微信
    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"
    INTERNAL = "internal"


@dataclass
class Alert:
    """Alert record."""
    id: UUID
    alert_type: str
    severity: AlertSeverity
    title: str
    message: str
    source: str  # Component that generated the alert
    status: AlertStatus = AlertStatus.ACTIVE
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    notifications_sent: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "alert_type": self.alert_type,
            "severity": self.severity.value,
            "title": self.title,
            "message": self.message,
            "source": self.source,
            "status": self.status.value,
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "context": self.context,
            "created_at": self.created_at.isoformat(),
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "acknowledged_by": self.acknowledged_by,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "resolved_by": self.resolved_by,
            "notifications_sent": self.notifications_sent
        }


@dataclass
class NotificationConfig:
    """Notification configuration."""
    channel: NotificationChannel
    enabled: bool = True
    min_severity: AlertSeverity = AlertSeverity.WARNING
    recipients: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)


class AlertManager:
    """
    Alert management engine.

    Handles alert creation, notification dispatch, and lifecycle management.
    """

    # Default escalation times (in minutes)
    ESCALATION_CONFIG = {
        AlertSeverity.CRITICAL: 5,   # Escalate after 5 minutes
        AlertSeverity.HIGH: 15,      # Escalate after 15 minutes
        AlertSeverity.WARNING: 60,   # Escalate after 1 hour
        AlertSeverity.INFO: 0        # Never escalate
    }

    # Silence duration presets (in minutes)
    SILENCE_PRESETS = {
        "15m": 15,
        "1h": 60,
        "4h": 240,
        "24h": 1440
    }

    def __init__(self):
        """Initialize the alert manager."""
        self._alerts: Dict[UUID, Alert] = {}
        self._notification_configs: Dict[str, NotificationConfig] = {}
        self._silence_rules: List[Dict[str, Any]] = []
        self._notification_handlers: Dict[NotificationChannel, Callable] = {}
        self._escalation_task: Optional[asyncio.Task] = None

        # Register default handlers
        self._register_default_handlers()

    def _register_default_handlers(self):
        """Register default notification handlers."""
        self._notification_handlers[NotificationChannel.INTERNAL] = self._send_internal_notification
        self._notification_handlers[NotificationChannel.WEBHOOK] = self._send_webhook_notification

    async def create_alert(
        self,
        alert_type: str,
        severity: AlertSeverity,
        title: str,
        message: str,
        source: str,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Alert:
        """
        Create a new alert.

        Args:
            alert_type: Type of alert
            severity: Alert severity
            title: Alert title
            message: Alert message
            source: Source component
            tenant_id: Optional tenant
            user_id: Optional user
            context: Additional context

        Returns:
            Created alert
        """
        # Check silence rules
        if self._is_silenced(alert_type, source, tenant_id):
            logger.info(f"Alert silenced: {alert_type} from {source}")
            return None

        alert = Alert(
            id=uuid4(),
            alert_type=alert_type,
            severity=severity,
            title=title,
            message=message,
            source=source,
            tenant_id=tenant_id,
            user_id=user_id,
            context=context or {}
        )

        self._alerts[alert.id] = alert

        logger.info(f"Alert created: {alert.id} - {severity.value} - {title}")

        # Send notifications
        await self._dispatch_notifications(alert)

        return alert

    async def _dispatch_notifications(self, alert: Alert):
        """Dispatch notifications for an alert."""
        for config_name, config in self._notification_configs.items():
            if not config.enabled:
                continue

            # Check severity threshold
            severity_order = [AlertSeverity.INFO, AlertSeverity.WARNING,
                             AlertSeverity.HIGH, AlertSeverity.CRITICAL]
            if severity_order.index(alert.severity) < severity_order.index(config.min_severity):
                continue

            # Get handler
            handler = self._notification_handlers.get(config.channel)
            if handler:
                try:
                    await handler(alert, config)
                    alert.notifications_sent.append(config.channel.value)
                    logger.info(f"Notification sent: {config.channel.value} for alert {alert.id}")
                except Exception as e:
                    logger.error(f"Failed to send notification via {config.channel.value}: {e}")

    async def _send_internal_notification(self, alert: Alert, config: NotificationConfig):
        """Send internal notification (logging)."""
        logger.warning(
            f"[ALERT-{alert.severity.value.upper()}] {alert.title}: {alert.message}"
        )

    async def _send_webhook_notification(self, alert: Alert, config: NotificationConfig):
        """Send webhook notification."""
        webhook_url = config.config.get("url")
        if not webhook_url:
            return

        payload = {
            "alert_id": str(alert.id),
            "alert_type": alert.alert_type,
            "severity": alert.severity.value,
            "title": alert.title,
            "message": alert.message,
            "source": alert.source,
            "timestamp": alert.created_at.isoformat(),
            "context": alert.context
        }

        # In production, would use aiohttp or httpx
        logger.info(f"Webhook notification would be sent to: {webhook_url}")
        logger.debug(f"Payload: {json.dumps(payload)}")

    async def send_wechat_work_notification(
        self,
        alert: Alert,
        config: NotificationConfig
    ):
        """
        Send notification via WeChat Work (企业微信).

        Args:
            alert: Alert to send
            config: Notification configuration
        """
        webhook_key = config.config.get("webhook_key")
        if not webhook_key:
            logger.error("WeChat Work webhook key not configured")
            return

        # Format message for WeChat Work
        severity_emoji = {
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.HIGH: "🔴",
            AlertSeverity.CRITICAL: "🚨"
        }

        content = (
            f"{severity_emoji.get(alert.severity, '')} **{alert.title}**\n\n"
            f"**严重程度**: {alert.severity.value}\n"
            f"**来源**: {alert.source}\n"
            f"**时间**: {alert.created_at.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"{alert.message}"
        )

        # WeChat Work webhook format
        payload = {
            "msgtype": "markdown",
            "markdown": {
                "content": content
            }
        }

        webhook_url = f"https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key={webhook_key}"

        # In production, would send HTTP request
        logger.info(f"WeChat Work notification prepared for: {webhook_url}")
        logger.debug(f"Content: {content}")

    async def send_email_notification(
        self,
        alert: Alert,
        config: NotificationConfig
    ):
        """
        Send email notification.

        Args:
            alert: Alert to send
            config: Notification configuration
        """
        recipients = config.recipients
        if not recipients:
            logger.warning("No email recipients configured")
            return

        subject = f"[{alert.severity.value.upper()}] {alert.title}"
        body = (
            f"Alert Type: {alert.alert_type}\n"
            f"Severity: {alert.severity.value}\n"
            f"Source: {alert.source}\n"
            f"Time: {alert.created_at.isoformat()}\n\n"
            f"Message:\n{alert.message}\n\n"
            f"Context:\n{json.dumps(alert.context, indent=2)}"
        )

        # In production, would use email library
        logger.info(f"Email notification prepared for: {recipients}")
        logger.debug(f"Subject: {subject}")

    async def acknowledge_alert(
        self,
        alert_id: UUID,
        acknowledged_by: str,
        notes: Optional[str] = None
    ) -> bool:
        """
        Acknowledge an alert.

        Args:
            alert_id: Alert UUID
            acknowledged_by: User acknowledging
            notes: Optional notes

        Returns:
            True if acknowledged
        """
        alert = self._alerts.get(alert_id)
        if not alert:
            return False

        if alert.status != AlertStatus.ACTIVE:
            return False

        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_at = datetime.now()
        alert.acknowledged_by = acknowledged_by

        if notes:
            alert.context["acknowledgment_notes"] = notes

        logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
        return True

    async def resolve_alert(
        self,
        alert_id: UUID,
        resolved_by: str,
        resolution_notes: Optional[str] = None
    ) -> bool:
        """
        Resolve an alert.

        Args:
            alert_id: Alert UUID
            resolved_by: User resolving
            resolution_notes: Resolution notes

        Returns:
            True if resolved
        """
        alert = self._alerts.get(alert_id)
        if not alert:
            return False

        if alert.status == AlertStatus.RESOLVED:
            return False

        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.now()
        alert.resolved_by = resolved_by

        if resolution_notes:
            alert.context["resolution_notes"] = resolution_notes

        logger.info(f"Alert resolved: {alert_id} by {resolved_by}")
        return True

    def silence_alert_type(
        self,
        alert_type: str,
        duration_minutes: int,
        source: Optional[str] = None,
        tenant_id: Optional[str] = None
    ) -> str:
        """
        Silence an alert type for a duration.

        Args:
            alert_type: Type to silence
            duration_minutes: Duration in minutes
            source: Optional source filter
            tenant_id: Optional tenant filter

        Returns:
            Silence rule ID
        """
        rule_id = str(uuid4())
        rule = {
            "id": rule_id,
            "alert_type": alert_type,
            "source": source,
            "tenant_id": tenant_id,
            "expires_at": datetime.now() + timedelta(minutes=duration_minutes),
            "created_at": datetime.now()
        }

        self._silence_rules.append(rule)
        logger.info(f"Silence rule created: {rule_id} for {alert_type}")
        return rule_id

    def _is_silenced(
        self,
        alert_type: str,
        source: str,
        tenant_id: Optional[str]
    ) -> bool:
        """Check if an alert type is silenced."""
        now = datetime.now()

        # Clean expired rules
        self._silence_rules = [
            r for r in self._silence_rules
            if r["expires_at"] > now
        ]

        # Check matching rules
        for rule in self._silence_rules:
            if rule["alert_type"] != alert_type:
                continue
            if rule["source"] and rule["source"] != source:
                continue
            if rule["tenant_id"] and rule["tenant_id"] != tenant_id:
                continue
            return True

        return False

    async def get_active_alerts(
        self,
        tenant_id: Optional[str] = None,
        severity: Optional[AlertSeverity] = None,
        source: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get active alerts.

        Args:
            tenant_id: Filter by tenant
            severity: Filter by severity
            source: Filter by source
            limit: Max results

        Returns:
            List of active alerts
        """
        alerts = [
            a for a in self._alerts.values()
            if a.status == AlertStatus.ACTIVE
        ]

        if tenant_id:
            alerts = [a for a in alerts if a.tenant_id == tenant_id]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        if source:
            alerts = [a for a in alerts if a.source == source]

        # Sort by severity and created_at
        severity_order = {
            AlertSeverity.CRITICAL: 0,
            AlertSeverity.HIGH: 1,
            AlertSeverity.WARNING: 2,
            AlertSeverity.INFO: 3
        }
        alerts.sort(key=lambda x: (severity_order[x.severity], -x.created_at.timestamp()))

        return [a.to_dict() for a in alerts[:limit]]

    async def get_alert_statistics(
        self,
        days: int = 7,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get alert statistics.

        Args:
            days: Analysis period
            tenant_id: Filter by tenant

        Returns:
            Statistics dictionary
        """
        cutoff = datetime.now() - timedelta(days=days)

        alerts = [
            a for a in self._alerts.values()
            if a.created_at >= cutoff
        ]

        if tenant_id:
            alerts = [a for a in alerts if a.tenant_id == tenant_id]

        # Count by severity
        by_severity = {}
        for severity in AlertSeverity:
            by_severity[severity.value] = len([
                a for a in alerts if a.severity == severity
            ])

        # Count by status
        by_status = {}
        for status in AlertStatus:
            by_status[status.value] = len([
                a for a in alerts if a.status == status
            ])

        # Count by type
        by_type = {}
        for alert in alerts:
            by_type[alert.alert_type] = by_type.get(alert.alert_type, 0) + 1

        # Calculate resolution metrics
        resolved = [
            a for a in alerts
            if a.status == AlertStatus.RESOLVED
        ]
        resolution_times = []
        for alert in resolved:
            if alert.resolved_at and alert.created_at:
                time_to_resolve = (alert.resolved_at - alert.created_at).total_seconds()
                resolution_times.append(time_to_resolve)

        avg_resolution_time = sum(resolution_times) / len(resolution_times) if resolution_times else 0

        return {
            "period_days": days,
            "total_alerts": len(alerts),
            "by_severity": by_severity,
            "by_status": by_status,
            "by_type": by_type,
            "active_count": by_status.get("active", 0),
            "resolved_count": len(resolved),
            "avg_resolution_time_seconds": round(avg_resolution_time, 2),
            "generated_at": datetime.now().isoformat()
        }

    def configure_notification(
        self,
        name: str,
        channel: NotificationChannel,
        enabled: bool = True,
        min_severity: AlertSeverity = AlertSeverity.WARNING,
        recipients: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Configure a notification channel.

        Args:
            name: Configuration name
            channel: Notification channel
            enabled: Whether enabled
            min_severity: Minimum severity
            recipients: Recipients list
            config: Channel-specific config
        """
        self._notification_configs[name] = NotificationConfig(
            channel=channel,
            enabled=enabled,
            min_severity=min_severity,
            recipients=recipients or [],
            config=config or {}
        )
        logger.info(f"Notification configured: {name} via {channel.value}")

    def get_notification_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get current notification configurations."""
        return {
            name: {
                "channel": config.channel.value,
                "enabled": config.enabled,
                "min_severity": config.min_severity.value,
                "recipients": config.recipients
            }
            for name, config in self._notification_configs.items()
        }
