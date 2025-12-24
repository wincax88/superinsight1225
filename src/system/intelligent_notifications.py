"""
Intelligent Notification System for SuperInsight Platform.

Provides smart notification filtering, deduplication, and escalation
to reduce false positives and notification fatigue while ensuring
critical issues are properly communicated.
"""

import logging
import time
import threading
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import hashlib
import json

from src.system.notification import notification_system, NotificationPriority, NotificationChannel
from src.system.error_handler import ErrorCategory, ErrorSeverity

logger = logging.getLogger(__name__)


class NotificationFilterType(Enum):
    """Types of notification filters."""
    FREQUENCY = "frequency"
    PATTERN = "pattern"
    SEVERITY = "severity"
    TIME_BASED = "time_based"
    CORRELATION = "correlation"
    BUSINESS_RULES = "business_rules"


class EscalationLevel(Enum):
    """Escalation levels for notifications."""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class NotificationContext:
    """Enhanced notification context with filtering metadata."""
    error_id: str
    service_name: str
    error_category: ErrorCategory
    error_severity: ErrorSeverity
    message: str
    timestamp: float
    signature: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Filtering context
    similar_count: int = 0
    escalation_level: EscalationLevel = EscalationLevel.NONE
    suppression_reason: Optional[str] = None
    correlation_group: Optional[str] = None


@dataclass
class FilterRule:
    """Notification filter rule configuration."""
    name: str
    filter_type: NotificationFilterType
    enabled: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1  # Higher priority rules are evaluated first


@dataclass
class EscalationRule:
    """Escalation rule for progressive notification enhancement."""
    trigger_count: int
    time_window: float  # seconds
    escalation_level: EscalationLevel
    actions: List[str] = field(default_factory=list)


class IntelligentNotificationFilter:
    """
    Intelligent notification filtering system with machine learning-like capabilities.
    
    Features:
    - Frequency-based filtering with adaptive thresholds
    - Pattern recognition for transient vs persistent issues
    - Correlation analysis for related errors
    - Business rules for context-aware filtering
    - Progressive escalation for persistent issues
    - Learning from user feedback and resolution patterns
    """
    
    def __init__(self):
        self.notification_history: deque = deque(maxlen=2000)
        self.suppression_cache: Dict[str, Dict[str, Any]] = {}
        self.correlation_groups: Dict[str, List[str]] = {}
        self.filter_rules: List[FilterRule] = []
        self.escalation_rules: List[EscalationRule] = []
        self.user_feedback: Dict[str, Dict[str, Any]] = {}
        self.pattern_cache: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.RLock()
        
        # Performance metrics
        self.filter_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {
            'total_processed': 0,
            'filtered': 0,
            'escalated': 0,
            'false_positives': 0,
            'false_negatives': 0
        })
        
        self._initialize_default_rules()
        
    def _initialize_default_rules(self):
        """Initialize default filtering and escalation rules."""
        # Frequency-based filtering
        self.filter_rules.append(FilterRule(
            name="high_frequency_suppression",
            filter_type=NotificationFilterType.FREQUENCY,
            priority=10,
            parameters={
                'window_seconds': 300,  # 5 minutes
                'max_notifications': 3,
                'decay_factor': 0.8,  # Reduce threshold over time
                'burst_protection': True
            }
        ))
        
        # Pattern-based filtering for transient errors
        self.filter_rules.append(FilterRule(
            name="transient_error_suppression",
            filter_type=NotificationFilterType.PATTERN,
            priority=8,
            parameters={
                'patterns': [
                    'timeout', 'connection reset', 'temporary', 'rate limit',
                    'service unavailable', '502', '503', '504', 'network error'
                ],
                'suppress_duration': 600,  # 10 minutes
                'escalation_threshold': 5,
                'pattern_confidence': 0.7
            }
        ))
        
        # Severity-based filtering
        self.filter_rules.append(FilterRule(
            name="severity_based_filtering",
            filter_type=NotificationFilterType.SEVERITY,
            priority=9,
            parameters={
                'low_severity_limit': 2,  # Max 2 low severity per hour
                'medium_severity_limit': 5,  # Max 5 medium severity per hour
                'time_window': 3600  # 1 hour
            }
        ))
        
        # Time-based filtering (business hours)
        self.filter_rules.append(FilterRule(
            name="business_hours_filtering",
            filter_type=NotificationFilterType.TIME_BASED,
            priority=5,
            parameters={
                'business_start': 8,  # 8 AM
                'business_end': 18,   # 6 PM
                'weekend_suppression': True,
                'off_hours_severity_threshold': ErrorSeverity.HIGH,
                'timezone': 'UTC'
            }
        ))
        
        # Correlation-based filtering
        self.filter_rules.append(FilterRule(
            name="correlation_based_filtering",
            filter_type=NotificationFilterType.CORRELATION,
            priority=7,
            parameters={
                'correlation_window': 900,  # 15 minutes
                'min_correlation_strength': 0.6,
                'max_correlated_notifications': 2
            }
        ))
        
        # Business rules filtering
        self.filter_rules.append(FilterRule(
            name="business_rules_filtering",
            filter_type=NotificationFilterType.BUSINESS_RULES,
            priority=6,
            parameters={
                'maintenance_windows': [],
                'known_issues': [],
                'service_priorities': {
                    'critical': ['database', 'authentication', 'billing'],
                    'high': ['ai_annotation', 'quality_check'],
                    'medium': ['data_extraction', 'export'],
                    'low': ['monitoring', 'logging']
                }
            }
        ))
        
        # Escalation rules
        self.escalation_rules.extend([
            EscalationRule(
                trigger_count=3,
                time_window=600,  # 10 minutes
                escalation_level=EscalationLevel.LOW,
                actions=['increase_priority', 'add_context']
            ),
            EscalationRule(
                trigger_count=5,
                time_window=900,  # 15 minutes
                escalation_level=EscalationLevel.MEDIUM,
                actions=['increase_priority', 'add_context', 'notify_oncall']
            ),
            EscalationRule(
                trigger_count=8,
                time_window=1800,  # 30 minutes
                escalation_level=EscalationLevel.HIGH,
                actions=['increase_priority', 'add_context', 'notify_oncall', 'create_incident']
            ),
            EscalationRule(
                trigger_count=12,
                time_window=3600,  # 1 hour
                escalation_level=EscalationLevel.CRITICAL,
                actions=['increase_priority', 'add_context', 'notify_oncall', 'create_incident', 'page_management']
            )
        ])
        
    def should_send_notification(self, error_context) -> Tuple[bool, Optional[NotificationContext]]:
        """
        Determine if a notification should be sent using intelligent filtering.
        
        Returns:
            Tuple of (should_send, notification_context)
        """
        with self.lock:
            # Create notification context
            notification_context = self._create_notification_context(error_context)
            
            # Update statistics
            self.filter_stats['total']['total_processed'] += 1
            
            # Apply filtering rules in priority order
            for rule in sorted(self.filter_rules, key=lambda r: r.priority, reverse=True):
                if not rule.enabled:
                    continue
                    
                should_filter, reason = self._apply_filter_rule(rule, notification_context)
                
                if should_filter:
                    notification_context.suppression_reason = reason
                    self.filter_stats[rule.name]['filtered'] += 1
                    
                    # Check if we should escalate despite filtering
                    should_escalate = self._check_escalation(notification_context)
                    
                    if should_escalate:
                        notification_context.suppression_reason = None
                        self.filter_stats[rule.name]['escalated'] += 1
                        logger.info(f"Escalating filtered notification: {notification_context.error_id}")
                        break
                    else:
                        logger.debug(f"Suppressing notification: {notification_context.error_id} - {reason}")
                        self._record_suppression(notification_context)
                        return False, notification_context
            
            # If we reach here, notification should be sent
            self._record_notification(notification_context)
            return True, notification_context
            
    def _create_notification_context(self, error_context) -> NotificationContext:
        """Create notification context with enhanced metadata."""
        # Create error signature for deduplication
        signature = self._create_error_signature(error_context)
        
        # Count similar recent errors
        similar_count = self._count_similar_errors(signature, time.time() - 3600)  # Last hour
        
        # Determine correlation group
        correlation_group = self._find_correlation_group(error_context)
        
        return NotificationContext(
            error_id=error_context.error_id,
            service_name=error_context.service_name or 'unknown',
            error_category=error_context.category,
            error_severity=error_context.severity,
            message=error_context.message,
            timestamp=time.time(),
            signature=signature,
            similar_count=similar_count,
            correlation_group=correlation_group,
            metadata={
                'user_id': getattr(error_context, 'user_id', None),
                'request_id': getattr(error_context, 'request_id', None),
                'traceback_hash': hash(getattr(error_context, 'traceback_str', '')[:500])
            }
        )
        
    def _create_error_signature(self, error_context) -> str:
        """Create a unique signature for error deduplication."""
        # Normalize message for better grouping
        normalized_message = self._normalize_error_message(error_context.message)
        
        signature_data = {
            'category': error_context.category.value,
            'service': error_context.service_name or 'unknown',
            'message_hash': hashlib.md5(normalized_message.encode()).hexdigest()[:8],
            'severity': error_context.severity.value
        }
        
        return json.dumps(signature_data, sort_keys=True)
        
    def _normalize_error_message(self, message: str) -> str:
        """Normalize error message for better grouping."""
        import re
        
        # Remove timestamps, IDs, and other variable parts
        normalized = re.sub(r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}', '[TIMESTAMP]', message)
        normalized = re.sub(r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b', '[UUID]', normalized)
        normalized = re.sub(r'\b\d+\b', '[NUMBER]', normalized)
        normalized = re.sub(r'\b0x[0-9a-f]+\b', '[HEX]', normalized)
        normalized = re.sub(r'["\']([^"\']*)["\']', '[STRING]', normalized)
        
        return normalized.lower().strip()
        
    def _count_similar_errors(self, signature: str, since_timestamp: float) -> int:
        """Count similar errors since the given timestamp."""
        count = 0
        for notification in self.notification_history:
            if (notification['timestamp'] >= since_timestamp and 
                notification['signature'] == signature):
                count += 1
        return count
        
    def _find_correlation_group(self, error_context) -> Optional[str]:
        """Find correlation group for the error."""
        # Simple correlation based on service and time proximity
        service_name = error_context.service_name or 'unknown'
        current_time = time.time()
        
        # Look for recent errors in the same service
        for notification in reversed(list(self.notification_history)[-50:]):  # Last 50 notifications
            if (current_time - notification['timestamp'] < 300 and  # 5 minutes
                notification.get('service_name') == service_name):
                return f"service_{service_name}_{int(current_time // 300)}"  # 5-minute windows
                
        return None
        
    def _apply_filter_rule(self, rule: FilterRule, context: NotificationContext) -> Tuple[bool, Optional[str]]:
        """Apply a specific filter rule to the notification context."""
        try:
            if rule.filter_type == NotificationFilterType.FREQUENCY:
                return self._apply_frequency_filter(rule, context)
            elif rule.filter_type == NotificationFilterType.PATTERN:
                return self._apply_pattern_filter(rule, context)
            elif rule.filter_type == NotificationFilterType.SEVERITY:
                return self._apply_severity_filter(rule, context)
            elif rule.filter_type == NotificationFilterType.TIME_BASED:
                return self._apply_time_based_filter(rule, context)
            elif rule.filter_type == NotificationFilterType.CORRELATION:
                return self._apply_correlation_filter(rule, context)
            elif rule.filter_type == NotificationFilterType.BUSINESS_RULES:
                return self._apply_business_rules_filter(rule, context)
            else:
                return False, None
                
        except Exception as e:
            logger.error(f"Error applying filter rule {rule.name}: {e}")
            return False, None
            
    def _apply_frequency_filter(self, rule: FilterRule, context: NotificationContext) -> Tuple[bool, Optional[str]]:
        """Apply frequency-based filtering."""
        params = rule.parameters
        window_seconds = params.get('window_seconds', 300)
        max_notifications = params.get('max_notifications', 3)
        
        # Count recent similar notifications
        window_start = context.timestamp - window_seconds
        recent_count = sum(
            1 for notification in self.notification_history
            if (notification['timestamp'] >= window_start and
                notification['signature'] == context.signature)
        )
        
        if recent_count >= max_notifications:
            return True, f"Frequency limit exceeded: {recent_count}/{max_notifications} in {window_seconds}s"
            
        return False, None
        
    def _apply_pattern_filter(self, rule: FilterRule, context: NotificationContext) -> Tuple[bool, Optional[str]]:
        """Apply pattern-based filtering for transient errors."""
        params = rule.parameters
        patterns = params.get('patterns', [])
        suppress_duration = params.get('suppress_duration', 600)
        escalation_threshold = params.get('escalation_threshold', 5)
        
        message_lower = context.message.lower()
        
        # Check if message matches transient patterns
        matching_patterns = [pattern for pattern in patterns if pattern in message_lower]
        
        if not matching_patterns:
            return False, None
            
        # Check if we should suppress based on recent similar errors
        if context.similar_count < escalation_threshold:
            return True, f"Transient error pattern detected: {matching_patterns[0]}"
            
        return False, None
        
    def _apply_severity_filter(self, rule: FilterRule, context: NotificationContext) -> Tuple[bool, Optional[str]]:
        """Apply severity-based filtering."""
        params = rule.parameters
        time_window = params.get('time_window', 3600)
        
        # Get severity limits
        severity_limits = {
            ErrorSeverity.LOW: params.get('low_severity_limit', 2),
            ErrorSeverity.MEDIUM: params.get('medium_severity_limit', 5),
            ErrorSeverity.HIGH: params.get('high_severity_limit', 10),
            ErrorSeverity.CRITICAL: params.get('critical_severity_limit', 999)  # No limit for critical
        }
        
        limit = severity_limits.get(context.error_severity, 999)
        
        if limit == 999:  # No limit
            return False, None
            
        # Count recent notifications of same severity
        window_start = context.timestamp - time_window
        severity_count = sum(
            1 for notification in self.notification_history
            if (notification['timestamp'] >= window_start and
                notification.get('error_severity') == context.error_severity.value)
        )
        
        if severity_count >= limit:
            return True, f"Severity limit exceeded: {severity_count}/{limit} {context.error_severity.value} errors in {time_window}s"
            
        return False, None
        
    def _apply_time_based_filter(self, rule: FilterRule, context: NotificationContext) -> Tuple[bool, Optional[str]]:
        """Apply time-based filtering (business hours, weekends, etc.)."""
        params = rule.parameters
        business_start = params.get('business_start', 8)
        business_end = params.get('business_end', 18)
        weekend_suppression = params.get('weekend_suppression', True)
        off_hours_threshold = params.get('off_hours_severity_threshold', ErrorSeverity.HIGH)
        
        import datetime
        dt = datetime.datetime.fromtimestamp(context.timestamp)
        
        # Check if it's weekend
        if weekend_suppression and dt.weekday() >= 5:  # Saturday = 5, Sunday = 6
            if context.error_severity.value not in ['high', 'critical']:
                return True, "Weekend suppression for non-critical errors"
                
        # Check if it's off-hours
        if dt.hour < business_start or dt.hour >= business_end:
            if context.error_severity != off_hours_threshold and context.error_severity != ErrorSeverity.CRITICAL:
                return True, f"Off-hours suppression for {context.error_severity.value} severity"
                
        return False, None
        
    def _apply_correlation_filter(self, rule: FilterRule, context: NotificationContext) -> Tuple[bool, Optional[str]]:
        """Apply correlation-based filtering."""
        params = rule.parameters
        correlation_window = params.get('correlation_window', 900)
        max_correlated = params.get('max_correlated_notifications', 2)
        
        if not context.correlation_group:
            return False, None
            
        # Count recent notifications in the same correlation group
        window_start = context.timestamp - correlation_window
        correlated_count = sum(
            1 for notification in self.notification_history
            if (notification['timestamp'] >= window_start and
                notification.get('correlation_group') == context.correlation_group)
        )
        
        if correlated_count >= max_correlated:
            return True, f"Correlation limit exceeded: {correlated_count}/{max_correlated} in group {context.correlation_group}"
            
        return False, None
        
    def _apply_business_rules_filter(self, rule: FilterRule, context: NotificationContext) -> Tuple[bool, Optional[str]]:
        """Apply business rules filtering."""
        params = rule.parameters
        
        # Check maintenance windows
        maintenance_windows = params.get('maintenance_windows', [])
        for window in maintenance_windows:
            if self._is_in_maintenance_window(context.timestamp, window):
                return True, f"Maintenance window: {window['name']}"
                
        # Check known issues
        known_issues = params.get('known_issues', [])
        for issue in known_issues:
            if self._matches_known_issue(context, issue):
                return True, f"Known issue: {issue['name']}"
                
        # Check service priorities
        service_priorities = params.get('service_priorities', {})
        service_priority = self._get_service_priority(context.service_name, service_priorities)
        
        # Apply priority-based filtering
        if service_priority == 'low' and context.error_severity == ErrorSeverity.LOW:
            return True, "Low priority service with low severity error"
            
        return False, None
        
    def _is_in_maintenance_window(self, timestamp: float, window: Dict[str, Any]) -> bool:
        """Check if timestamp is within a maintenance window."""
        # Simplified implementation
        start_time = window.get('start_time')
        end_time = window.get('end_time')
        
        if not start_time or not end_time:
            return False
            
        import datetime
        dt = datetime.datetime.fromtimestamp(timestamp)
        
        # This is a simplified check - real implementation would handle
        # recurring windows, timezones, etc.
        return start_time <= dt.hour < end_time
        
    def _matches_known_issue(self, context: NotificationContext, issue: Dict[str, Any]) -> bool:
        """Check if the error matches a known issue."""
        # Check service match
        if issue.get('service') and issue['service'] != context.service_name:
            return False
            
        # Check pattern match
        if issue.get('pattern'):
            if issue['pattern'].lower() not in context.message.lower():
                return False
                
        # Check category match
        if issue.get('category') and issue['category'] != context.error_category.value:
            return False
            
        return True
        
    def _get_service_priority(self, service_name: str, priorities: Dict[str, List[str]]) -> str:
        """Get priority level for a service."""
        for priority, services in priorities.items():
            if service_name in services:
                return priority
        return 'medium'  # Default priority
        
    def _check_escalation(self, context: NotificationContext) -> bool:
        """Check if a suppressed notification should be escalated."""
        # Check escalation rules
        for rule in self.escalation_rules:
            if self._should_escalate_by_rule(context, rule):
                context.escalation_level = rule.escalation_level
                self._apply_escalation_actions(context, rule.actions)
                return True
                
        return False
        
    def _should_escalate_by_rule(self, context: NotificationContext, rule: EscalationRule) -> bool:
        """Check if context meets escalation rule criteria."""
        window_start = context.timestamp - rule.time_window
        
        # Count similar errors in the time window
        similar_count = sum(
            1 for notification in self.notification_history
            if (notification['timestamp'] >= window_start and
                notification['signature'] == context.signature)
        )
        
        return similar_count >= rule.trigger_count
        
    def _apply_escalation_actions(self, context: NotificationContext, actions: List[str]):
        """Apply escalation actions to the notification context."""
        for action in actions:
            if action == 'increase_priority':
                # This would be handled by the caller
                pass
            elif action == 'add_context':
                context.metadata['escalated'] = True
                context.metadata['escalation_reason'] = f"Escalated due to {context.similar_count} similar errors"
            elif action == 'notify_oncall':
                context.metadata['notify_oncall'] = True
            elif action == 'create_incident':
                context.metadata['create_incident'] = True
            elif action == 'page_management':
                context.metadata['page_management'] = True
                
    def _record_notification(self, context: NotificationContext):
        """Record a sent notification."""
        record = {
            'timestamp': context.timestamp,
            'signature': context.signature,
            'error_id': context.error_id,
            'service_name': context.service_name,
            'error_category': context.error_category.value,
            'error_severity': context.error_severity.value,
            'escalation_level': context.escalation_level.value,
            'correlation_group': context.correlation_group
        }
        
        self.notification_history.append(record)
        
    def _record_suppression(self, context: NotificationContext):
        """Record a suppressed notification."""
        suppression_key = f"{context.signature}_{int(context.timestamp // 300)}"  # 5-minute buckets
        
        if suppression_key not in self.suppression_cache:
            self.suppression_cache[suppression_key] = {
                'count': 0,
                'first_seen': context.timestamp,
                'last_seen': context.timestamp,
                'reason': context.suppression_reason
            }
            
        self.suppression_cache[suppression_key]['count'] += 1
        self.suppression_cache[suppression_key]['last_seen'] = context.timestamp
        
    def add_filter_rule(self, rule: FilterRule):
        """Add a custom filter rule."""
        with self.lock:
            self.filter_rules.append(rule)
            logger.info(f"Added filter rule: {rule.name}")
            
    def remove_filter_rule(self, rule_name: str):
        """Remove a filter rule by name."""
        with self.lock:
            self.filter_rules = [rule for rule in self.filter_rules if rule.name != rule_name]
            logger.info(f"Removed filter rule: {rule_name}")
            
    def update_filter_rule(self, rule_name: str, updates: Dict[str, Any]):
        """Update a filter rule."""
        with self.lock:
            for rule in self.filter_rules:
                if rule.name == rule_name:
                    for key, value in updates.items():
                        if hasattr(rule, key):
                            setattr(rule, key, value)
                        elif key == 'parameters':
                            rule.parameters.update(value)
                    logger.info(f"Updated filter rule: {rule_name}")
                    break
                    
    def provide_feedback(self, error_id: str, feedback_type: str, details: Dict[str, Any]):
        """Provide feedback on notification filtering decisions."""
        with self.lock:
            self.user_feedback[error_id] = {
                'feedback_type': feedback_type,  # 'false_positive', 'false_negative', 'correct'
                'details': details,
                'timestamp': time.time()
            }
            
            # Update filter statistics
            if feedback_type == 'false_positive':
                # Find which rule caused the false positive and update stats
                for rule_name in self.filter_stats:
                    self.filter_stats[rule_name]['false_positives'] += 1
            elif feedback_type == 'false_negative':
                for rule_name in self.filter_stats:
                    self.filter_stats[rule_name]['false_negatives'] += 1
                    
            logger.info(f"Received feedback for {error_id}: {feedback_type}")
            
    def get_filter_statistics(self) -> Dict[str, Any]:
        """Get comprehensive filtering statistics."""
        with self.lock:
            total_stats = {
                'total_processed': sum(stats['total_processed'] for stats in self.filter_stats.values()),
                'total_filtered': sum(stats['filtered'] for stats in self.filter_stats.values()),
                'total_escalated': sum(stats['escalated'] for stats in self.filter_stats.values()),
                'filter_rate': 0.0,
                'escalation_rate': 0.0
            }
            
            if total_stats['total_processed'] > 0:
                total_stats['filter_rate'] = total_stats['total_filtered'] / total_stats['total_processed']
                total_stats['escalation_rate'] = total_stats['total_escalated'] / total_stats['total_processed']
                
            return {
                'total_stats': total_stats,
                'rule_stats': dict(self.filter_stats),
                'notification_history_size': len(self.notification_history),
                'suppression_cache_size': len(self.suppression_cache),
                'active_rules': len([rule for rule in self.filter_rules if rule.enabled]),
                'feedback_count': len(self.user_feedback)
            }
            
    def optimize_filters(self):
        """Optimize filter rules based on feedback and performance."""
        with self.lock:
            logger.info("Optimizing notification filters based on feedback")
            
            # Analyze feedback to adjust filter parameters
            false_positive_count = sum(
                1 for feedback in self.user_feedback.values()
                if feedback['feedback_type'] == 'false_positive'
            )
            
            false_negative_count = sum(
                1 for feedback in self.user_feedback.values()
                if feedback['feedback_type'] == 'false_negative'
            )
            
            # Adjust filter aggressiveness based on feedback
            if false_positive_count > false_negative_count * 2:
                # Too many false positives, make filters less aggressive
                self._reduce_filter_aggressiveness()
            elif false_negative_count > false_positive_count * 2:
                # Too many false negatives, make filters more aggressive
                self._increase_filter_aggressiveness()
                
            logger.info(f"Filter optimization completed. FP: {false_positive_count}, FN: {false_negative_count}")
            
    def _reduce_filter_aggressiveness(self):
        """Reduce filter aggressiveness to reduce false positives."""
        for rule in self.filter_rules:
            if rule.filter_type == NotificationFilterType.FREQUENCY:
                # Increase max notifications allowed
                current_max = rule.parameters.get('max_notifications', 3)
                rule.parameters['max_notifications'] = min(current_max + 1, 10)
            elif rule.filter_type == NotificationFilterType.PATTERN:
                # Increase escalation threshold
                current_threshold = rule.parameters.get('escalation_threshold', 5)
                rule.parameters['escalation_threshold'] = max(current_threshold - 1, 2)
                
    def _increase_filter_aggressiveness(self):
        """Increase filter aggressiveness to reduce false negatives."""
        for rule in self.filter_rules:
            if rule.filter_type == NotificationFilterType.FREQUENCY:
                # Decrease max notifications allowed
                current_max = rule.parameters.get('max_notifications', 3)
                rule.parameters['max_notifications'] = max(current_max - 1, 1)
            elif rule.filter_type == NotificationFilterType.PATTERN:
                # Decrease escalation threshold
                current_threshold = rule.parameters.get('escalation_threshold', 5)
                rule.parameters['escalation_threshold'] = min(current_threshold + 1, 10)


# Global instance
intelligent_notification_filter = IntelligentNotificationFilter()


def filter_notification(error_context) -> Tuple[bool, Optional[NotificationContext]]:
    """
    Main entry point for intelligent notification filtering.
    
    Returns:
        Tuple of (should_send, notification_context)
    """
    return intelligent_notification_filter.should_send_notification(error_context)


def provide_notification_feedback(error_id: str, feedback_type: str, details: Dict[str, Any]):
    """Provide feedback on notification filtering decisions."""
    intelligent_notification_filter.provide_feedback(error_id, feedback_type, details)


def get_notification_filter_stats() -> Dict[str, Any]:
    """Get notification filtering statistics."""
    return intelligent_notification_filter.get_filter_statistics()


def optimize_notification_filters():
    """Optimize notification filters based on feedback."""
    intelligent_notification_filter.optimize_filters()