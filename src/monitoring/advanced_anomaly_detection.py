"""
Advanced Anomaly Detection for SuperInsight Platform.

Provides ML-based anomaly detection capabilities including:
- Isolation Forest for multivariate anomaly detection
- EWMA (Exponentially Weighted Moving Average) for trend detection
- Seasonal decomposition for periodic pattern detection
- Alert aggregation and deduplication
- Automated response mechanisms
"""

import logging
import time
import asyncio
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Tuple
from enum import Enum
import hashlib
import json

logger = logging.getLogger(__name__)


class AnomalyType(str, Enum):
    """Types of detected anomalies."""
    POINT = "point"              # Single point anomaly
    CONTEXTUAL = "contextual"    # Anomaly in specific context
    COLLECTIVE = "collective"    # Group of points forming anomaly
    TREND = "trend"              # Trend-based anomaly
    SEASONAL = "seasonal"        # Seasonal pattern deviation


class AnomalySeverity(str, Enum):
    """Anomaly severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DetectedAnomaly:
    """Detected anomaly record."""
    id: str
    metric_name: str
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    timestamp: float
    value: float
    expected_value: float
    deviation_score: float
    context: Dict[str, Any] = field(default_factory=dict)
    aggregated_count: int = 1
    first_detected: float = field(default_factory=time.time)
    last_detected: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "metric_name": self.metric_name,
            "anomaly_type": self.anomaly_type.value,
            "severity": self.severity.value,
            "timestamp": self.timestamp,
            "value": self.value,
            "expected_value": self.expected_value,
            "deviation_score": self.deviation_score,
            "context": self.context,
            "aggregated_count": self.aggregated_count,
            "first_detected": self.first_detected,
            "last_detected": self.last_detected,
            "duration_seconds": self.last_detected - self.first_detected
        }


@dataclass
class AutomatedResponse:
    """Automated response action."""
    action_type: str
    target: str
    parameters: Dict[str, Any]
    triggered_by: str
    timestamp: float
    success: bool = False
    result: Optional[str] = None


class IsolationForest:
    """
    Simple Isolation Forest implementation for anomaly detection.

    Uses random subsampling and isolation trees to detect anomalies
    based on how easily a point can be isolated from others.
    """

    def __init__(self, n_trees: int = 100, sample_size: int = 256):
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.trees: List[Dict[str, Any]] = []
        self._fitted = False

    def fit(self, data: np.ndarray):
        """Fit the isolation forest on training data."""
        if len(data) < 2:
            return

        self.trees = []
        n_samples = min(self.sample_size, len(data))

        for _ in range(self.n_trees):
            # Random subsample
            indices = np.random.choice(len(data), size=n_samples, replace=False)
            subsample = data[indices]

            # Build isolation tree
            tree = self._build_tree(subsample, height_limit=int(np.ceil(np.log2(n_samples))))
            self.trees.append(tree)

        self._fitted = True

    def _build_tree(self, data: np.ndarray, height_limit: int, current_height: int = 0) -> Dict[str, Any]:
        """Build a single isolation tree."""
        if current_height >= height_limit or len(data) <= 1:
            return {"type": "leaf", "size": len(data)}

        # Select random feature and split point
        n_features = data.shape[1] if len(data.shape) > 1 else 1

        if n_features == 1:
            feature_idx = 0
            values = data if len(data.shape) == 1 else data[:, 0]
        else:
            feature_idx = np.random.randint(n_features)
            values = data[:, feature_idx]

        min_val, max_val = values.min(), values.max()

        if min_val == max_val:
            return {"type": "leaf", "size": len(data)}

        split_value = np.random.uniform(min_val, max_val)

        # Split data
        left_mask = values < split_value
        right_mask = ~left_mask

        left_data = data[left_mask]
        right_data = data[right_mask]

        if len(left_data) == 0 or len(right_data) == 0:
            return {"type": "leaf", "size": len(data)}

        return {
            "type": "node",
            "feature": feature_idx,
            "split_value": split_value,
            "left": self._build_tree(left_data, height_limit, current_height + 1),
            "right": self._build_tree(right_data, height_limit, current_height + 1)
        }

    def _path_length(self, point: np.ndarray, tree: Dict[str, Any], current_height: int = 0) -> float:
        """Calculate path length for a point in a tree."""
        if tree["type"] == "leaf":
            n = tree["size"]
            if n > 1:
                # Average path length for unsuccessful search in BST
                c = 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n
            else:
                c = 0
            return current_height + c

        feature_idx = tree["feature"]
        split_value = tree["split_value"]

        value = point[feature_idx] if len(point.shape) > 0 else point

        if value < split_value:
            return self._path_length(point, tree["left"], current_height + 1)
        else:
            return self._path_length(point, tree["right"], current_height + 1)

    def score_samples(self, data: np.ndarray) -> np.ndarray:
        """Score samples for anomaly (higher score = more anomalous)."""
        if not self._fitted:
            return np.zeros(len(data))

        scores = np.zeros(len(data))
        n = self.sample_size
        c = 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n if n > 1 else 0

        for i, point in enumerate(data):
            avg_path_length = np.mean([self._path_length(point, tree) for tree in self.trees])
            # Anomaly score: higher is more anomalous
            scores[i] = 2 ** (-avg_path_length / c) if c > 0 else 0

        return scores

    def predict(self, data: np.ndarray, threshold: float = 0.6) -> np.ndarray:
        """Predict anomalies (1 = anomaly, 0 = normal)."""
        scores = self.score_samples(data)
        return (scores > threshold).astype(int)


class EWMADetector:
    """
    Exponentially Weighted Moving Average detector for trend anomalies.

    Detects anomalies based on deviation from exponentially weighted moving average.
    """

    def __init__(self, alpha: float = 0.3, threshold_std: float = 3.0):
        self.alpha = alpha
        self.threshold_std = threshold_std
        self.ewma: Optional[float] = None
        self.ewma_var: Optional[float] = None

    def update(self, value: float) -> Tuple[bool, float, float]:
        """
        Update EWMA with new value and detect anomaly.

        Returns:
            Tuple of (is_anomaly, expected_value, deviation_score)
        """
        if self.ewma is None:
            self.ewma = value
            self.ewma_var = 0
            return False, value, 0.0

        # Update EWMA
        diff = value - self.ewma
        self.ewma = self.alpha * value + (1 - self.alpha) * self.ewma

        # Update variance estimate
        self.ewma_var = self.alpha * (diff ** 2) + (1 - self.alpha) * self.ewma_var
        std = np.sqrt(self.ewma_var) if self.ewma_var > 0 else 1.0

        # Calculate deviation score
        deviation_score = abs(diff) / std if std > 0 else 0
        is_anomaly = deviation_score > self.threshold_std

        return is_anomaly, self.ewma, deviation_score

    def reset(self):
        """Reset the detector state."""
        self.ewma = None
        self.ewma_var = None


class SeasonalDetector:
    """
    Seasonal pattern detector for periodic anomalies.

    Detects deviations from expected seasonal patterns.
    """

    def __init__(self, period: int = 24, threshold_std: float = 2.5):
        self.period = period
        self.threshold_std = threshold_std
        self.seasonal_means: Dict[int, float] = {}
        self.seasonal_stds: Dict[int, float] = {}
        self.seasonal_counts: Dict[int, int] = defaultdict(int)
        self.history: deque = deque(maxlen=period * 10)

    def update(self, value: float, position: int) -> Tuple[bool, float, float]:
        """
        Update seasonal model and detect anomaly.

        Args:
            value: Current value
            position: Position in seasonal cycle (e.g., hour of day)

        Returns:
            Tuple of (is_anomaly, expected_value, deviation_score)
        """
        position = position % self.period
        self.history.append((position, value))

        # Update seasonal statistics
        if position in self.seasonal_means:
            n = self.seasonal_counts[position]
            old_mean = self.seasonal_means[position]

            # Online mean update
            new_mean = old_mean + (value - old_mean) / (n + 1)

            # Online variance update (Welford's algorithm)
            if position in self.seasonal_stds:
                old_var = self.seasonal_stds[position] ** 2
                new_var = old_var + (value - old_mean) * (value - new_mean) / (n + 1)
                self.seasonal_stds[position] = np.sqrt(max(0, new_var))

            self.seasonal_means[position] = new_mean
        else:
            self.seasonal_means[position] = value
            self.seasonal_stds[position] = 0

        self.seasonal_counts[position] += 1

        # Detect anomaly
        expected = self.seasonal_means[position]
        std = self.seasonal_stds.get(position, 1.0) or 1.0

        deviation_score = abs(value - expected) / std if std > 0 else 0
        is_anomaly = deviation_score > self.threshold_std and self.seasonal_counts[position] > 5

        return is_anomaly, expected, deviation_score


class AlertAggregator:
    """
    Alert aggregation and deduplication system.

    Groups similar alerts to reduce noise and improve actionability.
    """

    def __init__(
        self,
        aggregation_window: int = 300,  # 5 minutes
        max_aggregated_alerts: int = 100
    ):
        self.aggregation_window = aggregation_window
        self.max_aggregated_alerts = max_aggregated_alerts
        self.active_anomalies: Dict[str, DetectedAnomaly] = {}
        self.resolved_anomalies: deque = deque(maxlen=1000)

    def _generate_anomaly_key(self, metric_name: str, anomaly_type: AnomalyType) -> str:
        """Generate unique key for anomaly aggregation."""
        return hashlib.md5(f"{metric_name}:{anomaly_type.value}".encode()).hexdigest()[:12]

    def add_anomaly(
        self,
        metric_name: str,
        anomaly_type: AnomalyType,
        severity: AnomalySeverity,
        value: float,
        expected_value: float,
        deviation_score: float,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[DetectedAnomaly, bool]:
        """
        Add anomaly with aggregation.

        Returns:
            Tuple of (anomaly, is_new) where is_new indicates if this is a new anomaly
        """
        current_time = time.time()
        key = self._generate_anomaly_key(metric_name, anomaly_type)

        if key in self.active_anomalies:
            existing = self.active_anomalies[key]

            # Check if within aggregation window
            if current_time - existing.last_detected < self.aggregation_window:
                # Aggregate
                existing.aggregated_count += 1
                existing.last_detected = current_time
                existing.value = value  # Update to latest value
                existing.deviation_score = max(existing.deviation_score, deviation_score)

                # Upgrade severity if needed
                severity_order = [AnomalySeverity.LOW, AnomalySeverity.MEDIUM,
                                 AnomalySeverity.HIGH, AnomalySeverity.CRITICAL]
                if severity_order.index(severity) > severity_order.index(existing.severity):
                    existing.severity = severity

                return existing, False

        # Create new anomaly
        anomaly = DetectedAnomaly(
            id=f"anom_{key}_{int(current_time)}",
            metric_name=metric_name,
            anomaly_type=anomaly_type,
            severity=severity,
            timestamp=current_time,
            value=value,
            expected_value=expected_value,
            deviation_score=deviation_score,
            context=context or {},
            first_detected=current_time,
            last_detected=current_time
        )

        self.active_anomalies[key] = anomaly

        # Cleanup old anomalies
        self._cleanup_expired()

        return anomaly, True

    def resolve_anomaly(self, metric_name: str, anomaly_type: AnomalyType) -> Optional[DetectedAnomaly]:
        """Resolve an active anomaly."""
        key = self._generate_anomaly_key(metric_name, anomaly_type)

        if key in self.active_anomalies:
            anomaly = self.active_anomalies.pop(key)
            self.resolved_anomalies.append(anomaly)
            return anomaly

        return None

    def _cleanup_expired(self):
        """Remove expired anomalies."""
        current_time = time.time()
        expired_keys = []

        for key, anomaly in self.active_anomalies.items():
            if current_time - anomaly.last_detected > self.aggregation_window * 2:
                expired_keys.append(key)

        for key in expired_keys:
            anomaly = self.active_anomalies.pop(key)
            self.resolved_anomalies.append(anomaly)

    def get_active_anomalies(self) -> List[DetectedAnomaly]:
        """Get all active anomalies."""
        self._cleanup_expired()
        return list(self.active_anomalies.values())

    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregation statistics."""
        active = self.get_active_anomalies()

        by_severity = defaultdict(int)
        by_type = defaultdict(int)
        total_aggregated = 0

        for anomaly in active:
            by_severity[anomaly.severity.value] += 1
            by_type[anomaly.anomaly_type.value] += 1
            total_aggregated += anomaly.aggregated_count

        return {
            "active_count": len(active),
            "resolved_count": len(self.resolved_anomalies),
            "total_aggregated": total_aggregated,
            "by_severity": dict(by_severity),
            "by_type": dict(by_type),
            "aggregation_ratio": total_aggregated / len(active) if active else 0
        }


class AutomatedResponseManager:
    """
    Automated response manager for anomaly remediation.

    Executes predefined responses based on anomaly type and severity.
    """

    def __init__(self):
        self.response_handlers: Dict[str, Callable] = {}
        self.response_rules: List[Dict[str, Any]] = []
        self.response_history: deque = deque(maxlen=500)
        self._setup_default_handlers()

    def _setup_default_handlers(self):
        """Setup default response handlers."""
        self.register_handler("log", self._handle_log)
        self.register_handler("scale", self._handle_scale)
        self.register_handler("restart", self._handle_restart)
        self.register_handler("alert", self._handle_alert)
        self.register_handler("throttle", self._handle_throttle)

    def register_handler(self, action_type: str, handler: Callable):
        """Register a response handler."""
        self.response_handlers[action_type] = handler
        logger.info(f"Registered automated response handler: {action_type}")

    def add_response_rule(
        self,
        metric_pattern: str,
        anomaly_type: AnomalyType,
        min_severity: AnomalySeverity,
        action_type: str,
        action_params: Dict[str, Any],
        cooldown_seconds: int = 300
    ):
        """Add an automated response rule."""
        rule = {
            "metric_pattern": metric_pattern,
            "anomaly_type": anomaly_type,
            "min_severity": min_severity,
            "action_type": action_type,
            "action_params": action_params,
            "cooldown_seconds": cooldown_seconds,
            "last_triggered": 0
        }
        self.response_rules.append(rule)
        logger.info(f"Added response rule for {metric_pattern}")

    async def process_anomaly(self, anomaly: DetectedAnomaly) -> List[AutomatedResponse]:
        """Process anomaly and execute matching response rules."""
        responses = []
        current_time = time.time()

        severity_order = [AnomalySeverity.LOW, AnomalySeverity.MEDIUM,
                         AnomalySeverity.HIGH, AnomalySeverity.CRITICAL]

        for rule in self.response_rules:
            # Check if rule matches
            if not self._matches_pattern(anomaly.metric_name, rule["metric_pattern"]):
                continue

            if anomaly.anomaly_type != rule["anomaly_type"]:
                continue

            if severity_order.index(anomaly.severity) < severity_order.index(rule["min_severity"]):
                continue

            # Check cooldown
            if current_time - rule["last_triggered"] < rule["cooldown_seconds"]:
                continue

            # Execute response
            response = await self._execute_response(
                action_type=rule["action_type"],
                target=anomaly.metric_name,
                parameters=rule["action_params"],
                triggered_by=anomaly.id
            )

            if response:
                responses.append(response)
                rule["last_triggered"] = current_time

        return responses

    def _matches_pattern(self, metric_name: str, pattern: str) -> bool:
        """Check if metric name matches pattern (supports * wildcard)."""
        if pattern == "*":
            return True

        if "*" not in pattern:
            return metric_name == pattern

        # Simple wildcard matching
        parts = pattern.split("*")
        if len(parts) == 2:
            return metric_name.startswith(parts[0]) and metric_name.endswith(parts[1])

        return pattern.replace("*", "") in metric_name

    async def _execute_response(
        self,
        action_type: str,
        target: str,
        parameters: Dict[str, Any],
        triggered_by: str
    ) -> Optional[AutomatedResponse]:
        """Execute a response action."""
        handler = self.response_handlers.get(action_type)

        if not handler:
            logger.warning(f"No handler for action type: {action_type}")
            return None

        response = AutomatedResponse(
            action_type=action_type,
            target=target,
            parameters=parameters,
            triggered_by=triggered_by,
            timestamp=time.time()
        )

        try:
            if asyncio.iscoroutinefunction(handler):
                result = await handler(target, parameters)
            else:
                result = handler(target, parameters)

            response.success = True
            response.result = str(result) if result else "Completed"

            logger.info(f"Executed automated response: {action_type} for {target}")

        except Exception as e:
            response.success = False
            response.result = str(e)
            logger.error(f"Automated response failed: {action_type} - {e}")

        self.response_history.append(response)
        return response

    # Default handlers
    async def _handle_log(self, target: str, params: Dict[str, Any]) -> str:
        """Log the anomaly for investigation."""
        level = params.get("level", "warning")
        message = params.get("message", f"Anomaly detected in {target}")

        log_func = getattr(logger, level, logger.warning)
        log_func(f"[AUTO-RESPONSE] {message}")

        return f"Logged at {level} level"

    async def _handle_scale(self, target: str, params: Dict[str, Any]) -> str:
        """Scale resources (placeholder for actual implementation)."""
        direction = params.get("direction", "up")
        amount = params.get("amount", 1)

        logger.info(f"[AUTO-RESPONSE] Would scale {target} {direction} by {amount}")
        return f"Scale {direction} by {amount} (simulated)"

    async def _handle_restart(self, target: str, params: Dict[str, Any]) -> str:
        """Restart service (placeholder for actual implementation)."""
        service = params.get("service", target)

        logger.info(f"[AUTO-RESPONSE] Would restart service: {service}")
        return f"Restart {service} (simulated)"

    async def _handle_alert(self, target: str, params: Dict[str, Any]) -> str:
        """Send alert notification."""
        channels = params.get("channels", ["internal"])
        priority = params.get("priority", "high")

        logger.info(f"[AUTO-RESPONSE] Would send {priority} alert via {channels} for {target}")
        return f"Alert sent via {channels}"

    async def _handle_throttle(self, target: str, params: Dict[str, Any]) -> str:
        """Apply throttling (placeholder for actual implementation)."""
        limit = params.get("limit", 100)
        duration = params.get("duration", 60)

        logger.info(f"[AUTO-RESPONSE] Would throttle {target} to {limit} for {duration}s")
        return f"Throttled to {limit} (simulated)"

    def get_response_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent response history."""
        responses = list(self.response_history)[-limit:]
        return [
            {
                "action_type": r.action_type,
                "target": r.target,
                "parameters": r.parameters,
                "triggered_by": r.triggered_by,
                "timestamp": r.timestamp,
                "success": r.success,
                "result": r.result
            }
            for r in responses
        ]


class AdvancedAnomalyDetector:
    """
    Advanced anomaly detection system with multiple detection methods.

    Combines:
    - Isolation Forest for multivariate detection
    - EWMA for trend detection
    - Seasonal decomposition for periodic patterns
    - Alert aggregation and deduplication
    - Automated response mechanisms
    """

    def __init__(
        self,
        isolation_threshold: float = 0.6,
        ewma_alpha: float = 0.3,
        seasonal_period: int = 24
    ):
        self.isolation_threshold = isolation_threshold

        # Detectors per metric
        self.isolation_forests: Dict[str, IsolationForest] = {}
        self.ewma_detectors: Dict[str, EWMADetector] = {}
        self.seasonal_detectors: Dict[str, SeasonalDetector] = {}

        # Historical data for training
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

        # Alert aggregation
        self.aggregator = AlertAggregator()

        # Automated response
        self.response_manager = AutomatedResponseManager()

        # Configuration
        self.ewma_alpha = ewma_alpha
        self.seasonal_period = seasonal_period
        self.min_training_samples = 50

        # Detection statistics
        self.detection_stats = {
            "total_points_analyzed": 0,
            "anomalies_detected": 0,
            "by_method": defaultdict(int)
        }

    def _get_ewma_detector(self, metric_name: str) -> EWMADetector:
        """Get or create EWMA detector for metric."""
        if metric_name not in self.ewma_detectors:
            self.ewma_detectors[metric_name] = EWMADetector(alpha=self.ewma_alpha)
        return self.ewma_detectors[metric_name]

    def _get_seasonal_detector(self, metric_name: str) -> SeasonalDetector:
        """Get or create seasonal detector for metric."""
        if metric_name not in self.seasonal_detectors:
            self.seasonal_detectors[metric_name] = SeasonalDetector(period=self.seasonal_period)
        return self.seasonal_detectors[metric_name]

    def _get_isolation_forest(self, metric_name: str) -> IsolationForest:
        """Get or create Isolation Forest for metric."""
        if metric_name not in self.isolation_forests:
            self.isolation_forests[metric_name] = IsolationForest()
        return self.isolation_forests[metric_name]

    def _determine_severity(self, deviation_score: float) -> AnomalySeverity:
        """Determine anomaly severity based on deviation score."""
        if deviation_score >= 5.0:
            return AnomalySeverity.CRITICAL
        elif deviation_score >= 4.0:
            return AnomalySeverity.HIGH
        elif deviation_score >= 3.0:
            return AnomalySeverity.MEDIUM
        else:
            return AnomalySeverity.LOW

    async def analyze_metric(
        self,
        metric_name: str,
        value: float,
        timestamp: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> List[DetectedAnomaly]:
        """
        Analyze a metric value for anomalies using multiple methods.

        Args:
            metric_name: Name of the metric
            value: Current metric value
            timestamp: Optional timestamp (defaults to current time)
            context: Optional context information

        Returns:
            List of detected anomalies (may be empty if no anomalies)
        """
        timestamp = timestamp or time.time()
        detected_anomalies = []

        # Store in history
        self.metric_history[metric_name].append((timestamp, value))
        self.detection_stats["total_points_analyzed"] += 1

        # 1. EWMA-based trend detection
        ewma_detector = self._get_ewma_detector(metric_name)
        is_trend_anomaly, expected_ewma, deviation_ewma = ewma_detector.update(value)

        if is_trend_anomaly:
            severity = self._determine_severity(deviation_ewma)
            anomaly, is_new = self.aggregator.add_anomaly(
                metric_name=metric_name,
                anomaly_type=AnomalyType.TREND,
                severity=severity,
                value=value,
                expected_value=expected_ewma,
                deviation_score=deviation_ewma,
                context=context
            )

            if is_new:
                detected_anomalies.append(anomaly)
                self.detection_stats["anomalies_detected"] += 1
                self.detection_stats["by_method"]["ewma"] += 1

        # 2. Seasonal pattern detection
        hour_of_day = datetime.fromtimestamp(timestamp).hour
        seasonal_detector = self._get_seasonal_detector(metric_name)
        is_seasonal_anomaly, expected_seasonal, deviation_seasonal = seasonal_detector.update(
            value, hour_of_day
        )

        if is_seasonal_anomaly:
            severity = self._determine_severity(deviation_seasonal)
            anomaly, is_new = self.aggregator.add_anomaly(
                metric_name=metric_name,
                anomaly_type=AnomalyType.SEASONAL,
                severity=severity,
                value=value,
                expected_value=expected_seasonal,
                deviation_score=deviation_seasonal,
                context=context
            )

            if is_new:
                detected_anomalies.append(anomaly)
                self.detection_stats["anomalies_detected"] += 1
                self.detection_stats["by_method"]["seasonal"] += 1

        # 3. Isolation Forest detection (requires sufficient history)
        history = list(self.metric_history[metric_name])
        if len(history) >= self.min_training_samples:
            forest = self._get_isolation_forest(metric_name)

            # Retrain periodically (every 100 samples)
            if len(history) % 100 == 0:
                values_array = np.array([v for _, v in history]).reshape(-1, 1)
                forest.fit(values_array)

            if forest._fitted:
                score = forest.score_samples(np.array([[value]]))[0]

                if score > self.isolation_threshold:
                    # Calculate deviation based on score
                    deviation_if = (score - 0.5) * 10  # Scale to comparable range
                    severity = self._determine_severity(deviation_if)

                    # Get expected value from recent mean
                    recent_values = [v for _, v in history[-20:]]
                    expected_if = np.mean(recent_values)

                    anomaly, is_new = self.aggregator.add_anomaly(
                        metric_name=metric_name,
                        anomaly_type=AnomalyType.POINT,
                        severity=severity,
                        value=value,
                        expected_value=expected_if,
                        deviation_score=deviation_if,
                        context=context
                    )

                    if is_new:
                        detected_anomalies.append(anomaly)
                        self.detection_stats["anomalies_detected"] += 1
                        self.detection_stats["by_method"]["isolation_forest"] += 1

        # 4. Process automated responses for new anomalies
        for anomaly in detected_anomalies:
            await self.response_manager.process_anomaly(anomaly)

        return detected_anomalies

    def get_active_anomalies(self) -> List[Dict[str, Any]]:
        """Get all currently active anomalies."""
        return [a.to_dict() for a in self.aggregator.get_active_anomalies()]

    def get_detection_statistics(self) -> Dict[str, Any]:
        """Get detection statistics."""
        return {
            "total_points_analyzed": self.detection_stats["total_points_analyzed"],
            "anomalies_detected": self.detection_stats["anomalies_detected"],
            "detection_rate": (
                self.detection_stats["anomalies_detected"] /
                max(1, self.detection_stats["total_points_analyzed"])
            ),
            "by_method": dict(self.detection_stats["by_method"]),
            "aggregation_stats": self.aggregator.get_statistics(),
            "metrics_tracked": len(self.metric_history)
        }

    def add_response_rule(
        self,
        metric_pattern: str,
        anomaly_type: AnomalyType,
        min_severity: AnomalySeverity,
        action_type: str,
        action_params: Dict[str, Any],
        cooldown_seconds: int = 300
    ):
        """Add an automated response rule."""
        self.response_manager.add_response_rule(
            metric_pattern=metric_pattern,
            anomaly_type=anomaly_type,
            min_severity=min_severity,
            action_type=action_type,
            action_params=action_params,
            cooldown_seconds=cooldown_seconds
        )

    def get_response_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get automated response history."""
        return self.response_manager.get_response_history(limit)

    def reset_metric(self, metric_name: str):
        """Reset all detectors for a metric."""
        if metric_name in self.ewma_detectors:
            self.ewma_detectors[metric_name].reset()

        if metric_name in self.seasonal_detectors:
            del self.seasonal_detectors[metric_name]

        if metric_name in self.isolation_forests:
            del self.isolation_forests[metric_name]

        if metric_name in self.metric_history:
            self.metric_history[metric_name].clear()

        self.aggregator.resolve_anomaly(metric_name, AnomalyType.TREND)
        self.aggregator.resolve_anomaly(metric_name, AnomalyType.SEASONAL)
        self.aggregator.resolve_anomaly(metric_name, AnomalyType.POINT)


# Global instance
advanced_anomaly_detector = AdvancedAnomalyDetector()
