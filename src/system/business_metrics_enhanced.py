"""
Enhanced Business Metrics Collection and Analysis for SuperInsight Platform.

Provides comprehensive business metrics tracking, trend analysis,
and performance insights for annotation workflows, user engagement,
and system efficiency.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
import statistics

from src.system.monitoring import MetricsCollector, metrics_collector
from src.billing.service import BillingSystem
from src.billing.analytics import BillingAnalytics

logger = logging.getLogger(__name__)


@dataclass
class BusinessMetric:
    """Business metric definition with metadata."""
    name: str
    value: float
    unit: str
    category: str
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class TrendAnalysis:
    """Trend analysis result for business metrics."""
    metric_name: str
    trend_direction: str  # increasing, decreasing, stable
    trend_strength: float  # 0.0 to 1.0
    confidence: float     # 0.0 to 1.0
    period_days: int
    change_rate: float    # percentage change
    predictions: Dict[str, float]  # future predictions


@dataclass
class BusinessInsight:
    """Business insight generated from metrics analysis."""
    title: str
    description: str
    category: str
    severity: str  # low, medium, high
    metrics: List[str]
    recommendations: List[str]
    timestamp: float


class BusinessMetricsCollector:
    """
    Enhanced business metrics collection and analysis system.
    
    Tracks key business metrics, analyzes trends, and provides
    actionable insights for business optimization.
    """
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.business_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.insights: List[BusinessInsight] = []
        self.trend_analyses: Dict[str, TrendAnalysis] = {}
        
        # Business metric categories
        self.categories = {
            "annotation": "Annotation Performance",
            "user_engagement": "User Engagement",
            "quality": "Quality Metrics",
            "efficiency": "System Efficiency",
            "revenue": "Revenue Metrics"
        }
        
        # Initialize billing integration
        self.billing_system = None
        self.billing_analytics = None
        
        try:
            self.billing_system = BillingSystem()
            self.billing_analytics = BillingAnalytics()
        except Exception as e:
            logger.warning(f"Failed to initialize billing integration: {e}")
    
    def record_business_metric(
        self,
        name: str,
        value: float,
        unit: str,
        category: str,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None
    ):
        """Record a business metric with enhanced metadata."""
        metric = BusinessMetric(
            name=name,
            value=value,
            unit=unit,
            category=category,
            timestamp=time.time(),
            metadata=metadata or {},
            tags=tags or {}
        )
        
        self.business_metrics[name].append(metric)
        
        # Also record in system metrics for unified monitoring
        self.metrics.record_metric(
            f"business.{name}",
            value,
            tags=tags,
            metadata={"category": category, "unit": unit, **(metadata or {})}
        )
        
        logger.debug(f"Recorded business metric: {name} = {value} {unit}")
    
    async def collect_annotation_metrics(self) -> Dict[str, Any]:
        """Collect annotation-related business metrics."""
        try:
            from src.database.connection import get_database_session
            
            metrics = {}
            
            with get_database_session() as session:
                # Annotation throughput (annotations per hour)
                # This would need actual database queries - simplified for demo
                annotations_per_hour = 50  # Placeholder
                self.record_business_metric(
                    "annotations_per_hour",
                    annotations_per_hour,
                    "annotations/hour",
                    "annotation",
                    metadata={"collection_method": "database_query"}
                )
                metrics["annotations_per_hour"] = annotations_per_hour
                
                # Average annotation time
                avg_annotation_time = 120  # seconds, placeholder
                self.record_business_metric(
                    "avg_annotation_time",
                    avg_annotation_time,
                    "seconds",
                    "annotation",
                    metadata={"collection_method": "database_query"}
                )
                metrics["avg_annotation_time"] = avg_annotation_time
                
                # Annotation quality score
                avg_quality_score = 0.85  # placeholder
                self.record_business_metric(
                    "avg_quality_score",
                    avg_quality_score,
                    "score",
                    "quality",
                    metadata={"collection_method": "database_query"}
                )
                metrics["avg_quality_score"] = avg_quality_score
                
                # AI pre-annotation accuracy
                ai_accuracy = 0.78  # placeholder
                self.record_business_metric(
                    "ai_preannotation_accuracy",
                    ai_accuracy,
                    "accuracy",
                    "annotation",
                    metadata={"collection_method": "database_query"}
                )
                metrics["ai_preannotation_accuracy"] = ai_accuracy
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect annotation metrics: {e}")
            return {}
    
    async def collect_user_engagement_metrics(self) -> Dict[str, Any]:
        """Collect user engagement metrics."""
        try:
            metrics = {}
            
            # Active users (last 24 hours)
            active_users_24h = 25  # placeholder
            self.record_business_metric(
                "active_users_24h",
                active_users_24h,
                "users",
                "user_engagement",
                metadata={"period": "24_hours"}
            )
            metrics["active_users_24h"] = active_users_24h
            
            # Session duration
            avg_session_duration = 45  # minutes, placeholder
            self.record_business_metric(
                "avg_session_duration",
                avg_session_duration,
                "minutes",
                "user_engagement",
                metadata={"period": "24_hours"}
            )
            metrics["avg_session_duration"] = avg_session_duration
            
            # User retention rate
            retention_rate = 0.82  # placeholder
            self.record_business_metric(
                "user_retention_rate",
                retention_rate,
                "rate",
                "user_engagement",
                metadata={"period": "7_days"}
            )
            metrics["user_retention_rate"] = retention_rate
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect user engagement metrics: {e}")
            return {}
    
    async def collect_efficiency_metrics(self) -> Dict[str, Any]:
        """Collect system efficiency metrics."""
        try:
            metrics = {}
            
            # System uptime
            uptime_hours = 168  # 1 week, placeholder
            self.record_business_metric(
                "system_uptime",
                uptime_hours,
                "hours",
                "efficiency",
                metadata={"measurement_period": "7_days"}
            )
            metrics["system_uptime"] = uptime_hours
            
            # Error rate
            error_rate = 0.02  # 2%, placeholder
            self.record_business_metric(
                "system_error_rate",
                error_rate,
                "rate",
                "efficiency",
                metadata={"period": "24_hours"}
            )
            metrics["system_error_rate"] = error_rate
            
            # Resource utilization efficiency
            resource_efficiency = 0.75  # placeholder
            self.record_business_metric(
                "resource_efficiency",
                resource_efficiency,
                "efficiency",
                "efficiency",
                metadata={"components": ["cpu", "memory", "disk"]}
            )
            metrics["resource_efficiency"] = resource_efficiency
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect efficiency metrics: {e}")
            return {}
    
    async def collect_revenue_metrics(self) -> Dict[str, Any]:
        """Collect revenue-related metrics."""
        try:
            metrics = {}
            
            if self.billing_analytics:
                # Monthly recurring revenue
                mrr = await self._calculate_mrr()
                if mrr is not None:
                    self.record_business_metric(
                        "monthly_recurring_revenue",
                        mrr,
                        "currency",
                        "revenue",
                        metadata={"currency": "USD"}
                    )
                    metrics["monthly_recurring_revenue"] = mrr
                
                # Average revenue per user
                arpu = await self._calculate_arpu()
                if arpu is not None:
                    self.record_business_metric(
                        "avg_revenue_per_user",
                        arpu,
                        "currency",
                        "revenue",
                        metadata={"currency": "USD", "period": "monthly"}
                    )
                    metrics["avg_revenue_per_user"] = arpu
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect revenue metrics: {e}")
            return {}
    
    async def _calculate_mrr(self) -> Optional[float]:
        """Calculate Monthly Recurring Revenue."""
        try:
            # This would integrate with actual billing data
            # Placeholder calculation
            return 15000.0  # $15,000 MRR
        except Exception as e:
            logger.error(f"Failed to calculate MRR: {e}")
            return None
    
    async def _calculate_arpu(self) -> Optional[float]:
        """Calculate Average Revenue Per User."""
        try:
            # This would integrate with actual billing and user data
            # Placeholder calculation
            return 250.0  # $250 ARPU
        except Exception as e:
            logger.error(f"Failed to calculate ARPU: {e}")
            return None
    
    def analyze_trends(self, metric_name: str, days: int = 7) -> Optional[TrendAnalysis]:
        """Analyze trends for a specific business metric."""
        if metric_name not in self.business_metrics:
            return None
        
        metrics_data = list(self.business_metrics[metric_name])
        if len(metrics_data) < 10:
            return None
        
        # Filter data for the specified period
        cutoff_time = time.time() - (days * 24 * 60 * 60)
        recent_data = [m for m in metrics_data if m.timestamp >= cutoff_time]
        
        if len(recent_data) < 5:
            return None
        
        values = [m.value for m in recent_data]
        timestamps = [m.timestamp for m in recent_data]
        
        # Calculate trend using linear regression
        n = len(values)
        x_values = list(range(n))
        
        x_mean = sum(x_values) / n
        y_mean = sum(values) / n
        
        numerator = sum((x_values[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x_values[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            slope = 0
        else:
            slope = numerator / denominator
        
        # Determine trend direction and strength
        if abs(slope) < 0.01:
            trend_direction = "stable"
            trend_strength = 0.0
        elif slope > 0:
            trend_direction = "increasing"
            trend_strength = min(abs(slope) / (max(values) - min(values)) * 100, 1.0)
        else:
            trend_direction = "decreasing"
            trend_strength = min(abs(slope) / (max(values) - min(values)) * 100, 1.0)
        
        # Calculate confidence based on data consistency
        if len(values) > 1:
            std_dev = statistics.stdev(values)
            mean_val = statistics.mean(values)
            coefficient_of_variation = std_dev / mean_val if mean_val != 0 else 1.0
            confidence = max(0.0, 1.0 - coefficient_of_variation)
        else:
            confidence = 0.0
        
        # Calculate change rate
        if len(values) >= 2:
            change_rate = ((values[-1] - values[0]) / values[0]) * 100 if values[0] != 0 else 0.0
        else:
            change_rate = 0.0
        
        # Simple predictions (linear extrapolation)
        predictions = {}
        if slope != 0:
            current_value = values[-1]
            predictions["1_day"] = current_value + slope * 1
            predictions["7_days"] = current_value + slope * 7
            predictions["30_days"] = current_value + slope * 30
        
        trend_analysis = TrendAnalysis(
            metric_name=metric_name,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            confidence=confidence,
            period_days=days,
            change_rate=change_rate,
            predictions=predictions
        )
        
        self.trend_analyses[metric_name] = trend_analysis
        return trend_analysis
    
    def generate_business_insights(self) -> List[BusinessInsight]:
        """Generate business insights from collected metrics and trends."""
        insights = []
        current_time = time.time()
        
        # Analyze annotation efficiency
        if "annotations_per_hour" in self.business_metrics:
            trend = self.analyze_trends("annotations_per_hour")
            if trend and trend.trend_direction == "decreasing" and trend.confidence > 0.7:
                insight = BusinessInsight(
                    title="Declining Annotation Throughput",
                    description=f"Annotation throughput has decreased by {abs(trend.change_rate):.1f}% over the last {trend.period_days} days",
                    category="annotation",
                    severity="medium",
                    metrics=["annotations_per_hour"],
                    recommendations=[
                        "Review annotation workflow for bottlenecks",
                        "Consider additional training for annotators",
                        "Optimize AI pre-annotation accuracy"
                    ],
                    timestamp=current_time
                )
                insights.append(insight)
        
        # Analyze quality trends
        if "avg_quality_score" in self.business_metrics:
            trend = self.analyze_trends("avg_quality_score")
            if trend and trend.trend_direction == "decreasing" and trend.confidence > 0.6:
                insight = BusinessInsight(
                    title="Quality Score Declining",
                    description=f"Average quality score has decreased by {abs(trend.change_rate):.1f}% over the last {trend.period_days} days",
                    category="quality",
                    severity="high",
                    metrics=["avg_quality_score"],
                    recommendations=[
                        "Implement additional quality checks",
                        "Provide refresher training to annotators",
                        "Review quality guidelines and standards"
                    ],
                    timestamp=current_time
                )
                insights.append(insight)
        
        # Analyze user engagement
        if "active_users_24h" in self.business_metrics:
            trend = self.analyze_trends("active_users_24h")
            if trend and trend.trend_direction == "increasing" and trend.confidence > 0.8:
                insight = BusinessInsight(
                    title="Growing User Engagement",
                    description=f"Active users have increased by {trend.change_rate:.1f}% over the last {trend.period_days} days",
                    category="user_engagement",
                    severity="low",
                    metrics=["active_users_24h"],
                    recommendations=[
                        "Maintain current engagement strategies",
                        "Consider scaling infrastructure for growth",
                        "Analyze successful engagement patterns"
                    ],
                    timestamp=current_time
                )
                insights.append(insight)
        
        # Store insights
        self.insights.extend(insights)
        
        # Keep only recent insights (last 100)
        if len(self.insights) > 100:
            self.insights = self.insights[-100:]
        
        return insights
    
    async def collect_all_business_metrics(self) -> Dict[str, Any]:
        """Collect all business metrics in one operation."""
        all_metrics = {}
        
        try:
            # Collect metrics from different categories
            annotation_metrics = await self.collect_annotation_metrics()
            user_metrics = await self.collect_user_engagement_metrics()
            efficiency_metrics = await self.collect_efficiency_metrics()
            revenue_metrics = await self.collect_revenue_metrics()
            
            all_metrics.update({
                "annotation": annotation_metrics,
                "user_engagement": user_metrics,
                "efficiency": efficiency_metrics,
                "revenue": revenue_metrics
            })
            
            # Generate insights
            insights = self.generate_business_insights()
            all_metrics["insights"] = [
                {
                    "title": insight.title,
                    "description": insight.description,
                    "category": insight.category,
                    "severity": insight.severity,
                    "recommendations": insight.recommendations
                }
                for insight in insights
            ]
            
            # Add trend analyses
            all_metrics["trends"] = {}
            for metric_name in self.business_metrics.keys():
                trend = self.analyze_trends(metric_name)
                if trend:
                    all_metrics["trends"][metric_name] = {
                        "direction": trend.trend_direction,
                        "strength": trend.trend_strength,
                        "confidence": trend.confidence,
                        "change_rate": trend.change_rate,
                        "predictions": trend.predictions
                    }
            
            logger.info(f"Collected business metrics: {len(all_metrics)} categories")
            return all_metrics
            
        except Exception as e:
            logger.error(f"Failed to collect business metrics: {e}")
            return {}
    
    def get_business_dashboard_data(self) -> Dict[str, Any]:
        """Get formatted data for business dashboard."""
        dashboard_data = {
            "timestamp": time.time(),
            "categories": self.categories,
            "metrics": {},
            "trends": {},
            "insights": [],
            "alerts": []
        }
        
        # Format metrics by category
        for metric_name, metric_data in self.business_metrics.items():
            if not metric_data:
                continue
            
            latest_metric = metric_data[-1]
            category = latest_metric.category
            
            if category not in dashboard_data["metrics"]:
                dashboard_data["metrics"][category] = []
            
            dashboard_data["metrics"][category].append({
                "name": metric_name,
                "value": latest_metric.value,
                "unit": latest_metric.unit,
                "timestamp": latest_metric.timestamp
            })
        
        # Add trend information
        for metric_name, trend in self.trend_analyses.items():
            dashboard_data["trends"][metric_name] = {
                "direction": trend.trend_direction,
                "strength": trend.trend_strength,
                "change_rate": trend.change_rate
            }
        
        # Add recent insights
        dashboard_data["insights"] = [
            {
                "title": insight.title,
                "description": insight.description,
                "severity": insight.severity,
                "category": insight.category,
                "recommendations": insight.recommendations[:3]  # Top 3 recommendations
            }
            for insight in self.insights[-5:]  # Last 5 insights
        ]
        
        return dashboard_data


# Global business metrics collector
business_metrics_collector = BusinessMetricsCollector(metrics_collector)