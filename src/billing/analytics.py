"""
Billing analytics service for SuperInsight platform.

Provides advanced billing analysis, trends, and forecasting.
"""

from datetime import date, datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from decimal import Decimal
from collections import defaultdict
import statistics

from src.billing.service import BillingSystem
from src.billing.models import BillingRecord, BillingReport


class BillingAnalytics:
    """
    Advanced billing analytics and reporting service.
    
    Provides trend analysis, forecasting, and comparative analytics.
    """
    
    def __init__(self, billing_system: BillingSystem):
        """Initialize analytics service with billing system."""
        self.billing_system = billing_system
    
    def calculate_cost_trends(self, tenant_id: str, days: int = 30) -> Dict[str, Any]:
        """
        Calculate cost trends over specified number of days.
        
        Args:
            tenant_id: Tenant identifier
            days: Number of days to analyze
            
        Returns:
            Trend analysis data
        """
        end_date = date.today()
        start_date = end_date - timedelta(days=days)
        
        report = self.billing_system.generate_report(tenant_id, start_date, end_date)
        
        # Calculate daily trends
        daily_costs = []
        daily_annotations = []
        
        for day_str, stats in report.daily_breakdown.items():
            daily_costs.append(stats["cost"])
            daily_annotations.append(stats["annotations"])
        
        # Calculate trend metrics
        avg_daily_cost = statistics.mean(daily_costs) if daily_costs else 0
        avg_daily_annotations = statistics.mean(daily_annotations) if daily_annotations else 0
        
        # Calculate growth rate (comparing first and second half)
        mid_point = len(daily_costs) // 2
        if mid_point > 0:
            first_half_avg = statistics.mean(daily_costs[:mid_point])
            second_half_avg = statistics.mean(daily_costs[mid_point:])
            growth_rate = ((second_half_avg - first_half_avg) / first_half_avg * 100) if first_half_avg > 0 else 0
        else:
            growth_rate = 0
        
        return {
            "period_days": days,
            "total_cost": float(report.total_cost),
            "average_daily_cost": avg_daily_cost,
            "average_daily_annotations": avg_daily_annotations,
            "growth_rate_percent": growth_rate,
            "daily_breakdown": report.daily_breakdown,
            "trend_direction": "increasing" if growth_rate > 5 else "decreasing" if growth_rate < -5 else "stable"
        }
    
    def analyze_user_productivity(self, tenant_id: str, days: int = 30) -> Dict[str, Any]:
        """
        Analyze user productivity and efficiency metrics.
        
        Args:
            tenant_id: Tenant identifier
            days: Number of days to analyze
            
        Returns:
            User productivity analysis
        """
        end_date = date.today()
        start_date = end_date - timedelta(days=days)
        
        report = self.billing_system.generate_report(tenant_id, start_date, end_date)
        
        # Calculate productivity metrics per user
        user_metrics = []
        for user_id, stats in report.user_breakdown.items():
            time_hours = stats["time_spent"] / 3600.0 if stats["time_spent"] > 0 else 0
            annotations_per_hour = stats["annotations"] / time_hours if time_hours > 0 else 0
            cost_per_annotation = stats["cost"] / stats["annotations"] if stats["annotations"] > 0 else 0
            
            user_metrics.append({
                "user_id": user_id,
                "total_annotations": stats["annotations"],
                "total_time_hours": time_hours,
                "annotations_per_hour": annotations_per_hour,
                "cost_per_annotation": cost_per_annotation,
                "total_cost": stats["cost"],
                "efficiency_score": annotations_per_hour * 10  # Simple efficiency metric
            })
        
        # Sort by efficiency
        user_metrics.sort(key=lambda x: x["efficiency_score"], reverse=True)
        
        # Calculate team averages
        if user_metrics:
            avg_annotations_per_hour = statistics.mean([u["annotations_per_hour"] for u in user_metrics])
            avg_cost_per_annotation = statistics.mean([u["cost_per_annotation"] for u in user_metrics])
        else:
            avg_annotations_per_hour = 0
            avg_cost_per_annotation = 0
        
        return {
            "analysis_period_days": days,
            "total_users": len(user_metrics),
            "team_averages": {
                "annotations_per_hour": avg_annotations_per_hour,
                "cost_per_annotation": avg_cost_per_annotation
            },
            "user_rankings": user_metrics,
            "top_performers": user_metrics[:5],
            "improvement_candidates": user_metrics[-5:] if len(user_metrics) > 5 else []
        }
    
    def forecast_monthly_cost(self, tenant_id: str, target_month: str) -> Dict[str, Any]:
        """
        Forecast monthly cost based on historical trends.
        
        Args:
            tenant_id: Tenant identifier
            target_month: Target month in YYYY-MM format
            
        Returns:
            Cost forecast data
        """
        # Get historical data (last 3 months)
        target_date = datetime.strptime(target_month, "%Y-%m").date()
        
        historical_costs = []
        for i in range(3, 0, -1):
            month_date = target_date.replace(day=1) - timedelta(days=i * 30)
            month_str = month_date.strftime("%Y-%m")
            
            try:
                bill = self.billing_system.calculate_monthly_bill(tenant_id, month_str)
                historical_costs.append(float(bill.total_cost))
            except:
                continue
        
        if not historical_costs:
            return {
                "target_month": target_month,
                "forecast_cost": 0.0,
                "confidence": "low",
                "message": "Insufficient historical data for forecasting"
            }
        
        # Simple linear trend forecast
        if len(historical_costs) >= 2:
            # Calculate trend
            x_values = list(range(len(historical_costs)))
            y_values = historical_costs
            
            # Simple linear regression
            n = len(x_values)
            sum_x = sum(x_values)
            sum_y = sum(y_values)
            sum_xy = sum(x * y for x, y in zip(x_values, y_values))
            sum_x2 = sum(x * x for x in x_values)
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            intercept = (sum_y - slope * sum_x) / n
            
            # Forecast next month
            next_x = len(historical_costs)
            forecast_cost = slope * next_x + intercept
            
            # Calculate confidence based on variance
            variance = statistics.variance(historical_costs) if len(historical_costs) > 1 else 0
            confidence = "high" if variance < 1000 else "medium" if variance < 5000 else "low"
        else:
            # Use average if only one data point
            forecast_cost = historical_costs[0]
            confidence = "low"
        
        return {
            "target_month": target_month,
            "forecast_cost": max(0, forecast_cost),  # Ensure non-negative
            "historical_costs": historical_costs,
            "confidence": confidence,
            "trend": "increasing" if len(historical_costs) >= 2 and historical_costs[-1] > historical_costs[0] else "stable"
        }
    
    def compare_periods(self, tenant_id: str, period1_start: date, period1_end: date,
                       period2_start: date, period2_end: date) -> Dict[str, Any]:
        """
        Compare billing metrics between two periods.
        
        Args:
            tenant_id: Tenant identifier
            period1_start: First period start date
            period1_end: First period end date
            period2_start: Second period start date
            period2_end: Second period end date
            
        Returns:
            Period comparison analysis
        """
        # Generate reports for both periods
        report1 = self.billing_system.generate_report(tenant_id, period1_start, period1_end)
        report2 = self.billing_system.generate_report(tenant_id, period2_start, period2_end)
        
        # Calculate changes
        cost_change = float(report2.total_cost - report1.total_cost)
        cost_change_percent = (cost_change / float(report1.total_cost) * 100) if report1.total_cost > 0 else 0
        
        annotation_change = report2.total_annotations - report1.total_annotations
        annotation_change_percent = (annotation_change / report1.total_annotations * 100) if report1.total_annotations > 0 else 0
        
        time_change = report2.total_time_spent - report1.total_time_spent
        time_change_percent = (time_change / report1.total_time_spent * 100) if report1.total_time_spent > 0 else 0
        
        # User activity comparison
        period1_users = set(report1.user_breakdown.keys())
        period2_users = set(report2.user_breakdown.keys())
        
        new_users = period2_users - period1_users
        returning_users = period1_users & period2_users
        inactive_users = period1_users - period2_users
        
        return {
            "period1": {
                "start_date": period1_start.isoformat(),
                "end_date": period1_end.isoformat(),
                "total_cost": float(report1.total_cost),
                "total_annotations": report1.total_annotations,
                "total_time_spent": report1.total_time_spent,
                "active_users": len(period1_users)
            },
            "period2": {
                "start_date": period2_start.isoformat(),
                "end_date": period2_end.isoformat(),
                "total_cost": float(report2.total_cost),
                "total_annotations": report2.total_annotations,
                "total_time_spent": report2.total_time_spent,
                "active_users": len(period2_users)
            },
            "changes": {
                "cost_change": cost_change,
                "cost_change_percent": cost_change_percent,
                "annotation_change": annotation_change,
                "annotation_change_percent": annotation_change_percent,
                "time_change": time_change,
                "time_change_percent": time_change_percent
            },
            "user_activity": {
                "new_users": len(new_users),
                "returning_users": len(returning_users),
                "inactive_users": len(inactive_users),
                "new_user_list": list(new_users),
                "inactive_user_list": list(inactive_users)
            }
        }
    
    def generate_cost_optimization_recommendations(self, tenant_id: str, days: int = 30) -> Dict[str, Any]:
        """
        Generate cost optimization recommendations based on usage patterns.
        
        Args:
            tenant_id: Tenant identifier
            days: Number of days to analyze
            
        Returns:
            Cost optimization recommendations
        """
        # Get productivity analysis
        productivity = self.analyze_user_productivity(tenant_id, days)
        
        # Get cost trends
        trends = self.calculate_cost_trends(tenant_id, days)
        
        recommendations = []
        
        # Check for low productivity users
        if productivity["improvement_candidates"]:
            low_performers = [u for u in productivity["improvement_candidates"] 
                            if u["efficiency_score"] < productivity["team_averages"]["annotations_per_hour"] * 5]
            if low_performers:
                recommendations.append({
                    "type": "productivity",
                    "priority": "high",
                    "title": "Improve Low Performer Productivity",
                    "description": f"Consider training for {len(low_performers)} users with below-average efficiency",
                    "potential_savings": sum(u["total_cost"] * 0.2 for u in low_performers),  # Assume 20% improvement
                    "affected_users": [u["user_id"] for u in low_performers]
                })
        
        # Check for high cost per annotation
        avg_cost = productivity["team_averages"]["cost_per_annotation"]
        if avg_cost > 1.0:  # Threshold for high cost
            recommendations.append({
                "type": "cost_efficiency",
                "priority": "medium",
                "title": "Reduce Cost Per Annotation",
                "description": f"Current cost per annotation (${avg_cost:.2f}) is above optimal range",
                "potential_savings": float(trends["total_cost"]) * 0.15,  # Assume 15% reduction possible
                "suggestion": "Consider optimizing annotation workflows or adjusting billing rates"
            })
        
        # Check for rapid cost growth
        if trends["growth_rate_percent"] > 20:
            recommendations.append({
                "type": "cost_control",
                "priority": "high",
                "title": "Control Rapid Cost Growth",
                "description": f"Costs are growing at {trends['growth_rate_percent']:.1f}% rate",
                "potential_savings": float(trends["total_cost"]) * 0.1,  # Assume 10% control possible
                "suggestion": "Review project scope and resource allocation"
            })
        
        # Calculate total potential savings
        total_potential_savings = sum(r.get("potential_savings", 0) for r in recommendations)
        
        return {
            "analysis_period_days": days,
            "current_monthly_cost": float(trends["total_cost"]) * (30 / days),  # Extrapolate to monthly
            "recommendations": recommendations,
            "total_potential_savings": total_potential_savings,
            "optimization_score": max(0, 100 - len(recommendations) * 20),  # Simple scoring
            "generated_at": datetime.now().isoformat()
        }