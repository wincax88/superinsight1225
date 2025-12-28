"""
Risk Assessment Engine for AI Agent System.

Implements risk factor identification, risk probability calculation,
risk mitigation suggestions, and risk monitoring alerts.
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, List, Optional, Callable, Tuple
import math
import random

logger = logging.getLogger(__name__)


class RiskCategory(str, Enum):
    """Categories of risks."""
    OPERATIONAL = "operational"         # Day-to-day operational risks
    STRATEGIC = "strategic"             # Long-term strategic risks
    FINANCIAL = "financial"             # Financial and economic risks
    TECHNICAL = "technical"             # Technical and technology risks
    COMPLIANCE = "compliance"           # Regulatory and compliance risks
    SECURITY = "security"               # Security and data risks
    REPUTATIONAL = "reputational"       # Brand and reputation risks
    EXTERNAL = "external"               # External market/environment risks


class RiskSeverity(str, Enum):
    """Severity levels for risks."""
    CRITICAL = "critical"       # Immediate threat, requires urgent action
    HIGH = "high"               # Significant impact, needs priority attention
    MEDIUM = "medium"           # Moderate impact, should be addressed
    LOW = "low"                 # Minor impact, monitor and address when convenient
    NEGLIGIBLE = "negligible"   # Minimal impact, acceptable risk


class RiskStatus(str, Enum):
    """Status of a risk."""
    IDENTIFIED = "identified"       # Newly identified
    ANALYZING = "analyzing"         # Under analysis
    ASSESSED = "assessed"           # Assessment complete
    MITIGATING = "mitigating"       # Mitigation in progress
    MONITORING = "monitoring"       # Being monitored
    RESOLVED = "resolved"           # Risk resolved
    ACCEPTED = "accepted"           # Risk accepted (no action)


class AlertLevel(str, Enum):
    """Alert levels for risk monitoring."""
    EMERGENCY = "emergency"     # Immediate action required
    WARNING = "warning"         # Attention needed
    INFO = "info"               # Informational
    CLEAR = "clear"             # Risk cleared


@dataclass
class RiskFactor:
    """A factor contributing to risk."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    category: RiskCategory = RiskCategory.OPERATIONAL
    weight: float = 1.0  # Contribution to overall risk
    current_value: float = 0.0  # Current risk level (0-1)
    threshold: float = 0.5  # Alert threshold
    trend: str = "stable"  # increasing, decreasing, stable
    data_source: str = ""
    last_updated: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskIndicator:
    """An indicator for measuring risk."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    metric_type: str = "gauge"  # gauge, counter, rate
    current_value: float = 0.0
    historical_values: List[Tuple[datetime, float]] = field(default_factory=list)
    thresholds: Dict[str, float] = field(default_factory=dict)
    unit: str = ""


@dataclass
class Risk:
    """A risk item."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    category: RiskCategory = RiskCategory.OPERATIONAL
    severity: RiskSeverity = RiskSeverity.MEDIUM
    status: RiskStatus = RiskStatus.IDENTIFIED
    probability: float = 0.5  # Likelihood of occurrence (0-1)
    impact: float = 0.5  # Impact if occurs (0-1)
    risk_score: float = 0.0  # Calculated: probability * impact
    factors: List[RiskFactor] = field(default_factory=list)
    indicators: List[RiskIndicator] = field(default_factory=list)
    affected_areas: List[str] = field(default_factory=list)
    owner: str = ""
    due_date: Optional[datetime] = None
    mitigation_plan: Optional[str] = None
    contingency_plan: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def calculate_risk_score(self) -> float:
        """Calculate risk score."""
        self.risk_score = self.probability * self.impact
        return self.risk_score

    def determine_severity(self) -> RiskSeverity:
        """Determine severity based on risk score."""
        score = self.calculate_risk_score()

        if score >= 0.8:
            self.severity = RiskSeverity.CRITICAL
        elif score >= 0.6:
            self.severity = RiskSeverity.HIGH
        elif score >= 0.4:
            self.severity = RiskSeverity.MEDIUM
        elif score >= 0.2:
            self.severity = RiskSeverity.LOW
        else:
            self.severity = RiskSeverity.NEGLIGIBLE

        return self.severity


@dataclass
class RiskAssessment:
    """Complete risk assessment."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    scope: str = ""
    risks: List[Risk] = field(default_factory=list)
    overall_risk_score: float = 0.0
    overall_severity: RiskSeverity = RiskSeverity.MEDIUM
    risk_matrix: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    assessed_at: datetime = field(default_factory=datetime.now)
    valid_until: Optional[datetime] = None
    assessor: str = ""
    methodology: str = "standard"
    confidence_level: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskAlert:
    """An alert for risk monitoring."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    risk_id: str = ""
    risk_name: str = ""
    level: AlertLevel = AlertLevel.INFO
    message: str = ""
    trigger_factor: str = ""
    trigger_value: float = 0.0
    threshold: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MitigationStrategy:
    """A strategy for mitigating risk."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    risk_id: str = ""
    strategy_type: str = "reduce"  # avoid, reduce, transfer, accept
    expected_reduction: float = 0.0  # Expected reduction in risk (0-1)
    cost: float = 0.0
    effort: str = "medium"  # low, medium, high
    timeline: str = ""
    steps: List[str] = field(default_factory=list)
    effectiveness: Optional[float] = None
    status: str = "proposed"  # proposed, approved, implementing, completed
    owner: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class RiskIdentifier:
    """Identifier for discovering risks."""

    def __init__(self):
        """Initialize the risk identifier."""
        self.risk_patterns: Dict[RiskCategory, List[Dict[str, Any]]] = {}
        self.identification_history: List[Dict[str, Any]] = []
        self._load_default_patterns()

    def _load_default_patterns(self) -> None:
        """Load default risk patterns."""
        self.risk_patterns = {
            RiskCategory.OPERATIONAL: [
                {
                    "pattern": "high_latency",
                    "indicators": ["response_time", "timeout_rate"],
                    "threshold": 0.7,
                    "description": "High latency may cause service degradation"
                },
                {
                    "pattern": "resource_exhaustion",
                    "indicators": ["cpu_usage", "memory_usage", "disk_usage"],
                    "threshold": 0.85,
                    "description": "Resource exhaustion may cause system failures"
                }
            ],
            RiskCategory.SECURITY: [
                {
                    "pattern": "unusual_access",
                    "indicators": ["failed_logins", "access_anomalies"],
                    "threshold": 0.6,
                    "description": "Unusual access patterns may indicate security threats"
                },
                {
                    "pattern": "data_exposure",
                    "indicators": ["sensitive_data_access", "export_volume"],
                    "threshold": 0.5,
                    "description": "Potential data exposure risk"
                }
            ],
            RiskCategory.TECHNICAL: [
                {
                    "pattern": "dependency_failure",
                    "indicators": ["service_health", "api_errors"],
                    "threshold": 0.4,
                    "description": "External dependency failures may impact functionality"
                },
                {
                    "pattern": "technical_debt",
                    "indicators": ["code_complexity", "bug_rate", "outdated_deps"],
                    "threshold": 0.6,
                    "description": "Technical debt increasing maintenance burden"
                }
            ],
            RiskCategory.COMPLIANCE: [
                {
                    "pattern": "policy_violation",
                    "indicators": ["compliance_score", "audit_findings"],
                    "threshold": 0.3,
                    "description": "Potential compliance policy violations"
                }
            ],
            RiskCategory.FINANCIAL: [
                {
                    "pattern": "cost_overrun",
                    "indicators": ["budget_utilization", "cost_growth_rate"],
                    "threshold": 0.8,
                    "description": "Budget overrun risk"
                }
            ]
        }

    def identify_risks(
        self,
        data: Dict[str, Any],
        categories: Optional[List[RiskCategory]] = None
    ) -> List[Risk]:
        """Identify risks from data."""
        identified_risks = []
        categories = categories or list(RiskCategory)

        for category in categories:
            patterns = self.risk_patterns.get(category, [])

            for pattern in patterns:
                # Check if any indicators trigger
                triggered = False
                trigger_values = []

                for indicator in pattern.get("indicators", []):
                    value = data.get(indicator, 0.0)
                    if value >= pattern.get("threshold", 0.5):
                        triggered = True
                        trigger_values.append((indicator, value))

                if triggered:
                    # Create risk
                    risk = Risk(
                        name=pattern.get("pattern", "Unknown Risk"),
                        description=pattern.get("description", ""),
                        category=category,
                        probability=max(v for _, v in trigger_values),
                        impact=0.5 + (0.3 * len(trigger_values)),  # More triggers = higher impact
                        factors=[
                            RiskFactor(
                                name=ind,
                                current_value=val,
                                category=category
                            )
                            for ind, val in trigger_values
                        ]
                    )
                    risk.determine_severity()
                    identified_risks.append(risk)

        # Record identification
        self.identification_history.append({
            "timestamp": datetime.now().isoformat(),
            "risks_found": len(identified_risks),
            "categories_checked": [c.value for c in categories]
        })

        return identified_risks

    def add_risk_pattern(
        self,
        category: RiskCategory,
        pattern: Dict[str, Any]
    ) -> None:
        """Add a custom risk pattern."""
        if category not in self.risk_patterns:
            self.risk_patterns[category] = []
        self.risk_patterns[category].append(pattern)


class RiskCalculator:
    """Calculator for risk probabilities and scores."""

    def __init__(self):
        """Initialize the calculator."""
        self.calculation_methods = {
            "simple": self._simple_calculation,
            "weighted": self._weighted_calculation,
            "bayesian": self._bayesian_calculation,
            "monte_carlo": self._monte_carlo_calculation
        }

    def calculate_probability(
        self,
        factors: List[RiskFactor],
        method: str = "weighted"
    ) -> float:
        """Calculate overall probability from factors."""
        calc_method = self.calculation_methods.get(method, self._weighted_calculation)
        return calc_method(factors)

    def _simple_calculation(self, factors: List[RiskFactor]) -> float:
        """Simple average calculation."""
        if not factors:
            return 0.0
        return sum(f.current_value for f in factors) / len(factors)

    def _weighted_calculation(self, factors: List[RiskFactor]) -> float:
        """Weighted average calculation."""
        if not factors:
            return 0.0

        total_weight = sum(f.weight for f in factors)
        if total_weight == 0:
            return 0.0

        weighted_sum = sum(f.current_value * f.weight for f in factors)
        return weighted_sum / total_weight

    def _bayesian_calculation(self, factors: List[RiskFactor]) -> float:
        """Bayesian probability calculation."""
        if not factors:
            return 0.0

        # Prior probability
        prior = 0.5

        # Update with each factor as evidence
        probability = prior
        for factor in factors:
            # Likelihood ratio
            likelihood = factor.current_value if factor.current_value > 0 else 0.01

            # Bayes update (simplified)
            probability = (probability * likelihood) / (
                probability * likelihood + (1 - probability) * (1 - likelihood)
            )

        return probability

    def _monte_carlo_calculation(
        self,
        factors: List[RiskFactor],
        simulations: int = 1000
    ) -> float:
        """Monte Carlo simulation for probability."""
        if not factors:
            return 0.0

        occurrences = 0
        for _ in range(simulations):
            # Simulate each factor
            all_triggered = True
            for factor in factors:
                if random.random() > factor.current_value:
                    all_triggered = False
                    break

            if all_triggered:
                occurrences += 1

        return occurrences / simulations

    def calculate_aggregate_risk(
        self,
        risks: List[Risk]
    ) -> Tuple[float, RiskSeverity]:
        """Calculate aggregate risk from multiple risks."""
        if not risks:
            return 0.0, RiskSeverity.NEGLIGIBLE

        # Calculate weighted average
        total_weight = 0.0
        weighted_score = 0.0

        severity_weights = {
            RiskSeverity.CRITICAL: 5,
            RiskSeverity.HIGH: 4,
            RiskSeverity.MEDIUM: 3,
            RiskSeverity.LOW: 2,
            RiskSeverity.NEGLIGIBLE: 1
        }

        for risk in risks:
            weight = severity_weights.get(risk.severity, 1)
            weighted_score += risk.risk_score * weight
            total_weight += weight

        aggregate_score = weighted_score / total_weight if total_weight > 0 else 0.0

        # Determine aggregate severity
        if aggregate_score >= 0.7:
            severity = RiskSeverity.CRITICAL
        elif aggregate_score >= 0.5:
            severity = RiskSeverity.HIGH
        elif aggregate_score >= 0.3:
            severity = RiskSeverity.MEDIUM
        elif aggregate_score >= 0.1:
            severity = RiskSeverity.LOW
        else:
            severity = RiskSeverity.NEGLIGIBLE

        return aggregate_score, severity


class MitigationAdvisor:
    """Advisor for risk mitigation strategies."""

    def __init__(self):
        """Initialize the advisor."""
        self.strategy_templates: Dict[RiskCategory, List[Dict[str, Any]]] = {}
        self._load_strategy_templates()

    def _load_strategy_templates(self) -> None:
        """Load default mitigation strategy templates."""
        self.strategy_templates = {
            RiskCategory.OPERATIONAL: [
                {
                    "name": "Implement redundancy",
                    "type": "reduce",
                    "reduction": 0.4,
                    "effort": "high",
                    "steps": [
                        "Identify critical components",
                        "Design redundant architecture",
                        "Implement failover mechanisms",
                        "Test failover procedures"
                    ]
                },
                {
                    "name": "Enhance monitoring",
                    "type": "reduce",
                    "reduction": 0.3,
                    "effort": "medium",
                    "steps": [
                        "Deploy comprehensive monitoring",
                        "Set up alerting thresholds",
                        "Create runbooks for common issues",
                        "Train team on response procedures"
                    ]
                }
            ],
            RiskCategory.SECURITY: [
                {
                    "name": "Implement access controls",
                    "type": "reduce",
                    "reduction": 0.5,
                    "effort": "medium",
                    "steps": [
                        "Review current access policies",
                        "Implement principle of least privilege",
                        "Enable multi-factor authentication",
                        "Regular access audits"
                    ]
                },
                {
                    "name": "Security training program",
                    "type": "reduce",
                    "reduction": 0.2,
                    "effort": "low",
                    "steps": [
                        "Develop security awareness content",
                        "Conduct regular training sessions",
                        "Phishing simulation exercises",
                        "Track and measure improvement"
                    ]
                }
            ],
            RiskCategory.TECHNICAL: [
                {
                    "name": "Technical debt reduction",
                    "type": "reduce",
                    "reduction": 0.3,
                    "effort": "high",
                    "steps": [
                        "Identify and prioritize technical debt",
                        "Allocate dedicated refactoring time",
                        "Update outdated dependencies",
                        "Improve test coverage"
                    ]
                }
            ],
            RiskCategory.FINANCIAL: [
                {
                    "name": "Budget contingency",
                    "type": "transfer",
                    "reduction": 0.4,
                    "effort": "low",
                    "steps": [
                        "Establish contingency fund",
                        "Define trigger conditions",
                        "Set approval process for contingency use"
                    ]
                }
            ],
            RiskCategory.COMPLIANCE: [
                {
                    "name": "Compliance automation",
                    "type": "reduce",
                    "reduction": 0.5,
                    "effort": "high",
                    "steps": [
                        "Map compliance requirements",
                        "Implement automated controls",
                        "Set up continuous monitoring",
                        "Regular compliance assessments"
                    ]
                }
            ]
        }

    def suggest_mitigations(
        self,
        risk: Risk
    ) -> List[MitigationStrategy]:
        """Suggest mitigation strategies for a risk."""
        suggestions = []
        templates = self.strategy_templates.get(risk.category, [])

        for template in templates:
            # Calculate expected effectiveness based on risk characteristics
            base_reduction = template.get("reduction", 0.3)

            # Adjust based on severity (higher severity = less effective mitigation)
            severity_factor = {
                RiskSeverity.CRITICAL: 0.6,
                RiskSeverity.HIGH: 0.8,
                RiskSeverity.MEDIUM: 1.0,
                RiskSeverity.LOW: 1.2,
                RiskSeverity.NEGLIGIBLE: 1.5
            }.get(risk.severity, 1.0)

            expected_reduction = min(1.0, base_reduction * severity_factor)

            strategy = MitigationStrategy(
                name=template.get("name", "Unknown Strategy"),
                risk_id=risk.id,
                strategy_type=template.get("type", "reduce"),
                expected_reduction=expected_reduction,
                effort=template.get("effort", "medium"),
                steps=template.get("steps", [])
            )

            # Generate description
            strategy.description = (
                f"Apply {strategy.name} to reduce {risk.name} risk. "
                f"Expected reduction: {expected_reduction:.0%}. "
                f"Effort level: {strategy.effort}."
            )

            suggestions.append(strategy)

        # Sort by expected reduction (highest first)
        suggestions.sort(key=lambda x: x.expected_reduction, reverse=True)

        return suggestions

    def estimate_residual_risk(
        self,
        risk: Risk,
        strategies: List[MitigationStrategy]
    ) -> float:
        """Estimate residual risk after applying mitigations."""
        current_risk = risk.risk_score
        residual = current_risk

        for strategy in strategies:
            reduction = strategy.expected_reduction
            residual *= (1 - reduction)

        return residual


class RiskMonitor:
    """Monitor for continuous risk tracking."""

    def __init__(self):
        """Initialize the monitor."""
        self.active_risks: Dict[str, Risk] = {}
        self.alerts: List[RiskAlert] = []
        self.alert_handlers: List[Callable[[RiskAlert], None]] = []
        self.check_interval: int = 300  # seconds
        self.history: List[Dict[str, Any]] = []

    def register_risk(self, risk: Risk) -> None:
        """Register a risk for monitoring."""
        self.active_risks[risk.id] = risk
        risk.status = RiskStatus.MONITORING
        logger.info(f"Registered risk for monitoring: {risk.name}")

    def unregister_risk(self, risk_id: str) -> bool:
        """Unregister a risk from monitoring."""
        if risk_id in self.active_risks:
            del self.active_risks[risk_id]
            return True
        return False

    def register_alert_handler(
        self,
        handler: Callable[[RiskAlert], None]
    ) -> None:
        """Register an alert handler."""
        self.alert_handlers.append(handler)

    def check_risks(
        self,
        current_data: Dict[str, Any]
    ) -> List[RiskAlert]:
        """Check all registered risks and generate alerts."""
        new_alerts = []

        for risk in self.active_risks.values():
            # Update risk factors with current data
            for factor in risk.factors:
                if factor.name in current_data:
                    old_value = factor.current_value
                    new_value = current_data[factor.name]
                    factor.current_value = new_value
                    factor.last_updated = datetime.now()

                    # Determine trend
                    if new_value > old_value:
                        factor.trend = "increasing"
                    elif new_value < old_value:
                        factor.trend = "decreasing"
                    else:
                        factor.trend = "stable"

                    # Check threshold
                    if new_value >= factor.threshold:
                        alert = self._create_alert(risk, factor)
                        new_alerts.append(alert)

            # Recalculate risk score
            risk.calculate_risk_score()
            risk.determine_severity()
            risk.updated_at = datetime.now()

        # Process alerts
        for alert in new_alerts:
            self.alerts.append(alert)
            for handler in self.alert_handlers:
                try:
                    handler(alert)
                except Exception as e:
                    logger.error(f"Alert handler error: {e}")

        # Record history
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "risks_checked": len(self.active_risks),
            "alerts_generated": len(new_alerts)
        })

        return new_alerts

    def _create_alert(
        self,
        risk: Risk,
        factor: RiskFactor
    ) -> RiskAlert:
        """Create an alert for a triggered risk factor."""
        # Determine alert level
        if factor.current_value >= 0.9:
            level = AlertLevel.EMERGENCY
        elif factor.current_value >= 0.7:
            level = AlertLevel.WARNING
        else:
            level = AlertLevel.INFO

        alert = RiskAlert(
            risk_id=risk.id,
            risk_name=risk.name,
            level=level,
            message=f"Risk factor '{factor.name}' exceeded threshold: "
                    f"{factor.current_value:.2f} >= {factor.threshold:.2f}",
            trigger_factor=factor.name,
            trigger_value=factor.current_value,
            threshold=factor.threshold
        )

        return alert

    def acknowledge_alert(
        self,
        alert_id: str,
        acknowledged_by: str
    ) -> bool:
        """Acknowledge an alert."""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                alert.acknowledged_by = acknowledged_by
                alert.acknowledged_at = datetime.now()
                return True
        return False

    def get_active_alerts(
        self,
        level: Optional[AlertLevel] = None,
        risk_id: Optional[str] = None
    ) -> List[RiskAlert]:
        """Get active (unresolved) alerts."""
        active = [a for a in self.alerts if not a.resolved]

        if level:
            active = [a for a in active if a.level == level]

        if risk_id:
            active = [a for a in active if a.risk_id == risk_id]

        return active

    def get_risk_summary(self) -> Dict[str, Any]:
        """Get summary of monitored risks."""
        summary = {
            "total_risks": len(self.active_risks),
            "by_severity": {},
            "by_category": {},
            "active_alerts": len(self.get_active_alerts()),
            "emergency_alerts": len(self.get_active_alerts(AlertLevel.EMERGENCY))
        }

        for risk in self.active_risks.values():
            # By severity
            severity = risk.severity.value
            summary["by_severity"][severity] = summary["by_severity"].get(severity, 0) + 1

            # By category
            category = risk.category.value
            summary["by_category"][category] = summary["by_category"].get(category, 0) + 1

        return summary


class RiskAssessmentEngine:
    """Main engine for comprehensive risk assessment."""

    def __init__(self):
        """Initialize the risk assessment engine."""
        self.identifier = RiskIdentifier()
        self.calculator = RiskCalculator()
        self.advisor = MitigationAdvisor()
        self.monitor = RiskMonitor()
        self.assessments: List[RiskAssessment] = []

    def perform_assessment(
        self,
        data: Dict[str, Any],
        scope: str = "general",
        categories: Optional[List[RiskCategory]] = None
    ) -> RiskAssessment:
        """Perform a comprehensive risk assessment."""
        assessment = RiskAssessment(
            name=f"Risk Assessment - {scope}",
            scope=scope,
            methodology="comprehensive"
        )

        # Step 1: Identify risks
        identified_risks = self.identifier.identify_risks(data, categories)
        assessment.risks = identified_risks

        # Step 2: Calculate aggregate risk
        if identified_risks:
            aggregate_score, aggregate_severity = self.calculator.calculate_aggregate_risk(
                identified_risks
            )
            assessment.overall_risk_score = aggregate_score
            assessment.overall_severity = aggregate_severity

        # Step 3: Generate risk matrix
        assessment.risk_matrix = self._generate_risk_matrix(identified_risks)

        # Step 4: Generate recommendations
        assessment.recommendations = self._generate_recommendations(identified_risks)

        # Step 5: Calculate confidence
        assessment.confidence_level = self._calculate_confidence(data, identified_risks)

        # Set validity period
        assessment.valid_until = datetime.now() + timedelta(days=30)

        # Register high-severity risks for monitoring
        for risk in identified_risks:
            if risk.severity in [RiskSeverity.CRITICAL, RiskSeverity.HIGH]:
                self.monitor.register_risk(risk)

        self.assessments.append(assessment)
        return assessment

    def _generate_risk_matrix(
        self,
        risks: List[Risk]
    ) -> Dict[str, Any]:
        """Generate a risk matrix."""
        matrix = {
            "high_probability_high_impact": [],
            "high_probability_low_impact": [],
            "low_probability_high_impact": [],
            "low_probability_low_impact": []
        }

        for risk in risks:
            prob = "high" if risk.probability >= 0.5 else "low"
            impact = "high" if risk.impact >= 0.5 else "low"
            key = f"{prob}_probability_{impact}_impact"
            matrix[key].append({
                "id": risk.id,
                "name": risk.name,
                "score": risk.risk_score
            })

        return matrix

    def _generate_recommendations(
        self,
        risks: List[Risk]
    ) -> List[str]:
        """Generate recommendations based on risks."""
        recommendations = []

        # Sort risks by score
        sorted_risks = sorted(risks, key=lambda r: r.risk_score, reverse=True)

        # Top 3 recommendations
        for i, risk in enumerate(sorted_risks[:3]):
            mitigations = self.advisor.suggest_mitigations(risk)
            if mitigations:
                best = mitigations[0]
                recommendations.append(
                    f"{i + 1}. Address {risk.severity.value} risk '{risk.name}': "
                    f"{best.description}"
                )

        # General recommendations
        if any(r.severity == RiskSeverity.CRITICAL for r in risks):
            recommendations.append(
                "URGENT: Critical risks identified - immediate action required"
            )

        if len([r for r in risks if r.category == RiskCategory.SECURITY]) >= 3:
            recommendations.append(
                "Multiple security risks detected - consider security review"
            )

        return recommendations

    def _calculate_confidence(
        self,
        data: Dict[str, Any],
        risks: List[Risk]
    ) -> float:
        """Calculate confidence level of the assessment."""
        # Factors affecting confidence:
        # 1. Data completeness
        # 2. Risk factor coverage
        # 3. Historical accuracy

        data_completeness = min(1.0, len(data) / 10)  # More data = higher confidence

        factor_coverage = 0.0
        if risks:
            total_factors = sum(len(r.factors) for r in risks)
            factor_coverage = min(1.0, total_factors / (len(risks) * 3))

        # Base confidence
        confidence = (data_completeness * 0.4 + factor_coverage * 0.4 + 0.2)

        return confidence

    def get_mitigation_plan(
        self,
        assessment_id: str
    ) -> Dict[str, Any]:
        """Get a comprehensive mitigation plan for an assessment."""
        assessment = None
        for a in self.assessments:
            if a.id == assessment_id:
                assessment = a
                break

        if not assessment:
            return {"error": "Assessment not found"}

        plan = {
            "assessment_id": assessment_id,
            "created_at": datetime.now().isoformat(),
            "risks_addressed": [],
            "total_estimated_reduction": 0.0,
            "priority_order": []
        }

        for risk in sorted(assessment.risks, key=lambda r: r.risk_score, reverse=True):
            mitigations = self.advisor.suggest_mitigations(risk)

            risk_plan = {
                "risk_id": risk.id,
                "risk_name": risk.name,
                "current_score": risk.risk_score,
                "strategies": []
            }

            residual = risk.risk_score
            for strategy in mitigations[:2]:  # Top 2 strategies per risk
                new_residual = residual * (1 - strategy.expected_reduction)
                risk_plan["strategies"].append({
                    "name": strategy.name,
                    "reduction": strategy.expected_reduction,
                    "residual_after": new_residual,
                    "steps": strategy.steps
                })
                residual = new_residual

            risk_plan["final_residual"] = residual
            plan["risks_addressed"].append(risk_plan)
            plan["total_estimated_reduction"] += (risk.risk_score - residual)

        plan["priority_order"] = [r["risk_id"] for r in plan["risks_addressed"]]

        return plan


# Global instance
_risk_engine: Optional[RiskAssessmentEngine] = None


def get_risk_engine() -> RiskAssessmentEngine:
    """Get or create global risk assessment engine instance."""
    global _risk_engine
    if _risk_engine is None:
        _risk_engine = RiskAssessmentEngine()
    return _risk_engine


def quick_risk_assessment(
    data: Dict[str, Any]
) -> Dict[str, Any]:
    """Helper function for quick risk assessment."""
    engine = get_risk_engine()
    assessment = engine.perform_assessment(data)

    return {
        "assessment_id": assessment.id,
        "overall_risk_score": assessment.overall_risk_score,
        "overall_severity": assessment.overall_severity.value,
        "risks_found": len(assessment.risks),
        "recommendations": assessment.recommendations[:3],
        "confidence": assessment.confidence_level
    }
