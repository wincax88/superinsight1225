"""
Decision Tree Module for AI Agent System.

Implements decision path analysis, option evaluation,
multi-objective decision optimization, and outcome prediction.
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional, Callable, Tuple
import random
import math

logger = logging.getLogger(__name__)


class DecisionNodeType(str, Enum):
    """Types of decision tree nodes."""
    ROOT = "root"               # Root node
    DECISION = "decision"       # Decision point
    CHANCE = "chance"           # Probabilistic outcome
    OUTCOME = "outcome"         # Final outcome
    ACTION = "action"           # Action to take


class DecisionStatus(str, Enum):
    """Status of decision process."""
    PENDING = "pending"
    EVALUATING = "evaluating"
    DECIDED = "decided"
    EXECUTED = "executed"
    CANCELLED = "cancelled"


class OptimizationObjective(str, Enum):
    """Types of optimization objectives."""
    MAXIMIZE_VALUE = "maximize_value"
    MINIMIZE_RISK = "minimize_risk"
    MAXIMIZE_PROBABILITY = "maximize_probability"
    BALANCED = "balanced"


@dataclass
class DecisionCriteria:
    """Criteria for evaluating decisions."""
    name: str
    weight: float = 1.0
    minimize: bool = False  # If True, lower values are better
    threshold: Optional[float] = None
    description: str = ""


@dataclass
class DecisionOption:
    """A possible decision option."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    probability: float = 1.0  # Probability of success
    expected_value: float = 0.0
    risk_level: float = 0.0  # 0-1, higher is riskier
    cost: float = 0.0
    time_to_implement: float = 0.0
    dependencies: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def evaluate_criteria(
        self,
        criteria: List[DecisionCriteria]
    ) -> Dict[str, float]:
        """Evaluate this option against criteria."""
        scores = {}
        for criterion in criteria:
            value = self.metadata.get(criterion.name, 0.0)
            if criterion.minimize:
                value = 1.0 - value if value <= 1.0 else 1.0 / value
            scores[criterion.name] = value * criterion.weight
        return scores


@dataclass
class DecisionNode:
    """A node in the decision tree."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    node_type: DecisionNodeType = DecisionNodeType.DECISION
    name: str = ""
    description: str = ""
    options: List[DecisionOption] = field(default_factory=list)
    children: Dict[str, 'DecisionNode'] = field(default_factory=dict)
    parent_id: Optional[str] = None
    selected_option: Optional[str] = None
    probability: float = 1.0
    value: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_option(self, option: DecisionOption) -> None:
        """Add an option to this node."""
        self.options.append(option)

    def add_child(
        self,
        option_id: str,
        child: 'DecisionNode'
    ) -> None:
        """Add a child node for an option."""
        child.parent_id = self.id
        self.children[option_id] = child

    def get_expected_value(self) -> float:
        """Calculate expected value recursively."""
        if self.node_type == DecisionNodeType.OUTCOME:
            return self.value

        if not self.options:
            return 0.0

        if self.node_type == DecisionNodeType.DECISION:
            # Decision node: take max expected value
            option_values = []
            for option in self.options:
                child = self.children.get(option.id)
                if child:
                    ev = option.probability * child.get_expected_value()
                else:
                    ev = option.probability * option.expected_value
                option_values.append(ev)
            return max(option_values) if option_values else 0.0

        elif self.node_type == DecisionNodeType.CHANCE:
            # Chance node: weighted sum of outcomes
            total = 0.0
            for option in self.options:
                child = self.children.get(option.id)
                if child:
                    total += option.probability * child.get_expected_value()
                else:
                    total += option.probability * option.expected_value
            return total

        return 0.0


@dataclass
class DecisionPath:
    """A path through the decision tree."""
    nodes: List[str] = field(default_factory=list)
    options_selected: List[str] = field(default_factory=list)
    total_probability: float = 1.0
    expected_value: float = 0.0
    total_risk: float = 0.0
    total_cost: float = 0.0


@dataclass
class DecisionTree:
    """A complete decision tree."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    root: Optional[DecisionNode] = None
    nodes: Dict[str, DecisionNode] = field(default_factory=dict)
    criteria: List[DecisionCriteria] = field(default_factory=list)
    objective: OptimizationObjective = OptimizationObjective.BALANCED
    status: DecisionStatus = DecisionStatus.PENDING
    best_path: Optional[DecisionPath] = None
    all_paths: List[DecisionPath] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    decided_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_node(self, node: DecisionNode) -> None:
        """Add a node to the tree."""
        self.nodes[node.id] = node
        if node.node_type == DecisionNodeType.ROOT:
            self.root = node

    def get_node(self, node_id: str) -> Optional[DecisionNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)


@dataclass
class DecisionResult:
    """Result of decision analysis."""
    tree_id: str
    recommended_path: Optional[DecisionPath] = None
    alternative_paths: List[DecisionPath] = field(default_factory=list)
    option_rankings: List[Tuple[str, float]] = field(default_factory=list)
    sensitivity_analysis: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    reasoning: List[str] = field(default_factory=list)
    risk_assessment: Dict[str, Any] = field(default_factory=dict)


class DecisionTreeBuilder:
    """Builder for constructing decision trees."""

    def __init__(self):
        """Initialize the builder."""
        self.tree: Optional[DecisionTree] = None

    def create_tree(
        self,
        name: str,
        description: str = "",
        objective: OptimizationObjective = OptimizationObjective.BALANCED
    ) -> DecisionTree:
        """Create a new decision tree."""
        self.tree = DecisionTree(
            name=name,
            description=description,
            objective=objective
        )
        return self.tree

    def add_root(
        self,
        name: str,
        description: str = ""
    ) -> DecisionNode:
        """Add root node to the tree."""
        root = DecisionNode(
            node_type=DecisionNodeType.ROOT,
            name=name,
            description=description
        )
        if self.tree:
            self.tree.add_node(root)
        return root

    def add_decision_node(
        self,
        name: str,
        parent_id: str,
        parent_option_id: str,
        description: str = ""
    ) -> DecisionNode:
        """Add a decision node."""
        node = DecisionNode(
            node_type=DecisionNodeType.DECISION,
            name=name,
            description=description,
            parent_id=parent_id
        )

        if self.tree:
            self.tree.add_node(node)
            parent = self.tree.get_node(parent_id)
            if parent:
                parent.add_child(parent_option_id, node)

        return node

    def add_chance_node(
        self,
        name: str,
        parent_id: str,
        parent_option_id: str,
        description: str = ""
    ) -> DecisionNode:
        """Add a chance node."""
        node = DecisionNode(
            node_type=DecisionNodeType.CHANCE,
            name=name,
            description=description,
            parent_id=parent_id
        )

        if self.tree:
            self.tree.add_node(node)
            parent = self.tree.get_node(parent_id)
            if parent:
                parent.add_child(parent_option_id, node)

        return node

    def add_outcome_node(
        self,
        name: str,
        parent_id: str,
        parent_option_id: str,
        value: float,
        description: str = ""
    ) -> DecisionNode:
        """Add an outcome node."""
        node = DecisionNode(
            node_type=DecisionNodeType.OUTCOME,
            name=name,
            description=description,
            parent_id=parent_id,
            value=value
        )

        if self.tree:
            self.tree.add_node(node)
            parent = self.tree.get_node(parent_id)
            if parent:
                parent.add_child(parent_option_id, node)

        return node

    def add_option(
        self,
        node_id: str,
        name: str,
        probability: float = 1.0,
        expected_value: float = 0.0,
        risk_level: float = 0.0,
        cost: float = 0.0,
        **kwargs
    ) -> Optional[DecisionOption]:
        """Add an option to a node."""
        if not self.tree:
            return None

        node = self.tree.get_node(node_id)
        if not node:
            return None

        option = DecisionOption(
            name=name,
            probability=probability,
            expected_value=expected_value,
            risk_level=risk_level,
            cost=cost,
            metadata=kwargs
        )

        node.add_option(option)
        return option

    def add_criteria(
        self,
        name: str,
        weight: float = 1.0,
        minimize: bool = False,
        threshold: Optional[float] = None
    ) -> None:
        """Add evaluation criteria."""
        if self.tree:
            criteria = DecisionCriteria(
                name=name,
                weight=weight,
                minimize=minimize,
                threshold=threshold
            )
            self.tree.criteria.append(criteria)


class DecisionAnalyzer:
    """Analyzer for decision trees."""

    def __init__(self):
        """Initialize the analyzer."""
        self.analysis_history: List[DecisionResult] = []

    def analyze(
        self,
        tree: DecisionTree,
        max_paths: int = 100
    ) -> DecisionResult:
        """Analyze a decision tree and find optimal paths."""
        result = DecisionResult(tree_id=tree.id)

        if not tree.root:
            result.reasoning.append("No root node found")
            return result

        # Find all paths
        all_paths = self._enumerate_paths(tree.root, tree)
        tree.all_paths = all_paths[:max_paths]

        # Rank paths based on objective
        ranked_paths = self._rank_paths(all_paths, tree.objective)

        if ranked_paths:
            result.recommended_path = ranked_paths[0]
            result.alternative_paths = ranked_paths[1:5]  # Top 5 alternatives

            # Calculate confidence
            if len(ranked_paths) >= 2:
                best_score = self._calculate_path_score(ranked_paths[0], tree.objective)
                second_score = self._calculate_path_score(ranked_paths[1], tree.objective)
                result.confidence = min(1.0, (best_score - second_score) / max(0.01, best_score) + 0.5)
            else:
                result.confidence = 0.7

        # Rank options
        result.option_rankings = self._rank_options(tree)

        # Sensitivity analysis
        result.sensitivity_analysis = self._perform_sensitivity_analysis(tree)

        # Risk assessment
        result.risk_assessment = self._assess_risks(tree)

        # Generate reasoning
        result.reasoning = self._generate_reasoning(tree, result)

        # Update tree
        tree.best_path = result.recommended_path
        tree.status = DecisionStatus.DECIDED
        tree.decided_at = datetime.now()

        self.analysis_history.append(result)
        return result

    def _enumerate_paths(
        self,
        node: DecisionNode,
        tree: DecisionTree,
        current_path: Optional[DecisionPath] = None
    ) -> List[DecisionPath]:
        """Enumerate all paths from a node."""
        if current_path is None:
            current_path = DecisionPath()

        current_path.nodes.append(node.id)
        paths = []

        if node.node_type == DecisionNodeType.OUTCOME or not node.options:
            # Terminal node
            current_path.expected_value = node.value
            paths.append(current_path)
            return paths

        for option in node.options:
            new_path = DecisionPath(
                nodes=list(current_path.nodes),
                options_selected=list(current_path.options_selected) + [option.id],
                total_probability=current_path.total_probability * option.probability,
                total_risk=current_path.total_risk + option.risk_level,
                total_cost=current_path.total_cost + option.cost
            )

            child = node.children.get(option.id)
            if child:
                child_paths = self._enumerate_paths(child, tree, new_path)
                paths.extend(child_paths)
            else:
                new_path.expected_value = option.expected_value
                paths.append(new_path)

        return paths

    def _rank_paths(
        self,
        paths: List[DecisionPath],
        objective: OptimizationObjective
    ) -> List[DecisionPath]:
        """Rank paths based on objective."""
        scored_paths = [
            (path, self._calculate_path_score(path, objective))
            for path in paths
        ]

        scored_paths.sort(key=lambda x: x[1], reverse=True)
        return [path for path, _ in scored_paths]

    def _calculate_path_score(
        self,
        path: DecisionPath,
        objective: OptimizationObjective
    ) -> float:
        """Calculate score for a path based on objective."""
        if objective == OptimizationObjective.MAXIMIZE_VALUE:
            return path.expected_value * path.total_probability

        elif objective == OptimizationObjective.MINIMIZE_RISK:
            return (1.0 - path.total_risk) * path.total_probability

        elif objective == OptimizationObjective.MAXIMIZE_PROBABILITY:
            return path.total_probability

        else:  # BALANCED
            value_score = path.expected_value * path.total_probability
            risk_score = 1.0 - path.total_risk
            cost_score = 1.0 / (1.0 + path.total_cost)
            return (value_score * 0.4 + risk_score * 0.3 + cost_score * 0.3)

    def _rank_options(
        self,
        tree: DecisionTree
    ) -> List[Tuple[str, float]]:
        """Rank all options across the tree."""
        option_scores = {}

        for node in tree.nodes.values():
            for option in node.options:
                # Calculate option score
                score = option.expected_value * option.probability * (1 - option.risk_level)
                option_scores[option.name] = score

        sorted_options = sorted(option_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_options

    def _perform_sensitivity_analysis(
        self,
        tree: DecisionTree
    ) -> Dict[str, Any]:
        """Perform sensitivity analysis on key parameters."""
        analysis = {
            "probability_sensitivity": [],
            "value_sensitivity": [],
            "critical_thresholds": []
        }

        if not tree.root:
            return analysis

        # Test probability changes
        for node in tree.nodes.values():
            for option in node.options:
                original_prob = option.probability

                # Test +/- 10%
                for delta in [-0.1, 0.1]:
                    new_prob = max(0, min(1, original_prob + delta))
                    option.probability = new_prob
                    new_ev = tree.root.get_expected_value()
                    option.probability = original_prob

                    analysis["probability_sensitivity"].append({
                        "option": option.name,
                        "delta": delta,
                        "impact": new_ev
                    })

        return analysis

    def _assess_risks(
        self,
        tree: DecisionTree
    ) -> Dict[str, Any]:
        """Assess risks in the decision."""
        risks = {
            "overall_risk": 0.0,
            "high_risk_options": [],
            "low_probability_paths": [],
            "risk_factors": []
        }

        # Calculate overall risk
        total_risk = 0.0
        risk_count = 0

        for node in tree.nodes.values():
            for option in node.options:
                total_risk += option.risk_level
                risk_count += 1

                if option.risk_level > 0.7:
                    risks["high_risk_options"].append({
                        "option": option.name,
                        "risk_level": option.risk_level
                    })

        risks["overall_risk"] = total_risk / risk_count if risk_count > 0 else 0.0

        # Identify low probability paths
        for path in tree.all_paths:
            if path.total_probability < 0.1:
                risks["low_probability_paths"].append({
                    "probability": path.total_probability,
                    "nodes": len(path.nodes)
                })

        return risks

    def _generate_reasoning(
        self,
        tree: DecisionTree,
        result: DecisionResult
    ) -> List[str]:
        """Generate reasoning for the decision recommendation."""
        reasoning = []

        if result.recommended_path:
            reasoning.append(
                f"Recommended path has {len(result.recommended_path.nodes)} steps "
                f"with {result.recommended_path.total_probability:.1%} success probability"
            )

            reasoning.append(
                f"Expected value: {result.recommended_path.expected_value:.2f}, "
                f"Risk: {result.recommended_path.total_risk:.2f}, "
                f"Cost: {result.recommended_path.total_cost:.2f}"
            )

        if result.alternative_paths:
            reasoning.append(
                f"Analyzed {len(tree.all_paths)} possible paths, "
                f"{len(result.alternative_paths)} viable alternatives identified"
            )

        if result.risk_assessment.get("high_risk_options"):
            count = len(result.risk_assessment["high_risk_options"])
            reasoning.append(f"Warning: {count} high-risk options identified")

        reasoning.append(f"Confidence in recommendation: {result.confidence:.1%}")

        return reasoning


class MultiObjectiveOptimizer:
    """Optimizer for multi-objective decision problems."""

    def __init__(self):
        """Initialize the optimizer."""
        self.objectives: List[DecisionCriteria] = []

    def set_objectives(self, objectives: List[DecisionCriteria]) -> None:
        """Set optimization objectives."""
        self.objectives = objectives

    def find_pareto_optimal(
        self,
        options: List[DecisionOption]
    ) -> List[DecisionOption]:
        """Find Pareto-optimal options."""
        pareto_front = []

        for option in options:
            dominated = False

            for other in options:
                if option.id == other.id:
                    continue

                if self._dominates(other, option):
                    dominated = True
                    break

            if not dominated:
                pareto_front.append(option)

        return pareto_front

    def _dominates(
        self,
        option_a: DecisionOption,
        option_b: DecisionOption
    ) -> bool:
        """Check if option_a dominates option_b."""
        at_least_one_better = False

        for obj in self.objectives:
            value_a = option_a.metadata.get(obj.name, 0)
            value_b = option_b.metadata.get(obj.name, 0)

            if obj.minimize:
                if value_a > value_b:
                    return False
                if value_a < value_b:
                    at_least_one_better = True
            else:
                if value_a < value_b:
                    return False
                if value_a > value_b:
                    at_least_one_better = True

        return at_least_one_better

    def weighted_sum_optimization(
        self,
        options: List[DecisionOption]
    ) -> Optional[DecisionOption]:
        """Find best option using weighted sum method."""
        if not options or not self.objectives:
            return None

        best_option = None
        best_score = float('-inf')

        for option in options:
            score = 0.0
            for obj in self.objectives:
                value = option.metadata.get(obj.name, 0)
                if obj.minimize:
                    value = -value
                score += value * obj.weight

            if score > best_score:
                best_score = score
                best_option = option

        return best_option


class OutcomePredictor:
    """Predictor for decision outcomes."""

    def __init__(self):
        """Initialize the predictor."""
        self.prediction_history: List[Dict[str, Any]] = []

    def predict_outcome(
        self,
        tree: DecisionTree,
        path: DecisionPath,
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """Predict the outcome of following a path."""
        prediction = {
            "expected_value": 0.0,
            "value_range": (0.0, 0.0),
            "probability_of_success": 0.0,
            "confidence_interval": (0.0, 0.0),
            "risk_factors": [],
            "scenarios": []
        }

        # Calculate expected value
        prediction["expected_value"] = path.expected_value * path.total_probability

        # Calculate value range (Monte Carlo simulation simplified)
        values = []
        for _ in range(100):
            simulated_value = self._simulate_path(tree, path)
            values.append(simulated_value)

        values.sort()
        lower_idx = int((1 - confidence_level) / 2 * len(values))
        upper_idx = int((1 + confidence_level) / 2 * len(values))

        prediction["value_range"] = (min(values), max(values))
        prediction["confidence_interval"] = (values[lower_idx], values[upper_idx])

        # Probability of success (value > 0)
        prediction["probability_of_success"] = sum(1 for v in values if v > 0) / len(values)

        # Identify risk factors
        for node_id in path.nodes:
            node = tree.get_node(node_id)
            if node:
                for option in node.options:
                    if option.risk_level > 0.5:
                        prediction["risk_factors"].append({
                            "node": node.name,
                            "option": option.name,
                            "risk": option.risk_level
                        })

        # Generate scenarios
        prediction["scenarios"] = [
            {
                "name": "Best Case",
                "value": max(values),
                "probability": 0.1
            },
            {
                "name": "Expected Case",
                "value": prediction["expected_value"],
                "probability": 0.5
            },
            {
                "name": "Worst Case",
                "value": min(values),
                "probability": 0.1
            }
        ]

        self.prediction_history.append(prediction)
        return prediction

    def _simulate_path(
        self,
        tree: DecisionTree,
        path: DecisionPath
    ) -> float:
        """Simulate a single path execution."""
        value = 0.0
        probability = 1.0

        for i, node_id in enumerate(path.nodes):
            node = tree.get_node(node_id)
            if not node:
                continue

            if i < len(path.options_selected):
                option_id = path.options_selected[i]
                for option in node.options:
                    if option.id == option_id:
                        # Add randomness
                        success = random.random() < option.probability
                        if success:
                            value += option.expected_value * (0.8 + random.random() * 0.4)
                        probability *= option.probability
                        break

        return value


# Global instances
_decision_analyzer: Optional[DecisionAnalyzer] = None
_outcome_predictor: Optional[OutcomePredictor] = None


def get_decision_analyzer() -> DecisionAnalyzer:
    """Get or create global decision analyzer instance."""
    global _decision_analyzer
    if _decision_analyzer is None:
        _decision_analyzer = DecisionAnalyzer()
    return _decision_analyzer


def get_outcome_predictor() -> OutcomePredictor:
    """Get or create global outcome predictor instance."""
    global _outcome_predictor
    if _outcome_predictor is None:
        _outcome_predictor = OutcomePredictor()
    return _outcome_predictor


def create_simple_decision_tree(
    decision_name: str,
    options: List[Dict[str, Any]]
) -> Tuple[DecisionTree, DecisionResult]:
    """Helper to create and analyze a simple decision tree."""
    builder = DecisionTreeBuilder()

    tree = builder.create_tree(
        name=decision_name,
        objective=OptimizationObjective.BALANCED
    )

    root = builder.add_root(name=decision_name)

    for opt in options:
        builder.add_option(
            node_id=root.id,
            name=opt.get("name", "Option"),
            probability=opt.get("probability", 1.0),
            expected_value=opt.get("value", 0.0),
            risk_level=opt.get("risk", 0.0),
            cost=opt.get("cost", 0.0)
        )

        # Add outcome node for each option
        option_obj = root.options[-1]
        builder.add_outcome_node(
            name=f"{opt.get('name', 'Option')}_outcome",
            parent_id=root.id,
            parent_option_id=option_obj.id,
            value=opt.get("value", 0.0)
        )

    analyzer = get_decision_analyzer()
    result = analyzer.analyze(tree)

    return tree, result
