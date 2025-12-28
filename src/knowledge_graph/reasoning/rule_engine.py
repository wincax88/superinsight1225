"""
Rule-based reasoning engine for Knowledge Graph.

Provides rule definition, evaluation, and inference chain tracking.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class RuleType(str, Enum):
    """Type of reasoning rule."""

    IMPLICATION = "implication"  # A -> B
    TRANSITIVITY = "transitivity"  # A->B, B->C => A->C
    SYMMETRY = "symmetry"  # A->B => B->A
    INHERITANCE = "inheritance"  # subclass relationships
    COMPOSITION = "composition"  # combine multiple rules
    CONSTRAINT = "constraint"  # validation rules
    AGGREGATION = "aggregation"  # aggregate relationships


class RulePriority(str, Enum):
    """Priority levels for rule execution."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ConditionOperator(str, Enum):
    """Operators for rule conditions."""

    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    MATCHES = "matches"  # regex
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    IN = "in"
    NOT_IN = "not_in"
    EXISTS = "exists"
    NOT_EXISTS = "not_exists"
    IS_TYPE = "is_type"


class Condition(BaseModel):
    """A single condition in a rule."""

    field: str = Field(..., description="Field to evaluate")
    operator: ConditionOperator = Field(..., description="Comparison operator")
    value: Any = Field(None, description="Value to compare against")
    negate: bool = Field(default=False, description="Negate the condition")

    def evaluate(self, context: dict[str, Any]) -> bool:
        """Evaluate the condition against a context."""
        field_value = self._get_field_value(context, self.field)

        result = self._evaluate_operator(field_value)
        return not result if self.negate else result

    def _get_field_value(self, context: dict[str, Any], field_path: str) -> Any:
        """Get field value from context using dot notation."""
        parts = field_path.split(".")
        value = context
        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            elif hasattr(value, part):
                value = getattr(value, part)
            else:
                return None
        return value

    def _evaluate_operator(self, field_value: Any) -> bool:
        """Evaluate the operator against the field value."""
        if self.operator == ConditionOperator.EXISTS:
            return field_value is not None

        if self.operator == ConditionOperator.NOT_EXISTS:
            return field_value is None

        if field_value is None:
            return False

        if self.operator == ConditionOperator.EQUALS:
            return field_value == self.value
        elif self.operator == ConditionOperator.NOT_EQUALS:
            return field_value != self.value
        elif self.operator == ConditionOperator.CONTAINS:
            return self.value in str(field_value)
        elif self.operator == ConditionOperator.STARTS_WITH:
            return str(field_value).startswith(str(self.value))
        elif self.operator == ConditionOperator.ENDS_WITH:
            return str(field_value).endswith(str(self.value))
        elif self.operator == ConditionOperator.MATCHES:
            return bool(re.match(str(self.value), str(field_value)))
        elif self.operator == ConditionOperator.GREATER_THAN:
            return field_value > self.value
        elif self.operator == ConditionOperator.LESS_THAN:
            return field_value < self.value
        elif self.operator == ConditionOperator.IN:
            return field_value in self.value
        elif self.operator == ConditionOperator.NOT_IN:
            return field_value not in self.value
        elif self.operator == ConditionOperator.IS_TYPE:
            return type(field_value).__name__ == self.value

        return False


class Action(BaseModel):
    """An action to perform when rule conditions are met."""

    action_type: str = Field(..., description="Type of action")
    target: str = Field(..., description="Target of the action")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Action parameters")

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """Execute the action and return the result."""
        return {
            "action_type": self.action_type,
            "target": self.target,
            "parameters": self.parameters,
            "context": context,
            "timestamp": datetime.now().isoformat(),
        }


class Rule(BaseModel):
    """A reasoning rule with conditions and actions."""

    rule_id: str = Field(..., description="Unique rule identifier")
    name: str = Field(..., description="Human-readable rule name")
    description: str = Field(default="", description="Rule description")
    rule_type: RuleType = Field(default=RuleType.IMPLICATION, description="Type of rule")
    priority: RulePriority = Field(default=RulePriority.MEDIUM, description="Rule priority")
    conditions: list[Condition] = Field(default_factory=list, description="Rule conditions (AND)")
    alternative_conditions: list[list[Condition]] = Field(
        default_factory=list, description="Alternative condition sets (OR)"
    )
    actions: list[Action] = Field(default_factory=list, description="Actions to perform")
    enabled: bool = Field(default=True, description="Whether rule is active")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def evaluate(self, context: dict[str, Any]) -> bool:
        """Evaluate whether rule conditions are satisfied."""
        if not self.enabled:
            return False

        # Check main conditions (all must be true - AND)
        main_result = all(cond.evaluate(context) for cond in self.conditions)

        if main_result:
            return True

        # Check alternative conditions (any set can be true - OR)
        for alt_conditions in self.alternative_conditions:
            if all(cond.evaluate(context) for cond in alt_conditions):
                return True

        return False


class InferenceStep(BaseModel):
    """A single step in an inference chain."""

    step_id: int = Field(..., description="Step number")
    rule_id: str = Field(..., description="Rule that was applied")
    rule_name: str = Field(..., description="Name of the rule")
    input_facts: dict[str, Any] = Field(..., description="Input facts for this step")
    output_facts: dict[str, Any] = Field(..., description="Output facts from this step")
    confidence: float = Field(default=1.0, description="Confidence in this inference")
    timestamp: datetime = Field(default_factory=datetime.now, description="When step was executed")


class InferenceChain(BaseModel):
    """Complete inference chain with all steps."""

    chain_id: str = Field(..., description="Unique chain identifier")
    query: str = Field(..., description="Original query or goal")
    initial_facts: dict[str, Any] = Field(..., description="Starting facts")
    steps: list[InferenceStep] = Field(default_factory=list, description="Inference steps")
    final_result: dict[str, Any] = Field(default_factory=dict, description="Final result")
    total_confidence: float = Field(default=1.0, description="Combined confidence")
    success: bool = Field(default=False, description="Whether inference succeeded")
    execution_time_ms: float = Field(default=0.0, description="Execution time")


class InferredFact(BaseModel):
    """A fact inferred through reasoning."""

    fact_id: str = Field(..., description="Unique fact identifier")
    fact_type: str = Field(..., description="Type of fact")
    subject: str = Field(..., description="Subject entity")
    predicate: str = Field(..., description="Relationship/property")
    object: str = Field(..., description="Object entity/value")
    confidence: float = Field(default=1.0, description="Confidence score")
    source_rules: list[str] = Field(default_factory=list, description="Rules that inferred this")
    supporting_facts: list[str] = Field(default_factory=list, description="Supporting fact IDs")
    timestamp: datetime = Field(default_factory=datetime.now, description="When inferred")


@dataclass
class RuleEngine:
    """Rule-based reasoning engine."""

    rules: dict[str, Rule] = field(default_factory=dict)
    facts: dict[str, InferredFact] = field(default_factory=dict)
    inference_history: list[InferenceChain] = field(default_factory=list)
    custom_actions: dict[str, Callable] = field(default_factory=dict)
    max_iterations: int = 100

    def add_rule(self, rule: Rule) -> None:
        """Add a rule to the engine."""
        self.rules[rule.rule_id] = rule
        logger.info(f"Added rule: {rule.rule_id} - {rule.name}")

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a rule from the engine."""
        if rule_id in self.rules:
            del self.rules[rule_id]
            logger.info(f"Removed rule: {rule_id}")
            return True
        return False

    def add_fact(self, fact: InferredFact) -> None:
        """Add a fact to the knowledge base."""
        self.facts[fact.fact_id] = fact

    def register_action(self, action_type: str, handler: Callable) -> None:
        """Register a custom action handler."""
        self.custom_actions[action_type] = handler

    def infer(
        self,
        initial_facts: dict[str, Any],
        query: Optional[str] = None,
    ) -> InferenceChain:
        """Perform forward chaining inference."""
        import time
        import uuid

        start_time = time.time()
        chain_id = str(uuid.uuid4())[:8]

        chain = InferenceChain(
            chain_id=chain_id,
            query=query or "Forward inference",
            initial_facts=initial_facts,
        )

        context = dict(initial_facts)
        step_count = 0
        iterations = 0
        rules_fired = set()

        # Get rules sorted by priority
        sorted_rules = self._get_sorted_rules()

        # Forward chaining loop
        while iterations < self.max_iterations:
            iterations += 1
            fired_this_round = False

            for rule in sorted_rules:
                if rule.rule_id in rules_fired:
                    continue

                if rule.evaluate(context):
                    step_count += 1

                    # Execute actions
                    step_output = {}
                    for action in rule.actions:
                        if action.action_type in self.custom_actions:
                            result = self.custom_actions[action.action_type](
                                context, action.parameters
                            )
                            step_output.update(result)
                        else:
                            result = action.execute(context)
                            step_output[action.target] = result

                    # Update context with new facts
                    context.update(step_output)

                    # Record step
                    step = InferenceStep(
                        step_id=step_count,
                        rule_id=rule.rule_id,
                        rule_name=rule.name,
                        input_facts=dict(context),
                        output_facts=step_output,
                        confidence=self._calculate_step_confidence(rule, context),
                    )
                    chain.steps.append(step)

                    rules_fired.add(rule.rule_id)
                    fired_this_round = True

            # No more rules can fire
            if not fired_this_round:
                break

        # Finalize chain
        chain.final_result = context
        chain.total_confidence = self._calculate_chain_confidence(chain)
        chain.success = len(chain.steps) > 0
        chain.execution_time_ms = (time.time() - start_time) * 1000

        self.inference_history.append(chain)

        logger.info(
            f"Inference chain {chain_id}: {len(chain.steps)} steps, "
            f"confidence={chain.total_confidence:.2f}"
        )

        return chain

    def backward_chain(
        self,
        goal: dict[str, Any],
        known_facts: dict[str, Any],
    ) -> InferenceChain:
        """Perform backward chaining to prove a goal."""
        import time
        import uuid

        start_time = time.time()
        chain_id = str(uuid.uuid4())[:8]

        chain = InferenceChain(
            chain_id=chain_id,
            query=f"Prove: {goal}",
            initial_facts=known_facts,
        )

        # Try to prove the goal
        success, steps = self._prove_goal(goal, known_facts, [], set())

        chain.steps = steps
        chain.final_result = {"goal": goal, "proven": success}
        chain.total_confidence = self._calculate_chain_confidence(chain)
        chain.success = success
        chain.execution_time_ms = (time.time() - start_time) * 1000

        self.inference_history.append(chain)

        return chain

    def _prove_goal(
        self,
        goal: dict[str, Any],
        facts: dict[str, Any],
        steps: list[InferenceStep],
        visited: set[str],
    ) -> tuple[bool, list[InferenceStep]]:
        """Recursively try to prove a goal."""
        # Check if goal is already in facts
        if self._goal_satisfied(goal, facts):
            return True, steps

        # Find rules that could prove this goal
        applicable_rules = self._find_rules_for_goal(goal)

        for rule in applicable_rules:
            if rule.rule_id in visited:
                continue

            visited.add(rule.rule_id)

            # Try to satisfy rule conditions
            if self._can_satisfy_conditions(rule.conditions, facts, visited):
                step = InferenceStep(
                    step_id=len(steps) + 1,
                    rule_id=rule.rule_id,
                    rule_name=rule.name,
                    input_facts=dict(facts),
                    output_facts=goal,
                    confidence=0.9,
                )
                steps.append(step)

                # Update facts with goal
                facts.update(goal)
                return True, steps

        return False, steps

    def _goal_satisfied(self, goal: dict[str, Any], facts: dict[str, Any]) -> bool:
        """Check if a goal is satisfied by current facts."""
        for key, value in goal.items():
            if key not in facts or facts[key] != value:
                return False
        return True

    def _find_rules_for_goal(self, goal: dict[str, Any]) -> list[Rule]:
        """Find rules that can potentially prove a goal."""
        applicable = []
        for rule in self.rules.values():
            # Check if any action targets match goal keys
            for action in rule.actions:
                if action.target in goal:
                    applicable.append(rule)
                    break
        return applicable

    def _can_satisfy_conditions(
        self,
        conditions: list[Condition],
        facts: dict[str, Any],
        visited: set[str],
    ) -> bool:
        """Check if conditions can be satisfied."""
        for condition in conditions:
            if not condition.evaluate(facts):
                # Try to prove the condition as a sub-goal
                sub_goal = {condition.field: condition.value}
                success, _ = self._prove_goal(sub_goal, facts, [], visited)
                if not success:
                    return False
        return True

    def _get_sorted_rules(self) -> list[Rule]:
        """Get rules sorted by priority."""
        priority_order = {
            RulePriority.CRITICAL: 0,
            RulePriority.HIGH: 1,
            RulePriority.MEDIUM: 2,
            RulePriority.LOW: 3,
        }
        return sorted(
            [r for r in self.rules.values() if r.enabled],
            key=lambda r: priority_order.get(r.priority, 2),
        )

    def _calculate_step_confidence(
        self,
        rule: Rule,
        context: dict[str, Any],
    ) -> float:
        """Calculate confidence for a single inference step."""
        base_confidence = rule.metadata.get("confidence", 1.0)

        # Adjust based on condition matching quality
        condition_scores = []
        for condition in rule.conditions:
            if condition.evaluate(context):
                condition_scores.append(1.0)
            else:
                condition_scores.append(0.5)

        if condition_scores:
            avg_condition_score = sum(condition_scores) / len(condition_scores)
            return base_confidence * avg_condition_score

        return base_confidence

    def _calculate_chain_confidence(self, chain: InferenceChain) -> float:
        """Calculate combined confidence for the entire chain."""
        if not chain.steps:
            return 0.0

        # Multiply step confidences (conservative approach)
        confidence = 1.0
        for step in chain.steps:
            confidence *= step.confidence

        return confidence

    def get_inference_history(self, limit: int = 10) -> list[InferenceChain]:
        """Get recent inference history."""
        return self.inference_history[-limit:]

    def explain_inference(self, chain: InferenceChain) -> str:
        """Generate human-readable explanation of inference chain."""
        lines = [
            f"推理链 {chain.chain_id}: {chain.query}",
            f"初始事实: {chain.initial_facts}",
            "",
            "推理步骤:",
        ]

        for step in chain.steps:
            lines.append(f"  步骤 {step.step_id}: 应用规则 '{step.rule_name}'")
            lines.append(f"    规则ID: {step.rule_id}")
            lines.append(f"    输出: {step.output_facts}")
            lines.append(f"    置信度: {step.confidence:.2%}")
            lines.append("")

        lines.extend([
            f"最终结果: {chain.final_result}",
            f"总置信度: {chain.total_confidence:.2%}",
            f"成功: {'是' if chain.success else '否'}",
            f"执行时间: {chain.execution_time_ms:.2f}ms",
        ])

        return "\n".join(lines)

    # Built-in rule templates

    def create_transitivity_rule(
        self,
        rule_id: str,
        relation: str,
    ) -> Rule:
        """Create a transitivity rule for a relation."""
        return Rule(
            rule_id=rule_id,
            name=f"Transitivity: {relation}",
            description=f"If A {relation} B and B {relation} C, then A {relation} C",
            rule_type=RuleType.TRANSITIVITY,
            conditions=[
                Condition(
                    field=f"relation.{relation}.source",
                    operator=ConditionOperator.EXISTS,
                ),
                Condition(
                    field=f"relation.{relation}.target",
                    operator=ConditionOperator.EXISTS,
                ),
            ],
            actions=[
                Action(
                    action_type="infer_relation",
                    target=f"transitive_{relation}",
                    parameters={"relation": relation},
                )
            ],
            metadata={"confidence": 0.95},
        )

    def create_symmetry_rule(
        self,
        rule_id: str,
        relation: str,
    ) -> Rule:
        """Create a symmetry rule for a relation."""
        return Rule(
            rule_id=rule_id,
            name=f"Symmetry: {relation}",
            description=f"If A {relation} B, then B {relation} A",
            rule_type=RuleType.SYMMETRY,
            conditions=[
                Condition(
                    field=f"relation.{relation}",
                    operator=ConditionOperator.EXISTS,
                ),
            ],
            actions=[
                Action(
                    action_type="infer_symmetric",
                    target=f"symmetric_{relation}",
                    parameters={"relation": relation},
                )
            ],
            metadata={"confidence": 1.0},
        )

    def create_inheritance_rule(
        self,
        rule_id: str,
        parent_type: str,
        child_type: str,
        inherited_property: str,
    ) -> Rule:
        """Create an inheritance rule."""
        return Rule(
            rule_id=rule_id,
            name=f"Inheritance: {child_type} from {parent_type}",
            description=f"{child_type} inherits {inherited_property} from {parent_type}",
            rule_type=RuleType.INHERITANCE,
            conditions=[
                Condition(
                    field="entity.type",
                    operator=ConditionOperator.EQUALS,
                    value=child_type,
                ),
                Condition(
                    field=f"entity.parent.{inherited_property}",
                    operator=ConditionOperator.EXISTS,
                ),
            ],
            actions=[
                Action(
                    action_type="inherit_property",
                    target=inherited_property,
                    parameters={
                        "parent_type": parent_type,
                        "property": inherited_property,
                    },
                )
            ],
            metadata={"confidence": 0.9},
        )


# Global instance
_rule_engine: Optional[RuleEngine] = None


def get_rule_engine() -> RuleEngine:
    """Get or create the global rule engine instance."""
    global _rule_engine
    if _rule_engine is None:
        _rule_engine = RuleEngine()
    return _rule_engine
