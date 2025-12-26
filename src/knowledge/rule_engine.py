"""
Rule Engine for SuperInsight Platform.

Manages knowledge rules and provides rule learning capabilities.
"""

import logging
import re
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Callable
from uuid import UUID
from collections import defaultdict

from .models import (
    KnowledgeRule,
    RuleType,
    CaseEntry,
    CaseStatus
)

logger = logging.getLogger(__name__)


class RuleEngine:
    """
    Rule engine for managing and applying knowledge rules.

    Features:
    - Rule CRUD operations
    - Rule application
    - Rule learning from cases
    - Rule evaluation and optimization
    """

    def __init__(self):
        """Initialize RuleEngine."""
        self._rules: Dict[UUID, KnowledgeRule] = {}
        self._type_index: Dict[RuleType, List[UUID]] = defaultdict(list)
        self._condition_handlers: Dict[str, Callable] = {}
        self._action_handlers: Dict[str, Callable] = {}

        # Register default handlers
        self._register_default_handlers()

    def add_rule(self, rule: KnowledgeRule) -> UUID:
        """
        Add a rule.

        Args:
            rule: Rule to add

        Returns:
            Rule ID
        """
        self._rules[rule.id] = rule
        self._type_index[rule.rule_type].append(rule.id)

        logger.info(f"Added rule: {rule.id} - {rule.name}")
        return rule.id

    def update_rule(self, rule_id: UUID,
                   updates: Dict[str, Any]) -> Optional[UUID]:
        """
        Update a rule.

        Args:
            rule_id: Rule ID
            updates: Field updates

        Returns:
            Rule ID if updated, None if not found
        """
        if rule_id not in self._rules:
            return None

        rule = self._rules[rule_id]

        for key, value in updates.items():
            if hasattr(rule, key):
                setattr(rule, key, value)

        rule.updated_at = datetime.now()

        logger.info(f"Updated rule: {rule_id}")
        return rule_id

    def get_rule(self, rule_id: UUID) -> Optional[KnowledgeRule]:
        """Get a rule by ID."""
        return self._rules.get(rule_id)

    def delete_rule(self, rule_id: UUID) -> bool:
        """Delete a rule."""
        if rule_id not in self._rules:
            return False

        rule = self._rules[rule_id]
        self._type_index[rule.rule_type].remove(rule_id)
        del self._rules[rule_id]

        logger.info(f"Deleted rule: {rule_id}")
        return True

    def get_rules_by_type(self, rule_type: RuleType) -> List[KnowledgeRule]:
        """Get rules by type."""
        rule_ids = self._type_index.get(rule_type, [])
        return [self._rules[rid] for rid in rule_ids if rid in self._rules]

    def apply_rules(self, context: Dict[str, Any],
                   rule_type: Optional[RuleType] = None) -> List[Tuple[KnowledgeRule, Any]]:
        """
        Apply matching rules to context.

        Args:
            context: Context data for rule evaluation
            rule_type: Optional filter by rule type

        Returns:
            List of (rule, result) tuples
        """
        results = []

        # Get applicable rules
        if rule_type:
            rules = self.get_rules_by_type(rule_type)
        else:
            rules = list(self._rules.values())

        # Sort by priority
        rules.sort(key=lambda r: r.priority, reverse=True)

        # Apply each enabled rule
        for rule in rules:
            if not rule.is_enabled:
                continue

            try:
                if self._evaluate_condition(rule.condition, context):
                    result = self._execute_action(rule.action, context)
                    rule.apply()
                    results.append((rule, result))
                    logger.debug(f"Rule {rule.name} matched and applied")

            except Exception as e:
                logger.warning(f"Failed to apply rule {rule.id}: {e}")

        return results

    def learn_rule(self, cases: List[CaseEntry],
                  rule_type: RuleType = RuleType.ASSOCIATION,
                  min_support: float = 0.1,
                  min_confidence: float = 0.5) -> List[KnowledgeRule]:
        """
        Learn rules from cases.

        Args:
            cases: Cases to learn from
            rule_type: Type of rules to learn
            min_support: Minimum support threshold
            min_confidence: Minimum confidence threshold

        Returns:
            List of learned rules
        """
        if len(cases) < 2:
            return []

        learned_rules = []

        # Extract patterns based on rule type
        if rule_type == RuleType.ASSOCIATION:
            learned_rules = self._learn_association_rules(cases, min_support, min_confidence)
        elif rule_type == RuleType.CLASSIFICATION:
            learned_rules = self._learn_classification_rules(cases, min_support, min_confidence)
        elif rule_type == RuleType.INFERENCE:
            learned_rules = self._learn_inference_rules(cases, min_support, min_confidence)

        # Add learned rules
        for rule in learned_rules:
            self.add_rule(rule)

        logger.info(f"Learned {len(learned_rules)} rules from {len(cases)} cases")
        return learned_rules

    def evaluate_rule(self, rule_id: UUID) -> Dict[str, Any]:
        """
        Evaluate rule effectiveness.

        Args:
            rule_id: Rule ID

        Returns:
            Evaluation metrics
        """
        if rule_id not in self._rules:
            return {"error": "Rule not found"}

        rule = self._rules[rule_id]

        return {
            "rule_id": str(rule_id),
            "name": rule.name,
            "hit_count": rule.hit_count,
            "success_count": rule.success_count,
            "success_rate": rule.success_rate,
            "confidence": rule.confidence,
            "support": rule.support,
            "is_effective": rule.success_rate >= 0.7 and rule.hit_count >= 5
        }

    def optimize_rules(self) -> Dict[str, Any]:
        """
        Optimize rule set.

        Returns:
            Optimization results
        """
        disabled = 0
        removed = 0
        adjusted = 0

        rules_to_remove = []

        for rule_id, rule in self._rules.items():
            # Disable ineffective rules
            if rule.hit_count >= 10 and rule.success_rate < 0.3:
                rule.is_enabled = False
                disabled += 1

            # Mark for removal rules with very low performance
            if rule.hit_count >= 20 and rule.success_rate < 0.1:
                rules_to_remove.append(rule_id)

            # Adjust confidence based on success rate
            if rule.hit_count >= 5:
                old_confidence = rule.confidence
                rule.confidence = (rule.confidence + rule.success_rate) / 2
                if abs(old_confidence - rule.confidence) > 0.1:
                    adjusted += 1

        # Remove ineffective rules
        for rule_id in rules_to_remove:
            self.delete_rule(rule_id)
            removed += 1

        return {
            "disabled": disabled,
            "removed": removed,
            "adjusted": adjusted,
            "total_rules": len(self._rules)
        }

    def _register_default_handlers(self) -> None:
        """Register default condition and action handlers."""
        # Condition handlers
        self._condition_handlers["equals"] = lambda ctx, key, value: ctx.get(key) == value
        self._condition_handlers["contains"] = lambda ctx, key, value: value in str(ctx.get(key, ""))
        self._condition_handlers["greater_than"] = lambda ctx, key, value: ctx.get(key, 0) > value
        self._condition_handlers["less_than"] = lambda ctx, key, value: ctx.get(key, 0) < value
        self._condition_handlers["exists"] = lambda ctx, key, _: key in ctx
        self._condition_handlers["matches"] = lambda ctx, key, pattern: bool(re.match(pattern, str(ctx.get(key, ""))))

        # Action handlers
        self._action_handlers["set"] = lambda ctx, key, value: ctx.update({key: value}) or value
        self._action_handlers["append"] = lambda ctx, key, value: ctx.get(key, []) + [value]
        self._action_handlers["transform"] = lambda ctx, key, func: func(ctx.get(key))

    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate a condition expression."""
        # Simple condition parsing: "handler:key:value"
        parts = condition.split(":")
        if len(parts) < 2:
            return False

        handler_name = parts[0]
        key = parts[1]
        value = parts[2] if len(parts) > 2 else None

        if handler_name in self._condition_handlers:
            try:
                return self._condition_handlers[handler_name](context, key, value)
            except Exception:
                return False

        # Fallback: simple equality check
        return context.get(key) == value

    def _execute_action(self, action: str, context: Dict[str, Any]) -> Any:
        """Execute an action expression."""
        # Simple action parsing: "handler:key:value"
        parts = action.split(":")
        if len(parts) < 2:
            return None

        handler_name = parts[0]
        key = parts[1]
        value = parts[2] if len(parts) > 2 else None

        if handler_name in self._action_handlers:
            try:
                return self._action_handlers[handler_name](context, key, value)
            except Exception:
                return None

        return None

    def _learn_association_rules(self, cases: List[CaseEntry],
                                min_support: float,
                                min_confidence: float) -> List[KnowledgeRule]:
        """Learn association rules from cases."""
        rules = []
        total_cases = len(cases)

        # Extract item sets from case tags
        item_sets = []
        for case in cases:
            if case.status == CaseStatus.RESOLVED:
                items = set(case.tags)
                items.add(f"outcome:{case.outcome}" if case.outcome else "outcome:unknown")
                item_sets.append(items)

        if not item_sets:
            return []

        # Find frequent item pairs
        pair_counts = defaultdict(int)
        item_counts = defaultdict(int)

        for items in item_sets:
            item_list = list(items)
            for item in item_list:
                item_counts[item] += 1
            for i, item1 in enumerate(item_list):
                for item2 in item_list[i+1:]:
                    pair_counts[(item1, item2)] += 1
                    pair_counts[(item2, item1)] += 1

        # Generate rules from frequent pairs
        for (antecedent, consequent), count in pair_counts.items():
            support = count / total_cases
            if support < min_support:
                continue

            confidence = count / item_counts[antecedent] if item_counts[antecedent] > 0 else 0
            if confidence < min_confidence:
                continue

            rule = KnowledgeRule(
                name=f"Association: {antecedent} -> {consequent}",
                description=f"When {antecedent}, then {consequent}",
                rule_type=RuleType.ASSOCIATION,
                condition=f"contains:tags:{antecedent}",
                action=f"append:suggestions:{consequent}",
                confidence=confidence,
                support=support,
                source_cases=[c.id for c in cases]
            )
            rules.append(rule)

        return rules

    def _learn_classification_rules(self, cases: List[CaseEntry],
                                   min_support: float,
                                   min_confidence: float) -> List[KnowledgeRule]:
        """Learn classification rules from cases."""
        rules = []

        # Group cases by outcome
        outcome_groups = defaultdict(list)
        for case in cases:
            if case.status == CaseStatus.RESOLVED and case.outcome:
                outcome_groups[case.outcome].append(case)

        total_cases = len(cases)

        # Find common patterns for each outcome
        for outcome, outcome_cases in outcome_groups.items():
            support = len(outcome_cases) / total_cases
            if support < min_support:
                continue

            # Find common tags
            if outcome_cases:
                common_tags = set(outcome_cases[0].tags)
                for case in outcome_cases[1:]:
                    common_tags &= set(case.tags)

                for tag in common_tags:
                    # Calculate confidence
                    tag_cases = [c for c in cases if tag in c.tags]
                    if tag_cases:
                        confidence = len([c for c in tag_cases if c.outcome == outcome]) / len(tag_cases)

                        if confidence >= min_confidence:
                            rule = KnowledgeRule(
                                name=f"Classification: {tag} -> {outcome}",
                                description=f"Cases with tag '{tag}' tend to have outcome '{outcome}'",
                                rule_type=RuleType.CLASSIFICATION,
                                condition=f"contains:tags:{tag}",
                                action=f"set:predicted_outcome:{outcome}",
                                confidence=confidence,
                                support=support,
                                source_cases=[c.id for c in outcome_cases]
                            )
                            rules.append(rule)

        return rules

    def _learn_inference_rules(self, cases: List[CaseEntry],
                              min_support: float,
                              min_confidence: float) -> List[KnowledgeRule]:
        """Learn inference rules from cases."""
        rules = []

        # Look for problem-solution patterns
        problem_solutions = defaultdict(list)

        for case in cases:
            if case.status == CaseStatus.RESOLVED and case.solution:
                # Extract key terms from problem
                problem_key = self._extract_key_terms(case.problem)
                if problem_key:
                    problem_solutions[problem_key].append(case.solution)

        total_cases = len(cases)

        for problem_key, solutions in problem_solutions.items():
            support = len(solutions) / total_cases
            if support < min_support:
                continue

            # Find most common solution
            solution_counts = defaultdict(int)
            for sol in solutions:
                sol_key = self._extract_key_terms(sol)
                solution_counts[sol_key] += 1

            if solution_counts:
                best_solution, count = max(solution_counts.items(), key=lambda x: x[1])
                confidence = count / len(solutions)

                if confidence >= min_confidence:
                    rule = KnowledgeRule(
                        name=f"Inference: {problem_key[:30]}...",
                        description=f"For problems involving '{problem_key}', suggest '{best_solution}'",
                        rule_type=RuleType.INFERENCE,
                        condition=f"contains:problem:{problem_key}",
                        action=f"set:suggested_solution:{best_solution}",
                        confidence=confidence,
                        support=support
                    )
                    rules.append(rule)

        return rules

    def _extract_key_terms(self, text: str) -> str:
        """Extract key terms from text."""
        # Simple extraction - remove common words and take first few words
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                     "have", "has", "had", "do", "does", "did", "will", "would",
                     "could", "should", "may", "might", "must", "shall", "can",
                     "的", "是", "在", "有", "和", "了", "不", "我", "这", "那"}

        words = text.lower().split()
        key_words = [w for w in words if w not in stop_words and len(w) > 2]

        return " ".join(key_words[:5])

    def get_statistics(self) -> Dict[str, Any]:
        """Get rule engine statistics."""
        type_counts = {}
        for rule_type in RuleType:
            type_counts[rule_type.value] = len(self._type_index.get(rule_type, []))

        enabled_rules = sum(1 for r in self._rules.values() if r.is_enabled)
        total_hits = sum(r.hit_count for r in self._rules.values())
        total_successes = sum(r.success_count for r in self._rules.values())

        return {
            "total_rules": len(self._rules),
            "enabled_rules": enabled_rules,
            "by_type": type_counts,
            "total_hits": total_hits,
            "total_successes": total_successes,
            "overall_success_rate": total_successes / total_hits if total_hits > 0 else 0
        }


# Global instance
_rule_engine: Optional[RuleEngine] = None


def get_rule_engine() -> RuleEngine:
    """Get or create global RuleEngine instance."""
    global _rule_engine

    if _rule_engine is None:
        _rule_engine = RuleEngine()

    return _rule_engine
