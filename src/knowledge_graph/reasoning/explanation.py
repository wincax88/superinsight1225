"""
Reasoning explanation module for Knowledge Graph.

Provides inference result validation, explanation generation, and confidence calculation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ExplanationType(str, Enum):
    """Type of explanation."""

    RULE_BASED = "rule_based"
    SIMILARITY_BASED = "similarity_based"
    PATH_BASED = "path_based"
    EVIDENCE_BASED = "evidence_based"
    ANALOGY_BASED = "analogy_based"
    COUNTERFACTUAL = "counterfactual"


class ConfidenceLevel(str, Enum):
    """Confidence level categories."""

    VERY_HIGH = "very_high"  # > 0.9
    HIGH = "high"  # > 0.75
    MEDIUM = "medium"  # > 0.5
    LOW = "low"  # > 0.25
    VERY_LOW = "very_low"  # <= 0.25


class ValidationStatus(str, Enum):
    """Validation result status."""

    VALID = "valid"
    INVALID = "invalid"
    UNCERTAIN = "uncertain"
    PARTIAL = "partial"


class EvidenceItem(BaseModel):
    """A single piece of evidence supporting an inference."""

    evidence_id: str = Field(..., description="Unique evidence identifier")
    evidence_type: str = Field(..., description="Type of evidence")
    content: str = Field(..., description="Evidence content/description")
    source: str = Field(default="", description="Source of evidence")
    weight: float = Field(default=1.0, description="Weight in overall assessment")
    confidence: float = Field(default=1.0, description="Confidence in this evidence")
    timestamp: datetime = Field(default_factory=datetime.now)


class ReasoningPath(BaseModel):
    """A path of reasoning from premises to conclusion."""

    path_id: str = Field(..., description="Unique path identifier")
    premises: list[str] = Field(default_factory=list, description="Starting premises")
    steps: list[str] = Field(default_factory=list, description="Reasoning steps")
    conclusion: str = Field(..., description="Final conclusion")
    rules_applied: list[str] = Field(
        default_factory=list, description="Rules used in reasoning"
    )
    confidence: float = Field(default=1.0, description="Path confidence")


class Explanation(BaseModel):
    """Complete explanation for an inference result."""

    explanation_id: str = Field(..., description="Unique explanation identifier")
    inference_id: str = Field(..., description="Related inference ID")
    explanation_type: ExplanationType = Field(..., description="Type of explanation")
    summary: str = Field(..., description="Brief summary")
    detailed_explanation: str = Field(..., description="Detailed explanation")
    evidence: list[EvidenceItem] = Field(
        default_factory=list, description="Supporting evidence"
    )
    reasoning_paths: list[ReasoningPath] = Field(
        default_factory=list, description="Reasoning paths"
    )
    confidence: float = Field(default=1.0, description="Overall confidence")
    confidence_level: ConfidenceLevel = Field(
        default=ConfidenceLevel.MEDIUM, description="Confidence category"
    )
    alternatives: list[str] = Field(
        default_factory=list, description="Alternative explanations"
    )
    caveats: list[str] = Field(
        default_factory=list, description="Limitations and caveats"
    )
    timestamp: datetime = Field(default_factory=datetime.now)


class ValidationResult(BaseModel):
    """Result of validating an inference."""

    validation_id: str = Field(..., description="Unique validation identifier")
    inference_id: str = Field(..., description="Inference being validated")
    status: ValidationStatus = Field(..., description="Validation status")
    is_valid: bool = Field(default=False, description="Whether inference is valid")
    confidence: float = Field(default=0.0, description="Validation confidence")
    checks_passed: list[str] = Field(
        default_factory=list, description="Checks that passed"
    )
    checks_failed: list[str] = Field(
        default_factory=list, description="Checks that failed"
    )
    warnings: list[str] = Field(default_factory=list, description="Validation warnings")
    suggestions: list[str] = Field(
        default_factory=list, description="Improvement suggestions"
    )
    timestamp: datetime = Field(default_factory=datetime.now)


class ConfidenceBreakdown(BaseModel):
    """Detailed breakdown of confidence calculation."""

    overall_confidence: float = Field(..., description="Final confidence score")
    confidence_level: ConfidenceLevel = Field(..., description="Confidence category")
    components: dict[str, float] = Field(
        default_factory=dict, description="Component scores"
    )
    weights: dict[str, float] = Field(
        default_factory=dict, description="Component weights"
    )
    contributing_factors: list[str] = Field(
        default_factory=list, description="Positive factors"
    )
    reducing_factors: list[str] = Field(
        default_factory=list, description="Negative factors"
    )
    explanation: str = Field(default="", description="Natural language explanation")


@dataclass
class ReasoningExplainer:
    """Explainer for reasoning and inference results."""

    explanation_templates: dict[str, str] = field(default_factory=dict)
    confidence_thresholds: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        # Initialize default templates
        self.explanation_templates = {
            "rule_based": "基于规则 '{rule_name}'，从 {premises} 推导出 {conclusion}。",
            "similarity_based": "基于相似度分析，{entity1} 与 {entity2} 的相似度为 {similarity:.2%}。",
            "path_based": "通过路径 {path} 连接 {source} 和 {target}。",
            "evidence_based": "基于 {evidence_count} 条证据支持，推断 {conclusion}。",
            "analogy_based": "类比 {source_case}，可以推断 {target_case}。",
            "counterfactual": "如果没有 {factor}，结果可能是 {alternative}。",
        }

        self.confidence_thresholds = {
            "very_high": 0.9,
            "high": 0.75,
            "medium": 0.5,
            "low": 0.25,
        }

    def explain_inference(
        self,
        inference_id: str,
        inference_result: dict[str, Any],
        explanation_type: ExplanationType = ExplanationType.RULE_BASED,
    ) -> Explanation:
        """Generate explanation for an inference result."""
        import uuid

        explanation_id = f"exp_{uuid.uuid4().hex[:8]}"

        # Generate summary based on type
        summary = self._generate_summary(inference_result, explanation_type)

        # Generate detailed explanation
        detailed = self._generate_detailed_explanation(inference_result, explanation_type)

        # Extract evidence
        evidence = self._extract_evidence(inference_result)

        # Build reasoning paths
        reasoning_paths = self._build_reasoning_paths(inference_result)

        # Calculate confidence
        confidence = self._calculate_explanation_confidence(inference_result)
        confidence_level = self._get_confidence_level(confidence)

        # Generate alternatives and caveats
        alternatives = self._generate_alternatives(inference_result)
        caveats = self._generate_caveats(inference_result, confidence)

        return Explanation(
            explanation_id=explanation_id,
            inference_id=inference_id,
            explanation_type=explanation_type,
            summary=summary,
            detailed_explanation=detailed,
            evidence=evidence,
            reasoning_paths=reasoning_paths,
            confidence=confidence,
            confidence_level=confidence_level,
            alternatives=alternatives,
            caveats=caveats,
        )

    def validate_inference(
        self,
        inference_id: str,
        inference_result: dict[str, Any],
        known_facts: Optional[dict[str, Any]] = None,
    ) -> ValidationResult:
        """Validate an inference result."""
        import uuid

        validation_id = f"val_{uuid.uuid4().hex[:8]}"
        checks_passed = []
        checks_failed = []
        warnings = []
        suggestions = []

        # Check 1: Completeness
        if self._check_completeness(inference_result):
            checks_passed.append("完整性检查：推理结果包含所有必需字段")
        else:
            checks_failed.append("完整性检查：推理结果缺少必需字段")
            suggestions.append("确保推理结果包含 source、target 和 relation 字段")

        # Check 2: Consistency
        if self._check_consistency(inference_result, known_facts):
            checks_passed.append("一致性检查：推理结果与已知事实一致")
        else:
            checks_failed.append("一致性检查：推理结果与已知事实存在冲突")
            warnings.append("存在与已知事实的潜在冲突")

        # Check 3: Confidence threshold
        confidence = inference_result.get("confidence", 0.0)
        if confidence >= 0.5:
            checks_passed.append(f"置信度检查：{confidence:.2%} >= 50%")
        else:
            checks_failed.append(f"置信度检查：{confidence:.2%} < 50%")
            suggestions.append("考虑收集更多证据以提高置信度")

        # Check 4: Evidence support
        evidence_count = len(inference_result.get("evidence", []))
        if evidence_count >= 1:
            checks_passed.append(f"证据支持检查：有 {evidence_count} 条证据支持")
        else:
            warnings.append("警告：没有直接证据支持此推理")
            suggestions.append("建议添加支持性证据")

        # Check 5: Logical validity
        if self._check_logical_validity(inference_result):
            checks_passed.append("逻辑有效性检查：推理链逻辑有效")
        else:
            checks_failed.append("逻辑有效性检查：推理链存在逻辑问题")

        # Determine status
        if not checks_failed:
            status = ValidationStatus.VALID
            is_valid = True
        elif len(checks_passed) > len(checks_failed):
            status = ValidationStatus.PARTIAL
            is_valid = False
        elif checks_failed:
            status = ValidationStatus.INVALID
            is_valid = False
        else:
            status = ValidationStatus.UNCERTAIN
            is_valid = False

        validation_confidence = len(checks_passed) / (
            len(checks_passed) + len(checks_failed)
        ) if (checks_passed or checks_failed) else 0.5

        return ValidationResult(
            validation_id=validation_id,
            inference_id=inference_id,
            status=status,
            is_valid=is_valid,
            confidence=validation_confidence,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            warnings=warnings,
            suggestions=suggestions,
        )

    def calculate_confidence(
        self,
        inference_result: dict[str, Any],
    ) -> ConfidenceBreakdown:
        """Calculate detailed confidence breakdown."""
        components = {}
        weights = {}

        # Component 1: Rule confidence
        rule_confidence = inference_result.get("rule_confidence", 0.8)
        components["rule_confidence"] = rule_confidence
        weights["rule_confidence"] = 0.3

        # Component 2: Evidence strength
        evidence = inference_result.get("evidence", [])
        if evidence:
            evidence_scores = [e.get("confidence", 0.5) for e in evidence]
            evidence_strength = sum(evidence_scores) / len(evidence_scores)
        else:
            evidence_strength = 0.5
        components["evidence_strength"] = evidence_strength
        weights["evidence_strength"] = 0.3

        # Component 3: Path reliability
        path_length = inference_result.get("path_length", 1)
        path_reliability = max(0.3, 1.0 - (path_length - 1) * 0.1)
        components["path_reliability"] = path_reliability
        weights["path_reliability"] = 0.2

        # Component 4: Source quality
        source_quality = inference_result.get("source_quality", 0.7)
        components["source_quality"] = source_quality
        weights["source_quality"] = 0.2

        # Calculate weighted average
        overall = sum(
            components[k] * weights[k] for k in components
        )

        # Identify factors
        contributing = []
        reducing = []

        for component, score in components.items():
            if score >= 0.8:
                contributing.append(f"{component}: {score:.2%}")
            elif score < 0.5:
                reducing.append(f"{component}: {score:.2%}")

        confidence_level = self._get_confidence_level(overall)

        explanation = self._generate_confidence_explanation(
            overall, contributing, reducing
        )

        return ConfidenceBreakdown(
            overall_confidence=overall,
            confidence_level=confidence_level,
            components=components,
            weights=weights,
            contributing_factors=contributing,
            reducing_factors=reducing,
            explanation=explanation,
        )

    def generate_natural_language_explanation(
        self,
        explanation: Explanation,
        language: str = "zh",
    ) -> str:
        """Generate natural language explanation."""
        if language == "zh":
            return self._generate_chinese_explanation(explanation)
        else:
            return self._generate_english_explanation(explanation)

    def _generate_chinese_explanation(self, explanation: Explanation) -> str:
        """Generate Chinese explanation."""
        lines = []

        # Summary
        lines.append(f"【摘要】{explanation.summary}")
        lines.append("")

        # Detailed explanation
        lines.append("【详细说明】")
        lines.append(explanation.detailed_explanation)
        lines.append("")

        # Evidence
        if explanation.evidence:
            lines.append(f"【证据支持】共 {len(explanation.evidence)} 条证据：")
            for i, evidence in enumerate(explanation.evidence, 1):
                lines.append(f"  {i}. {evidence.content} (置信度: {evidence.confidence:.0%})")
            lines.append("")

        # Reasoning paths
        if explanation.reasoning_paths:
            lines.append("【推理路径】")
            for path in explanation.reasoning_paths:
                lines.append(f"  前提: {', '.join(path.premises)}")
                lines.append(f"  步骤: {' -> '.join(path.steps)}")
                lines.append(f"  结论: {path.conclusion}")
                lines.append(f"  置信度: {path.confidence:.0%}")
                lines.append("")

        # Confidence
        lines.append(f"【置信度】{explanation.confidence:.0%} ({explanation.confidence_level.value})")
        lines.append("")

        # Caveats
        if explanation.caveats:
            lines.append("【注意事项】")
            for caveat in explanation.caveats:
                lines.append(f"  - {caveat}")
            lines.append("")

        # Alternatives
        if explanation.alternatives:
            lines.append("【其他可能】")
            for alt in explanation.alternatives:
                lines.append(f"  - {alt}")

        return "\n".join(lines)

    def _generate_english_explanation(self, explanation: Explanation) -> str:
        """Generate English explanation."""
        lines = []

        lines.append(f"[Summary] {explanation.summary}")
        lines.append("")

        lines.append("[Detailed Explanation]")
        lines.append(explanation.detailed_explanation)
        lines.append("")

        if explanation.evidence:
            lines.append(f"[Evidence] {len(explanation.evidence)} pieces of evidence:")
            for i, evidence in enumerate(explanation.evidence, 1):
                lines.append(f"  {i}. {evidence.content} (confidence: {evidence.confidence:.0%})")
            lines.append("")

        if explanation.reasoning_paths:
            lines.append("[Reasoning Paths]")
            for path in explanation.reasoning_paths:
                lines.append(f"  Premises: {', '.join(path.premises)}")
                lines.append(f"  Steps: {' -> '.join(path.steps)}")
                lines.append(f"  Conclusion: {path.conclusion}")
                lines.append("")

        lines.append(f"[Confidence] {explanation.confidence:.0%} ({explanation.confidence_level.value})")

        if explanation.caveats:
            lines.append("")
            lines.append("[Caveats]")
            for caveat in explanation.caveats:
                lines.append(f"  - {caveat}")

        return "\n".join(lines)

    def _generate_summary(
        self,
        inference_result: dict[str, Any],
        explanation_type: ExplanationType,
    ) -> str:
        """Generate summary based on inference type."""
        source = inference_result.get("source", "实体A")
        target = inference_result.get("target", "实体B")
        relation = inference_result.get("relation", "关系")
        confidence = inference_result.get("confidence", 0.0)

        if explanation_type == ExplanationType.RULE_BASED:
            return f"通过规则推理，推断 {source} 与 {target} 之间存在 {relation} 关系（置信度 {confidence:.0%}）"
        elif explanation_type == ExplanationType.SIMILARITY_BASED:
            return f"基于相似性分析，{source} 与 {target} 高度相关"
        elif explanation_type == ExplanationType.PATH_BASED:
            return f"通过路径分析，发现 {source} 到 {target} 的连接"
        elif explanation_type == ExplanationType.EVIDENCE_BASED:
            evidence_count = len(inference_result.get("evidence", []))
            return f"基于 {evidence_count} 条证据，推断 {source} 与 {target} 之间的关系"
        else:
            return f"推理结果：{source} -> {relation} -> {target}"

    def _generate_detailed_explanation(
        self,
        inference_result: dict[str, Any],
        explanation_type: ExplanationType,
    ) -> str:
        """Generate detailed explanation."""
        source = inference_result.get("source", "实体A")
        target = inference_result.get("target", "实体B")
        relation = inference_result.get("relation", "关系")

        details = [
            f"源实体：{source}",
            f"目标实体：{target}",
            f"关系类型：{relation}",
        ]

        if "rule_name" in inference_result:
            details.append(f"应用规则：{inference_result['rule_name']}")

        if "steps" in inference_result:
            details.append(f"推理步骤数：{len(inference_result['steps'])}")

        if "path" in inference_result:
            details.append(f"连接路径：{' -> '.join(inference_result['path'])}")

        return "\n".join(details)

    def _extract_evidence(
        self,
        inference_result: dict[str, Any],
    ) -> list[EvidenceItem]:
        """Extract evidence from inference result."""
        evidence_list = []
        raw_evidence = inference_result.get("evidence", [])

        for i, ev in enumerate(raw_evidence):
            if isinstance(ev, dict):
                evidence_list.append(
                    EvidenceItem(
                        evidence_id=ev.get("id", f"ev_{i}"),
                        evidence_type=ev.get("type", "fact"),
                        content=ev.get("content", str(ev)),
                        source=ev.get("source", ""),
                        weight=ev.get("weight", 1.0),
                        confidence=ev.get("confidence", 0.8),
                    )
                )
            else:
                evidence_list.append(
                    EvidenceItem(
                        evidence_id=f"ev_{i}",
                        evidence_type="fact",
                        content=str(ev),
                    )
                )

        return evidence_list

    def _build_reasoning_paths(
        self,
        inference_result: dict[str, Any],
    ) -> list[ReasoningPath]:
        """Build reasoning paths from inference result."""
        paths = []

        if "steps" in inference_result:
            import uuid

            premises = inference_result.get("premises", [])
            steps = inference_result.get("steps", [])
            conclusion = inference_result.get("conclusion", "")
            rules = inference_result.get("rules_applied", [])

            paths.append(
                ReasoningPath(
                    path_id=f"path_{uuid.uuid4().hex[:8]}",
                    premises=premises if isinstance(premises, list) else [str(premises)],
                    steps=steps if isinstance(steps, list) else [str(steps)],
                    conclusion=conclusion,
                    rules_applied=rules if isinstance(rules, list) else [str(rules)],
                    confidence=inference_result.get("confidence", 0.8),
                )
            )

        return paths

    def _calculate_explanation_confidence(
        self,
        inference_result: dict[str, Any],
    ) -> float:
        """Calculate confidence for explanation."""
        base_confidence = inference_result.get("confidence", 0.5)

        # Adjust based on evidence
        evidence = inference_result.get("evidence", [])
        if evidence:
            evidence_boost = min(0.2, len(evidence) * 0.05)
            base_confidence = min(1.0, base_confidence + evidence_boost)

        return base_confidence

    def _get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Get confidence level category."""
        if confidence > self.confidence_thresholds["very_high"]:
            return ConfidenceLevel.VERY_HIGH
        elif confidence > self.confidence_thresholds["high"]:
            return ConfidenceLevel.HIGH
        elif confidence > self.confidence_thresholds["medium"]:
            return ConfidenceLevel.MEDIUM
        elif confidence > self.confidence_thresholds["low"]:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW

    def _generate_alternatives(
        self,
        inference_result: dict[str, Any],
    ) -> list[str]:
        """Generate alternative explanations."""
        alternatives = []

        confidence = inference_result.get("confidence", 0.5)
        if confidence < 0.8:
            alternatives.append("可能存在其他未被考虑的关系类型")

        if "similar_entities" in inference_result:
            alternatives.append("相似实体可能具有类似的关系模式")

        return alternatives

    def _generate_caveats(
        self,
        inference_result: dict[str, Any],
        confidence: float,
    ) -> list[str]:
        """Generate caveats and limitations."""
        caveats = []

        if confidence < 0.7:
            caveats.append("推理置信度较低，建议人工验证")

        evidence = inference_result.get("evidence", [])
        if len(evidence) < 2:
            caveats.append("证据数量有限，可能需要更多支持")

        if inference_result.get("path_length", 1) > 3:
            caveats.append("推理路径较长，可能存在累积误差")

        return caveats

    def _check_completeness(self, inference_result: dict[str, Any]) -> bool:
        """Check if inference result has all required fields."""
        required_fields = ["source", "target", "relation"]
        return all(field in inference_result for field in required_fields)

    def _check_consistency(
        self,
        inference_result: dict[str, Any],
        known_facts: Optional[dict[str, Any]],
    ) -> bool:
        """Check consistency with known facts."""
        if not known_facts:
            return True

        # Check for contradictions
        source = inference_result.get("source")
        target = inference_result.get("target")
        relation = inference_result.get("relation")

        # Check if there's a contradicting fact
        key = f"{source}_{target}"
        if key in known_facts:
            existing_relation = known_facts[key].get("relation")
            if existing_relation and existing_relation != relation:
                return False

        return True

    def _check_logical_validity(self, inference_result: dict[str, Any]) -> bool:
        """Check logical validity of inference."""
        # Check for circular reasoning
        source = inference_result.get("source")
        target = inference_result.get("target")

        if source == target:
            return False

        # Check for valid steps
        steps = inference_result.get("steps", [])
        if steps and not all(steps):
            return False

        return True

    def _generate_confidence_explanation(
        self,
        confidence: float,
        contributing: list[str],
        reducing: list[str],
    ) -> str:
        """Generate natural language confidence explanation."""
        level = "高" if confidence >= 0.75 else "中" if confidence >= 0.5 else "低"

        explanation = f"总体置信度为 {confidence:.0%}（{level}）。"

        if contributing:
            explanation += f" 主要贡献因素：{', '.join(contributing)}。"

        if reducing:
            explanation += f" 降低因素：{', '.join(reducing)}。"

        return explanation


# Global instance
_reasoning_explainer: Optional[ReasoningExplainer] = None


def get_reasoning_explainer() -> ReasoningExplainer:
    """Get or create the global reasoning explainer instance."""
    global _reasoning_explainer
    if _reasoning_explainer is None:
        _reasoning_explainer = ReasoningExplainer()
    return _reasoning_explainer
