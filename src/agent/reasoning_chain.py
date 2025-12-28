"""
Reasoning Chain Module for AI Agent System.

Implements multi-step reasoning logic, reasoning process recording,
hypothesis verification, and confidence evaluation.
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional, Callable, Tuple

logger = logging.getLogger(__name__)


class ReasoningStepType(str, Enum):
    """Types of reasoning steps."""
    OBSERVATION = "observation"      # Observe input data
    HYPOTHESIS = "hypothesis"        # Form hypothesis
    INFERENCE = "inference"          # Make inference
    VERIFICATION = "verification"    # Verify hypothesis
    CONCLUSION = "conclusion"        # Draw conclusion
    BACKTRACK = "backtrack"          # Backtrack due to failure


class ReasoningStatus(str, Enum):
    """Status of reasoning process."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BACKTRACKED = "backtracked"


@dataclass
class ReasoningStep:
    """A single step in the reasoning chain."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    step_type: ReasoningStepType = ReasoningStepType.OBSERVATION
    description: str = ""
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    reasoning: str = ""
    evidence: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    status: ReasoningStatus = ReasoningStatus.PENDING
    error_message: Optional[str] = None
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "step_type": self.step_type.value,
            "description": self.description,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "evidence": self.evidence,
            "dependencies": self.dependencies,
            "status": self.status.value,
            "error_message": self.error_message,
            "execution_time": self.execution_time,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class Hypothesis:
    """A hypothesis to be verified."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    statement: str = ""
    supporting_evidence: List[str] = field(default_factory=list)
    contradicting_evidence: List[str] = field(default_factory=list)
    confidence: float = 0.0
    verified: bool = False
    verification_result: Optional[bool] = None
    verification_reasoning: str = ""
    created_at: datetime = field(default_factory=datetime.now)

    def calculate_confidence(self) -> float:
        """Calculate confidence based on evidence."""
        support_count = len(self.supporting_evidence)
        contradict_count = len(self.contradicting_evidence)
        total = support_count + contradict_count

        if total == 0:
            return 0.5  # No evidence, neutral confidence

        # Base confidence from evidence ratio
        base_confidence = support_count / total

        # Adjust for total evidence (more evidence = more confident)
        evidence_factor = min(1.0, total / 5)  # Max out at 5 pieces of evidence

        # Final confidence
        self.confidence = base_confidence * (0.5 + 0.5 * evidence_factor)
        return self.confidence


@dataclass
class ReasoningChain:
    """A chain of reasoning steps."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    goal: str = ""
    steps: List[ReasoningStep] = field(default_factory=list)
    hypotheses: List[Hypothesis] = field(default_factory=list)
    current_step_index: int = 0
    status: ReasoningStatus = ReasoningStatus.PENDING
    overall_confidence: float = 0.0
    conclusion: Optional[Dict[str, Any]] = None
    backtrack_history: List[Tuple[int, str]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    total_execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_step(self, step: ReasoningStep) -> None:
        """Add a step to the chain."""
        self.steps.append(step)

    def get_current_step(self) -> Optional[ReasoningStep]:
        """Get the current step."""
        if 0 <= self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None

    def advance(self) -> bool:
        """Advance to the next step."""
        if self.current_step_index < len(self.steps) - 1:
            self.current_step_index += 1
            return True
        return False

    def backtrack(self, reason: str) -> bool:
        """Backtrack to the previous step."""
        if self.current_step_index > 0:
            self.backtrack_history.append((self.current_step_index, reason))
            current = self.get_current_step()
            if current:
                current.status = ReasoningStatus.BACKTRACKED
            self.current_step_index -= 1
            return True
        return False

    def calculate_overall_confidence(self) -> float:
        """Calculate overall confidence from all steps."""
        if not self.steps:
            return 0.0

        completed_steps = [s for s in self.steps if s.status == ReasoningStatus.COMPLETED]
        if not completed_steps:
            return 0.0

        # Weight later steps more heavily (conclusions more important than observations)
        weights = [1.0 + i * 0.2 for i in range(len(completed_steps))]
        total_weight = sum(weights)

        weighted_confidence = sum(
            step.confidence * weight
            for step, weight in zip(completed_steps, weights)
        )

        self.overall_confidence = weighted_confidence / total_weight
        return self.overall_confidence

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "goal": self.goal,
            "steps": [s.to_dict() for s in self.steps],
            "hypotheses": [
                {
                    "id": h.id,
                    "statement": h.statement,
                    "confidence": h.confidence,
                    "verified": h.verified,
                    "verification_result": h.verification_result
                }
                for h in self.hypotheses
            ],
            "current_step_index": self.current_step_index,
            "status": self.status.value,
            "overall_confidence": self.overall_confidence,
            "conclusion": self.conclusion,
            "backtrack_count": len(self.backtrack_history),
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "total_execution_time": self.total_execution_time
        }


class ReasoningChainBuilder:
    """Builder for constructing reasoning chains."""

    def __init__(self):
        """Initialize the reasoning chain builder."""
        self.chain: Optional[ReasoningChain] = None
        self.step_executors: Dict[ReasoningStepType, Callable] = {}
        self.hypothesis_verifiers: List[Callable] = []
        self.max_backtrack_depth = 3
        self.confidence_threshold = 0.6

    def create_chain(
        self,
        name: str,
        goal: str,
        description: str = ""
    ) -> ReasoningChain:
        """Create a new reasoning chain."""
        self.chain = ReasoningChain(
            name=name,
            goal=goal,
            description=description
        )
        return self.chain

    def add_observation_step(
        self,
        description: str,
        input_data: Dict[str, Any],
        dependencies: Optional[List[str]] = None
    ) -> ReasoningStep:
        """Add an observation step."""
        step = ReasoningStep(
            step_type=ReasoningStepType.OBSERVATION,
            description=description,
            input_data=input_data,
            dependencies=dependencies or []
        )
        if self.chain:
            self.chain.add_step(step)
        return step

    def add_hypothesis_step(
        self,
        statement: str,
        based_on: Optional[List[str]] = None
    ) -> ReasoningStep:
        """Add a hypothesis formation step."""
        hypothesis = Hypothesis(statement=statement)
        if self.chain:
            self.chain.hypotheses.append(hypothesis)

        step = ReasoningStep(
            step_type=ReasoningStepType.HYPOTHESIS,
            description=f"Form hypothesis: {statement}",
            input_data={"statement": statement},
            dependencies=based_on or [],
            metadata={"hypothesis_id": hypothesis.id}
        )
        if self.chain:
            self.chain.add_step(step)
        return step

    def add_inference_step(
        self,
        description: str,
        inference_logic: str,
        dependencies: Optional[List[str]] = None
    ) -> ReasoningStep:
        """Add an inference step."""
        step = ReasoningStep(
            step_type=ReasoningStepType.INFERENCE,
            description=description,
            input_data={"logic": inference_logic},
            reasoning=inference_logic,
            dependencies=dependencies or []
        )
        if self.chain:
            self.chain.add_step(step)
        return step

    def add_verification_step(
        self,
        hypothesis_id: str,
        verification_method: str,
        dependencies: Optional[List[str]] = None
    ) -> ReasoningStep:
        """Add a verification step for a hypothesis."""
        step = ReasoningStep(
            step_type=ReasoningStepType.VERIFICATION,
            description=f"Verify hypothesis: {hypothesis_id}",
            input_data={
                "hypothesis_id": hypothesis_id,
                "method": verification_method
            },
            dependencies=dependencies or []
        )
        if self.chain:
            self.chain.add_step(step)
        return step

    def add_conclusion_step(
        self,
        description: str,
        dependencies: Optional[List[str]] = None
    ) -> ReasoningStep:
        """Add a conclusion step."""
        step = ReasoningStep(
            step_type=ReasoningStepType.CONCLUSION,
            description=description,
            dependencies=dependencies or []
        )
        if self.chain:
            self.chain.add_step(step)
        return step

    def register_step_executor(
        self,
        step_type: ReasoningStepType,
        executor: Callable[[ReasoningStep], Dict[str, Any]]
    ) -> None:
        """Register a custom executor for a step type."""
        self.step_executors[step_type] = executor

    def register_hypothesis_verifier(
        self,
        verifier: Callable[[Hypothesis, Dict[str, Any]], Tuple[bool, float, str]]
    ) -> None:
        """Register a hypothesis verifier function."""
        self.hypothesis_verifiers.append(verifier)


class ReasoningEngine:
    """Engine for executing reasoning chains."""

    def __init__(self):
        """Initialize the reasoning engine."""
        self.active_chains: Dict[str, ReasoningChain] = {}
        self.completed_chains: List[ReasoningChain] = []
        self.step_executors: Dict[ReasoningStepType, Callable] = {}
        self.default_confidence = 0.5
        self.max_iterations = 100
        self.backtrack_threshold = 0.3

        # Register default step executors
        self._register_default_executors()

    def _register_default_executors(self) -> None:
        """Register default step executors."""
        self.step_executors[ReasoningStepType.OBSERVATION] = self._execute_observation
        self.step_executors[ReasoningStepType.HYPOTHESIS] = self._execute_hypothesis
        self.step_executors[ReasoningStepType.INFERENCE] = self._execute_inference
        self.step_executors[ReasoningStepType.VERIFICATION] = self._execute_verification
        self.step_executors[ReasoningStepType.CONCLUSION] = self._execute_conclusion

    def _execute_observation(
        self,
        step: ReasoningStep,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute an observation step."""
        # Analyze input data
        input_data = step.input_data

        observations = {
            "data_type": type(input_data).__name__,
            "key_fields": list(input_data.keys()) if isinstance(input_data, dict) else [],
            "data_size": len(str(input_data)),
            "observed_patterns": []
        }

        # Pattern detection (simplified)
        if isinstance(input_data, dict):
            for key, value in input_data.items():
                if isinstance(value, (int, float)):
                    observations["observed_patterns"].append(f"numeric_{key}")
                elif isinstance(value, str):
                    observations["observed_patterns"].append(f"text_{key}")
                elif isinstance(value, list):
                    observations["observed_patterns"].append(f"list_{key}")

        step.output_data = observations
        step.confidence = 0.9  # Observations are usually high confidence
        step.reasoning = f"Observed {len(observations.get('key_fields', []))} data fields"

        return observations

    def _execute_hypothesis(
        self,
        step: ReasoningStep,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a hypothesis formation step."""
        statement = step.input_data.get("statement", "")

        # Get supporting context from previous steps
        supporting_evidence = []
        for completed_step in context.get("completed_steps", []):
            if completed_step.status == ReasoningStatus.COMPLETED:
                supporting_evidence.append(completed_step.description)

        result = {
            "hypothesis": statement,
            "initial_support": len(supporting_evidence),
            "supporting_evidence": supporting_evidence
        }

        step.output_data = result
        step.evidence = supporting_evidence
        step.confidence = 0.5 + (0.1 * min(len(supporting_evidence), 3))  # More evidence = higher confidence
        step.reasoning = f"Hypothesis formed based on {len(supporting_evidence)} observations"

        return result

    def _execute_inference(
        self,
        step: ReasoningStep,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute an inference step."""
        logic = step.input_data.get("logic", "")

        # Get relevant context
        relevant_data = {}
        for dep_id in step.dependencies:
            for completed_step in context.get("completed_steps", []):
                if completed_step.id == dep_id:
                    relevant_data[dep_id] = completed_step.output_data

        # Simplified inference (in production, use LLM or rule engine)
        inference_result = {
            "inference_logic": logic,
            "based_on": list(relevant_data.keys()),
            "derived_conclusions": [],
            "confidence_factors": []
        }

        # Generate derived conclusions based on available data
        for step_id, data in relevant_data.items():
            if isinstance(data, dict):
                patterns = data.get("observed_patterns", [])
                for pattern in patterns:
                    inference_result["derived_conclusions"].append(
                        f"Pattern '{pattern}' suggests further investigation"
                    )

        step.output_data = inference_result
        step.confidence = 0.7 if inference_result["derived_conclusions"] else 0.4
        step.reasoning = f"Inferred {len(inference_result['derived_conclusions'])} conclusions"

        return inference_result

    def _execute_verification(
        self,
        step: ReasoningStep,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a verification step."""
        hypothesis_id = step.input_data.get("hypothesis_id", "")
        method = step.input_data.get("method", "default")

        # Find the hypothesis
        chain = context.get("chain")
        hypothesis = None
        if chain:
            for h in chain.hypotheses:
                if h.id == hypothesis_id:
                    hypothesis = h
                    break

        verification_result = {
            "hypothesis_id": hypothesis_id,
            "method": method,
            "verified": False,
            "confidence": 0.0,
            "reasoning": ""
        }

        if hypothesis:
            # Calculate hypothesis confidence
            hypothesis.calculate_confidence()

            # Verification logic
            if hypothesis.confidence >= 0.6:
                verification_result["verified"] = True
                verification_result["reasoning"] = "Sufficient supporting evidence found"
            elif hypothesis.confidence <= 0.3:
                verification_result["verified"] = False
                verification_result["reasoning"] = "Insufficient or contradicting evidence"
            else:
                verification_result["verified"] = None  # Inconclusive
                verification_result["reasoning"] = "Evidence is inconclusive, needs more investigation"

            verification_result["confidence"] = hypothesis.confidence

            # Update hypothesis
            hypothesis.verified = True
            hypothesis.verification_result = verification_result["verified"]
            hypothesis.verification_reasoning = verification_result["reasoning"]

        step.output_data = verification_result
        step.confidence = verification_result["confidence"]
        step.reasoning = verification_result["reasoning"]

        return verification_result

    def _execute_conclusion(
        self,
        step: ReasoningStep,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a conclusion step."""
        chain = context.get("chain")

        conclusion = {
            "summary": step.description,
            "verified_hypotheses": [],
            "rejected_hypotheses": [],
            "key_insights": [],
            "confidence": 0.0
        }

        if chain:
            for hypothesis in chain.hypotheses:
                if hypothesis.verified:
                    if hypothesis.verification_result:
                        conclusion["verified_hypotheses"].append(hypothesis.statement)
                    elif hypothesis.verification_result is False:
                        conclusion["rejected_hypotheses"].append(hypothesis.statement)

            # Extract key insights from inference steps
            for s in chain.steps:
                if s.step_type == ReasoningStepType.INFERENCE and s.status == ReasoningStatus.COMPLETED:
                    insights = s.output_data.get("derived_conclusions", [])
                    conclusion["key_insights"].extend(insights[:3])  # Top 3 insights per step

            # Calculate overall confidence
            conclusion["confidence"] = chain.calculate_overall_confidence()

        step.output_data = conclusion
        step.confidence = conclusion["confidence"]
        step.reasoning = f"Concluded with {len(conclusion['verified_hypotheses'])} verified hypotheses"

        return conclusion

    def execute_chain(
        self,
        chain: ReasoningChain,
        initial_context: Optional[Dict[str, Any]] = None
    ) -> ReasoningChain:
        """Execute a reasoning chain."""
        start_time = time.time()

        logger.info(f"Starting reasoning chain execution: {chain.name}")

        chain.status = ReasoningStatus.IN_PROGRESS
        self.active_chains[chain.id] = chain

        context = initial_context or {}
        context["chain"] = chain
        context["completed_steps"] = []

        iteration = 0
        backtrack_count = 0
        max_backtracks = len(chain.steps) * 2

        try:
            while chain.current_step_index < len(chain.steps) and iteration < self.max_iterations:
                iteration += 1
                current_step = chain.get_current_step()

                if not current_step or current_step.status == ReasoningStatus.COMPLETED:
                    if not chain.advance():
                        break
                    continue

                # Execute the step
                current_step.status = ReasoningStatus.IN_PROGRESS
                step_start = time.time()

                try:
                    executor = self.step_executors.get(current_step.step_type)
                    if executor:
                        result = executor(current_step, context)
                        current_step.status = ReasoningStatus.COMPLETED
                        context["completed_steps"].append(current_step)
                    else:
                        logger.warning(f"No executor for step type: {current_step.step_type}")
                        current_step.status = ReasoningStatus.FAILED
                        current_step.error_message = "No executor available"

                except Exception as e:
                    current_step.status = ReasoningStatus.FAILED
                    current_step.error_message = str(e)
                    logger.error(f"Step execution failed: {e}")

                current_step.execution_time = time.time() - step_start

                # Check if we need to backtrack
                if current_step.status == ReasoningStatus.FAILED or \
                   current_step.confidence < self.backtrack_threshold:
                    if backtrack_count < max_backtracks:
                        reason = current_step.error_message or f"Low confidence: {current_step.confidence}"
                        if chain.backtrack(reason):
                            backtrack_count += 1
                            logger.info(f"Backtracking: {reason}")
                            continue

                # Advance to next step
                chain.advance()

            # Set conclusion
            if chain.steps:
                last_step = chain.steps[-1]
                if last_step.step_type == ReasoningStepType.CONCLUSION:
                    chain.conclusion = last_step.output_data

            # Calculate final confidence
            chain.calculate_overall_confidence()

            chain.status = ReasoningStatus.COMPLETED
            chain.completed_at = datetime.now()

        except Exception as e:
            chain.status = ReasoningStatus.FAILED
            logger.error(f"Reasoning chain execution failed: {e}")

        chain.total_execution_time = time.time() - start_time

        # Move to completed
        if chain.id in self.active_chains:
            del self.active_chains[chain.id]
        self.completed_chains.append(chain)

        logger.info(
            f"Reasoning chain completed: {chain.name}, "
            f"confidence={chain.overall_confidence:.2f}, "
            f"time={chain.total_execution_time:.2f}s"
        )

        return chain

    def get_chain_summary(self, chain: ReasoningChain) -> Dict[str, Any]:
        """Get a summary of a reasoning chain execution."""
        step_summary = {}
        for step_type in ReasoningStepType:
            steps_of_type = [s for s in chain.steps if s.step_type == step_type]
            completed = [s for s in steps_of_type if s.status == ReasoningStatus.COMPLETED]
            step_summary[step_type.value] = {
                "total": len(steps_of_type),
                "completed": len(completed),
                "avg_confidence": sum(s.confidence for s in completed) / len(completed) if completed else 0.0
            }

        return {
            "chain_id": chain.id,
            "name": chain.name,
            "goal": chain.goal,
            "status": chain.status.value,
            "total_steps": len(chain.steps),
            "completed_steps": sum(1 for s in chain.steps if s.status == ReasoningStatus.COMPLETED),
            "backtrack_count": len(chain.backtrack_history),
            "overall_confidence": chain.overall_confidence,
            "step_summary": step_summary,
            "hypotheses_verified": sum(1 for h in chain.hypotheses if h.verification_result),
            "hypotheses_rejected": sum(1 for h in chain.hypotheses if h.verification_result is False),
            "execution_time": chain.total_execution_time,
            "conclusion": chain.conclusion
        }


# Global reasoning engine instance
_reasoning_engine: Optional[ReasoningEngine] = None


def get_reasoning_engine() -> ReasoningEngine:
    """Get or create global reasoning engine instance."""
    global _reasoning_engine
    if _reasoning_engine is None:
        _reasoning_engine = ReasoningEngine()
    return _reasoning_engine


def create_analysis_reasoning_chain(
    query: str,
    data: Dict[str, Any],
    analysis_type: str = "general"
) -> ReasoningChain:
    """Create a reasoning chain for data analysis."""
    builder = ReasoningChainBuilder()

    chain = builder.create_chain(
        name=f"Analysis: {query[:50]}",
        goal=f"Analyze data to answer: {query}",
        description=f"Multi-step reasoning for {analysis_type} analysis"
    )

    # Step 1: Observe input data
    obs_step = builder.add_observation_step(
        description="Observe and catalog input data",
        input_data=data
    )

    # Step 2: Form initial hypothesis
    hyp_step = builder.add_hypothesis_step(
        statement=f"The data can answer the query: {query}",
        based_on=[obs_step.id]
    )
    hypothesis_id = chain.hypotheses[-1].id if chain.hypotheses else ""

    # Step 3: Inference step
    inf_step = builder.add_inference_step(
        description="Derive insights from observed patterns",
        inference_logic=f"Apply {analysis_type} analysis to observed data patterns",
        dependencies=[obs_step.id, hyp_step.id]
    )

    # Step 4: Verify hypothesis
    ver_step = builder.add_verification_step(
        hypothesis_id=hypothesis_id,
        verification_method="evidence_based",
        dependencies=[inf_step.id]
    )

    # Step 5: Draw conclusion
    builder.add_conclusion_step(
        description=f"Conclusion for query: {query}",
        dependencies=[ver_step.id]
    )

    return chain
