"""
Module 07 - Deterministic Mapping Engine
Maps (PromptSpec, EvidenceBundle, ReasoningTrace) to verdict deterministically.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from core.por.reasoning_trace import ReasoningTrace, ReasoningStep
from core.schemas.evidence import EvidenceBundle
from core.schemas.prompts import PromptSpec
from core.schemas.verdict import DeterministicVerdict, Outcome
from core.schemas.verification import CheckResult


@dataclass(frozen=True)
class EvaluationVariables:
    """Structured evaluation variables extracted from the trace."""
    event_observed: Optional[bool]
    numeric_value: Optional[float]
    timestamp: Optional[str]
    conflict_detected: bool
    insufficient_evidence: bool
    source_summary: list[str]
    fallback_used: bool = False
    quorum_strength: Optional[float] = None
    raw: dict[str, Any] = None

    def __post_init__(self):
        if self.raw is None:
            object.__setattr__(self, 'raw', {})


@dataclass(frozen=True)
class ConfidencePolicy:
    """Policy for computing verdict confidence."""
    default_confidence: float = 0.7
    min_confidence_for_yesno: float = 0.55
    conflict_reduction: float = 0.8
    fallback_reduction: float = 0.9
    insufficient_confidence: float = 0.0
    invalid_confidence: float = 0.3

    @classmethod
    def from_prompt_spec(cls, prompt_spec: PromptSpec) -> "ConfidencePolicy":
        pd = (prompt_spec.extra or {}).get("confidence_policy", {})
        return cls(
            default_confidence=pd.get("default_confidence", 0.7),
            min_confidence_for_yesno=pd.get("min_confidence_for_yesno", 0.55),
            conflict_reduction=pd.get("conflict_reduction", 0.8),
            fallback_reduction=pd.get("fallback_reduction", 0.9),
            insufficient_confidence=pd.get("insufficient_confidence", 0.0),
            invalid_confidence=pd.get("invalid_confidence", 0.3),
        )


def _find_final_evaluation_step(trace: ReasoningTrace) -> Optional[ReasoningStep]:
    """Find the final step with evaluation_variables (map > aggregate > any)."""
    if not trace.steps:
        return None
    for step in reversed(trace.steps):
        if step.type == "map":
            return step
    for step in reversed(trace.steps):
        if step.type == "aggregate":
            return step
    for step in reversed(trace.steps):
        if "evaluation_variables" in step.output:
            return step
    return trace.steps[-1] if trace.steps else None


def extract_evaluation_variables(trace: ReasoningTrace) -> tuple[Optional[EvaluationVariables], CheckResult]:
    """Extract evaluation_variables from the trace's final step."""
    final_step = _find_final_evaluation_step(trace)
    
    if final_step is None:
        return None, CheckResult.failed(
            check_id="extract_evaluation_variables",
            message="No steps found in trace",
            details={"trace_id": trace.trace_id},
        )
    
    ev = final_step.output.get("evaluation_variables", {})
    if not ev:
        return None, CheckResult.failed(
            check_id="extract_evaluation_variables",
            message=f"No evaluation_variables in step '{final_step.step_id}'",
            details={"step_id": final_step.step_id, "step_type": final_step.type},
        )
    
    variables = EvaluationVariables(
        event_observed=ev.get("event_observed"),
        numeric_value=ev.get("numeric_value"),
        timestamp=ev.get("timestamp"),
        conflict_detected=ev.get("conflict_detected", False),
        insufficient_evidence=ev.get("insufficient_evidence", False),
        source_summary=ev.get("source_summary", []),
        fallback_used=ev.get("fallback_used", False),
        quorum_strength=ev.get("quorum_strength"),
        raw=ev,
    )
    
    return variables, CheckResult.passed(
        check_id="extract_evaluation_variables",
        message="Extracted evaluation_variables",
        details={"step_id": final_step.step_id, "has_event": variables.event_observed is not None},
    )


def check_validity(eval_vars: EvaluationVariables, prompt_spec: PromptSpec) -> tuple[bool, CheckResult]:
    """Check if evaluation variables are valid for verdict determination."""
    if eval_vars.insufficient_evidence:
        return False, CheckResult.failed(
            check_id="validity", message="Insufficient evidence flagged",
            details={"insufficient_evidence": True},
        )
    
    has_obs = eval_vars.event_observed is not None
    has_num = eval_vars.numeric_value is not None
    threshold = prompt_spec.prediction_semantics.threshold
    
    # If we have event_observed, that's sufficient for binary decision
    if has_obs:
        return True, CheckResult.passed(
            check_id="validity", message="Evaluation variables valid (event_observed)",
            details={"has_event": has_obs, "has_numeric": has_num},
        )
    
    # If no event_observed, we need numeric_value for threshold comparison
    if threshold is not None and not has_num:
        return False, CheckResult.failed(
            check_id="validity", message="Threshold defined but no numeric_value",
            details={"threshold": threshold},
        )
    
    if not has_obs and not has_num:
        return False, CheckResult.failed(
            check_id="validity", message="No event_observed or numeric_value",
            details={"has_event": has_obs, "has_numeric": has_num},
        )
    
    return True, CheckResult.passed(
        check_id="validity", message="Evaluation variables valid",
        details={"has_event": has_obs, "has_numeric": has_num},
    )


def check_conflicts(eval_vars: EvaluationVariables, prompt_spec: PromptSpec) -> tuple[bool, CheckResult]:
    """Check for conflicts and determine if they prevent verdict."""
    if not eval_vars.conflict_detected:
        return True, CheckResult.passed(
            check_id="conflict", message="No conflicts",
            details={"conflict_detected": False},
        )
    
    # Quorum achieved resolves conflict
    if eval_vars.quorum_strength is not None and eval_vars.quorum_strength >= 0.5:
        return True, CheckResult.warning(
            check_id="conflict", message="Conflict but quorum achieved",
            details={"quorum_strength": eval_vars.quorum_strength},
        )
    
    # Check policy
    conflict_policy = (prompt_spec.extra or {}).get("conflict_policy", {})
    if not conflict_policy.get("allow_verdict_with_conflict", True):
        return False, CheckResult.failed(
            check_id="conflict", message="Conflict forbids verdict",
            details={"allow_verdict_with_conflict": False},
        )
    
    return True, CheckResult.warning(
        check_id="conflict", message="Conflict - reduced confidence",
        details={"quorum_strength": eval_vars.quorum_strength},
    )


def _parse_threshold(threshold_str: str, value: float) -> tuple[bool, float]:
    """Parse threshold string and compare. Returns (result, threshold_value)."""
    s = threshold_str.strip()
    if s.startswith(">="):
        t = float(s[2:].strip())
        return value >= t, t
    elif s.startswith("<="):
        t = float(s[2:].strip())
        return value <= t, t
    elif s.startswith(">"):
        t = float(s[1:].strip())
        return value > t, t
    elif s.startswith("<"):
        t = float(s[1:].strip())
        return value < t, t
    elif s.startswith("==") or s.startswith("="):
        t = float(s.lstrip("=").strip())
        return abs(value - t) < 1e-9, t
    else:
        t = float(s)
        return value > t, t


def determine_binary_outcome(eval_vars: EvaluationVariables, prompt_spec: PromptSpec) -> tuple[Outcome, CheckResult]:
    """Determine YES/NO/INVALID from evaluation variables."""
    # Check event_observed first
    if eval_vars.event_observed is True:
        return "YES", CheckResult.passed(
            check_id="binary_decision", message="Event observed: YES",
            details={"event_observed": True},
        )
    elif eval_vars.event_observed is False:
        return "NO", CheckResult.passed(
            check_id="binary_decision", message="Event not observed: NO",
            details={"event_observed": False},
        )
    
    # Numeric threshold comparison
    threshold_str = prompt_spec.prediction_semantics.threshold
    if threshold_str is not None and eval_vars.numeric_value is not None:
        try:
            result, threshold = _parse_threshold(threshold_str, eval_vars.numeric_value)
            outcome: Outcome = "YES" if result else "NO"
            return outcome, CheckResult.passed(
                check_id="binary_decision",
                message=f"Numeric: {eval_vars.numeric_value} vs {threshold_str} -> {outcome}",
                details={"numeric_value": eval_vars.numeric_value, "threshold": threshold_str},
            )
        except (ValueError, TypeError) as e:
            return "INVALID", CheckResult.failed(
                check_id="binary_decision", message=f"Threshold parse error: {e}",
                details={"threshold": threshold_str, "error": str(e)},
            )
    
    return "INVALID", CheckResult.failed(
        check_id="binary_decision", message="Cannot determine outcome",
        details={"event_observed": eval_vars.event_observed, "numeric_value": eval_vars.numeric_value},
    )


def compute_confidence(outcome: Outcome, eval_vars: EvaluationVariables, policy: ConfidencePolicy) -> tuple[float, CheckResult]:
    """Compute confidence score based on outcome and evaluation variables."""
    if outcome == "INVALID":
        conf = policy.insufficient_confidence if eval_vars.insufficient_evidence else policy.invalid_confidence
        return conf, CheckResult.passed(
            check_id="confidence", message=f"INVALID confidence: {conf}",
            details={"outcome": outcome, "confidence": conf},
        )
    
    conf = policy.default_confidence
    if eval_vars.conflict_detected:
        conf *= policy.conflict_reduction
    if eval_vars.fallback_used:
        conf *= policy.fallback_reduction
    if eval_vars.quorum_strength is not None:
        conf *= max(0.5, eval_vars.quorum_strength)
    
    conf = max(0.0, min(1.0, conf))
    
    if conf < policy.min_confidence_for_yesno:
        return conf, CheckResult.warning(
            check_id="confidence", message=f"Below min: {conf}",
            details={"confidence": conf, "min": policy.min_confidence_for_yesno},
        )
    
    return conf, CheckResult.passed(
        check_id="confidence", message=f"Confidence: {conf}",
        details={"confidence": conf, "conflict": eval_vars.conflict_detected},
    )


def _determine_rule_id(validity_ok: bool, conflict_ok: bool, outcome: Outcome) -> str:
    """Determine which rule decided the verdict."""
    if not validity_ok:
        return "R_VALIDITY"
    if not conflict_ok:
        return "R_CONFLICT"
    if outcome == "INVALID":
        return "R_INVALID_FALLBACK"
    return "R_BINARY_DECISION"


def map_to_verdict(
    prompt_spec: PromptSpec,
    evidence: EvidenceBundle,
    trace: ReasoningTrace,
    *,
    resolution_time: Optional[datetime] = None,
) -> tuple[DeterministicVerdict, list[CheckResult]]:
    """
    Map inputs to a deterministic verdict (main entry point).
    
    Args:
        prompt_spec: The prompt specification
        evidence: The evidence bundle
        trace: The reasoning trace from the auditor
        resolution_time: Optional timestamp (None for deterministic hashing)
    
    Returns:
        Tuple of (DeterministicVerdict, list[CheckResult])
    """
    checks: list[CheckResult] = []
    res_time = resolution_time or datetime.now(timezone.utc)
    market_id = prompt_spec.market.market_id
    
    # Step 1: Extract evaluation variables
    eval_vars, extract_check = extract_evaluation_variables(trace)
    checks.append(extract_check)
    
    if eval_vars is None:
        return DeterministicVerdict(
            market_id=market_id, outcome="INVALID", confidence=0.0,
            resolution_time=res_time, resolution_rule_id="R_VALIDITY",
        ), checks
    
    # Step 2: Validity
    validity_ok, validity_check = check_validity(eval_vars, prompt_spec)
    checks.append(validity_check)
    
    if not validity_ok:
        policy = ConfidencePolicy.from_prompt_spec(prompt_spec)
        # Use insufficient_confidence if that was the reason
        conf = policy.insufficient_confidence if eval_vars.insufficient_evidence else policy.invalid_confidence
        return DeterministicVerdict(
            market_id=market_id, outcome="INVALID", confidence=conf,
            resolution_time=res_time, resolution_rule_id="R_VALIDITY",
            selected_leaf_refs=eval_vars.source_summary,
        ), checks
    
    # Step 3: Conflicts
    conflict_ok, conflict_check = check_conflicts(eval_vars, prompt_spec)
    checks.append(conflict_check)
    
    if not conflict_ok:
        policy = ConfidencePolicy.from_prompt_spec(prompt_spec)
        return DeterministicVerdict(
            market_id=market_id, outcome="INVALID", confidence=policy.invalid_confidence,
            resolution_time=res_time, resolution_rule_id="R_CONFLICT",
            selected_leaf_refs=eval_vars.source_summary,
        ), checks
    
    # Step 4: Binary decision
    outcome, decision_check = determine_binary_outcome(eval_vars, prompt_spec)
    checks.append(decision_check)
    
    # Step 5: Confidence
    policy = ConfidencePolicy.from_prompt_spec(prompt_spec)
    confidence, confidence_check = compute_confidence(outcome, eval_vars, policy)
    checks.append(confidence_check)
    
    rule_id = _determine_rule_id(validity_ok, conflict_ok, outcome)
    
    return DeterministicVerdict(
        market_id=market_id, outcome=outcome, confidence=confidence,
        resolution_time=res_time, resolution_rule_id=rule_id,
        selected_leaf_refs=eval_vars.source_summary,
    ), checks


__all__ = [
    "EvaluationVariables", "ConfidencePolicy", "extract_evaluation_variables",
    "check_validity", "check_conflicts", "determine_binary_outcome",
    "compute_confidence", "map_to_verdict",
]