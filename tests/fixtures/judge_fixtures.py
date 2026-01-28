"""
Judge-specific fixtures for M07 tests.

Provides factory functions for:
- Traces with specific evaluation variables
- Edge case scenarios (conflicts, insufficient evidence, etc.)
"""

from typing import Any, Optional

from core.por.reasoning_trace import ReasoningTrace, ReasoningStep

from .common import make_reasoning_trace, make_reasoning_step


# =============================================================================
# Evaluation Variables Helper
# =============================================================================

def make_trace_with_eval_vars(
    trace_id: str = "trace_test_001",
    event_observed: Optional[bool] = None,
    numeric_value: Optional[float] = None,
    conflict_detected: bool = False,
    insufficient_evidence: bool = False,
    fallback_used: bool = False,
    quorum_strength: Optional[float] = None,
    source_summary: Optional[list[str]] = None,
) -> ReasoningTrace:
    """
    Create a ReasoningTrace with specific evaluation variables.
    
    This is the primary factory for Judge tests, allowing fine-grained
    control over the evaluation variables that drive verdict computation.
    
    Args:
        trace_id: Unique trace identifier.
        event_observed: Whether the event was observed (True/False/None).
        numeric_value: Extracted numeric value (e.g., price).
        conflict_detected: Whether conflicting evidence was found.
        insufficient_evidence: Whether there was not enough evidence.
        fallback_used: Whether fallback sources were used.
        quorum_strength: Quorum strength if multi-source.
        source_summary: List of evidence IDs used.
    
    Returns:
        A ReasoningTrace with the specified evaluation variables.
    
    Examples:
        >>> trace = make_trace_with_eval_vars(event_observed=True)
        >>> judge.evaluate(trace)  # Should produce YES
        
        >>> trace = make_trace_with_eval_vars(conflict_detected=True)
        >>> judge.evaluate(trace)  # Should produce INVALID
    """
    eval_vars: dict[str, Any] = {
        "conflict_detected": conflict_detected,
        "insufficient_evidence": insufficient_evidence,
        "source_summary": source_summary or ["ev_001"],
    }
    
    if event_observed is not None:
        eval_vars["event_observed"] = event_observed
    if numeric_value is not None:
        eval_vars["numeric_value"] = numeric_value
    if fallback_used:
        eval_vars["fallback_used"] = fallback_used
    if quorum_strength is not None:
        eval_vars["quorum_strength"] = quorum_strength
    
    return make_reasoning_trace(
        trace_id=trace_id,
        evaluation_variables=eval_vars,
    )


# =============================================================================
# Edge Case Trace Factories
# =============================================================================

def make_conflict_trace(
    trace_id: str = "trace_conflict_001",
    source_a_value: float = 105000.0,
    source_b_value: float = 95000.0,
) -> ReasoningTrace:
    """
    Create a trace where conflicting evidence was detected.
    
    Simulates a scenario where two sources report different values.
    """
    return ReasoningTrace(
        trace_id=trace_id,
        steps=[
            make_reasoning_step(
                step_id="step_0001",
                step_type="extract",
                action="Extract price from source A",
                evidence_ids=["ev_001"],
                output={"price": source_a_value, "source": "A"},
            ),
            make_reasoning_step(
                step_id="step_0002",
                step_type="extract",
                action="Extract price from source B",
                evidence_ids=["ev_002"],
                output={"price": source_b_value, "source": "B"},
            ),
            make_reasoning_step(
                step_id="step_0003",
                step_type="check",
                action="Check for conflicts",
                prior_step_ids=["step_0001", "step_0002"],
                output={
                    "conflict_detected": True,
                    "difference": abs(source_a_value - source_b_value),
                },
            ),
            make_reasoning_step(
                step_id="step_0004",
                step_type="map",
                action="Map to evaluation variables",
                prior_step_ids=["step_0003"],
                output={
                    "evaluation_variables": {
                        "conflict_detected": True,
                        "insufficient_evidence": False,
                        "source_summary": ["ev_001", "ev_002"],
                    }
                },
            ),
        ],
        evidence_refs=["ev_001", "ev_002"],
    )


def make_insufficient_evidence_trace(
    trace_id: str = "trace_insufficient_001",
) -> ReasoningTrace:
    """
    Create a trace where insufficient evidence was available.
    
    Simulates a scenario where the collector couldn't gather enough data.
    """
    return ReasoningTrace(
        trace_id=trace_id,
        steps=[
            make_reasoning_step(
                step_id="step_0001",
                step_type="search",
                action="Search for evidence",
                output={"found": False, "reason": "No sources available"},
            ),
            make_reasoning_step(
                step_id="step_0002",
                step_type="map",
                action="Map to evaluation variables",
                prior_step_ids=["step_0001"],
                output={
                    "evaluation_variables": {
                        "conflict_detected": False,
                        "insufficient_evidence": True,
                        "source_summary": [],
                    }
                },
            ),
        ],
        evidence_refs=[],
    )


def make_threshold_comparison_trace(
    trace_id: str = "trace_threshold_001",
    numeric_value: float = 105000.0,
    threshold: float = 100000.0,
) -> ReasoningTrace:
    """
    Create a trace that performs a threshold comparison.
    
    Useful for testing numeric comparisons in prediction markets.
    """
    above_threshold = numeric_value > threshold
    
    return ReasoningTrace(
        trace_id=trace_id,
        steps=[
            make_reasoning_step(
                step_id="step_0001",
                step_type="extract",
                action="Extract numeric value",
                evidence_ids=["ev_001"],
                output={"value": numeric_value},
            ),
            make_reasoning_step(
                step_id="step_0002",
                step_type="check",
                action=f"Compare {numeric_value} > {threshold}",
                prior_step_ids=["step_0001"],
                output={
                    "threshold": threshold,
                    "value": numeric_value,
                    "above_threshold": above_threshold,
                },
            ),
            make_reasoning_step(
                step_id="step_0003",
                step_type="map",
                action="Map to evaluation variables",
                prior_step_ids=["step_0002"],
                output={
                    "evaluation_variables": {
                        "event_observed": above_threshold,
                        "numeric_value": numeric_value,
                        "conflict_detected": False,
                        "insufficient_evidence": False,
                        "source_summary": ["ev_001"],
                    }
                },
            ),
        ],
        evidence_refs=["ev_001"],
    )


def make_quorum_trace(
    trace_id: str = "trace_quorum_001",
    num_sources: int = 3,
    agreeing_sources: int = 2,
    value: float = 105000.0,
) -> ReasoningTrace:
    """
    Create a trace with quorum-based evidence aggregation.
    
    Simulates multi-source verification with quorum strength.
    """
    evidence_ids = [f"ev_{i:03d}" for i in range(num_sources)]
    quorum_strength = agreeing_sources / num_sources
    
    extract_steps = [
        make_reasoning_step(
            step_id=f"step_{i+1:04d}",
            step_type="extract",
            action=f"Extract from source {i}",
            evidence_ids=[evidence_ids[i]],
            output={"value": value if i < agreeing_sources else value * 0.9},
            prior_step_ids=[f"step_{i:04d}"] if i > 0 else [],
        )
        for i in range(num_sources)
    ]
    
    aggregate_step = make_reasoning_step(
        step_id=f"step_{num_sources+1:04d}",
        step_type="aggregate",
        action="Aggregate sources with quorum",
        prior_step_ids=[f"step_{i+1:04d}" for i in range(num_sources)],
        output={
            "quorum_strength": quorum_strength,
            "agreeing_sources": agreeing_sources,
            "total_sources": num_sources,
        },
    )
    
    map_step = make_reasoning_step(
        step_id=f"step_{num_sources+2:04d}",
        step_type="map",
        action="Map to evaluation variables",
        prior_step_ids=[f"step_{num_sources+1:04d}"],
        output={
            "evaluation_variables": {
                "event_observed": True,
                "numeric_value": value,
                "conflict_detected": agreeing_sources < num_sources,
                "insufficient_evidence": False,
                "quorum_strength": quorum_strength,
                "source_summary": evidence_ids,
            }
        },
    )
    
    return ReasoningTrace(
        trace_id=trace_id,
        steps=extract_steps + [aggregate_step, map_step],
        evidence_refs=evidence_ids,
    )


def make_fallback_trace(
    trace_id: str = "trace_fallback_001",
    primary_failed: bool = True,
    fallback_value: float = 105000.0,
) -> ReasoningTrace:
    """
    Create a trace where fallback sources were used.
    
    Simulates primary source failure with fallback recovery.
    """
    return ReasoningTrace(
        trace_id=trace_id,
        steps=[
            make_reasoning_step(
                step_id="step_0001",
                step_type="search",
                action="Try primary source",
                evidence_ids=[],
                output={"success": not primary_failed, "error": "timeout" if primary_failed else None},
            ),
            make_reasoning_step(
                step_id="step_0002",
                step_type="search",
                action="Try fallback source",
                evidence_ids=["ev_fallback"],
                prior_step_ids=["step_0001"],
                output={"success": True, "fallback_used": True},
            ),
            make_reasoning_step(
                step_id="step_0003",
                step_type="extract",
                action="Extract from fallback",
                evidence_ids=["ev_fallback"],
                prior_step_ids=["step_0002"],
                output={"value": fallback_value},
            ),
            make_reasoning_step(
                step_id="step_0004",
                step_type="map",
                action="Map to evaluation variables",
                prior_step_ids=["step_0003"],
                output={
                    "evaluation_variables": {
                        "event_observed": True,
                        "numeric_value": fallback_value,
                        "conflict_detected": False,
                        "insufficient_evidence": False,
                        "fallback_used": True,
                        "source_summary": ["ev_fallback"],
                    }
                },
            ),
        ],
        evidence_refs=["ev_fallback"],
    )