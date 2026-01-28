"""
Module 07 - Judge Mapping Tests

Tests for deterministic mapping engine:
1. event_observed=True → YES, confidence >= min threshold
2. event_observed=False → NO
3. event_observed=None or insufficient_evidence=True → INVALID with low confidence
4. conflict_detected=True reduces confidence
5. Missing confidence policy uses defaults deterministically
"""
import pytest
from datetime import datetime, timezone

from agents.judge import (
    JudgeAgent,
    map_to_verdict,
    extract_evaluation_variables,
    check_validity,
    check_conflicts,
    determine_binary_outcome,
    compute_confidence,
    EvaluationVariables,
    ConfidencePolicy,
)
from core.schemas.verdict import Outcome

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fixtures import (
    make_prompt_spec,
    make_evidence_bundle,
    make_reasoning_trace,
    make_trace_with_eval_vars,
)


class TestEvaluationVariablesExtraction:
    """Tests for extracting evaluation variables from trace."""
    
    def test_extract_from_map_step(self):
        """Should extract evaluation_variables from final map step."""
        trace = make_trace_with_eval_vars(event_observed=True)
        eval_vars, check = extract_evaluation_variables(trace)
        
        assert eval_vars is not None
        assert eval_vars.event_observed is True
        assert check.ok is True
    
    def test_extract_from_aggregate_step(self):
        """Should extract from aggregate step if no map step."""
        from core.por.reasoning_trace import ReasoningStep
        
        trace = make_reasoning_trace(steps=[
            ReasoningStep(
                step_id="step_0001",
                type="aggregate",
                action="Aggregate results",
                output={"evaluation_variables": {"event_observed": False}},
            ),
        ])
        
        eval_vars, check = extract_evaluation_variables(trace)
        assert eval_vars is not None
        assert eval_vars.event_observed is False
        assert check.ok is True
    
    def test_missing_evaluation_variables_fails(self):
        """Should fail if no evaluation_variables in output."""
        from core.por.reasoning_trace import ReasoningStep
        
        trace = make_reasoning_trace(steps=[
            ReasoningStep(
                step_id="step_0001",
                type="map",
                action="Map without eval vars",
                output={"some_other_key": "value"},
            ),
        ])
        
        eval_vars, check = extract_evaluation_variables(trace)
        assert eval_vars is None
        assert check.ok is False
        assert "evaluation_variables" in check.message.lower()
    
    def test_empty_trace_fails(self):
        """Should fail for empty trace."""
        from core.por.reasoning_trace import ReasoningTrace
        
        trace = ReasoningTrace(trace_id="empty", steps=[])
        eval_vars, check = extract_evaluation_variables(trace)
        
        assert eval_vars is None
        assert check.ok is False


class TestBinaryOutcomeDecision:
    """Tests for YES/NO/INVALID decision logic."""
    
    def test_event_observed_true_yields_yes(self):
        """event_observed=True should yield YES."""
        trace = make_trace_with_eval_vars(event_observed=True)
        prompt_spec = make_prompt_spec()
        evidence = make_evidence_bundle()
        
        verdict, checks = map_to_verdict(prompt_spec, evidence, trace)
        
        assert verdict.outcome == "YES"
        assert verdict.confidence >= 0.55  # min_confidence_for_yesno default
    
    def test_event_observed_false_yields_no(self):
        """event_observed=False should yield NO."""
        trace = make_trace_with_eval_vars(event_observed=False)
        prompt_spec = make_prompt_spec()
        evidence = make_evidence_bundle()
        
        verdict, checks = map_to_verdict(prompt_spec, evidence, trace)
        
        assert verdict.outcome == "NO"
        assert verdict.confidence >= 0.55
    
    def test_event_observed_none_with_threshold_comparison(self):
        """Should use numeric_value and threshold when event_observed is None."""
        trace = make_trace_with_eval_vars(
            event_observed=None,
            numeric_value=105000.0,
        )
        prompt_spec = make_prompt_spec(threshold="> 100000")
        evidence = make_evidence_bundle()
        
        verdict, checks = map_to_verdict(prompt_spec, evidence, trace)
        
        assert verdict.outcome == "YES"  # 105000 > 100000
    
    def test_threshold_less_than(self):
        """Should handle < threshold correctly."""
        trace = make_trace_with_eval_vars(
            event_observed=None,
            numeric_value=50.0,
        )
        prompt_spec = make_prompt_spec(threshold="< 100")
        evidence = make_evidence_bundle()
        
        verdict, checks = map_to_verdict(prompt_spec, evidence, trace)
        
        assert verdict.outcome == "YES"  # 50 < 100
    
    def test_threshold_greater_than_or_equal(self):
        """Should handle >= threshold correctly."""
        trace = make_trace_with_eval_vars(
            event_observed=None,
            numeric_value=100.0,
        )
        prompt_spec = make_prompt_spec(threshold=">= 100")
        evidence = make_evidence_bundle()
        
        verdict, checks = map_to_verdict(prompt_spec, evidence, trace)
        
        assert verdict.outcome == "YES"  # 100 >= 100
    
    def test_insufficient_evidence_yields_invalid(self):
        """insufficient_evidence=True should yield INVALID."""
        trace = make_trace_with_eval_vars(
            event_observed=True,  # Even with this, should be INVALID
            insufficient_evidence=True,
        )
        prompt_spec = make_prompt_spec()
        evidence = make_evidence_bundle()
        
        verdict, checks = map_to_verdict(prompt_spec, evidence, trace)
        
        assert verdict.outcome == "INVALID"
        assert verdict.confidence <= 0.3  # Low confidence for invalid
        assert verdict.resolution_rule_id == "R_VALIDITY"
    
    def test_no_observation_or_numeric_yields_invalid(self):
        """Missing both event_observed and numeric_value yields INVALID."""
        trace = make_trace_with_eval_vars(
            event_observed=None,
            numeric_value=None,
        )
        prompt_spec = make_prompt_spec(threshold=None)  # No threshold either
        evidence = make_evidence_bundle()
        
        verdict, checks = map_to_verdict(prompt_spec, evidence, trace)
        
        assert verdict.outcome == "INVALID"


class TestConfidenceComputation:
    """Tests for confidence score computation."""
    
    def test_default_confidence_used_when_no_policy(self):
        """Should use default confidence (0.7) when no policy specified."""
        trace = make_trace_with_eval_vars(event_observed=True)
        prompt_spec = make_prompt_spec(confidence_policy=None)
        evidence = make_evidence_bundle()
        
        verdict, checks = map_to_verdict(prompt_spec, evidence, trace)
        
        assert verdict.confidence == 0.7  # default
    
    def test_custom_confidence_policy(self):
        """Should use custom confidence policy when specified."""
        trace = make_trace_with_eval_vars(event_observed=True)
        prompt_spec = make_prompt_spec(
            confidence_policy={"default_confidence": 0.9}
        )
        evidence = make_evidence_bundle()
        
        verdict, checks = map_to_verdict(prompt_spec, evidence, trace)
        
        assert verdict.confidence == 0.9
    
    def test_conflict_detected_reduces_confidence(self):
        """conflict_detected=True should reduce confidence."""
        trace = make_trace_with_eval_vars(
            event_observed=True,
            conflict_detected=True,
        )
        prompt_spec = make_prompt_spec(
            confidence_policy={"default_confidence": 1.0, "conflict_reduction": 0.8}
        )
        evidence = make_evidence_bundle()
        
        verdict, checks = map_to_verdict(prompt_spec, evidence, trace)
        
        # 1.0 * 0.8 = 0.8
        assert verdict.confidence == 0.8
    
    def test_fallback_used_reduces_confidence(self):
        """fallback_used=True should reduce confidence."""
        trace = make_trace_with_eval_vars(
            event_observed=True,
            fallback_used=True,
        )
        prompt_spec = make_prompt_spec(
            confidence_policy={"default_confidence": 1.0, "fallback_reduction": 0.9}
        )
        evidence = make_evidence_bundle()
        
        verdict, checks = map_to_verdict(prompt_spec, evidence, trace)
        
        # 1.0 * 0.9 = 0.9
        assert verdict.confidence == 0.9
    
    def test_multiple_reductions_stack(self):
        """Multiple reduction factors should stack multiplicatively."""
        trace = make_trace_with_eval_vars(
            event_observed=True,
            conflict_detected=True,
            fallback_used=True,
        )
        prompt_spec = make_prompt_spec(
            confidence_policy={
                "default_confidence": 1.0,
                "conflict_reduction": 0.8,
                "fallback_reduction": 0.9,
            }
        )
        evidence = make_evidence_bundle()
        
        verdict, checks = map_to_verdict(prompt_spec, evidence, trace)
        
        # 1.0 * 0.8 * 0.9 = 0.72
        assert abs(verdict.confidence - 0.72) < 0.001
    
    def test_quorum_strength_affects_confidence(self):
        """quorum_strength should affect confidence."""
        trace = make_trace_with_eval_vars(
            event_observed=True,
            quorum_strength=0.8,
        )
        prompt_spec = make_prompt_spec(
            confidence_policy={"default_confidence": 1.0}
        )
        evidence = make_evidence_bundle()
        
        verdict, checks = map_to_verdict(prompt_spec, evidence, trace)
        
        # 1.0 * max(0.5, 0.8) = 0.8
        assert verdict.confidence == 0.8
    
    def test_invalid_outcome_low_confidence(self):
        """INVALID outcome should have low confidence."""
        trace = make_trace_with_eval_vars(insufficient_evidence=True)
        prompt_spec = make_prompt_spec(
            confidence_policy={"insufficient_confidence": 0.0}
        )
        evidence = make_evidence_bundle()
        
        verdict, checks = map_to_verdict(prompt_spec, evidence, trace)
        
        assert verdict.outcome == "INVALID"
        assert verdict.confidence == 0.0


class TestVerdictMetadata:
    """Tests for verdict metadata and references."""
    
    def test_market_id_matches(self):
        """Verdict market_id should match PromptSpec market_id."""
        prompt_spec = make_prompt_spec(market_id="my_special_market")
        evidence = make_evidence_bundle()
        trace = make_trace_with_eval_vars(event_observed=True)
        
        verdict, checks = map_to_verdict(prompt_spec, evidence, trace)
        
        assert verdict.market_id == "my_special_market"
    
    def test_resolution_rule_id_set_correctly(self):
        """resolution_rule_id should indicate which rule decided."""
        # Test R_BINARY_DECISION
        trace = make_trace_with_eval_vars(event_observed=True)
        prompt_spec = make_prompt_spec()
        evidence = make_evidence_bundle()
        
        verdict, _ = map_to_verdict(prompt_spec, evidence, trace)
        assert verdict.resolution_rule_id == "R_BINARY_DECISION"
        
        # Test R_VALIDITY
        trace_invalid = make_trace_with_eval_vars(insufficient_evidence=True)
        verdict_invalid, _ = map_to_verdict(prompt_spec, evidence, trace_invalid)
        assert verdict_invalid.resolution_rule_id == "R_VALIDITY"
    
    def test_selected_leaf_refs_from_source_summary(self):
        """selected_leaf_refs should come from source_summary."""
        trace = make_trace_with_eval_vars(
            event_observed=True,
            source_summary=["ev_001", "ev_002", "ev_003"],
        )
        prompt_spec = make_prompt_spec()
        evidence = make_evidence_bundle()
        
        verdict, checks = map_to_verdict(prompt_spec, evidence, trace)
        
        assert verdict.selected_leaf_refs == ["ev_001", "ev_002", "ev_003"]


class TestCheckResults:
    """Tests for structured check results."""
    
    def test_checks_returned_for_all_stages(self):
        """Should return CheckResults for each verification stage."""
        trace = make_trace_with_eval_vars(event_observed=True)
        prompt_spec = make_prompt_spec()
        evidence = make_evidence_bundle()
        
        verdict, checks = map_to_verdict(prompt_spec, evidence, trace)
        
        check_ids = [c.check_id for c in checks]
        
        assert "extract_evaluation_variables" in check_ids
        assert "validity" in check_ids
        assert "conflict" in check_ids
        assert "binary_decision" in check_ids
        assert "confidence" in check_ids
    
    def test_failed_check_has_error_severity(self):
        """Failed checks should have error severity."""
        trace = make_trace_with_eval_vars(insufficient_evidence=True)
        prompt_spec = make_prompt_spec()
        evidence = make_evidence_bundle()
        
        verdict, checks = map_to_verdict(prompt_spec, evidence, trace)
        
        validity_check = next(c for c in checks if c.check_id == "validity")
        assert validity_check.ok is False
        assert validity_check.severity == "error"
    
    def test_warning_check_for_conflict(self):
        """Conflict with quorum should produce warning, not error."""
        trace = make_trace_with_eval_vars(
            event_observed=True,
            conflict_detected=True,
            quorum_strength=0.7,
        )
        prompt_spec = make_prompt_spec()
        evidence = make_evidence_bundle()
        
        verdict, checks = map_to_verdict(prompt_spec, evidence, trace)
        
        conflict_check = next(c for c in checks if c.check_id == "conflict")
        assert conflict_check.ok is True  # Warning, not failure
        assert conflict_check.severity == "warn"


class TestJudgeAgentIntegration:
    """Integration tests for JudgeAgent."""
    
    def test_judge_produces_valid_verdict(self):
        """JudgeAgent should produce valid verdict."""
        judge = JudgeAgent()
        prompt_spec = make_prompt_spec()
        evidence = make_evidence_bundle()
        trace = make_trace_with_eval_vars(event_observed=True)
        
        verdict, result = judge.judge(prompt_spec, evidence, trace)
        
        assert verdict.outcome == "YES"
        assert verdict.market_id == "test_market_001"
        assert result.ok is True
    
    def test_judge_computes_commitments(self):
        """JudgeAgent should compute hash commitments."""
        judge = JudgeAgent(compute_commitments=True)
        prompt_spec = make_prompt_spec()
        evidence = make_evidence_bundle()
        trace = make_trace_with_eval_vars(event_observed=True)
        
        verdict, result = judge.judge(prompt_spec, evidence, trace)
        
        assert verdict.prompt_spec_hash is not None
        assert verdict.prompt_spec_hash.startswith("0x")
        assert verdict.evidence_root is not None
        assert verdict.reasoning_root is not None
    
    def test_deterministic_across_runs(self):
        """Same inputs should produce same verdict."""
        judge = JudgeAgent()
        prompt_spec = make_prompt_spec()
        evidence = make_evidence_bundle()
        trace = make_trace_with_eval_vars(event_observed=True, numeric_value=105000.0)
        
        # Fixed resolution time for determinism
        fixed_time = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        
        verdict1, _ = judge.judge(prompt_spec, evidence, trace, resolution_time=fixed_time)
        verdict2, _ = judge.judge(prompt_spec, evidence, trace, resolution_time=fixed_time)
        
        assert verdict1.outcome == verdict2.outcome
        assert verdict1.confidence == verdict2.confidence
        assert verdict1.resolution_rule_id == verdict2.resolution_rule_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])