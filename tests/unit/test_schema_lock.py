"""
Module 07 - Schema Lock Tests

Tests for schema lock validation:
- Wrong output_schema_ref fails
- Missing strict_mode fails (if strict pipeline enabled)
- Missing evaluation_variables fails
"""
import pytest

from agents.judge import (
    JudgeAgent,
    SchemaLock,
    EXPECTED_OUTPUT_SCHEMA_REF,
    VALID_OUTCOMES,
)
from core.por.reasoning_trace import ReasoningStep

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fixtures import (
    make_prompt_spec,
    make_evidence_bundle,
    make_reasoning_trace,
    make_trace_with_eval_vars,
)


class TestSchemaLockPromptSpec:
    """Tests for PromptSpec schema lock validation."""
    
    def test_valid_prompt_spec_passes(self):
        """Valid PromptSpec should pass all checks."""
        prompt_spec = make_prompt_spec(
            strict_mode=True,
            output_schema_ref="core.schemas.verdict.DeterministicVerdict",
        )
        
        checks = SchemaLock.verify_prompt_spec(prompt_spec, require_strict_mode=True)
        
        failed_checks = [c for c in checks if not c.ok]
        assert len(failed_checks) == 0, f"Failed checks: {[c.message for c in failed_checks]}"
    
    def test_wrong_output_schema_ref_fails(self):
        """Wrong output_schema_ref should fail validation."""
        prompt_spec = make_prompt_spec(
            output_schema_ref="some.wrong.Schema",
        )
        
        checks = SchemaLock.verify_prompt_spec(prompt_spec, require_strict_mode=False)
        
        schema_check = next(c for c in checks if "output_schema_ref" in c.check_id)
        assert schema_check.ok is False
        assert EXPECTED_OUTPUT_SCHEMA_REF in schema_check.details.get("expected", "")
    
    def test_missing_strict_mode_fails(self):
        """Missing strict_mode should fail when required."""
        prompt_spec = make_prompt_spec(strict_mode=False)
        
        checks = SchemaLock.verify_prompt_spec(prompt_spec, require_strict_mode=True)
        
        strict_check = next(c for c in checks if "strict_mode" in c.check_id)
        assert strict_check.ok is False
    
    def test_missing_strict_mode_ok_when_not_required(self):
        """Missing strict_mode should pass when not required."""
        prompt_spec = make_prompt_spec(strict_mode=False)
        
        checks = SchemaLock.verify_prompt_spec(prompt_spec, require_strict_mode=False)
        
        # Should not have strict_mode check when not required
        strict_checks = [c for c in checks if "strict_mode" in c.check_id]
        assert len(strict_checks) == 0
    
    def test_market_spec_present(self):
        """MarketSpec should be validated as present."""
        prompt_spec = make_prompt_spec()
        
        checks = SchemaLock.verify_prompt_spec(prompt_spec)
        
        market_check = next(c for c in checks if "market_spec" in c.check_id)
        assert market_check.ok is True
    
    def test_prediction_semantics_present(self):
        """PredictionSemantics should be validated as present."""
        prompt_spec = make_prompt_spec()
        
        checks = SchemaLock.verify_prompt_spec(prompt_spec)
        
        semantics_check = next(c for c in checks if "prediction_semantics" in c.check_id)
        assert semantics_check.ok is True


class TestSchemaLockTrace:
    """Tests for ReasoningTrace schema lock validation."""
    
    def test_valid_trace_passes(self):
        """Valid trace should pass all checks."""
        trace = make_trace_with_eval_vars(event_observed=True)
        
        checks = SchemaLock.verify_trace(trace, require_evaluation_variables=True)
        
        failed_checks = [c for c in checks if not c.ok]
        assert len(failed_checks) == 0, f"Failed checks: {[c.message for c in failed_checks]}"
    
    def test_missing_evaluation_variables_fails(self):
        """Missing evaluation_variables should fail when required."""
        trace = make_reasoning_trace(steps=[
            ReasoningStep(
                step_id="step_0001",
                type="map",
                action="Map without eval vars",
                output={"some_other_key": "value"},
            ),
        ])
        
        checks = SchemaLock.verify_trace(trace, require_evaluation_variables=True)
        
        eval_check = next(c for c in checks if "evaluation_variables" in c.check_id)
        assert eval_check.ok is False
    
    def test_empty_trace_fails(self):
        """Empty trace should fail validation."""
        from core.por.reasoning_trace import ReasoningTrace
        
        trace = ReasoningTrace(trace_id="empty", steps=[])
        
        checks = SchemaLock.verify_trace(trace)
        
        steps_check = next(c for c in checks if "trace_steps" in c.check_id)
        assert steps_check.ok is False
    
    def test_unique_step_ids(self):
        """Step IDs must be unique."""
        trace = make_trace_with_eval_vars(event_observed=True)
        
        checks = SchemaLock.verify_trace(trace)
        
        unique_check = next(c for c in checks if "unique_step_ids" in c.check_id)
        assert unique_check.ok is True
    
    def test_valid_prior_step_refs(self):
        """Prior step references must point to earlier steps."""
        trace = make_trace_with_eval_vars(event_observed=True)
        
        checks = SchemaLock.verify_trace(trace)
        
        prior_check = next(c for c in checks if "prior_step_refs" in c.check_id)
        assert prior_check.ok is True
    
    def test_final_step_type_check(self):
        """Should validate final step is map or aggregate."""
        trace = make_trace_with_eval_vars(event_observed=True)
        
        checks = SchemaLock.verify_trace(trace)
        
        final_check = next(c for c in checks if "final_step" in c.check_id)
        assert final_check.ok is True


class TestSchemaLockOutcome:
    """Tests for outcome validation."""
    
    def test_valid_outcomes(self):
        """Valid outcomes should pass."""
        for outcome in ["YES", "NO", "INVALID"]:
            check = SchemaLock.verify_outcome(outcome)
            assert check.ok is True, f"Outcome '{outcome}' should be valid"
    
    def test_invalid_outcome_fails(self):
        """Invalid outcomes should fail."""
        for invalid in ["MAYBE", "yes", "no", "True", "False", ""]:
            check = SchemaLock.verify_outcome(invalid)
            assert check.ok is False, f"Outcome '{invalid}' should be invalid"
    
    def test_valid_outcomes_constant(self):
        """VALID_OUTCOMES should contain exactly YES, NO, INVALID."""
        assert VALID_OUTCOMES == frozenset({"YES", "NO", "INVALID"})


class TestJudgeAgentSchemaLock:
    """Integration tests for JudgeAgent schema lock enforcement."""
    
    def test_judge_rejects_wrong_output_schema(self):
        """JudgeAgent should reject wrong output_schema_ref."""
        judge = JudgeAgent(require_strict_mode=False)
        prompt_spec = make_prompt_spec(
            output_schema_ref="wrong.schema.Reference",
        )
        evidence = make_evidence_bundle()
        trace = make_trace_with_eval_vars(event_observed=True)
        
        verdict, result = judge.judge(prompt_spec, evidence, trace)
        
        assert result.ok is False
        assert verdict.outcome == "INVALID"
    
    def test_judge_rejects_missing_strict_mode(self):
        """JudgeAgent should reject missing strict_mode when required."""
        judge = JudgeAgent(require_strict_mode=True)
        prompt_spec = make_prompt_spec(strict_mode=False)
        evidence = make_evidence_bundle()
        trace = make_trace_with_eval_vars(event_observed=True)
        
        verdict, result = judge.judge(prompt_spec, evidence, trace)
        
        assert result.ok is False
        assert verdict.outcome == "INVALID"
    
    def test_judge_allows_missing_strict_mode_when_not_required(self):
        """JudgeAgent should allow missing strict_mode when not required."""
        judge = JudgeAgent(require_strict_mode=False)
        prompt_spec = make_prompt_spec(strict_mode=False)
        evidence = make_evidence_bundle()
        trace = make_trace_with_eval_vars(event_observed=True)
        
        verdict, result = judge.judge(prompt_spec, evidence, trace)
        
        assert verdict.outcome == "YES"
        # Result might have warnings but should be ok if verdict is valid
    
    def test_judge_rejects_missing_evaluation_variables(self):
        """JudgeAgent should reject trace without evaluation_variables."""
        judge = JudgeAgent()
        prompt_spec = make_prompt_spec()
        evidence = make_evidence_bundle()
        trace = make_reasoning_trace(steps=[
            ReasoningStep(
                step_id="step_0001",
                type="map",
                action="Map without eval vars",
                output={"no_eval_vars": True},
            ),
        ])
        
        verdict, result = judge.judge(prompt_spec, evidence, trace)
        
        assert result.ok is False
        assert verdict.outcome == "INVALID"
    
    def test_challenge_ref_for_schema_failure(self):
        """Should provide ChallengeRef when schema lock fails."""
        judge = JudgeAgent()
        prompt_spec = make_prompt_spec(
            output_schema_ref="wrong.schema.Reference",
        )
        evidence = make_evidence_bundle()
        trace = make_trace_with_eval_vars(event_observed=True)
        
        verdict, result = judge.judge(prompt_spec, evidence, trace)
        
        assert result.challenge is not None
        assert result.challenge.kind in ("por_bundle", "reasoning_leaf")


class TestSchemaLockConstants:
    """Tests for schema lock constants."""
    
    def test_expected_output_schema_ref(self):
        """EXPECTED_OUTPUT_SCHEMA_REF should be correct."""
        assert EXPECTED_OUTPUT_SCHEMA_REF == "core.schemas.verdict.DeterministicVerdict"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])