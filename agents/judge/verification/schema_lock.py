"""
Module 07 - Schema Lock Verification
Ensures Judge only relies on schema-locked, deterministic structures.

Owner: Protocol/Judging Engineer
Module ID: M07

This module validates:
1. PromptSpec.output_schema_ref points to DeterministicVerdict
2. PromptSpec.extra["strict_mode"] is True (for strict pipeline)
3. ReasoningTrace has final evaluation_variables
4. Outcome is one of YES/NO/INVALID
"""

from __future__ import annotations

from typing import Optional

from core.por.reasoning_trace import ReasoningTrace, ReasoningStep
from core.schemas.prompts import PromptSpec
from core.schemas.verification import CheckResult


# Expected output schema reference
EXPECTED_OUTPUT_SCHEMA_REF = "core.schemas.verdict.DeterministicVerdict"

# Valid outcomes
VALID_OUTCOMES = frozenset({"YES", "NO", "INVALID"})


class SchemaLock:
    """
    Validates schema-locked, deterministic structures for the Judge.
    
    This prevents "flexible" prompt modules from breaking Judge determinism.
    """
    
    @staticmethod
    def verify_prompt_spec(
        prompt_spec: PromptSpec,
        *,
        require_strict_mode: bool = True,
    ) -> list[CheckResult]:
        """
        Verify that the PromptSpec conforms to schema lock requirements.
        
        Args:
            prompt_spec: The PromptSpec to validate
            require_strict_mode: Whether to require strict_mode=True
        
        Returns:
            List of CheckResults
        """
        checks: list[CheckResult] = []
        
        # Check 1: Output schema reference
        actual_ref = prompt_spec.output_schema_ref
        if actual_ref != EXPECTED_OUTPUT_SCHEMA_REF:
            checks.append(CheckResult.failed(
                check_id="schema_lock.output_schema_ref",
                message=f"output_schema_ref must be '{EXPECTED_OUTPUT_SCHEMA_REF}'",
                details={
                    "expected": EXPECTED_OUTPUT_SCHEMA_REF,
                    "actual": actual_ref,
                },
            ))
        else:
            checks.append(CheckResult.passed(
                check_id="schema_lock.output_schema_ref",
                message="output_schema_ref correctly references DeterministicVerdict",
                details={"output_schema_ref": actual_ref},
            ))
        
        # Check 2: Strict mode (if required)
        if require_strict_mode:
            extra = prompt_spec.extra or {}
            strict_mode = extra.get("strict_mode", False)
            
            if not strict_mode:
                checks.append(CheckResult.failed(
                    check_id="schema_lock.strict_mode",
                    message="strict_mode must be True for deterministic judging",
                    details={
                        "strict_mode": strict_mode,
                        "extra_keys": list(extra.keys()),
                    },
                ))
            else:
                checks.append(CheckResult.passed(
                    check_id="schema_lock.strict_mode",
                    message="strict_mode is enabled",
                    details={"strict_mode": strict_mode},
                ))
        
        # Check 3: Market specification exists
        if prompt_spec.market is None:
            checks.append(CheckResult.failed(
                check_id="schema_lock.market_spec",
                message="MarketSpec is required",
                details={},
            ))
        else:
            checks.append(CheckResult.passed(
                check_id="schema_lock.market_spec",
                message="MarketSpec is present",
                details={"market_id": prompt_spec.market.market_id},
            ))
        
        # Check 4: Prediction semantics exists
        if prompt_spec.prediction_semantics is None:
            checks.append(CheckResult.failed(
                check_id="schema_lock.prediction_semantics",
                message="PredictionSemantics is required",
                details={},
            ))
        else:
            checks.append(CheckResult.passed(
                check_id="schema_lock.prediction_semantics",
                message="PredictionSemantics is present",
                details={
                    "target_entity": prompt_spec.prediction_semantics.target_entity,
                    "predicate": prompt_spec.prediction_semantics.predicate,
                },
            ))
        
        # Check 5: Resolution rules exist
        if (prompt_spec.market is not None and 
            prompt_spec.market.resolution_rules is not None and
            prompt_spec.market.resolution_rules.rules):
            checks.append(CheckResult.passed(
                check_id="schema_lock.resolution_rules",
                message="Resolution rules are defined",
                details={
                    "rule_count": len(prompt_spec.market.resolution_rules.rules),
                    "rule_ids": [r.rule_id for r in prompt_spec.market.resolution_rules.rules],
                },
            ))
        else:
            checks.append(CheckResult.warning(
                check_id="schema_lock.resolution_rules",
                message="No explicit resolution rules defined - using default mapping",
                details={},
            ))
        
        return checks
    
    @staticmethod
    def verify_trace(
        trace: ReasoningTrace,
        *,
        require_evaluation_variables: bool = True,
    ) -> list[CheckResult]:
        """
        Verify that the ReasoningTrace conforms to schema lock requirements.
        
        Args:
            trace: The ReasoningTrace to validate
            require_evaluation_variables: Whether to require final evaluation_variables
        
        Returns:
            List of CheckResults
        """
        checks: list[CheckResult] = []
        
        # Check 1: Trace has steps
        if not trace.steps:
            checks.append(CheckResult.failed(
                check_id="schema_lock.trace_steps",
                message="ReasoningTrace must have at least one step",
                details={"trace_id": trace.trace_id, "step_count": 0},
            ))
            return checks  # Can't continue without steps
        
        checks.append(CheckResult.passed(
            check_id="schema_lock.trace_steps",
            message=f"ReasoningTrace has {len(trace.steps)} steps",
            details={"trace_id": trace.trace_id, "step_count": len(trace.steps)},
        ))
        
        # Check 2: Find final evaluation step
        final_step = SchemaLock._find_evaluation_step(trace)
        
        if final_step is None:
            if require_evaluation_variables:
                checks.append(CheckResult.failed(
                    check_id="schema_lock.final_step",
                    message="No final 'map' or 'aggregate' step found",
                    details={
                        "step_types": [s.type for s in trace.steps],
                    },
                ))
            else:
                checks.append(CheckResult.warning(
                    check_id="schema_lock.final_step",
                    message="No final 'map' or 'aggregate' step found",
                    details={
                        "step_types": [s.type for s in trace.steps],
                    },
                ))
        else:
            checks.append(CheckResult.passed(
                check_id="schema_lock.final_step",
                message=f"Found final evaluation step '{final_step.step_id}'",
                details={
                    "step_id": final_step.step_id,
                    "step_type": final_step.type,
                },
            ))
        
        # Check 3: Final step has evaluation_variables
        if require_evaluation_variables:
            has_eval_vars = SchemaLock._has_evaluation_variables(trace)
            
            if not has_eval_vars:
                checks.append(CheckResult.failed(
                    check_id="schema_lock.evaluation_variables",
                    message="Final step must have evaluation_variables in output",
                    details={
                        "final_step_id": final_step.step_id if final_step else None,
                        "output_keys": list(final_step.output.keys()) if final_step else [],
                    },
                ))
            else:
                eval_vars = SchemaLock._get_evaluation_variables(trace)
                checks.append(CheckResult.passed(
                    check_id="schema_lock.evaluation_variables",
                    message="evaluation_variables present in final step",
                    details={
                        "eval_var_keys": list(eval_vars.keys()) if eval_vars else [],
                    },
                ))
        
        # Check 4: Step IDs are unique
        step_ids = [s.step_id for s in trace.steps]
        unique_ids = set(step_ids)
        
        if len(step_ids) != len(unique_ids):
            duplicates = [sid for sid in step_ids if step_ids.count(sid) > 1]
            checks.append(CheckResult.failed(
                check_id="schema_lock.unique_step_ids",
                message="Step IDs must be unique",
                details={"duplicates": list(set(duplicates))},
            ))
        else:
            checks.append(CheckResult.passed(
                check_id="schema_lock.unique_step_ids",
                message="All step IDs are unique",
                details={"step_count": len(step_ids)},
            ))
        
        # Check 5: Prior step references are valid
        seen_ids: set[str] = set()
        invalid_refs: list[tuple[str, str]] = []
        
        for step in trace.steps:
            for prior_id in step.prior_step_ids:
                if prior_id not in seen_ids:
                    invalid_refs.append((step.step_id, prior_id))
            seen_ids.add(step.step_id)
        
        if invalid_refs:
            checks.append(CheckResult.failed(
                check_id="schema_lock.prior_step_refs",
                message="Invalid prior step references found",
                details={"invalid_refs": invalid_refs},
            ))
        else:
            checks.append(CheckResult.passed(
                check_id="schema_lock.prior_step_refs",
                message="All prior step references are valid",
                details={},
            ))
        
        return checks
    
    @staticmethod
    def _find_evaluation_step(trace: ReasoningTrace) -> Optional[ReasoningStep]:
        """Find the final evaluation step (map or aggregate)."""
        if not trace.steps:
            return None
        
        # Look for last "map" step
        for step in reversed(trace.steps):
            if step.type == "map":
                return step
        
        # Look for last "aggregate" step
        for step in reversed(trace.steps):
            if step.type == "aggregate":
                return step
        
        return None
    
    @staticmethod
    def _has_evaluation_variables(trace: ReasoningTrace) -> bool:
        """Check if trace has evaluation_variables in any step output."""
        for step in reversed(trace.steps):
            if "evaluation_variables" in step.output:
                return True
        return False
    
    @staticmethod
    def _get_evaluation_variables(trace: ReasoningTrace) -> Optional[dict]:
        """Get evaluation_variables from trace."""
        for step in reversed(trace.steps):
            if "evaluation_variables" in step.output:
                return step.output["evaluation_variables"]
        return None
    
    @staticmethod
    def verify_outcome(outcome: str) -> CheckResult:
        """
        Verify that an outcome is valid.
        
        Args:
            outcome: The outcome string to validate
        
        Returns:
            CheckResult
        """
        if outcome in VALID_OUTCOMES:
            return CheckResult.passed(
                check_id="schema_lock.outcome",
                message=f"Outcome '{outcome}' is valid",
                details={"outcome": outcome, "valid_outcomes": list(VALID_OUTCOMES)},
            )
        else:
            return CheckResult.failed(
                check_id="schema_lock.outcome",
                message=f"Invalid outcome: '{outcome}'",
                details={"outcome": outcome, "valid_outcomes": list(VALID_OUTCOMES)},
            )


__all__ = [
    "SchemaLock",
    "EXPECTED_OUTPUT_SCHEMA_REF",
    "VALID_OUTCOMES",
]