"""
Verdict Builder

Constructs DeterministicVerdict from reasoning traces and evidence.
Handles:
- Hash computation for audit trail
- Confidence validation and adjustment
- Rule selection and application
- Justification generation
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, TYPE_CHECKING

from core.schemas import (
    DeterministicVerdict,
    EvidenceBundle,
    PromptSpec,
    ReasoningTrace,
)
from core.schemas.canonical import dumps_canonical

if TYPE_CHECKING:
    from agents.context import AgentContext


class VerdictBuilder:
    """
    Builds DeterministicVerdict from reasoning traces.
    
    Responsibilities:
    - Validate reasoning trace
    - Compute artifact hashes
    - Finalize confidence score
    - Select resolution rule
    - Generate justification
    """
    
    # Minimum confidence thresholds
    MIN_CONFIDENCE_FOR_YES_NO = 0.55
    MIN_CONFIDENCE_FOR_HIGH = 0.75
    
    def __init__(self, strict_mode: bool = True) -> None:
        """
        Initialize builder.
        
        Args:
            strict_mode: If True, exclude timestamps from hashing
        """
        self.strict_mode = strict_mode
    
    def build(
        self,
        ctx: "AgentContext",
        prompt_spec: PromptSpec,
        evidence_bundle: EvidenceBundle,
        reasoning_trace: ReasoningTrace,
        *,
        override_outcome: str | None = None,
        override_confidence: float | None = None,
        override_rule_id: str | None = None,
    ) -> DeterministicVerdict:
        """
        Build a DeterministicVerdict from reasoning trace.
        
        Args:
            ctx: Agent context
            prompt_spec: The prompt specification
            evidence_bundle: Collected evidence
            reasoning_trace: The reasoning trace from auditor
            override_outcome: Override the preliminary outcome
            override_confidence: Override the confidence score
            override_rule_id: Override the resolution rule
        
        Returns:
            DeterministicVerdict ready for settlement
        """
        # Determine outcome
        outcome = self._determine_outcome(
            prompt_spec,
            reasoning_trace,
            override_outcome,
        )
        
        # Determine confidence
        confidence = self._determine_confidence(
            reasoning_trace,
            outcome,
            override_confidence,
        )
        
        # Determine rule
        rule_id = self._determine_rule(
            prompt_spec,
            reasoning_trace,
            outcome,
            override_rule_id,
        )
        
        # Compute hashes
        prompt_spec_hash = self._compute_prompt_spec_hash(prompt_spec)
        evidence_root = self._compute_evidence_root(evidence_bundle)
        reasoning_root = self._compute_reasoning_root(reasoning_trace)
        
        # Generate justification
        justification = self._generate_justification(
            prompt_spec,
            evidence_bundle,
            reasoning_trace,
            outcome,
            confidence,
            rule_id,
        )
        justification_hash = self._hash_string(justification)
        
        # Get selected evidence refs
        selected_leaf_refs = reasoning_trace.get_evidence_refs()
        
        # Build verdict
        verdict = DeterministicVerdict(
            market_id=prompt_spec.market_id,
            outcome=outcome,
            confidence=confidence,
            resolution_time=ctx.now() if not self.strict_mode else None,
            resolution_rule_id=rule_id,
            prompt_spec_hash=prompt_spec_hash,
            evidence_root=evidence_root,
            reasoning_root=reasoning_root,
            justification_hash=justification_hash,
            selected_leaf_refs=selected_leaf_refs,
            metadata={
                "strict_mode": self.strict_mode,
                "trace_id": reasoning_trace.trace_id,
                "bundle_id": evidence_bundle.bundle_id,
                "step_count": reasoning_trace.step_count,
                "conflict_count": len(reasoning_trace.conflicts),
                "justification": justification,
            },
        )
        
        return verdict
    
    def _determine_outcome(
        self,
        prompt_spec: PromptSpec,
        trace: ReasoningTrace,
        override: str | None,
    ) -> str:
        """Determine the final outcome."""
        if override is not None:
            return override

        preliminary = trace.preliminary_outcome

        if preliminary and prompt_spec.market.is_valid_outcome(preliminary):
            return preliminary
        elif preliminary == "UNCERTAIN" or preliminary is None:
            return "INVALID"
        else:
            # Unknown outcome, fall back to INVALID
            return "INVALID"
    
    def _determine_confidence(
        self,
        trace: ReasoningTrace,
        outcome: str,
        override: float | None,
    ) -> float:
        """Determine the final confidence score."""
        if override is not None:
            return max(0.0, min(1.0, override))

        base_confidence = trace.preliminary_confidence or 0.5

        # Adjust based on outcome
        if outcome == "INVALID":
            # For INVALID, confidence represents certainty that resolution is impossible
            return max(0.0, min(1.0, base_confidence))

        # For definitive outcomes, ensure minimum threshold
        if base_confidence < self.MIN_CONFIDENCE_FOR_YES_NO:
            base_confidence = self.MIN_CONFIDENCE_FOR_YES_NO

        return max(0.0, min(1.0, base_confidence))
    
    def _determine_rule(
        self,
        prompt_spec: PromptSpec,
        trace: ReasoningTrace,
        outcome: str,
        override: str | None,
    ) -> str:
        """Determine the resolution rule to cite."""
        if override is not None:
            return override
        
        # Use recommended rule from trace if available
        if trace.recommended_rule_id:
            return trace.recommended_rule_id
        
        # Select based on outcome
        rules = prompt_spec.market.resolution_rules.get_sorted_rules()
        
        if outcome == "INVALID":
            # Look for fallback rule
            for rule in rules:
                if "invalid" in rule.rule_id.lower() or "fallback" in rule.rule_id.lower():
                    return rule.rule_id
        
        # Default to highest priority rule
        if rules:
            return rules[0].rule_id
        
        return "R_DEFAULT"
    
    def _compute_prompt_spec_hash(self, prompt_spec: PromptSpec) -> str:
        """Compute hash of prompt spec."""
        # Exclude timestamps for deterministic hashing
        spec_dict = prompt_spec.model_dump(exclude={"created_at"})
        canonical = dumps_canonical(spec_dict)
        return self._hash_string(canonical)
    
    def _compute_evidence_root(self, bundle: EvidenceBundle) -> str:
        """Compute merkle root of evidence (simplified: hash of all evidence IDs)."""
        evidence_ids = sorted([item.evidence_id for item in bundle.items])
        combined = "|".join(evidence_ids)
        return self._hash_string(combined)
    
    def _compute_reasoning_root(self, trace: ReasoningTrace) -> str:
        """Compute hash of reasoning trace."""
        # Hash step IDs and conclusions
        step_data = []
        for step in trace.steps:
            step_data.append({
                "step_id": step.step_id,
                "step_type": step.step_type,
                "conclusion": step.conclusion,
            })
        canonical = dumps_canonical(step_data)
        return self._hash_string(canonical)
    
    def _generate_justification(
        self,
        prompt_spec: PromptSpec,
        evidence_bundle: EvidenceBundle,
        trace: ReasoningTrace,
        outcome: str,
        confidence: float,
        rule_id: str,
    ) -> str:
        """Generate human-readable justification."""
        parts = []
        
        # Header
        parts.append(f"Market: {prompt_spec.market.question}")
        parts.append(f"Outcome: {outcome}")
        parts.append(f"Confidence: {confidence:.0%}")
        parts.append(f"Rule Applied: {rule_id}")
        parts.append("")
        
        # Evidence summary
        parts.append("Evidence Summary:")
        parts.append(trace.evidence_summary or f"Analyzed {len(evidence_bundle.items)} evidence items.")
        parts.append("")
        
        # Reasoning summary
        parts.append("Reasoning:")
        parts.append(trace.reasoning_summary or f"Completed {trace.step_count} reasoning steps.")
        parts.append("")
        
        # Key conclusions from steps
        conclusions = [
            step.conclusion for step in trace.steps
            if step.conclusion and step.step_type in ("threshold_check", "conclusion", "rule_application")
        ]
        if conclusions:
            parts.append("Key Conclusions:")
            for i, conclusion in enumerate(conclusions[:5], 1):
                parts.append(f"  {i}. {conclusion}")
        
        return "\n".join(parts)
    
    def _hash_string(self, s: str) -> str:
        """Compute SHA256 hash of string."""
        return f"0x{hashlib.sha256(s.encode()).hexdigest()}"


class VerdictValidator:
    """
    Validates DeterministicVerdict for correctness.
    """
    
    def validate(
        self,
        verdict: DeterministicVerdict,
        prompt_spec: PromptSpec,
        evidence_bundle: EvidenceBundle,
        reasoning_trace: ReasoningTrace,
    ) -> tuple[bool, list[str]]:
        """
        Validate a verdict.
        
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors: list[str] = []
        
        # Check market ID matches
        if verdict.market_id != prompt_spec.market_id:
            errors.append(f"Market ID mismatch: {verdict.market_id} != {prompt_spec.market_id}")
        
        # Check outcome is valid for this market
        if not prompt_spec.market.is_valid_outcome(verdict.outcome):
            errors.append(f"Invalid outcome: {verdict.outcome}. Valid: {prompt_spec.market.get_valid_outcomes()}")
        
        # Check confidence bounds
        if not (0.0 <= verdict.confidence <= 1.0):
            errors.append(f"Confidence out of bounds: {verdict.confidence}")
        
        # Check rule exists in prompt spec
        rule = prompt_spec.market.resolution_rules.get_rule_by_id(verdict.resolution_rule_id)
        if rule is None and verdict.resolution_rule_id != "R_DEFAULT":
            errors.append(f"Unknown resolution rule: {verdict.resolution_rule_id}")
        
        # Check hashes are present
        if not verdict.prompt_spec_hash:
            errors.append("Missing prompt_spec_hash")
        if not verdict.evidence_root:
            errors.append("Missing evidence_root")
        if not verdict.reasoning_root:
            errors.append("Missing reasoning_root")
        
        return len(errors) == 0, errors
