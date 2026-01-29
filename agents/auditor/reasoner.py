"""
Rule-Based Reasoning Engine

Generates reasoning traces using deterministic rule-based logic.
Used as fallback when LLM is unavailable.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, TYPE_CHECKING

from core.schemas import (
    ConflictRecord,
    EvidenceBundle,
    EvidenceItem,
    EvidenceRef,
    PromptSpec,
    ReasoningStep,
    ReasoningTrace,
)

if TYPE_CHECKING:
    from agents.context import AgentContext


class RuleBasedReasoner:
    """
    Deterministic rule-based reasoning engine.
    
    Generates reasoning traces by:
    1. Validating evidence
    2. Extracting relevant values
    3. Applying resolution rules
    4. Detecting and resolving conflicts
    5. Computing confidence
    6. Drawing conclusions
    """
    
    def __init__(self, strict_mode: bool = True) -> None:
        """
        Initialize reasoner.
        
        Args:
            strict_mode: If True, exclude timestamps from output
        """
        self.strict_mode = strict_mode
    
    def reason(
        self,
        ctx: "AgentContext",
        prompt_spec: PromptSpec,
        evidence_bundle: EvidenceBundle,
    ) -> ReasoningTrace:
        """
        Generate a reasoning trace from evidence.
        
        Args:
            ctx: Agent context
            prompt_spec: The prompt specification
            evidence_bundle: Collected evidence
        
        Returns:
            ReasoningTrace with all reasoning steps
        """
        # Initialize trace
        trace = ReasoningTrace(
            trace_id=self._generate_trace_id(evidence_bundle.bundle_id),
            market_id=prompt_spec.market_id,
            bundle_id=evidence_bundle.bundle_id,
            created_at=ctx.now() if not self.strict_mode else None,
        )
        
        step_counter = 0
        confidence = 0.5  # Start neutral
        
        # Step 1: Validity checks
        valid_evidence = evidence_bundle.get_valid_evidence()
        
        step_counter += 1
        trace.add_step(ReasoningStep(
            step_id=f"step_{step_counter:03d}",
            step_type="validity_check",
            description=f"Check validity of {len(evidence_bundle.items)} evidence items",
            input_summary=f"Total items: {len(evidence_bundle.items)}",
            output_summary=f"Valid items: {len(valid_evidence)}, Invalid: {len(evidence_bundle.items) - len(valid_evidence)}",
            conclusion=f"{len(valid_evidence)} valid evidence items available",
            confidence_delta=0.0 if valid_evidence else -0.3,
        ))
        
        if not valid_evidence:
            confidence -= 0.3
        
        # Step 2: Analyze each evidence item
        extracted_values: dict[str, Any] = {}
        
        for item in valid_evidence:
            step_counter += 1
            value = self._extract_primary_value(item, prompt_spec)
            extracted_values[item.evidence_id] = value
            
            trace.add_step(ReasoningStep(
                step_id=f"step_{step_counter:03d}",
                step_type="evidence_analysis",
                description=f"Analyze evidence from {item.provenance.source_id}",
                evidence_refs=[EvidenceRef(
                    evidence_id=item.evidence_id,
                    requirement_id=item.requirement_id,
                    source_id=item.provenance.source_id,
                    field_used=self._get_primary_field(item),
                    value_at_reference=value,
                )],
                input_summary=f"Source: {item.provenance.source_id}, Tier: {item.provenance.tier}",
                output_summary=f"Extracted value: {value}",
                confidence_delta=0.05 if item.provenance.tier >= 2 else 0.0,
            ))
            
            if item.provenance.tier >= 2:
                confidence += 0.05
        
        # Step 3: Detect and resolve conflicts
        conflicts = self._detect_conflicts(valid_evidence, extracted_values, prompt_spec)
        
        for conflict in conflicts:
            trace.add_conflict(conflict)
            step_counter += 1
            
            trace.add_step(ReasoningStep(
                step_id=f"step_{step_counter:03d}",
                step_type="conflict_resolution",
                description=f"Resolve conflict: {conflict.description}",
                evidence_refs=[
                    EvidenceRef(evidence_id=eid)
                    for eid in conflict.evidence_ids
                ],
                input_summary=f"Conflicting evidence: {conflict.evidence_ids}",
                output_summary=f"Resolution: {conflict.resolution}",
                conclusion=f"Using evidence {conflict.winning_evidence_id}",
                confidence_delta=-0.1,
            ))
            confidence -= 0.1
        
        # Step 4: Apply resolution rules
        semantics = prompt_spec.prediction_semantics
        rules = prompt_spec.market.resolution_rules.get_sorted_rules()
        
        # Determine the best value to use (highest tier, conflict winner)
        best_value = self._get_best_value(valid_evidence, extracted_values, conflicts)
        best_evidence = self._get_best_evidence(valid_evidence, conflicts)
        
        # Apply threshold rule if applicable
        threshold = self._parse_threshold(semantics.threshold)
        outcome = "UNCERTAIN"
        rule_applied = None
        
        if threshold is not None and best_value is not None:
            # Threshold comparison
            step_counter += 1
            comparison_result = self._compare_threshold(best_value, threshold, prompt_spec)
            
            trace.add_step(ReasoningStep(
                step_id=f"step_{step_counter:03d}",
                step_type="threshold_check",
                description=f"Compare value to threshold",
                evidence_refs=[EvidenceRef(
                    evidence_id=best_evidence.evidence_id if best_evidence else "unknown",
                    value_at_reference=best_value,
                )] if best_evidence else [],
                rule_id="R_THRESHOLD",
                input_summary=f"Value: {best_value}, Threshold: {threshold}",
                output_summary=f"Comparison: {best_value} vs {threshold} = {comparison_result}",
                conclusion=f"Threshold {'met' if comparison_result else 'not met'}",
                confidence_delta=0.15,
                depends_on=[f"step_{i:03d}" for i in range(1, step_counter)],
            ))
            
            outcome = "YES" if comparison_result else "NO"
            rule_applied = "R_THRESHOLD"
            confidence += 0.15
        
        elif best_value is not None:
            # Non-threshold evaluation
            step_counter += 1
            
            trace.add_step(ReasoningStep(
                step_id=f"step_{step_counter:03d}",
                step_type="rule_application",
                description="Apply binary decision rule",
                evidence_refs=[EvidenceRef(
                    evidence_id=best_evidence.evidence_id if best_evidence else "unknown",
                    value_at_reference=best_value,
                )] if best_evidence else [],
                rule_id="R_BINARY_DECISION",
                input_summary=f"Best evidence value: {best_value}",
                output_summary="Evaluating truthiness of evidence",
                confidence_delta=0.1,
            ))
            
            # Simple truthiness check
            if self._is_truthy(best_value):
                outcome = "YES"
            else:
                outcome = "NO"
            rule_applied = "R_BINARY_DECISION"
            confidence += 0.1
        
        else:
            # No usable value
            step_counter += 1
            
            trace.add_step(ReasoningStep(
                step_id=f"step_{step_counter:03d}",
                step_type="rule_application",
                description="Apply invalid fallback rule",
                rule_id="R_INVALID_FALLBACK",
                input_summary="No usable evidence value found",
                output_summary="Cannot determine outcome",
                conclusion="Evidence insufficient for resolution",
                confidence_delta=-0.2,
            ))
            
            outcome = "INVALID"
            rule_applied = "R_INVALID_FALLBACK"
            confidence -= 0.2
        
        # Step 5: Confidence assessment
        step_counter += 1
        confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
        
        trace.add_step(ReasoningStep(
            step_id=f"step_{step_counter:03d}",
            step_type="confidence_assessment",
            description="Assess overall confidence",
            input_summary=f"Valid sources: {len(valid_evidence)}, Conflicts: {len(conflicts)}",
            output_summary=f"Final confidence: {confidence:.2f}",
            confidence_delta=0.0,
        ))
        
        # Step 6: Conclusion
        step_counter += 1
        
        trace.add_step(ReasoningStep(
            step_id=f"step_{step_counter:03d}",
            step_type="conclusion",
            description="Draw final conclusion",
            input_summary=f"Outcome: {outcome}, Confidence: {confidence:.2f}",
            output_summary=f"Preliminary verdict: {outcome} with {confidence:.0%} confidence",
            conclusion=f"Based on {len(valid_evidence)} evidence items, recommend {outcome}",
            depends_on=[f"step_{i:03d}" for i in range(1, step_counter)],
        ))
        
        # Set trace summary fields
        trace.evidence_summary = self._summarize_evidence(valid_evidence)
        trace.reasoning_summary = self._summarize_reasoning(trace, outcome, confidence)
        trace.preliminary_outcome = outcome
        trace.preliminary_confidence = confidence
        trace.recommended_rule_id = rule_applied
        
        return trace
    
    def _generate_trace_id(self, bundle_id: str) -> str:
        """Generate deterministic trace ID."""
        hash_bytes = hashlib.sha256(bundle_id.encode()).digest()
        return f"trace_{hash_bytes[:8].hex()}"
    
    def _extract_primary_value(self, item: EvidenceItem, prompt_spec: PromptSpec) -> Any:
        """Extract the primary value from an evidence item."""
        # Check extracted fields first
        if item.extracted_fields:
            # Direct price fields
            for key in ["price_usd", "usd", "price", "amount", "value"]:
                if key in item.extracted_fields:
                    val = item.extracted_fields[key]
                    try:
                        return float(val) if val is not None else None
                    except (ValueError, TypeError):
                        pass
            
            # Nested fields (e.g., bitcoin_usd)
            for key, val in item.extracted_fields.items():
                if isinstance(val, (int, float)):
                    if "usd" in key.lower() or "price" in key.lower():
                        return float(val)
                elif isinstance(val, dict):
                    # Handle CoinGecko style: {"bitcoin": {"usd": 50000}}
                    nested_price = self._find_price_in_dict(val)
                    if nested_price is not None:
                        return nested_price
        
        # Check parsed value
        if item.parsed_value is not None:
            if isinstance(item.parsed_value, (int, float)):
                return float(item.parsed_value)
            if isinstance(item.parsed_value, dict):
                # Try to find price in nested dict
                return self._find_price_in_dict(item.parsed_value)
        
        return None
    
    def _find_price_in_dict(self, d: dict, depth: int = 0) -> float | None:
        """Recursively find a price value in a dict."""
        if depth > 3:
            return None
        
        for key, val in d.items():
            if isinstance(val, (int, float)):
                if any(p in key.lower() for p in ["price", "usd", "amount", "value"]):
                    return float(val)
            elif isinstance(val, dict):
                result = self._find_price_in_dict(val, depth + 1)
                if result is not None:
                    return result
            elif isinstance(val, str):
                try:
                    return float(val)
                except ValueError:
                    pass
        
        return None
    
    def _get_primary_field(self, item: EvidenceItem) -> str | None:
        """Get the name of the primary field used."""
        if item.extracted_fields:
            for key in ["price_usd", "usd", "price", "amount", "value"]:
                if key in item.extracted_fields:
                    return key
        return None
    
    def _detect_conflicts(
        self,
        evidence: list[EvidenceItem],
        values: dict[str, Any],
        prompt_spec: PromptSpec,
    ) -> list[ConflictRecord]:
        """Detect conflicts between evidence items."""
        conflicts: list[ConflictRecord] = []
        
        # Get numeric values only
        numeric_values = {
            eid: val for eid, val in values.items()
            if isinstance(val, (int, float))
        }
        
        if len(numeric_values) < 2:
            return conflicts
        
        # Check for significant differences (>10%)
        vals = list(numeric_values.values())
        eids = list(numeric_values.keys())
        
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                if vals[i] == 0 or vals[j] == 0:
                    continue
                
                diff_pct = abs(vals[i] - vals[j]) / max(vals[i], vals[j])
                if diff_pct > 0.1:  # >10% difference
                    # Find winning evidence
                    ev_i = next((e for e in evidence if e.evidence_id == eids[i]), None)
                    ev_j = next((e for e in evidence if e.evidence_id == eids[j]), None)
                    
                    winner_id = eids[i]
                    if ev_i and ev_j:
                        if ev_j.provenance.tier > ev_i.provenance.tier:
                            winner_id = eids[j]
                    
                    conflicts.append(ConflictRecord(
                        conflict_id=f"conflict_{len(conflicts) + 1:03d}",
                        evidence_ids=[eids[i], eids[j]],
                        description=f"Values differ by {diff_pct:.1%}: {vals[i]} vs {vals[j]}",
                        resolution="Use higher provenance tier source",
                        resolution_rationale=f"Evidence {winner_id} has higher provenance tier",
                        winning_evidence_id=winner_id,
                    ))
        
        return conflicts
    
    def _get_best_value(
        self,
        evidence: list[EvidenceItem],
        values: dict[str, Any],
        conflicts: list[ConflictRecord],
    ) -> Any:
        """Get the best value considering conflicts."""
        if not evidence:
            return None
        
        # If there are conflicts, use the winning evidence
        conflict_winners = {c.winning_evidence_id for c in conflicts if c.winning_evidence_id}
        
        # Filter to winners if conflicts exist
        candidates = evidence
        if conflict_winners:
            candidates = [e for e in evidence if e.evidence_id in conflict_winners]
        
        if not candidates:
            candidates = evidence
        
        # Sort by provenance tier (highest first)
        candidates = sorted(candidates, key=lambda e: e.provenance.tier, reverse=True)
        
        for ev in candidates:
            val = values.get(ev.evidence_id)
            if val is not None:
                return val
        
        return None
    
    def _get_best_evidence(
        self,
        evidence: list[EvidenceItem],
        conflicts: list[ConflictRecord],
    ) -> EvidenceItem | None:
        """Get the best evidence item."""
        if not evidence:
            return None
        
        # If there are conflicts, prefer winners
        conflict_winners = {c.winning_evidence_id for c in conflicts if c.winning_evidence_id}
        
        candidates = evidence
        if conflict_winners:
            candidates = [e for e in evidence if e.evidence_id in conflict_winners]
        
        if not candidates:
            candidates = evidence
        
        # Sort by provenance tier
        return max(candidates, key=lambda e: e.provenance.tier)
    
    def _parse_threshold(self, threshold: str | None) -> float | None:
        """Parse threshold string to float."""
        if threshold is None:
            return None
        
        # Remove common formatting
        cleaned = str(threshold).replace(",", "").replace("$", "").strip()
        
        # Handle "100k" style
        multipliers = {"k": 1000, "m": 1000000, "b": 1000000000}
        for suffix, mult in multipliers.items():
            if cleaned.lower().endswith(suffix):
                try:
                    return float(cleaned[:-1]) * mult
                except ValueError:
                    pass
        
        try:
            return float(cleaned)
        except ValueError:
            return None
    
    def _compare_threshold(
        self,
        value: Any,
        threshold: float,
        prompt_spec: PromptSpec,
    ) -> bool:
        """Compare value to threshold."""
        if value is None:
            return False
        
        try:
            val = float(value)
        except (ValueError, TypeError):
            return False
        
        # Determine comparison direction from predicate
        predicate = prompt_spec.prediction_semantics.predicate.lower()
        event_def = prompt_spec.market.event_definition.lower()
        
        if any(op in predicate or op in event_def for op in ["above", "over", "exceed", "greater", ">"]):
            return val > threshold
        elif any(op in predicate or op in event_def for op in ["below", "under", "less", "<"]):
            return val < threshold
        elif any(op in predicate or op in event_def for op in ["equal", "=", "=="]):
            return abs(val - threshold) < 0.01 * threshold  # Within 1%
        
        # Default: above
        return val > threshold
    
    def _is_truthy(self, value: Any) -> bool:
        """Check if a value is truthy for binary decisions."""
        if value is None:
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value > 0
        if isinstance(value, str):
            return value.lower() in ("true", "yes", "1", "resolved", "confirmed")
        return bool(value)
    
    def _summarize_evidence(self, evidence: list[EvidenceItem]) -> str:
        """Generate evidence summary."""
        if not evidence:
            return "No valid evidence collected."
        
        sources = [e.provenance.source_id for e in evidence]
        tiers = [e.provenance.tier for e in evidence]
        
        return (
            f"Analyzed {len(evidence)} evidence items from sources: {', '.join(set(sources))}. "
            f"Provenance tiers range from {min(tiers)} to {max(tiers)}."
        )
    
    def _summarize_reasoning(
        self,
        trace: ReasoningTrace,
        outcome: str,
        confidence: float,
    ) -> str:
        """Generate reasoning summary."""
        return (
            f"Completed {trace.step_count} reasoning steps. "
            f"Detected {len(trace.conflicts)} conflicts. "
            f"Preliminary outcome: {outcome} with {confidence:.0%} confidence."
        )
