"""
LLM-Based Reasoning Engine

Generates reasoning traces using LLM for deeper analysis.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, TYPE_CHECKING

from core.schemas import (
    ConflictRecord,
    EvidenceBundle,
    EvidenceRef,
    PromptSpec,
    ReasoningStep,
    ReasoningTrace,
)
from .prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

if TYPE_CHECKING:
    from agents.context import AgentContext


class LLMReasoner:
    """
    LLM-based reasoning engine.
    
    Uses LLM to generate detailed reasoning traces with deeper analysis.
    """
    
    MAX_RETRIES = 2
    
    def __init__(self, strict_mode: bool = True) -> None:
        """
        Initialize LLM reasoner.
        
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
        Generate a reasoning trace using LLM.
        
        Args:
            ctx: Agent context with LLM client
            prompt_spec: The prompt specification
            evidence_bundle: Collected evidence
        
        Returns:
            ReasoningTrace with all reasoning steps
        """
        if not ctx.llm:
            raise ValueError("LLM client required for LLM reasoning")
        
        # Prepare evidence for LLM
        evidence_json = self._prepare_evidence_json(evidence_bundle)
        
        # Prepare resolution rules
        rules_text = self._format_resolution_rules(prompt_spec)
        
        # Build prompt
        semantics = prompt_spec.prediction_semantics
        user_prompt = USER_PROMPT_TEMPLATE.format(
            question=prompt_spec.market.question,
            event_definition=prompt_spec.market.event_definition,
            resolution_rules=rules_text,
            evidence_json=evidence_json,
            target_entity=semantics.target_entity,
            predicate=semantics.predicate,
            threshold=semantics.threshold or "N/A",
            timeframe=semantics.timeframe or "N/A",
        )
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        
        # Get LLM response
        response = ctx.llm.chat(messages)
        raw_output = response.content
        
        # Parse response with retries
        last_error: str | None = None
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                parsed = self._extract_json(raw_output)
                trace = self._build_trace(ctx, prompt_spec, evidence_bundle, parsed)
                return trace
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                last_error = str(e)
                ctx.warning(f"JSON parse attempt {attempt + 1} failed: {last_error}")
                
                if attempt < self.MAX_RETRIES:
                    # Request repair
                    repair_prompt = f"The JSON was invalid: {last_error}. Please fix and return valid JSON only."
                    messages.append({"role": "assistant", "content": raw_output})
                    messages.append({"role": "user", "content": repair_prompt})
                    
                    response = ctx.llm.chat(messages)
                    raw_output = response.content
        
        raise ValueError(f"Failed to generate reasoning trace after {self.MAX_RETRIES + 1} attempts: {last_error}")
    
    def _prepare_evidence_json(self, bundle: EvidenceBundle) -> str:
        """Prepare evidence bundle as JSON for LLM."""
        items = []
        for item in bundle.items:
            items.append({
                "evidence_id": item.evidence_id,
                "requirement_id": item.requirement_id,
                "source_id": item.provenance.source_id,
                "source_uri": item.provenance.source_uri,
                "provenance_tier": item.provenance.tier,
                "success": item.success,
                "error": item.error,
                "status_code": item.status_code,
                "extracted_fields": item.extracted_fields,
                "parsed_value": self._safe_value(item.parsed_value),
            })
        
        return json.dumps({
            "bundle_id": bundle.bundle_id,
            "market_id": bundle.market_id,
            "total_items": len(items),
            "items": items,
        }, indent=2)
    
    def _safe_value(self, value: Any) -> Any:
        """Make value JSON-safe."""
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            # Truncate large dicts
            if len(str(value)) > 1000:
                return {"_truncated": True, "keys": list(value.keys())[:10]}
            return value
        if isinstance(value, list):
            if len(value) > 10:
                return value[:10] + [f"... and {len(value) - 10} more"]
            return value
        return str(value)[:500]
    
    def _format_resolution_rules(self, prompt_spec: PromptSpec) -> str:
        """Format resolution rules for LLM."""
        rules = prompt_spec.market.resolution_rules.get_sorted_rules()
        lines = []
        for rule in rules:
            lines.append(f"- {rule.rule_id} (priority {rule.priority}): {rule.description}")
        return "\n".join(lines)
    
    def _extract_json(self, text: str) -> dict[str, Any]:
        """Extract JSON from LLM response."""
        # Try to extract from code block
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if json_match:
            return json.loads(json_match.group(1).strip())
        
        # Try bare JSON
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
        
        return json.loads(text.strip())
    
    def _build_trace(
        self,
        ctx: "AgentContext",
        prompt_spec: PromptSpec,
        evidence_bundle: EvidenceBundle,
        data: dict[str, Any],
    ) -> ReasoningTrace:
        """Build ReasoningTrace from LLM JSON output."""
        # Generate trace ID
        trace_id = data.get("trace_id") or self._generate_trace_id(evidence_bundle.bundle_id)
        
        # Build steps
        steps: list[ReasoningStep] = []
        for i, step_data in enumerate(data.get("steps", [])):
            # Build evidence refs
            evidence_refs: list[EvidenceRef] = []
            for ref_data in step_data.get("evidence_refs", []):
                evidence_refs.append(EvidenceRef(
                    evidence_id=ref_data.get("evidence_id", ""),
                    requirement_id=ref_data.get("requirement_id"),
                    source_id=ref_data.get("source_id"),
                    field_used=ref_data.get("field_used"),
                    value_at_reference=ref_data.get("value_at_reference"),
                ))
            
            steps.append(ReasoningStep(
                step_id=step_data.get("step_id", f"step_{i + 1:03d}"),
                step_type=step_data.get("step_type", "inference"),
                description=step_data.get("description", ""),
                evidence_refs=evidence_refs,
                rule_id=step_data.get("rule_id"),
                input_summary=step_data.get("input_summary"),
                output_summary=step_data.get("output_summary"),
                conclusion=step_data.get("conclusion"),
                confidence_delta=step_data.get("confidence_delta"),
                depends_on=step_data.get("depends_on", []),
            ))
        
        # Build conflicts
        conflicts: list[ConflictRecord] = []
        for conf_data in data.get("conflicts", []):
            conflicts.append(ConflictRecord(
                conflict_id=conf_data.get("conflict_id", f"conflict_{len(conflicts) + 1}"),
                evidence_ids=conf_data.get("evidence_ids", []),
                description=conf_data.get("description", ""),
                resolution=conf_data.get("resolution"),
                resolution_rationale=conf_data.get("resolution_rationale"),
                winning_evidence_id=conf_data.get("winning_evidence_id"),
            ))
        
        # Map outcome
        outcome = data.get("preliminary_outcome", "UNCERTAIN")
        if outcome not in ("YES", "NO", "INVALID", "UNCERTAIN"):
            outcome = "UNCERTAIN"
        
        # Build trace
        trace = ReasoningTrace(
            trace_id=trace_id,
            market_id=prompt_spec.market_id,
            bundle_id=evidence_bundle.bundle_id,
            steps=steps,
            conflicts=conflicts,
            evidence_summary=data.get("evidence_summary"),
            reasoning_summary=data.get("reasoning_summary"),
            preliminary_outcome=outcome,
            preliminary_confidence=data.get("preliminary_confidence", 0.5),
            recommended_rule_id=data.get("recommended_rule_id"),
            created_at=ctx.now() if not self.strict_mode else None,
            metadata={
                "reasoner": "llm",
                "strict_mode": self.strict_mode,
            },
        )
        
        return trace
    
    def _generate_trace_id(self, bundle_id: str) -> str:
        """Generate deterministic trace ID."""
        hash_bytes = hashlib.sha256(bundle_id.encode()).digest()
        return f"trace_{hash_bytes[:8].hex()}"
