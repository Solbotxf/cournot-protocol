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


# Default limits for evidence JSON to stay under model context (e.g. 128K)
DEFAULT_MAX_AUDIT_EVIDENCE_CHARS = 300_000
DEFAULT_MAX_AUDIT_EVIDENCE_ITEMS = 100
# Token estimation and hard cap for secondary truncation
DEFAULT_MODEL_CONTEXT_LIMIT = 128_000
DEFAULT_AUDIT_RESERVE_TOKENS = 15_000


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
        
        # Read audit evidence limits from context (set by pipeline/config)
        max_chars = ctx.extra.get(
            "max_audit_evidence_chars", DEFAULT_MAX_AUDIT_EVIDENCE_CHARS
        )
        max_items = ctx.extra.get(
            "max_audit_evidence_items", DEFAULT_MAX_AUDIT_EVIDENCE_ITEMS
        )
        
        # Prepare resolution rules (fixed size)
        rules_text = self._format_resolution_rules(prompt_spec)
        semantics = prompt_spec.prediction_semantics
        
        model_limit = ctx.extra.get(
            "model_context_limit", DEFAULT_MODEL_CONTEXT_LIMIT
        )
        reserve_tokens = ctx.extra.get(
            "audit_reserve_tokens", DEFAULT_AUDIT_RESERVE_TOKENS
        )
        target_tokens = model_limit - reserve_tokens
        
        # Prepare evidence and build messages; if estimated tokens exceed limit, reduce evidence and retry
        evidence_json = self._prepare_evidence_json(
            evidence_bundle, max_chars=max_chars, max_items=max_items, ctx=ctx
        )
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
        
        # Secondary truncation: if estimated tokens exceed target, reduce evidence and rebuild
        while self._estimate_tokens(messages) > target_tokens and max_chars > 2000:
            max_chars = max(2000, max_chars // 2)
            evidence_json = self._prepare_evidence_json(
                evidence_bundle, max_chars=max_chars, max_items=max_items, ctx=ctx
            )
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
            ctx.info(
                f"Audit messages over context limit; reduced evidence to max_chars={max_chars}"
            )
        
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
    
    def _prepare_evidence_json(
        self,
        bundle: EvidenceBundle,
        *,
        max_chars: int = DEFAULT_MAX_AUDIT_EVIDENCE_CHARS,
        max_items: int = DEFAULT_MAX_AUDIT_EVIDENCE_ITEMS,
        ctx: "AgentContext | None" = None,
    ) -> str:
        """Prepare evidence bundle as JSON for LLM, with truncation to stay under context limit."""
        total_bundle_items = len(bundle.items)
        # Apply item cap: keep first max_items, note the rest
        items_to_serialize = bundle.items[:max_items]
        truncated_by_count = total_bundle_items > max_items
        
        items = []
        for item in items_to_serialize:
            # Truncate extracted_fields the same way as parsed_value
            safe_extracted = {
                k: self._safe_value(v) for k, v in item.extracted_fields.items()
            }
            items.append({
                "evidence_id": item.evidence_id,
                "requirement_id": item.requirement_id,
                "source_id": item.provenance.source_id,
                "source_uri": item.provenance.source_uri,
                "provenance_tier": item.provenance.tier,
                "success": item.success,
                "error": item.error,
                "status_code": item.status_code,
                "extracted_fields": safe_extracted,
                "parsed_value": self._safe_value(item.parsed_value),
            })
        
        payload: dict[str, Any] = {
            "bundle_id": bundle.bundle_id,
            "market_id": bundle.market_id,
            "total_items": total_bundle_items,
            "items": items,
        }
        if truncated_by_count:
            payload["_truncation_note"] = (
                f"Only first {max_items} of {total_bundle_items} evidence items shown."
            )
        
        evidence_json = json.dumps(payload, indent=2)
        truncated_by_size = False
        
        # If still over max_chars, reduce by dropping items from the end until under
        while len(evidence_json) > max_chars and len(items) > 1:
            truncated_by_size = True
            items = items[:-1]
            payload["items"] = items
            payload["_truncation_note"] = (
                f"Evidence truncated to {len(items)} items (total {total_bundle_items}) "
                f"to fit context limit ({max_chars} chars)."
            )
            evidence_json = json.dumps(payload, indent=2)
        
        if ctx and (truncated_by_count or truncated_by_size):
            ctx.info(
                f"Audit evidence truncated: {len(items)} items shown, "
                f"total_items={total_bundle_items}, json_len={len(evidence_json)}"
            )
        
        return evidence_json
    
    def _safe_value(self, value: Any) -> Any:
        """Make value JSON-safe and bounded for LLM context."""
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            if isinstance(value, str) and len(value) > 500:
                return value[:500] + "..."
            return value
        if isinstance(value, dict):
            # Truncate large dicts by size or key count
            if len(str(value)) > 1000:
                return {"_truncated": True, "keys": list(value.keys())[:10]}
            items = list(value.items())
            if len(items) > 30:
                safe = {k: self._safe_value(v) for k, v in items[:30]}
                safe["_truncated_keys"] = len(items) - 30
                return safe
            return {k: self._safe_value(v) for k, v in value.items()}
        if isinstance(value, list):
            if len(value) > 10:
                return [self._safe_value(v) for v in value[:10]] + [
                    f"... and {len(value) - 10} more"
                ]
            return [self._safe_value(v) for v in value]
        return str(value)[:500]
    
    def _estimate_tokens(self, messages: list[dict[str, Any]]) -> int:
        """Estimate total token count for messages using character heuristic (~4 chars/token)."""
        total_chars = 0
        for m in messages:
            content = m.get("content")
            if isinstance(content, str):
                total_chars += len(content)
        return (total_chars + 3) // 4
    
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
