"""
LLM-Based Prompt Compiler

Uses LLM to convert user questions into structured PromptSpec + ToolPlan.
Includes JSON validation with repair loop.
"""

from __future__ import annotations

import logging
import json
import re
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Any, TYPE_CHECKING

from pydantic import ValidationError

from core.schemas import (
    DataRequirement,
    DisputePolicy,
    MarketSpec,
    PredictionSemantics,
    PromptSpec,
    ResolutionRule,
    ResolutionRules,
    ResolutionWindow,
    SelectionPolicy,
    SourcePolicy,
    SourceTarget,
    ToolPlan,
)
from .prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, JSON_REPAIR_PROMPT

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from agents.context import AgentContext


class LLMPromptCompiler:
    """
    Compiles user questions to PromptSpec using LLM.
    
    Features:
    - Deterministic market IDs from question hash
    - JSON validation with repair loop (max 2 retries)
    - Strict schema enforcement
    """
    
    MAX_RETRIES = 2
    
    def __init__(self, strict_mode: bool = True) -> None:
        """
        Initialize compiler.
        
        Args:
            strict_mode: If True, enforce strict validation
        """
        self.strict_mode = strict_mode
    
    def compile(
        self,
        ctx: AgentContext,
        user_input: str,
    ) -> tuple[PromptSpec, ToolPlan]:
        """
        Compile user input to PromptSpec and ToolPlan.
        
        Args:
            ctx: Agent context with LLM client
            user_input: User's prediction question
        
        Returns:
            Tuple of (PromptSpec, ToolPlan)
        
        Raises:
            ValueError: If compilation fails after retries
        """
        if not ctx.llm:
            raise ValueError("LLM client required for LLM compilation")
        
        current_time = ctx.now().isoformat()
        
        # Build user prompt
        user_prompt = USER_PROMPT_TEMPLATE.format(
            user_input=user_input,
            current_time=current_time,
        )
        
        # First attempt
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        
        response = ctx.llm.chat(messages)
        raw_output = response.content

        logger.debug(f"LLM Prompt raw output: {raw_output}")
        # Try to parse with retries
        last_error: str | None = None
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                parsed = self._extract_json(raw_output)
                prompt_spec, tool_plan = self._build_from_llm_output(
                    ctx, user_input, parsed
                )
                return prompt_spec, tool_plan
            except (json.JSONDecodeError, ValidationError, KeyError) as e:
                last_error = str(e)
                ctx.warning(f"JSON parse attempt {attempt + 1} failed: {last_error}")
                
                if attempt < self.MAX_RETRIES:
                    # Request repair
                    repair_prompt = JSON_REPAIR_PROMPT.format(
                        error=last_error,
                        previous_output=raw_output[:500],
                    )
                    messages.append({"role": "assistant", "content": raw_output})
                    messages.append({"role": "user", "content": repair_prompt})
                    
                    response = ctx.llm.chat(messages)
                    raw_output = response.content
        
        raise ValueError(f"Failed to compile prompt after {self.MAX_RETRIES + 1} attempts: {last_error}")
    
    def _extract_json(self, text: str) -> dict[str, Any]:
        """
        Extract JSON from LLM response.
        
        Handles markdown code blocks and bare JSON.
        """
        # Try to extract from code block
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if json_match:
            return json.loads(json_match.group(1).strip())
        
        # Try bare JSON (find first { to last })
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
        
        # Last resort: try whole text
        return json.loads(text.strip())
    
    def _build_from_llm_output(
        self,
        ctx: "AgentContext",
        user_input: str,
        data: dict[str, Any],
    ) -> tuple[PromptSpec, ToolPlan]:
        """
        Build PromptSpec and ToolPlan from LLM JSON output.
        
        Validates and normalizes the output.
        """
        now = ctx.now()
        
        # Generate deterministic market ID
        market_id = data.get("market_id") or self._generate_market_id(user_input)
        
        # Build resolution window
        window_data = data.get("resolution_window", {})
        resolution_window = ResolutionWindow(
            start=self._parse_datetime(window_data.get("start"), now),
            end=self._parse_datetime(window_data.get("end"), now + timedelta(days=7)),
        )
        
        # Build resolution deadline
        resolution_deadline = self._parse_datetime(
            data.get("resolution_deadline"),
            resolution_window.end,
        )
        
        # Determine market type and possible outcomes
        market_type = data.get("market_type", "binary")
        possible_outcomes = data.get("possible_outcomes", ["YES", "NO"])
        if market_type == "multi_choice" and possible_outcomes == ["YES", "NO"]:
            # LLM indicated multi-choice but didn't provide outcomes; fall back to binary
            market_type = "binary"
        if market_type == "binary":
            possible_outcomes = ["YES", "NO"]

        # Build resolution rules
        rules_data = data.get("resolution_rules", [])
        if not rules_data:
            rules_data = self._default_resolution_rules(market_type)
        
        resolution_rules = ResolutionRules(
            rules=[
                ResolutionRule(
                    rule_id=r.get("rule_id", f"R_{i}"),
                    description=r.get("description", ""),
                    priority=r.get("priority", 0),
                )
                for i, r in enumerate(rules_data)
            ]
        )
        
        # Build allowed sources
        sources_data = data.get("allowed_sources", [])
        allowed_sources = [
            SourcePolicy(
                source_id=s.get("source_id", f"source_{i}"),
                kind=s.get("kind", "api"),
                allow=s.get("allow", True),
                min_provenance_tier=s.get("min_provenance_tier", 0),
            )
            for i, s in enumerate(sources_data)
        ]
        
        # Build dispute policy
        dispute_data = data.get("dispute_policy", {})
        dispute_policy = DisputePolicy(
            dispute_window_seconds=dispute_data.get("dispute_window_seconds", 86400),
            allow_challenges=dispute_data.get("allow_challenges", True),
        )
        
        # Build MarketSpec
        market = MarketSpec(
            market_id=market_id,
            question=data.get("question", user_input),
            event_definition=data.get("event_definition", user_input),
            timezone=data.get("timezone", "UTC"),
            resolution_deadline=resolution_deadline,
            resolution_window=resolution_window,
            resolution_rules=resolution_rules,
            allowed_sources=allowed_sources,
            min_provenance_tier=data.get("min_provenance_tier", 0),
            dispute_policy=dispute_policy,
            market_type=market_type,
            possible_outcomes=possible_outcomes,
        )
        
        # Build data requirements
        reqs_data = data.get("data_requirements", [])
        data_requirements = []
        for i, req_data in enumerate(reqs_data):
            # Build source targets
            targets_data = req_data.get("source_targets", [])
            source_targets = [
                SourceTarget(
                    source_id=t.get("source_id", "http"),
                    uri=t.get("uri", ""),
                    method=t.get("method", "GET"),
                    expected_content_type=t.get("expected_content_type", "json"),
                    headers=t.get("headers", {}),
                    params=t.get("params", {}),
                    operation=t.get("operation"),
                    search_query=t.get("search_query"),
                )
                for t in targets_data
                if t.get("uri")  # Only include targets with URIs
            ]
            
            # Must have at least one source target
            if not source_targets:
                continue
            
            # Build selection policy
            sel_data = req_data.get("selection_policy", {})
            selection_policy = SelectionPolicy(
                strategy=sel_data.get("strategy", "single_best"),
                min_sources=sel_data.get("min_sources", 1),
                max_sources=sel_data.get("max_sources", 3),
                quorum=sel_data.get("quorum", 1),
            )
            
            data_requirements.append(
                DataRequirement(
                    requirement_id=req_data.get("requirement_id", f"req_{i:03d}"),
                    description=req_data.get("description", ""),
                    source_targets=source_targets,
                    selection_policy=selection_policy,
                    min_provenance_tier=req_data.get("min_provenance_tier", 0),
                    expected_fields=req_data.get("expected_fields"),
                )
            )
        
        # Build prediction semantics
        semantics_data = data.get("prediction_semantics", {})
        if not semantics_data:
            semantics_data = {
                "target_entity": data.get("target_entity", "unknown"),
                "predicate": data.get("predicate", "unknown"),
                "threshold": data.get("threshold"),
                "timeframe": data.get("timeframe"),
            }
        
        prediction_semantics = PredictionSemantics(
            target_entity=semantics_data.get("target_entity", "unknown"),
            predicate=semantics_data.get("predicate", "unknown"),
            threshold=semantics_data.get("threshold"),
            timeframe=semantics_data.get("timeframe"),
        )
        
        # Build PromptSpec
        prompt_spec = PromptSpec(
            market=market,
            prediction_semantics=prediction_semantics,
            data_requirements=data_requirements,
            forbidden_behaviors=data.get("forbidden_behaviors", []),
            created_at=now if not self.strict_mode else None,
            extra={
                "strict_mode": self.strict_mode,
                "compiler": "llm",
                "assumptions": data.get("assumptions", []),
                "confidence_policy": data.get("confidence_policy", {}),
            },
        )
        
        # Build ToolPlan
        tool_plan = ToolPlan(
            plan_id=f"plan_{market_id}",
            requirements=[req.requirement_id for req in data_requirements],
            sources=list({
                target.source_id
                for req in data_requirements
                for target in req.source_targets
            }),
            min_provenance_tier=market.min_provenance_tier,
            allow_fallbacks=True,
        )
        
        return prompt_spec, tool_plan
    
    def _generate_market_id(self, question: str) -> str:
        """Generate deterministic market ID from question."""
        hash_bytes = hashlib.sha256(question.encode()).digest()
        return f"mk_{hash_bytes[:8].hex()}"
    
    def _parse_datetime(self, value: str | datetime | None, default: datetime) -> datetime:
        """Parse datetime from string or return default."""
        if value is None:
            return default
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except (ValueError, AttributeError):
            return default
    
    def _default_resolution_rules(self, market_type: str = "binary") -> list[dict[str, Any]]:
        """Return default resolution rules."""
        rules = [
            {"rule_id": "R_VALIDITY", "description": "Check if evidence is sufficient and valid", "priority": 100},
            {"rule_id": "R_CONFLICT", "description": "Handle conflicting evidence from multiple sources", "priority": 90},
        ]
        if market_type == "multi_choice":
            rules.append({"rule_id": "R_MULTI_CHOICE", "description": "Match evidence against enumerated outcomes and select the best match", "priority": 80})
        else:
            rules.append({"rule_id": "R_BINARY_DECISION", "description": "Map evidence to YES/NO outcome", "priority": 80})
        rules.extend([
            {"rule_id": "R_CONFIDENCE", "description": "Assign confidence score based on evidence quality", "priority": 70},
            {"rule_id": "R_INVALID_FALLBACK", "description": "Return INVALID if resolution is impossible", "priority": 0},
        ])
        return rules
