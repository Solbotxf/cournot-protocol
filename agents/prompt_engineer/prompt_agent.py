"""
Module 04 - Prompt Engineer Agent.

Compiles raw user text into a deterministic, strictly validated PromptSpec
and ToolPlan where:
- The prediction is unambiguous
- Evidence sources are explicit (URLs/endpoints)
- Resolution rules are deterministic
- Downstream output is constrained to YES/NO/INVALID with confidence

This module is pluggable: the compiler can be replaced without changing
orchestrator/collector contracts.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from agents.base_agent import AgentContext, BaseAgent
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
from core.schemas.errors import PromptCompilationException

from .input_parser import InputParser
from .models import (
    CompilationResult,
    NormalizedUserRequest,
    PromptModule,
    generate_deterministic_id,
    generate_requirement_id,
)
from .source_builders import SourceTargetBuilder


class StrictPromptCompilerV1:
    """
    Default strict prompt compiler implementation.
    
    Compiles user input into a fully deterministic PromptSpec with
    explicit source targets and resolution rules.
    """
    
    module_id: str = "strict_compiler"
    version: str = "1.0.0"
    
    DEFAULT_CONFIDENCE_POLICY = {
        "min_confidence_for_yesno": 0.55,
        "default_confidence": 0.7,
        "single_source_max": 0.85,
        "quorum_bonus": 0.1,
    }
    DEFAULT_WINDOW_DAYS = 7
    
    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
    
    def compile(
        self,
        user_input: str,
        *,
        ctx: Optional[AgentContext] = None
    ) -> tuple[PromptSpec, ToolPlan]:
        """Compile user input into PromptSpec and ToolPlan."""
        normalized = self._parse_input(user_input)
        self._enforce_strictness(normalized)
        market_spec = self._build_market_spec(normalized)
        data_requirements = self._build_data_requirements(normalized)
        prediction_semantics = self._build_prediction_semantics(normalized)
        tool_plan = self._build_tool_plan(normalized, data_requirements, market_spec.market_id)
        
        prompt_spec = PromptSpec(
            market=market_spec,
            prediction_semantics=prediction_semantics,
            data_requirements=data_requirements,
            output_schema_ref="core.schemas.verdict.DeterministicVerdict",
            forbidden_behaviors=[
                "Do not hallucinate evidence",
                "Do not use sources outside allowed list",
                "Do not output ambiguous outcomes",
                "Do not exceed confidence bounds",
            ],
            created_at=None,
            tool_plan=tool_plan,
            extra={
                "strict_mode": True,
                "compiler": {"module_id": self.module_id, "version": self.version},
                "assumptions": normalized.assumptions,
                "confidence_policy": normalized.confidence_policy or self.DEFAULT_CONFIDENCE_POLICY,
                "id_scheme": "deterministic_sha256",
            }
        )
        return prompt_spec, tool_plan
    
    def _parse_input(self, user_input: str) -> NormalizedUserRequest:
        """Parse raw input into normalized request."""
        normalized = NormalizedUserRequest(question=user_input.strip(), raw_input=user_input)
        
        normalized.threshold = InputParser.extract_threshold(user_input)
        normalized.source_preferences = InputParser.extract_sources(user_input)
        
        date_str = InputParser.extract_date(user_input)
        tz = InputParser.extract_timezone(user_input)
        normalized.time_config = {"timezone": tz, "window_start": None, "window_end": None, "date_str": date_str}
        
        if date_str:
            try:
                normalized.time_config["window_end"] = self._parse_date(date_str, tz)
            except ValueError:
                normalized.assumptions.append(f"Could not parse date '{date_str}', using default window")
        
        if not normalized.time_config.get("window_end"):
            now = datetime.now(timezone.utc)
            normalized.time_config["window_end"] = now + timedelta(days=self.DEFAULT_WINDOW_DAYS)
            normalized.assumptions.append(f"No explicit timeframe; defaulting to {self.DEFAULT_WINDOW_DAYS}-day window")
        
        if not normalized.time_config.get("window_start"):
            normalized.time_config["window_start"] = datetime.now(timezone.utc)
        
        entity = InputParser.detect_entity(user_input)
        predicate = InputParser.detect_predicate(user_input)
        
        if normalized.threshold:
            normalized.event_definition = (
                f"{entity} {predicate.replace('_', ' ')} {normalized.threshold} "
                f"by {normalized.time_config['window_end'].isoformat()}"
            )
        else:
            normalized.event_definition = (
                f"{entity} {predicate.replace('_', ' ')} by {normalized.time_config['window_end'].isoformat()}"
            )
        
        for url in InputParser.extract_urls(user_input):
            normalized.source_targets.append(SourceTargetBuilder.build_generic_http_target(url))
        
        self._build_source_targets_from_preferences(normalized, entity)
        normalized.confidence_policy = dict(self.DEFAULT_CONFIDENCE_POLICY)
        normalized.resolution_policy = {
            "invalid_conditions": [
                "Required evidence missing",
                "Evidence below minimum provenance tier",
                "Conflicting evidence without quorum resolution",
            ],
            "conflict_policy": "prefer_higher_provenance",
        }
        return normalized
    
    def _parse_date(self, date_str: str, tz: str) -> datetime:
        """Parse a date string into datetime."""
        try:
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            return dt.astimezone(timezone.utc)
        except ValueError:
            pass
        
        for fmt in ["%Y-%m-%d", "%B %d, %Y", "%B %d %Y", "%b %d, %Y", "%b %d %Y"]:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.replace(hour=23, minute=59, second=59, tzinfo=timezone.utc)
            except ValueError:
                continue
        raise ValueError(f"Cannot parse date: {date_str}")
    
    def _build_source_targets_from_preferences(self, normalized: NormalizedUserRequest, entity: str) -> None:
        """Build source targets from detected preferences."""
        preferences = normalized.source_preferences
        date_str = normalized.time_config.get("date_str", "")
        
        if not preferences and entity.upper() in ["BTC", "ETH", "SOL", "XRP", "ADA", "DOT"]:
            preferences = ["exchange"]
            normalized.assumptions.append(f"Detected crypto asset {entity}, adding exchange sources")
        
        for pref in preferences:
            if pref == "exchange":
                normalized.source_targets.append(SourceTargetBuilder.build_coinbase_target(entity, date_str or ""))
                normalized.source_targets.append(SourceTargetBuilder.build_binance_target(entity, date_str or ""))
            elif pref == "polymarket":
                normalized.source_targets.append(SourceTarget(
                    source_id="polymarket", uri="https://gamma-api.polymarket.com/markets",
                    method="GET", expected_content_type="json",
                    params={"q": normalized.question[:100]}, notes="Polymarket market search"
                ))
            elif pref == "government":
                normalized.source_targets.append(SourceTarget(
                    source_id="government", uri="https://api.bls.gov/publicAPI/v2/timeseries/data/",
                    method="POST", expected_content_type="json", notes="BLS API for economic data"
                ))
        
        if not normalized.source_targets:
            normalized.source_targets.append(SourceTarget(
                source_id="web", uri="https://www.google.com/search", method="GET",
                expected_content_type="html", params={"q": normalized.question[:200]},
                notes="Web search fallback"
            ))
            normalized.assumptions.append("No specific sources identified, added web search fallback")
    
    def _enforce_strictness(self, normalized: NormalizedUserRequest) -> None:
        """Enforce strictness requirements."""
        errors = []
        if not normalized.event_definition:
            errors.append("Event definition could not be determined")
        if not normalized.time_config.get("window_end"):
            errors.append("Resolution timeframe could not be determined")
        window_end = normalized.time_config.get("window_end")
        if window_end and window_end < datetime.now(timezone.utc):
            errors.append("Resolution window end is in the past")
        if not normalized.source_targets:
            errors.append("No data sources could be identified")
        if errors:
            raise PromptCompilationException(
                f"Strictness validation failed: {'; '.join(errors)}",
                details={"errors": errors, "raw_input": normalized.raw_input, "assumptions": normalized.assumptions}
            )
    
    def _build_market_spec(self, normalized: NormalizedUserRequest) -> MarketSpec:
        """Build MarketSpec from normalized request."""
        window_start = normalized.time_config["window_start"]
        window_end = normalized.time_config["window_end"]
        market_id = generate_deterministic_id("mkt", normalized.event_definition, window_end.isoformat())
        
        rules = ResolutionRules(rules=[
            ResolutionRule(rule_id="validity_rule", description="If required evidence is missing or below provenance tier, outcome is INVALID", priority=100),
            ResolutionRule(rule_id="conflict_rule", description="If conflicting evidence, prefer higher provenance tier; apply quorum if same tier", priority=90),
            ResolutionRule(rule_id="binary_decision_rule", description="If event_definition evaluates True, outcome is YES; otherwise NO", priority=80),
            ResolutionRule(rule_id="confidence_rule", description="Confidence based on source tier and quorum: single-source capped, quorum adds bonus", priority=70),
        ])
        
        source_policies = []
        seen_sources = set()
        for target in normalized.source_targets:
            if target.source_id not in seen_sources:
                source_policies.append(SourcePolicy(
                    source_id=target.source_id, kind="api" if "api" in target.uri.lower() else "web",
                    allow=True, min_provenance_tier=0
                ))
                seen_sources.add(target.source_id)
        
        return MarketSpec(
            market_id=market_id, question=normalized.question, event_definition=normalized.event_definition,
            timezone=normalized.time_config.get("timezone", "UTC"), resolution_deadline=window_end,
            resolution_window=ResolutionWindow(start=window_start, end=window_end),
            resolution_rules=rules, allowed_sources=source_policies, min_provenance_tier=0,
            dispute_policy=DisputePolicy(dispute_window_seconds=86400, allow_challenges=True),
            metadata={"assumptions": normalized.assumptions, "threshold": normalized.threshold}
        )
    
    def _build_data_requirements(self, normalized: NormalizedUserRequest) -> list[DataRequirement]:
        """Build data requirements with explicit source targets."""
        requirements = []
        targets_by_source: dict[str, list[SourceTarget]] = {}
        for target in normalized.source_targets:
            targets_by_source.setdefault(target.source_id, []).append(target)
        
        for idx, (source_id, targets) in enumerate(targets_by_source.items()):
            req_id = generate_requirement_id(idx + 1)
            if len(targets) > 1:
                selection_policy = SelectionPolicy(strategy="fallback_chain", min_sources=1, max_sources=len(targets), tie_breaker="highest_provenance")
            else:
                selection_policy = SelectionPolicy(strategy="single_best", min_sources=1, max_sources=1, tie_breaker="highest_provenance")

            requirements.append(DataRequirement(
                requirement_id=req_id,
                description=f"Data from {source_id} source(s) for resolution",
                preferred_sources=[source_id],
                min_provenance_tier=0,
                source_targets=targets,
                selection_policy=selection_policy,
            ))
        return requirements
    
    def _build_prediction_semantics(self, normalized: NormalizedUserRequest) -> PredictionSemantics:
        """Build prediction semantics from normalized request."""
        return PredictionSemantics(
            target_entity=InputParser.detect_entity(normalized.raw_input),
            predicate=InputParser.detect_predicate(normalized.raw_input),
            threshold=normalized.threshold, timeframe=normalized.time_config.get("date_str")
        )
    
    def _build_tool_plan(self, normalized: NormalizedUserRequest, requirements: list[DataRequirement], market_id: str) -> ToolPlan:
        """Build tool execution plan."""
        sources = list({t.source_id for t in normalized.source_targets})
        plan_id = generate_deterministic_id("plan", market_id, "|".join(sources))
        return ToolPlan(
            plan_id=plan_id, requirements=[req.requirement_id for req in requirements],
            sources=sources, min_provenance_tier=0, allow_fallbacks=True,
            extra={"execution_mode": "sequential", "selection_hints": {req.requirement_id: req.selection_policy.strategy for req in requirements}}
        )


class PromptEngineerAgent(BaseAgent[str, tuple[PromptSpec, ToolPlan]]):
    """Prompt Engineer Agent - converts user input to structured PromptSpec."""
    
    agent_type: str = "prompt_engineer"
    agent_version: str = "1.0.0"
    
    def __init__(self, module: Optional[PromptModule] = None, context: Optional[AgentContext] = None, config: dict[str, Any] | None = None):
        super().__init__(context)
        self.module = module or StrictPromptCompilerV1(config)
        self.config = config or {}
    
    def run(self, input_data: str) -> tuple[PromptSpec, ToolPlan]:
        """Compile user input into PromptSpec and ToolPlan."""
        if not input_data or not input_data.strip():
            raise PromptCompilationException("Empty input provided", details={"input": input_data})
        return self.module.compile(input_data, ctx=self.context)
    
    def validate_input(self, input_data: str) -> bool:
        """Validate that input is non-empty string."""
        return bool(input_data and input_data.strip())
    
    def get_module_info(self) -> dict[str, str]:
        """Get information about the current module."""
        return {"module_id": self.module.module_id, "version": self.module.version}