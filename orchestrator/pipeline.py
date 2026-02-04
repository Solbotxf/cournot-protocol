"""
Module 09A - Pipeline Integration

Deterministic, testable, in-process pipeline runner composing Modules 04-08.

Key features:
- Registry-based agent selection (no direct instantiation in production)
- Fail-closed in production when missing required capabilities
- Configurable step overrides
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from agents import AgentContext, AgentStep, get_registry
from agents.base import AgentCapability, AgentResult

from core.por.por_bundle import PoRBundle
from core.por.proof_of_reasoning import PoRRoots, build_por_bundle, compute_roots
from core.por.reasoning_trace import ReasoningTrace
from core.schemas.evidence import EvidenceBundle
from core.schemas.prompts import PromptSpec
from core.schemas.transport import ToolExecutionLog, ToolPlan
from core.schemas.verdict import DeterministicVerdict
from core.schemas.verification import CheckResult, VerificationResult

from orchestrator.sop_executor import PipelineState, SOPExecutor, make_step


logger = logging.getLogger(__name__)


# Map pipeline steps to AgentsConfig attribute names
_STEP_TO_AGENT_KEY = {
    AgentStep.PROMPT_ENGINEER: "prompt_engineer",
    AgentStep.COLLECTOR: "collector",
    AgentStep.AUDITOR: "auditor",
    AgentStep.JUDGE: "judge",
    AgentStep.SENTINEL: "sentinel",
}


# =============================================================================
# Execution Mode
# =============================================================================

class ExecutionMode(str, Enum):
    """Pipeline execution mode."""
    PRODUCTION = "production"  # Fail closed on missing capabilities
    DEVELOPMENT = "development"  # Use fallbacks when possible
    TEST = "test"  # Use mock/deterministic agents


# =============================================================================
# Step Overrides
# =============================================================================

@dataclass
class StepOverrides:
    """
    Override specific agents for each step.
    
    If set, bypasses registry selection for that step.
    """
    prompt_engineer: Optional[str] = None  # Agent name override
    collector: Optional[str] = None
    auditor: Optional[str] = None
    judge: Optional[str] = None
    sentinel: Optional[str] = None


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class PipelineConfig:
    """Configuration for pipeline execution."""
    
    # Execution mode
    mode: ExecutionMode = ExecutionMode.DEVELOPMENT
    
    # Strict mode settings
    strict_mode: bool = True
    deterministic_timestamps: bool = True
    
    # Feature flags
    enable_replay: bool = False
    enable_sentinel_verify: bool = True
    
    # Timeouts
    max_runtime_s: int = 60
    
    # Step overrides (agent names)
    step_overrides: StepOverrides = field(default_factory=StepOverrides)
    
    # Required capabilities per mode
    require_llm: bool = False  # Set True to require LLM agents
    require_network: bool = False  # Set True to require network access
    
    # Audit evidence context limits (to stay under model context, e.g. 128K tokens)
    max_audit_evidence_chars: int = 300_000
    max_audit_evidence_items: int = 100
    
    # Debug
    debug: bool = False
    
    @property
    def is_production(self) -> bool:
        return self.mode == ExecutionMode.PRODUCTION
    
    @property
    def allow_fallbacks(self) -> bool:
        return self.mode in (ExecutionMode.DEVELOPMENT, ExecutionMode.TEST)


# =============================================================================
# Run Result
# =============================================================================

@dataclass
class RunResult:
    """Complete result of a pipeline run."""
    prompt_spec: Optional[PromptSpec] = None
    tool_plan: Optional[ToolPlan] = None
    evidence_bundle: Optional[EvidenceBundle] = None
    execution_log: Optional[ToolExecutionLog] = None
    audit_trace: Optional[ReasoningTrace] = None
    audit_verification: Optional[VerificationResult] = None
    verdict: Optional[DeterministicVerdict] = None
    judge_verification: Optional[VerificationResult] = None
    por_bundle: Optional[PoRBundle] = None
    roots: Optional[PoRRoots] = None
    sentinel_verification: Optional[VerificationResult] = None
    challenges: Optional[list[Any]] = None
    ok: bool = False
    checks: list[CheckResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    
    @property
    def market_id(self) -> Optional[str]:
        return self.verdict.market_id if self.verdict else None
    
    @property
    def outcome(self) -> Optional[str]:
        return self.verdict.outcome if self.verdict else None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "market_id": self.market_id,
            "outcome": self.outcome,
            "confidence": self.verdict.confidence if self.verdict else None,
            "errors": self.errors,
            "check_count": len(self.checks),
            "has_por_bundle": self.por_bundle is not None,
        }


# =============================================================================
# PoR Package (for Sentinel)
# =============================================================================

@dataclass
class PoRPackage:
    """Internal container for sentinel verification."""
    bundle: PoRBundle
    prompt_spec: PromptSpec
    tool_plan: Optional[ToolPlan]
    evidence: EvidenceBundle
    trace: ReasoningTrace
    verdict: DeterministicVerdict


# =============================================================================
# Capability Error
# =============================================================================

class CapabilityError(Exception):
    """Raised when required capabilities are not available."""
    pass


# =============================================================================
# Pipeline Class
# =============================================================================

class Pipeline:
    """
    Main pipeline runner orchestrating the full resolution flow.
    
    Uses registry-based agent selection with configurable overrides.
    Fails closed in production mode when capabilities are missing.
    """
    
    def __init__(
        self,
        *,
        config: Optional[PipelineConfig] = None,
        context: Optional[AgentContext] = None,
    ):
        """
        Initialize pipeline.
        
        Args:
            config: Pipeline configuration
            context: Agent context (created if not provided)
        """
        self.config = config or PipelineConfig()
        self._context = context
        self._executor = SOPExecutor(stop_on_error=True)
        self._registry = get_registry()
    
    def _get_context(self) -> AgentContext:
        """Get or create agent context."""
        if self._context is not None:
            return self._context
        return AgentContext.create_minimal()
    
    def _check_capabilities(self, ctx: AgentContext) -> list[str]:
        """
        Check required capabilities are available.
        
        Returns list of missing capability errors.
        """
        errors = []
        
        logger.debug(f"checking context {ctx}")
        if self.config.require_llm and ctx.llm is None:
            errors.append("LLM capability required but not available")
        
        if self.config.require_network and ctx.http is None:
            errors.append("NETWORK capability required but not available")
        
        return errors

    def _resolve_ctx(self, ctx: AgentContext, step: AgentStep) -> AgentContext:
        """Return ctx with per-agent LLM override applied, if configured."""
        if ctx.config is None:
            return ctx
        agent_key = _STEP_TO_AGENT_KEY.get(step)
        if not agent_key:
            return ctx
        agent_cfg = getattr(ctx.config.agents, agent_key, None)
        if not agent_cfg or agent_cfg.llm_override is None:
            return ctx
        return ctx.with_llm_override(agent_cfg.llm_override)

    def _select_agent(
        self,
        step: AgentStep,
        override_name: Optional[str],
        ctx: AgentContext,
    ) -> Any:
        """
        Select an agent for a step using registry.
        
        Args:
            step: The pipeline step
            override_name: Optional agent name override
            ctx: Agent context
        
        Returns:
            Selected agent instance
        
        Raises:
            CapabilityError: If production mode and no suitable agent found
        """
        try:
            # If override specified, get that specific agent
            if override_name:
                if self._registry.has_agent(override_name):
                    return self._registry.get_agent_by_name(override_name, ctx)
                elif self.config.is_production:
                    raise CapabilityError(f"Specified agent '{override_name}' not found for {step}")
                else:
                    logger.warning(f"Override agent '{override_name}' not found, using default selection")
            
            # Get available agents for this step
            agents = self._registry.list_agents(step)

            if not agents:
                if self.config.is_production:
                    raise CapabilityError(f"No agents registered for {step}")
                return None
            
            # In production, check capabilities and fail closed
            if self.config.is_production:
                # Try to get a suitable agent
                for agent_info in agents:
                    # Skip fallbacks in production if we have primary agents
                    if agent_info.is_fallback and any(not a.is_fallback for a in agents):
                        continue
                    
                    # Check if agent's required capabilities are available
                    if AgentCapability.LLM in agent_info.capabilities and ctx.llm is None:
                        continue
                    if AgentCapability.NETWORK in agent_info.capabilities and ctx.http is None:
                        continue
                    
                    # This agent is suitable
                    return agent_info.factory(ctx)
                
                # No suitable agent found in production
                raise CapabilityError(
                    f"No agent with available capabilities for {step}"
                )
            
            # Development/test mode: use best available with fallbacks
            for agent_info in agents:
                # Skip agents requiring unavailable capabilities
                if AgentCapability.LLM in agent_info.capabilities and ctx.llm is None:
                    continue
                if AgentCapability.NETWORK in agent_info.capabilities and ctx.http is None:
                    continue
                return agent_info.factory(ctx)
            
            # Use fallback even if capabilities don't match
            if self.config.allow_fallbacks:
                fallbacks = [a for a in agents if a.is_fallback]
                if fallbacks:
                    return fallbacks[0].factory(ctx)
            
            return None
            
        except ValueError as e:
            if self.config.is_production:
                raise CapabilityError(str(e))
            logger.warning(f"Agent selection failed: {e}")
            return None
    
    def run_from_prompt(
        self,
        prompt_spec: PromptSpec,
        tool_plan: ToolPlan,
    ) -> RunResult:
        """
        Execute pipeline steps 2-5 from a pre-built PromptSpec and ToolPlan.

        Skips prompt compilation (step 1) and sentinel verification (step 6).
        Runs: evidence collection → audit → judge → build PoR bundle.

        Args:
            prompt_spec: Compiled prompt specification
            tool_plan: Tool execution plan

        Returns:
            RunResult with all artifacts from steps 2-5
        """
        ctx = self._get_context()

        # Pre-populate state with prompt step output
        state = PipelineState(user_input="")
        state.context = ctx
        state.prompt_spec = prompt_spec
        state.tool_plan = tool_plan

        # Pass audit evidence limits to agents via context
        ctx.extra["max_audit_evidence_chars"] = self.config.max_audit_evidence_chars
        ctx.extra["max_audit_evidence_items"] = self.config.max_audit_evidence_items

        # Build only steps 2-5
        overrides = self.config.step_overrides
        steps = [
            make_step("evidence_collection",
                lambda s: self._step_evidence_collection(s, overrides.collector)),
            make_step("audit",
                lambda s: self._step_audit(s, overrides.auditor)),
            make_step("judge",
                lambda s: self._step_judge(s, overrides.judge)),
            make_step("build_por_bundle", self._step_build_por_bundle),
        ]

        state = self._executor.execute(steps, state)
        return self._state_to_result(state)

    def run(self, user_input: str) -> RunResult:
        """Execute the full pipeline."""
        ctx = self._get_context()
        
        # Check required capabilities in production mode
        if self.config.is_production:
            cap_errors = self._check_capabilities(ctx)
            if cap_errors:
                return RunResult(
                    ok=False,
                    errors=cap_errors,
                    checks=[CheckResult.failed("capability_check", e) for e in cap_errors],
                )
        
        state = PipelineState(user_input=user_input)
        state.context = ctx  # Store context in state for steps
        
        # Pass audit evidence limits to agents (e.g. Auditor) via context
        ctx.extra["max_audit_evidence_chars"] = self.config.max_audit_evidence_chars
        ctx.extra["max_audit_evidence_items"] = self.config.max_audit_evidence_items
        
        steps = self._build_steps(ctx)
        state = self._executor.execute(steps, state)
        
        return self._state_to_result(state)
    
    def _build_steps(self, ctx: AgentContext) -> list:
        """Build pipeline steps based on config."""
        steps = []
        overrides = self.config.step_overrides
        
        # Step 1: Prompt compilation
        steps.append(make_step("prompt_compilation", 
            lambda s: self._step_prompt_compilation(s, overrides.prompt_engineer)))
        
        # Step 2: Evidence collection
        steps.append(make_step("evidence_collection",
            lambda s: self._step_evidence_collection(s, overrides.collector)))
        
        # Step 3: Audit
        steps.append(make_step("audit",
            lambda s: self._step_audit(s, overrides.auditor)))
        
        # Step 4: Judge
        steps.append(make_step("judge",
            lambda s: self._step_judge(s, overrides.judge)))
        
        # Step 5: Build PoR bundle
        steps.append(make_step("build_por_bundle", self._step_build_por_bundle))
        
        # Step 6: Sentinel verify (optional)
        if self.config.enable_sentinel_verify:
            steps.append(make_step("sentinel_verify",
                lambda s: self._step_sentinel_verify(s, overrides.sentinel)))
        
        return steps
    
    def _step_prompt_compilation(
        self,
        state: PipelineState,
        override: Optional[str],
    ) -> PipelineState:
        """Step 1: Compile user input into PromptSpec and ToolPlan."""
        ctx = self._resolve_ctx(state.context, AgentStep.PROMPT_ENGINEER)
        
        try:
            agent = self._select_agent(AgentStep.PROMPT_ENGINEER, override, ctx)
            if agent is None:
                state.add_error("No prompt engineer agent available")
                return state
            
            logger.info(f"Running prompt engineer: {agent.name}")
            result: AgentResult = agent.run(ctx, state.user_input)
            
            if not result.success:
                state.add_error(f"Prompt compilation failed: {result.error}")
                return state
            
            prompt_spec, tool_plan = result.output
            
            # Validate strict mode
            if self.config.strict_mode:
                if not prompt_spec.extra.get("strict_mode", False):
                    state.add_check(CheckResult.failed(
                        "strict_mode_check",
                        "PromptSpec.extra['strict_mode'] must be True",
                    ))
            
            # Validate timestamps
            if self.config.deterministic_timestamps and prompt_spec.created_at is not None:
                state.add_check(CheckResult.failed(
                    "timestamp_determinism_prompt",
                    "PromptSpec.created_at must be None for determinism",
                ))
            
            state.prompt_spec = prompt_spec
            state.tool_plan = tool_plan
            state.add_check(CheckResult.passed(
                "prompt_compilation",
                f"Compiled prompt for {prompt_spec.market.market_id}",
            ))
            
        except CapabilityError as e:
            state.add_error(str(e))
        except Exception as e:
            logger.exception("Prompt compilation failed")
            state.add_error(f"Prompt compilation error: {e}")
        
        return state
    
    def _step_evidence_collection(
        self,
        state: PipelineState,
        override: Optional[str],
    ) -> PipelineState:
        """Step 2: Collect evidence."""
        ctx = self._resolve_ctx(state.context, AgentStep.COLLECTOR)
        # ctx.logger.debug(f"getting llm client: {ctx.llm}")
        
        if not state.prompt_spec or not state.tool_plan:
            state.add_error("Cannot collect evidence: missing prompt_spec or tool_plan")
            return state
        
        try:
            agent = self._select_agent(AgentStep.COLLECTOR, override, ctx)
            if agent is None:
                state.add_error("No collector agent available")
                return state
            
            logger.info(f"Running collector: {agent.name} with prompt_spec {state.prompt_spec} and tool plan {state.tool_plan}")
            result: AgentResult = agent.run(ctx, state.prompt_spec, state.tool_plan)
            
            if not result.success:
                state.add_error(f"Evidence collection failed: {result.error}")
                return state
            
            evidence_bundle, execution_log = result.output
            state.evidence_bundle = evidence_bundle
            state.execution_log = execution_log
            
            state.add_check(CheckResult.passed(
                "evidence_collection",
                f"Collected {len(evidence_bundle.items)} evidence items",
            ))
            
        except CapabilityError as e:
            state.add_error(str(e))
        except Exception as e:
            logger.exception("Evidence collection failed")
            state.add_error(f"Evidence collection error: {e}")
        
        return state
    
    def _step_audit(
        self,
        state: PipelineState,
        override: Optional[str],
    ) -> PipelineState:
        """Step 3: Produce reasoning trace."""
        ctx = self._resolve_ctx(state.context, AgentStep.AUDITOR)
        
        if not state.prompt_spec or not state.evidence_bundle:
            state.add_error("Cannot audit: missing prompt_spec or evidence_bundle")
            return state
        
        try:
            agent = self._select_agent(AgentStep.AUDITOR, override, ctx)
            if agent is None:
                state.add_error("No auditor agent available")
                return state
            
            logger.info(f"Running auditor: {agent.name}")
            result: AgentResult = agent.run(ctx, state.prompt_spec, state.evidence_bundle)
            
            if not result.success:
                state.add_error(f"Audit failed: {result.error}")
                return state
            
            trace = result.output
            state.audit_trace = trace
            state.audit_verification = result.verification
            
            if result.verification:
                state.merge_verification(result.verification)
            
            state.add_check(CheckResult.passed(
                "audit",
                f"Generated reasoning trace with {trace.step_count} steps",
            ))
            
        except CapabilityError as e:
            state.add_error(str(e))
        except Exception as e:
            logger.exception("Audit failed")
            state.add_error(f"Audit error: {e}")
        
        return state
    
    def _step_judge(
        self,
        state: PipelineState,
        override: Optional[str],
    ) -> PipelineState:
        """Step 4: Produce verdict."""
        ctx = self._resolve_ctx(state.context, AgentStep.JUDGE)
        
        if not state.prompt_spec or not state.evidence_bundle or not state.audit_trace:
            state.add_error("Cannot judge: missing required artifacts")
            return state
        
        try:
            agent = self._select_agent(AgentStep.JUDGE, override, ctx)
            if agent is None:
                state.add_error("No judge agent available")
                return state
            
            logger.info(f"Running judge: {agent.name}")
            result: AgentResult = agent.run(
                ctx, state.prompt_spec, state.evidence_bundle, state.audit_trace
            )
            
            if not result.success:
                state.add_error(f"Judge failed: {result.error}")
                return state
            
            verdict = result.output
            state.verdict = verdict
            state.judge_verification = result.verification
            
            if result.verification:
                state.merge_verification(result.verification)
            
            state.add_check(CheckResult.passed(
                "judge",
                f"Verdict: {verdict.outcome} ({verdict.confidence:.0%} confidence)",
            ))
            
        except CapabilityError as e:
            state.add_error(str(e))
        except Exception as e:
            logger.exception("Judge failed")
            state.add_error(f"Judge error: {e}")
        
        return state
    
    def _step_build_por_bundle(self, state: PipelineState) -> PipelineState:
        """Step 5: Build PoR bundle."""
        if not all([state.prompt_spec, state.evidence_bundle, state.audit_trace, state.verdict]):
            state.add_error("Cannot build PoR bundle: missing artifacts")
            return state
        
        try:
            roots = compute_roots(
                state.prompt_spec,
                state.evidence_bundle,
                state.audit_trace,
                state.verdict,
            )
            state.roots = roots
            
            por_bundle = build_por_bundle(
                state.prompt_spec,
                state.evidence_bundle,
                state.audit_trace,
                state.verdict,
                include_por_root=True,
                metadata={"pipeline_version": "09A", "mode": self.config.mode.value},
            )
            state.por_bundle = por_bundle
            
            state.add_check(CheckResult.passed(
                "por_bundle_built",
                f"Built bundle with root {roots.por_root[:18]}...",
            ))
            
        except Exception as e:
            logger.exception("PoR bundle build failed")
            state.add_error(f"PoR bundle error: {e}")
        
        return state
    
    def _step_sentinel_verify(
        self,
        state: PipelineState,
        override: Optional[str],
    ) -> PipelineState:
        """Step 6: Verify with Sentinel."""
        ctx = self._resolve_ctx(state.context, AgentStep.SENTINEL)
        
        if not all([state.por_bundle, state.prompt_spec, state.evidence_bundle, 
                    state.audit_trace, state.verdict]):
            state.add_error("Cannot verify: missing artifacts")
            return state
        
        try:
            agent = self._select_agent(AgentStep.SENTINEL, override, ctx)
            if agent is None:
                if self.config.is_production:
                    state.add_error("No sentinel agent available")
                else:
                    state.add_check(CheckResult.warning(
                        "sentinel_skip",
                        "Sentinel verification skipped (no agent available)",
                    ))
                return state
            
            logger.info(f"Running sentinel: {agent.name}")
            
            # Build proof bundle for sentinel
            from agents.sentinel import build_proof_bundle
            proof_bundle = build_proof_bundle(
                state.prompt_spec,
                state.tool_plan,
                state.evidence_bundle,
                state.audit_trace,
                state.verdict,
                state.execution_log,
            )
            
            result: AgentResult = agent.run(ctx, proof_bundle)
            
            if not result.success:
                state.add_error(f"Sentinel verification failed: {result.error}")
                return state
            
            verification_result, report = result.output
            state.sentinel_verification = verification_result
            
            if verification_result:
                state.merge_verification(verification_result)
            
            if report.verified:
                state.add_check(CheckResult.passed(
                    "sentinel_verify",
                    f"Sentinel verified: {report.passed_checks}/{report.total_checks} checks passed",
                ))
            else:
                state.add_check(CheckResult.failed(
                    "sentinel_verify",
                    f"Sentinel verification failed: {report.errors}",
                ))
            
        except CapabilityError as e:
            state.add_error(str(e))
        except Exception as e:
            logger.exception("Sentinel verification failed")
            state.add_error(f"Sentinel error: {e}")
        
        return state
    
    def _state_to_result(self, state: PipelineState) -> RunResult:
        """Convert final PipelineState to RunResult."""
        ok = state.ok and state.por_bundle is not None
        
        return RunResult(
            prompt_spec=state.prompt_spec,
            tool_plan=state.tool_plan,
            evidence_bundle=state.evidence_bundle,
            execution_log=getattr(state, 'execution_log', None),
            audit_trace=state.audit_trace,
            audit_verification=state.audit_verification,
            verdict=state.verdict,
            judge_verification=state.judge_verification,
            por_bundle=state.por_bundle,
            roots=state.roots,
            sentinel_verification=state.sentinel_verification,
            challenges=state.challenges,
            ok=ok,
            checks=state.checks,
            errors=state.errors,
        )


# =============================================================================
# Factory Functions
# =============================================================================

def create_pipeline(
    *,
    mode: ExecutionMode = ExecutionMode.DEVELOPMENT,
    context: Optional[AgentContext] = None,
    strict_mode: bool = True,
    enable_sentinel: bool = True,
    step_overrides: Optional[StepOverrides] = None,
    require_llm: bool = False,
    require_network: bool = False,
) -> Pipeline:
    """
    Convenience function to create a pipeline with common configuration.
    
    Args:
        mode: Execution mode (production/development/test)
        context: Optional agent context
        strict_mode: Require strict mode in prompts
        enable_sentinel: Enable sentinel verification
        step_overrides: Override agents for specific steps
        require_llm: Require LLM capability
        require_network: Require network capability
    
    Returns:
        Configured Pipeline instance
    """
    config = PipelineConfig(
        mode=mode,
        strict_mode=strict_mode,
        enable_sentinel_verify=enable_sentinel,
        step_overrides=step_overrides or StepOverrides(),
        require_llm=require_llm,
        require_network=require_network,
    )
    return Pipeline(config=config, context=context)


def create_production_pipeline(
    context: AgentContext,
    *,
    require_llm: bool = True,
    require_network: bool = True,
) -> Pipeline:
    """
    Create a production pipeline that fails closed on missing capabilities.
    
    Args:
        context: Agent context with required capabilities
        require_llm: Require LLM capability (default True)
        require_network: Require network capability (default True)
    
    Returns:
        Production-configured Pipeline
    """
    return create_pipeline(
        mode=ExecutionMode.PRODUCTION,
        context=context,
        require_llm=require_llm,
        require_network=require_network,
    )


def create_test_pipeline(
    mock_responses: Optional[dict[str, Any]] = None,
) -> Pipeline:
    """
    Create a test pipeline using mock/deterministic agents.
    
    Args:
        mock_responses: Optional mock responses for collector
    
    Returns:
        Test-configured Pipeline with mock context
    """
    # PromptEngineerLLM requires an LLM client; provide a mock
    mock_llm_response = json.dumps({
        "market_id": "mk_test_pipeline",
        "question": "Will BTC be above $100k?",
        "event_definition": "price(BTC_USD) > 100000",
        "target_entity": "bitcoin",
        "predicate": "price above threshold",
        "threshold": "100000",
        "resolution_window": {
            "start": "2025-01-01T00:00:00Z",
            "end": "2025-12-31T23:59:59Z",
        },
        "resolution_deadline": "2025-12-31T23:59:59Z",
        "data_requirements": [{
            "requirement_id": "req_001",
            "description": "Get BTC price",
            "source_targets": [{
                "source_id": "coingecko",
                "uri": "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd",
                "method": "GET",
                "expected_content_type": "json",
            }],
            "selection_policy": {
                "strategy": "single_best",
                "min_sources": 1,
                "max_sources": 1,
                "quorum": 1,
            },
        }],
        "resolution_rules": [
            {"rule_id": "R_VALIDITY", "description": "Check validity", "priority": 100},
            {"rule_id": "R_THRESHOLD", "description": "Compare to threshold", "priority": 80},
            {"rule_id": "R_INVALID_FALLBACK", "description": "Fallback", "priority": 0},
        ],
        "allowed_sources": [
            {"source_id": "coingecko", "kind": "api", "allow": True},
        ],
    })
    ctx = AgentContext.create_mock(llm_responses=[mock_llm_response])

    # Configure step overrides to use deterministic agents
    overrides = StepOverrides(
        prompt_engineer="PromptEngineerLLM",
        collector="CollectorMock",
        auditor="AuditorRuleBased",
        judge="JudgeRuleBased",
        sentinel="SentinelStrict",
    )

    return create_pipeline(
        mode=ExecutionMode.TEST,
        context=ctx,
        step_overrides=overrides,
    )
