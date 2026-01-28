"""
Module 09A - Pipeline Integration

Deterministic, testable, in-process pipeline runner composing Modules 04-08.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

from core.por.por_bundle import PoRBundle
from core.por.proof_of_reasoning import PoRRoots, build_por_bundle, compute_roots
from core.por.reasoning_trace import ReasoningTrace
from core.schemas.evidence import EvidenceBundle
from core.schemas.prompts import PromptSpec
from core.schemas.transport import ToolPlan
from core.schemas.verdict import DeterministicVerdict
from core.schemas.verification import CheckResult, VerificationResult

from orchestrator.sop_executor import PipelineState, SOPExecutor, make_step


# Agent Protocols (for dependency injection / mocking)

@runtime_checkable
class PromptEngineerProtocol(Protocol):
    def run(self, user_input: str, *, ctx: Optional[Any] = None) -> tuple[PromptSpec, ToolPlan]: ...

@runtime_checkable
class CollectorProtocol(Protocol):
    def collect(self, prompt_spec: PromptSpec, tool_plan: ToolPlan, *, ctx: Optional[Any] = None) -> EvidenceBundle: ...

@runtime_checkable
class AuditorProtocol(Protocol):
    def audit(self, prompt_spec: PromptSpec, evidence: EvidenceBundle, *, ctx: Optional[Any] = None) -> tuple[ReasoningTrace, VerificationResult]: ...

@runtime_checkable
class JudgeProtocol(Protocol):
    def judge(self, prompt_spec: PromptSpec, evidence: EvidenceBundle, trace: ReasoningTrace, *, ctx: Optional[Any] = None) -> tuple[DeterministicVerdict, VerificationResult]: ...

@runtime_checkable
class SentinelProtocol(Protocol):
    def verify(self, package: Any, *, mode: str = "verify") -> tuple[VerificationResult, list[Any]]: ...


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class PipelineConfig:
    """Configuration for pipeline execution."""
    strict_mode: bool = True
    enable_replay: bool = False
    enable_sentinel_verify: bool = True
    max_runtime_s: int = 60
    deterministic_timestamps: bool = True
    debug: bool = False


# =============================================================================
# Run Result
# =============================================================================

@dataclass
class RunResult:
    """Complete result of a pipeline run."""
    prompt_spec: Optional[PromptSpec] = None
    tool_plan: Optional[ToolPlan] = None
    evidence_bundle: Optional[EvidenceBundle] = None
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
            "ok": self.ok, "market_id": self.market_id, "outcome": self.outcome,
            "confidence": self.verdict.confidence if self.verdict else None,
            "errors": self.errors, "check_count": len(self.checks),
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


def _enforce_commitment_safe_timestamps(
    prompt_spec: PromptSpec, evidence_bundle: EvidenceBundle, verdict: DeterministicVerdict,
) -> None:
    """Enforce None timestamps in committed objects for determinism (no-op if schemas default to None)."""
    pass  # Schemas already default timestamps to None


# =============================================================================
# Pipeline Class
# =============================================================================

class Pipeline:
    """Main pipeline runner orchestrating the full resolution flow."""
    
    def __init__(
        self, *, config: Optional[PipelineConfig] = None,
        prompt_engineer: Optional[PromptEngineerProtocol] = None,
        collector: Optional[CollectorProtocol] = None,
        auditor: Optional[AuditorProtocol] = None,
        judge: Optional[JudgeProtocol] = None,
        sentinel: Optional[SentinelProtocol] = None,
    ):
        self.config = config or PipelineConfig()
        self._prompt_engineer = prompt_engineer
        self._collector = collector
        self._auditor = auditor
        self._judge = judge
        self._sentinel = sentinel
        self._executor = SOPExecutor(stop_on_error=True)
    
    def run(self, user_input: str) -> RunResult:
        """Execute the full pipeline."""
        state = PipelineState(user_input=user_input)
        steps = self._build_steps()
        state = self._executor.execute(steps, state)
        return self._state_to_result(state)
    
    def _build_steps(self) -> list:
        """Build pipeline steps based on config."""
        steps = []
        if self._prompt_engineer:
            steps.append(make_step("prompt_compilation", self._step_prompt_compilation))
        if self._collector:
            steps.append(make_step("evidence_collection", self._step_evidence_collection))
        if self._auditor:
            steps.append(make_step("audit", self._step_audit))
        if self._judge:
            steps.append(make_step("judge", self._step_judge))
        steps.append(make_step("build_por_bundle", self._step_build_por_bundle))
        if self.config.enable_sentinel_verify and self._sentinel:
            steps.append(make_step("sentinel_verify", self._step_sentinel_verify))
        return steps
    
    def _step_prompt_compilation(self, state: PipelineState) -> PipelineState:
        """Step 1: Compile user input into PromptSpec and ToolPlan."""
        if not self._prompt_engineer:
            state.add_error("Prompt engineer not configured")
            return state
        prompt_spec, tool_plan = self._prompt_engineer.run(state.user_input)
        if self.config.strict_mode:
            if not prompt_spec.extra.get("strict_mode", False):
                state.add_check(CheckResult(
                    check_id="strict_mode_check", ok=False, severity="error",
                    message="PromptSpec.extra['strict_mode'] must be True", details={},
                ))
        if self.config.deterministic_timestamps and prompt_spec.created_at is not None:
            state.add_check(CheckResult(
                check_id="timestamp_determinism_prompt", ok=False, severity="error",
                message="PromptSpec.created_at must be None", details={},
            ))
        state.prompt_spec = prompt_spec
        state.tool_plan = tool_plan
        state.add_check(CheckResult.passed("prompt_compilation", f"Compiled prompt for {prompt_spec.market.market_id}"))
        return state
    
    def _step_evidence_collection(self, state: PipelineState) -> PipelineState:
        """Step 2: Collect evidence."""
        if not self._collector:
            state.add_error("Collector not configured")
            return state
        if not state.prompt_spec or not state.tool_plan:
            state.add_error("Cannot collect evidence: missing prompt_spec or tool_plan")
            return state
        evidence_bundle = self._collector.collect(state.prompt_spec, state.tool_plan)
        state.evidence_bundle = evidence_bundle
        state.add_check(CheckResult.passed("evidence_collection", f"Collected {len(evidence_bundle.items)} items"))
        return state
    
    def _step_audit(self, state: PipelineState) -> PipelineState:
        """Step 3: Produce reasoning trace."""
        if not self._auditor:
            state.add_error("Auditor not configured")
            return state
        if not state.prompt_spec or not state.evidence_bundle:
            state.add_error("Cannot audit: missing artifacts")
            return state
        trace, verification = self._auditor.audit(state.prompt_spec, state.evidence_bundle)
        state.audit_trace = trace
        state.audit_verification = verification
        state.merge_verification(verification)
        return state
    
    def _step_judge(self, state: PipelineState) -> PipelineState:
        """Step 4: Produce verdict."""
        if not self._judge:
            state.add_error("Judge not configured")
            return state
        if not state.prompt_spec or not state.evidence_bundle or not state.audit_trace:
            state.add_error("Cannot judge: missing artifacts")
            return state
        verdict, verification = self._judge.judge(state.prompt_spec, state.evidence_bundle, state.audit_trace)
        state.verdict = verdict
        state.judge_verification = verification
        state.merge_verification(verification)
        return state
    
    def _step_build_por_bundle(self, state: PipelineState) -> PipelineState:
        """Step 5: Build PoR bundle."""
        if not all([state.prompt_spec, state.evidence_bundle, state.audit_trace, state.verdict]):
            state.add_error("Cannot build PoR bundle: missing artifacts")
            return state
        if self.config.deterministic_timestamps:
            _enforce_commitment_safe_timestamps(state.prompt_spec, state.evidence_bundle, state.verdict)
        roots = compute_roots(state.prompt_spec, state.evidence_bundle, state.audit_trace, state.verdict)
        state.roots = roots
        por_bundle = build_por_bundle(
            state.prompt_spec, state.evidence_bundle, state.audit_trace, state.verdict,
            include_por_root=True, metadata={"pipeline_version": "09A"},
        )
        state.por_bundle = por_bundle
        state.add_check(CheckResult.passed("por_bundle_built", f"Built bundle with root {roots.por_root[:18]}..."))
        return state
    
    def _step_sentinel_verify(self, state: PipelineState) -> PipelineState:
        """Step 6: Verify with Sentinel."""
        if not self._sentinel:
            state.add_error("Sentinel not configured")
            return state
        if not all([state.por_bundle, state.prompt_spec, state.evidence_bundle, state.audit_trace, state.verdict]):
            state.add_error("Cannot verify: missing artifacts")
            return state
        package = PoRPackage(
            bundle=state.por_bundle, prompt_spec=state.prompt_spec, tool_plan=state.tool_plan,
            evidence=state.evidence_bundle, trace=state.audit_trace, verdict=state.verdict,
        )
        mode = "replay" if self.config.enable_replay else "verify"
        verification, challenges = self._sentinel.verify(package, mode=mode)
        state.sentinel_verification = verification
        state.challenges = challenges
        state.merge_verification(verification)
        return state
    
    def _state_to_result(self, state: PipelineState) -> RunResult:
        """Convert final PipelineState to RunResult."""
        ok = state.ok and state.por_bundle is not None
        return RunResult(
            prompt_spec=state.prompt_spec, tool_plan=state.tool_plan,
            evidence_bundle=state.evidence_bundle, audit_trace=state.audit_trace,
            audit_verification=state.audit_verification, verdict=state.verdict,
            judge_verification=state.judge_verification, por_bundle=state.por_bundle,
            roots=state.roots, sentinel_verification=state.sentinel_verification,
            challenges=state.challenges, ok=ok, checks=state.checks, errors=state.errors,
        )


def create_pipeline(
    *, prompt_engineer: Optional[PromptEngineerProtocol] = None,
    collector: Optional[CollectorProtocol] = None, auditor: Optional[AuditorProtocol] = None,
    judge: Optional[JudgeProtocol] = None, sentinel: Optional[SentinelProtocol] = None,
    strict_mode: bool = True, enable_sentinel: bool = True,
) -> Pipeline:
    """Convenience function to create a pipeline with common configuration."""
    config = PipelineConfig(
        strict_mode=strict_mode,
        enable_sentinel_verify=enable_sentinel and sentinel is not None,
    )
    return Pipeline(
        config=config, prompt_engineer=prompt_engineer, collector=collector,
        auditor=auditor, judge=judge, sentinel=sentinel,
    )