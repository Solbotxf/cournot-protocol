"""
Module 09C - CLI Step Commands

Execute individual pipeline steps for debugging and testing.

Usage:
    cournot step prompt "<query>" --out prompt_spec.json
    cournot step collect --prompt prompt_spec.json --out evidence.json
    cournot step audit --prompt prompt_spec.json --evidence evidence.json --out trace.json
    cournot step judge --prompt prompt_spec.json --evidence evidence.json --trace trace.json --out verdict.json
    cournot step sentinel --prompt prompt_spec.json --evidence evidence.json --trace trace.json --verdict verdict.json
"""

from __future__ import annotations

import json
import logging
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

from cournot_cli.config import CLIConfig


logger = logging.getLogger(__name__)


# Exit codes
EXIT_SUCCESS = 0
EXIT_RUNTIME_ERROR = 1
EXIT_VERIFICATION_FAILED = 2


def load_json_artifact(path: Path, model_class: type) -> Any:
    """Load a JSON artifact and validate with Pydantic model."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return model_class.model_validate(data)


def save_json_artifact(path: Path, artifact: Any) -> None:
    """Save an artifact as JSON."""
    if hasattr(artifact, "model_dump"):
        data = artifact.model_dump(mode="json")
    elif hasattr(artifact, "to_dict"):
        data = artifact.to_dict()
    else:
        data = artifact
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def create_context(config: CLIConfig):
    """Create agent context from config (HTTP/LLM when configured)."""
    from cournot_cli.commands.run import create_agent_context
    return create_agent_context(config)


def select_agent(registry, step, ctx, agent_name: str | None):
    """Select agent, preferring fallback when LLM not available."""
    from agents.base import AgentCapability
    
    if agent_name:
        return registry.get_agent_by_name(agent_name, ctx)
    
    # If no LLM, prefer fallback agents
    agents = registry.list_agents(step)
    if ctx.llm is None:
        # Find a fallback or deterministic agent
        for agent_info in agents:
            if agent_info.is_fallback or AgentCapability.DETERMINISTIC in agent_info.capabilities:
                return agent_info.factory(ctx)
    
    # Default selection
    return registry.get_agent(step, ctx)


# =============================================================================
# Prompt Step
# =============================================================================

def step_prompt_cmd(args: Namespace) -> int:
    """Execute the prompt engineer step."""
    from agents import AgentContext, AgentStep, get_registry
    from core.schemas import PromptSpec
    from core.schemas.transport import ToolPlan
    
    query = args.query
    out_path = args.out
    output_json = args.json
    debug = args.debug
    agent_name = getattr(args, "agent", None)
    
    config: CLIConfig = getattr(args, "cli_config", CLIConfig())
    
    try:
        ctx = create_context(config)
        registry = get_registry()
        
        # Select agent
        agent = select_agent(registry, AgentStep.PROMPT_ENGINEER, ctx, agent_name)
        
        logger.info(f"Running prompt engineer: {agent.name}")
        result = agent.run(ctx, query)
        
        if not result.success:
            print(f"Prompt compilation failed: {result.error}", file=sys.stderr)
            return EXIT_RUNTIME_ERROR
        
        prompt_spec, tool_plan = result.output
        
        # Output
        if out_path:
            out_dir = Path(out_path)
            out_dir.mkdir(parents=True, exist_ok=True)
            save_json_artifact(out_dir / "prompt_spec.json", prompt_spec)
            save_json_artifact(out_dir / "tool_plan.json", tool_plan)
            print(f"Saved to: {out_dir}")
        
        if output_json:
            print(json.dumps({
                "market_id": prompt_spec.market.market_id,
                "ok": True,
                "prompt_spec": prompt_spec.model_dump(mode="json"),
                "tool_plan": tool_plan.model_dump(mode="json"),
            }, indent=2))
        else:
            print(f"market_id: {prompt_spec.market.market_id}")
            print(f"question: {prompt_spec.market.question}")
            print(f"requirements: {len(prompt_spec.data_requirements)}")
            print(f"sources: {len(tool_plan.sources)}")
        
        return EXIT_SUCCESS
        
    except Exception as e:
        if debug:
            raise
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_RUNTIME_ERROR


# =============================================================================
# Collect Step
# =============================================================================

def step_collect_cmd(args: Namespace) -> int:
    """Execute the collector step."""
    from agents import AgentContext, AgentStep, get_registry
    from core.schemas import PromptSpec, EvidenceBundle
    from core.schemas.transport import ToolPlan
    
    prompt_path = Path(args.prompt)
    out_path = args.out
    output_json = args.json
    debug = args.debug
    agent_name = getattr(args, "agent", None)
    
    config: CLIConfig = getattr(args, "cli_config", CLIConfig())
    
    try:
        # Load inputs
        prompt_spec = load_json_artifact(prompt_path / "prompt_spec.json", PromptSpec)
        tool_plan = load_json_artifact(prompt_path / "tool_plan.json", ToolPlan)
        
        ctx = create_context(config)
        registry = get_registry()
        
        # Select agent
        agent = select_agent(registry, AgentStep.COLLECTOR, ctx, agent_name)
        
        logger.info(f"Running collector: {agent.name}")
        result = agent.run(ctx, prompt_spec, tool_plan)
        
        if not result.success:
            print(f"Collection failed: {result.error}", file=sys.stderr)
            return EXIT_RUNTIME_ERROR
        
        evidence_bundle, execution_log = result.output
        
        # Output
        if out_path:
            out_dir = Path(out_path)
            out_dir.mkdir(parents=True, exist_ok=True)
            save_json_artifact(out_dir / "evidence_bundle.json", evidence_bundle)
            if execution_log:
                save_json_artifact(out_dir / "execution_log.json", execution_log)
            print(f"Saved to: {out_dir}")
        
        if output_json:
            print(json.dumps({
                "bundle_id": evidence_bundle.bundle_id,
                "ok": True,
                "evidence_count": len(evidence_bundle.items),
            }, indent=2))
        else:
            print(f"bundle_id: {evidence_bundle.bundle_id}")
            print(f"evidence_count: {len(evidence_bundle.items)}")
            for item in evidence_bundle.items[:5]:
                status = "✓" if item.success else "✗"
                tier = item.provenance.tier if item.provenance else "unknown"
                print(f"  {status} {item.evidence_id}: tier={tier}")
        
        return EXIT_SUCCESS
        
    except Exception as e:
        if debug:
            raise
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_RUNTIME_ERROR


# =============================================================================
# Audit Step
# =============================================================================

def step_audit_cmd(args: Namespace) -> int:
    """Execute the auditor step."""
    from agents import AgentContext, AgentStep, get_registry
    from core.schemas import PromptSpec, EvidenceBundle
    from core.schemas.reasoning import ReasoningTrace
    
    prompt_path = Path(args.prompt)
    evidence_path = Path(args.evidence)
    out_path = args.out
    output_json = args.json
    debug = args.debug
    agent_name = getattr(args, "agent", None)
    
    config: CLIConfig = getattr(args, "cli_config", CLIConfig())
    
    try:
        # Load inputs
        prompt_spec = load_json_artifact(prompt_path / "prompt_spec.json", PromptSpec)
        evidence_bundle = load_json_artifact(evidence_path / "evidence_bundle.json", EvidenceBundle)
        
        ctx = create_context(config)
        registry = get_registry()
        
        # Select agent
        agent = select_agent(registry, AgentStep.AUDITOR, ctx, agent_name)
        
        logger.info(f"Running auditor: {agent.name}")
        result = agent.run(ctx, prompt_spec, evidence_bundle)
        
        if not result.success:
            print(f"Audit failed: {result.error}", file=sys.stderr)
            return EXIT_RUNTIME_ERROR
        
        trace = result.output
        
        # Output
        if out_path:
            out_dir = Path(out_path)
            out_dir.mkdir(parents=True, exist_ok=True)
            save_json_artifact(out_dir / "reasoning_trace.json", trace)
            print(f"Saved to: {out_dir}")
        
        step_count = len(trace.steps) if hasattr(trace, 'steps') else 0
        
        if output_json:
            print(json.dumps({
                "trace_id": trace.trace_id,
                "ok": True,
                "step_count": step_count,
                "preliminary_outcome": trace.preliminary_outcome if hasattr(trace, 'preliminary_outcome') else None,
            }, indent=2))
        else:
            print(f"trace_id: {trace.trace_id}")
            print(f"step_count: {step_count}")
            if hasattr(trace, 'preliminary_outcome'):
                print(f"preliminary_outcome: {trace.preliminary_outcome}")
            if hasattr(trace, 'steps'):
                for step in trace.steps[:5]:
                    print(f"  - {step.step_id}: {step.step_type}")
        
        return EXIT_SUCCESS
        
    except Exception as e:
        if debug:
            raise
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_RUNTIME_ERROR


# =============================================================================
# Judge Step
# =============================================================================

def step_judge_cmd(args: Namespace) -> int:
    """Execute the judge step."""
    from agents import AgentContext, AgentStep, get_registry
    from core.schemas import PromptSpec, EvidenceBundle, DeterministicVerdict
    from core.schemas.reasoning import ReasoningTrace
    
    prompt_path = Path(args.prompt)
    evidence_path = Path(args.evidence)
    trace_path = Path(args.trace)
    out_path = args.out
    output_json = args.json
    debug = args.debug
    agent_name = getattr(args, "agent", None)
    
    config: CLIConfig = getattr(args, "cli_config", CLIConfig())
    
    try:
        # Load inputs
        prompt_spec = load_json_artifact(prompt_path / "prompt_spec.json", PromptSpec)
        evidence_bundle = load_json_artifact(evidence_path / "evidence_bundle.json", EvidenceBundle)
        trace = load_json_artifact(trace_path / "reasoning_trace.json", ReasoningTrace)
        
        ctx = create_context(config)
        registry = get_registry()
        
        # Select agent
        agent = select_agent(registry, AgentStep.JUDGE, ctx, agent_name)
        
        logger.info(f"Running judge: {agent.name}")
        result = agent.run(ctx, prompt_spec, evidence_bundle, trace)
        
        if not result.success:
            print(f"Judge failed: {result.error}", file=sys.stderr)
            return EXIT_RUNTIME_ERROR
        
        verdict = result.output
        
        # Output
        if out_path:
            out_dir = Path(out_path)
            out_dir.mkdir(parents=True, exist_ok=True)
            save_json_artifact(out_dir / "verdict.json", verdict)
            print(f"Saved to: {out_dir}")
        
        if output_json:
            print(json.dumps({
                "market_id": verdict.market_id,
                "outcome": verdict.outcome,
                "confidence": verdict.confidence,
                "ok": True,
            }, indent=2))
        else:
            print(f"market_id: {verdict.market_id}")
            print(f"outcome: {verdict.outcome}")
            print(f"confidence: {verdict.confidence:.2f}")
            print(f"rule: {verdict.resolution_rule_id}")
        
        return EXIT_SUCCESS
        
    except Exception as e:
        if debug:
            raise
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_RUNTIME_ERROR


# =============================================================================
# Sentinel Step
# =============================================================================

def step_sentinel_cmd(args: Namespace) -> int:
    """Execute the sentinel verification step."""
    from agents import AgentContext, AgentStep, get_registry
    from agents.sentinel import build_proof_bundle
    from core.schemas import PromptSpec, EvidenceBundle, DeterministicVerdict
    from core.schemas.reasoning import ReasoningTrace
    from core.schemas.transport import ToolPlan
    
    prompt_path = Path(args.prompt)
    evidence_path = Path(args.evidence)
    trace_path = Path(args.trace)
    verdict_path = Path(args.verdict)
    output_json = args.json
    debug = args.debug
    agent_name = getattr(args, "agent", None)
    
    config: CLIConfig = getattr(args, "cli_config", CLIConfig())
    
    try:
        # Load inputs
        prompt_spec = load_json_artifact(prompt_path / "prompt_spec.json", PromptSpec)
        
        # Try to load tool_plan (optional)
        tool_plan = None
        tool_plan_path = prompt_path / "tool_plan.json"
        if tool_plan_path.exists():
            tool_plan = load_json_artifact(tool_plan_path, ToolPlan)
        
        evidence_bundle = load_json_artifact(evidence_path / "evidence_bundle.json", EvidenceBundle)
        trace = load_json_artifact(trace_path / "reasoning_trace.json", ReasoningTrace)
        verdict = load_json_artifact(verdict_path / "verdict.json", DeterministicVerdict)
        
        # Build proof bundle
        proof_bundle = build_proof_bundle(
            prompt_spec, tool_plan, evidence_bundle, trace, verdict, None
        )
        
        ctx = create_context(config)
        registry = get_registry()
        
        # Select agent
        agent = select_agent(registry, AgentStep.SENTINEL, ctx, agent_name)
        
        logger.info(f"Running sentinel: {agent.name}")
        result = agent.run(ctx, proof_bundle)
        
        if not result.success:
            print(f"Sentinel failed: {result.error}", file=sys.stderr)
            return EXIT_RUNTIME_ERROR
        
        verification_result, report = result.output
        
        if output_json:
            print(json.dumps({
                "verified": report.verified,
                "total_checks": report.total_checks,
                "passed_checks": report.passed_checks,
                "failed_checks": report.failed_checks,
                "errors": report.errors,
            }, indent=2))
        else:
            status = "✓ VERIFIED" if report.verified else "✗ FAILED"
            print(f"status: {status}")
            print(f"checks: {report.passed_checks}/{report.total_checks} passed")
            if report.errors:
                print(f"\nerrors:")
                for err in report.errors[:10]:
                    print(f"  - {err}")
        
        return EXIT_SUCCESS if report.verified else EXIT_VERIFICATION_FAILED
        
    except Exception as e:
        if debug:
            raise
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_RUNTIME_ERROR


# =============================================================================
# Step Dispatcher
# =============================================================================

STEP_COMMANDS = {
    "prompt": step_prompt_cmd,
    "collect": step_collect_cmd,
    "audit": step_audit_cmd,
    "judge": step_judge_cmd,
    "sentinel": step_sentinel_cmd,
}


def step_cmd(args: Namespace) -> int:
    """Dispatch to the appropriate step command."""
    step_name = args.step_name
    
    if step_name not in STEP_COMMANDS:
        print(f"Unknown step: {step_name}", file=sys.stderr)
        print(f"Available steps: {', '.join(STEP_COMMANDS.keys())}")
        return EXIT_RUNTIME_ERROR
    
    return STEP_COMMANDS[step_name](args)
