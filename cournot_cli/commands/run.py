"""
Module 09C - CLI Run Command (Production)

Execute the full pipeline using the real agents from agents/ package.

Usage:
    cournot run "<query>" --out pack.zip
"""

from __future__ import annotations

import json
import logging
import sys
from argparse import Namespace
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, TYPE_CHECKING

from cournot_cli.config import CLIConfig

# Type checking imports (not executed at runtime)
if TYPE_CHECKING:
    from agents.prompt_engineer import PromptEngineerAgent
    from agents.collector import CollectorAgent, CollectorConfig
    from agents.auditor import AuditorAgent
    from agents.judge import JudgeAgent
    from agents.validator import SentinelAgent, PoRPackage
    from orchestrator.pipeline import Pipeline, PipelineConfig, RunResult


logger = logging.getLogger(__name__)


# Exit codes
EXIT_SUCCESS = 0
EXIT_RUNTIME_ERROR = 1
EXIT_VERIFICATION_FAILED = 2


@dataclass
class RunSummary:
    """Summary of a pipeline run for CLI output."""
    market_id: str = ""
    outcome: str = ""
    confidence: float = 0.0
    por_root: str = ""
    prompt_spec_hash: str = ""
    evidence_root: str = ""
    reasoning_root: str = ""
    saved_to: str | None = None
    ok: bool = True
    verification_ok: bool = True
    checks: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if not d["saved_to"]:
            del d["saved_to"]
        if not d["checks"]:
            del d["checks"]
        if not d["errors"]:
            del d["errors"]
        return d


def _import_agents():
    """Import agents lazily to allow CLI to work without agents installed."""
    try:
        from agents.prompt_engineer import PromptEngineerAgent
        from agents.collector import CollectorAgent, CollectorConfig
        from agents.auditor import AuditorAgent
        from agents.judge import JudgeAgent
        from agents.validator import SentinelAgent, PoRPackage
        from orchestrator.pipeline import Pipeline, PipelineConfig, RunResult
        from orchestrator.artifacts.io import save_pack
        
        return {
            "PromptEngineerAgent": PromptEngineerAgent,
            "CollectorAgent": CollectorAgent,
            "CollectorConfig": CollectorConfig,
            "AuditorAgent": AuditorAgent,
            "JudgeAgent": JudgeAgent,
            "SentinelAgent": SentinelAgent,
            "PoRPackage": PoRPackage,
            "Pipeline": Pipeline,
            "PipelineConfig": PipelineConfig,
            "RunResult": RunResult,
            "save_pack": save_pack,
        }
    except ImportError as e:
        raise ImportError(
            f"Could not import agents package: {e}\n"
            "The 'run' command requires the agents package to be installed.\n"
            "Make sure the agents/ directory is in your PYTHONPATH."
        ) from e


def create_agents(config: CLIConfig) -> dict[str, Any]:
    """
    Create all pipeline agents using the real implementations.
    
    Args:
        config: CLI configuration
    
    Returns:
        Dictionary of agent instances
    """
    modules = _import_agents()
    
    # Create Prompt Engineer
    prompt_engineer = modules["PromptEngineerAgent"]()
    
    # Create Collector with configuration
    collector_config = modules["CollectorConfig"](
        default_timeout_s=config.collector.default_timeout_s,
        strict_tier_policy=config.collector.strict_tier_policy,
        include_timestamps=config.collector.include_timestamps,
        collector_id=config.collector.collector_id,
    )
    collector = modules["CollectorAgent"](config=collector_config)
    
    # Create Auditor
    auditor = modules["AuditorAgent"]()
    
    # Create Judge
    judge = modules["JudgeAgent"](require_strict_mode=config.strict_mode)
    
    # Create Sentinel (optional)
    sentinel = modules["SentinelAgent"](strict_mode=config.strict_mode) if config.enable_sentinel else None
    
    return {
        "prompt_engineer": prompt_engineer,
        "collector": collector,
        "auditor": auditor,
        "judge": judge,
        "sentinel": sentinel,
        "_modules": modules,  # Keep reference for later use
    }


def run_pipeline(
    query: str,
    config: CLIConfig,
    *,
    strict: bool | None = None,
    enable_sentinel: bool = True,
    replay_mode: bool = False,
) -> tuple[Any, Any]:
    """
    Execute the pipeline with the given query.
    
    Args:
        query: The prediction market query
        config: CLI configuration
        strict: Override strict mode (None = use config)
        enable_sentinel: Whether to enable sentinel verification
        replay_mode: Whether to enable replay mode
    
    Returns:
        Tuple of (RunResult, PoRPackage or None)
    """
    logger.info("Creating agents...")
    agents = create_agents(config)
    modules = agents["_modules"]
    
    # Determine strict mode
    if strict is None:
        strict = config.strict_mode
    
    # Create pipeline configuration
    pipeline_config = modules["PipelineConfig"](
        strict_mode=strict,
        enable_sentinel_verify=enable_sentinel and config.enable_sentinel,
        enable_replay=replay_mode,
    )
    
    # Create pipeline with real agents
    pipeline = modules["Pipeline"](
        config=pipeline_config,
        prompt_engineer=agents["prompt_engineer"],
        collector=agents["collector"],
        auditor=agents["auditor"],
        judge=agents["judge"],
        sentinel=agents["sentinel"] if enable_sentinel else None,
    )
    
    # Run pipeline
    logger.info(f"Running pipeline for query: {query[:50]}...")
    result = pipeline.run(query)
    
    # Build PoRPackage for saving
    package = None
    if result.por_bundle is not None:
        package = modules["PoRPackage"](
            bundle=result.por_bundle,
            prompt_spec=result.prompt_spec,
            tool_plan=result.tool_plan,
            evidence=result.evidence_bundle,
            trace=result.audit_trace,
            verdict=result.verdict,
        )
    
    return result, package


def build_summary(result: Any, saved_to: str | None = None, debug: bool = False) -> RunSummary:
    """Build a RunSummary from pipeline result."""
    summary = RunSummary(
        market_id=result.market_id or "",
        outcome=result.outcome or "",
        confidence=result.verdict.confidence if result.verdict else 0.0,
        por_root=result.por_bundle.por_root if result.por_bundle else "",
        prompt_spec_hash=result.roots.prompt_spec_hash if result.roots else "",
        evidence_root=result.roots.evidence_root if result.roots else "",
        reasoning_root=result.roots.reasoning_root if result.roots else "",
        saved_to=saved_to,
        ok=result.ok,
        verification_ok=True,
        errors=list(result.errors) if result.errors else [],
    )
    
    # Check sentinel verification
    if result.sentinel_verification is not None:
        summary.verification_ok = result.sentinel_verification.ok
    
    # Include checks if debug
    if debug and result.checks:
        summary.checks = [
            {"check_id": c.check_id, "ok": c.ok, "message": c.message}
            for c in result.checks
        ]
    
    return summary


def print_summary_human(summary: RunSummary) -> None:
    """Print summary in human-readable format."""
    print(f"market_id: {summary.market_id}")
    print(f"outcome: {summary.outcome}")
    print(f"confidence: {summary.confidence:.2f}")
    print(f"por_root: {summary.por_root}")
    if summary.saved_to:
        print(f"saved: {summary.saved_to}")
    print(f"ok: {str(summary.ok).lower()}")
    print(f"verification_ok: {str(summary.verification_ok).lower()}")
    
    if summary.errors:
        print(f"errors: {len(summary.errors)}")
        for err in summary.errors[:5]:
            print(f"  - {err}")
    
    if summary.checks:
        print(f"checks: {len(summary.checks)}")
        for check in summary.checks[:10]:
            status = "✓" if check["ok"] else "✗"
            print(f"  {status} {check['check_id']}: {check['message']}")


def print_summary_json(summary: RunSummary) -> None:
    """Print summary as JSON."""
    print(json.dumps(summary.to_dict(), indent=2))


def run_cmd(args: Namespace) -> int:
    """
    Execute the run command.
    
    Args:
        args: Parsed command-line arguments
    
    Returns:
        Exit code
    """
    query = args.query
    out_path = args.out
    enable_sentinel = not args.no_sentinel
    replay_mode = args.replay
    strict = args.strict
    output_json = args.json
    debug = args.debug
    
    # Get configuration from args (set by main.py)
    config: CLIConfig = getattr(args, "cli_config", None)
    if config is None:
        print("Error: Configuration not loaded", file=sys.stderr)
        return EXIT_RUNTIME_ERROR
    
    # Run pipeline
    try:
        result, package = run_pipeline(
            query,
            config,
            strict=strict,
            enable_sentinel=enable_sentinel,
            replay_mode=replay_mode,
        )
    except ImportError as e:
        print(f"Import error: {e}", file=sys.stderr)
        return EXIT_RUNTIME_ERROR
    except Exception as e:
        if debug:
            raise
        print(f"Pipeline error: {e}", file=sys.stderr)
        return EXIT_RUNTIME_ERROR
    
    # Save pack if requested
    saved_to = None
    if out_path and package:
        try:
            modules = _import_agents()
            save_path = modules["save_pack"](package, out_path)
            saved_to = str(save_path)
            logger.info(f"Saved artifact pack to: {saved_to}")
        except Exception as e:
            if debug:
                raise
            print(f"Failed to save pack: {e}", file=sys.stderr)
            return EXIT_RUNTIME_ERROR
    
    # Build and print summary
    summary = build_summary(result, saved_to=saved_to, debug=debug)
    
    if output_json:
        print_summary_json(summary)
    else:
        print_summary_human(summary)
    
    # Determine exit code
    if not result.ok:
        return EXIT_RUNTIME_ERROR
    if not summary.verification_ok:
        return EXIT_VERIFICATION_FAILED
    return EXIT_SUCCESS