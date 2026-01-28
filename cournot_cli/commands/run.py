"""
Module 09C - CLI Run Command (Production)

Execute the full pipeline and optionally save an artifact pack.

Usage:
    cournot run "<query>" --out pack.zip

Requires agents to be configured via cournot.json or environment variables.
"""

from __future__ import annotations

import json
import logging
import sys
from argparse import Namespace
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from cournot_cli.config import CLIConfig
from cournot_cli.agents import (
    load_all_agents,
    AgentNotConfiguredError,
    AgentLoadError,
    LocalSentinel,
)

from orchestrator.pipeline import Pipeline, PipelineConfig, RunResult, PoRPackage
from orchestrator.artifacts.io import save_pack


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


def run_pipeline(
    query: str,
    config: CLIConfig,
    *,
    strict: bool | None = None,
    enable_sentinel: bool = True,
    replay_mode: bool = False,
) -> tuple[RunResult, PoRPackage | None]:
    """
    Execute the pipeline with the given query.
    
    Args:
        query: The prediction market query
        config: CLI configuration with agent settings
        strict: Override strict mode (None = use config)
        enable_sentinel: Whether to enable sentinel verification
        replay_mode: Whether to enable replay mode
    
    Returns:
        Tuple of (RunResult, PoRPackage or None)
    
    Raises:
        AgentNotConfiguredError: If required agents are not configured
    """
    # Load agents from configuration
    logger.info("Loading agents from configuration...")
    agents = load_all_agents(config.agents)
    
    # Determine strict mode
    if strict is None:
        strict = config.strict_mode
    
    # Create pipeline configuration
    pipeline_config = PipelineConfig(
        strict_mode=strict,
        enable_sentinel_verify=enable_sentinel,
        enable_replay=replay_mode,
    )
    
    # Get sentinel (use local if not configured but enabled)
    sentinel = None
    if enable_sentinel:
        if "sentinel" in agents:
            sentinel = agents["sentinel"]
        else:
            # Fall back to local sentinel
            logger.info("Using local sentinel verification")
            sentinel = LocalSentinel()
    
    # Create pipeline
    pipeline = Pipeline(
        config=pipeline_config,
        prompt_engineer=agents["prompt_engineer"],
        collector=agents["collector"],
        auditor=agents["auditor"],
        judge=agents["judge"],
        sentinel=sentinel,
    )
    
    # Run pipeline
    logger.info(f"Running pipeline for query: {query[:50]}...")
    result = pipeline.run(query)
    
    # Build PoRPackage for saving
    package = None
    if result.por_bundle is not None:
        package = PoRPackage(
            bundle=result.por_bundle,
            prompt_spec=result.prompt_spec,
            tool_plan=result.tool_plan,
            evidence=result.evidence_bundle,
            trace=result.audit_trace,
            verdict=result.verdict,
        )
    
    return result, package


def build_summary(result: RunResult, saved_to: str | None = None, debug: bool = False) -> RunSummary:
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
    except AgentNotConfiguredError as e:
        print(f"Agent configuration error:\n{e}", file=sys.stderr)
        print("\nRun 'cournot config --init' to create a configuration file.", file=sys.stderr)
        return EXIT_RUNTIME_ERROR
    except AgentLoadError as e:
        print(f"Agent error: {e}", file=sys.stderr)
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
            save_path = save_pack(package, out_path)
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