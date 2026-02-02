"""
Module 09C - CLI Run Command

Execute the full pipeline using registry-based agent selection.

Usage:
    cournot run "<query>" --out pack.zip
    cournot run "<query>" --mode production --require-llm
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
    execution_mode: str = ""
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


def _runtime_config_from_cli(config: CLIConfig):
    """Build RuntimeConfig from CLIConfig for agent context (serper, etc.)."""
    from core.config import RuntimeConfig, SerperConfig
    runtime = RuntimeConfig.from_env()
    runtime.serper = SerperConfig(
        api_key=config.serper.api_key or None,
    )
    return runtime


def create_agent_context(config: CLIConfig):
    """
    Create an AgentContext based on configuration.
    
    Args:
        config: CLI configuration
    
    Returns:
        AgentContext instance
    """
    from agents import AgentContext

    runtime_config = _runtime_config_from_cli(config)
    
    # Start with minimal context, with config for serper etc.
    ctx = AgentContext.create_minimal()
    ctx = AgentContext(
        llm=ctx.llm,
        http=ctx.http,
        recorder=ctx.recorder,
        config=runtime_config,
        clock=ctx.clock,
        cache=ctx.cache,
        logger=ctx.logger,
    )
    
    # Add LLM if configured
    if config.llm.provider and config.llm.api_key:
        logger.debug(f"getting llm config: {config.llm}")
        try:
            from core.llm import create_llm_client
            llm = create_llm_client(
                provider=config.llm.provider,
                api_key=config.llm.api_key,
                model=config.llm.model,
                endpoint=config.llm.endpoint or None,
            )
            ctx = AgentContext(
                llm=llm,
                http=ctx.http,
                recorder=ctx.recorder,
                config=ctx.config,
                clock=ctx.clock,
                cache=ctx.cache,
                logger=ctx.logger,
            )
        except Exception as e:
            logger.warning(f"Failed to create LLM client: {e}")
    
    # Add HTTP client if needed
    if config.require_network or config.execution_mode == "production":
        try:
            from core.http import HttpClient
            http = HttpClient()
            ctx = AgentContext(
                llm=ctx.llm,
                http=http,
                recorder=ctx.recorder,
                config=ctx.config,
                clock=ctx.clock,
                cache=ctx.cache,
                logger=ctx.logger,
            )
        except Exception as e:
            logger.warning(f"Failed to create HTTP client: {e}")
    
    return ctx


def run_pipeline(
    query: str,
    config: CLIConfig,
    *,
    strict: bool | None = None,
    enable_sentinel: bool = True,
    mode: str | None = None,
):
    """
    Execute the pipeline with the given query.
    
    Args:
        query: The prediction market query
        config: CLI configuration
        strict: Override strict mode (None = use config)
        enable_sentinel: Whether to enable sentinel verification
        mode: Override execution mode
    
    Returns:
        Tuple of (RunResult, PoRPackage or None, execution_mode)
    """
    from orchestrator import (
        Pipeline,
        PipelineConfig,
        ExecutionMode,
        PoRPackage,
    )
    
    logger.info("Creating agent context...")
    ctx = create_agent_context(config)
    
    # Determine execution mode
    exec_mode_str = mode or config.execution_mode
    if exec_mode_str == "production":
        exec_mode = ExecutionMode.PRODUCTION
    elif exec_mode_str == "test":
        exec_mode = ExecutionMode.TEST
    else:
        exec_mode = ExecutionMode.DEVELOPMENT
    
    # Determine strict mode
    if strict is None:
        strict = config.strict_mode
    
    # Create pipeline configuration
    pipeline_config = PipelineConfig(
        mode=exec_mode,
        strict_mode=strict,
        enable_sentinel_verify=enable_sentinel and config.enable_sentinel,
        enable_replay=config.enable_replay,
        require_llm=config.require_llm,
        require_network=config.require_network,
        max_audit_evidence_chars=config.max_audit_evidence_chars,
        max_audit_evidence_items=config.max_audit_evidence_items,
    )
    
    # Create pipeline
    pipeline = Pipeline(config=pipeline_config, context=ctx)
    
    # Run pipeline
    logger.info(f"Running pipeline for query: {query[:50]}...")
    logger.info(f"Execution mode: {exec_mode.value}")
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
    
    return result, package, exec_mode.value


def build_summary(
    result,
    execution_mode: str,
    saved_to: str | None = None,
    debug: bool = False,
) -> RunSummary:
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
        execution_mode=execution_mode,
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
    print(f"mode: {summary.execution_mode}")
    if summary.saved_to:
        print(f"saved: {summary.saved_to}")
    print(f"ok: {str(summary.ok).lower()}")
    print(f"verification_ok: {str(summary.verification_ok).lower()}")
    
    if summary.errors:
        print(f"\nerrors ({len(summary.errors)}):")
        for err in summary.errors[:5]:
            print(f"  - {err}")
    
    if summary.checks:
        passed = sum(1 for c in summary.checks if c["ok"])
        failed = len(summary.checks) - passed
        print(f"\nchecks: {passed} passed, {failed} failed")
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
    strict = args.strict
    mode = getattr(args, "mode", None)
    output_json = args.json
    debug = args.debug
    
    # Get configuration from args (set by main.py)
    config: CLIConfig = getattr(args, "cli_config", None)
    if config is None:
        from cournot_cli.config import CLIConfig
        config = CLIConfig()
    
    # Run pipeline
    try:
        result, package, exec_mode = run_pipeline(
            query,
            config,
            strict=strict,
            enable_sentinel=enable_sentinel,
            mode=mode,
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
            from orchestrator.artifacts.io import save_pack
            save_path = save_pack(package, out_path)
            saved_to = str(save_path)
            logger.info(f"Saved artifact pack to: {saved_to}")
        except Exception as e:
            if debug:
                raise
            print(f"Failed to save pack: {e}", file=sys.stderr)
            return EXIT_RUNTIME_ERROR
    
    # Build and print summary
    summary = build_summary(result, exec_mode, saved_to=saved_to, debug=debug)
    
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
