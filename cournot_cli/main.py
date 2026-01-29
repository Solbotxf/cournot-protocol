"""
Module 09C - CLI Main Entry Point

Parses command-line arguments and dispatches to subcommands.

Usage:
    python -m cournot_cli run "<query>" [--out PATH] [--json] [--debug]
    python -m cournot_cli verify <pack_path> [--no-sentinel] [--json] [--debug]
    python -m cournot_cli replay <pack_path> [--timeout N] [--json] [--debug]
    python -m cournot_cli pack --from-dir PATH --out PATH
    python -m cournot_cli step prompt "<query>" --out ./prompt
    python -m cournot_cli step collect --prompt ./prompt --out ./evidence
    python -m cournot_cli step audit --prompt ./prompt --evidence ./evidence --out ./trace
    python -m cournot_cli step judge --prompt ./prompt --evidence ./evidence --trace ./trace --out ./verdict
    python -m cournot_cli step sentinel --prompt ./prompt --evidence ./evidence --trace ./trace --verdict ./verdict
    python -m cournot_cli config --init

Environment Variables:
    COURNOT_STRICT_MODE         Enable strict mode (default: true)
    COURNOT_ENABLE_SENTINEL     Enable sentinel verification (default: true)
    COURNOT_EXECUTION_MODE      Execution mode: production, development, test
    COURNOT_LOG_LEVEL           Log level (default: INFO)
    COURNOT_LLM_PROVIDER        LLM provider (anthropic, openai, etc.)
    COURNOT_LLM_API_KEY         LLM API key
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Sequence

from cournot_cli.commands import run, verify, replay, pack, steps
from cournot_cli.config import load_config, get_default_config_template, CLIConfig


# Exit codes
EXIT_SUCCESS = 0
EXIT_RUNTIME_ERROR = 1
EXIT_VERIFICATION_FAILED = 2


def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """Configure logging for the CLI."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="cournot",
        description="Cournot Protocol CLI - Run pipelines, verify proofs, and manage artifact packs.",
    )
    parser.add_argument(
        "--version", action="version", version="%(prog)s 0.1.0"
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=None,
        help="Path to configuration file (default: ./cournot.json or ~/.config/cournot/config.json)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (overrides config)",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # --- run command ---
    run_parser = subparsers.add_parser(
        "run",
        help="Run the full pipeline on a user query",
        description="Execute the Cournot pipeline and optionally save an artifact pack.",
    )
    run_parser.add_argument(
        "query",
        type=str,
        help="The prediction market query to resolve",
    )
    run_parser.add_argument(
        "--out", "-o",
        type=str,
        default=None,
        help="Output path for artifact pack (zip if .zip extension, otherwise directory)",
    )
    run_parser.add_argument(
        "--mode",
        type=str,
        choices=["production", "development", "test"],
        default=None,
        help="Execution mode (default: from config or development)",
    )
    run_parser.add_argument(
        "--no-sentinel",
        action="store_true",
        default=False,
        help="Disable sentinel verification",
    )
    run_parser.add_argument(
        "--strict",
        dest="strict",
        action="store_true",
        default=None,
        help="Enable strict mode (default)",
    )
    run_parser.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        help="Disable strict mode",
    )
    run_parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output machine-readable JSON summary",
    )
    run_parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Include detailed checks in output",
    )
    run_parser.set_defaults(func=run.run_cmd)
    
    # --- verify command ---
    verify_parser = subparsers.add_parser(
        "verify",
        help="Verify an artifact pack offline",
        description="Verify file hashes, semantic consistency, and optionally sentinel verification.",
    )
    verify_parser.add_argument(
        "pack_path",
        type=str,
        help="Path to artifact pack (directory or zip file)",
    )
    verify_parser.add_argument(
        "--no-sentinel",
        action="store_true",
        default=False,
        help="Skip sentinel verification (only pack-level validation)",
    )
    verify_parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output machine-readable JSON report",
    )
    verify_parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Include detailed checks and challenge objects",
    )
    verify_parser.set_defaults(func=verify.verify_cmd)
    
    # --- replay command ---
    replay_parser = subparsers.add_parser(
        "replay",
        help="Replay evidence collection and compare (online)",
        description="Re-collect evidence and compare with pack contents to detect divergence.",
    )
    replay_parser.add_argument(
        "pack_path",
        type=str,
        help="Path to artifact pack (directory or zip file)",
    )
    replay_parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Timeout in seconds for replay operations (default: 30)",
    )
    replay_parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output machine-readable JSON report",
    )
    replay_parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Include detailed divergence information",
    )
    replay_parser.set_defaults(func=replay.replay_cmd)
    
    # --- pack command ---
    pack_parser = subparsers.add_parser(
        "pack",
        help="Create a pack from existing artifact JSON files",
        description="Build an artifact pack from a directory containing JSON artifact files.",
    )
    pack_parser.add_argument(
        "--from-dir",
        type=str,
        required=True,
        help="Directory containing artifact JSON files",
    )
    pack_parser.add_argument(
        "--out", "-o",
        type=str,
        required=True,
        help="Output path for artifact pack (zip if .zip extension)",
    )
    pack_parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output machine-readable JSON summary",
    )
    pack_parser.set_defaults(func=pack.pack_cmd)
    
    # --- step command ---
    step_parser = subparsers.add_parser(
        "step",
        help="Run individual pipeline steps",
        description="Execute a single pipeline step for debugging or testing.",
    )
    step_subparsers = step_parser.add_subparsers(dest="step_name", help="Step to run")
    
    # step prompt
    step_prompt = step_subparsers.add_parser(
        "prompt",
        help="Run prompt engineer step",
    )
    step_prompt.add_argument("query", type=str, help="Query to compile")
    step_prompt.add_argument("--out", "-o", type=str, help="Output directory")
    step_prompt.add_argument("--agent", type=str, help="Specific agent name to use")
    step_prompt.add_argument("--json", action="store_true", help="JSON output")
    step_prompt.add_argument("--debug", action="store_true", help="Debug mode")
    step_prompt.set_defaults(func=steps.step_prompt_cmd)
    
    # step collect
    step_collect = step_subparsers.add_parser(
        "collect",
        help="Run collector step",
    )
    step_collect.add_argument("--prompt", type=str, required=True, help="Directory with prompt_spec.json")
    step_collect.add_argument("--out", "-o", type=str, help="Output directory")
    step_collect.add_argument("--agent", type=str, help="Specific agent name to use")
    step_collect.add_argument("--json", action="store_true", help="JSON output")
    step_collect.add_argument("--debug", action="store_true", help="Debug mode")
    step_collect.set_defaults(func=steps.step_collect_cmd)
    
    # step audit
    step_audit = step_subparsers.add_parser(
        "audit",
        help="Run auditor step",
    )
    step_audit.add_argument("--prompt", type=str, required=True, help="Directory with prompt_spec.json")
    step_audit.add_argument("--evidence", type=str, required=True, help="Directory with evidence_bundle.json")
    step_audit.add_argument("--out", "-o", type=str, help="Output directory")
    step_audit.add_argument("--agent", type=str, help="Specific agent name to use")
    step_audit.add_argument("--json", action="store_true", help="JSON output")
    step_audit.add_argument("--debug", action="store_true", help="Debug mode")
    step_audit.set_defaults(func=steps.step_audit_cmd)
    
    # step judge
    step_judge = step_subparsers.add_parser(
        "judge",
        help="Run judge step",
    )
    step_judge.add_argument("--prompt", type=str, required=True, help="Directory with prompt_spec.json")
    step_judge.add_argument("--evidence", type=str, required=True, help="Directory with evidence_bundle.json")
    step_judge.add_argument("--trace", type=str, required=True, help="Directory with reasoning_trace.json")
    step_judge.add_argument("--out", "-o", type=str, help="Output directory")
    step_judge.add_argument("--agent", type=str, help="Specific agent name to use")
    step_judge.add_argument("--json", action="store_true", help="JSON output")
    step_judge.add_argument("--debug", action="store_true", help="Debug mode")
    step_judge.set_defaults(func=steps.step_judge_cmd)
    
    # step sentinel
    step_sentinel = step_subparsers.add_parser(
        "sentinel",
        help="Run sentinel verification step",
    )
    step_sentinel.add_argument("--prompt", type=str, required=True, help="Directory with prompt_spec.json")
    step_sentinel.add_argument("--evidence", type=str, required=True, help="Directory with evidence_bundle.json")
    step_sentinel.add_argument("--trace", type=str, required=True, help="Directory with reasoning_trace.json")
    step_sentinel.add_argument("--verdict", type=str, required=True, help="Directory with verdict.json")
    step_sentinel.add_argument("--agent", type=str, help="Specific agent name to use")
    step_sentinel.add_argument("--json", action="store_true", help="JSON output")
    step_sentinel.add_argument("--debug", action="store_true", help="Debug mode")
    step_sentinel.set_defaults(func=steps.step_sentinel_cmd)
    
    step_parser.set_defaults(func=lambda args: step_parser.print_help() or EXIT_SUCCESS)
    
    # --- config command ---
    config_parser = subparsers.add_parser(
        "config",
        help="Manage CLI configuration",
        description="Initialize or display configuration.",
    )
    config_parser.add_argument(
        "--init",
        action="store_true",
        default=False,
        help="Create a template configuration file",
    )
    config_parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="Show current configuration",
    )
    config_parser.add_argument(
        "--path",
        type=str,
        default="cournot.json",
        help="Path for config file (default: cournot.json)",
    )
    config_parser.set_defaults(func=config_cmd)
    
    # --- agents command ---
    agents_parser = subparsers.add_parser(
        "agents",
        help="List registered agents",
        description="Show all registered agents and their capabilities.",
    )
    agents_parser.add_argument(
        "--step",
        type=str,
        choices=["prompt_engineer", "collector", "auditor", "judge", "sentinel"],
        help="Filter by step",
    )
    agents_parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="JSON output",
    )
    agents_parser.set_defaults(func=agents_cmd)
    
    return parser


def config_cmd(args: argparse.Namespace) -> int:
    """Handle config command."""
    if args.init:
        config_path = Path(args.path)
        if config_path.exists():
            print(f"Error: Config file already exists: {config_path}", file=sys.stderr)
            return EXIT_RUNTIME_ERROR
        
        config_path.write_text(get_default_config_template())
        print(f"Created configuration file: {config_path}")
        print("\nEdit this file to configure your settings.")
        print("You can also use environment variables (COURNOT_* prefix).")
        return EXIT_SUCCESS
    
    if args.show:
        config = load_config(Path(args.path) if args.path else None)
        config_dict = {
            "strict_mode": config.strict_mode,
            "enable_sentinel": config.enable_sentinel,
            "enable_replay": config.enable_replay,
            "execution_mode": config.execution_mode,
            "require_llm": config.require_llm,
            "require_network": config.require_network,
            "pipeline_timeout": config.pipeline_timeout,
            "replay_timeout": config.replay_timeout,
            "log_level": config.log_level,
            "llm": {
                "provider": config.llm.provider or "(not configured)",
                "model": config.llm.model or "(not configured)",
            },
        }
        print(json.dumps(config_dict, indent=2))
        return EXIT_SUCCESS
    
    # Default: show help
    print("Usage: cournot config [--init|--show]")
    print("  --init  Create a template configuration file")
    print("  --show  Show current configuration")
    return EXIT_SUCCESS


def agents_cmd(args: argparse.Namespace) -> int:
    """Handle agents command."""
    from agents import get_registry, AgentStep
    
    registry = get_registry()
    
    step_filter = None
    if args.step:
        step_map = {
            "prompt_engineer": AgentStep.PROMPT_ENGINEER,
            "collector": AgentStep.COLLECTOR,
            "auditor": AgentStep.AUDITOR,
            "judge": AgentStep.JUDGE,
            "sentinel": AgentStep.SENTINEL,
        }
        step_filter = step_map.get(args.step)
    
    agents_list = registry.list_agents(step_filter)
    
    if args.json:
        data = [
            {
                "name": a.name,
                "version": a.version,
                "step": a.step.value,
                "capabilities": [c.value for c in a.capabilities],
                "priority": a.priority,
                "is_fallback": a.is_fallback,
            }
            for a in agents_list
        ]
        print(json.dumps(data, indent=2))
    else:
        if not agents_list:
            print("No agents registered")
            return EXIT_SUCCESS
        
        # Group by step
        by_step: dict[str, list] = {}
        for a in agents_list:
            step = a.step.value
            if step not in by_step:
                by_step[step] = []
            by_step[step].append(a)
        
        for step, step_agents in sorted(by_step.items()):
            print(f"\n{step}:")
            for a in step_agents:
                caps = ", ".join(c.value for c in a.capabilities)
                fallback_marker = " [fallback]" if a.is_fallback else ""
                print(f"  - {a.name} (v{a.version}) [priority={a.priority}]{fallback_marker}")
                print(f"    capabilities: {caps}")
    
    return EXIT_SUCCESS


def main(argv: Sequence[str] | None = None) -> int:
    """
    Main entry point for the CLI.
    
    Args:
        argv: Command-line arguments (defaults to sys.argv[1:])
    
    Returns:
        Exit code (0=success, 1=error, 2=verification failed)
    """
    parser = create_parser()
    args = parser.parse_args(argv)
    
    if args.command is None:
        parser.print_help()
        return EXIT_RUNTIME_ERROR
    
    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        return EXIT_RUNTIME_ERROR
    
    # Setup logging
    log_level = args.log_level or config.log_level
    setup_logging(level=log_level, log_file=config.log_file)
    
    # Attach config to args for commands to use
    args.cli_config = config
    
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return EXIT_RUNTIME_ERROR
    except Exception as e:
        if hasattr(args, "debug") and args.debug:
            traceback.print_exc()
        else:
            print(f"Error: {e}", file=sys.stderr)
        return EXIT_RUNTIME_ERROR


if __name__ == "__main__":
    sys.exit(main())
