"""
Module 09C - CLI Pack Command

Create an artifact pack from existing JSON files.

Usage:
    cournot pack --from-dir ./artifacts --out pack.zip
"""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.schemas.prompts import PromptSpec
from core.schemas.transport import ToolPlan
from core.schemas.evidence import EvidenceBundle
from core.schemas.verdict import DeterministicVerdict
from core.por.reasoning_trace import ReasoningTrace
from core.por.por_bundle import PoRBundle
from core.por.proof_of_reasoning import compute_roots, build_por_bundle

from orchestrator.pipeline import PoRPackage
from orchestrator.artifacts.io import save_pack


# Exit codes
EXIT_SUCCESS = 0
EXIT_RUNTIME_ERROR = 1


# Expected file names in source directory
EXPECTED_FILES = {
    "prompt_spec": "prompt_spec.json",
    "tool_plan": "tool_plan.json",  # optional
    "evidence_bundle": "evidence_bundle.json",
    "reasoning_trace": "reasoning_trace.json",
    "verdict": "verdict.json",
    "por_bundle": "por_bundle.json",  # optional - will be generated if missing
}


@dataclass
class PackSummary:
    """Summary of pack creation for CLI output."""
    source_dir: str = ""
    output_path: str = ""
    market_id: str = ""
    por_root: str = ""
    files_found: list[str] = None
    files_missing: list[str] = None
    success: bool = False
    error: str | None = None
    
    def __post_init__(self):
        if self.files_found is None:
            self.files_found = []
        if self.files_missing is None:
            self.files_missing = []
    
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if d["error"] is None:
            del d["error"]
        return d


def load_json_file(path: Path) -> Any:
    """Load and parse a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_artifact_files(source_dir: Path) -> tuple[dict[str, Path], list[str], list[str]]:
    """
    Find artifact files in the source directory.
    
    Returns:
        Tuple of (found_files, found_names, missing_names)
    """
    found: dict[str, Path] = {}
    found_names: list[str] = []
    missing_names: list[str] = []
    
    required = ["prompt_spec", "evidence_bundle", "reasoning_trace", "verdict"]
    optional = ["tool_plan", "por_bundle"]
    
    for key, filename in EXPECTED_FILES.items():
        file_path = source_dir / filename
        if file_path.exists():
            found[key] = file_path
            found_names.append(key)
        elif key in required:
            missing_names.append(key)
    
    return found, found_names, missing_names


def load_artifacts(files: dict[str, Path]) -> dict[str, Any]:
    """Load all artifacts from files."""
    artifacts = {}
    
    # Load prompt_spec
    if "prompt_spec" in files:
        data = load_json_file(files["prompt_spec"])
        artifacts["prompt_spec"] = PromptSpec.model_validate(data)
    
    # Load tool_plan (optional)
    if "tool_plan" in files:
        data = load_json_file(files["tool_plan"])
        artifacts["tool_plan"] = ToolPlan.model_validate(data)
    else:
        artifacts["tool_plan"] = None
    
    # Load evidence_bundle
    if "evidence_bundle" in files:
        data = load_json_file(files["evidence_bundle"])
        artifacts["evidence_bundle"] = EvidenceBundle.model_validate(data)
    
    # Load reasoning_trace
    if "reasoning_trace" in files:
        data = load_json_file(files["reasoning_trace"])
        artifacts["reasoning_trace"] = ReasoningTrace.model_validate(data)
    
    # Load verdict
    if "verdict" in files:
        data = load_json_file(files["verdict"])
        artifacts["verdict"] = DeterministicVerdict.model_validate(data)
    
    # Load por_bundle (optional - will be generated if missing)
    if "por_bundle" in files:
        data = load_json_file(files["por_bundle"])
        artifacts["por_bundle"] = PoRBundle.model_validate(data)
    else:
        artifacts["por_bundle"] = None
    
    return artifacts


def build_package(artifacts: dict[str, Any]) -> PoRPackage:
    """Build a PoRPackage from loaded artifacts."""
    prompt_spec = artifacts["prompt_spec"]
    tool_plan = artifacts["tool_plan"]
    evidence = artifacts["evidence_bundle"]
    trace = artifacts["reasoning_trace"]
    verdict = artifacts["verdict"]
    por_bundle = artifacts["por_bundle"]
    
    # Generate por_bundle if not provided
    if por_bundle is None:
        por_bundle = build_por_bundle(
            prompt_spec=prompt_spec,
            evidence=evidence,
            trace=trace,
            verdict=verdict,
        )
    
    return PoRPackage(
        bundle=por_bundle,
        prompt_spec=prompt_spec,
        tool_plan=tool_plan,
        evidence=evidence,
        trace=trace,
        verdict=verdict,
    )


def print_summary_human(summary: PackSummary) -> None:
    """Print summary in human-readable format."""
    if summary.success:
        print(f"Created pack: {summary.output_path}")
        print(f"market_id: {summary.market_id}")
        print(f"por_root: {summary.por_root}")
        print(f"files_included: {', '.join(summary.files_found)}")
    else:
        print(f"Failed to create pack", file=sys.stderr)
        if summary.error:
            print(f"Error: {summary.error}", file=sys.stderr)
        if summary.files_missing:
            print(f"Missing required files: {', '.join(summary.files_missing)}", file=sys.stderr)


def print_summary_json(summary: PackSummary) -> None:
    """Print summary as JSON."""
    print(json.dumps(summary.to_dict(), indent=2))


def pack_cmd(args: Namespace) -> int:
    """
    Execute the pack command.
    
    Args:
        args: Parsed command-line arguments
    
    Returns:
        Exit code
    """
    source_dir = Path(args.from_dir)
    out_path = Path(args.out)
    output_json = args.json
    
    summary = PackSummary(
        source_dir=str(source_dir),
        output_path=str(out_path),
    )
    
    # Check source directory exists
    if not source_dir.exists():
        summary.error = f"Source directory not found: {source_dir}"
        if output_json:
            print_summary_json(summary)
        else:
            print_summary_human(summary)
        return EXIT_RUNTIME_ERROR
    
    if not source_dir.is_dir():
        summary.error = f"Not a directory: {source_dir}"
        if output_json:
            print_summary_json(summary)
        else:
            print_summary_human(summary)
        return EXIT_RUNTIME_ERROR
    
    # Find artifact files
    files, found_names, missing_names = find_artifact_files(source_dir)
    summary.files_found = found_names
    summary.files_missing = missing_names
    
    if missing_names:
        summary.error = f"Missing required files: {', '.join(missing_names)}"
        if output_json:
            print_summary_json(summary)
        else:
            print_summary_human(summary)
        return EXIT_RUNTIME_ERROR
    
    # Load artifacts
    try:
        artifacts = load_artifacts(files)
    except Exception as e:
        summary.error = f"Failed to load artifacts: {e}"
        if output_json:
            print_summary_json(summary)
        else:
            print_summary_human(summary)
        return EXIT_RUNTIME_ERROR
    
    # Build package
    try:
        package = build_package(artifacts)
    except Exception as e:
        summary.error = f"Failed to build package: {e}"
        if output_json:
            print_summary_json(summary)
        else:
            print_summary_human(summary)
        return EXIT_RUNTIME_ERROR
    
    # Save pack
    try:
        saved_path = save_pack(package, out_path)
        summary.output_path = str(saved_path)
        summary.market_id = package.bundle.market_id
        summary.por_root = package.bundle.por_root or ""
        summary.success = True
    except Exception as e:
        summary.error = f"Failed to save pack: {e}"
        if output_json:
            print_summary_json(summary)
        else:
            print_summary_human(summary)
        return EXIT_RUNTIME_ERROR
    
    if output_json:
        print_summary_json(summary)
    else:
        print_summary_human(summary)
    
    return EXIT_SUCCESS