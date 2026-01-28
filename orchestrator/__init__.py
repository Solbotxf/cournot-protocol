"""
Module 09A - Pipeline Integration (In-Process Runtime Wiring)

This module provides a deterministic, testable, in-process pipeline runner
that composes Modules 04-08 into an executable flow.

Public API:
- Pipeline: Main pipeline runner class
- PipelineConfig: Configuration for pipeline execution
- RunResult: Complete result of a pipeline run
- SOPExecutor: Step executor for composable pipeline steps
- PipelineState: State container for pipeline execution
- resolve_market: Translate verdict to market resolution record
"""

from orchestrator.pipeline import (
    Pipeline,
    PipelineConfig,
    RunResult,
)
from orchestrator.sop_executor import (
    PipelineState,
    SOPExecutor,
    SOPStep,
)
from orchestrator.market_resolver import resolve_market
from .prompt_engineer import compile_prompt, create_strict_compiler, validate_prompt_spec


__all__ = [
    # Main pipeline
    "Pipeline",
    "PipelineConfig",
    "RunResult",
    # SOP executor
    "SOPExecutor",
    "SOPStep",
    "PipelineState",
    # Market resolver
    "resolve_market",
    # Prompt engineer
    "compile_prompt",
    "validate_prompt_spec",
    "create_strict_compiler",
]