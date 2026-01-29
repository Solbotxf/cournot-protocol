"""
Module 09A - Pipeline Integration (In-Process Runtime Wiring)

This module provides a deterministic, testable, in-process pipeline runner
that composes Modules 04-08 into an executable flow.

Key features:
- Registry-based agent selection (no direct agent instantiation)
- Fail-closed in production mode when capabilities are missing
- Configurable step overrides for testing and customization

Public API:
- Pipeline: Main pipeline runner class
- PipelineConfig: Configuration for pipeline execution
- ExecutionMode: Pipeline execution mode (production/development/test)
- StepOverrides: Override specific agents for each step
- RunResult: Complete result of a pipeline run
- CapabilityError: Raised when required capabilities are missing
- SOPExecutor: Step executor for composable pipeline steps
- PipelineState: State container for pipeline execution
- resolve_market: Translate verdict to market resolution record
"""

from orchestrator.pipeline import (
    CapabilityError,
    ExecutionMode,
    Pipeline,
    PipelineConfig,
    PoRPackage,
    RunResult,
    StepOverrides,
    create_pipeline,
    create_production_pipeline,
    create_test_pipeline,
)
from orchestrator.sop_executor import (
    FunctionStep,
    PipelineState,
    SOPExecutor,
    SOPStep,
    make_step,
)
from orchestrator.market_resolver import (
    MarketResolution,
    create_resolution,
    is_definitive_resolution,
    resolution_summary,
    resolve_market,
)


__all__ = [
    # Main pipeline
    "Pipeline",
    "PipelineConfig",
    "ExecutionMode",
    "StepOverrides",
    "RunResult",
    "PoRPackage",
    "CapabilityError",
    # Factory functions
    "create_pipeline",
    "create_production_pipeline",
    "create_test_pipeline",
    # SOP executor
    "SOPExecutor",
    "SOPStep",
    "FunctionStep",
    "PipelineState",
    "make_step",
    # Market resolver
    "MarketResolution",
    "resolve_market",
    "create_resolution",
    "is_definitive_resolution",
    "resolution_summary",
]
