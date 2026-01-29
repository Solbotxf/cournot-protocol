"""
Module 09A - SOP Executor

Purpose: Keep pipeline steps composable and testable with minimal abstraction.

Provides:
- SOPStep: Protocol for individual pipeline steps
- PipelineState: Dataclass holding artifacts incrementally
- SOPExecutor: Runner that executes steps in sequence

This abstraction allows later modules (09B API, 09C persistence) to be added
without rewriting the core flow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, TYPE_CHECKING

from core.por.por_bundle import PoRBundle
from core.por.proof_of_reasoning import PoRRoots
from core.por.reasoning_trace import ReasoningTrace
from core.schemas.evidence import EvidenceBundle
from core.schemas.prompts import PromptSpec
from core.schemas.transport import ToolExecutionLog, ToolPlan
from core.schemas.verdict import DeterministicVerdict
from core.schemas.verification import CheckResult, VerificationResult

if TYPE_CHECKING:
    from agents.context import AgentContext


@dataclass
class PipelineState:
    """
    Holds artifacts incrementally as pipeline progresses.
    
    Each step may read from and write to this state.
    Fields are Optional to allow incremental population.
    """
    
    # Input
    user_input: str = ""
    
    # Agent context (set by pipeline runner)
    context: Optional["AgentContext"] = None
    
    # Step 1: Prompt compilation
    prompt_spec: Optional[PromptSpec] = None
    tool_plan: Optional[ToolPlan] = None
    
    # Step 2: Evidence collection
    evidence_bundle: Optional[EvidenceBundle] = None
    execution_log: Optional[ToolExecutionLog] = None
    
    # Step 3: Audit / reasoning trace
    audit_trace: Optional[ReasoningTrace] = None
    audit_verification: Optional[VerificationResult] = None
    
    # Step 4: Judge verdict
    verdict: Optional[DeterministicVerdict] = None
    judge_verification: Optional[VerificationResult] = None
    
    # Step 5: PoR bundle
    roots: Optional[PoRRoots] = None
    por_bundle: Optional[PoRBundle] = None
    
    # Step 6: Sentinel verification (optional)
    sentinel_verification: Optional[VerificationResult] = None
    challenges: Optional[list[Any]] = None
    
    # Aggregated results
    checks: list[CheckResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    ok: bool = True
    
    def add_check(self, check: CheckResult) -> None:
        """Add a check result to the aggregated checks."""
        self.checks.append(check)
        if not check.ok and check.severity == "error":
            self.ok = False
    
    def add_checks(self, checks: list[CheckResult]) -> None:
        """Add multiple check results."""
        for check in checks:
            self.add_check(check)
    
    def add_error(self, error: str) -> None:
        """Add an error message and mark state as not ok."""
        self.errors.append(error)
        self.ok = False
    
    def merge_verification(self, result: VerificationResult) -> None:
        """Merge a verification result into the state."""
        self.add_checks(result.checks)
        if not result.ok:
            self.ok = False
            if result.error:
                self.add_error(result.error.message)


class SOPStep(Protocol):
    """
    Protocol for a single pipeline step.
    
    Each step has a name and a run method that transforms state.
    Steps should be side-effect free except for state mutation.
    """
    
    @property
    def name(self) -> str:
        """Unique name for this step."""
        ...
    
    def run(self, state: PipelineState) -> PipelineState:
        """
        Execute this step, potentially modifying state.
        
        Args:
            state: Current pipeline state
            
        Returns:
            Updated pipeline state (may be the same object)
        """
        ...


@dataclass
class FunctionStep:
    """
    Adapter to create SOPStep from a plain function.
    
    Example:
        step = FunctionStep("my_step", lambda s: do_something(s))
    """
    
    _name: str
    _func: Callable[[PipelineState], PipelineState]
    
    @property
    def name(self) -> str:
        return self._name
    
    def run(self, state: PipelineState) -> PipelineState:
        return self._func(state)


class SOPExecutor:
    """
    Executor that runs a sequence of SOPSteps.
    
    Provides:
    - Sequential execution of steps
    - Error handling and state tracking
    - Step timing (optional)
    """
    
    def __init__(self, *, stop_on_error: bool = True):
        """
        Initialize executor.
        
        Args:
            stop_on_error: If True, stop execution on first step error.
                          If False, continue and aggregate errors.
        """
        self.stop_on_error = stop_on_error
        self._step_results: list[tuple[str, bool, Optional[str]]] = []
    
    def execute(
        self,
        steps: list[SOPStep],
        state: PipelineState,
    ) -> PipelineState:
        """
        Execute all steps in sequence.
        
        Args:
            steps: List of steps to execute
            state: Initial pipeline state
            
        Returns:
            Final pipeline state after all steps
        """
        self._step_results = []
        
        for step in steps:
            try:
                state = step.run(state)
                self._step_results.append((step.name, True, None))
                
                # Check if we should stop on error
                if self.stop_on_error and not state.ok:
                    break
                    
            except Exception as e:
                error_msg = f"Step '{step.name}' failed: {e}"
                state.add_error(error_msg)
                self._step_results.append((step.name, False, str(e)))
                
                if self.stop_on_error:
                    break
        
        return state
    
    @property
    def step_results(self) -> list[tuple[str, bool, Optional[str]]]:
        """
        Get results of each step execution.
        
        Returns:
            List of (step_name, success, error_message) tuples
        """
        return self._step_results.copy()
    
    def get_failed_steps(self) -> list[str]:
        """Get names of failed steps."""
        return [name for name, success, _ in self._step_results if not success]
    
    def all_steps_succeeded(self) -> bool:
        """Check if all executed steps succeeded."""
        return all(success for _, success, _ in self._step_results)


def make_step(name: str, func: Callable[[PipelineState], PipelineState]) -> SOPStep:
    """
    Convenience function to create a step from a function.
    
    Args:
        name: Step name
        func: Function that takes and returns PipelineState
        
    Returns:
        SOPStep wrapping the function
    """
    return FunctionStep(name, func)
