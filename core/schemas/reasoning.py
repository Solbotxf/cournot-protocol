"""
Module 01 - Schemas & Canonicalization
File: reasoning.py

Purpose: Reasoning trace schemas for the Auditor agent.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .versioning import SCHEMA_VERSION


# Step types in reasoning
StepType = Literal[
    "evidence_analysis",      # Analyzing a piece of evidence
    "comparison",             # Comparing multiple pieces of evidence
    "inference",              # Drawing an inference from evidence
    "rule_application",       # Applying a resolution rule
    "confidence_assessment",  # Assessing confidence level
    "conflict_resolution",    # Resolving conflicting evidence
    "threshold_check",        # Checking against a threshold
    "validity_check",         # Checking evidence validity
    "conclusion",             # Drawing a conclusion
]


class EvidenceRef(BaseModel):
    """Reference to a piece of evidence used in reasoning."""
    
    model_config = ConfigDict(extra="forbid")
    
    evidence_id: str = Field(..., description="ID of the referenced evidence")
    requirement_id: str | None = Field(default=None)
    source_id: str | None = Field(default=None)
    field_used: str | None = Field(default=None, description="Specific field used from evidence")
    value_at_reference: Any = Field(default=None, description="Value at time of reference")


class ReasoningStep(BaseModel):
    """
    A single step in the reasoning trace.
    
    Each step shows how the agent processes evidence toward a conclusion.
    """
    
    model_config = ConfigDict(extra="forbid")
    
    step_id: str = Field(..., description="Unique identifier for this step", min_length=1)
    step_type: StepType = Field(..., description="Type of reasoning step")
    description: str = Field(..., description="Human-readable description of this step")
    
    # Evidence references
    evidence_refs: list[EvidenceRef] = Field(
        default_factory=list,
        description="Evidence items referenced in this step",
    )
    
    # Rule reference (if applying a rule)
    rule_id: str | None = Field(default=None, description="Resolution rule being applied")
    
    # Input/output of this step
    input_summary: str | None = Field(default=None, description="Summary of inputs to this step")
    output_summary: str | None = Field(default=None, description="Summary of output from this step")
    
    # Intermediate conclusion
    conclusion: str | None = Field(default=None, description="Conclusion drawn in this step")
    confidence_delta: float | None = Field(
        default=None,
        ge=-1.0,
        le=1.0,
        description="Change in confidence from this step",
    )
    
    # Dependencies
    depends_on: list[str] = Field(
        default_factory=list,
        description="Step IDs this step depends on",
    )
    
    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConflictRecord(BaseModel):
    """Record of a conflict between evidence items."""
    
    model_config = ConfigDict(extra="forbid")
    
    conflict_id: str = Field(..., min_length=1)
    evidence_ids: list[str] = Field(..., min_length=2)
    description: str = Field(...)
    resolution: str | None = Field(default=None)
    resolution_rationale: str | None = Field(default=None)
    winning_evidence_id: str | None = Field(default=None)


class ReasoningTrace(BaseModel):
    """
    Complete reasoning trace from evidence to conclusion.
    
    This is the primary output of the Auditor agent.
    """
    
    model_config = ConfigDict(extra="forbid")
    
    schema_version: str = Field(default=SCHEMA_VERSION)
    trace_id: str = Field(..., description="Unique identifier for this trace", min_length=1)
    market_id: str = Field(..., description="Market being reasoned about", min_length=1)
    bundle_id: str = Field(..., description="Evidence bundle used", min_length=1)
    
    # Reasoning steps
    steps: list[ReasoningStep] = Field(default_factory=list)
    
    # Conflicts detected and resolved
    conflicts: list[ConflictRecord] = Field(default_factory=list)
    
    # Summary
    evidence_summary: str | None = Field(
        default=None,
        description="Summary of all evidence considered",
    )
    reasoning_summary: str | None = Field(
        default=None,
        description="Summary of reasoning process",
    )
    
    # Preliminary conclusion (for Judge to finalize)
    preliminary_outcome: Literal["YES", "NO", "INVALID", "UNCERTAIN"] | None = Field(
        default=None,
        description="Preliminary outcome before Judge review",
    )
    preliminary_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Preliminary confidence score",
    )
    recommended_rule_id: str | None = Field(
        default=None,
        description="Recommended resolution rule to apply",
    )
    
    # Timing (optional for determinism)
    created_at: datetime | None = Field(default=None)
    
    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict)
    
    @property
    def step_count(self) -> int:
        """Number of reasoning steps."""
        return len(self.steps)
    
    @property
    def has_conflicts(self) -> bool:
        """Check if any conflicts were detected."""
        return len(self.conflicts) > 0
    
    def get_step(self, step_id: str) -> ReasoningStep | None:
        """Get a step by ID."""
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None
    
    def get_steps_by_type(self, step_type: StepType) -> list[ReasoningStep]:
        """Get all steps of a given type."""
        return [s for s in self.steps if s.step_type == step_type]
    
    def get_evidence_refs(self) -> list[str]:
        """Get all unique evidence IDs referenced."""
        refs: set[str] = set()
        for step in self.steps:
            for ref in step.evidence_refs:
                refs.add(ref.evidence_id)
        return list(refs)
    
    def add_step(self, step: ReasoningStep) -> None:
        """Add a reasoning step."""
        self.steps.append(step)
    
    def add_conflict(self, conflict: ConflictRecord) -> None:
        """Add a conflict record."""
        self.conflicts.append(conflict)


class AuditResult(BaseModel):
    """
    Result of an audit operation.
    
    Wraps ReasoningTrace with execution status.
    """
    
    model_config = ConfigDict(extra="forbid")
    
    success: bool = Field(...)
    trace: ReasoningTrace | None = Field(default=None)
    error: str | None = Field(default=None)
    warnings: list[str] = Field(default_factory=list)
    
    @classmethod
    def from_trace(cls, trace: ReasoningTrace) -> "AuditResult":
        """Create a successful result from a trace."""
        return cls(success=True, trace=trace)
    
    @classmethod
    def failure(cls, error: str) -> "AuditResult":
        """Create a failure result."""
        return cls(success=False, error=error)
