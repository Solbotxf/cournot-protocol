"""
Module 03 - Reasoning Trace Models

Defines the canonical trace format that the Auditor emits. This is what gets
Merkle-committed so a challenger can pinpoint a broken step.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from core.schemas.versioning import SCHEMA_VERSION


class TracePolicy(BaseModel):
    """
    Policy governing how reasoning traces should be constructed and validated.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(default=SCHEMA_VERSION)
    decoding_policy: str = Field(
        default="strict",
        description="Policy for decoding LLM outputs: 'strict', 'lenient', etc.",
    )
    allow_external_sources: bool = Field(
        default=False,
        description="Whether steps can reference external sources not in the evidence bundle",
    )
    max_steps: int = Field(
        default=200,
        ge=1,
        description="Maximum number of reasoning steps allowed",
    )
    extra: dict[str, Any] = Field(default_factory=dict)


class ReasoningStep(BaseModel):
    """
    A single step in the reasoning trace.

    Each step references evidence, prior steps, and produces an output.
    Steps are ordered and form a DAG where each step can only reference
    prior steps (by step_id).
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(default=SCHEMA_VERSION)
    step_id: str = Field(
        ...,
        description="Deterministic identifier, e.g., 'step_0001'",
        min_length=1,
    )
    type: Literal["search", "extract", "check", "deduce", "aggregate", "map"] = Field(
        ...,
        description="Type of reasoning operation performed",
    )
    inputs: dict[str, Any] = Field(
        default_factory=dict,
        description="Input data for this step (evidence refs, prior outputs, etc.)",
    )
    action: str = Field(
        ...,
        description="Description of the action performed in this step",
    )
    output: dict[str, Any] = Field(
        default_factory=dict,
        description="Output produced by this step",
    )
    evidence_ids: list[str] = Field(
        default_factory=list,
        description="Evidence IDs from the EvidenceBundle that this step references",
    )
    prior_step_ids: list[str] = Field(
        default_factory=list,
        description="Step IDs that this step depends on (must be earlier steps)",
    )
    hash_commitment: Optional[str] = Field(
        default=None,
        description="Optional hash commitment for this step (can be filled later)",
    )


class ReasoningTrace(BaseModel):
    """
    Complete reasoning trace produced by the Auditor.

    Contains an ordered list of ReasoningSteps that form a directed acyclic graph
    of reasoning from evidence to conclusion. The trace is Merkle-committed so
    individual steps can be challenged.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(default=SCHEMA_VERSION)
    trace_id: str = Field(
        ...,
        description="Unique identifier for this trace",
        min_length=1,
    )
    policy: TracePolicy = Field(
        default_factory=TracePolicy,
        description="Policy that governs this trace's construction",
    )
    steps: list[ReasoningStep] = Field(
        default_factory=list,
        description="Ordered list of reasoning steps",
    )
    evidence_refs: list[str] = Field(
        default_factory=list,
        description="All evidence IDs referenced by this trace (subset of EvidenceBundle)",
    )

    @field_validator("evidence_refs")
    @classmethod
    def validate_evidence_refs_unique(cls, v: list[str]) -> list[str]:
        """Ensure evidence_refs are unique."""
        if len(v) != len(set(v)):
            raise ValueError("evidence_refs must contain unique values")
        return v

    @model_validator(mode="after")
    def validate_trace_structure(self) -> "ReasoningTrace":
        """
        Validate the structural integrity of the trace:
        1. Number of steps doesn't exceed policy.max_steps
        2. All step_ids are unique
        3. All prior_step_ids reference earlier steps
        """
        # Check max_steps
        if len(self.steps) > self.policy.max_steps:
            raise ValueError(
                f"Number of steps ({len(self.steps)}) exceeds "
                f"policy.max_steps ({self.policy.max_steps})"
            )

        # Build a set of seen step_ids to validate uniqueness and ordering
        seen_step_ids: set[str] = set()

        for i, step in enumerate(self.steps):
            # Check step_id uniqueness
            if step.step_id in seen_step_ids:
                raise ValueError(f"Duplicate step_id found: '{step.step_id}'")

            # Check that all prior_step_ids reference earlier steps
            for prior_id in step.prior_step_ids:
                if prior_id not in seen_step_ids:
                    raise ValueError(
                        f"Step '{step.step_id}' references prior_step_id '{prior_id}' "
                        f"which is not an earlier step"
                    )

            seen_step_ids.add(step.step_id)

        return self

    def get_step_by_id(self, step_id: str) -> Optional[ReasoningStep]:
        """Get a step by its ID."""
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None

    def get_all_evidence_ids(self) -> set[str]:
        """Get all evidence IDs referenced across all steps."""
        evidence_ids: set[str] = set()
        for step in self.steps:
            evidence_ids.update(step.evidence_ids)
        return evidence_ids

    def validate_evidence_coverage(self, bundle_evidence_ids: set[str]) -> list[str]:
        """
        Validate that all evidence IDs in the trace exist in the bundle.
        Returns list of invalid evidence IDs (empty if all valid).
        """
        trace_evidence_ids = self.get_all_evidence_ids()
        trace_evidence_ids.update(self.evidence_refs)
        invalid_ids = trace_evidence_ids - bundle_evidence_ids
        return list(invalid_ids)