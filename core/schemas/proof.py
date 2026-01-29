"""
Module 01 - Schemas & Canonicalization
File: proof.py

Purpose: Proof bundle schemas for complete pipeline verification.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .versioning import SCHEMA_VERSION
from .prompts import PromptSpec
from .evidence import EvidenceBundle
from .reasoning import ReasoningTrace
from .verdict import DeterministicVerdict
from .transport import ToolPlan, ToolExecutionLog


class ProofBundle(BaseModel):
    """
    Complete proof bundle containing all artifacts for verification.
    
    This is the primary input to the Sentinel for full verification.
    """
    
    model_config = ConfigDict(extra="forbid")
    
    schema_version: str = Field(default=SCHEMA_VERSION)
    bundle_id: str = Field(..., description="Unique identifier for this proof bundle", min_length=1)
    market_id: str = Field(..., description="Market this proof is for", min_length=1)
    
    # Pipeline artifacts
    prompt_spec: PromptSpec = Field(..., description="The compiled prompt specification")
    tool_plan: ToolPlan = Field(..., description="The tool execution plan")
    evidence_bundle: EvidenceBundle = Field(..., description="Collected evidence")
    execution_log: ToolExecutionLog | None = Field(default=None, description="Tool execution log")
    reasoning_trace: ReasoningTrace = Field(..., description="Reasoning trace from auditor")
    verdict: DeterministicVerdict = Field(..., description="Final verdict from judge")
    
    # Metadata
    created_at: datetime | None = Field(default=None)
    pipeline_version: str = Field(default="v1")
    metadata: dict[str, Any] = Field(default_factory=dict)
    
    @property
    def is_complete(self) -> bool:
        """Check if all required artifacts are present."""
        return all([
            self.prompt_spec is not None,
            self.tool_plan is not None,
            self.evidence_bundle is not None,
            self.reasoning_trace is not None,
            self.verdict is not None,
        ])
    
    @property
    def artifact_count(self) -> int:
        """Count of artifacts in bundle."""
        count = 0
        if self.prompt_spec:
            count += 1
        if self.tool_plan:
            count += 1
        if self.evidence_bundle:
            count += 1
        if self.execution_log:
            count += 1
        if self.reasoning_trace:
            count += 1
        if self.verdict:
            count += 1
        return count


class SentinelReport(BaseModel):
    """
    Complete verification report from the Sentinel.
    
    Contains all check results and final verification status.
    """
    
    model_config = ConfigDict(extra="forbid")
    
    schema_version: str = Field(default=SCHEMA_VERSION)
    report_id: str = Field(..., min_length=1)
    bundle_id: str = Field(..., min_length=1)
    market_id: str = Field(..., min_length=1)
    
    # Overall status
    verified: bool = Field(..., description="Whether the proof bundle passed verification")
    
    # Check categories
    completeness_checks: list[dict[str, Any]] = Field(default_factory=list)
    hash_checks: list[dict[str, Any]] = Field(default_factory=list)
    consistency_checks: list[dict[str, Any]] = Field(default_factory=list)
    provenance_checks: list[dict[str, Any]] = Field(default_factory=list)
    reasoning_checks: list[dict[str, Any]] = Field(default_factory=list)
    
    # Summary
    total_checks: int = Field(default=0)
    passed_checks: int = Field(default=0)
    failed_checks: int = Field(default=0)
    warning_checks: int = Field(default=0)
    
    # Errors
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    
    # Timing
    verification_time_ms: float | None = Field(default=None)
    created_at: datetime | None = Field(default=None)
    
    @property
    def pass_rate(self) -> float:
        """Get the pass rate of checks."""
        if self.total_checks == 0:
            return 0.0
        return self.passed_checks / self.total_checks
    
    def add_check(
        self,
        category: str,
        check_id: str,
        passed: bool,
        message: str,
        severity: str = "error",
    ) -> None:
        """Add a check result to the report."""
        check = {
            "check_id": check_id,
            "passed": passed,
            "message": message,
            "severity": severity,
        }
        
        # Add to appropriate category
        if category == "completeness":
            self.completeness_checks.append(check)
        elif category == "hash":
            self.hash_checks.append(check)
        elif category == "consistency":
            self.consistency_checks.append(check)
        elif category == "provenance":
            self.provenance_checks.append(check)
        elif category == "reasoning":
            self.reasoning_checks.append(check)
        
        # Update counts
        self.total_checks += 1
        if passed:
            self.passed_checks += 1
        elif severity == "warn":
            self.warning_checks += 1
            self.warnings.append(message)
        else:
            self.failed_checks += 1
            self.errors.append(message)
            self.verified = False
