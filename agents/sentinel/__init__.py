"""
Sentinel Agent Module

Verifies complete proof bundles for correctness.

Usage:
    from agents.sentinel import verify_proof, verify_artifacts, SentinelStrict
    
    # Using convenience function with ProofBundle
    result = verify_proof(ctx, proof_bundle)
    verification_result, sentinel_report = result.output
    
    # Using convenience function with individual artifacts
    result = verify_artifacts(
        ctx, prompt_spec, tool_plan, evidence_bundle,
        reasoning_trace, verdict
    )
    
    # Using specific agent
    sentinel = SentinelStrict()
    result = sentinel.run(ctx, proof_bundle)
"""

from .agent import (
    SentinelStrict,
    SentinelBasic,
    build_proof_bundle,
    get_sentinel,
    verify_artifacts,
    verify_proof,
)

from .engine import VerificationEngine

__all__ = [
    # Agents
    "SentinelStrict",
    "SentinelBasic",
    # Functions
    "get_sentinel",
    "verify_proof",
    "verify_artifacts",
    "build_proof_bundle",
    # Engine
    "VerificationEngine",
]
