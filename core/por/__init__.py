"""
Module 03 - Proof of Reasoning (PoR) Bundle & Commitments

This module provides:
- ReasoningTrace: The canonical trace format for Auditor reasoning
- PoRBundle: The verifiable package containing commitments and verdict
- Root computation functions for deterministic Merkle commitments
- Bundle building and verification utilities

Public API:
- Models: ReasoningTrace, ReasoningStep, TracePolicy, PoRBundle, TEEAttestation
- Types: PoRRoots
- Exceptions: PromptSpecTimestampError
- Functions: compute_roots, build_por_bundle, verify_por_bundle
"""

from core.por.por_bundle import PoRBundle, TEEAttestation
from core.por.proof_of_reasoning import (
    PoRRoots,
    PromptSpecTimestampError,
    build_por_bundle,
    compute_evidence_leaf_hashes,
    compute_evidence_root,
    compute_por_root,
    compute_prompt_spec_hash,
    compute_reasoning_leaf_hashes,
    compute_reasoning_root,
    compute_roots,
    compute_verdict_hash,
    get_evidence_leaf_hash,
    get_reasoning_leaf_hash,
    verify_por_bundle,
    verify_por_bundle_structure,
)
from core.por.reasoning_trace import ReasoningStep, ReasoningTrace, TracePolicy

__all__ = [
    # Models
    "ReasoningTrace",
    "ReasoningStep",
    "TracePolicy",
    "PoRBundle",
    "TEEAttestation",
    # Types
    "PoRRoots",
    # Exceptions
    "PromptSpecTimestampError",
    # Root computation
    "compute_roots",
    "compute_prompt_spec_hash",
    "compute_evidence_root",
    "compute_evidence_leaf_hashes",
    "compute_reasoning_root",
    "compute_reasoning_leaf_hashes",
    "compute_verdict_hash",
    "compute_por_root",
    # Bundle operations
    "build_por_bundle",
    "verify_por_bundle",
    "verify_por_bundle_structure",
    # Leaf helpers
    "get_evidence_leaf_hash",
    "get_reasoning_leaf_hash",
]