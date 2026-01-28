"""
Module 06 - Auditor Reasoning Submodule

Provides claim extraction and contradiction detection for the Auditor agent.

Public API:
- Claim, ClaimSet: Data structures for extracted claims
- ClaimExtractor, extract_claims: Claim extraction from evidence
- ContradictionChecker, check_contradictions: Contradiction detection
- Contradiction: Detected contradiction data structure
"""

from .claim_extraction import (
    Claim,
    ClaimExtractor,
    ClaimSet,
    extract_claims,
)
from .contradiction_checks import (
    Contradiction,
    ContradictionChecker,
    check_contradictions,
)
from .dspy_programs import (
    DSPY_AVAILABLE,
    DSPyClaimExtractor,
    DSPyContradictionChecker,
    is_dspy_available,
)

__all__ = [
    # Claim extraction
    "Claim",
    "ClaimSet",
    "ClaimExtractor",
    "extract_claims",
    # Contradiction detection
    "Contradiction",
    "ContradictionChecker",
    "check_contradictions",
    # DSPy (optional)
    "DSPY_AVAILABLE",
    "DSPyClaimExtractor",
    "DSPyContradictionChecker",
    "is_dspy_available",
]