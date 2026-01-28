"""
Module 08 - Validator/Sentinel (Replay Executor & Challenges)

This module provides an independent validator (sentinel) that can:
1. Verify a provided PoR package end-to-end
2. Optionally replay evidence collection deterministically
3. Detect mismatches and policy failures
4. Generate pinpoint challenges with Merkle proofs for dispute resolution

Public API:
- SentinelAgent: Main validator agent with verify() method
- PoRPackage: Container for PoR artifacts
- Challenge: Serializable challenge packet for disputes
- ReplayExecutor: Evidence replay for verification

Owner: Protocol Verification Engineer
Module ID: M08
"""

from .challenges import (
    Challenge,
    ChallengeKindType,
    MerkleProofData,
    make_merkle_leaf_challenge,
    make_por_root_challenge,
    make_prompt_hash_challenge,
    make_replay_divergence_challenge,
    make_root_mismatch_challenge,
    make_verdict_hash_challenge,
)
from .pinpoint import (
    create_evidence_mismatch_challenge,
    create_reasoning_mismatch_challenge,
)
from .replay_executor import (
    CollectorProtocol,
    ReplayExecutor,
    ReplayResult,
)
from .sentinel_agent import (
    PoRPackage,
    SentinelAgent,
    SentinelVerificationResult,
    VerifyMode,
)
from .verification_helpers import (
    check_evaluation_variables,
    check_output_schema_ref,
    check_strict_mode_flag,
    verify_commitments,
    verify_verdict_integrity,
)

__all__ = [
    # Main agent
    "SentinelAgent",
    "PoRPackage",
    "SentinelVerificationResult",
    "VerifyMode",
    # Challenge types
    "Challenge",
    "ChallengeKindType",
    "MerkleProofData",
    # Challenge constructors
    "make_prompt_hash_challenge",
    "make_verdict_hash_challenge",
    "make_merkle_leaf_challenge",
    "make_root_mismatch_challenge",
    "make_por_root_challenge",
    "make_replay_divergence_challenge",
    # Pinpoint helpers
    "create_evidence_mismatch_challenge",
    "create_reasoning_mismatch_challenge",
    # Verification helpers
    "check_strict_mode_flag",
    "check_output_schema_ref",
    "check_evaluation_variables",
    "verify_commitments",
    "verify_verdict_integrity",
    # Replay
    "ReplayExecutor",
    "ReplayResult",
    "CollectorProtocol",
]