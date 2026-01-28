"""
Module 07 - Judge Verification Submodule
Exports verification utilities for the Judge agent.

Owner: Protocol/Judging Engineer
Module ID: M07
"""

from .deterministic_mapping import (
    EvaluationVariables,
    ConfidencePolicy,
    extract_evaluation_variables,
    check_validity,
    check_conflicts,
    determine_binary_outcome,
    compute_confidence,
    map_to_verdict,
)

from .schema_lock import (
    SchemaLock,
    EXPECTED_OUTPUT_SCHEMA_REF,
    VALID_OUTCOMES,
)


__all__ = [
    # Deterministic mapping
    "EvaluationVariables",
    "ConfidencePolicy",
    "extract_evaluation_variables",
    "check_validity",
    "check_conflicts",
    "determine_binary_outcome",
    "compute_confidence",
    "map_to_verdict",
    # Schema lock
    "SchemaLock",
    "EXPECTED_OUTPUT_SCHEMA_REF",
    "VALID_OUTCOMES",
]