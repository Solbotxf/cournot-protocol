"""
Module 01 - Schemas & Canonicalization
File: __init__.py

Purpose: Export the public API for the schemas module.
This is the main entry point for other modules to import schema definitions.
"""

# Version constants
from .versioning import (
    PROTOCOL_VERSION,
    SCHEMA_VERSION,
    SUPPORTED_PROTOCOL_VERSIONS,
    SUPPORTED_SCHEMA_VERSIONS,
    ProtocolVersion,
    SchemaVersion,
    UnsupportedProtocolVersionError,
    UnsupportedSchemaVersionError,
    assert_supported_protocol_version,
    assert_supported_schema_version,
    is_compatible_protocol_version,
    is_compatible_schema_version,
)

# Canonical serialization API
from .canonical import (
    CANONICAL_JSON_SEPARATORS,
    canonical_equals,
    canonicalize_value,
    dumps_canonical,
    ensure_utc,
    format_datetime_canonical,
    loads_canonical,
    normalize_datetime,
    to_canonical_json_dict,
)

# Error models and exceptions
from .errors import (
    CanonicalizationException,
    CournotError,
    CournotException,
    DeterminismException,
    ErrorCodes,
    MerkleVerificationException,
    ProvenanceError,
    ProvenanceException,
    SchemaValidationException,
    SOPExecutionException,
    TimeWindowError,
    TimeWindowException,
    TracePolicyException,
    ValidationError,
)

# Market schemas
from .market import (
    DisputePolicy,
    MarketSpec,
    ResolutionRule,
    ResolutionRules,
    ResolutionWindow,
    SourcePolicy,
)

# Transport schemas
from .transport import (
    ToolCallRecord,
    ToolExecutionLog,
    ToolPlan,
)

# Prompt schemas
from .prompts import (
    ContentTypeHint,
    DataRequirement,
    HttpMethod,
    PredictionSemantics,
    PromptSpec,
    PromptValidationResult,
    SelectionPolicy,
    SelectionStrategy,
    SourceTarget,
    TieBreakerStrategy,
)

# Evidence schemas
from .evidence import (
    CollectionResult,
    EvidenceBundle,
    EvidenceItem,
    Provenance,
    ProvenanceTier,
)

from .reasoning import (
    AuditResult,
    ConflictRecord,
    EvidenceRef,
    ReasoningStep,
    ReasoningTrace,
    StepType,
)

from .proof import (
    ProofBundle,
    SentinelReport,
)

# Verdict schemas
from .verdict import (
    DeterministicVerdict,
    Outcome,
    VerdictJustification,
    VerdictValidationResult,
)

# Verification schemas
from .verification import (
    ChallengeKind,
    ChallengeRef,
    CheckResult,
    CheckSeverity,
    ProvenanceCheck,
    TraceConsistencyCheck,
    VerificationResult,
)


# Define __all__ for explicit public API
__all__ = [
    # Versioning
    "SCHEMA_VERSION",
    "PROTOCOL_VERSION",
    "SUPPORTED_SCHEMA_VERSIONS",
    "SUPPORTED_PROTOCOL_VERSIONS",
    "SchemaVersion",
    "ProtocolVersion",
    "assert_supported_schema_version",
    "assert_supported_protocol_version",
    "is_compatible_schema_version",
    "is_compatible_protocol_version",
    "UnsupportedSchemaVersionError",
    "UnsupportedProtocolVersionError",
    # Canonical serialization
    "dumps_canonical",
    "to_canonical_json_dict",
    "canonicalize_value",
    "ensure_utc",
    "normalize_datetime",
    "format_datetime_canonical",
    "loads_canonical",
    "canonical_equals",
    "CANONICAL_JSON_SEPARATORS",
    # Errors
    "CournotError",
    "CournotException",
    "CanonicalizationException",
    "SchemaValidationException",
    "ProvenanceException",
    "TimeWindowException",
    "DeterminismException",
    "TracePolicyException",
    "MerkleVerificationException",
    "SOPExecutionException",
    "ErrorCodes",
    "ValidationError",
    "ProvenanceError",
    "TimeWindowError",
    # Market
    "MarketSpec",
    "ResolutionWindow",
    "ResolutionRule",
    "ResolutionRules",
    "SourcePolicy",
    "DisputePolicy",
    # Transport
    "ToolPlan",
    "ToolCallRecord",
    "ToolExecutionLog",
    # Prompts
    "PromptSpec",
    "PredictionSemantics",
    "DataRequirement",
    "PromptValidationResult",
    "SourceTarget",
    "SelectionPolicy",
    "HttpMethod",
    "ContentTypeHint",
    "SelectionStrategy",
    "TieBreakerStrategy",
    # Evidence
    "EvidenceBundle",
    "EvidenceItem",
    "SourceDescriptor",
    "RetrievalReceipt",
    "ProvenanceProof",
    "RetrievalMethod",
    "ProvenanceKind",
    "ContentType",
    # Verdict
    "DeterministicVerdict",
    "Outcome",
    "VerdictJustification",
    "VerdictValidationResult",
    # Verification
    "VerificationResult",
    "CheckResult",
    "ChallengeRef",
    "CheckSeverity",
    "ChallengeKind",
    "ProvenanceCheck",
    "TraceConsistencyCheck",
]