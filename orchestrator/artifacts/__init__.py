"""
Module 09B - Artifact Packaging & IO

Provides functionality for saving, loading, and validating artifact packs.
"""

from orchestrator.artifacts.manifest import (
    FORMAT_VERSION,
    PROTOCOL_VERSION,
    REQUIRED_FILES,
    OPTIONAL_FILES,
    ALL_ARTIFACT_FILES,
    ManifestFileEntry,
    PackManifest,
    parse_format_version,
    is_compatible_format_version,
)

from orchestrator.artifacts.io import (
    PackIOError,
    PackHashMismatchError,
    PackMissingFileError,
    dump_json,
    compute_sha256,
    save_pack_dir,
    save_pack_zip,
    save_pack,
    load_pack_dir,
    load_pack_zip,
    load_pack,
    load_manifest,
    validate_pack_files,
)

from orchestrator.artifacts.pack import (
    validate_pack,
    validate_market_id_consistency,
    validate_verdict_hash,
    validate_prompt_spec_hash,
    validate_evidence_root,
    validate_reasoning_root,
    validate_por_root,
    validate_embedded_verdict,
    validate_evidence_references,
    validate_manifest_consistency,
)

__all__ = [
    # Manifest
    "FORMAT_VERSION",
    "PROTOCOL_VERSION",
    "REQUIRED_FILES",
    "OPTIONAL_FILES",
    "ALL_ARTIFACT_FILES",
    "ManifestFileEntry",
    "PackManifest",
    "parse_format_version",
    "is_compatible_format_version",
    # IO
    "PackIOError",
    "PackHashMismatchError",
    "PackMissingFileError",
    "dump_json",
    "compute_sha256",
    "save_pack_dir",
    "save_pack_zip",
    "save_pack",
    "load_pack_dir",
    "load_pack_zip",
    "load_pack",
    "load_manifest",
    "validate_pack_files",
    # Validation
    "validate_pack",
    "validate_market_id_consistency",
    "validate_verdict_hash",
    "validate_prompt_spec_hash",
    "validate_evidence_root",
    "validate_reasoning_root",
    "validate_por_root",
    "validate_embedded_verdict",
    "validate_evidence_references",
    "validate_manifest_consistency",
]
