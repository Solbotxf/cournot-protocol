"""
Module 01 - Schemas & Canonicalization
File: versioning.py

Purpose: Centralize protocol/schema version constants.
This file must remain tiny (<80 LOC) and have no imports from other schema files
to avoid circular dependencies.
"""

from typing import Literal

# Current schema version - used across all models
SCHEMA_VERSION: str = "v1"

# Protocol version for wire format compatibility
PROTOCOL_VERSION: str = "v1"

# Type alias for schema version (future-proof for migrations)
SchemaVersion = Literal["v1"]

# Type alias for protocol version
ProtocolVersion = Literal["v1"]

# Supported versions for forward compatibility
SUPPORTED_SCHEMA_VERSIONS: frozenset[str] = frozenset({"v1"})
SUPPORTED_PROTOCOL_VERSIONS: frozenset[str] = frozenset({"v1"})


class UnsupportedSchemaVersionError(ValueError):
    """Raised when an unsupported schema version is encountered."""

    def __init__(self, version: str, supported: frozenset[str] | None = None) -> None:
        self.version = version
        self.supported = supported or SUPPORTED_SCHEMA_VERSIONS
        super().__init__(
            f"Unsupported schema version: '{version}'. "
            f"Supported versions: {sorted(self.supported)}"
        )


class UnsupportedProtocolVersionError(ValueError):
    """Raised when an unsupported protocol version is encountered."""

    def __init__(self, version: str, supported: frozenset[str] | None = None) -> None:
        self.version = version
        self.supported = supported or SUPPORTED_PROTOCOL_VERSIONS
        super().__init__(
            f"Unsupported protocol version: '{version}'. "
            f"Supported versions: {sorted(self.supported)}"
        )


def assert_supported_schema_version(version: str) -> None:
    """
    Validate that the given schema version is supported.

    Args:
        version: The schema version string to validate.

    Raises:
        UnsupportedSchemaVersionError: If the version is not supported.
    """
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise UnsupportedSchemaVersionError(version)


def assert_supported_protocol_version(version: str) -> None:
    """
    Validate that the given protocol version is supported.

    Args:
        version: The protocol version string to validate.

    Raises:
        UnsupportedProtocolVersionError: If the version is not supported.
    """
    if version not in SUPPORTED_PROTOCOL_VERSIONS:
        raise UnsupportedProtocolVersionError(version)


def is_compatible_schema_version(version: str) -> bool:
    """Check if a schema version is compatible without raising."""
    return version in SUPPORTED_SCHEMA_VERSIONS


def is_compatible_protocol_version(version: str) -> bool:
    """Check if a protocol version is compatible without raising."""
    return version in SUPPORTED_PROTOCOL_VERSIONS