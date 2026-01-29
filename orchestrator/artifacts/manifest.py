"""
Module 09B - Artifact Packaging & IO
File: manifest.py

Purpose: Manifest specification for artifact packs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from core.schemas.versioning import SCHEMA_VERSION

FORMAT_VERSION = "pack.v1"
PROTOCOL_VERSION = SCHEMA_VERSION

REQUIRED_FILES = frozenset({
    "prompt_spec",
    "evidence_bundle",
    "reasoning_trace",
    "verdict",
    "por_bundle",
})

OPTIONAL_FILES = frozenset({
    "tool_plan",
})

ALL_ARTIFACT_FILES = REQUIRED_FILES | OPTIONAL_FILES


@dataclass
class ManifestFileEntry:
    """Entry describing a single file in the pack."""
    path: str
    sha256: str
    size: int  # bytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "bytes": self.size,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ManifestFileEntry":
        return cls(
            path=data["path"],
            sha256=data["sha256"],
            size=data.get("bytes", data.get("size", 0)),
        )


@dataclass
class PackManifest:
    """Manifest describing an artifact pack."""
    format_version: str = FORMAT_VERSION
    protocol_version: str = PROTOCOL_VERSION
    market_id: str = ""
    por_root: str = ""
    files: dict[str, ManifestFileEntry] = field(default_factory=dict)
    created_at: str | None = None
    notes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "protocol_version": self.protocol_version,
            "market_id": self.market_id,
            "por_root": self.por_root,
            "files": {k: v.to_dict() for k, v in self.files.items()},
            "created_at": self.created_at,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PackManifest":
        files = {}
        for key, entry_data in data.get("files", {}).items():
            files[key] = ManifestFileEntry.from_dict(entry_data)
        return cls(
            format_version=data.get("format_version", FORMAT_VERSION),
            protocol_version=data.get("protocol_version", PROTOCOL_VERSION),
            market_id=data.get("market_id", ""),
            por_root=data.get("por_root", ""),
            files=files,
            created_at=data.get("created_at"),
            notes=data.get("notes", {}),
        )

    def add_file(self, key: str, path: str, sha256: str, size: int) -> None:
        """Add a file entry to the manifest."""
        self.files[key] = ManifestFileEntry(path=path, sha256=sha256, size=size)

    def get_file(self, key: str) -> ManifestFileEntry | None:
        """Get a file entry by key."""
        return self.files.get(key)

    def has_required_files(self) -> tuple[bool, list[str]]:
        """Check if all required files are present."""
        missing = [f for f in REQUIRED_FILES if f not in self.files]
        return len(missing) == 0, missing

    def get_file_keys(self) -> set[str]:
        """Get all file keys in manifest."""
        return set(self.files.keys())


def parse_format_version(version: str) -> tuple[str, int]:
    """Parse format version string into (prefix, version_number)."""
    if "." not in version:
        return version, 0
    parts = version.rsplit(".", 1)
    prefix = parts[0]
    try:
        num = int(parts[1].lstrip("v"))
    except ValueError:
        num = 0
    return prefix, num


def is_compatible_format_version(version: str) -> bool:
    """Check if format version is compatible with current implementation."""
    prefix, num = parse_format_version(version)
    current_prefix, current_num = parse_format_version(FORMAT_VERSION)
    # Same prefix required, version must be <= current
    return prefix == current_prefix and num <= current_num


__all__ = [
    "FORMAT_VERSION",
    "PROTOCOL_VERSION",
    "REQUIRED_FILES",
    "OPTIONAL_FILES",
    "ALL_ARTIFACT_FILES",
    "ManifestFileEntry",
    "PackManifest",
    "parse_format_version",
    "is_compatible_format_version",
]
