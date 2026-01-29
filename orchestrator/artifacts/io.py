"""
Module 09B - Artifact Packaging & IO
File: io.py

Purpose: Save and load artifact packs to/from disk (directory or zip).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from core.schemas.canonical import dumps_canonical
from core.schemas.evidence import EvidenceBundle
from core.schemas.prompts import PromptSpec
from core.schemas.transport import ToolPlan
from core.schemas.verdict import DeterministicVerdict
from core.schemas.verification import CheckResult, VerificationResult
from core.por.por_bundle import PoRBundle
from core.por.reasoning_trace import ReasoningTrace

from orchestrator.pipeline import PoRPackage
from orchestrator.artifacts.manifest import (
    PackManifest, ManifestFileEntry, FORMAT_VERSION,
    REQUIRED_FILES, is_compatible_format_version,
)


# File name constants
MANIFEST_FILE = "manifest.json"
PROMPT_SPEC_FILE = "prompt_spec.json"
TOOL_PLAN_FILE = "tool_plan.json"
EVIDENCE_BUNDLE_FILE = "evidence_bundle.json"
REASONING_TRACE_FILE = "reasoning_trace.json"
VERDICT_FILE = "verdict.json"
POR_BUNDLE_FILE = "por_bundle.json"
CHECKS_DIR = "checks"

FILE_MAP = {
    "prompt_spec": PROMPT_SPEC_FILE,
    "tool_plan": TOOL_PLAN_FILE,
    "evidence_bundle": EVIDENCE_BUNDLE_FILE,
    "reasoning_trace": REASONING_TRACE_FILE,
    "verdict": VERDICT_FILE,
    "por_bundle": POR_BUNDLE_FILE,
}


class PackIOError(Exception):
    """Error during pack IO operations."""
    pass


class PackHashMismatchError(PackIOError):
    """Hash mismatch detected during pack loading."""
    def __init__(self, file_key: str, expected: str, actual: str):
        self.file_key = file_key
        self.expected = expected
        self.actual = actual
        super().__init__(f"Hash mismatch for {file_key}: expected {expected}, got {actual}")


class PackMissingFileError(PackIOError):
    """Required file missing from pack."""
    def __init__(self, file_key: str):
        self.file_key = file_key
        super().__init__(f"Required file missing: {file_key}")


def dump_json(obj: Any) -> str:
    """Serialize object to canonical JSON string."""
    if hasattr(obj, "model_dump"):
        # Don't use exclude_none=True as some fields are required but allow None
        data = obj.model_dump(mode="json", by_alias=True)
        return dumps_canonical(data)
    elif hasattr(obj, "to_dict"):
        return dumps_canonical(obj.to_dict())
    elif isinstance(obj, dict):
        return dumps_canonical(obj)
    else:
        return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def compute_sha256(data: bytes) -> str:
    """Compute SHA-256 hex digest of data."""
    return hashlib.sha256(data).hexdigest()


def _write_json_file(path: Path, obj: Any) -> tuple[str, int]:
    """Write object as JSON file, return (sha256, size)."""
    content = dump_json(obj)
    data = content.encode("utf-8")
    path.write_bytes(data)
    return compute_sha256(data), len(data)


def _read_json_file(path: Path) -> tuple[Any, str]:
    """Read JSON file, return (parsed_data, sha256)."""
    data = path.read_bytes()
    sha256 = compute_sha256(data)
    parsed = json.loads(data.decode("utf-8"))
    return parsed, sha256


def save_pack_dir(
    package: PoRPackage,
    out_dir: str | Path,
    *,
    include_checks: dict[str, list[CheckResult]] | None = None,
) -> Path:
    """
    Save PoRPackage to a directory.
    
    Args:
        package: The package to save
        out_dir: Output directory path
        include_checks: Optional dict of check lists (pipeline, audit, judge, sentinel)
    
    Returns:
        Path to the created directory
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    manifest = PackManifest(
        market_id=package.bundle.market_id,
        por_root=package.bundle.por_root or "",
    )
    
    # Write required files
    artifacts = [
        ("prompt_spec", PROMPT_SPEC_FILE, package.prompt_spec),
        ("evidence_bundle", EVIDENCE_BUNDLE_FILE, package.evidence),
        ("reasoning_trace", REASONING_TRACE_FILE, package.trace),
        ("verdict", VERDICT_FILE, package.verdict),
        ("por_bundle", POR_BUNDLE_FILE, package.bundle),
    ]
    
    for key, filename, obj in artifacts:
        file_path = out_path / filename
        sha256, size = _write_json_file(file_path, obj)
        manifest.add_file(key, filename, sha256, size)
    
    # Write optional tool_plan
    if package.tool_plan is not None:
        file_path = out_path / TOOL_PLAN_FILE
        sha256, size = _write_json_file(file_path, package.tool_plan)
        manifest.add_file("tool_plan", TOOL_PLAN_FILE, sha256, size)
    
    # Write checks if provided
    if include_checks:
        checks_dir = out_path / CHECKS_DIR
        checks_dir.mkdir(exist_ok=True)
        for check_name, checks in include_checks.items():
            if checks:
                check_file = checks_dir / f"{check_name}_checks.json"
                check_data = [c.model_dump(mode="json") for c in checks]
                sha256, size = _write_json_file(check_file, check_data)
                manifest.add_file(
                    f"checks_{check_name}",
                    f"{CHECKS_DIR}/{check_name}_checks.json",
                    sha256, size
                )
    
    # Write manifest last
    manifest_path = out_path / MANIFEST_FILE
    manifest_content = dump_json(manifest.to_dict())
    manifest_path.write_text(manifest_content, encoding="utf-8")
    
    return out_path


def save_pack_zip(
    package: PoRPackage,
    out_zip: str | Path,
    *,
    include_checks: dict[str, list[CheckResult]] | None = None,
) -> Path:
    """
    Save PoRPackage to a zip file.
    
    Args:
        package: The package to save
        out_zip: Output zip file path
        include_checks: Optional dict of check lists
    
    Returns:
        Path to the created zip file
    """
    out_path = Path(out_zip)
    
    # Create temp directory, save as dir, then zip
    with tempfile.TemporaryDirectory() as tmpdir:
        pack_dir = Path(tmpdir) / "pack"
        save_pack_dir(package, pack_dir, include_checks=include_checks)
        
        # Create zip file
        with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(pack_dir):
                for file in files:
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(pack_dir)
                    zf.write(file_path, arcname)
    
    return out_path


def load_manifest(path: Path) -> PackManifest:
    """Load manifest from directory or zip."""
    if path.is_dir():
        manifest_path = path / MANIFEST_FILE
        if not manifest_path.exists():
            raise PackMissingFileError("manifest")
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    elif path.suffix == ".zip" or zipfile.is_zipfile(path):
        with zipfile.ZipFile(path, "r") as zf:
            try:
                data = json.loads(zf.read(MANIFEST_FILE).decode("utf-8"))
            except KeyError:
                raise PackMissingFileError("manifest")
    else:
        raise PackIOError(f"Unknown pack format: {path}")
    
    return PackManifest.from_dict(data)


def validate_pack_files(path: str | Path) -> VerificationResult:
    """
    Validate file hashes in a pack against manifest.
    
    Returns VerificationResult with check for each file.
    """
    path = Path(path)
    checks: list[CheckResult] = []
    
    try:
        manifest = load_manifest(path)
    except PackIOError as e:
        return VerificationResult(
            ok=False,
            checks=[CheckResult.failed("manifest_load", str(e))],
        )
    
    # Check format version
    if not is_compatible_format_version(manifest.format_version):
        checks.append(CheckResult.failed(
            "format_version",
            f"Incompatible format version: {manifest.format_version}",
        ))
    else:
        checks.append(CheckResult.passed("format_version", "Format version compatible"))
    
    # Check required files present
    has_required, missing = manifest.has_required_files()
    if not has_required:
        checks.append(CheckResult.failed(
            "required_files",
            f"Missing required files: {missing}",
        ))
    else:
        checks.append(CheckResult.passed("required_files", "All required files present"))
    
    # Verify each file hash
    is_zip = path.suffix == ".zip" or (path.is_file() and zipfile.is_zipfile(path))
    
    if is_zip:
        with zipfile.ZipFile(path, "r") as zf:
            for key, entry in manifest.files.items():
                try:
                    data = zf.read(entry.path)
                    actual_sha256 = compute_sha256(data)
                    if actual_sha256 == entry.sha256:
                        checks.append(CheckResult.passed(f"hash_{key}", f"{key} hash valid"))
                    else:
                        checks.append(CheckResult.failed(
                            f"hash_{key}",
                            f"{key} hash mismatch",
                            {"expected": entry.sha256, "actual": actual_sha256},
                        ))
                except KeyError:
                    checks.append(CheckResult.failed(f"hash_{key}", f"{key} file not found"))
    else:
        for key, entry in manifest.files.items():
            file_path = path / entry.path
            if not file_path.exists():
                checks.append(CheckResult.failed(f"hash_{key}", f"{key} file not found"))
                continue
            data = file_path.read_bytes()
            actual_sha256 = compute_sha256(data)
            if actual_sha256 == entry.sha256:
                checks.append(CheckResult.passed(f"hash_{key}", f"{key} hash valid"))
            else:
                checks.append(CheckResult.failed(
                    f"hash_{key}",
                    f"{key} hash mismatch",
                    {"expected": entry.sha256, "actual": actual_sha256},
                ))
    
    all_ok = all(c.ok for c in checks)
    return VerificationResult(ok=all_ok, checks=checks)


def _load_artifact_data(path: Path, manifest: PackManifest, verify_hashes: bool) -> dict[str, Any]:
    """Load all artifact data from pack."""
    is_zip = path.suffix == ".zip" or (path.is_file() and zipfile.is_zipfile(path))
    artifacts: dict[str, Any] = {}
    
    def read_file(entry: ManifestFileEntry, key: str) -> bytes:
        if is_zip:
            with zipfile.ZipFile(path, "r") as zf:
                try:
                    return zf.read(entry.path)
                except KeyError:
                    raise PackMissingFileError(key)
        else:
            file_path = path / entry.path
            if not file_path.exists():
                raise PackMissingFileError(key)
            return file_path.read_bytes()
    
    for key in ["prompt_spec", "evidence_bundle", "reasoning_trace", "verdict", "por_bundle", "tool_plan"]:
        entry = manifest.get_file(key)
        if entry is None:
            if key in REQUIRED_FILES:
                raise PackMissingFileError(key)
            continue
        
        data = read_file(entry, key)
        
        if verify_hashes:
            actual_sha256 = compute_sha256(data)
            if actual_sha256 != entry.sha256:
                raise PackHashMismatchError(key, entry.sha256, actual_sha256)
        
        artifacts[key] = json.loads(data.decode("utf-8"))
    
    return artifacts


def load_pack_dir(dir_path: str | Path, *, verify_hashes: bool = True) -> PoRPackage:
    """Load PoRPackage from a directory."""
    path = Path(dir_path)
    if not path.is_dir():
        raise PackIOError(f"Not a directory: {path}")
    return _load_pack_from_path(path, verify_hashes=verify_hashes)


def load_pack_zip(zip_path: str | Path, *, verify_hashes: bool = True) -> PoRPackage:
    """Load PoRPackage from a zip file."""
    path = Path(zip_path)
    if not zipfile.is_zipfile(path):
        raise PackIOError(f"Not a zip file: {path}")
    return _load_pack_from_path(path, verify_hashes=verify_hashes)


def load_pack(path: str | Path, *, verify_hashes: bool = True) -> PoRPackage:
    """
    Load PoRPackage from path (auto-detect directory or zip).
    
    Args:
        path: Path to directory or zip file
        verify_hashes: Whether to verify file hashes against manifest
    
    Returns:
        Loaded PoRPackage
    """
    path = Path(path)
    return _load_pack_from_path(path, verify_hashes=verify_hashes)


def _load_pack_from_path(path: Path, *, verify_hashes: bool = True) -> PoRPackage:
    """Internal: load pack from path."""
    manifest = load_manifest(path)
    
    if not is_compatible_format_version(manifest.format_version):
        raise PackIOError(f"Incompatible format version: {manifest.format_version}")
    
    artifacts = _load_artifact_data(path, manifest, verify_hashes)
    
    # Parse artifacts into models
    prompt_spec = PromptSpec.model_validate(artifacts["prompt_spec"])
    evidence = EvidenceBundle.model_validate(artifacts["evidence_bundle"])
    trace = ReasoningTrace.model_validate(artifacts["reasoning_trace"])
    verdict = DeterministicVerdict.model_validate(artifacts["verdict"])
    bundle = PoRBundle.model_validate(artifacts["por_bundle"])
    
    tool_plan = None
    if "tool_plan" in artifacts:
        tool_plan = ToolPlan.model_validate(artifacts["tool_plan"])
    
    return PoRPackage(
        bundle=bundle,
        prompt_spec=prompt_spec,
        tool_plan=tool_plan,
        evidence=evidence,
        trace=trace,
        verdict=verdict,
    )


def save_pack(
    package: PoRPackage,
    out_path: str | Path,
    *,
    include_checks: dict[str, list[CheckResult]] | None = None,
) -> Path:
    """
    Save PoRPackage to path (auto-detect format from extension).
    
    If path ends with .zip, saves as zip. Otherwise saves as directory.
    """
    out_path = Path(out_path)
    if out_path.suffix == ".zip":
        return save_pack_zip(package, out_path, include_checks=include_checks)
    else:
        return save_pack_dir(package, out_path, include_checks=include_checks)


__all__ = [
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
]
