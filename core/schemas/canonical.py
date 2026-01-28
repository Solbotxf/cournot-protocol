"""
Module 01 - Schemas & Canonicalization
File: canonical.py

Purpose: Deterministic serialization utilities for hashing, Merkle leaves,
and deterministic verification.

CRITICAL: All outputs from this module MUST be deterministic across runs.
"""

import json
import math
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel

from .errors import CanonicalizationException

# Canonical JSON separators - no whitespace
CANONICAL_JSON_SEPARATORS: tuple[str, str] = (",", ":")


def ensure_utc(dt: datetime) -> datetime:
    """
    Ensure a datetime is timezone-aware and in UTC.

    Args:
        dt: A datetime object (naive or aware).

    Returns:
        A timezone-aware datetime in UTC.

    Rules:
        - If naive (no tzinfo): treat as UTC
        - If aware: convert to UTC
    """
    if dt.tzinfo is None:
        # Naive datetime - assume UTC
        return dt.replace(tzinfo=timezone.utc)
    else:
        # Aware datetime - convert to UTC
        return dt.astimezone(timezone.utc)


def normalize_datetime(dt: datetime) -> datetime:
    """
    Normalize a datetime for canonical serialization.

    This ensures consistent handling regardless of input timezone.

    Args:
        dt: A datetime object.

    Returns:
        A normalized UTC datetime.
    """
    return ensure_utc(dt)


def format_datetime_canonical(dt: datetime) -> str:
    """
    Format a datetime as ISO-8601 with Z suffix for UTC.

    Args:
        dt: A datetime object.

    Returns:
        ISO-8601 formatted string with Z suffix (e.g., "2026-01-27T21:35:00Z").
    """
    utc_dt = ensure_utc(dt)
    # Format without timezone, then add Z
    # Use microseconds=0 format if no microseconds, otherwise include them
    if utc_dt.microsecond == 0:
        return utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        return utc_dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _validate_float(value: float, path: str = "") -> None:
    """
    Validate that a float is finite (not NaN or Infinity).

    Args:
        value: The float to validate.
        path: Path for error reporting.

    Raises:
        CanonicalizationException: If the float is NaN or Infinity.
    """
    if not math.isfinite(value):
        raise CanonicalizationException(
            message=f"Non-finite float value encountered: {value}",
            details={"path": path, "value": str(value)},
        )


def canonicalize_value(value: Any, path: str = "") -> Any:
    """
    Recursively canonicalize a value for deterministic JSON serialization.

    Args:
        value: Any Python value to canonicalize.
        path: Current path for error reporting.

    Returns:
        A JSON-serializable canonical representation.

    Raises:
        CanonicalizationException: If the value cannot be canonicalized
            (e.g., contains NaN/Infinity floats).
    """
    if value is None:
        return None

    if isinstance(value, bool):
        # Must check bool before int since bool is subclass of int
        return value

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        _validate_float(value, path)
        return value

    if isinstance(value, str):
        return value

    if isinstance(value, datetime):
        return format_datetime_canonical(value)

    if isinstance(value, Enum):
        # Represent enums as their string values
        return value.value

    if isinstance(value, BaseModel):
        # Use Pydantic's model_dump with appropriate settings
        dumped = value.model_dump(
            mode="json",
            by_alias=True,
            exclude_none=True,
        )
        # Recursively canonicalize the dumped dict
        return canonicalize_value(dumped, path)

    if isinstance(value, dict):
        # Recursively canonicalize dict values
        # Keys will be sorted during JSON serialization
        return {
            k: canonicalize_value(v, f"{path}.{k}" if path else k)
            for k, v in value.items()
            if v is not None  # Exclude None values
        }

    if isinstance(value, (list, tuple)):
        # Recursively canonicalize list/tuple items
        return [
            canonicalize_value(item, f"{path}[{i}]")
            for i, item in enumerate(value)
        ]

    if isinstance(value, bytes):
        # Convert bytes to hex string for JSON compatibility
        return value.hex()

    # For other types, attempt string conversion
    try:
        return str(value)
    except Exception as e:
        raise CanonicalizationException(
            message=f"Cannot canonicalize value of type {type(value).__name__}",
            details={"path": path, "type": type(value).__name__, "error": str(e)},
        ) from e


def to_canonical_json_dict(obj: Any) -> dict[str, Any]:
    """
    Convert an object to a canonical JSON-serializable dictionary.

    This is the primary entry point for converting Pydantic models
    and other objects to a deterministic dictionary representation.

    Args:
        obj: A Pydantic model, dict, or other serializable object.

    Returns:
        A dictionary suitable for deterministic JSON serialization.

    Raises:
        CanonicalizationException: If the object cannot be canonicalized.
    """
    canonicalized = canonicalize_value(obj)

    if not isinstance(canonicalized, dict):
        raise CanonicalizationException(
            message="to_canonical_json_dict expects an object that serializes to a dict",
            details={"type": type(obj).__name__, "result_type": type(canonicalized).__name__},
        )

    return canonicalized


def dumps_canonical(obj: Any) -> str:
    """
    Serialize an object to canonical JSON string.

    This produces a deterministic JSON string suitable for hashing
    and Merkle commitments.

    Args:
        obj: A Pydantic model, dict, or other serializable object.

    Returns:
        A canonical JSON string with:
            - Sorted keys
            - No extra whitespace
            - None fields excluded
            - Datetimes as ISO-8601 with Z suffix
            - Enums as string values
            - No NaN/Infinity floats

    Raises:
        CanonicalizationException: If serialization fails.

    Example:
        >>> from datetime import datetime
        >>> data = {"b": 2, "a": 1, "time": datetime(2026, 1, 27, 21, 35, 0)}
        >>> dumps_canonical(data)
        '{"a":1,"b":2,"time":"2026-01-27T21:35:00Z"}'
    """
    try:
        canonicalized = canonicalize_value(obj)
        return json.dumps(
            canonicalized,
            sort_keys=True,
            separators=CANONICAL_JSON_SEPARATORS,
            ensure_ascii=False,
        )
    except CanonicalizationException:
        raise
    except Exception as e:
        raise CanonicalizationException(
            message=f"Failed to serialize to canonical JSON: {e}",
            details={"type": type(obj).__name__, "error": str(e)},
        ) from e


def loads_canonical(json_str: str) -> Any:
    """
    Parse a canonical JSON string.

    Note: This does NOT restore datetime objects - they remain as strings.
    Use this primarily for verification and comparison.

    Args:
        json_str: A JSON string.

    Returns:
        The parsed JSON data structure.
    """
    return json.loads(json_str)


def canonical_equals(obj1: Any, obj2: Any) -> bool:
    """
    Check if two objects are canonically equal.

    This compares the canonical JSON representations of two objects.

    Args:
        obj1: First object.
        obj2: Second object.

    Returns:
        True if the canonical representations are identical.
    """
    try:
        return dumps_canonical(obj1) == dumps_canonical(obj2)
    except CanonicalizationException:
        return False