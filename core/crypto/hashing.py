"""
Module 02 - Hashing Utilities
Basic hashing and canonical hashing utilities for Merkle commitments.

Owner: Protocol/Crypto Engineer
Module ID: M02

This module provides:
- SHA-256 hashing for raw bytes
- Canonical hashing for objects (via dumps_canonical)
- Hex encoding/decoding with 0x prefix

Security/Determinism Notes:
- Always hash raw bytes exactly as specified
- No auto-stripping of whitespace beyond canonical JSON
- All operations are deterministic
"""
from __future__ import annotations

import hashlib
from typing import Any

from core.schemas.canonical import dumps_canonical


def sha256(data: bytes) -> bytes:
    """
    Compute SHA-256 hash of raw bytes.
    
    Args:
        data: Raw bytes to hash
        
    Returns:
        32-byte SHA-256 digest
        
    Example:
        >>> sha256(b"hello").hex()
        '2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824'
    """
    return hashlib.sha256(data).digest()


def hash_bytes(data: bytes) -> bytes:
    """
    Alias for sha256() - compute SHA-256 hash of raw bytes.
    
    Args:
        data: Raw bytes to hash
        
    Returns:
        32-byte SHA-256 digest
    """
    return sha256(data)


def hash_canonical(obj: Any) -> bytes:
    """
    Hash an object using canonical JSON serialization.
    
    This is the standard way to compute a "leaf" hash for Merkle trees.
    The object is first serialized to canonical JSON (deterministic),
    then the UTF-8 encoded bytes are hashed with SHA-256.
    
    Rule: leaf = sha256(dumps_canonical(obj).encode("utf-8"))
    
    Args:
        obj: Any object that can be canonically serialized
             (Pydantic model, dict, list, primitives)
        
    Returns:
        32-byte SHA-256 digest of the canonical JSON
        
    Raises:
        ValueError: If object cannot be canonically serialized
        
    Example:
        >>> hash_canonical({"b": 2, "a": 1}).hex()  # Keys sorted
        'a1b2c3...'  # deterministic regardless of dict insertion order
    """
    canonical_json = dumps_canonical(obj)
    return sha256(canonical_json.encode("utf-8"))


def to_hex(data: bytes) -> str:
    """
    Convert bytes to hexadecimal string with 0x prefix.
    
    Args:
        data: Raw bytes
        
    Returns:
        Hex string with 0x prefix (e.g., "0x1234abcd")
        
    Example:
        >>> to_hex(bytes.fromhex("deadbeef"))
        '0xdeadbeef'
    """
    return "0x" + data.hex()


def from_hex(hex_string: str) -> bytes:
    """
    Convert hexadecimal string (with 0x prefix) to bytes.
    
    Args:
        hex_string: Hex string with 0x prefix
        
    Returns:
        Decoded bytes
        
    Raises:
        ValueError: If string doesn't start with 0x, has odd length,
                   or contains invalid hex characters
                   
    Example:
        >>> from_hex("0xdeadbeef").hex()
        'deadbeef'
    """
    # Validate 0x prefix
    if not hex_string.startswith("0x"):
        raise ValueError(
            f"Hex string must start with '0x' prefix, got: {hex_string[:10]}..."
        )
    
    # Remove prefix
    hex_content = hex_string[2:]
    
    # Validate even length
    if len(hex_content) % 2 != 0:
        raise ValueError(
            f"Hex string must have even length after 0x prefix, "
            f"got length {len(hex_content)}"
        )
    
    # Decode (will raise ValueError for invalid hex chars)
    try:
        return bytes.fromhex(hex_content)
    except ValueError as e:
        raise ValueError(f"Invalid hex characters in string: {e}") from e


def hash_concat(left: bytes, right: bytes) -> bytes:
    """
    Hash the concatenation of two byte sequences.
    
    This is used for computing Merkle parent hashes:
    parent = sha256(left + right)
    
    Args:
        left: Left child hash (typically 32 bytes)
        right: Right child hash (typically 32 bytes)
        
    Returns:
        32-byte SHA-256 digest of concatenation
    """
    return sha256(left + right)


__all__ = [
    "sha256",
    "hash_bytes",
    "hash_canonical",
    "to_hex",
    "from_hex",
    "hash_concat",
]