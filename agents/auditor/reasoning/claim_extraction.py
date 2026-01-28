"""
Module 06 - Claim Extraction

Purpose: Extract structured claims from evidence items that can be reasoned over
deterministically. Claims are the atomic units of information extracted from evidence.

Owner: AI/Reasoning Engineer
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from core.crypto.hashing import hash_canonical, to_hex
from core.schemas.canonical import dumps_canonical
from core.schemas.evidence import EvidenceBundle, EvidenceItem
from core.schemas.prompts import DataRequirement, PromptSpec


@dataclass(frozen=True)
class Claim:
    """
    A single extracted claim from evidence.

    Claims are atomic statements that can be reasoned over and verified.
    Each claim is tied to a specific piece of evidence.
    """

    claim_id: str
    kind: str  # "numeric", "boolean", "text_assertion", "timestamp", "categorical"
    path: str  # JSON path or extraction rule used
    value: Any  # The extracted value
    evidence_id: str  # Reference to the source evidence
    confidence: float = 1.0  # Confidence in extraction (0.0 to 1.0)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate claim structure."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")


@dataclass
class ClaimSet:
    """
    Collection of claims extracted from evidence.

    Provides helpers for filtering and grouping claims.
    """

    claims: list[Claim] = field(default_factory=list)

    def add(self, claim: Claim) -> None:
        """Add a claim to the set."""
        self.claims.append(claim)

    def by_evidence_id(self, evidence_id: str) -> list[Claim]:
        """Get all claims from a specific evidence item."""
        return [c for c in self.claims if c.evidence_id == evidence_id]

    def by_kind(self, kind: str) -> list[Claim]:
        """Get all claims of a specific kind."""
        return [c for c in self.claims if c.kind == kind]

    def by_path(self, path: str) -> list[Claim]:
        """Get all claims extracted from a specific path."""
        return [c for c in self.claims if c.path == path]

    def get_unique_paths(self) -> set[str]:
        """Get all unique extraction paths."""
        return {c.path for c in self.claims}

    def get_unique_evidence_ids(self) -> set[str]:
        """Get all unique evidence IDs referenced."""
        return {c.evidence_id for c in self.claims}

    def __len__(self) -> int:
        return len(self.claims)

    def __iter__(self):
        return iter(self.claims)


def _generate_claim_id(evidence_id: str, path: str, value: Any) -> str:
    """
    Generate a deterministic claim ID.

    claim_id = "cl_" + sha256(evidence_id + "|" + path + "|" + canonical(value))[:12]
    """
    canonical_value = dumps_canonical(value) if value is not None else "null"
    content = f"{evidence_id}|{path}|{canonical_value}"
    hash_bytes = hash_canonical(content)
    return "cl_" + to_hex(hash_bytes)[2:14]  # Take 12 hex chars after 0x


def _infer_claim_kind(value: Any) -> str:
    """Infer the kind of claim from the value type."""
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "numeric"
    if isinstance(value, str):
        # Check if it looks like a timestamp
        if re.match(r"^\d{4}-\d{2}-\d{2}", value):
            return "timestamp"
        return "text_assertion"
    return "categorical"


def _extract_from_json(
    content: dict[str, Any],
    evidence_id: str,
    expected_fields: Optional[list[str]] = None,
) -> list[Claim]:
    """
    Extract claims from JSON content.

    Strategy:
    1. If expected_fields provided, extract only those
    2. Otherwise, extract common fields: result, value, price, timestamp, status
    """
    claims: list[Claim] = []

    # Default common fields to extract
    common_fields = ["result", "value", "price", "close", "timestamp", "status",
                     "outcome", "resolved", "data", "answer", "amount"]

    fields_to_extract = expected_fields if expected_fields else common_fields

    def extract_recursive(obj: Any, current_path: str = "") -> None:
        """Recursively extract values from nested structures."""
        if isinstance(obj, dict):
            for key, val in obj.items():
                new_path = f"{current_path}.{key}" if current_path else key

                # Check if this key should be extracted
                key_lower = key.lower()
                should_extract = (
                    expected_fields is not None and key in expected_fields
                ) or (
                    expected_fields is None and key_lower in [f.lower() for f in common_fields]
                )

                if should_extract and val is not None:
                    # Don't extract nested dicts/lists directly - just primitives
                    if not isinstance(val, (dict, list)):
                        claim = Claim(
                            claim_id=_generate_claim_id(evidence_id, new_path, val),
                            kind=_infer_claim_kind(val),
                            path=new_path,
                            value=val,
                            evidence_id=evidence_id,
                            confidence=1.0,
                        )
                        claims.append(claim)

                # Continue recursing for nested structures
                if isinstance(val, (dict, list)):
                    extract_recursive(val, new_path)

        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                new_path = f"{current_path}[{i}]"
                extract_recursive(item, new_path)

    extract_recursive(content)
    return claims


def _extract_from_text(
    content: str,
    evidence_id: str,
    patterns: Optional[list[str]] = None,
) -> list[Claim]:
    """
    Extract claims from text content using regex patterns.

    If no patterns provided, uses default numeric and boolean extraction.
    """
    claims: list[Claim] = []

    # Default patterns for common extractions
    default_patterns = [
        # Numbers with optional decimal
        (r"(\d+(?:\.\d+)?)", "numeric"),
        # Boolean-like words
        (r"\b(true|false|yes|no)\b", "boolean"),
        # Dates (YYYY-MM-DD)
        (r"(\d{4}-\d{2}-\d{2})", "timestamp"),
    ]

    patterns_to_use = patterns if patterns else [p[0] for p in default_patterns]

    for i, pattern in enumerate(patterns_to_use):
        try:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for j, match in enumerate(matches[:5]):  # Limit to first 5 matches
                # Determine kind
                if isinstance(match, tuple):
                    match = match[0]

                kind = "text_assertion"
                value: Any = match

                # Try to infer type
                if match.lower() in ("true", "yes"):
                    kind = "boolean"
                    value = True
                elif match.lower() in ("false", "no"):
                    kind = "boolean"
                    value = False
                elif re.match(r"^\d+(?:\.\d+)?$", match):
                    kind = "numeric"
                    value = float(match) if "." in match else int(match)
                elif re.match(r"^\d{4}-\d{2}-\d{2}", match):
                    kind = "timestamp"

                path = f"text_match[{i}][{j}]"
                claim = Claim(
                    claim_id=_generate_claim_id(evidence_id, path, value),
                    kind=kind,
                    path=path,
                    value=value,
                    evidence_id=evidence_id,
                    confidence=0.8,  # Lower confidence for text extraction
                )
                claims.append(claim)
        except re.error:
            # Invalid regex pattern - skip
            continue

    return claims


class ClaimExtractor:
    """
    Extracts structured claims from evidence items.

    Supports JSON and text content types. Extraction is deterministic
    given the same evidence and configuration.
    """

    def __init__(
        self,
        expected_fields: Optional[dict[str, list[str]]] = None,
        text_patterns: Optional[dict[str, list[str]]] = None,
    ):
        """
        Initialize the claim extractor.

        Args:
            expected_fields: Map of requirement_id -> list of expected JSON fields
            text_patterns: Map of requirement_id -> list of regex patterns for text
        """
        self.expected_fields = expected_fields or {}
        self.text_patterns = text_patterns or {}

    def extract_from_item(
        self,
        item: EvidenceItem,
        requirement: Optional[DataRequirement] = None,
    ) -> list[Claim]:
        """
        Extract claims from a single evidence item.

        Args:
            item: The evidence item to extract from
            requirement: Optional requirement for context (expected_fields)

        Returns:
            List of extracted claims
        """
        # Determine expected fields
        expected = None
        if requirement and requirement.expected_fields:
            expected = requirement.expected_fields
        elif item.requirement_id and item.requirement_id in self.expected_fields:
            expected = self.expected_fields[item.requirement_id]

        # Extract based on content type
        if item.content_type == "json" and isinstance(item.content, dict):
            return _extract_from_json(item.content, item.evidence_id, expected)
        elif item.content_type in ("text", "html"):
            content = str(item.content)
            patterns = None
            if item.requirement_id and item.requirement_id in self.text_patterns:
                patterns = self.text_patterns[item.requirement_id]
            return _extract_from_text(content, item.evidence_id, patterns)
        else:
            # For bytes or unknown types, try to extract as text
            try:
                content = str(item.content)
                return _extract_from_text(content, item.evidence_id)
            except Exception:
                return []

    def extract(
        self,
        prompt_spec: PromptSpec,
        evidence: EvidenceBundle,
    ) -> ClaimSet:
        """
        Extract all claims from an evidence bundle.

        Claims are extracted in evidence item order (deterministic).
        Within each item, claims are ordered by their extraction path.

        Args:
            prompt_spec: The prompt specification (for expected_fields hints)
            evidence: The evidence bundle to extract from

        Returns:
            ClaimSet containing all extracted claims
        """
        claim_set = ClaimSet()

        # Build requirement lookup
        req_lookup: dict[str, DataRequirement] = {}
        for req in prompt_spec.data_requirements:
            req_lookup[req.requirement_id] = req

        # Process items in order (preserving determinism)
        for item in evidence.items:
            requirement = None
            if item.requirement_id:
                requirement = req_lookup.get(item.requirement_id)

            item_claims = self.extract_from_item(item, requirement)

            # Sort claims by path for deterministic ordering within item
            item_claims.sort(key=lambda c: c.path)

            for claim in item_claims:
                claim_set.add(claim)

        return claim_set


# Convenience function for simple extraction
def extract_claims(
    prompt_spec: PromptSpec,
    evidence: EvidenceBundle,
) -> ClaimSet:
    """
    Extract claims from evidence using default configuration.

    This is the main entry point for claim extraction.
    """
    extractor = ClaimExtractor()
    return extractor.extract(prompt_spec, evidence)