"""
Module 06 - Contradiction Checks

Purpose: Detect conflicting claims across sources and flag them for
Judge/Invalid handling. Contradictions are identified when claims of
the same semantic key have incompatible values.

Owner: AI/Reasoning Engineer
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from core.schemas.verification import CheckResult

from .claim_extraction import Claim, ClaimSet


@dataclass
class Contradiction:
    """
    Represents a detected contradiction between claims.
    """

    contradiction_id: str
    kind: str  # "numeric_mismatch", "boolean_conflict", "value_divergence"
    claims: list[Claim]  # The conflicting claims
    semantic_key: tuple[str, str]  # (kind, path) that groups these claims
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def evidence_ids(self) -> list[str]:
        """Get all evidence IDs involved in this contradiction."""
        return list({c.evidence_id for c in self.claims})

    @property
    def claim_ids(self) -> list[str]:
        """Get all claim IDs involved in this contradiction."""
        return [c.claim_id for c in self.claims]


def _normalize_path(path: str) -> str:
    """
    Normalize a path for semantic grouping.

    Removes array indices and normalizes case.
    e.g., "data[0].price" -> "data.price"
    """
    import re
    # Remove array indices
    normalized = re.sub(r"\[\d+\]", "", path)
    # Normalize case
    return normalized.lower()


def _are_numeric_values_conflicting(
    values: list[tuple[float, str]],  # (value, evidence_id)
    tolerance: float = 0.01,
) -> bool:
    """
    Check if numeric values conflict beyond tolerance.

    Uses relative tolerance for larger values, absolute for smaller.
    """
    if len(values) < 2:
        return False

    nums = [v[0] for v in values]
    min_val = min(nums)
    max_val = max(nums)

    if max_val == 0 and min_val == 0:
        return False

    # Use relative tolerance for larger values
    if abs(max_val) > 1:
        relative_diff = (max_val - min_val) / abs(max_val)
        return relative_diff > tolerance
    else:
        # Absolute tolerance for small values
        return (max_val - min_val) > tolerance


def _are_boolean_values_conflicting(values: list[tuple[bool, str]]) -> bool:
    """Check if boolean values conflict (True vs False)."""
    if len(values) < 2:
        return False

    bools = {v[0] for v in values}
    return len(bools) > 1  # Has both True and False


def _generate_contradiction_id(semantic_key: tuple[str, str], claim_ids: list[str]) -> str:
    """Generate a deterministic contradiction ID."""
    from core.crypto.hashing import hash_canonical, to_hex

    content = {
        "semantic_key": list(semantic_key),
        "claim_ids": sorted(claim_ids),
    }
    hash_bytes = hash_canonical(content)
    return "ctr_" + to_hex(hash_bytes)[2:14]


class ContradictionChecker:
    """
    Detects contradictions across claims in a ClaimSet.

    Contradictions are identified by grouping claims by semantic key
    (kind, normalized_path) and checking for incompatible values.
    """

    def __init__(
        self,
        numeric_tolerance: float = 0.01,
        ignore_paths: Optional[set[str]] = None,
    ):
        """
        Initialize the contradiction checker.

        Args:
            numeric_tolerance: Relative tolerance for numeric comparisons
            ignore_paths: Set of paths to ignore in contradiction detection
        """
        self.numeric_tolerance = numeric_tolerance
        self.ignore_paths = ignore_paths or set()

    def _group_claims_by_semantic_key(
        self,
        claims: ClaimSet,
    ) -> dict[tuple[str, str], list[Claim]]:
        """
        Group claims by their semantic key (kind, normalized_path).
        """
        groups: dict[tuple[str, str], list[Claim]] = {}

        for claim in claims:
            # Skip ignored paths
            normalized = _normalize_path(claim.path)
            if normalized in self.ignore_paths:
                continue

            key = (claim.kind, normalized)
            if key not in groups:
                groups[key] = []
            groups[key].append(claim)

        return groups

    def _check_group_for_contradictions(
        self,
        semantic_key: tuple[str, str],
        claims: list[Claim],
    ) -> Optional[Contradiction]:
        """
        Check a group of claims for contradictions.

        Returns a Contradiction if found, None otherwise.
        """
        if len(claims) < 2:
            return None

        kind = semantic_key[0]

        if kind == "numeric":
            # Extract numeric values with evidence IDs
            values: list[tuple[float, str]] = []
            for claim in claims:
                try:
                    val = float(claim.value)
                    values.append((val, claim.evidence_id))
                except (ValueError, TypeError):
                    continue

            if _are_numeric_values_conflicting(values, self.numeric_tolerance):
                return Contradiction(
                    contradiction_id=_generate_contradiction_id(
                        semantic_key, [c.claim_id for c in claims]
                    ),
                    kind="numeric_mismatch",
                    claims=claims,
                    semantic_key=semantic_key,
                    details={
                        "values": [(c.value, c.evidence_id) for c in claims],
                        "tolerance": self.numeric_tolerance,
                    },
                )

        elif kind == "boolean":
            # Extract boolean values
            values_bool: list[tuple[bool, str]] = []
            for claim in claims:
                if isinstance(claim.value, bool):
                    values_bool.append((claim.value, claim.evidence_id))

            if _are_boolean_values_conflicting(values_bool):
                return Contradiction(
                    contradiction_id=_generate_contradiction_id(
                        semantic_key, [c.claim_id for c in claims]
                    ),
                    kind="boolean_conflict",
                    claims=claims,
                    semantic_key=semantic_key,
                    details={
                        "values": [(c.value, c.evidence_id) for c in claims],
                    },
                )

        elif kind in ("text_assertion", "categorical"):
            # Check for value divergence (different values from different sources)
            unique_values = set()
            for claim in claims:
                if claim.value is not None:
                    # Normalize string values for comparison
                    val = str(claim.value).strip().lower()
                    unique_values.add(val)

            # If we have multiple distinct values from different evidence sources
            if len(unique_values) > 1:
                evidence_ids = {c.evidence_id for c in claims}
                if len(evidence_ids) > 1:  # Divergence across sources
                    return Contradiction(
                        contradiction_id=_generate_contradiction_id(
                            semantic_key, [c.claim_id for c in claims]
                        ),
                        kind="value_divergence",
                        claims=claims,
                        semantic_key=semantic_key,
                        details={
                            "values": [(c.value, c.evidence_id) for c in claims],
                            "unique_values": list(unique_values),
                        },
                    )

        return None

    def check(self, claims: ClaimSet) -> list[CheckResult]:
        """
        Check a ClaimSet for contradictions.

        Args:
            claims: The set of claims to check

        Returns:
            List of CheckResult objects (one per detected contradiction)
        """
        results: list[CheckResult] = []

        # Group claims by semantic key
        groups = self._group_claims_by_semantic_key(claims)

        # Check each group for contradictions
        contradictions: list[Contradiction] = []
        for semantic_key, group_claims in groups.items():
            contradiction = self._check_group_for_contradictions(
                semantic_key, group_claims
            )
            if contradiction:
                contradictions.append(contradiction)

        # Convert contradictions to CheckResults
        for ctr in contradictions:
            severity = "warn"
            if ctr.kind in ("boolean_conflict", "numeric_mismatch"):
                severity = "error"

            results.append(
                CheckResult(
                    check_id=f"contradiction_{ctr.contradiction_id}",
                    ok=False,
                    severity=severity,
                    message=f"Contradiction detected: {ctr.kind} at path '{ctr.semantic_key[1]}'",
                    details={
                        "contradiction_id": ctr.contradiction_id,
                        "kind": ctr.kind,
                        "semantic_key": list(ctr.semantic_key),
                        "claim_ids": ctr.claim_ids,
                        "evidence_ids": ctr.evidence_ids,
                        "values": ctr.details.get("values", []),
                    },
                )
            )

        # If no contradictions found, add a passing check
        if not contradictions:
            results.append(
                CheckResult(
                    check_id="contradiction_check_passed",
                    ok=True,
                    severity="info",
                    message=f"No contradictions detected among {len(claims)} claims",
                    details={"claim_count": len(claims)},
                )
            )

        return results

    def get_contradictions(self, claims: ClaimSet) -> list[Contradiction]:
        """
        Get all contradictions found in the claim set.

        Returns the full Contradiction objects (not just CheckResults).
        """
        contradictions: list[Contradiction] = []

        groups = self._group_claims_by_semantic_key(claims)

        for semantic_key, group_claims in groups.items():
            contradiction = self._check_group_for_contradictions(
                semantic_key, group_claims
            )
            if contradiction:
                contradictions.append(contradiction)

        return contradictions


# Convenience function
def check_contradictions(claims: ClaimSet) -> list[CheckResult]:
    """
    Check a ClaimSet for contradictions using default configuration.

    This is the main entry point for contradiction checking.
    """
    checker = ContradictionChecker()
    return checker.check(claims)