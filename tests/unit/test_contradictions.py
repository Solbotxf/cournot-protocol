"""
Module 06 - Tests for Contradiction Checks

Tests:
- Numeric conflict detection
- Boolean conflict detection
- Value divergence detection
- No false positives when claims agree
"""

import pytest

from agents.auditor.reasoning.claim_extraction import Claim, ClaimSet
from agents.auditor.reasoning.contradiction_checks import (
    Contradiction,
    ContradictionChecker,
    check_contradictions,
    _are_numeric_values_conflicting,
    _are_boolean_values_conflicting,
    _normalize_path,
)


# =============================================================================
# Test Path Normalization
# =============================================================================


class TestPathNormalization:
    """Tests for path normalization."""

    def test_normalize_removes_array_indices(self):
        """Test that array indices are removed."""
        assert _normalize_path("data[0].price") == "data.price"
        assert _normalize_path("items[123].value") == "items.value"
        assert _normalize_path("[0]") == ""

    def test_normalize_case(self):
        """Test that paths are lowercased."""
        assert _normalize_path("Price") == "price"
        assert _normalize_path("Data.PRICE") == "data.price"

    def test_normalize_simple_path(self):
        """Test that simple paths are preserved."""
        assert _normalize_path("price") == "price"
        assert _normalize_path("data.value") == "data.value"


# =============================================================================
# Test Numeric Conflict Detection
# =============================================================================


class TestNumericConflicts:
    """Tests for numeric conflict detection."""

    def test_no_conflict_same_values(self):
        """Test that identical values don't conflict."""
        values = [(100.0, "ev_001"), (100.0, "ev_002")]
        assert not _are_numeric_values_conflicting(values)

    def test_no_conflict_within_tolerance(self):
        """Test that values within tolerance don't conflict."""
        values = [(100.0, "ev_001"), (100.5, "ev_002")]
        assert not _are_numeric_values_conflicting(values, tolerance=0.01)

    def test_conflict_beyond_tolerance(self):
        """Test that values beyond tolerance conflict."""
        values = [(100.0, "ev_001"), (110.0, "ev_002")]
        assert _are_numeric_values_conflicting(values, tolerance=0.01)

    def test_single_value_no_conflict(self):
        """Test that single value doesn't conflict."""
        values = [(100.0, "ev_001")]
        assert not _are_numeric_values_conflicting(values)

    def test_empty_values_no_conflict(self):
        """Test that empty values don't conflict."""
        values = []
        assert not _are_numeric_values_conflicting(values)

    def test_zero_values_handled(self):
        """Test that zero values are handled correctly."""
        values = [(0.0, "ev_001"), (0.0, "ev_002")]
        assert not _are_numeric_values_conflicting(values)


# =============================================================================
# Test Boolean Conflict Detection
# =============================================================================


class TestBooleanConflicts:
    """Tests for boolean conflict detection."""

    def test_no_conflict_all_true(self):
        """Test that all True doesn't conflict."""
        values = [(True, "ev_001"), (True, "ev_002")]
        assert not _are_boolean_values_conflicting(values)

    def test_no_conflict_all_false(self):
        """Test that all False doesn't conflict."""
        values = [(False, "ev_001"), (False, "ev_002")]
        assert not _are_boolean_values_conflicting(values)

    def test_conflict_true_and_false(self):
        """Test that True and False conflict."""
        values = [(True, "ev_001"), (False, "ev_002")]
        assert _are_boolean_values_conflicting(values)

    def test_single_value_no_conflict(self):
        """Test that single value doesn't conflict."""
        values = [(True, "ev_001")]
        assert not _are_boolean_values_conflicting(values)


# =============================================================================
# Test ContradictionChecker
# =============================================================================


class TestContradictionChecker:
    """Tests for the ContradictionChecker class."""

    def test_detect_numeric_mismatch(self):
        """Test detection of numeric mismatch."""
        claim_set = ClaimSet()
        claim_set.add(Claim("cl_1", "numeric", "price", 100.0, "ev_001", 1.0))
        claim_set.add(Claim("cl_2", "numeric", "price", 150.0, "ev_002", 1.0))

        checker = ContradictionChecker()
        results = checker.check(claim_set)

        # Should detect contradiction
        failed_checks = [r for r in results if not r.ok]
        assert len(failed_checks) >= 1

        # Check that it's a numeric mismatch
        assert any("numeric_mismatch" in r.details.get("kind", "") for r in failed_checks)

    def test_detect_boolean_conflict(self):
        """Test detection of boolean conflict."""
        claim_set = ClaimSet()
        claim_set.add(Claim("cl_1", "boolean", "resolved", True, "ev_001", 1.0))
        claim_set.add(Claim("cl_2", "boolean", "resolved", False, "ev_002", 1.0))

        checker = ContradictionChecker()
        results = checker.check(claim_set)

        # Should detect contradiction
        failed_checks = [r for r in results if not r.ok]
        assert len(failed_checks) >= 1

        # Check that it's a boolean conflict
        assert any("boolean_conflict" in r.details.get("kind", "") for r in failed_checks)

    def test_no_contradiction_when_agreeing(self):
        """Test no contradiction when claims agree."""
        claim_set = ClaimSet()
        claim_set.add(Claim("cl_1", "numeric", "price", 100.0, "ev_001", 1.0))
        claim_set.add(Claim("cl_2", "numeric", "price", 100.5, "ev_002", 1.0))

        checker = ContradictionChecker(numeric_tolerance=0.01)
        results = checker.check(claim_set)

        # All checks should pass
        assert all(r.ok for r in results)

    def test_different_paths_no_conflict(self):
        """Test that different paths don't conflict."""
        claim_set = ClaimSet()
        claim_set.add(Claim("cl_1", "numeric", "price", 100.0, "ev_001", 1.0))
        claim_set.add(Claim("cl_2", "numeric", "volume", 150.0, "ev_002", 1.0))

        checker = ContradictionChecker()
        results = checker.check(claim_set)

        # Should not detect contradiction (different paths)
        failed_checks = [r for r in results if not r.ok]
        # The only failures should not be for numeric_mismatch between these
        for check in failed_checks:
            if check.details.get("kind") == "numeric_mismatch":
                # If there's a numeric mismatch, it shouldn't be between price and volume
                semantic_key = check.details.get("semantic_key", [])
                assert semantic_key[1] != "price" or semantic_key[1] != "volume"

    def test_ignore_paths(self):
        """Test that ignored paths are not checked."""
        claim_set = ClaimSet()
        claim_set.add(Claim("cl_1", "numeric", "timestamp", 1000, "ev_001", 1.0))
        claim_set.add(Claim("cl_2", "numeric", "timestamp", 2000, "ev_002", 1.0))

        checker = ContradictionChecker(ignore_paths={"timestamp"})
        results = checker.check(claim_set)

        # Should not detect contradiction (path ignored)
        failed_checks = [r for r in results if not r.ok]
        for check in failed_checks:
            # Make sure timestamp isn't flagged
            semantic_key = check.details.get("semantic_key", [])
            assert "timestamp" not in str(semantic_key)

    def test_get_contradictions_method(self):
        """Test the get_contradictions method returns Contradiction objects."""
        claim_set = ClaimSet()
        claim_set.add(Claim("cl_1", "boolean", "status", True, "ev_001", 1.0))
        claim_set.add(Claim("cl_2", "boolean", "status", False, "ev_002", 1.0))

        checker = ContradictionChecker()
        contradictions = checker.get_contradictions(claim_set)

        assert len(contradictions) >= 1
        assert isinstance(contradictions[0], Contradiction)
        assert contradictions[0].kind == "boolean_conflict"
        assert len(contradictions[0].evidence_ids) == 2


# =============================================================================
# Test Convenience Function
# =============================================================================


class TestCheckContradictions:
    """Tests for the check_contradictions convenience function."""

    def test_check_contradictions_function(self):
        """Test the convenience function."""
        claim_set = ClaimSet()
        claim_set.add(Claim("cl_1", "numeric", "price", 100.0, "ev_001", 1.0))
        claim_set.add(Claim("cl_2", "numeric", "price", 200.0, "ev_002", 1.0))

        results = check_contradictions(claim_set)

        assert len(results) > 0
        # Should have at least one failed check
        assert any(not r.ok for r in results)

    def test_empty_claim_set(self):
        """Test with empty claim set."""
        claim_set = ClaimSet()
        results = check_contradictions(claim_set)

        # Should return passing check
        assert len(results) > 0
        assert all(r.ok for r in results)


# =============================================================================
# Test Value Divergence
# =============================================================================


class TestValueDivergence:
    """Tests for text/categorical value divergence detection."""

    def test_detect_text_divergence(self):
        """Test detection of divergent text values."""
        claim_set = ClaimSet()
        claim_set.add(Claim("cl_1", "text_assertion", "status", "completed", "ev_001", 1.0))
        claim_set.add(Claim("cl_2", "text_assertion", "status", "pending", "ev_002", 1.0))

        checker = ContradictionChecker()
        contradictions = checker.get_contradictions(claim_set)

        # Should detect divergence
        assert len(contradictions) >= 1
        assert any(c.kind == "value_divergence" for c in contradictions)

    def test_no_divergence_same_text(self):
        """Test no divergence when text values agree."""
        claim_set = ClaimSet()
        claim_set.add(Claim("cl_1", "text_assertion", "status", "completed", "ev_001", 1.0))
        claim_set.add(Claim("cl_2", "text_assertion", "status", "COMPLETED", "ev_002", 1.0))

        checker = ContradictionChecker()
        contradictions = checker.get_contradictions(claim_set)

        # Should not detect divergence (same value, different case)
        divergence_contradictions = [c for c in contradictions if c.kind == "value_divergence"]
        assert len(divergence_contradictions) == 0

    def test_no_divergence_same_source(self):
        """Test no divergence when values are from same source."""
        claim_set = ClaimSet()
        claim_set.add(Claim("cl_1", "text_assertion", "field_a", "value_x", "ev_001", 1.0))
        claim_set.add(Claim("cl_2", "text_assertion", "field_a", "value_y", "ev_001", 1.0))

        checker = ContradictionChecker()
        contradictions = checker.get_contradictions(claim_set)

        # Should not detect divergence (same source, might be different contexts)
        divergence_contradictions = [c for c in contradictions if c.kind == "value_divergence"]
        # This is from same evidence source, so no cross-source divergence
        assert len(divergence_contradictions) == 0


# =============================================================================
# Test CheckResult Details
# =============================================================================


class TestCheckResultDetails:
    """Tests for CheckResult details in contradiction checks."""

    def test_check_result_contains_evidence_ids(self):
        """Test that CheckResult details contain evidence IDs."""
        claim_set = ClaimSet()
        claim_set.add(Claim("cl_1", "numeric", "price", 100.0, "ev_001", 1.0))
        claim_set.add(Claim("cl_2", "numeric", "price", 200.0, "ev_002", 1.0))

        results = check_contradictions(claim_set)
        failed = [r for r in results if not r.ok]

        assert len(failed) > 0
        for check in failed:
            assert "evidence_ids" in check.details
            assert "ev_001" in check.details["evidence_ids"]
            assert "ev_002" in check.details["evidence_ids"]

    def test_check_result_contains_claim_ids(self):
        """Test that CheckResult details contain claim IDs."""
        claim_set = ClaimSet()
        claim_set.add(Claim("cl_1", "boolean", "flag", True, "ev_001", 1.0))
        claim_set.add(Claim("cl_2", "boolean", "flag", False, "ev_002", 1.0))

        results = check_contradictions(claim_set)
        failed = [r for r in results if not r.ok]

        assert len(failed) > 0
        for check in failed:
            assert "claim_ids" in check.details
            assert "cl_1" in check.details["claim_ids"]
            assert "cl_2" in check.details["claim_ids"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])