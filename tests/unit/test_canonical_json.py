"""
Module 01 - Schemas & Canonicalization
File: tests/unit/test_canonical_json.py

Purpose: Unit tests for canonical JSON serialization.
These tests ensure deterministic serialization across runs.

Updated to include tests for:
- SourceTarget and SelectionPolicy schemas
- source_targets order preservation
- PromptSpec.created_at optional behavior for deterministic hashing
"""

import json
import math
from datetime import datetime, timezone, timedelta
from enum import Enum

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from core.schemas import (
    CanonicalizationException,
    DataRequirement,
    DeterministicVerdict,
    EvidenceBundle,
    EvidenceItem,
    MarketSpec,
    PredictionSemantics,
    PromptSpec,
    Provenance,
    ResolutionRule,
    ResolutionRules,
    ResolutionWindow,
    SelectionPolicy,
    SourcePolicy,
    SourceTarget,
    DisputePolicy,
    dumps_canonical,
    ensure_utc,
    normalize_datetime,
    to_canonical_json_dict,
    format_datetime_canonical,
    canonicalize_value,
    loads_canonical,
    canonical_equals,
)


# =============================================================================
# Test Fixtures
# =============================================================================


class SampleEnum(str, Enum):
    """Sample enum for testing."""
    OPTION_A = "option_a"
    OPTION_B = "option_b"


class SampleModel(BaseModel):
    """Sample Pydantic model for testing."""
    model_config = ConfigDict(extra="forbid")

    name: str
    value: int
    optional_field: str | None = None
    nested: dict | None = None


@pytest.fixture
def sample_datetime_naive() -> datetime:
    """A naive datetime (no timezone info)."""
    return datetime(2026, 1, 27, 21, 35, 0)


@pytest.fixture
def sample_datetime_utc() -> datetime:
    """A UTC-aware datetime."""
    return datetime(2026, 1, 27, 21, 35, 0, tzinfo=timezone.utc)


@pytest.fixture
def sample_datetime_offset() -> datetime:
    """A datetime with non-UTC offset."""
    tz = timezone(timedelta(hours=-5))  # EST
    return datetime(2026, 1, 27, 16, 35, 0, tzinfo=tz)


@pytest.fixture
def sample_source_target() -> SourceTarget:
    """A sample SourceTarget for testing."""
    return SourceTarget(
        source_id="http",
        uri="https://api.example.com/data?param=value",
        method="GET",
        expected_content_type="json",
        headers={"Authorization": "Bearer token"},
    )


@pytest.fixture
def sample_selection_policy() -> SelectionPolicy:
    """A sample SelectionPolicy for testing."""
    return SelectionPolicy(
        strategy="fallback_chain",
        min_sources=1,
        max_sources=3,
        quorum=1,
        tie_breaker="highest_provenance",
    )


@pytest.fixture
def sample_data_requirement(sample_source_target, sample_selection_policy) -> DataRequirement:
    """A sample DataRequirement with source_targets."""
    return DataRequirement(
        requirement_id="req-001",
        description="Fetch price data from exchange",
        source_targets=[sample_source_target],
        selection_policy=sample_selection_policy,
    )


@pytest.fixture
def sample_market_spec() -> MarketSpec:
    """A complete MarketSpec for testing."""
    return MarketSpec(
        market_id="test-market-001",
        question="Will event X happen by date Y?",
        event_definition="Event X is defined as: specific observable occurrence.",
        resolution_deadline=datetime(2026, 3, 31, 23, 59, 59, tzinfo=timezone.utc),
        resolution_window=ResolutionWindow(
            start=datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            end=datetime(2026, 3, 31, 23, 59, 59, tzinfo=timezone.utc),
        ),
        resolution_rules=ResolutionRules(
            rules=[
                ResolutionRule(
                    rule_id="rule-yes",
                    description="Resolve YES if confirmed by primary source",
                    priority=1,
                ),
                ResolutionRule(
                    rule_id="rule-no",
                    description="Resolve NO if deadline passes without confirmation",
                    priority=0,
                ),
            ]
        ),
        allowed_sources=[
            SourcePolicy(
                source_id="primary-api",
                kind="api",
                allow=True,
                min_provenance_tier=0,
            ),
        ],
        dispute_policy=DisputePolicy(
            dispute_window_seconds=86400,
            allow_challenges=True,
        ),
    )


@pytest.fixture
def sample_evidence_item() -> EvidenceItem:
    """A sample EvidenceItem for testing."""
    return EvidenceItem(
        evidence_id="evidence-001",
        requirement_id="req_001",
        provenance=Provenance(
            source_id="test-source",
            source_uri="https://api.example.com/data",
            tier=1,
        ),
        content_type="application/json",
        parsed_value={"key": "value"},
    )


# =============================================================================
# Test A: Deterministic Ordering
# =============================================================================


class TestDeterministicOrdering:
    """Tests for deterministic JSON key ordering."""

    def test_dict_keys_sorted(self):
        """Keys should be sorted alphabetically in output."""
        data = {"zebra": 1, "apple": 2, "mango": 3}
        result = dumps_canonical(data)
        expected = '{"apple":2,"mango":3,"zebra":1}'
        assert result == expected

    def test_nested_dict_keys_sorted(self):
        """Nested dict keys should also be sorted."""
        data = {"outer": {"z": 1, "a": 2}, "inner": {"y": 3, "b": 4}}
        result = dumps_canonical(data)
        parsed = json.loads(result)
        assert list(parsed.keys()) == ["inner", "outer"]
        assert list(parsed["inner"].keys()) == ["b", "y"]
        assert list(parsed["outer"].keys()) == ["a", "z"]

    def test_model_fields_sorted(self):
        """Pydantic model fields should be sorted."""
        model = SampleModel(name="test", value=42)
        result = dumps_canonical(model)
        parsed = json.loads(result)
        assert list(parsed.keys()) == ["name", "value"]

    def test_repeated_serialization_identical(self):
        """Multiple serializations should produce identical output."""
        data = {"b": 2, "a": 1, "c": {"y": 1, "x": 2}}
        results = [dumps_canonical(data) for _ in range(10)]
        assert len(set(results)) == 1, "All serializations should be identical"

    def test_random_insertion_order_deterministic(self):
        """Dict with random insertion order should serialize identically."""
        import random

        keys = list("abcdefghij")
        for _ in range(10):
            random.shuffle(keys)
            data = {k: i for i, k in enumerate(keys)}
            result = dumps_canonical(data)
            parsed = json.loads(result)
            assert list(parsed.keys()) == sorted(keys)


# =============================================================================
# Test B: Datetime Normalization
# =============================================================================


class TestDatetimeNormalization:
    """Tests for datetime handling and UTC normalization."""

    def test_ensure_utc_naive(self, sample_datetime_naive):
        """Naive datetime should be treated as UTC."""
        result = ensure_utc(sample_datetime_naive)
        assert result.tzinfo == timezone.utc
        assert result.hour == 21

    def test_ensure_utc_already_utc(self, sample_datetime_utc):
        """UTC datetime should remain unchanged."""
        result = ensure_utc(sample_datetime_utc)
        assert result == sample_datetime_utc

    def test_ensure_utc_converts_offset(self, sample_datetime_offset):
        """Non-UTC datetime should be converted to UTC."""
        result = ensure_utc(sample_datetime_offset)
        assert result.tzinfo == timezone.utc
        assert result.hour == 21  # 16:35 EST = 21:35 UTC

    def test_normalize_datetime(self, sample_datetime_naive):
        """normalize_datetime should call ensure_utc."""
        result = normalize_datetime(sample_datetime_naive)
        assert result.tzinfo == timezone.utc

    def test_datetime_format_with_z_suffix(self, sample_datetime_utc):
        """Datetimes should serialize with Z suffix."""
        result = format_datetime_canonical(sample_datetime_utc)
        assert result == "2026-01-27T21:35:00Z"
        assert result.endswith("Z")

    def test_naive_and_aware_serialize_same(
        self, sample_datetime_naive, sample_datetime_utc
    ):
        """Naive UTC and aware UTC should serialize identically."""
        result_naive = format_datetime_canonical(sample_datetime_naive)
        result_aware = format_datetime_canonical(sample_datetime_utc)
        assert result_naive == result_aware

    def test_offset_converts_to_utc_in_output(self, sample_datetime_offset):
        """Non-UTC datetime should serialize as UTC."""
        result = format_datetime_canonical(sample_datetime_offset)
        assert result == "2026-01-27T21:35:00Z"

    def test_datetime_with_microseconds(self):
        """Datetimes with microseconds should include them."""
        dt = datetime(2026, 1, 27, 21, 35, 0, 123456, tzinfo=timezone.utc)
        result = format_datetime_canonical(dt)
        assert result == "2026-01-27T21:35:00.123456Z"

    def test_datetime_in_model(self, sample_market_spec):
        """Datetimes in models should serialize correctly."""
        result = dumps_canonical(sample_market_spec)
        assert '"2026-03-31T23:59:59Z"' in result


# =============================================================================
# Test C: Exclude None Fields
# =============================================================================


class TestExcludeNone:
    """Tests for None field exclusion."""

    def test_none_excluded_from_dict(self):
        """None values should not appear in output."""
        data = {"a": 1, "b": None, "c": 3}
        result = dumps_canonical(data)
        parsed = json.loads(result)
        assert "b" not in parsed
        assert parsed == {"a": 1, "c": 3}

    def test_none_excluded_from_model(self):
        """Optional None fields in models should not appear."""
        model = SampleModel(name="test", value=42, optional_field=None)
        result = dumps_canonical(model)
        parsed = json.loads(result)
        assert "optional_field" not in parsed

    def test_empty_string_not_excluded(self):
        """Empty strings should NOT be excluded (only None)."""
        data = {"a": "", "b": 1}
        result = dumps_canonical(data)
        parsed = json.loads(result)
        assert "a" in parsed
        assert parsed["a"] == ""

    def test_zero_not_excluded(self):
        """Zero values should NOT be excluded (only None)."""
        data = {"a": 0, "b": 1}
        result = dumps_canonical(data)
        parsed = json.loads(result)
        assert "a" in parsed
        assert parsed["a"] == 0

    def test_false_not_excluded(self):
        """False values should NOT be excluded (only None)."""
        data = {"a": False, "b": True}
        result = dumps_canonical(data)
        parsed = json.loads(result)
        assert "a" in parsed
        assert parsed["a"] is False

    def test_nested_none_excluded(self):
        """None values in nested structures should be excluded."""
        data = {"outer": {"a": 1, "b": None}, "c": None}
        result = dumps_canonical(data)
        parsed = json.loads(result)
        assert parsed == {"outer": {"a": 1}}


# =============================================================================
# Test D: Float Safety
# =============================================================================


class TestFloatSafety:
    """Tests for NaN/Infinity rejection."""

    def test_nan_raises_exception(self):
        """NaN should raise CanonicalizationException."""
        data = {"value": float("nan")}
        with pytest.raises(CanonicalizationException) as exc_info:
            dumps_canonical(data)
        assert "Non-finite float" in str(exc_info.value)

    def test_positive_infinity_raises(self):
        """Positive infinity should raise CanonicalizationException."""
        data = {"value": float("inf")}
        with pytest.raises(CanonicalizationException):
            dumps_canonical(data)

    def test_negative_infinity_raises(self):
        """Negative infinity should raise CanonicalizationException."""
        data = {"value": float("-inf")}
        with pytest.raises(CanonicalizationException):
            dumps_canonical(data)

    def test_normal_float_works(self):
        """Normal floats should serialize correctly."""
        data = {"value": 3.14159}
        result = dumps_canonical(data)
        parsed = json.loads(result)
        assert math.isclose(parsed["value"], 3.14159)

    def test_float_in_nested_structure(self):
        """NaN in nested structure should also raise."""
        data = {"outer": {"inner": float("nan")}}
        with pytest.raises(CanonicalizationException):
            dumps_canonical(data)

    def test_float_in_list(self):
        """NaN in list should also raise."""
        data = {"values": [1.0, float("nan"), 3.0]}
        with pytest.raises(CanonicalizationException):
            dumps_canonical(data)

    def test_very_large_float_works(self):
        """Very large (but finite) floats should work."""
        data = {"value": 1e308}
        result = dumps_canonical(data)
        parsed = json.loads(result)
        assert parsed["value"] == 1e308


# =============================================================================
# Test E: Round Trip Stability
# =============================================================================


class TestRoundTripStability:
    """Tests for serialization/deserialization stability."""

    def test_simple_dict_roundtrip(self):
        """Simple dict should round-trip correctly."""
        data = {"a": 1, "b": "hello", "c": [1, 2, 3]}
        serialized = dumps_canonical(data)
        deserialized = loads_canonical(serialized)
        assert deserialized == data

    def test_model_roundtrip(self, sample_market_spec):
        """Model should round-trip to dict form."""
        serialized = dumps_canonical(sample_market_spec)
        deserialized = loads_canonical(serialized)
        assert deserialized["market_id"] == "test-market-001"
        assert "question" in deserialized

    def test_canonical_dict_roundtrip(self, sample_market_spec):
        """to_canonical_json_dict should produce consistent output."""
        dict1 = to_canonical_json_dict(sample_market_spec)
        serialized = dumps_canonical(dict1)
        dict2 = loads_canonical(serialized)
        assert dict1 == dict2

    def test_nested_model_roundtrip(self, sample_evidence_item):
        """Nested model should round-trip correctly."""
        serialized = dumps_canonical(sample_evidence_item)
        deserialized = loads_canonical(serialized)
        assert deserialized["evidence_id"] == "evidence-001"
        assert deserialized["provenance"]["source_id"] == "test-source"


# =============================================================================
# Test: Enum Serialization
# =============================================================================


class TestEnumSerialization:
    """Tests for enum value serialization."""

    def test_enum_serializes_to_value(self):
        """Enums should serialize to their string value."""
        data = {"status": SampleEnum.OPTION_A}
        result = dumps_canonical(data)
        parsed = json.loads(result)
        assert parsed["status"] == "option_a"

    def test_enum_in_model(self):
        """Enums in models should serialize correctly."""
        verdict = DeterministicVerdict(
            market_id="test-001",
            outcome="YES",
            confidence=0.95,
            resolution_time=datetime(2026, 1, 27, 21, 35, 0, tzinfo=timezone.utc),
            resolution_rule_id="rule-yes",
        )
        result = dumps_canonical(verdict)
        parsed = json.loads(result)
        assert parsed["outcome"] == "YES"


# =============================================================================
# Test: Canonical Equality
# =============================================================================


class TestCanonicalEquality:
    """Tests for canonical_equals function."""

    def test_equal_dicts(self):
        """Equal dicts should be canonically equal."""
        dict1 = {"a": 1, "b": 2}
        dict2 = {"b": 2, "a": 1}
        assert canonical_equals(dict1, dict2)

    def test_unequal_dicts(self):
        """Unequal dicts should not be canonically equal."""
        dict1 = {"a": 1, "b": 2}
        dict2 = {"a": 1, "b": 3}
        assert not canonical_equals(dict1, dict2)


# =============================================================================
# Test: SourceTarget and SelectionPolicy (NEW)
# =============================================================================


class TestSourceTarget:
    """Tests for SourceTarget schema."""

    def test_source_target_creation(self, sample_source_target):
        """SourceTarget should be created with valid data."""
        assert sample_source_target.source_id == "http"
        assert sample_source_target.uri == "https://api.example.com/data?param=value"
        assert sample_source_target.method == "GET"

    def test_source_target_serialization(self, sample_source_target):
        """SourceTarget should serialize correctly."""
        result = dumps_canonical(sample_source_target)
        parsed = json.loads(result)
        assert parsed["source_id"] == "http"
        assert parsed["uri"] == "https://api.example.com/data?param=value"

    def test_uri_treated_as_opaque_string(self):
        """URI should be treated as opaque - no normalization."""
        # URIs with different representations should NOT be normalized
        target1 = SourceTarget(
            source_id="http",
            uri="https://api.example.com/data?b=2&a=1",
        )
        target2 = SourceTarget(
            source_id="http",
            uri="https://api.example.com/data?a=1&b=2",
        )
        # These should serialize differently (no query param sorting)
        result1 = dumps_canonical(target1)
        result2 = dumps_canonical(target2)
        assert result1 != result2

    def test_source_target_with_body(self):
        """SourceTarget with body should serialize correctly."""
        target = SourceTarget(
            source_id="http",
            uri="https://api.example.com/rpc",
            method="POST",
            body={"jsonrpc": "2.0", "method": "getData"},
        )
        result = dumps_canonical(target)
        parsed = json.loads(result)
        assert parsed["method"] == "POST"
        assert parsed["body"]["jsonrpc"] == "2.0"

    def test_source_target_without_operation_excluded_from_canonical(self):
        """SourceTarget with operation/search_query None should not include them in canonical (backward compatible hash)."""
        target = SourceTarget(
            source_id="http",
            uri="https://api.example.com/data",
        )
        result = dumps_canonical(target)
        parsed = json.loads(result)
        assert "operation" not in parsed
        assert "search_query" not in parsed

    def test_source_target_with_operation_search_query_serialization(self):
        """SourceTarget with operation and search_query should serialize and round-trip correctly."""
        target = SourceTarget(
            source_id="http",
            uri="https://www.example.org",
            operation="search",
            search_query='site:example.org "exact phrase"',
        )
        result = dumps_canonical(target)
        parsed = json.loads(result)
        assert parsed.get("operation") == "search"
        assert parsed.get("search_query") == 'site:example.org "exact phrase"'
        # Round-trip
        loaded = SourceTarget.model_validate(parsed)
        assert loaded.operation == "search"
        assert loaded.search_query == 'site:example.org "exact phrase"'


class TestSelectionPolicy:
    """Tests for SelectionPolicy schema."""

    def test_selection_policy_creation(self, sample_selection_policy):
        """SelectionPolicy should be created with valid data."""
        assert sample_selection_policy.strategy == "fallback_chain"
        assert sample_selection_policy.min_sources == 1

    def test_quorum_validation(self):
        """Quorum cannot exceed max_sources."""
        with pytest.raises(ValidationError):
            SelectionPolicy(
                strategy="multi_source_quorum",
                max_sources=2,
                quorum=3,  # Invalid: quorum > max_sources
            )

    def test_min_max_validation(self):
        """min_sources cannot exceed max_sources."""
        with pytest.raises(ValidationError):
            SelectionPolicy(
                strategy="single_best",
                min_sources=5,
                max_sources=3,  # Invalid: min > max
            )


class TestDataRequirementWithSourceTargets:
    """Tests for DataRequirement with source_targets."""

    def test_data_requirement_requires_source_targets(self):
        """DataRequirement must have at least one source_target."""
        with pytest.raises(ValidationError):
            DataRequirement(
                requirement_id="req-001",
                description="Test requirement",
                source_targets=[],  # Empty list should fail
                selection_policy=SelectionPolicy(strategy="single_best"),
            )

    def test_data_requirement_serialization(self, sample_data_requirement):
        """DataRequirement should serialize with source_targets."""
        result = dumps_canonical(sample_data_requirement)
        parsed = json.loads(result)
        assert "source_targets" in parsed
        assert len(parsed["source_targets"]) == 1
        assert parsed["source_targets"][0]["source_id"] == "http"


# =============================================================================
# Test: Source Targets Order Preservation (NEW - Section 4.4)
# =============================================================================


class TestSourceTargetsOrderPreservation:
    """Tests for source_targets order preservation during serialization."""

    def test_source_targets_order_preserved(self):
        """source_targets order MUST be preserved (not sorted)."""
        targets = [
            SourceTarget(source_id="source-z", uri="https://z.example.com"),
            SourceTarget(source_id="source-a", uri="https://a.example.com"),
            SourceTarget(source_id="source-m", uri="https://m.example.com"),
        ]
        req = DataRequirement(
            requirement_id="req-001",
            description="Test",
            source_targets=targets,
            selection_policy=SelectionPolicy(strategy="fallback_chain"),
        )
        
        result = dumps_canonical(req)
        parsed = json.loads(result)
        
        # Order should be preserved: z, a, m (NOT sorted to a, m, z)
        assert parsed["source_targets"][0]["source_id"] == "source-z"
        assert parsed["source_targets"][1]["source_id"] == "source-a"
        assert parsed["source_targets"][2]["source_id"] == "source-m"

    def test_source_targets_order_semantically_meaningful(self):
        """Different orders should produce different canonical outputs."""
        targets_order1 = [
            SourceTarget(source_id="primary", uri="https://primary.com"),
            SourceTarget(source_id="fallback", uri="https://fallback.com"),
        ]
        targets_order2 = [
            SourceTarget(source_id="fallback", uri="https://fallback.com"),
            SourceTarget(source_id="primary", uri="https://primary.com"),
        ]
        
        req1 = DataRequirement(
            requirement_id="req-001",
            description="Test",
            source_targets=targets_order1,
            selection_policy=SelectionPolicy(strategy="fallback_chain"),
        )
        req2 = DataRequirement(
            requirement_id="req-001",
            description="Test",
            source_targets=targets_order2,
            selection_policy=SelectionPolicy(strategy="fallback_chain"),
        )
        
        # Different order = different canonical output
        assert dumps_canonical(req1) != dumps_canonical(req2)


# =============================================================================
# Test: Timestamp Determinism (NEW - Section 4.5)
# =============================================================================


class TestTimestampDeterminism:
    """Tests for PromptSpec.created_at optional behavior for deterministic hashing."""

    def test_prompt_spec_created_at_optional(self, sample_market_spec):
        """PromptSpec.created_at should be optional and default to None."""
        spec = PromptSpec(
            market=sample_market_spec,
            prediction_semantics=PredictionSemantics(
                target_entity="Company X",
                predicate="will acquire Company Y",
            ),
            # Note: created_at not provided - should default to None
        )
        assert spec.created_at is None

    def test_prompt_spec_without_created_at_excludes_field(self, sample_market_spec):
        """When created_at is None, it should be excluded from serialization."""
        spec = PromptSpec(
            market=sample_market_spec,
            prediction_semantics=PredictionSemantics(
                target_entity="Company X",
                predicate="will acquire Company Y",
            ),
        )
        
        result = dumps_canonical(spec)
        parsed = json.loads(result)
        
        # created_at should NOT be in the output
        assert "created_at" not in parsed

    def test_identical_inputs_produce_identical_hash(self, sample_market_spec):
        """Identical inputs should produce identical canonical output (for hashing)."""
        spec1 = PromptSpec(
            market=sample_market_spec,
            prediction_semantics=PredictionSemantics(
                target_entity="Company X",
                predicate="will acquire Company Y",
            ),
        )
        spec2 = PromptSpec(
            market=sample_market_spec,
            prediction_semantics=PredictionSemantics(
                target_entity="Company X",
                predicate="will acquire Company Y",
            ),
        )
        
        # Both should produce identical canonical output
        assert dumps_canonical(spec1) == dumps_canonical(spec2)

    def test_prompt_spec_with_created_at_includes_field(self, sample_market_spec):
        """When created_at is provided, it should be included in serialization."""
        timestamp = datetime(2026, 1, 27, 21, 35, 0, tzinfo=timezone.utc)
        spec = PromptSpec(
            market=sample_market_spec,
            prediction_semantics=PredictionSemantics(
                target_entity="Company X",
                predicate="will acquire Company Y",
            ),
            created_at=timestamp,
        )
        
        result = dumps_canonical(spec)
        parsed = json.loads(result)
        
        # created_at SHOULD be in the output when provided
        assert "created_at" in parsed
        assert parsed["created_at"] == "2026-01-27T21:35:00Z"

    def test_different_timestamps_produce_different_hash(self, sample_market_spec):
        """Different timestamps should produce different canonical output."""
        spec1 = PromptSpec(
            market=sample_market_spec,
            prediction_semantics=PredictionSemantics(
                target_entity="Company X",
                predicate="will acquire Company Y",
            ),
            created_at=datetime(2026, 1, 27, 21, 35, 0, tzinfo=timezone.utc),
        )
        spec2 = PromptSpec(
            market=sample_market_spec,
            prediction_semantics=PredictionSemantics(
                target_entity="Company X",
                predicate="will acquire Company Y",
            ),
            created_at=datetime(2026, 1, 27, 21, 36, 0, tzinfo=timezone.utc),  # Different time
        )
        
        # Different timestamps = different canonical output
        assert dumps_canonical(spec1) != dumps_canonical(spec2)


# =============================================================================
# Test: Complex Schema Integration
# =============================================================================


class TestComplexSchemaIntegration:
    """Integration tests with complex schemas."""

    def test_evidence_bundle_serialization(self, sample_evidence_item):
        """EvidenceBundle should serialize correctly."""
        bundle = EvidenceBundle(
            bundle_id="bundle-001",
            market_id="mk_test",
            plan_id="plan_test",
            items=[sample_evidence_item],
            collected_at=datetime(2026, 1, 27, 21, 0, 0, tzinfo=timezone.utc),
        )
        result = dumps_canonical(bundle)
        parsed = json.loads(result)
        
        assert parsed["bundle_id"] == "bundle-001"
        assert len(parsed["items"]) == 1
        assert parsed["items"][0]["evidence_id"] == "evidence-001"

    def test_deterministic_verdict_serialization(self):
        """DeterministicVerdict should serialize correctly."""
        verdict = DeterministicVerdict(
            market_id="market-001",
            outcome="YES",
            confidence=0.95,
            resolution_time=datetime(2026, 1, 27, 21, 35, 0, tzinfo=timezone.utc),
            resolution_rule_id="rule-yes",
            evidence_root="0x1234567890abcdef",
            reasoning_root="0xfedcba0987654321",
            selected_leaf_refs=["leaf-1", "leaf-2"],
        )
        result = dumps_canonical(verdict)
        parsed = json.loads(result)
        
        assert parsed["outcome"] == "YES"
        assert parsed["confidence"] == 0.95
        assert parsed["evidence_root"] == "0x1234567890abcdef"
        assert parsed["selected_leaf_refs"] == ["leaf-1", "leaf-2"]

    def test_market_spec_full_serialization(self, sample_market_spec):
        """Full MarketSpec should serialize correctly."""
        result = dumps_canonical(sample_market_spec)
        parsed = json.loads(result)
        
        # Check nested structures are preserved
        assert "resolution_window" in parsed
        assert "resolution_rules" in parsed
        assert "dispute_policy" in parsed
        assert len(parsed["resolution_rules"]["rules"]) == 2

    def test_full_prompt_spec_with_data_requirements(self, sample_market_spec):
        """Full PromptSpec with DataRequirements should serialize correctly."""
        spec = PromptSpec(
            market=sample_market_spec,
            prediction_semantics=PredictionSemantics(
                target_entity="Company X",
                predicate="will acquire Company Y",
                threshold="official announcement",
                timeframe="before March 31, 2026",
            ),
            data_requirements=[
                DataRequirement(
                    requirement_id="req-001",
                    description="Fetch SEC filings",
                    source_targets=[
                        SourceTarget(
                            source_id="sec-edgar",
                            uri="https://www.sec.gov/cgi-bin/browse-edgar",
                            method="GET",
                        ),
                    ],
                    selection_policy=SelectionPolicy(strategy="single_best"),
                ),
            ],
        )
        
        result = dumps_canonical(spec)
        parsed = json.loads(result)
        
        assert len(parsed["data_requirements"]) == 1
        assert parsed["data_requirements"][0]["source_targets"][0]["source_id"] == "sec-edgar"


# =============================================================================
# Test: No Whitespace
# =============================================================================


class TestNoWhitespace:
    """Tests for compact output (no whitespace)."""

    def test_no_spaces_in_output(self):
        """Output should have no extraneous spaces."""
        data = {"a": 1, "b": [1, 2, 3], "c": {"d": 4}}
        result = dumps_canonical(data)
        # Should not have ": " or ", " - only ":" and ","
        assert ": " not in result
        assert ", " not in result

    def test_separators_are_minimal(self):
        """Separators should be minimal."""
        data = {"key": "value"}
        result = dumps_canonical(data)
        assert result == '{"key":"value"}'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])