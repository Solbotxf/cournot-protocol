"""
Module 05 - Collector Determinism Tests

Tests for deterministic evidence collection, stable IDs, and ordering.
"""

import pytest
from datetime import datetime, timezone
from typing import Any
from unittest.mock import Mock, patch

from core.schemas.canonical import dumps_canonical
from core.schemas.evidence import EvidenceBundle, EvidenceItem
from core.schemas.market import (
    DisputePolicy,
    MarketSpec,
    ResolutionRule,
    ResolutionRules,
    ResolutionWindow,
    SourcePolicy,
)
from core.schemas.prompts import (
    DataRequirement,
    PredictionSemantics,
    PromptSpec,
    SelectionPolicy,
    SourceTarget,
)
from core.schemas.transport import ToolPlan

from agents.collector import CollectorAgent, CollectorConfig
from agents.collector.data_sources import BaseSource, FetchedArtifact


class MockSource(BaseSource):
    """Mock source for testing."""
    
    source_id: str = "mock"
    
    def __init__(self, responses: dict[str, FetchedArtifact]):
        super().__init__()
        self.responses = responses
        self.call_count = 0
    
    def fetch(self, target, *, timeout_s=20):
        self.call_count += 1
        key = target.uri
        if key in self.responses:
            return self.responses[key]
        return FetchedArtifact(
            raw_bytes=b"default response",
            content_type="text",
            parsed="default response",
            status_code=200,
        )


def create_test_prompt_spec(
    market_id: str = "test_market_001",
    requirements: list[DataRequirement] | None = None
) -> PromptSpec:
    """Create a test PromptSpec."""
    if requirements is None:
        requirements = [
            DataRequirement(
                requirement_id="req_001",
                description="Test requirement",
                source_targets=[
                    SourceTarget(
                        source_id="mock",
                        uri="https://api.example.com/data",
                        method="GET",
                        expected_content_type="json",
                    )
                ],
                selection_policy=SelectionPolicy(
                    strategy="single_best",
                    min_sources=1,
                    max_sources=1,
                    quorum=1,
                ),
                min_provenance_tier=0,
            )
        ]
    
    return PromptSpec(
        market=MarketSpec(
            market_id=market_id,
            question="Test question?",
            event_definition="Test event definition",
            timezone="UTC",
            resolution_deadline=datetime(2026, 12, 31, tzinfo=timezone.utc),
            resolution_window=ResolutionWindow(
                start=datetime(2026, 1, 1, tzinfo=timezone.utc),
                end=datetime(2026, 12, 31, tzinfo=timezone.utc),
            ),
            resolution_rules=ResolutionRules(rules=[
                ResolutionRule(rule_id="rule_001", description="Default rule")
            ]),
            allowed_sources=[
                SourcePolicy(source_id="mock", kind="api", allow=True)
            ],
            min_provenance_tier=0,
            dispute_policy=DisputePolicy(dispute_window_seconds=86400),
        ),
        prediction_semantics=PredictionSemantics(
            target_entity="Test",
            predicate="will happen",
        ),
        data_requirements=requirements,
        # created_at is None for determinism
    )


def create_test_tool_plan(plan_id: str = "plan_001") -> ToolPlan:
    """Create a test ToolPlan."""
    return ToolPlan(
        plan_id=plan_id,
        requirements=["req_001"],
        sources=["mock"],
    )


class TestEvidenceIdDeterminism:
    """Tests for deterministic evidence ID generation."""
    
    def test_same_response_same_id(self):
        """Same request + response should produce same evidence ID."""
        # Create mock responses
        response = FetchedArtifact(
            raw_bytes=b'{"value": 100}',
            content_type="json",
            parsed={"value": 100},
            status_code=200,
        )
        
        mock_source = MockSource({
            "https://api.example.com/data": response
        })
        
        config = CollectorConfig(include_timestamps=False)
        agent = CollectorAgent(
            config=config,
            source_adapters={"mock": mock_source}
        )
        
        prompt_spec = create_test_prompt_spec()
        tool_plan = create_test_tool_plan()
        
        # Collect twice
        bundle1 = agent.collect(prompt_spec, tool_plan)
        bundle2 = agent.collect(prompt_spec, tool_plan)
        
        # IDs should be identical
        assert bundle1.items[0].evidence_id == bundle2.items[0].evidence_id
    
    def test_different_response_different_id(self):
        """Different responses should produce different evidence IDs."""
        response1 = FetchedArtifact(
            raw_bytes=b'{"value": 100}',
            content_type="json",
            parsed={"value": 100},
            status_code=200,
        )
        response2 = FetchedArtifact(
            raw_bytes=b'{"value": 200}',
            content_type="json",
            parsed={"value": 200},
            status_code=200,
        )
        
        config = CollectorConfig(include_timestamps=False)
        
        # First collection
        agent1 = CollectorAgent(
            config=config,
            source_adapters={"mock": MockSource({"https://api.example.com/data": response1})}
        )
        bundle1 = agent1.collect(create_test_prompt_spec(), create_test_tool_plan())
        
        # Second collection with different response
        agent2 = CollectorAgent(
            config=config,
            source_adapters={"mock": MockSource({"https://api.example.com/data": response2})}
        )
        bundle2 = agent2.collect(create_test_prompt_spec(), create_test_tool_plan())
        
        # IDs should be different
        assert bundle1.items[0].evidence_id != bundle2.items[0].evidence_id


class TestOrderingDeterminism:
    """Tests for deterministic evidence ordering."""
    
    def test_requirement_order_preserved(self):
        """Evidence items should be ordered by requirement order."""
        response1 = FetchedArtifact(
            raw_bytes=b'{"id": 1}',
            content_type="json",
            parsed={"id": 1},
            status_code=200,
        )
        response2 = FetchedArtifact(
            raw_bytes=b'{"id": 2}',
            content_type="json",
            parsed={"id": 2},
            status_code=200,
        )
        
        mock_source = MockSource({
            "https://api.example.com/data1": response1,
            "https://api.example.com/data2": response2,
        })
        
        requirements = [
            DataRequirement(
                requirement_id="req_001",
                description="First requirement",
                source_targets=[
                    SourceTarget(
                        source_id="mock",
                        uri="https://api.example.com/data1",
                        method="GET",
                        expected_content_type="json",
                    )
                ],
                selection_policy=SelectionPolicy(
                    strategy="single_best",
                    min_sources=1,
                    max_sources=1,
                    quorum=1,
                ),
            ),
            DataRequirement(
                requirement_id="req_002",
                description="Second requirement",
                source_targets=[
                    SourceTarget(
                        source_id="mock",
                        uri="https://api.example.com/data2",
                        method="GET",
                        expected_content_type="json",
                    )
                ],
                selection_policy=SelectionPolicy(
                    strategy="single_best",
                    min_sources=1,
                    max_sources=1,
                    quorum=1,
                ),
            ),
        ]
        
        config = CollectorConfig(include_timestamps=False)
        agent = CollectorAgent(
            config=config,
            source_adapters={"mock": mock_source}
        )
        
        prompt_spec = create_test_prompt_spec(requirements=requirements)
        tool_plan = ToolPlan(
            plan_id="plan_001",
            requirements=["req_001", "req_002"],
            sources=["mock"],
        )
        
        bundle = agent.collect(prompt_spec, tool_plan)
        
        # Check ordering
        assert len(bundle.items) == 2
        assert bundle.items[0].requirement_id == "req_001"
        assert bundle.items[1].requirement_id == "req_002"
    
    def test_source_target_order_preserved(self):
        """Within a requirement, source target order should be preserved."""
        response1 = FetchedArtifact(
            raw_bytes=b"",
            content_type="json",
            parsed=None,
            status_code=500,
            error="Server error",
        )
        response2 = FetchedArtifact(
            raw_bytes=b'{"success": true}',
            content_type="json",
            parsed={"success": True},
            status_code=200,
        )
        
        mock_source = MockSource({
            "https://api.example.com/primary": response1,  # Will fail
            "https://api.example.com/fallback": response2,  # Will succeed
        })
        
        requirements = [
            DataRequirement(
                requirement_id="req_001",
                description="Requirement with fallback",
                source_targets=[
                    SourceTarget(
                        source_id="mock",
                        uri="https://api.example.com/primary",
                        method="GET",
                        expected_content_type="json",
                    ),
                    SourceTarget(
                        source_id="mock",
                        uri="https://api.example.com/fallback",
                        method="GET",
                        expected_content_type="json",
                    ),
                ],
                selection_policy=SelectionPolicy(
                    strategy="fallback_chain",
                    min_sources=1,
                    max_sources=2,
                    quorum=1,
                ),
            ),
        ]
        
        config = CollectorConfig(include_timestamps=False)
        agent = CollectorAgent(
            config=config,
            source_adapters={"mock": mock_source}
        )
        
        prompt_spec = create_test_prompt_spec(requirements=requirements)
        tool_plan = create_test_tool_plan()
        
        bundle = agent.collect(prompt_spec, tool_plan)
        
        # Should have used fallback
        assert len(bundle.items) == 1
        assert "fallback" in bundle.items[0].source.uri


class TestCanonicalDumpsDeterminism:
    """Tests for canonical serialization determinism."""
    
    def test_canonical_dumps_stable(self):
        """Canonical dumps of same bundle should be identical."""
        response = FetchedArtifact(
            raw_bytes=b'{"value": 42}',
            content_type="json",
            parsed={"value": 42},
            status_code=200,
        )
        
        mock_source = MockSource({
            "https://api.example.com/data": response
        })
        
        config = CollectorConfig(include_timestamps=False)
        agent = CollectorAgent(
            config=config,
            source_adapters={"mock": mock_source}
        )
        
        prompt_spec = create_test_prompt_spec()
        tool_plan = create_test_tool_plan()
        
        bundle1 = agent.collect(prompt_spec, tool_plan)
        bundle2 = agent.collect(prompt_spec, tool_plan)
        
        # Canonical dumps should be identical
        dump1 = dumps_canonical(bundle1.items[0])
        dump2 = dumps_canonical(bundle2.items[0])
        
        assert dump1 == dump2
    
    def test_timestamps_excluded_when_disabled(self):
        """Timestamps should be excluded from evidence when disabled."""
        response = FetchedArtifact(
            raw_bytes=b'{"test": "data"}',
            content_type="json",
            parsed={"test": "data"},
            status_code=200,
        )
        
        mock_source = MockSource({
            "https://api.example.com/data": response
        })
        
        config = CollectorConfig(include_timestamps=False)
        agent = CollectorAgent(
            config=config,
            source_adapters={"mock": mock_source}
        )
        
        bundle = agent.collect(create_test_prompt_spec(), create_test_tool_plan())
        
        # Check timestamps are None
        assert bundle.collection_time is None
        assert bundle.items[0].retrieval.retrieved_at is None


class TestBundleIdDeterminism:
    """Tests for deterministic bundle ID generation."""
    
    def test_same_inputs_same_bundle_id(self):
        """Same market_id + plan_id should produce same bundle_id."""
        response = FetchedArtifact(
            raw_bytes=b'{"data": 1}',
            content_type="json",
            parsed={"data": 1},
            status_code=200,
        )
        
        mock_source = MockSource({
            "https://api.example.com/data": response
        })
        
        config = CollectorConfig(include_timestamps=False)
        agent = CollectorAgent(
            config=config,
            source_adapters={"mock": mock_source}
        )
        
        prompt_spec = create_test_prompt_spec()
        tool_plan = create_test_tool_plan()
        
        bundle1 = agent.collect(prompt_spec, tool_plan)
        bundle2 = agent.collect(prompt_spec, tool_plan)
        
        assert bundle1.bundle_id == bundle2.bundle_id
    
    def test_different_market_different_bundle_id(self):
        """Different market_id should produce different bundle_id."""
        response = FetchedArtifact(
            raw_bytes=b'{"data": 1}',
            content_type="json",
            parsed={"data": 1},
            status_code=200,
        )
        
        mock_source = MockSource({
            "https://api.example.com/data": response
        })
        
        config = CollectorConfig(include_timestamps=False)
        
        agent1 = CollectorAgent(
            config=config,
            source_adapters={"mock": mock_source}
        )
        agent2 = CollectorAgent(
            config=config,
            source_adapters={"mock": mock_source}
        )
        
        prompt_spec1 = create_test_prompt_spec(market_id="market_001")
        prompt_spec2 = create_test_prompt_spec(market_id="market_002")
        tool_plan = create_test_tool_plan()
        
        bundle1 = agent1.collect(prompt_spec1, tool_plan)
        bundle2 = agent2.collect(prompt_spec2, tool_plan)
        
        assert bundle1.bundle_id != bundle2.bundle_id


class TestFingerprintDeterminism:
    """Tests for request/response fingerprint determinism."""
    
    def test_request_fingerprint_stable(self):
        """Request fingerprint should be stable for same target."""
        response = FetchedArtifact(
            raw_bytes=b'{"data": "test"}',
            content_type="json",
            parsed={"data": "test"},
            status_code=200,
        )
        
        mock_source = MockSource({
            "https://api.example.com/data": response
        })
        
        config = CollectorConfig(include_timestamps=False)
        agent = CollectorAgent(
            config=config,
            source_adapters={"mock": mock_source}
        )
        
        prompt_spec = create_test_prompt_spec()
        tool_plan = create_test_tool_plan()
        
        bundle1 = agent.collect(prompt_spec, tool_plan)
        bundle2 = agent.collect(prompt_spec, tool_plan)
        
        fp1 = bundle1.items[0].retrieval.request_fingerprint
        fp2 = bundle2.items[0].retrieval.request_fingerprint
        
        assert fp1 == fp2
        assert fp1.startswith("0x")
    
    def test_response_fingerprint_stable(self):
        """Response fingerprint should be stable for same response."""
        response = FetchedArtifact(
            raw_bytes=b'{"data": "stable"}',
            content_type="json",
            parsed={"data": "stable"},
            status_code=200,
        )
        
        mock_source = MockSource({
            "https://api.example.com/data": response
        })
        
        config = CollectorConfig(include_timestamps=False)
        agent = CollectorAgent(
            config=config,
            source_adapters={"mock": mock_source}
        )
        
        prompt_spec = create_test_prompt_spec()
        tool_plan = create_test_tool_plan()
        
        bundle1 = agent.collect(prompt_spec, tool_plan)
        bundle2 = agent.collect(prompt_spec, tool_plan)
        
        fp1 = bundle1.items[0].retrieval.response_fingerprint
        fp2 = bundle2.items[0].retrieval.response_fingerprint
        
        assert fp1 == fp2
        assert fp1.startswith("0x")