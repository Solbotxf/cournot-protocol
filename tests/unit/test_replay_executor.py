"""
Module 08 - Validator/Sentinel Tests: Replay Executor

Tests for ReplayExecutor:
- Mock collector returns deterministic evidence bundle
- Replay divergence produces replay_divergence challenge
- Replay disabled or missing tool_plan handles gracefully

Owner: Protocol Verification Engineer
Module ID: M08
"""

import pytest
from typing import Any, Optional

from core.schemas.evidence import EvidenceBundle
from core.schemas.prompts import PromptSpec
from core.schemas.transport import ToolPlan

from agents.validator import (
    SentinelAgent,
    PoRPackage,
    ReplayExecutor,
    CollectorProtocol,
)

from fixtures import (
    make_valid_por_package,
    make_evidence_bundle,
    make_evidence_item,
    make_prompt_spec,
    make_tool_plan,
)


class MockCollector:
    """Mock collector for testing replay functionality."""

    def __init__(
        self,
        return_bundle: Optional[EvidenceBundle] = None,
        should_fail: bool = False,
    ):
        self.return_bundle = return_bundle
        self.should_fail = should_fail
        self.collect_calls: list[tuple[PromptSpec, ToolPlan]] = []

    def collect(
        self,
        prompt_spec: PromptSpec,
        tool_plan: ToolPlan,
    ) -> EvidenceBundle:
        """Mock collect method."""
        self.collect_calls.append((prompt_spec, tool_plan))

        if self.should_fail:
            raise RuntimeError("Mock collection failure")

        if self.return_bundle is not None:
            return self.return_bundle

        # Return a default bundle
        return make_evidence_bundle()


class TestReplayExecutorBasic:
    """Basic tests for ReplayExecutor."""

    def test_replay_executor_disabled_without_collector(self):
        """ReplayExecutor should be disabled without a collector."""
        executor = ReplayExecutor(collector=None)
        assert executor.is_enabled is False

    def test_replay_executor_enabled_with_collector(self):
        """ReplayExecutor should be enabled with a collector."""
        collector = MockCollector()
        executor = ReplayExecutor(collector=collector)
        assert executor.is_enabled is True

    def test_replay_executor_can_be_explicitly_disabled(self):
        """ReplayExecutor should respect explicit enabled=False."""
        collector = MockCollector()
        executor = ReplayExecutor(collector=collector, enabled=False)
        assert executor.is_enabled is False

    def test_replay_evidence_calls_collector(self):
        """replay_evidence should call the collector."""
        bundle = make_evidence_bundle()
        collector = MockCollector(return_bundle=bundle)
        executor = ReplayExecutor(collector=collector)

        prompt_spec = make_prompt_spec()
        tool_plan = make_tool_plan()

        replayed, result = executor.replay_evidence(prompt_spec, tool_plan)

        assert replayed is not None
        assert replayed.bundle_id == bundle.bundle_id
        assert len(collector.collect_calls) == 1
        assert result.ok is True

    def test_replay_evidence_handles_failure(self):
        """replay_evidence should handle collector failure gracefully."""
        collector = MockCollector(should_fail=True)
        executor = ReplayExecutor(collector=collector)

        prompt_spec = make_prompt_spec()
        tool_plan = make_tool_plan()

        replayed, result = executor.replay_evidence(prompt_spec, tool_plan)

        assert replayed is None
        assert result.ok is False
        assert any("failed" in c.message.lower() for c in result.checks)


class TestReplayExecutorComparison:
    """Tests for evidence comparison in ReplayExecutor."""

    def test_compare_identical_bundles_passes(self):
        """Comparing identical bundles should pass."""
        bundle = make_evidence_bundle()
        collector = MockCollector(return_bundle=bundle)
        executor = ReplayExecutor(collector=collector)

        result = executor.compare_evidence(bundle, bundle)

        assert result.verification.ok is True
        assert len(result.divergent_items) == 0

    def test_compare_different_item_count_detects_divergence(self):
        """Different item counts should be detected."""
        original = make_evidence_bundle(items=[
            make_evidence_item(evidence_id="ev_001"),
            make_evidence_item(evidence_id="ev_002"),
        ])
        replayed = make_evidence_bundle(items=[
            make_evidence_item(evidence_id="ev_001"),
        ])

        executor = ReplayExecutor(collector=None, enabled=False)
        result = executor.compare_evidence(original, replayed)

        # Should detect missing item
        assert result.verification.ok is False or len(result.divergent_items) > 0

    def test_compare_missing_items_creates_divergence(self):
        """Missing items in replay should create divergence entries."""
        original = make_evidence_bundle(items=[
            make_evidence_item(evidence_id="ev_001"),
            make_evidence_item(evidence_id="ev_002"),
        ])
        replayed = make_evidence_bundle(items=[
            make_evidence_item(evidence_id="ev_001"),
            # ev_002 missing
        ])

        executor = ReplayExecutor(collector=None, enabled=False)
        result = executor.compare_evidence(original, replayed)

        missing = [d for d in result.divergent_items if d.get("reason") == "missing_in_replay"]
        assert len(missing) >= 1
        assert any(d["evidence_id"] == "ev_002" for d in missing)

    def test_compare_different_content_detects_divergence(self):
        """Different content in same evidence_id should be detected."""
        original = make_evidence_bundle(items=[
            make_evidence_item(
                evidence_id="ev_001",
                content={"price": 100000.0},
            ),
        ])
        replayed = make_evidence_bundle(items=[
            make_evidence_item(
                evidence_id="ev_001",
                content={"price": 200000.0},  # Different content
            ),
        ])

        executor = ReplayExecutor(collector=None, enabled=False)
        result = executor.compare_evidence(original, replayed)

        # Should detect content mismatch (via hash or fingerprint)
        assert result.verification.ok is False or len(result.divergent_items) > 0


class TestReplayExecutorFullWorkflow:
    """Tests for full replay and compare workflow."""

    def test_replay_and_compare_identical(self):
        """Full workflow with identical results should pass."""
        original_bundle = make_evidence_bundle()
        collector = MockCollector(return_bundle=original_bundle)
        executor = ReplayExecutor(collector=collector)

        prompt_spec = make_prompt_spec()
        tool_plan = make_tool_plan()

        result = executor.replay_and_compare(prompt_spec, tool_plan, original_bundle)

        assert result.verification.ok is True
        assert len(result.divergent_items) == 0
        assert result.replayed_bundle is not None

    def test_replay_and_compare_divergent(self):
        """Full workflow with divergent results should detect differences."""
        original_bundle = make_evidence_bundle(items=[
            make_evidence_item(evidence_id="ev_001", content={"value": 100}),
        ])
        replayed_bundle = make_evidence_bundle(items=[
            make_evidence_item(evidence_id="ev_001", content={"value": 200}),
        ])

        collector = MockCollector(return_bundle=replayed_bundle)
        executor = ReplayExecutor(collector=collector)

        prompt_spec = make_prompt_spec()
        tool_plan = make_tool_plan()

        result = executor.replay_and_compare(prompt_spec, tool_plan, original_bundle)

        assert result.verification.ok is False or len(result.divergent_items) > 0
        assert result.replayed_bundle is not None


class TestSentinelReplayMode:
    """Tests for SentinelAgent replay mode."""

    def test_replay_mode_without_collector_warns(self):
        """Replay mode without collector should produce warning."""
        bundle, prompt_spec, evidence, trace, verdict, tool_plan = make_valid_por_package()

        package = PoRPackage(
            bundle=bundle,
            prompt_spec=prompt_spec,
            evidence=evidence,
            trace=trace,
            verdict=verdict,
            tool_plan=tool_plan,
        )

        # Sentinel without collector
        sentinel = SentinelAgent(strict_mode=True, collector=None)
        result, challenges = sentinel.verify(package, mode="replay")

        # Should still pass (replay is optional) but with warning
        warn_checks = [c for c in result.checks if c.severity == "warn" and "replay" in c.check_id.lower()]
        assert len(warn_checks) >= 1

    def test_replay_mode_without_tool_plan_warns(self):
        """Replay mode without tool_plan should produce warning."""
        bundle, prompt_spec, evidence, trace, verdict, _ = make_valid_por_package()

        package = PoRPackage(
            bundle=bundle,
            prompt_spec=prompt_spec,
            evidence=evidence,
            trace=trace,
            verdict=verdict,
            tool_plan=None,  # No tool_plan
        )

        collector = MockCollector()
        sentinel = SentinelAgent(strict_mode=True, collector=collector)
        result, challenges = sentinel.verify(package, mode="replay")

        # Should have warning about missing tool_plan
        warn_checks = [c for c in result.checks if "tool_plan" in c.check_id.lower()]
        assert len(warn_checks) >= 1

    def test_replay_mode_with_divergence_creates_challenge(self):
        """Replay mode with divergence should create replay_divergence challenge."""
        bundle, prompt_spec, evidence, trace, verdict, tool_plan = make_valid_por_package()

        # Create a different bundle for replay
        divergent_bundle = make_evidence_bundle(items=[
            make_evidence_item(
                evidence_id="ev_001",
                content={"price": 999999.0},  # Different from original
            ),
        ])

        collector = MockCollector(return_bundle=divergent_bundle)

        package = PoRPackage(
            bundle=bundle,
            prompt_spec=prompt_spec,
            evidence=evidence,
            trace=trace,
            verdict=verdict,
            tool_plan=tool_plan,
        )

        sentinel = SentinelAgent(strict_mode=True, collector=collector)
        result, challenges = sentinel.verify(package, mode="replay")

        # Should have replay_divergence challenge
        replay_challenges = [c for c in challenges if c.kind == "replay_divergence"]
        assert len(replay_challenges) >= 1

        challenge = replay_challenges[0]
        assert "original_evidence_root" in challenge.details
        assert "replayed_evidence_root" in challenge.details

    def test_replay_mode_with_matching_bundle_no_challenge(self):
        """Replay mode with matching bundle should not create challenge."""
        bundle, prompt_spec, evidence, trace, verdict, tool_plan = make_valid_por_package()

        # Return the same evidence bundle
        collector = MockCollector(return_bundle=evidence)

        package = PoRPackage(
            bundle=bundle,
            prompt_spec=prompt_spec,
            evidence=evidence,
            trace=trace,
            verdict=verdict,
            tool_plan=tool_plan,
        )

        sentinel = SentinelAgent(strict_mode=True, collector=collector)
        result, challenges = sentinel.verify(package, mode="replay")

        # Should pass without replay_divergence challenges
        replay_challenges = [c for c in challenges if c.kind == "replay_divergence"]
        assert len(replay_challenges) == 0

    def test_replay_collector_failure_handled_gracefully(self):
        """Collector failure during replay should be handled gracefully."""
        bundle, prompt_spec, evidence, trace, verdict, tool_plan = make_valid_por_package()

        # Collector that fails
        collector = MockCollector(should_fail=True)

        package = PoRPackage(
            bundle=bundle,
            prompt_spec=prompt_spec,
            evidence=evidence,
            trace=trace,
            verdict=verdict,
            tool_plan=tool_plan,
        )

        sentinel = SentinelAgent(strict_mode=True, collector=collector)

        # Should not raise exception
        result, challenges = sentinel.verify(package, mode="replay")

        # Verification should still work for non-replay checks
        # (depends on implementation - may pass or fail)
        assert result is not None


class TestReplayDeterminism:
    """Tests for determinism in replay mode."""

    def test_replay_produces_consistent_results(self):
        """Multiple replay calls should produce consistent results."""
        bundle, prompt_spec, evidence, trace, verdict, tool_plan = make_valid_por_package()

        collector = MockCollector(return_bundle=evidence)

        package = PoRPackage(
            bundle=bundle,
            prompt_spec=prompt_spec,
            evidence=evidence,
            trace=trace,
            verdict=verdict,
            tool_plan=tool_plan,
        )

        sentinel = SentinelAgent(strict_mode=True, collector=collector)

        # Run multiple times
        results = [sentinel.verify(package, mode="replay") for _ in range(3)]

        # All should have same ok status
        ok_values = [r[0].ok for r in results]
        assert all(v == ok_values[0] for v in ok_values)

        # All should have same number of challenges
        challenge_counts = [len(r[1]) for r in results]
        assert all(c == challenge_counts[0] for c in challenge_counts)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])