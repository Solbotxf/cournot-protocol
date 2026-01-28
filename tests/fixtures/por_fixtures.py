"""
PoR (Proof of Reasoning) bundle fixtures for M03+ and M08 tests.

Provides factory functions for:
- DeterministicVerdict
- ToolPlan
- Complete PoR packages (bundle + all artifacts)
- Tampered bundles for mismatch testing
"""

from typing import Any, Optional

from core.por.por_bundle import PoRBundle
from core.por.proof_of_reasoning import build_por_bundle
from core.schemas.verdict import DeterministicVerdict
from core.schemas.transport import ToolPlan

from .common import (
    make_prompt_spec,
    make_evidence_bundle,
    make_reasoning_trace,
)


# =============================================================================
# DeterministicVerdict Factory
# =============================================================================

def make_verdict(
    market_id: str = "test_market_001",
    outcome: str = "YES",
    confidence: float = 0.85,
    resolution_rule_id: str = "R_BINARY_DECISION",
) -> DeterministicVerdict:
    """
    Create a DeterministicVerdict for testing.
    
    Args:
        market_id: Market identifier (must match PromptSpec).
        outcome: Resolution outcome (YES, NO, INVALID).
        confidence: Confidence score [0, 1].
        resolution_rule_id: Applied resolution rule.
    
    Returns:
        A valid DeterministicVerdict with resolution_time=None for determinism.
    """
    return DeterministicVerdict(
        market_id=market_id,
        outcome=outcome,
        confidence=confidence,
        resolution_time=None,  # Optional for determinism
        resolution_rule_id=resolution_rule_id,
    )


# =============================================================================
# ToolPlan Factory
# =============================================================================

def make_tool_plan(
    plan_id: str = "plan_test_001",
    requirements: Optional[list[str]] = None,
    sources: Optional[list[str]] = None,
    min_provenance_tier: int = 0,
    allow_fallbacks: bool = True,
) -> ToolPlan:
    """
    Create a ToolPlan for testing.
    
    Args:
        plan_id: Unique plan identifier.
        requirements: List of requirement IDs to fulfill.
        sources: List of source IDs to use.
        min_provenance_tier: Minimum required provenance tier.
        allow_fallbacks: Whether fallback sources are allowed.
    
    Returns:
        A valid ToolPlan instance.
    """
    return ToolPlan(
        plan_id=plan_id,
        requirements=requirements or ["req_001"],
        sources=sources or ["exchange"],
        min_provenance_tier=min_provenance_tier,
        allow_fallbacks=allow_fallbacks,
    )


# =============================================================================
# Complete PoR Package Factory
# =============================================================================

def make_valid_por_package(
    market_id: str = "test_market_001",
    strict_mode: bool = True,
    outcome: str = "YES",
    confidence: float = 0.85,
) -> tuple[PoRBundle, Any, Any, Any, DeterministicVerdict, ToolPlan]:
    """
    Create a complete, valid PoR package for testing.
    
    This creates all artifacts with matching commitments:
    - PromptSpec (with strict_mode and created_at=None)
    - EvidenceBundle
    - ReasoningTrace (with evaluation_variables)
    - DeterministicVerdict
    - ToolPlan
    - PoRBundle (with correct hashes computed from artifacts)
    
    Args:
        market_id: Market identifier for all components.
        strict_mode: Whether strict mode is enabled.
        outcome: Verdict outcome (YES, NO, INVALID).
        confidence: Verdict confidence.
    
    Returns:
        Tuple of (bundle, prompt_spec, evidence, trace, verdict, tool_plan).
        All components are valid and bundle commitments match artifacts.
    """
    prompt_spec = make_prompt_spec(market_id=market_id, strict_mode=strict_mode)
    evidence = make_evidence_bundle()
    trace = make_reasoning_trace()
    verdict = make_verdict(market_id=market_id, outcome=outcome, confidence=confidence)
    tool_plan = make_tool_plan()
    
    # Build the bundle with correct commitments
    bundle = build_por_bundle(
        prompt_spec=prompt_spec,
        evidence=evidence,
        trace=trace,
        verdict=verdict,
        include_por_root=True,
    )
    
    return bundle, prompt_spec, evidence, trace, verdict, tool_plan


# =============================================================================
# Tampered Bundle Factory
# =============================================================================

def make_tampered_bundle(
    original_bundle: PoRBundle,
    tamper_field: str,
    tamper_value: Optional[str] = None,
) -> PoRBundle:
    """
    Create a tampered copy of a PoRBundle.
    
    Useful for testing mismatch detection and challenge generation.
    
    Args:
        original_bundle: The original valid bundle.
        tamper_field: Field to tamper. One of:
            - 'prompt_spec_hash'
            - 'evidence_root'
            - 'reasoning_root'
            - 'verdict_hash'
            - 'por_root'
        tamper_value: The tampered value to set.
            Default: "0x" + "ff" * 32 (obviously invalid hash).
    
    Returns:
        A new PoRBundle with the tampered field.
    
    Example:
        >>> bundle, *_ = make_valid_por_package()
        >>> bad = make_tampered_bundle(bundle, "evidence_root")
        >>> sentinel.verify(bad)  # Should fail with evidence_leaf challenge
    """
    if tamper_value is None:
        tamper_value = "0x" + "ff" * 32
    
    data = original_bundle.model_dump()
    
    if tamper_field not in data:
        raise ValueError(
            f"Unknown tamper_field: {tamper_field}. "
            f"Must be one of: prompt_spec_hash, evidence_root, "
            f"reasoning_root, verdict_hash, por_root"
        )
    
    data[tamper_field] = tamper_value
    return PoRBundle(**data)


# =============================================================================
# Specialized Package Factories
# =============================================================================

def make_invalid_verdict_package(
    market_id: str = "test_market_001",
) -> tuple[PoRBundle, Any, Any, Any, DeterministicVerdict, ToolPlan]:
    """
    Create a PoR package with INVALID outcome.
    
    Used for testing invalid verdict handling.
    """
    return make_valid_por_package(
        market_id=market_id,
        outcome="INVALID",
        confidence=0.0,
    )


def make_low_confidence_package(
    market_id: str = "test_market_001",
    confidence: float = 0.51,
) -> tuple[PoRBundle, Any, Any, Any, DeterministicVerdict, ToolPlan]:
    """
    Create a PoR package with low confidence.
    
    Used for testing confidence threshold handling.
    """
    return make_valid_por_package(
        market_id=market_id,
        confidence=confidence,
    )


def make_multi_evidence_package(
    market_id: str = "test_market_001",
    num_items: int = 3,
) -> tuple[PoRBundle, Any, Any, Any, DeterministicVerdict, ToolPlan]:
    """
    Create a PoR package with multiple evidence items.
    
    Used for testing Merkle tree handling with multiple leaves.
    """
    from tests.fixtures.common import make_evidence_item, make_evidence_bundle
    
    items = [
        make_evidence_item(
            evidence_id=f"ev_{i:03d}",
            content={"price": 105000.0 + i * 100, "source_index": i},
        )
        for i in range(num_items)
    ]
    
    evidence = make_evidence_bundle(items=items)
    prompt_spec = make_prompt_spec(market_id=market_id)
    trace = make_reasoning_trace(
        evidence_refs=[f"ev_{i:03d}" for i in range(num_items)],
    )
    verdict = make_verdict(market_id=market_id)
    tool_plan = make_tool_plan()
    
    bundle = build_por_bundle(
        prompt_spec=prompt_spec,
        evidence=evidence,
        trace=trace,
        verdict=verdict,
        include_por_root=True,
    )
    
    return bundle, prompt_spec, evidence, trace, verdict, tool_plan