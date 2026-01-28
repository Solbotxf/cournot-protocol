"""
Common test fixtures shared by all modules.

Provides factory functions for core Cournot data structures:
- MarketSpec
- PromptSpec
- EvidenceBundle / EvidenceItem
- ReasoningTrace / ReasoningStep

These are the foundational building blocks used by higher-level fixtures.
"""

from datetime import datetime, timezone, timedelta
from typing import Any, Optional

from core.schemas.market import (
    MarketSpec,
    ResolutionWindow,
    ResolutionRules,
    ResolutionRule,
    DisputePolicy,
    SourcePolicy,
)
from core.schemas.prompts import (
    PromptSpec,
    PredictionSemantics,
    DataRequirement,
    SourceTarget,
    SelectionPolicy,
)
from core.schemas.evidence import (
    EvidenceBundle,
    EvidenceItem,
    SourceDescriptor,
    RetrievalReceipt,
    ProvenanceProof,
)
from core.por.reasoning_trace import (
    ReasoningTrace,
    ReasoningStep,
    TracePolicy,
)


# =============================================================================
# MarketSpec Factory
# =============================================================================

def make_market_spec(
    market_id: str = "test_market_001",
    question: str = "Will BTC exceed $100,000 by end of 2026?",
    event_definition: str = "BTC-USD price > 100000 at 2026-12-31T23:59:59Z",
    resolution_deadline_days: int = 365,
    allowed_sources: Optional[list[SourcePolicy]] = None,
    dispute_window_seconds: int = 86400,
) -> MarketSpec:
    """
    Create a MarketSpec for testing.
    
    Args:
        market_id: Unique market identifier.
        question: Human-readable question.
        event_definition: Machine-parseable event definition.
        resolution_deadline_days: Days from now until resolution deadline.
        allowed_sources: List of allowed source policies.
        dispute_window_seconds: Dispute window duration.
    
    Returns:
        A valid MarketSpec instance.
    """
    now = datetime.now(timezone.utc)
    
    if allowed_sources is None:
        allowed_sources = [
            SourcePolicy(source_id="http", kind="api", allow=True),
            SourcePolicy(source_id="exchange", kind="api", allow=True),
        ]
    
    return MarketSpec(
        market_id=market_id,
        question=question,
        event_definition=event_definition,
        timezone="UTC",
        resolution_deadline=now + timedelta(days=resolution_deadline_days),
        resolution_window=ResolutionWindow(
            start=now,
            end=now + timedelta(days=resolution_deadline_days),
        ),
        resolution_rules=ResolutionRules(
            rules=[
                ResolutionRule(
                    rule_id="R_BINARY_DECISION",
                    description="If event observed, YES; else NO",
                    priority=10,
                ),
                ResolutionRule(
                    rule_id="R_VALIDITY",
                    description="If insufficient evidence, INVALID",
                    priority=20,
                ),
            ]
        ),
        allowed_sources=allowed_sources,
        min_provenance_tier=0,
        dispute_policy=DisputePolicy(
            dispute_window_seconds=dispute_window_seconds,
            allow_challenges=True,
        ),
    )


# =============================================================================
# PromptSpec Factory
# =============================================================================

def make_prompt_spec(
    market_id: str = "test_market_001",
    strict_mode: bool = True,
    output_schema_ref: str = "core.schemas.verdict.DeterministicVerdict",
    threshold: Optional[str] = "100000",
    timeframe: Optional[str] = "2026-12-31",
    confidence_policy: Optional[dict[str, Any]] = None,
    data_requirements: Optional[list[DataRequirement]] = None,
) -> PromptSpec:
    """
    Create a PromptSpec for testing.
    
    Args:
        market_id: Market identifier (used to create MarketSpec).
        strict_mode: Whether strict mode is enabled in extra.
        output_schema_ref: Reference to output schema.
        threshold: Prediction threshold value.
        timeframe: Prediction timeframe.
        confidence_policy: Optional confidence policy in extra.
        data_requirements: Custom data requirements (default creates one).
    
    Returns:
        A valid PromptSpec with created_at=None for deterministic hashing.
    """
    extra: dict[str, Any] = {}
    if strict_mode:
        extra["strict_mode"] = True
    if confidence_policy:
        extra["confidence_policy"] = confidence_policy
    else:
        extra["confidence_policy"] = {
            "min_confidence_for_yesno": 0.55,
            "default_confidence": 0.7,
        }
    
    if data_requirements is None:
        data_requirements = [
            DataRequirement(
                requirement_id="req_001",
                description="BTC price from exchange",
                source_targets=[
                    SourceTarget(
                        source_id="exchange",
                        uri="https://api.exchange.example.com/btc-usd",
                        method="GET",
                        expected_content_type="json",
                    ),
                ],
                selection_policy=SelectionPolicy(
                    strategy="single_best",
                    min_sources=1,
                    max_sources=1,
                    quorum=1,
                ),
                min_provenance_tier=0,
            ),
        ]
    
    return PromptSpec(
        market=make_market_spec(market_id),
        prediction_semantics=PredictionSemantics(
            target_entity="BTC-USD",
            predicate="price exceeds threshold",
            threshold=threshold,
            timeframe=timeframe,
        ),
        data_requirements=data_requirements,
        output_schema_ref=output_schema_ref,
        created_at=None,  # Important for deterministic hashing
        extra=extra,
    )


# =============================================================================
# EvidenceItem / EvidenceBundle Factories
# =============================================================================

def make_evidence_item(
    evidence_id: str = "ev_001",
    requirement_id: str = "req_001",
    source_id: str = "exchange",
    content: Optional[Any] = None,
    provenance_tier: int = 1,
    provenance_kind: str = "signature",
    confidence: float = 0.9,
) -> EvidenceItem:
    """
    Create an EvidenceItem for testing.
    
    Args:
        evidence_id: Unique evidence identifier.
        requirement_id: Linked requirement ID.
        source_id: Source identifier.
        content: Evidence content (default: BTC price data).
        provenance_tier: Provenance tier (0=highest, 3=lowest).
        provenance_kind: Type of provenance proof.
        confidence: Confidence score [0, 1].
    
    Returns:
        A valid EvidenceItem instance.
    """
    if content is None:
        content = {"price": 105000.0, "timestamp": "2026-12-31T23:59:00Z"}
    
    return EvidenceItem(
        evidence_id=evidence_id,
        requirement_id=requirement_id,
        source=SourceDescriptor(
            source_id=source_id,
            uri=f"https://api.{source_id}.example.com/data",
            provider=f"{source_id.title()} Provider",
        ),
        retrieval=RetrievalReceipt(
            retrieved_at=None,  # Optional for determinism
            method="http_get",
            request_fingerprint="0x1234567890abcdef",
            response_fingerprint="0xfedcba0987654321",
        ),
        provenance=ProvenanceProof(
            tier=provenance_tier,
            kind=provenance_kind,
        ),
        content_type="json",
        content=content,
        confidence=confidence,
    )


def make_evidence_bundle(
    bundle_id: str = "bundle_test_001",
    items: Optional[list[EvidenceItem]] = None,
    collector_id: str = "collector_v1",
) -> EvidenceBundle:
    """
    Create an EvidenceBundle for testing.
    
    Args:
        bundle_id: Unique bundle identifier.
        items: List of evidence items (default: one item).
        collector_id: Collector agent identifier.
    
    Returns:
        A valid EvidenceBundle instance.
    """
    if items is None:
        items = [make_evidence_item()]
    
    return EvidenceBundle(
        bundle_id=bundle_id,
        collector_id=collector_id,
        collection_time=None,  # Optional for determinism
        items=items,
        provenance_summary={"total_items": len(items)},
    )


# =============================================================================
# ReasoningStep / ReasoningTrace Factories
# =============================================================================

def make_reasoning_step(
    step_id: str,
    step_type: str = "extract",
    action: str = "Performing operation",
    evidence_ids: Optional[list[str]] = None,
    prior_step_ids: Optional[list[str]] = None,
    inputs: Optional[dict[str, Any]] = None,
    output: Optional[dict[str, Any]] = None,
) -> ReasoningStep:
    """
    Create a ReasoningStep for testing.
    
    Args:
        step_id: Unique step identifier.
        step_type: Type of step (extract, check, deduce, map, aggregate, search).
        action: Human-readable action description.
        evidence_ids: Referenced evidence IDs.
        prior_step_ids: Dependent step IDs.
        inputs: Step inputs.
        output: Step outputs.
    
    Returns:
        A valid ReasoningStep instance.
    """
    return ReasoningStep(
        step_id=step_id,
        type=step_type,
        action=action or f"Performing {step_type} operation",
        inputs=inputs or {"evidence_ids": evidence_ids or []},
        output=output or {},
        evidence_ids=evidence_ids or [],
        prior_step_ids=prior_step_ids or [],
    )


def make_reasoning_trace(
    trace_id: str = "trace_test_001",
    steps: Optional[list[ReasoningStep]] = None,
    evaluation_variables: Optional[dict[str, Any]] = None,
    evidence_refs: Optional[list[str]] = None,
) -> ReasoningTrace:
    """
    Create a ReasoningTrace for testing.
    
    Args:
        trace_id: Unique trace identifier.
        steps: List of reasoning steps (default: extract + map).
        evaluation_variables: Final evaluation variables for map step.
        evidence_refs: Referenced evidence IDs.
    
    Returns:
        A valid ReasoningTrace with evaluation_variables in final step.
    """
    if evaluation_variables is None:
        evaluation_variables = {
            "event_observed": True,
            "numeric_value": 105000.0,
            "conflict_detected": False,
            "insufficient_evidence": False,
            "source_summary": ["ev_001"],
        }
    
    if steps is None:
        steps = [
            make_reasoning_step(
                step_id="step_0001",
                step_type="extract",
                action="Extract price from evidence",
                evidence_ids=["ev_001"],
                output={"extracted_price": 105000.0},
            ),
            make_reasoning_step(
                step_id="step_0002",
                step_type="check",
                action="Check price against threshold",
                prior_step_ids=["step_0001"],
                output={"price_above_threshold": True},
            ),
            make_reasoning_step(
                step_id="step_0003",
                step_type="map",
                action="Map to evaluation variables",
                prior_step_ids=["step_0002"],
                output={"evaluation_variables": evaluation_variables},
            ),
        ]
    
    if evidence_refs is None:
        # Collect all evidence_ids from steps
        evidence_refs = []
        for step in steps:
            for eid in step.evidence_ids:
                if eid not in evidence_refs:
                    evidence_refs.append(eid)
        if not evidence_refs:
            evidence_refs = ["ev_001"]
    
    return ReasoningTrace(
        trace_id=trace_id,
        policy=TracePolicy(
            decoding_policy="strict",
            allow_external_sources=False,
            max_steps=200,
        ),
        steps=steps,
        evidence_refs=evidence_refs,
    )