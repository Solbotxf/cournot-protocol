"""
Module 09A - Market Resolver

Purpose: Translate DeterministicVerdict into a market resolution record.

This is a thin layer that produces the final resolution output
without any persistence. Persistence will be added in Module 09C.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.schemas.verdict import DeterministicVerdict, Outcome


@dataclass(frozen=True)
class MarketResolution:
    """
    Market resolution record.
    
    This is the final output that describes how a market was resolved.
    """
    
    market_id: str
    outcome: Outcome
    confidence: float
    resolution_rule_id: str
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "market_id": self.market_id,
            "outcome": self.outcome,
            "confidence": self.confidence,
            "resolution_rule_id": self.resolution_rule_id,
        }


def resolve_market(verdict: DeterministicVerdict) -> dict[str, Any]:
    """
    Translate a DeterministicVerdict into a market resolution record.
    
    Args:
        verdict: The verdict from the Judge
        
    Returns:
        Dictionary containing:
        - market_id: The market identifier
        - outcome: YES/NO/INVALID
        - confidence: Confidence score [0,1]
        - resolution_rule_id: The rule that decided the verdict
    """
    return {
        "market_id": verdict.market_id,
        "outcome": verdict.outcome,
        "confidence": verdict.confidence,
        "resolution_rule_id": verdict.resolution_rule_id,
    }


def create_resolution(verdict: DeterministicVerdict) -> MarketResolution:
    """
    Create a MarketResolution dataclass from a verdict.
    
    Args:
        verdict: The verdict from the Judge
        
    Returns:
        MarketResolution record
    """
    return MarketResolution(
        market_id=verdict.market_id,
        outcome=verdict.outcome,
        confidence=verdict.confidence,
        resolution_rule_id=verdict.resolution_rule_id,
    )


def is_definitive_resolution(verdict: DeterministicVerdict) -> bool:
    """
    Check if the resolution is definitive (YES or NO, not INVALID).
    
    Args:
        verdict: The verdict to check
        
    Returns:
        True if outcome is YES or NO
    """
    return verdict.outcome in ("YES", "NO")


def resolution_summary(verdict: DeterministicVerdict) -> str:
    """
    Generate a human-readable summary of the resolution.
    
    Args:
        verdict: The verdict to summarize
        
    Returns:
        Summary string
    """
    if verdict.outcome == "INVALID":
        return (
            f"Market {verdict.market_id} resolved as INVALID "
            f"(rule: {verdict.resolution_rule_id}, confidence: {verdict.confidence:.2f})"
        )
    else:
        return (
            f"Market {verdict.market_id} resolved as {verdict.outcome} "
            f"with {verdict.confidence:.1%} confidence "
            f"(rule: {verdict.resolution_rule_id})"
        )
