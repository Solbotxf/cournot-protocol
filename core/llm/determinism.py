"""
LLM Determinism Controls

Controls for ensuring deterministic LLM outputs:
- temperature=0 for reproducibility
- fixed seeds when supported
- schema-locked JSON decoding
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class DecodingPolicy:
    """
    Controls determinism for LLM calls.
    
    temperature=0 + fixed seed = most deterministic behavior
    schema_lock=True ensures structured JSON outputs
    """
    temperature: float = 0.0
    top_p: float = 1.0
    seed: Optional[int] = 42
    max_tokens: int = 2048
    schema_lock: bool = True
    stop_sequences: tuple[str, ...] = field(default_factory=tuple)
    
    def with_temperature(self, temp: float) -> "DecodingPolicy":
        """Return a new policy with modified temperature."""
        return DecodingPolicy(
            temperature=temp,
            top_p=self.top_p,
            seed=self.seed,
            max_tokens=self.max_tokens,
            schema_lock=self.schema_lock,
            stop_sequences=self.stop_sequences,
        )
    
    def with_max_tokens(self, tokens: int) -> "DecodingPolicy":
        """Return a new policy with modified max_tokens."""
        return DecodingPolicy(
            temperature=self.temperature,
            top_p=self.top_p,
            seed=self.seed,
            max_tokens=tokens,
            schema_lock=self.schema_lock,
            stop_sequences=self.stop_sequences,
        )


def policy_to_provider_args(policy: DecodingPolicy, provider: str = "openai") -> Dict[str, Any]:
    """
    Convert DecodingPolicy to provider-specific API arguments.
    
    Args:
        policy: The decoding policy
        provider: Provider name (openai, anthropic, google)
    
    Returns:
        Dict of API arguments
    """
    if provider == "openai":
        args = {
            "temperature": policy.temperature,
            "max_tokens": policy.max_tokens,
            "top_p": policy.top_p,
        }
        if policy.seed is not None:
            args["seed"] = policy.seed
        if policy.stop_sequences:
            args["stop"] = list(policy.stop_sequences)
        return args
    
    elif provider == "anthropic":
        args = {
            "temperature": policy.temperature,
            "max_tokens": policy.max_tokens,
            "top_p": policy.top_p,
        }
        if policy.stop_sequences:
            args["stop_sequences"] = list(policy.stop_sequences)
        return args
    
    elif provider == "google":
        return {
            "temperature": policy.temperature,
            "max_output_tokens": policy.max_tokens,
            "top_p": policy.top_p,
        }
    
    else:
        # Default to OpenAI-style args
        return {
            "temperature": policy.temperature,
            "max_tokens": policy.max_tokens,
            "top_p": policy.top_p,
        }


# Preset policies for common use cases
STRICT_POLICY = DecodingPolicy(
    temperature=0.0,
    seed=42,
    schema_lock=True,
)

CREATIVE_POLICY = DecodingPolicy(
    temperature=0.7,
    seed=None,
    schema_lock=False,
)
