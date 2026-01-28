from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class DecodingPolicy:
    """Controls determinism: temperature, top_p, seed, and schema-locked decoding."""
    temperature: float = 0.0
    top_p: float = 1.0
    seed: Optional[int] = 42
    max_tokens: int = 2048
    schema_lock: bool = True


def policy_to_provider_args(policy: DecodingPolicy) -> Dict[str, Any]:
    """Convert to provider-specific args (OpenAI, Anthropic, etc.)."""
    return {
        "temperature": policy.temperature,
        "top_p": policy.top_p,
        "seed": policy.seed,
        "max_tokens": policy.max_tokens,
    }
