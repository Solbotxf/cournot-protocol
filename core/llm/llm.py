from __future__ import annotations

from typing import Any, Dict, Optional

from .determinism import DecodingPolicy


class LLMClient:
    """Provider-agnostic LLM client interface."""

    def __init__(self, provider: str, model: str, default_policy: Optional[DecodingPolicy] = None):
        self.provider = provider
        self.model = model
        self.default_policy = default_policy or DecodingPolicy()

    def generate(self, prompt: str, *, policy: Optional[DecodingPolicy] = None, json_schema: Optional[Dict[str, Any]] = None) -> str:
        """Generate text from a prompt.
        
        - policy: decoding determinism controls
        - json_schema: optional schema lock constraints
        """
        raise NotImplementedError
