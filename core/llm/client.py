"""
LLM Client

Provider-agnostic LLM client interface with receipt recording.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

from .determinism import DecodingPolicy

if TYPE_CHECKING:
    from core.receipts import ReceiptRecorder

@dataclass
class LLMResponse:
    """
    Response from an LLM call.
    """
    content: str
    model: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    finish_reason: str = "stop"
    raw_response: Optional[dict[str, Any]] = None
    receipt_id: Optional[str] = None
    
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens
    
    def as_json(self) -> Optional[dict[str, Any]]:
        """
        Try to parse content as JSON.
        
        Returns None if parsing fails.
        """
        try:
            # Handle common cases of JSON wrapped in markdown
            text = self.content.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            return json.loads(text.strip())
        except json.JSONDecodeError:
            return None


class LLMClient:
    """
    Provider-agnostic LLM client.
    
    Usage:
        from core.llm import LLMClient, create_provider
        
        provider = create_provider("openai", api_key="...")
        client = LLMClient(provider)
        
        response = client.chat([
            {"role": "user", "content": "Hello!"}
        ])
        print(response.content)
    """
    
    def __init__(
        self,
        provider: "LLMProvider",
        *,
        default_policy: Optional[DecodingPolicy] = None,
        recorder: Optional["ReceiptRecorder"] = None,
    ) -> None:
        """
        Initialize LLM client.
        
        Args:
            provider: The LLM provider to use
            default_policy: Default decoding policy for calls
            recorder: Receipt recorder for audit logging
        """
        self.provider = provider
        self.default_policy = default_policy or DecodingPolicy()
        self.recorder = recorder
    
    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        policy: Optional[DecodingPolicy] = None,
        json_schema: Optional[dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """
        Send a chat completion request.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            policy: Decoding policy (temperature, etc.)
            json_schema: Optional JSON schema for structured output
            system_prompt: Optional system prompt to prepend
        
        Returns:
            LLMResponse with content and metadata
        """
        effective_policy = policy or self.default_policy
        
        # Prepend system prompt if provided
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages
        
        # Start receipt if recorder available
        receipt = None
        if self.recorder:
            receipt = self.recorder.start_llm_receipt(
                provider=self.provider.name,
                model=self.provider.model,
                messages=messages,
                temperature=effective_policy.temperature,
                max_tokens=effective_policy.max_tokens,
                json_schema=json_schema,
            )
        
        try:
            response = self.provider.chat(
                messages=messages,
                policy=effective_policy,
                json_schema=json_schema,
            )
            
            # Complete receipt
            if receipt and self.recorder:
                self.recorder.complete(
                    receipt,
                    response={
                        "content": response.content,
                        "finish_reason": response.finish_reason,
                    },
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    finish_reason=response.finish_reason,
                )
                response.receipt_id = receipt.receipt_id
            
            return response
            
        except Exception as e:
            if receipt and self.recorder:
                self.recorder.complete(receipt, error=str(e))
            raise
    
    def generate(
        self,
        prompt: str,
        *,
        policy: Optional[DecodingPolicy] = None,
        json_schema: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        Simple text generation from a prompt.
        
        This is a convenience wrapper around chat() with a single user message.
        
        Args:
            prompt: The prompt text
            policy: Decoding policy
            json_schema: Optional JSON schema for structured output
        
        Returns:
            Generated text content
        """
        response = self.chat(
            messages=[{"role": "user", "content": prompt}],
            policy=policy,
            json_schema=json_schema,
        )
        return response.content
    
    def generate_json(
        self,
        prompt: str,
        *,
        policy: Optional[DecodingPolicy] = None,
        schema: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Generate JSON output from a prompt.
        
        Args:
            prompt: The prompt text
            policy: Decoding policy
            schema: Optional JSON schema for validation
        
        Returns:
            Parsed JSON dict
        
        Raises:
            ValueError: If response cannot be parsed as JSON
        """
        response = self.chat(
            messages=[{"role": "user", "content": prompt}],
            policy=policy,
            json_schema=schema,
        )
        
        result = response.as_json()
        if result is None:
            raise ValueError(f"Failed to parse LLM response as JSON: {response.content[:200]}")
        
        return result


# Import here to avoid circular imports
from .providers import LLMProvider
