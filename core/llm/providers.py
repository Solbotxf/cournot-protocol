"""
LLM Provider Implementations

Provider-specific adapters for:
- OpenAI (GPT-4, GPT-3.5, etc.)
- Anthropic (Claude)
- Google (Gemini)
- Mock (for testing)
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Optional

from .client import LLMResponse
from .determinism import DecodingPolicy, policy_to_provider_args


# Canonical mapping from provider name to environment variable for API key.
PROVIDER_ENV_KEYS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "grok": "XAI_API_KEY",
}

# Default models per provider (mirrors factory defaults in create_provider).
PROVIDER_DEFAULT_MODELS: dict[str, str] = {
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-20250514",
    "google": "gemini-2.5-flash",
    "grok": "grok-4-fast",
}


def get_configured_providers() -> list[dict[str, str]]:
    """Return providers that have API keys set in the environment."""
    return [
        {"provider": name, "default_model": PROVIDER_DEFAULT_MODELS.get(name, "")}
        for name, env_var in PROVIDER_ENV_KEYS.items()
        if os.getenv(env_var)
    ]


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (openai, anthropic, google, etc.)."""
        ...
    
    @property
    @abstractmethod
    def model(self) -> str:
        """Model identifier."""
        ...
    
    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        policy: DecodingPolicy,
        json_schema: Optional[dict[str, Any]] = None,
    ) -> LLMResponse:
        """
        Send a chat completion request.
        
        Args:
            messages: List of message dicts
            policy: Decoding policy
            json_schema: Optional JSON schema for structured output
        
        Returns:
            LLMResponse with content and metadata
        """
        ...


class OpenAIProvider(LLMProvider):
    """
    OpenAI API provider.

    Requires: openai package
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        proxy: Optional[str] = None,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._base_url = base_url
        self._proxy = proxy
        self._client = None

    @property
    def name(self) -> str:
        return "openai"

    @property
    def model(self) -> str:
        return self._model

    def _get_client(self):
        """Lazy-load OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("openai package required: pip install openai")

            kwargs: dict[str, Any] = {
                "api_key": self._api_key,
            }
            if self._base_url:
                kwargs["base_url"] = self._base_url

            # Configure proxy via httpx client
            if self._proxy:
                try:
                    import httpx
                    kwargs["http_client"] = httpx.Client(proxy=self._proxy)
                except ImportError:
                    pass  # httpx not available, proxy won't work

            self._client = OpenAI(**kwargs)
        return self._client
    
    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        policy: DecodingPolicy,
        json_schema: Optional[dict[str, Any]] = None,
    ) -> LLMResponse:
        client = self._get_client()
        
        kwargs = policy_to_provider_args(policy, "openai")
        kwargs["model"] = self._model
        kwargs["messages"] = messages
        
        # Handle JSON mode
        if json_schema or policy.schema_lock:
            kwargs["response_format"] = {"type": "json_object"}
        
        response = client.chat.completions.create(**kwargs)
        
        choice = response.choices[0]
        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            provider=self.name,
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
            finish_reason=choice.finish_reason or "stop",
            raw_response=response.model_dump() if hasattr(response, 'model_dump') else None,
        )


class AnthropicProvider(LLMProvider):
    """
    Anthropic API provider.

    Requires: anthropic package
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        *,
        api_key: Optional[str] = None,
        proxy: Optional[str] = None,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._proxy = proxy
        self._client = None

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def model(self) -> str:
        return self._model

    def _get_client(self):
        """Lazy-load Anthropic client."""
        if self._client is None:
            try:
                from anthropic import Anthropic
            except ImportError:
                raise ImportError("anthropic package required: pip install anthropic")

            kwargs: dict[str, Any] = {"api_key": self._api_key}

            # Configure proxy via httpx client
            if self._proxy:
                try:
                    import httpx
                    kwargs["http_client"] = httpx.Client(proxy=self._proxy)
                except ImportError:
                    pass  # httpx not available, proxy won't work

            self._client = Anthropic(**kwargs)
        return self._client
    
    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        policy: DecodingPolicy,
        json_schema: Optional[dict[str, Any]] = None,
    ) -> LLMResponse:
        client = self._get_client()
        
        # Extract system message if present
        system_prompt = None
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                chat_messages.append(msg)
        
        kwargs = policy_to_provider_args(policy, "anthropic")
        kwargs["model"] = self._model
        kwargs["messages"] = chat_messages
        if system_prompt:
            kwargs["system"] = system_prompt
        
        response = client.messages.create(**kwargs)
        
        content = ""
        for block in response.content:
            if hasattr(block, 'text'):
                content += block.text
        
        return LLMResponse(
            content=content,
            model=response.model,
            provider=self.name,
            input_tokens=response.usage.input_tokens if response.usage else 0,
            output_tokens=response.usage.output_tokens if response.usage else 0,
            finish_reason=response.stop_reason or "stop",
            raw_response=response.model_dump() if hasattr(response, 'model_dump') else None,
        )


class GoogleProvider(LLMProvider):
    """
    Google Gemini API provider.

    Requires: google-genai package (pip install google-genai)
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        *,
        api_key: Optional[str] = None,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self._client = None

    @property
    def name(self) -> str:
        return "google"

    @property
    def model(self) -> str:
        return self._model

    def _get_client(self):
        """Lazy-load Google genai client."""
        if self._client is None:
            try:
                from google import genai
            except ImportError:
                raise ImportError(
                    "google-genai package required: pip install google-genai"
                )
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        policy: DecodingPolicy,
        json_schema: Optional[dict[str, Any]] = None,
    ) -> LLMResponse:
        from google.genai import types

        client = self._get_client()

        # Separate system instruction from conversation messages
        system_parts = []
        gemini_contents = []
        for msg in messages:
            if msg["role"] == "system":
                system_parts.append(msg["content"])
            else:
                role = "user" if msg["role"] == "user" else "model"
                gemini_contents.append(
                    types.Content(role=role, parts=[types.Part(text=msg["content"])])
                )

        config_kwargs: dict[str, Any] = {
            "temperature": policy.temperature,
            "max_output_tokens": policy.max_tokens,
            "top_p": policy.top_p,
        }
        if system_parts:
            config_kwargs["system_instruction"] = "\n".join(system_parts)
        if json_schema or policy.schema_lock:
            config_kwargs["response_mime_type"] = "application/json"

        response = client.models.generate_content(
            model=self._model,
            contents=gemini_contents,
            config=types.GenerateContentConfig(**config_kwargs),
        )

        # Extract text from response
        text = ""
        for candidate in getattr(response, "candidates", []):
            content = getattr(candidate, "content", None)
            if content:
                for part in getattr(content, "parts", []):
                    t = getattr(part, "text", None)
                    if t:
                        text += t

        # Extract token counts from usage metadata
        usage = getattr(response, "usage_metadata", None)
        input_tokens = getattr(usage, "prompt_token_count", 0) or 0
        output_tokens = getattr(usage, "candidates_token_count", 0) or 0

        return LLMResponse(
            content=text,
            model=self._model,
            provider=self.name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            finish_reason="stop",
        )


class GrokProvider(OpenAIProvider):
    """
    Grok (xAI) API provider.

    Uses the OpenAI-compatible endpoint at https://api.x.ai/v1.
    Requires: openai package
    """

    def __init__(
        self,
        model: str = "grok-4-latest",
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        proxy: Optional[str] = None,
    ) -> None:
        super().__init__(
            model=model,
            api_key=api_key or os.getenv("XAI_API_KEY"),
            base_url=base_url or "https://api.x.ai/v1",
            proxy=proxy,
        )

    @property
    def name(self) -> str:
        return "grok"


class MockProvider(LLMProvider):
    """
    Mock provider for testing.
    
    Can be configured with preset responses or a response function.
    """
    
    def __init__(
        self,
        model: str = "mock-model",
        *,
        responses: Optional[list[str]] = None,
        response_fn: Optional[callable] = None,
    ) -> None:
        self._model = model
        self._responses = responses or []
        self._response_fn = response_fn
        self._call_count = 0
        self._calls: list[dict[str, Any]] = []
    
    @property
    def name(self) -> str:
        return "mock"
    
    @property
    def model(self) -> str:
        return self._model
    
    @property
    def calls(self) -> list[dict[str, Any]]:
        """Get all recorded calls."""
        return self._calls
    
    def set_response(self, response: str) -> None:
        """Set a single response to return."""
        self._responses = [response]
        self._call_count = 0
    
    def set_responses(self, responses: list[str]) -> None:
        """Set multiple responses to return in sequence."""
        self._responses = responses
        self._call_count = 0
    
    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        policy: DecodingPolicy,
        json_schema: Optional[dict[str, Any]] = None,
    ) -> LLMResponse:
        # Record the call
        self._calls.append({
            "messages": messages,
            "policy": policy,
            "json_schema": json_schema,
        })
        
        # Get response
        if self._response_fn:
            content = self._response_fn(messages, policy, json_schema)
        elif self._responses:
            content = self._responses[self._call_count % len(self._responses)]
        else:
            content = '{"result": "mock response"}'
        
        self._call_count += 1
        
        return LLMResponse(
            content=content,
            model=self._model,
            provider=self.name,
            input_tokens=100,
            output_tokens=50,
            finish_reason="stop",
        )


def create_provider(
    provider_name: str,
    model: Optional[str] = None,
    proxy: Optional[str] = None,
    **kwargs: Any,
) -> LLMProvider:
    """
    Factory function to create an LLM provider.

    Args:
        provider_name: Provider name (openai, anthropic, google, grok, mock)
        model: Model identifier (optional, uses provider default)
        proxy: Proxy URL (e.g., "http://user:pass@host:port")
        **kwargs: Provider-specific arguments

    Returns:
        LLMProvider instance
    """
    provider_name = provider_name.lower()

    if provider_name == "openai":
        return OpenAIProvider(model=model or "gpt-4o", proxy=proxy, **kwargs)
    elif provider_name == "anthropic":
        return AnthropicProvider(model=model or "claude-sonnet-4-20250514", proxy=proxy, **kwargs)
    elif provider_name == "google":
        google_kwargs = {k: v for k, v in kwargs.items() if k not in ("base_url", "proxy")}
        return GoogleProvider(model=model or "gemini-2.5-flash", **google_kwargs)
    elif provider_name == "grok":
        return GrokProvider(model=model or "grok-4-latest", proxy=proxy, **kwargs)
    elif provider_name == "mock":
        return MockProvider(model=model or "mock-model", **kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider_name}")
