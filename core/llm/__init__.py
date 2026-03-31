"""
LLM Client Module

Provider-agnostic LLM client with support for:
- OpenAI (GPT-4, etc.)
- Anthropic (Claude)
- Google (Gemini)
- Local models (via OpenAI-compatible APIs)

All calls are recorded via ReceiptRecorder for auditability.
"""

from typing import Optional, Any

from .client import LLMClient, LLMResponse
from .credentials import (
    PROVIDER_ENV_KEYS,
    PROVIDER_DEFAULT_MODELS,
    get_configured_providers,
    resolve_credentials,
    get_native_google_key,
)
from .determinism import DecodingPolicy, policy_to_provider_args
from .providers import (
    LLMProvider,
    OpenAIProvider,
    AnthropicProvider,
    GoogleProvider,
    GrokProvider,
    MockProvider,
    create_provider,
)

def create_llm_client(
    provider: str,
    *,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    endpoint: Optional[str] = None,
    proxy: Optional[str] = None,
    default_policy: Optional[DecodingPolicy] = None,
    recorder: Optional[Any] = None,
    **kwargs: Any,
) -> LLMClient:
    """
    Convenience function to create an LLMClient.

    Args:
        provider: Provider name (openai, anthropic, google, grok, mock)
        api_key: API key for the provider
        model: Model identifier (optional, uses provider default)
        endpoint: Custom API endpoint (for OpenAI-compatible APIs)
        proxy: Proxy URL (e.g., "http://user:pass@host:port")
        default_policy: Default decoding policy
        recorder: Receipt recorder for audit logging
        **kwargs: Additional provider-specific arguments

    Returns:
        Configured LLMClient instance
    """
    provider_kwargs = {}
    if api_key:
        provider_kwargs["api_key"] = api_key
    if endpoint:
        provider_kwargs["base_url"] = endpoint
    provider_kwargs.update(kwargs)

    llm_provider = create_provider(provider, model=model, proxy=proxy, **provider_kwargs)

    return LLMClient(
        provider=llm_provider,
        default_policy=default_policy,
        recorder=recorder,
    )

__all__ = [
    "LLMClient",
    "LLMResponse",
    "DecodingPolicy",
    "policy_to_provider_args",
    "LLMProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "GoogleProvider",
    "GrokProvider",
    "MockProvider",
    "create_provider",
    "create_llm_client",
    "PROVIDER_ENV_KEYS",
    "PROVIDER_DEFAULT_MODELS",
    "get_configured_providers",
    "resolve_credentials",
    "get_native_google_key",
]
