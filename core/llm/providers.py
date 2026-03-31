"""
LLM Provider Implementations

Provider-specific adapters for:
- OpenAI (GPT-4, GPT-3.5, etc.)
- Anthropic (Claude)
- Google (Gemini)
- Mock (for testing)

Provider classes are intentionally *dumb*: they accept ``api_key``,
``base_url``, and ``model`` directly and never read environment variables.
Credential resolution lives in ``core.llm.credentials`` — the single
source of truth.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from .client import LLMResponse
from .credentials import (
    PROVIDER_DEFAULT_MODELS,
    PROVIDER_ENV_KEYS,          # re-exported for backwards compat
    get_configured_providers,   # re-exported for backwards compat
    resolve_credentials,
)
from .determinism import DecodingPolicy, policy_to_provider_args


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

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
        ...


# ── OpenAI ───────────────────────────────────────────────────────────


class OpenAIProvider(LLMProvider):
    """OpenAI API provider.  Requires: ``openai`` package."""

    def __init__(
        self,
        model: str = "gpt-4o",
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        proxy: Optional[str] = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
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
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("openai package required: pip install openai")

            kwargs: dict[str, Any] = {"api_key": self._api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            if self._proxy:
                try:
                    import httpx
                    kwargs["http_client"] = httpx.Client(proxy=self._proxy)
                except ImportError:
                    pass
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
            raw_response=response.model_dump() if hasattr(response, "model_dump") else None,
        )


# ── Anthropic ────────────────────────────────────────────────────────


class AnthropicProvider(LLMProvider):
    """Anthropic API provider.  Requires: ``anthropic`` package."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        proxy: Optional[str] = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._proxy = proxy
        self._client = None

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def model(self) -> str:
        return self._model

    def _get_client(self):
        if self._client is None:
            try:
                from anthropic import Anthropic
            except ImportError:
                raise ImportError("anthropic package required: pip install anthropic")

            kwargs: dict[str, Any] = {"api_key": self._api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            if self._proxy:
                try:
                    import httpx
                    kwargs["http_client"] = httpx.Client(proxy=self._proxy)
                except ImportError:
                    pass
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
            if hasattr(block, "text"):
                content += block.text

        return LLMResponse(
            content=content,
            model=response.model,
            provider=self.name,
            input_tokens=response.usage.input_tokens if response.usage else 0,
            output_tokens=response.usage.output_tokens if response.usage else 0,
            finish_reason=response.stop_reason or "stop",
            raw_response=response.model_dump() if hasattr(response, "model_dump") else None,
        )


# ── Google / Gemini ──────────────────────────────────────────────────


class GoogleProvider(LLMProvider):
    """Google Gemini API provider.

    When ``base_url`` is supplied (unified gateway), uses the OpenAI-
    compatible client.  Otherwise uses the native ``google-genai`` SDK.
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        proxy: Optional[str] = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._proxy = proxy
        self._client = None
        self._use_openai_compat = base_url is not None

    @property
    def name(self) -> str:
        return "google"

    @property
    def model(self) -> str:
        return self._model

    def _get_client(self):
        if self._client is None:
            if self._use_openai_compat:
                try:
                    from openai import OpenAI
                except ImportError:
                    raise ImportError("openai package required: pip install openai")
                kwargs: dict[str, Any] = {
                    "api_key": self._api_key,
                    "base_url": self._base_url,
                }
                if self._proxy:
                    try:
                        import httpx
                        kwargs["http_client"] = httpx.Client(proxy=self._proxy)
                    except ImportError:
                        pass
                self._client = OpenAI(**kwargs)
            else:
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
        if self._use_openai_compat:
            return self._chat_openai_compat(messages, policy=policy, json_schema=json_schema)
        return self._chat_native(messages, policy=policy, json_schema=json_schema)

    # -- OpenAI-compatible path (unified gateway) -------------------------

    def _chat_openai_compat(
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
            raw_response=response.model_dump() if hasattr(response, "model_dump") else None,
        )

    # -- Native google-genai path -----------------------------------------

    def _chat_native(
        self,
        messages: list[dict[str, Any]],
        *,
        policy: DecodingPolicy,
        json_schema: Optional[dict[str, Any]] = None,
    ) -> LLMResponse:
        from google.genai import types

        client = self._get_client()

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

        text = ""
        for candidate in getattr(response, "candidates", []):
            content = getattr(candidate, "content", None)
            if content:
                for part in getattr(content, "parts", []):
                    t = getattr(part, "text", None)
                    if t:
                        text += t

        finish_reason = "stop"
        for candidate in getattr(response, "candidates", []):
            fr = getattr(candidate, "finish_reason", None)
            if fr is not None:
                finish_reason = str(fr).rsplit(".", 1)[-1].lower()
                break

        usage = getattr(response, "usage_metadata", None)
        input_tokens = getattr(usage, "prompt_token_count", 0) or 0
        output_tokens = getattr(usage, "candidates_token_count", 0) or 0

        if finish_reason == "max_tokens" and (json_schema or policy.schema_lock):
            raise ValueError(
                f"Gemini response truncated at {output_tokens} output tokens "
                f"(limit {policy.max_tokens}). Increase max_tokens for this step."
            )

        return LLMResponse(
            content=text,
            model=self._model,
            provider=self.name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            finish_reason=finish_reason,
        )


# ── Grok ─────────────────────────────────────────────────────────────


class GrokProvider(OpenAIProvider):
    """Grok (xAI) API provider.  OpenAI-compatible endpoint."""

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
            api_key=api_key,
            base_url=base_url or "https://api.x.ai/v1",
            proxy=proxy,
        )

    @property
    def name(self) -> str:
        return "grok"


# ── Mock ─────────────────────────────────────────────────────────────


class MockProvider(LLMProvider):
    """Mock provider for testing."""

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
        return self._calls

    def set_response(self, response: str) -> None:
        self._responses = [response]
        self._call_count = 0

    def set_responses(self, responses: list[str]) -> None:
        self._responses = responses
        self._call_count = 0

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        policy: DecodingPolicy,
        json_schema: Optional[dict[str, Any]] = None,
    ) -> LLMResponse:
        self._calls.append({"messages": messages, "policy": policy, "json_schema": json_schema})
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


# ── Factory ──────────────────────────────────────────────────────────


def create_provider(
    provider_name: str,
    model: Optional[str] = None,
    proxy: Optional[str] = None,
    **kwargs: Any,
) -> LLMProvider:
    """Create an LLM provider.

    If ``api_key`` or ``base_url`` are not supplied in *kwargs*, they
    are resolved automatically via ``credentials.resolve_credentials``.
    """
    provider_name = provider_name.lower()

    # Fill in credentials from the central module when the caller
    # didn't supply them explicitly.
    if "api_key" not in kwargs or kwargs["api_key"] is None or "base_url" not in kwargs:
        resolved_key, resolved_base = resolve_credentials(provider_name)
        kwargs.setdefault("api_key", resolved_key)
        kwargs.setdefault("base_url", resolved_base)

    if provider_name == "openai":
        return OpenAIProvider(model=model or "gpt-4o", proxy=proxy, **kwargs)
    elif provider_name == "anthropic":
        return AnthropicProvider(model=model or "claude-sonnet-4-20250514", proxy=proxy, **kwargs)
    elif provider_name == "google":
        return GoogleProvider(model=model or "gemini-2.5-flash", proxy=proxy, **kwargs)
    elif provider_name == "grok":
        return GrokProvider(model=model or "grok-4-latest", proxy=proxy, **kwargs)
    elif provider_name == "mock":
        mock_kwargs = {k: v for k, v in kwargs.items() if k not in ("api_key", "base_url")}
        return MockProvider(model=model or "mock-model", **mock_kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider_name}")
