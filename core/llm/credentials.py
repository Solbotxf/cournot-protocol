"""
LLM Credential Resolution — single source of truth.

All API key and base-URL resolution goes through this module.
Nothing else in the codebase should read LLM-related env vars directly.

Two modes of operation
----------------------
**Unified mode** (``COURNOT_UNIFIED_API_BASE`` is set):
    Every provider is routed through a single gateway.  Only
    ``COURNOT_UNIFIED_*`` env vars are consulted — per-provider vars
    such as ``OPENAI_API_KEY`` are ignored.

    Env vars:
        COURNOT_UNIFIED_API_BASE        — gateway base URL
        COURNOT_UNIFIED_API_KEY         — default key for all providers
        COURNOT_UNIFIED_<PROV>_API_KEY  — optional per-provider override
                                          (e.g. COURNOT_UNIFIED_GOOGLE_API_KEY)
        GOOGLE_NATIVE_API_KEY           — real Google key for collectors
                                          that need native google-genai SDK
                                          (GoogleSearch grounding, UrlContext)

**Per-provider mode** (``COURNOT_UNIFIED_API_BASE`` is NOT set):
    Each provider reads its own env var.

    Env vars:
        OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, XAI_API_KEY
"""

from __future__ import annotations

import os
from typing import Any

# ── Canonical provider metadata ──────────────────────────────────────

PROVIDER_ENV_KEYS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "grok": "XAI_API_KEY",
}

PROVIDER_DEFAULT_MODELS: dict[str, str] = {
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-20250514",
    "google": "gemini-2.5-flash",
    "grok": "grok-4-fast",
}

_PROVIDER_DEFAULT_BASE_URLS: dict[str, str] = {
    "grok": "https://api.x.ai/v1",
}

# ── Public helpers ───────────────────────────────────────────────────


def is_unified_mode() -> bool:
    """Return True when a unified gateway base URL is configured."""
    return bool(os.getenv("COURNOT_UNIFIED_API_BASE"))


def _normalise_base(raw: str) -> str:
    base = raw.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    return base


def resolve_credentials(provider: str) -> tuple[str | None, str | None]:
    """Return ``(api_key, base_url)`` for *provider*.

    This is the ONE function that decides which key and URL to use.
    """
    provider = provider.lower()
    unified_base = os.getenv("COURNOT_UNIFIED_API_BASE")

    if unified_base:
        # ── Unified mode ─────────────────────────────────────────
        base_url = _normalise_base(unified_base)
        # Provider-specific unified key, then default unified key.
        api_key = (
            os.getenv(f"COURNOT_UNIFIED_{provider.upper()}_API_KEY")
            or os.getenv("COURNOT_UNIFIED_API_KEY")
        )
        return api_key, base_url

    # ── Per-provider mode ────────────────────────────────────────
    env_var = PROVIDER_ENV_KEYS.get(provider, f"{provider.upper()}_API_KEY")
    api_key = os.getenv(env_var)
    base_url = _PROVIDER_DEFAULT_BASE_URLS.get(provider)
    return api_key, base_url


def get_native_google_key() -> str | None:
    """Return the API key for the **native** google-genai SDK.

    Collectors that use ``GoogleSearch`` grounding or ``UrlContext`` MUST
    call this instead of ``resolve_credentials("google")``, because
    the unified gateway cannot replicate those Gemini-specific features.

    In unified mode  → ``GOOGLE_NATIVE_API_KEY``
    In per-provider  → ``GOOGLE_API_KEY``
    """
    if is_unified_mode():
        return os.getenv("GOOGLE_NATIVE_API_KEY")
    return os.getenv("GOOGLE_API_KEY")


def get_configured_providers() -> list[dict[str, str]]:
    """Return providers that have usable credentials.

    Used by ``GET /capabilities`` to advertise available providers.
    """
    result: list[dict[str, str]] = []
    for name in PROVIDER_ENV_KEYS:
        api_key, _ = resolve_credentials(name)
        if api_key:
            result.append({
                "provider": name,
                "default_model": PROVIDER_DEFAULT_MODELS.get(name, ""),
            })
    return result
