"""
Module 09D - API Dependencies

Dependency injection for the API.
Provides factories for pipeline and agent context.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

from orchestrator.pipeline import Pipeline, PipelineConfig, ExecutionMode
from agents import AgentContext
from core.config.runtime import LLMConfig, RuntimeConfig
from core.llm.providers import PROVIDER_DEFAULT_MODELS, PROVIDER_ENV_KEYS

logger = logging.getLogger(__name__)


def _load_runtime_config() -> RuntimeConfig:
    """Load RuntimeConfig from config file, then overlay environment variables.

    Search order for config file:
      1. ./cournot.json
      2. ./.cournot.json
      3. ~/.config/cournot/config.json

    Environment variables ALWAYS override config file values.
    The .env file is loaded automatically by core.config.runtime on import.
    """
    search_paths = [
        Path.cwd() / "cournot.json",
        Path.cwd() / ".cournot.json",
        Path.home() / ".config" / "cournot" / "config.json",
    ]

    config: RuntimeConfig | None = None

    for path in search_paths:
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                logger.info(f"Loaded config from {path}")
                config = RuntimeConfig.from_dict(data)
                break
            except Exception as e:
                logger.warning(f"Failed to parse {path}: {e}")

    if config is None:
        # No config file found — start with defaults
        config = RuntimeConfig()

    # Always apply environment variable overrides
    return config.with_env_overrides()


def build_llm_override(
    llm_provider: str | None,
    llm_model: str | None,
    *,
    agent_name: str | None = None,
) -> LLMConfig | None:
    """Build an LLMConfig override from per-request provider/model.

    Resolution order (first non-None wins):
      1. Per-request ``llm_provider`` / ``llm_model`` from the API body.
      2. Per-agent ``llm_override`` from ``cournot.json`` → ``agents.<agent_name>.llm_override``.
      3. None — the caller falls back to the server-wide default LLM.

    API keys are ALWAYS resolved server-side from environment variables.

    Args:
        llm_provider: Provider requested in the API call (e.g. 'openai').
        llm_model: Model requested in the API call (e.g. 'gpt-4o').
        agent_name: Optional agent section name (e.g. 'collector', 'auditor')
                    used to look up ``agents.<agent_name>.llm_override`` in config.

    Returns:
        LLMConfig override, or None when no override applies.

    Raises:
        InvalidRequestError: if the resolved provider has no API key configured.
    """
    config = _load_runtime_config()

    # --- Tier 2: per-agent config fallback ---
    agent_override: LLMConfig | None = None
    if agent_name is not None:
        agent_cfg = getattr(config.agents, agent_name, None)
        if agent_cfg is not None:
            agent_override = agent_cfg.llm_override  # may be None

    # --- Tier 1: per-request values take priority ---
    if llm_provider is None and llm_model is None:
        # No request-level override — use agent config if available
        if agent_override is None:
            return None
        # Use agent config, resolving its API key
        llm_provider = agent_override.provider
        llm_model = agent_override.model
    else:
        # Request provided at least one value; fill gaps from agent config,
        # then from the server-wide default.
        if llm_provider is None:
            llm_provider = (
                agent_override.provider if agent_override else config.llm.provider
            )
        if llm_model is None:
            llm_model = (
                agent_override.model if agent_override else None
            )

    # Resolve API key from environment
    env_var = PROVIDER_ENV_KEYS.get(llm_provider.lower())
    if env_var is None:
        # Fall back to the generic pattern for unknown providers
        env_var = f"{llm_provider.upper()}_API_KEY"

    api_key = os.getenv(env_var)
    if not api_key:
        from api.errors import InvalidRequestError
        raise InvalidRequestError(
            f"Provider '{llm_provider}' is not configured on this server "
            f"(missing {env_var} environment variable)"
        )

    # Carry over base_url from agent config when the provider matches
    base_url: str | None = None
    if agent_override and agent_override.provider == llm_provider.lower():
        base_url = agent_override.base_url

    return LLMConfig(
        provider=llm_provider.lower(),
        model=llm_model or PROVIDER_DEFAULT_MODELS.get(llm_provider.lower(), ""),
        api_key=api_key,
        base_url=base_url,
    )


def get_agent_context(
    *,
    with_llm: bool = False,
    with_http: bool = False,
    llm_override: LLMConfig | None = None,
) -> AgentContext:
    """
    Create an AgentContext with optional capabilities.

    LLM API keys are resolved via LLMConfig.__post_init__ which reads
    the standard provider env var (e.g. OPENAI_API_KEY, XAI_API_KEY).
    The .env file is loaded automatically by core.config.runtime.

    Args:
        with_llm: Enable LLM client
        with_http: Enable HTTP client for network requests
        llm_override: Optional per-request LLM config override

    Returns:
        Configured AgentContext
    """
    config = _load_runtime_config()

    if with_llm and not config.llm.api_key:
        logger.warning(
            "LLM requested but no api_key resolved. "
            "Set the provider env var (e.g. OPENAI_API_KEY) in .env "
            "or configure llm.api_key in cournot.json."
        )

    ctx = AgentContext.create(config)

    # Apply per-request LLM override if provided
    if llm_override is not None and with_llm:
        ctx = ctx.with_llm_override(llm_override)

    # If LLM not requested, drop it so capability checks behave correctly
    if not with_llm:
        ctx.llm = None

    # If HTTP not requested, drop it
    if not with_http:
        ctx.http = None

    return ctx


def get_pipeline(
    *,
    strict_mode: bool = True,
    enable_sentinel: bool = True,
    enable_replay: bool = False,
    mode: str = "development",
    with_llm: bool = False,
    with_http: bool = False,
) -> Pipeline:
    """
    Create a Pipeline instance with the given configuration.
    
    Uses the registry-based agent system. In development mode,
    fallback agents are used when LLM/network is unavailable.
    
    Args:
        strict_mode: Enable strict mode for deterministic hashing
        enable_sentinel: Enable sentinel verification
        enable_replay: Enable replay mode
        mode: Execution mode (production, development, test)
        with_llm: Request LLM capability
        with_http: Request HTTP capability
    
    Returns:
        Configured Pipeline instance
    """
    # Map mode string to enum
    if mode == "production":
        exec_mode = ExecutionMode.PRODUCTION
    elif mode == "test":
        exec_mode = ExecutionMode.TEST
    else:
        exec_mode = ExecutionMode.DEVELOPMENT
    
    # Create configuration
    config = PipelineConfig(
        mode=exec_mode,
        strict_mode=strict_mode,
        enable_sentinel_verify=enable_sentinel,
        enable_replay=enable_replay,
        require_llm=with_llm and mode == "production",
        require_network=with_http and mode == "production",
    )
    
    # Create context with requested capabilities
    ctx = get_agent_context(with_llm=with_llm, with_http=with_http)
    
    return Pipeline(config=config, context=ctx)


def get_verification_context() -> AgentContext:
    """Get a minimal context for verification operations."""
    return AgentContext.create_minimal()
