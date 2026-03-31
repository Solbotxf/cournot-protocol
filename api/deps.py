"""
Module 09D - API Dependencies

Dependency injection for the API.
Provides factories for pipeline and agent context.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from orchestrator.pipeline import Pipeline, PipelineConfig, ExecutionMode
from agents import AgentContext
from core.config.runtime import LLMConfig, RuntimeConfig
from core.llm.credentials import PROVIDER_DEFAULT_MODELS, resolve_credentials

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
        config = RuntimeConfig()

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
      2. Per-agent ``llm_override`` from ``cournot.json``.
      3. None — the caller falls back to the server-wide default LLM.

    Credentials are resolved via ``core.llm.credentials.resolve_credentials``
    — the single source of truth.
    """
    config = _load_runtime_config()

    # --- Tier 2: per-agent config fallback ---
    agent_override: LLMConfig | None = None
    if agent_name is not None:
        agent_cfg = getattr(config.agents, agent_name, None)
        if agent_cfg is not None:
            agent_override = agent_cfg.llm_override

    # --- Tier 1: per-request values take priority ---
    if llm_provider is None and llm_model is None:
        if agent_override is None:
            return None
        llm_provider = agent_override.provider
        llm_model = agent_override.model
    else:
        if llm_provider is None:
            llm_provider = (
                agent_override.provider if agent_override else config.llm.provider
            )
        if llm_model is None:
            llm_model = (
                agent_override.model if agent_override else None
            )

    # Resolve credentials from the central module.
    api_key, base_url = resolve_credentials(llm_provider.lower())

    if not api_key:
        from api.errors import InvalidRequestError
        raise InvalidRequestError(
            f"Provider '{llm_provider}' is not configured on this server. "
            f"Check COURNOT_UNIFIED_* or per-provider env vars."
        )

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
    """Create an AgentContext with optional capabilities.

    Credentials are resolved by ``LLMConfig.__post_init__`` which
    delegates to ``core.llm.credentials``.
    """
    config = _load_runtime_config()

    if with_llm and not config.llm.api_key:
        logger.warning(
            "LLM requested but no api_key resolved. "
            "Check COURNOT_UNIFIED_* or per-provider env vars in .env."
        )

    ctx = AgentContext.create(config)

    if llm_override is not None and with_llm:
        ctx = ctx.with_llm_override(llm_override)

    if not with_llm:
        ctx.llm = None
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
    """Create a Pipeline instance with the given configuration."""
    if mode == "production":
        exec_mode = ExecutionMode.PRODUCTION
    elif mode == "test":
        exec_mode = ExecutionMode.TEST
    else:
        exec_mode = ExecutionMode.DEVELOPMENT

    config = PipelineConfig(
        mode=exec_mode,
        strict_mode=strict_mode,
        enable_sentinel_verify=enable_sentinel,
        enable_replay=enable_replay,
        require_llm=with_llm and mode == "production",
        require_network=with_http and mode == "production",
    )

    ctx = get_agent_context(with_llm=with_llm, with_http=with_http)
    return Pipeline(config=config, context=ctx)


def get_verification_context() -> AgentContext:
    """Get a minimal context for verification operations."""
    return AgentContext.create_minimal()
