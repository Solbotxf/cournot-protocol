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
from core.config.runtime import RuntimeConfig

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


def get_agent_context(
    *,
    with_llm: bool = False,
    with_http: bool = False,
) -> AgentContext:
    """
    Create an AgentContext with optional capabilities.

    LLM API keys are resolved via LLMConfig.__post_init__ which reads
    the standard provider env var (e.g. OPENAI_API_KEY, XAI_API_KEY).
    The .env file is loaded automatically by core.config.runtime.

    Args:
        with_llm: Enable LLM client
        with_http: Enable HTTP client for network requests

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
