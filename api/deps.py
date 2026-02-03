"""
Module 09D - API Dependencies

Dependency injection for the API.
Provides factories for pipeline and agent context.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from orchestrator.pipeline import Pipeline, PipelineConfig, ExecutionMode
from agents import AgentContext

logger = logging.getLogger(__name__)


def _load_cli_config():
    """Load CLIConfig from cournot.json (with env var overrides)."""
    try:
        from cournot_cli.config import load_config
        return load_config()
    except Exception as e:
        logger.debug(f"Could not load cournot.json: {e}")
        return None


def get_agent_context(
    *,
    with_llm: bool = False,
    with_http: bool = False,
) -> AgentContext:
    """
    Create an AgentContext with optional capabilities.

    Resolution order for LLM config:
      1. Environment variables (COURNOT_LLM_PROVIDER, COURNOT_LLM_API_KEY, COURNOT_LLM_MODEL)
      2. cournot.json (searched in cwd, then ~/.config/cournot/)

    Args:
        with_llm: Enable LLM client
        with_http: Enable HTTP client for network requests

    Returns:
        Configured AgentContext
    """
    # Start with minimal context
    ctx = AgentContext.create_minimal()

    # Add LLM if requested and configured
    if with_llm:
        # Try env vars first
        provider = os.getenv("COURNOT_LLM_PROVIDER")
        api_key = os.getenv("COURNOT_LLM_API_KEY")
        model = os.getenv("COURNOT_LLM_MODEL")
        endpoint = None

        # Fall back to cournot.json
        if not provider or not api_key:
            cli_config = _load_cli_config()
            if cli_config and cli_config.llm.provider and cli_config.llm.api_key:
                provider = provider or cli_config.llm.provider
                api_key = api_key or cli_config.llm.api_key
                model = model or cli_config.llm.model
                endpoint = cli_config.llm.endpoint or None
                logger.info(f"Loaded LLM config from cournot.json (provider={provider})")

        if not provider or not api_key:
            logger.warning(
                "LLM requested but no provider/api_key found. "
                "Set COURNOT_LLM_PROVIDER/COURNOT_LLM_API_KEY env vars "
                "or configure llm.provider/llm.api_key in cournot.json."
            )
        else:
            try:
                from core.llm import create_llm_client
                llm = create_llm_client(
                    provider=provider,
                    api_key=api_key,
                    model=model,
                    endpoint=endpoint,
                )
                ctx = AgentContext(
                    llm=llm,
                    http=ctx.http,
                    recorder=ctx.recorder,
                    config=ctx.config,
                    cache=ctx.cache,
                    logger=ctx.logger,
                )
            except Exception as e:
                logger.error(f"Failed to create LLM client: {e}")

    # Add HTTP client if requested
    if with_http:
        try:
            from core.http import HttpClient
            http = HttpClient()
            ctx = AgentContext(
                llm=ctx.llm,
                http=http,
                recorder=ctx.recorder,
                config=ctx.config,
                cache=ctx.cache,
                logger=ctx.logger,
            )
        except Exception:
            pass  # Fall back to no HTTP

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
