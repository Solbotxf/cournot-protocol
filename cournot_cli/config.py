"""
Module 09C - CLI Configuration

Production configuration management for the Cournot CLI.
Supports environment variables and configuration files.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# Environment variable prefix
ENV_PREFIX = "COURNOT_"


@dataclass
class AgentConfig:
    """Configuration for an agent endpoint."""
    
    type: str = ""  # e.g., "http", "local", "mock"
    endpoint: str = ""
    api_key: str = ""
    timeout: int = 30
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineAgentsConfig:
    """Configuration for all pipeline agents."""
    
    prompt_engineer: AgentConfig = field(default_factory=AgentConfig)
    collector: AgentConfig = field(default_factory=AgentConfig)
    auditor: AgentConfig = field(default_factory=AgentConfig)
    judge: AgentConfig = field(default_factory=AgentConfig)
    sentinel: AgentConfig = field(default_factory=AgentConfig)


@dataclass
class CLIConfig:
    """Main CLI configuration."""
    
    # Pipeline settings
    strict_mode: bool = True
    enable_sentinel: bool = True
    enable_replay: bool = False
    
    # Timeouts
    pipeline_timeout: int = 300
    replay_timeout: int = 30
    
    # Agent configurations
    agents: PipelineAgentsConfig = field(default_factory=PipelineAgentsConfig)
    
    # Logging
    log_level: str = "INFO"
    log_file: str | None = None
    
    # Output
    default_output_format: str = "human"  # "human" or "json"


def load_config_from_env() -> CLIConfig:
    """Load configuration from environment variables."""
    config = CLIConfig()
    
    # Pipeline settings
    if os.getenv(f"{ENV_PREFIX}STRICT_MODE"):
        config.strict_mode = os.getenv(f"{ENV_PREFIX}STRICT_MODE", "true").lower() == "true"
    if os.getenv(f"{ENV_PREFIX}ENABLE_SENTINEL"):
        config.enable_sentinel = os.getenv(f"{ENV_PREFIX}ENABLE_SENTINEL", "true").lower() == "true"
    if os.getenv(f"{ENV_PREFIX}ENABLE_REPLAY"):
        config.enable_replay = os.getenv(f"{ENV_PREFIX}ENABLE_REPLAY", "false").lower() == "true"
    
    # Timeouts
    if os.getenv(f"{ENV_PREFIX}PIPELINE_TIMEOUT"):
        config.pipeline_timeout = int(os.getenv(f"{ENV_PREFIX}PIPELINE_TIMEOUT", "300"))
    if os.getenv(f"{ENV_PREFIX}REPLAY_TIMEOUT"):
        config.replay_timeout = int(os.getenv(f"{ENV_PREFIX}REPLAY_TIMEOUT", "30"))
    
    # Logging
    config.log_level = os.getenv(f"{ENV_PREFIX}LOG_LEVEL", "INFO")
    config.log_file = os.getenv(f"{ENV_PREFIX}LOG_FILE")
    
    # Agent configurations
    for agent_name in ["prompt_engineer", "collector", "auditor", "judge", "sentinel"]:
        agent_prefix = f"{ENV_PREFIX}{agent_name.upper()}_"
        agent_config = AgentConfig(
            type=os.getenv(f"{agent_prefix}TYPE", ""),
            endpoint=os.getenv(f"{agent_prefix}ENDPOINT", ""),
            api_key=os.getenv(f"{agent_prefix}API_KEY", ""),
            timeout=int(os.getenv(f"{agent_prefix}TIMEOUT", "30")),
        )
        setattr(config.agents, agent_name, agent_config)
    
    return config


def load_config_from_file(path: Path) -> CLIConfig:
    """Load configuration from a JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path, "r") as f:
        data = json.load(f)
    
    config = CLIConfig()
    
    # Pipeline settings
    config.strict_mode = data.get("strict_mode", config.strict_mode)
    config.enable_sentinel = data.get("enable_sentinel", config.enable_sentinel)
    config.enable_replay = data.get("enable_replay", config.enable_replay)
    
    # Timeouts
    config.pipeline_timeout = data.get("pipeline_timeout", config.pipeline_timeout)
    config.replay_timeout = data.get("replay_timeout", config.replay_timeout)
    
    # Logging
    config.log_level = data.get("log_level", config.log_level)
    config.log_file = data.get("log_file", config.log_file)
    
    # Agent configurations
    agents_data = data.get("agents", {})
    for agent_name in ["prompt_engineer", "collector", "auditor", "judge", "sentinel"]:
        if agent_name in agents_data:
            agent_data = agents_data[agent_name]
            agent_config = AgentConfig(
                type=agent_data.get("type", ""),
                endpoint=agent_data.get("endpoint", ""),
                api_key=agent_data.get("api_key", ""),
                timeout=agent_data.get("timeout", 30),
                extra=agent_data.get("extra", {}),
            )
            setattr(config.agents, agent_name, agent_config)
    
    return config


def load_config(config_path: Path | None = None) -> CLIConfig:
    """
    Load configuration from file and/or environment.
    
    Environment variables override file settings.
    
    Args:
        config_path: Optional path to config file
    
    Returns:
        Merged configuration
    """
    # Start with defaults
    config = CLIConfig()
    
    # Load from file if provided
    if config_path and config_path.exists():
        config = load_config_from_file(config_path)
    
    # Check for default config locations
    default_paths = [
        Path.cwd() / "cournot.json",
        Path.cwd() / ".cournot.json",
        Path.home() / ".config" / "cournot" / "config.json",
    ]
    
    if config_path is None:
        for default_path in default_paths:
            if default_path.exists():
                config = load_config_from_file(default_path)
                break
    
    # Override with environment variables
    env_config = load_config_from_env()
    
    # Merge env into config (env takes precedence)
    if os.getenv(f"{ENV_PREFIX}STRICT_MODE"):
        config.strict_mode = env_config.strict_mode
    if os.getenv(f"{ENV_PREFIX}ENABLE_SENTINEL"):
        config.enable_sentinel = env_config.enable_sentinel
    if os.getenv(f"{ENV_PREFIX}ENABLE_REPLAY"):
        config.enable_replay = env_config.enable_replay
    if os.getenv(f"{ENV_PREFIX}LOG_LEVEL"):
        config.log_level = env_config.log_level
    if os.getenv(f"{ENV_PREFIX}LOG_FILE"):
        config.log_file = env_config.log_file
    
    # Merge agent configs from env
    for agent_name in ["prompt_engineer", "collector", "auditor", "judge", "sentinel"]:
        agent_prefix = f"{ENV_PREFIX}{agent_name.upper()}_"
        if os.getenv(f"{agent_prefix}TYPE"):
            setattr(config.agents, agent_name, getattr(env_config.agents, agent_name))
    
    return config


def get_default_config_template() -> str:
    """Get a template configuration file."""
    return """{
  "strict_mode": true,
  "enable_sentinel": true,
  "enable_replay": false,
  "pipeline_timeout": 300,
  "replay_timeout": 30,
  "log_level": "INFO",
  "log_file": null,
  "agents": {
    "prompt_engineer": {
      "type": "http",
      "endpoint": "http://localhost:8001/prompt",
      "api_key": "",
      "timeout": 30
    },
    "collector": {
      "type": "http",
      "endpoint": "http://localhost:8002/collect",
      "api_key": "",
      "timeout": 60
    },
    "auditor": {
      "type": "http",
      "endpoint": "http://localhost:8003/audit",
      "api_key": "",
      "timeout": 60
    },
    "judge": {
      "type": "http",
      "endpoint": "http://localhost:8004/judge",
      "api_key": "",
      "timeout": 30
    },
    "sentinel": {
      "type": "http",
      "endpoint": "http://localhost:8005/verify",
      "api_key": "",
      "timeout": 30
    }
  }
}
"""