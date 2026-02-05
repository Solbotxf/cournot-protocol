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
class SerperConfig:
    """Configuration for Serper API (search operations via google.serper.dev)."""
    api_key: str = ""


@dataclass
class CollectorConfig:
    """Configuration for the Collector agent."""
    default_timeout_s: int = 20
    strict_tier_policy: bool = True
    include_timestamps: bool = False
    collector_id: str = "collector_v1"


@dataclass
class LLMConfig:
    """Configuration for LLM providers."""
    provider: str = ""  # "anthropic", "openai", etc.
    model: str = ""
    api_key: str = ""
    endpoint: str = ""
    temperature: float = 0.0
    max_tokens: int = 4096


@dataclass
class CLIConfig:
    """Main CLI configuration."""

    # Pipeline settings
    strict_mode: bool = True
    enable_sentinel: bool = True
    enable_replay: bool = False

    # Execution mode: "production", "development", "test"
    execution_mode: str = "development"

    # Capability requirements
    require_llm: bool = False
    require_network: bool = False

    # Timeouts
    pipeline_timeout: int = 300
    replay_timeout: int = 30

    # Audit evidence context limits (to stay under model context, e.g. 128K tokens)
    max_audit_evidence_chars: int = 300_000
    max_audit_evidence_items: int = 100

    # Collector settings
    collector: CollectorConfig = field(default_factory=CollectorConfig)

    # Serper API (for search operations in Collector)
    serper: SerperConfig = field(default_factory=SerperConfig)

    # LLM settings
    llm: LLMConfig = field(default_factory=LLMConfig)

    # Proxy settings (e.g., "http://user:pass@host:port")
    proxy: str = ""

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
    
    # Execution mode
    if os.getenv(f"{ENV_PREFIX}EXECUTION_MODE"):
        config.execution_mode = os.getenv(f"{ENV_PREFIX}EXECUTION_MODE", "development")
    
    # Capability requirements
    if os.getenv(f"{ENV_PREFIX}REQUIRE_LLM"):
        config.require_llm = os.getenv(f"{ENV_PREFIX}REQUIRE_LLM", "false").lower() == "true"
    if os.getenv(f"{ENV_PREFIX}REQUIRE_NETWORK"):
        config.require_network = os.getenv(f"{ENV_PREFIX}REQUIRE_NETWORK", "false").lower() == "true"
    
    # Timeouts
    if os.getenv(f"{ENV_PREFIX}PIPELINE_TIMEOUT"):
        config.pipeline_timeout = int(os.getenv(f"{ENV_PREFIX}PIPELINE_TIMEOUT", "300"))
    if os.getenv(f"{ENV_PREFIX}REPLAY_TIMEOUT"):
        config.replay_timeout = int(os.getenv(f"{ENV_PREFIX}REPLAY_TIMEOUT", "30"))
    
    # Audit evidence limits
    if os.getenv(f"{ENV_PREFIX}MAX_AUDIT_EVIDENCE_CHARS"):
        config.max_audit_evidence_chars = int(os.getenv(f"{ENV_PREFIX}MAX_AUDIT_EVIDENCE_CHARS", "300000"))
    if os.getenv(f"{ENV_PREFIX}MAX_AUDIT_EVIDENCE_ITEMS"):
        config.max_audit_evidence_items = int(os.getenv(f"{ENV_PREFIX}MAX_AUDIT_EVIDENCE_ITEMS", "100"))
    
    # Collector settings
    if os.getenv(f"{ENV_PREFIX}COLLECTOR_TIMEOUT"):
        config.collector.default_timeout_s = int(os.getenv(f"{ENV_PREFIX}COLLECTOR_TIMEOUT", "20"))
    if os.getenv(f"{ENV_PREFIX}COLLECTOR_STRICT_TIER"):
        config.collector.strict_tier_policy = os.getenv(f"{ENV_PREFIX}COLLECTOR_STRICT_TIER", "true").lower() == "true"
    
    # LLM settings
    if os.getenv(f"{ENV_PREFIX}LLM_PROVIDER"):
        config.llm.provider = os.getenv(f"{ENV_PREFIX}LLM_PROVIDER", "")
    if os.getenv(f"{ENV_PREFIX}LLM_MODEL"):
        config.llm.model = os.getenv(f"{ENV_PREFIX}LLM_MODEL", "")
    if os.getenv(f"{ENV_PREFIX}LLM_API_KEY"):
        config.llm.api_key = os.getenv(f"{ENV_PREFIX}LLM_API_KEY", "")
    if os.getenv(f"{ENV_PREFIX}LLM_ENDPOINT"):
        config.llm.endpoint = os.getenv(f"{ENV_PREFIX}LLM_ENDPOINT", "")
    
    # Proxy (use COURNOT_HTTP_PROXY for consistency with RuntimeConfig)
    if os.getenv(f"{ENV_PREFIX}HTTP_PROXY"):
        config.proxy = os.getenv(f"{ENV_PREFIX}HTTP_PROXY", "")

    # Logging
    config.log_level = os.getenv(f"{ENV_PREFIX}LOG_LEVEL", "INFO")
    config.log_file = os.getenv(f"{ENV_PREFIX}LOG_FILE")

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
    config.execution_mode = data.get("execution_mode", config.execution_mode)
    config.require_llm = data.get("require_llm", config.require_llm)
    config.require_network = data.get("require_network", config.require_network)
    
    # Timeouts
    config.pipeline_timeout = data.get("pipeline_timeout", config.pipeline_timeout)
    config.replay_timeout = data.get("replay_timeout", config.replay_timeout)
    
    # Audit evidence limits
    config.max_audit_evidence_chars = data.get("max_audit_evidence_chars", config.max_audit_evidence_chars)
    config.max_audit_evidence_items = data.get("max_audit_evidence_items", config.max_audit_evidence_items)
    
    # Collector settings
    collector_data = data.get("collector", {})
    config.collector = CollectorConfig(
        default_timeout_s=collector_data.get("default_timeout_s", 20),
        strict_tier_policy=collector_data.get("strict_tier_policy", True),
        include_timestamps=collector_data.get("include_timestamps", False),
        collector_id=collector_data.get("collector_id", "collector_v1"),
    )

    # Serper API (supports legacy google_cse key for api_key)
    serper_data = data.get("serper", {}) or data.get("google_cse", {})
    config.serper = SerperConfig(
        api_key=serper_data.get("api_key", ""),
    )
    
    # LLM settings
    llm_data = data.get("llm", {})
    config.llm = LLMConfig(
        provider=llm_data.get("provider", ""),
        model=llm_data.get("model", ""),
        api_key=llm_data.get("api_key", ""),
        endpoint=llm_data.get("endpoint", ""),
        temperature=llm_data.get("temperature", 0.0),
        max_tokens=llm_data.get("max_tokens", 4096),
    )
    
    # Proxy
    config.proxy = data.get("proxy", config.proxy)

    # Logging
    config.log_level = data.get("log_level", config.log_level)
    config.log_file = data.get("log_file", config.log_file)

    return config


def load_config(config_path: Path | None = None) -> CLIConfig:
    """
    Load configuration from file and/or environment.
    
    Environment variables override file settings.
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
    
    if os.getenv(f"{ENV_PREFIX}STRICT_MODE"):
        config.strict_mode = env_config.strict_mode
    if os.getenv(f"{ENV_PREFIX}ENABLE_SENTINEL"):
        config.enable_sentinel = env_config.enable_sentinel
    if os.getenv(f"{ENV_PREFIX}ENABLE_REPLAY"):
        config.enable_replay = env_config.enable_replay
    if os.getenv(f"{ENV_PREFIX}EXECUTION_MODE"):
        config.execution_mode = env_config.execution_mode
    if os.getenv(f"{ENV_PREFIX}LOG_LEVEL"):
        config.log_level = env_config.log_level
    if os.getenv(f"{ENV_PREFIX}LOG_FILE"):
        config.log_file = env_config.log_file
    if os.getenv(f"{ENV_PREFIX}LLM_PROVIDER"):
        config.llm = env_config.llm
    if os.getenv(f"{ENV_PREFIX}MAX_AUDIT_EVIDENCE_CHARS"):
        config.max_audit_evidence_chars = env_config.max_audit_evidence_chars
    if os.getenv(f"{ENV_PREFIX}MAX_AUDIT_EVIDENCE_ITEMS"):
        config.max_audit_evidence_items = env_config.max_audit_evidence_items
    
    return config


def get_default_config_template() -> str:
    """Get a template configuration file."""
    return """{
  "strict_mode": true,
  "enable_sentinel": true,
  "enable_replay": false,
  "execution_mode": "development",
  "require_llm": false,
  "require_network": false,
  "pipeline_timeout": 300,
  "replay_timeout": 30,
  "max_audit_evidence_chars": 300000,
  "max_audit_evidence_items": 100,
  "log_level": "INFO",
  "log_file": null,
  "collector": {
    "default_timeout_s": 20,
    "strict_tier_policy": true,
    "include_timestamps": false,
    "collector_id": "collector_v1"
  },
  "serper": {
    "api_key": ""
  },
  "llm": {
    "provider": "",
    "model": "",
    "api_key": "",
    "endpoint": "",
    "temperature": 0.0,
    "max_tokens": 4096
  }
}
"""
