"""
Runtime Configuration

Central configuration for pipeline execution, agent selection, and service setup.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional
from pathlib import Path
import yaml


@dataclass
class LLMConfig:
    """Configuration for LLM provider."""
    provider: str = "openai"
    model: str = "gpt-4o"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 4096
    
    def __post_init__(self):
        # Load API key from environment if not provided
        if self.api_key is None:
            env_var = f"{self.provider.upper()}_API_KEY"
            self.api_key = os.getenv(env_var)


@dataclass
class HttpConfig:
    """Configuration for HTTP client."""
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    user_agent: str = "cournot-protocol/1.0"


@dataclass
class AgentConfig:
    """Configuration for a specific agent."""
    name: str
    version: str = "v1"
    enabled: bool = True
    llm_override: Optional[LLMConfig] = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentsConfig:
    """Configuration for all agents."""
    prompt_engineer: AgentConfig = field(
        default_factory=lambda: AgentConfig(name="PromptEngineerLLM")
    )
    collector: AgentConfig = field(
        default_factory=lambda: AgentConfig(name="CollectorNetwork")
    )
    auditor: AgentConfig = field(
        default_factory=lambda: AgentConfig(name="AuditorLLM")
    )
    judge: AgentConfig = field(
        default_factory=lambda: AgentConfig(name="JudgeLLMStructured")
    )
    sentinel: AgentConfig = field(
        default_factory=lambda: AgentConfig(name="SentinelValidator")
    )


@dataclass
class PipelineConfig:
    """Configuration for pipeline execution."""
    strict_mode: bool = True
    enable_replay: bool = False
    enable_sentinel_verify: bool = True
    max_runtime_s: int = 300
    deterministic_timestamps: bool = True
    debug: bool = False
    record_receipts: bool = True


@dataclass
class RuntimeConfig:
    """
    Complete runtime configuration for the Cournot protocol.
    
    Can be loaded from:
    - Environment variables
    - YAML file
    - Programmatic construction
    """
    llm: LLMConfig = field(default_factory=LLMConfig)
    http: HttpConfig = field(default_factory=HttpConfig)
    agents: AgentsConfig = field(default_factory=AgentsConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    extra: dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_env(cls) -> "RuntimeConfig":
        """
        Load configuration from environment variables.
        
        Supported variables:
        - COURNOT_LLM_PROVIDER
        - COURNOT_LLM_MODEL
        - COURNOT_STRICT_MODE
        - COURNOT_DEBUG
        - OPENAI_API_KEY
        - ANTHROPIC_API_KEY
        - GOOGLE_API_KEY
        """
        llm = LLMConfig(
            provider=os.getenv("COURNOT_LLM_PROVIDER", "openai"),
            model=os.getenv("COURNOT_LLM_MODEL", "gpt-4o"),
        )
        
        pipeline = PipelineConfig(
            strict_mode=os.getenv("COURNOT_STRICT_MODE", "true").lower() == "true",
            debug=os.getenv("COURNOT_DEBUG", "false").lower() == "true",
        )
        
        return cls(llm=llm, pipeline=pipeline)
    
    @classmethod
    def from_yaml(cls, path: str | Path) -> "RuntimeConfig":
        """Load configuration from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path) as f:
            data = yaml.safe_load(f)
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RuntimeConfig":
        """Load configuration from a dictionary."""
        llm_data = data.get("llm", {})
        http_data = data.get("http", {})
        agents_data = data.get("agents", {})
        pipeline_data = data.get("pipeline", {})
        
        llm = LLMConfig(**llm_data) if llm_data else LLMConfig()
        http = HttpConfig(**http_data) if http_data else HttpConfig()
        pipeline = PipelineConfig(**pipeline_data) if pipeline_data else PipelineConfig()
        
        # Parse agents config
        agents = AgentsConfig()
        if agents_data:
            for agent_key, agent_conf in agents_data.items():
                if hasattr(agents, agent_key) and isinstance(agent_conf, dict):
                    setattr(agents, agent_key, AgentConfig(**agent_conf))
        
        return cls(
            llm=llm,
            http=http,
            agents=agents,
            pipeline=pipeline,
            extra=data.get("extra", {}),
        )
    
    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to a dictionary."""
        return {
            "llm": {
                "provider": self.llm.provider,
                "model": self.llm.model,
                "temperature": self.llm.temperature,
                "max_tokens": self.llm.max_tokens,
            },
            "http": {
                "timeout": self.http.timeout,
                "max_retries": self.http.max_retries,
            },
            "pipeline": {
                "strict_mode": self.pipeline.strict_mode,
                "enable_replay": self.pipeline.enable_replay,
                "enable_sentinel_verify": self.pipeline.enable_sentinel_verify,
                "deterministic_timestamps": self.pipeline.deterministic_timestamps,
                "debug": self.pipeline.debug,
            },
            "extra": self.extra,
        }


# Global default configuration
_default_config: Optional[RuntimeConfig] = None


def get_default_config() -> RuntimeConfig:
    """Get the default runtime configuration."""
    global _default_config
    if _default_config is None:
        _default_config = RuntimeConfig.from_env()
    return _default_config


def set_default_config(config: RuntimeConfig) -> None:
    """Set the default runtime configuration."""
    global _default_config
    _default_config = config
