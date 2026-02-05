"""
Runtime Configuration

Central configuration for pipeline execution, agent selection, and service setup.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


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


# Generic browser User-Agent (simulates a real user, not tied to any specific machine).
_DEFAULT_HTTP_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


@dataclass
class HttpConfig:
    """Configuration for HTTP client."""
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    user_agent: str = _DEFAULT_HTTP_USER_AGENT


@dataclass
class SerperConfig:
    """Configuration for Serper API (search operations via google.serper.dev)."""
    api_key: Optional[str] = None




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
    strict_mode: bool = False
    enable_replay: bool = False
    enable_sentinel_verify: bool = True
    max_runtime_s: int = 300
    deterministic_timestamps: bool = False
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
    serper: SerperConfig = field(default_factory=SerperConfig)
    proxy: Optional[str] = None  # e.g., "http://user:pass@host:port"
    extra: dict[str, Any] = field(default_factory=dict)
    
    @staticmethod
    def _get_env_overrides() -> dict[str, Any]:
        """
        Get configuration overrides from environment variables.

        This is the SINGLE source of truth for all env var reading.

        Supported variables:
        - COURNOT_LLM_PROVIDER: LLM provider name
        - COURNOT_LLM_MODEL: LLM model name
        - COURNOT_LLM_API_KEY: LLM API key (also checks {PROVIDER}_API_KEY)
        - COURNOT_STRICT_MODE: Enable strict mode (true/false)
        - COURNOT_DEBUG: Enable debug mode (true/false)
        - COURNOT_HTTP_PROXY: HTTP proxy URL
        - SERPER_API_KEY: Serper API key for search
        """
        overrides: dict[str, Any] = {}

        # LLM settings
        if os.getenv("COURNOT_LLM_PROVIDER"):
            overrides.setdefault("llm", {})["provider"] = os.getenv("COURNOT_LLM_PROVIDER")
        if os.getenv("COURNOT_LLM_MODEL"):
            overrides.setdefault("llm", {})["model"] = os.getenv("COURNOT_LLM_MODEL")
        if os.getenv("COURNOT_LLM_API_KEY"):
            overrides.setdefault("llm", {})["api_key"] = os.getenv("COURNOT_LLM_API_KEY")

        # Pipeline settings
        if os.getenv("COURNOT_STRICT_MODE"):
            overrides.setdefault("pipeline", {})["strict_mode"] = (
                os.getenv("COURNOT_STRICT_MODE", "true").lower() == "true"
            )
        if os.getenv("COURNOT_DEBUG"):
            overrides.setdefault("pipeline", {})["debug"] = (
                os.getenv("COURNOT_DEBUG", "false").lower() == "true"
            )

        # Serper API
        if os.getenv("SERPER_API_KEY"):
            overrides.setdefault("serper", {})["api_key"] = os.getenv("SERPER_API_KEY")

        # Proxy
        if os.getenv("COURNOT_HTTP_PROXY"):
            overrides["proxy"] = os.getenv("COURNOT_HTTP_PROXY")

        return overrides

    @classmethod
    def from_env(cls) -> "RuntimeConfig":
        """
        Load configuration purely from environment variables.

        Uses defaults for any values not specified in env vars.
        """
        overrides = cls._get_env_overrides()
        # Start with defaults and apply env overrides
        return cls.from_dict(overrides)
    
    @classmethod
    def from_yaml(cls, path: str | Path) -> "RuntimeConfig":
        """Load configuration from a YAML file."""
        import yaml
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path) as f:
            data = yaml.safe_load(f)
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RuntimeConfig":
        """Load configuration from a dictionary (supports partial data)."""
        llm_data = data.get("llm", {})
        http_data = data.get("http", {})
        agents_data = data.get("agents", {})
        pipeline_data = data.get("pipeline", {})
        serper_data = data.get("serper", {}) or data.get("google_cse", {})
        if isinstance(serper_data, dict):
            serper = SerperConfig(api_key=serper_data.get("api_key"))
        else:
            serper = SerperConfig()

        llm = LLMConfig(**llm_data) if llm_data else LLMConfig()
        http = HttpConfig(**http_data) if http_data else HttpConfig()
        pipeline = PipelineConfig(**pipeline_data) if pipeline_data else PipelineConfig()

        # Parse agents config
        agents = AgentsConfig()
        if agents_data:
            for agent_key, agent_conf in agents_data.items():
                if hasattr(agents, agent_key) and isinstance(agent_conf, dict):
                    conf = dict(agent_conf)
                    if isinstance(conf.get("llm_override"), dict):
                        conf["llm_override"] = LLMConfig(**conf["llm_override"])
                    setattr(agents, agent_key, AgentConfig(**conf))

        return cls(
            llm=llm,
            http=http,
            agents=agents,
            pipeline=pipeline,
            serper=serper,
            proxy=data.get("proxy"),
            extra=data.get("extra", {}),
        )

    def with_env_overrides(self) -> "RuntimeConfig":
        """
        Return a new config with environment variable overrides applied.

        This allows loading from a config file first, then overlaying env vars.
        """
        overrides = self._get_env_overrides()
        if not overrides:
            return self

        # Apply overrides to current config
        import copy
        new_config = copy.deepcopy(self)

        # LLM overrides
        if "llm" in overrides:
            for key, value in overrides["llm"].items():
                setattr(new_config.llm, key, value)

        # Pipeline overrides
        if "pipeline" in overrides:
            for key, value in overrides["pipeline"].items():
                setattr(new_config.pipeline, key, value)

        # Serper overrides
        if "serper" in overrides:
            for key, value in overrides["serper"].items():
                setattr(new_config.serper, key, value)

        # Proxy override
        if "proxy" in overrides:
            new_config.proxy = overrides["proxy"]

        return new_config
    
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
            "serper": {
                "api_key": self.serper.api_key,
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
