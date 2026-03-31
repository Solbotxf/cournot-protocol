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
    max_tokens: int = 8192
    
    def __post_init__(self):
        from core.llm.credentials import resolve_credentials, is_unified_mode
        resolved_key, resolved_base = resolve_credentials(self.provider)

        if is_unified_mode():
            # Unified mode: gateway credentials ALWAYS win over config-file
            # values (cournot.json may contain stale per-provider URLs).
            self.api_key = resolved_key
            self.base_url = resolved_base
        else:
            if self.api_key is None:
                self.api_key = resolved_key
            if self.base_url is None:
                self.base_url = resolved_base


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
class APIDataProviderConfig:
    """Per-provider configuration for the API data collector."""
    enabled: bool = True
    api_key: str | None = None
    base_url: str | None = None
    timeout: float = 30.0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class APIDataConfig:
    """Configuration for the CollectorAPIData agent."""
    enabled: bool = True
    use_llm_matching: bool = True
    providers: dict[str, APIDataProviderConfig] = field(default_factory=lambda: {
        "open_meteo": APIDataProviderConfig(enabled=True),
        "coinmarketcap": APIDataProviderConfig(enabled=False),
        "binance": APIDataProviderConfig(enabled=True),
    })


@dataclass
class PANSearchConfig:
    """Configuration for PAN (ENCOMPASS-style) collector search."""
    search_algo: str = "beam"          # "bon_global" | "bon_local" | "beam"
    default_branching: int = 3         # N candidates per branchpoint
    beam_width: int = 2                # K beams to keep (beam search only)
    max_expansions: int = 50           # safety cap on total generate_fn calls
    seed: Optional[int] = None         # RNG seed for determinism


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
    pan: PANSearchConfig = field(default_factory=PANSearchConfig)
    api_data: APIDataConfig = field(default_factory=APIDataConfig)
    proxy: Optional[str] = None  # e.g., "http://user:pass@host:port"
    extra: dict[str, Any] = field(default_factory=dict)
    
    @staticmethod
    def _get_env_overrides() -> dict[str, Any]:
        """
        Get configuration overrides from environment variables.

        LLM credentials (api_key, base_url) are NOT resolved here — they
        are handled by ``core.llm.credentials.resolve_credentials`` which
        is called from ``LLMConfig.__post_init__``.  This method only
        handles non-credential overrides and explicit ``COURNOT_LLM_*``
        provider/model selection.

        Supported variables:
        - COURNOT_LLM_PROVIDER: LLM provider name
        - COURNOT_LLM_MODEL: LLM model name
        - COURNOT_STRICT_MODE: Enable strict mode (true/false)
        - COURNOT_DEBUG: Enable debug mode (true/false)
        - COURNOT_HTTP_PROXY: HTTP proxy URL
        - SERPER_API_KEY: Serper API key for search
        - CMC_PRO_API_KEY: CoinMarketCap API key for crypto data
        """
        overrides: dict[str, Any] = {}

        # LLM provider/model selection (credentials resolved by LLMConfig)
        if os.getenv("COURNOT_LLM_PROVIDER"):
            overrides.setdefault("llm", {})["provider"] = os.getenv("COURNOT_LLM_PROVIDER")
        if os.getenv("COURNOT_LLM_MODEL"):
            overrides.setdefault("llm", {})["model"] = os.getenv("COURNOT_LLM_MODEL")

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

        # CoinMarketCap API key
        if os.getenv("CMC_PRO_API_KEY"):
            overrides.setdefault("api_data", {}).setdefault("providers", {}).setdefault(
                "coinmarketcap", {}
            )["api_key"] = os.getenv("CMC_PRO_API_KEY")
            overrides["api_data"]["providers"]["coinmarketcap"]["enabled"] = True

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
        pan_data = data.get("pan", {})
        if isinstance(serper_data, dict):
            serper = SerperConfig(api_key=serper_data.get("api_key"))
        else:
            serper = SerperConfig()

        llm = LLMConfig(**llm_data) if llm_data else LLMConfig()
        http = HttpConfig(**http_data) if http_data else HttpConfig()
        pipeline = PipelineConfig(**pipeline_data) if pipeline_data else PipelineConfig()
        pan = PANSearchConfig(**pan_data) if pan_data else PANSearchConfig()

        # Parse api_data config
        api_data_raw = data.get("api_data", {})
        if api_data_raw:
            providers_raw = api_data_raw.get("providers", {})
            providers: dict[str, APIDataProviderConfig] = {}
            for pid, pcfg in providers_raw.items():
                if isinstance(pcfg, dict):
                    providers[pid] = APIDataProviderConfig(**pcfg)
                else:
                    providers[pid] = pcfg
            api_data = APIDataConfig(
                enabled=api_data_raw.get("enabled", True),
                use_llm_matching=api_data_raw.get("use_llm_matching", True),
                providers=providers if providers else APIDataConfig().providers,
            )
        else:
            api_data = APIDataConfig()

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
            pan=pan,
            api_data=api_data,
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

        # API data overrides
        if "api_data" in overrides:
            ad = overrides["api_data"]
            if "enabled" in ad:
                new_config.api_data.enabled = ad["enabled"]
            if "use_llm_matching" in ad:
                new_config.api_data.use_llm_matching = ad["use_llm_matching"]
            for pid, pcfg in ad.get("providers", {}).items():
                if isinstance(pcfg, dict):
                    existing = new_config.api_data.providers.get(
                        pid, APIDataProviderConfig()
                    )
                    for k, v in pcfg.items():
                        setattr(existing, k, v)
                    new_config.api_data.providers[pid] = existing

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
            "pan": {
                "search_algo": self.pan.search_algo,
                "default_branching": self.pan.default_branching,
                "beam_width": self.pan.beam_width,
                "max_expansions": self.pan.max_expansions,
                "seed": self.pan.seed,
            },
            "api_data": {
                "enabled": self.api_data.enabled,
                "use_llm_matching": self.api_data.use_llm_matching,
                "providers": {
                    pid: {
                        "enabled": pcfg.enabled,
                        "api_key": pcfg.api_key,
                        "base_url": pcfg.base_url,
                        "timeout": pcfg.timeout,
                    }
                    for pid, pcfg in self.api_data.providers.items()
                },
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
