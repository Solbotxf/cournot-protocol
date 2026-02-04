"""
Agent Context

Provides dependency injection for agents, containing:
- LLM client
- HTTP client
- Receipt recorder
- Configuration
- Clock (can be frozen for determinism)
- Cache
- Logger

Agents receive context rather than creating their own clients,
enabling testability and consistent receipt recording.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from core.llm import LLMClient
    from core.http import HttpClient
    from core.receipts import ReceiptRecorder
    from core.config import RuntimeConfig
    from core.config.runtime import LLMConfig


class Clock(Protocol):
    """
    Protocol for time source.
    
    Can be real time or frozen for deterministic testing.
    """
    def now(self) -> datetime:
        """Get current UTC time."""
        ...


class RealClock:
    """Real-time clock implementation."""
    
    def now(self) -> datetime:
        return datetime.now(timezone.utc)


class FrozenClock:
    """
    Frozen clock for deterministic testing.
    
    Always returns the same time.
    """
    
    def __init__(self, frozen_time: Optional[datetime] = None) -> None:
        self._time = frozen_time or datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    
    def now(self) -> datetime:
        return self._time
    
    def set_time(self, time: datetime) -> None:
        """Set the frozen time."""
        self._time = time


class Cache(Protocol):
    """
    Protocol for caching.
    
    Can be in-memory, Redis, or no-op.
    """
    
    def get(self, key: str) -> Optional[Any]:
        """Get a cached value."""
        ...
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """Set a cached value."""
        ...
    
    def delete(self, key: str) -> None:
        """Delete a cached value."""
        ...


class InMemoryCache:
    """Simple in-memory cache implementation."""
    
    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}
    
    def get(self, key: str) -> Optional[Any]:
        return self._cache.get(key)
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        # Note: TTL not implemented for simplicity
        self._cache[key] = value
    
    def delete(self, key: str) -> None:
        self._cache.pop(key, None)
    
    def clear(self) -> None:
        self._cache.clear()


class NoOpCache:
    """No-op cache implementation."""
    
    def get(self, key: str) -> Optional[Any]:
        return None
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        pass
    
    def delete(self, key: str) -> None:
        pass


@dataclass
class AgentContext:
    """
    Context providing dependencies to agents.
    
    This is the primary mechanism for dependency injection.
    Agents should never create their own clients directly.
    
    Usage:
        ctx = AgentContext.create(config)
        
        # Pass to agent
        agent = MyAgent()
        result = agent.run(ctx, input_data)
        
        # Get receipts after execution
        receipts = ctx.recorder.get_receipts()
    """
    
    # Core dependencies
    llm: Optional["LLMClient"] = None
    http: Optional["HttpClient"] = None
    recorder: Optional["ReceiptRecorder"] = None
    config: Optional["RuntimeConfig"] = None
    
    # Utilities
    clock: Clock = field(default_factory=RealClock)
    cache: Cache = field(default_factory=NoOpCache)
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("cournot.agents"))
    
    # Execution context
    run_id: Optional[str] = None
    market_id: Optional[str] = None
    
    # Extra data for agent-specific needs
    extra: dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def create(
        cls,
        config: "RuntimeConfig",
        *,
        run_id: Optional[str] = None,
        market_id: Optional[str] = None,
        deterministic: bool = False,
    ) -> "AgentContext":
        """
        Create a fully configured context.
        
        Args:
            config: Runtime configuration
            run_id: Unique run identifier
            market_id: Market being processed
            deterministic: If True, use frozen clock
        
        Returns:
            Configured AgentContext
        """
        from core.receipts import ReceiptRecorder
        from core.llm import LLMClient, create_provider
        from core.llm.determinism import DecodingPolicy
        from core.http import HttpClient
        
        # Create receipt recorder
        recorder = ReceiptRecorder()
        
        # Create LLM client if configured
        llm = None
        if config.llm.api_key:
            provider = create_provider(
                config.llm.provider,
                model=config.llm.model,
                api_key=config.llm.api_key,
                base_url=config.llm.base_url,
            )
            llm = LLMClient(
                provider,
                default_policy=DecodingPolicy(
                    temperature=config.llm.temperature,
                    max_tokens=config.llm.max_tokens,
                ),
                recorder=recorder,
            )
        
        # Create HTTP client with generic browser-like headers (reduces 403 / connection reset)
        default_headers = {
            "User-Agent": config.http.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
        }
        http = HttpClient(
            timeout=config.http.timeout,
            recorder=recorder,
            default_headers=default_headers,
        )
        
        # Choose clock
        clock: Clock
        if deterministic or config.pipeline.deterministic_timestamps:
            clock = FrozenClock()
        else:
            clock = RealClock()
        
        # Choose cache
        cache: Cache
        if config.pipeline.debug:
            cache = InMemoryCache()
        else:
            cache = NoOpCache()
        
        # Create logger
        logger = logging.getLogger("cournot.agents")
        if config.pipeline.debug:
            logger.setLevel(logging.DEBUG)
        
        return cls(
            llm=llm,
            http=http,
            recorder=recorder,
            config=config,
            clock=clock,
            cache=cache,
            logger=logger,
            run_id=run_id,
            market_id=market_id,
        )
    
    @classmethod
    def create_minimal(cls) -> "AgentContext":
        """
        Create a minimal context for testing.
        
        No LLM or HTTP clients are created.
        """
        from core.receipts import ReceiptRecorder
        
        return cls(
            recorder=ReceiptRecorder(),
            clock=FrozenClock(),
            cache=InMemoryCache(),
        )
    
    @classmethod
    def create_mock(
        cls,
        *,
        llm_responses: Optional[list[str]] = None,
    ) -> "AgentContext":
        """
        Create a mock context for testing.
        
        Args:
            llm_responses: Preset LLM responses for MockProvider
        """
        from core.receipts import ReceiptRecorder
        from core.llm import LLMClient, MockProvider
        
        recorder = ReceiptRecorder()
        
        # Create mock LLM
        provider = MockProvider(responses=llm_responses or [])
        llm = LLMClient(provider, recorder=recorder)
        
        return cls(
            llm=llm,
            recorder=recorder,
            clock=FrozenClock(),
            cache=InMemoryCache(),
        )
    
    def now(self) -> datetime:
        """Get current time from clock."""
        return self.clock.now()
    
    def log(self, level: int, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log a message."""
        self.logger.log(level, msg, *args, **kwargs)
    
    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log a debug message."""
        self.logger.debug(msg, *args, **kwargs)
    
    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log an info message."""
        self.logger.info(msg, *args, **kwargs)
    
    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log a warning message."""
        self.logger.warning(msg, *args, **kwargs)
    
    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log an error message."""
        self.logger.error(msg, *args, **kwargs)
    
    def get_receipts(self) -> list:
        """Get all recorded receipts."""
        if self.recorder:
            return self.recorder.get_receipts()
        return []
    
    def get_receipt_refs(self) -> list:
        """Get receipt references."""
        if self.recorder:
            return self.recorder.get_receipt_refs()
        return []

    def with_llm_override(self, llm_config: "LLMConfig") -> "AgentContext":
        """Return a shallow copy with a different LLM client.

        All other fields (recorder, http, clock, cache) are shared.
        """
        import copy
        from core.llm import LLMClient, create_provider
        from core.llm.determinism import DecodingPolicy

        provider = create_provider(
            llm_config.provider,
            model=llm_config.model,
            api_key=llm_config.api_key,
            base_url=llm_config.base_url,
        )
        llm = LLMClient(
            provider,
            default_policy=DecodingPolicy(
                temperature=llm_config.temperature,
                max_tokens=llm_config.max_tokens,
            ),
            recorder=self.recorder,
        )
        ctx = copy.copy(self)
        ctx.llm = llm
        return ctx
