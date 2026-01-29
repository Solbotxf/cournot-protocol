"""
Agent Registry

Provides dynamic agent registration and selection.

Supports:
- Registration of multiple agents per step
- Selection based on capabilities and configuration
- Fallback chain support
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, TYPE_CHECKING

from .base import Agent, AgentCapability, AgentStep

if TYPE_CHECKING:
    from .context import AgentContext
    from core.config import RuntimeConfig


@dataclass
class AgentEntry:
    """
    Entry in the agent registry.
    """
    name: str
    version: str
    step: AgentStep
    factory: Callable[["AgentContext"], Agent]
    capabilities: set[AgentCapability] = field(default_factory=set)
    priority: int = 0  # Higher = preferred
    is_fallback: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class AgentRegistry:
    """
    Registry for agent implementations.
    
    Enables dynamic agent selection based on:
    - Configuration
    - Required capabilities
    - Availability
    
    Usage:
        registry = AgentRegistry()
        
        # Register agents
        registry.register(
            step=AgentStep.PROMPT_ENGINEER,
            name="PromptEngineerLLM",
            factory=lambda ctx: PromptEngineerLLM(ctx),
            capabilities={AgentCapability.LLM},
        )
        
        # Get agent for a step
        agent = registry.get_agent(
            step=AgentStep.PROMPT_ENGINEER,
            ctx=agent_context,
        )
    """
    
    def __init__(self) -> None:
        self._entries: dict[AgentStep, list[AgentEntry]] = {}
        self._by_name: dict[str, AgentEntry] = {}
    
    def register(
        self,
        step: AgentStep,
        name: str,
        factory: Callable[["AgentContext"], Agent],
        *,
        version: str = "v1",
        capabilities: Optional[set[AgentCapability]] = None,
        priority: int = 0,
        is_fallback: bool = False,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Register an agent implementation.
        
        Args:
            step: Pipeline step this agent fulfills
            name: Unique name for this agent
            factory: Function to create agent instance
            version: Agent version string
            capabilities: Set of capabilities
            priority: Selection priority (higher = preferred)
            is_fallback: Whether this is a fallback implementation
            metadata: Additional metadata
        """
        entry = AgentEntry(
            name=name,
            version=version,
            step=step,
            factory=factory,
            capabilities=capabilities or set(),
            priority=priority,
            is_fallback=is_fallback,
            metadata=metadata or {},
        )
        
        if step not in self._entries:
            self._entries[step] = []
        
        self._entries[step].append(entry)
        self._by_name[name] = entry
        
        # Keep sorted by priority (descending)
        self._entries[step].sort(key=lambda e: e.priority, reverse=True)
    
    def get_agent(
        self,
        step: AgentStep,
        ctx: "AgentContext",
        *,
        name: Optional[str] = None,
        required_capabilities: Optional[set[AgentCapability]] = None,
        prefer_deterministic: bool = False,
        allow_fallback: bool = True,
    ) -> Agent:
        """
        Get an agent for a pipeline step.
        
        Args:
            step: The pipeline step
            ctx: Agent context
            name: Specific agent name to use (overrides selection)
            required_capabilities: Capabilities the agent must have
            prefer_deterministic: Prefer deterministic agents
            allow_fallback: Allow fallback agents if primary fails
        
        Returns:
            Agent instance
        
        Raises:
            ValueError: If no suitable agent found
        """
        # If specific name requested, use that
        if name:
            return self.get_agent_by_name(name, ctx)
        
        # Get candidates for this step
        candidates = self._entries.get(step, [])
        if not candidates:
            raise ValueError(f"No agents registered for step: {step}")
        
        # Filter by capabilities
        if required_capabilities:
            candidates = [
                e for e in candidates
                if required_capabilities.issubset(e.capabilities)
            ]
        
        # Filter out fallbacks if not allowed
        if not allow_fallback:
            non_fallback = [e for e in candidates if not e.is_fallback]
            if non_fallback:
                candidates = non_fallback
        
        # Prefer deterministic if requested
        if prefer_deterministic:
            deterministic = [
                e for e in candidates
                if AgentCapability.DETERMINISTIC in e.capabilities
            ]
            if deterministic:
                candidates = deterministic
        
        if not candidates:
            raise ValueError(
                f"No suitable agent found for step {step} "
                f"with capabilities {required_capabilities}"
            )
        
        # Return highest priority candidate
        entry = candidates[0]
        return entry.factory(ctx)
    
    def get_agent_by_name(
        self,
        name: str,
        ctx: "AgentContext",
    ) -> Agent:
        """
        Get a specific agent by name.
        
        Args:
            name: Agent name
            ctx: Agent context
        
        Returns:
            Agent instance
        
        Raises:
            ValueError: If agent not found
        """
        entry = self._by_name.get(name)
        if not entry:
            raise ValueError(f"Agent not found: {name}")
        return entry.factory(ctx)
    
    def list_agents(self, step: Optional[AgentStep] = None) -> list[AgentEntry]:
        """
        List registered agents.
        
        Args:
            step: Filter by step (None = all agents)
        
        Returns:
            List of agent entries
        """
        if step:
            return list(self._entries.get(step, []))
        
        all_entries = []
        for entries in self._entries.values():
            all_entries.extend(entries)
        return all_entries
    
    def has_agent(self, name: str) -> bool:
        """Check if an agent is registered."""
        return name in self._by_name
    
    def has_step(self, step: AgentStep) -> bool:
        """Check if any agents are registered for a step."""
        return step in self._entries and len(self._entries[step]) > 0


# Global registry instance
_global_registry: Optional[AgentRegistry] = None


def get_registry() -> AgentRegistry:
    """Get the global agent registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = AgentRegistry()
    return _global_registry


def register_agent(
    step: AgentStep,
    name: str,
    factory: Callable[["AgentContext"], Agent],
    **kwargs: Any,
) -> None:
    """
    Register an agent in the global registry.
    
    Convenience function for module-level registration.
    """
    get_registry().register(step, name, factory, **kwargs)


def get_agent(
    step: AgentStep,
    ctx: "AgentContext",
    **kwargs: Any,
) -> Agent:
    """
    Get an agent from the global registry.
    
    Convenience function for getting agents.
    """
    return get_registry().get_agent(step, ctx, **kwargs)
