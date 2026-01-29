"""
Agents Module

Provides the base agent infrastructure including:
- Agent interface and base class
- AgentContext for dependency injection
- AgentResult for standardized outputs
- Agent registry for dynamic selection
- Prompt Engineer agents
"""

from .base import (
    Agent,
    AgentCapability,
    AgentFactory,
    AgentResult,
    AgentStep,
    BaseAgent,
)
from .context import AgentContext, FrozenClock, RealClock
from .registry import AgentRegistry, get_agent, get_registry, register_agent

# Import prompt_engineer to trigger auto-registration
from . import prompt_engineer

# Import collector to trigger auto-registration
from . import collector

# Import auditor to trigger auto-registration
from . import auditor

# Import judge to trigger auto-registration
from . import judge

# Import sentinel to trigger auto-registration
from . import sentinel

__all__ = [
    # Base
    "Agent",
    "AgentCapability",
    "AgentFactory",
    "AgentResult",
    "AgentStep",
    "BaseAgent",
    # Context
    "AgentContext",
    "FrozenClock",
    "RealClock",
    # Registry
    "AgentRegistry",
    "get_agent",
    "get_registry",
    "register_agent",
    # Prompt Engineer
    "prompt_engineer",
    # Collector
    "collector",
    # Auditor
    "auditor",
    # Judge
    "judge",
    # Sentinel
    "sentinel",
]