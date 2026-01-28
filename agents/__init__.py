"""
Base Agent Module

Provides the abstract base class for all Cournot protocol agents.
All agents (PromptEngineer, Collector, Auditor, Judge, Sentinel) inherit from BaseAgent.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class AgentContext:
    """
    Context object passed to agents during execution.

    Provides shared state, configuration, and utilities that agents
    may need during their operations.
    """

    # Unique run identifier
    run_id: Optional[str] = None

    # Configuration flags
    strict_mode: bool = True
    debug: bool = False

    # Shared context data (agent-specific)
    data: dict[str, Any] = field(default_factory=dict)

    # Logging/telemetry hook
    log_fn: Optional[Any] = None

    def log(self, level: str, message: str, **kwargs: Any) -> None:
        """Log a message if logging is configured."""
        if self.log_fn is not None:
            self.log_fn(level, message, **kwargs)


class BaseAgent(ABC):
    """
    Abstract base class for all Cournot protocol agents.

    Agents are stateless processors that implement specific roles
    in the resolution pipeline:
    - PromptEngineer: Compiles user input to PromptSpec
    - Collector: Gathers evidence according to ToolPlan
    - Auditor: Produces reasoning trace from evidence
    - Judge: Produces deterministic verdict
    - Sentinel: Verifies PoR bundles and generates challenges

    Attributes:
        role: The agent's role identifier (e.g., "prompt_engineer", "collector")
        name: Optional instance name for debugging/logging
    """

    # Subclasses should override this
    role: str = "base"

    def __init__(self, *, name: Optional[str] = None) -> None:
        """
        Initialize the agent.

        Args:
            name: Optional instance name for debugging/logging
        """
        self._name = name

    @property
    def name(self) -> str:
        """Get the agent's name (uses role if no name provided)."""
        return self._name or self.role

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, role={self.role!r})"