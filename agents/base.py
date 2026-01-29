"""
Agent Base Classes

Defines the core agent interface and base implementation.

All agents in the Cournot protocol must:
1. Implement the Agent protocol
2. Declare their capabilities
3. Return AgentResult from their primary method
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Protocol, TYPE_CHECKING, runtime_checkable

if TYPE_CHECKING:
    from .context import AgentContext
    from core.receipts import ReceiptRef
    from core.schemas.verification import VerificationResult


class AgentCapability(str, Enum):
    """
    Capabilities that an agent may have.
    
    Used for routing and selection.
    """
    LLM = "llm"              # Uses LLM for reasoning
    NETWORK = "network"       # Makes network requests
    DETERMINISTIC = "deterministic"  # Produces deterministic outputs
    POLYMARKET = "polymarket"  # Specialized for Polymarket data
    BLOCKCHAIN = "blockchain"  # Can query blockchain data
    REPLAY = "replay"         # Supports replay verification


@dataclass
class AgentResult:
    """
    Standardized result from an agent operation.
    
    Contains the primary output, verification status, and receipts.
    """
    # The primary output (type depends on agent)
    output: Any
    
    # Verification result with checks
    verification: Optional["VerificationResult"] = None
    
    # Receipt references for audit trail
    receipts: list["ReceiptRef"] = field(default_factory=list)
    
    # Whether the operation succeeded
    success: bool = True
    
    # Error message if failed
    error: Optional[str] = None
    
    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def failure(
        cls,
        error: str,
        output: Any = None,
        verification: Optional["VerificationResult"] = None,
    ) -> "AgentResult":
        """Create a failure result."""
        return cls(
            output=output,
            verification=verification,
            success=False,
            error=error,
        )


@runtime_checkable
class Agent(Protocol):
    """
    Protocol defining the agent interface.
    
    All agents must implement this protocol.
    """
    
    @property
    def name(self) -> str:
        """Unique name identifying this agent implementation."""
        ...
    
    @property
    def version(self) -> str:
        """Version string (must change when behavior changes)."""
        ...
    
    @property
    def capabilities(self) -> set[AgentCapability]:
        """Set of capabilities this agent has."""
        ...


class BaseAgent(ABC):
    """
    Abstract base class for agents.
    
    Provides common functionality and enforces the agent contract.
    """
    
    # Subclasses must define these
    _name: str
    _version: str
    _capabilities: set[AgentCapability]
    
    def __init__(
        self,
        *,
        name: Optional[str] = None,
        version: Optional[str] = None,
    ) -> None:
        """
        Initialize base agent.
        
        Args:
            name: Override default name
            version: Override default version
        """
        self._name_override = name
        self._version_override = version
    
    @property
    def name(self) -> str:
        """Agent name."""
        return self._name_override or getattr(self, '_name', self.__class__.__name__)
    
    @property
    def version(self) -> str:
        """Agent version."""
        return self._version_override or getattr(self, '_version', 'v1')
    
    @property
    def capabilities(self) -> set[AgentCapability]:
        """Agent capabilities."""
        return getattr(self, '_capabilities', set())
    
    def has_capability(self, cap: AgentCapability) -> bool:
        """Check if agent has a specific capability."""
        return cap in self.capabilities
    
    @property
    def uses_llm(self) -> bool:
        """Check if agent uses LLM."""
        return AgentCapability.LLM in self.capabilities
    
    @property
    def uses_network(self) -> bool:
        """Check if agent uses network."""
        return AgentCapability.NETWORK in self.capabilities
    
    @property
    def is_deterministic(self) -> bool:
        """Check if agent is deterministic."""
        return AgentCapability.DETERMINISTIC in self.capabilities
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, version={self.version!r})"


# Type alias for agent factory functions
AgentFactory = Callable[["AgentContext"], Agent]


class AgentStep(str, Enum):
    """
    Pipeline steps that agents can fulfill.
    """
    PROMPT_ENGINEER = "prompt_engineer"
    COLLECTOR = "collector"
    AUDITOR = "auditor"
    JUDGE = "judge"
    SENTINEL = "sentinel"
