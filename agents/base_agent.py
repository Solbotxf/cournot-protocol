"""
Base agent module for all Cournot protocol agents.

Provides the base class and common interfaces that all agents inherit from.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, Optional, TypeVar

# Type variables for input/output
InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


@dataclass
class AgentContext:
    """
    Context provided to agents during execution.
    
    Contains configuration, session information, and shared state
    that agents may need during their operation.
    """
    session_id: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    # Optional LLM configuration
    llm_config: dict[str, Any] = field(default_factory=dict)
    
    # Determinism settings
    deterministic_mode: bool = True
    random_seed: Optional[int] = None


class BaseAgent(ABC, Generic[InputT, OutputT]):
    """
    Abstract base class for all Cournot protocol agents.
    
    Provides a standard interface for agent execution with
    typed inputs and outputs.
    """
    
    # Agent identification
    agent_type: str = "base"
    agent_version: str = "1.0.0"
    
    def __init__(self, context: Optional[AgentContext] = None):
        """
        Initialize the agent with optional context.
        
        Args:
            context: Execution context for the agent
        """
        self.context = context or AgentContext()
    
    @abstractmethod
    def run(self, input_data: InputT) -> OutputT:
        """
        Execute the agent's primary function.
        
        Args:
            input_data: Typed input for the agent
            
        Returns:
            Typed output from the agent
        """
        pass
    
    def validate_input(self, input_data: InputT) -> bool:
        """
        Validate input before processing.
        
        Override in subclasses for custom validation.
        
        Args:
            input_data: Input to validate
            
        Returns:
            True if valid, False otherwise
        """
        return True
    
    def get_agent_info(self) -> dict[str, Any]:
        """
        Get agent identification information.
        
        Returns:
            Dictionary with agent type and version
        """
        return {
            "agent_type": self.agent_type,
            "agent_version": self.agent_version,
        }