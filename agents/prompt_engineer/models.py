"""
Module 04 - Internal data models for Prompt Engineer.

Contains internal data structures used during prompt compilation,
including normalized user request and compilation result.
"""

import hashlib
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from agents.context import AgentContext
from core.schemas import PromptSpec, SourceTarget, ToolPlan


@dataclass
class NormalizedUserRequest:
    """
    Internal normalized representation of a user request.
    
    Even if the user provides raw text, the module normalizes to this
    structure before generating PromptSpec.
    """
    question: str
    time_config: dict[str, Any] = field(default_factory=dict)
    event_definition: str = ""
    outcome_type: str = "binary"
    possible_outcomes: list[str] = field(default_factory=lambda: ["YES", "NO"])
    threshold: Optional[str] = None
    source_preferences: list[str] = field(default_factory=list)
    source_targets: list[SourceTarget] = field(default_factory=list)
    confidence_policy: dict[str, float] = field(default_factory=dict)
    resolution_policy: dict[str, Any] = field(default_factory=dict)
    assumptions: list[str] = field(default_factory=list)
    raw_input: str = ""


@dataclass 
class CompilationResult:
    """Result of prompt compilation."""
    prompt_spec: PromptSpec
    tool_plan: ToolPlan
    assumptions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class PromptModule(Protocol):
    """
    Protocol for pluggable prompt compilation modules.
    
    Implementations can be swapped to use different compilation strategies
    (e.g., LLM-first vs DSL-first) without changing the orchestrator contract.
    """
    
    module_id: str
    version: str
    
    def compile(
        self,
        user_input: str,
        *,
        ctx: Optional[AgentContext] = None
    ) -> tuple[PromptSpec, ToolPlan]:
        """
        Compile user input into a PromptSpec and ToolPlan.
        
        Args:
            user_input: Raw user text describing the prediction
            ctx: Optional agent context
            
        Returns:
            Tuple of (PromptSpec, ToolPlan)
            
        Raises:
            PromptCompilationException: If compilation fails
        """
        ...


def generate_deterministic_id(prefix: str, *components: str) -> str:
    """
    Generate a deterministic ID from components.
    
    Args:
        prefix: ID prefix (e.g., "mkt", "req", "plan")
        components: String components to hash
        
    Returns:
        Deterministic ID string
    """
    combined = "|".join(str(c) for c in components)
    hash_bytes = hashlib.sha256(combined.encode("utf-8")).digest()
    short_hash = hash_bytes[:8].hex()
    return f"{prefix}_{short_hash}"


def generate_requirement_id(index: int) -> str:
    """Generate sequential requirement ID."""
    return f"req_{index:04d}"