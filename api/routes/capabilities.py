"""
Capabilities Route

Discovery endpoint for available agents and configured LLM providers.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from agents.registry import get_registry
from agents.base import AgentStep
from core.llm.providers import get_configured_providers


logger = logging.getLogger(__name__)

router = APIRouter(tags=["capabilities"])


class ProviderInfo(BaseModel):
    """Information about a configured LLM provider."""

    provider: str = Field(..., description="Provider name")
    default_model: str = Field(..., description="Default model for this provider")


class AgentInfo(BaseModel):
    """Information about a registered agent."""

    name: str = Field(..., description="Agent name")
    version: str = Field(..., description="Agent version")
    capabilities: list[str] = Field(default_factory=list, description="Agent capabilities")
    priority: int = Field(default=0, description="Selection priority")
    is_fallback: bool = Field(default=False, description="Whether this is a fallback agent")


class StepAgents(BaseModel):
    """Agents available for a pipeline step."""

    step: str = Field(..., description="Pipeline step name")
    agents: list[AgentInfo] = Field(default_factory=list)


class CapabilitiesResponse(BaseModel):
    """Response for GET /capabilities."""

    ok: bool = True
    providers: list[ProviderInfo] = Field(
        default_factory=list,
        description="LLM providers configured on this server (have API keys)",
    )
    steps: list[StepAgents] = Field(
        default_factory=list,
        description="Available agents per pipeline step",
    )


@router.get("/capabilities", response_model=CapabilitiesResponse)
async def get_capabilities() -> CapabilitiesResponse:
    """
    Return available agents and configured LLM providers.

    Only providers that have API keys set in the server environment
    are included. This allows clients to populate provider/model
    dropdowns without exposing secrets.
    """
    # Get configured providers
    raw_providers = get_configured_providers()
    providers = [
        ProviderInfo(provider=p["provider"], default_model=p["default_model"])
        for p in raw_providers
    ]

    # Get agents per step from registry
    registry = get_registry()
    steps = []
    for step in AgentStep:
        entries = registry.list_agents(step)
        agents = [
            AgentInfo(
                name=e.name,
                version=e.version,
                capabilities=[c.value for c in e.capabilities],
                priority=e.priority,
                is_fallback=e.is_fallback,
            )
            for e in entries
        ]
        steps.append(StepAgents(step=step.value, agents=agents))

    return CapabilitiesResponse(
        ok=True,
        providers=providers,
        steps=steps,
    )
