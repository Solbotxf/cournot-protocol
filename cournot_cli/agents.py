"""
Module 09C - Agent Interfaces and Loading

Production agent interfaces for the Cournot CLI.
Defines protocols for agents and provides loading utilities.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable

from cournot_cli.config import AgentConfig, PipelineAgentsConfig

from core.schemas.prompts import PromptSpec
from core.schemas.transport import ToolPlan
from core.schemas.evidence import EvidenceBundle
from core.schemas.verdict import DeterministicVerdict
from core.schemas.verification import VerificationResult
from core.por.reasoning_trace import ReasoningTrace
from orchestrator.pipeline import PoRPackage


logger = logging.getLogger(__name__)


# =============================================================================
# Agent Protocols
# =============================================================================

@runtime_checkable
class PromptEngineerProtocol(Protocol):
    """Protocol for Prompt Engineer agents."""
    
    def run(self, user_input: str, **kwargs) -> tuple[PromptSpec, ToolPlan | None]:
        """
        Generate a PromptSpec and optional ToolPlan from user input.
        
        Args:
            user_input: The user's query or prediction request
            **kwargs: Additional configuration
        
        Returns:
            Tuple of (PromptSpec, optional ToolPlan)
        """
        ...


@runtime_checkable
class CollectorProtocol(Protocol):
    """Protocol for Evidence Collector agents."""
    
    def collect(
        self,
        prompt_spec: PromptSpec,
        tool_plan: ToolPlan | None,
        **kwargs,
    ) -> EvidenceBundle:
        """
        Collect evidence based on the prompt spec and tool plan.
        
        Args:
            prompt_spec: The prompt specification
            tool_plan: Optional tool plan for collection
            **kwargs: Additional configuration
        
        Returns:
            EvidenceBundle with collected evidence
        """
        ...


@runtime_checkable
class AuditorProtocol(Protocol):
    """Protocol for Auditor agents."""
    
    def audit(
        self,
        prompt_spec: PromptSpec,
        evidence: EvidenceBundle,
        **kwargs,
    ) -> tuple[ReasoningTrace, VerificationResult]:
        """
        Audit evidence and produce a reasoning trace.
        
        Args:
            prompt_spec: The prompt specification
            evidence: The collected evidence
            **kwargs: Additional configuration
        
        Returns:
            Tuple of (ReasoningTrace, VerificationResult)
        """
        ...


@runtime_checkable
class JudgeProtocol(Protocol):
    """Protocol for Judge agents."""
    
    def judge(
        self,
        prompt_spec: PromptSpec,
        evidence: EvidenceBundle,
        trace: ReasoningTrace,
        **kwargs,
    ) -> tuple[DeterministicVerdict, VerificationResult]:
        """
        Render a verdict based on evidence and reasoning.
        
        Args:
            prompt_spec: The prompt specification
            evidence: The collected evidence
            trace: The reasoning trace
            **kwargs: Additional configuration
        
        Returns:
            Tuple of (DeterministicVerdict, VerificationResult)
        """
        ...


@runtime_checkable
class SentinelProtocol(Protocol):
    """Protocol for Sentinel agents."""
    
    def verify(
        self,
        package: PoRPackage,
        *,
        mode: str = "verify",
    ) -> tuple[VerificationResult, list[dict[str, Any]]]:
        """
        Verify a PoR package.
        
        Args:
            package: The PoRPackage to verify
            mode: Verification mode ("verify" or "replay")
        
        Returns:
            Tuple of (VerificationResult, list of challenges)
        """
        ...


# =============================================================================
# Agent Loading Errors
# =============================================================================

class AgentConfigError(Exception):
    """Error in agent configuration."""
    pass


class AgentNotConfiguredError(AgentConfigError):
    """Agent is not configured."""
    pass


class AgentLoadError(AgentConfigError):
    """Failed to load agent."""
    pass


# =============================================================================
# HTTP Agent Implementations (Production)
# =============================================================================

class HTTPAgentBase(ABC):
    """Base class for HTTP-based agent implementations."""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.endpoint = config.endpoint
        self.api_key = config.api_key
        self.timeout = config.timeout
        
        if not self.endpoint:
            raise AgentConfigError(f"Endpoint not configured for {self.__class__.__name__}")
    
    def _make_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Make HTTP request to agent endpoint."""
        import urllib.request
        import urllib.error
        
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint,
            data=data,
            headers=headers,
            method="POST",
        )
        
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else ""
            raise AgentLoadError(f"HTTP {e.code}: {error_body}") from e
        except urllib.error.URLError as e:
            raise AgentLoadError(f"Connection error: {e.reason}") from e
        except TimeoutError:
            raise AgentLoadError(f"Request timed out after {self.timeout}s")


class HTTPPromptEngineer(HTTPAgentBase):
    """HTTP-based Prompt Engineer agent."""
    
    def run(self, user_input: str, **kwargs) -> tuple[PromptSpec, ToolPlan | None]:
        logger.info(f"Calling Prompt Engineer at {self.endpoint}")
        
        response = self._make_request({
            "user_input": user_input,
            **kwargs,
        })
        
        prompt_spec = PromptSpec.model_validate(response["prompt_spec"])
        tool_plan = None
        if "tool_plan" in response and response["tool_plan"]:
            tool_plan = ToolPlan.model_validate(response["tool_plan"])
        
        return prompt_spec, tool_plan


class HTTPCollector(HTTPAgentBase):
    """HTTP-based Evidence Collector agent."""
    
    def collect(
        self,
        prompt_spec: PromptSpec,
        tool_plan: ToolPlan | None,
        **kwargs,
    ) -> EvidenceBundle:
        logger.info(f"Calling Collector at {self.endpoint}")
        
        payload = {
            "prompt_spec": prompt_spec.model_dump(mode="json"),
            **kwargs,
        }
        if tool_plan:
            payload["tool_plan"] = tool_plan.model_dump(mode="json")
        
        response = self._make_request(payload)
        return EvidenceBundle.model_validate(response["evidence_bundle"])


class HTTPAuditor(HTTPAgentBase):
    """HTTP-based Auditor agent."""
    
    def audit(
        self,
        prompt_spec: PromptSpec,
        evidence: EvidenceBundle,
        **kwargs,
    ) -> tuple[ReasoningTrace, VerificationResult]:
        logger.info(f"Calling Auditor at {self.endpoint}")
        
        response = self._make_request({
            "prompt_spec": prompt_spec.model_dump(mode="json"),
            "evidence_bundle": evidence.model_dump(mode="json"),
            **kwargs,
        })
        
        trace = ReasoningTrace.model_validate(response["reasoning_trace"])
        verification = VerificationResult.model_validate(response["verification"])
        return trace, verification


class HTTPJudge(HTTPAgentBase):
    """HTTP-based Judge agent."""
    
    def judge(
        self,
        prompt_spec: PromptSpec,
        evidence: EvidenceBundle,
        trace: ReasoningTrace,
        **kwargs,
    ) -> tuple[DeterministicVerdict, VerificationResult]:
        logger.info(f"Calling Judge at {self.endpoint}")
        
        response = self._make_request({
            "prompt_spec": prompt_spec.model_dump(mode="json"),
            "evidence_bundle": evidence.model_dump(mode="json"),
            "reasoning_trace": trace.model_dump(mode="json"),
            **kwargs,
        })
        
        verdict = DeterministicVerdict.model_validate(response["verdict"])
        verification = VerificationResult.model_validate(response["verification"])
        return verdict, verification


class HTTPSentinel(HTTPAgentBase):
    """HTTP-based Sentinel agent."""
    
    def verify(
        self,
        package: PoRPackage,
        *,
        mode: str = "verify",
    ) -> tuple[VerificationResult, list[dict[str, Any]]]:
        logger.info(f"Calling Sentinel at {self.endpoint} (mode={mode})")
        
        response = self._make_request({
            "por_bundle": package.bundle.model_dump(mode="json"),
            "prompt_spec": package.prompt_spec.model_dump(mode="json"),
            "evidence_bundle": package.evidence.model_dump(mode="json"),
            "reasoning_trace": package.trace.model_dump(mode="json"),
            "verdict": package.verdict.model_dump(mode="json"),
            "mode": mode,
        })
        
        verification = VerificationResult.model_validate(response["verification"])
        challenges = response.get("challenges", [])
        return verification, challenges


# =============================================================================
# Local Sentinel (uses built-in verification)
# =============================================================================

class LocalSentinel:
    """Local Sentinel using built-in PoR verification."""
    
    def __init__(self, config: AgentConfig | None = None):
        self.config = config
    
    def verify(
        self,
        package: PoRPackage,
        *,
        mode: str = "verify",
    ) -> tuple[VerificationResult, list[dict[str, Any]]]:
        from core.por.proof_of_reasoning import verify_por_bundle
        
        logger.info(f"Running local Sentinel verification (mode={mode})")
        
        result = verify_por_bundle(
            package.bundle,
            prompt_spec=package.prompt_spec,
            evidence=package.evidence,
            trace=package.trace,
        )
        
        challenges = []
        if not result.ok and result.challenge:
            challenges.append({
                "kind": result.challenge.kind,
                "reason": result.challenge.reason,
                "evidence_id": result.challenge.evidence_id,
                "step_id": result.challenge.step_id,
            })
        
        return result, challenges


# =============================================================================
# Agent Loading
# =============================================================================

def load_agent(agent_name: str, config: AgentConfig) -> Any:
    """
    Load an agent based on configuration.
    
    Args:
        agent_name: Name of the agent (prompt_engineer, collector, etc.)
        config: Agent configuration
    
    Returns:
        Agent instance
    
    Raises:
        AgentNotConfiguredError: If agent is not configured
        AgentLoadError: If agent fails to load
    """
    if not config.type:
        raise AgentNotConfiguredError(
            f"Agent '{agent_name}' is not configured. "
            f"Set COURNOT_{agent_name.upper()}_TYPE environment variable or configure in cournot.json"
        )
    
    agent_classes = {
        "prompt_engineer": {"http": HTTPPromptEngineer},
        "collector": {"http": HTTPCollector},
        "auditor": {"http": HTTPAuditor},
        "judge": {"http": HTTPJudge},
        "sentinel": {"http": HTTPSentinel, "local": LocalSentinel},
    }
    
    if agent_name not in agent_classes:
        raise AgentLoadError(f"Unknown agent: {agent_name}")
    
    type_map = agent_classes[agent_name]
    
    if config.type not in type_map:
        available = ", ".join(type_map.keys())
        raise AgentLoadError(
            f"Unknown type '{config.type}' for agent '{agent_name}'. "
            f"Available types: {available}"
        )
    
    agent_class = type_map[config.type]
    
    try:
        return agent_class(config)
    except Exception as e:
        raise AgentLoadError(f"Failed to initialize {agent_name}: {e}") from e


def load_all_agents(agents_config: PipelineAgentsConfig) -> dict[str, Any]:
    """
    Load all configured agents.
    
    Args:
        agents_config: Configuration for all agents
    
    Returns:
        Dictionary of agent name -> agent instance
    
    Raises:
        AgentNotConfiguredError: If required agents are not configured
    """
    agents = {}
    errors = []
    
    required_agents = ["prompt_engineer", "collector", "auditor", "judge"]
    optional_agents = ["sentinel"]
    
    for agent_name in required_agents:
        config = getattr(agents_config, agent_name)
        try:
            agents[agent_name] = load_agent(agent_name, config)
        except AgentNotConfiguredError as e:
            errors.append(str(e))
        except AgentLoadError as e:
            errors.append(f"Failed to load {agent_name}: {e}")
    
    for agent_name in optional_agents:
        config = getattr(agents_config, agent_name)
        try:
            if config.type:  # Only load if configured
                agents[agent_name] = load_agent(agent_name, config)
        except AgentLoadError as e:
            logger.warning(f"Optional agent {agent_name} failed to load: {e}")
    
    if errors:
        raise AgentNotConfiguredError(
            "Failed to load required agents:\n" + "\n".join(f"  - {e}" for e in errors)
        )
    
    return agents