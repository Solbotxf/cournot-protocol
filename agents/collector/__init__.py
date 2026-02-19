"""
Collector Agent Module

Executes ToolPlans to collect evidence from external sources.

Usage:
    from agents.collector import collect_evidence, CollectorHTTP, CollectorMock
    
    # Using convenience function (auto-selects based on context)
    result = collect_evidence(ctx, prompt_spec, tool_plan)
    evidence_bundle, execution_log = result.output
    
    # Using specific agent
    collector = CollectorHTTP()
    result = collector.run(ctx, prompt_spec, tool_plan)
"""

from .agent import (
    CollectorHTTP,
    CollectorLLM,
    CollectorGraphRAG,
    CollectorMock,
    collect_evidence,
    get_collector,
)

from .pan_agent import PANCollectorAgent, PANCollectorConfig
from .pan_runtime import SearchAlgo, SearchConfig
from .gemini_grounded_agent import CollectorGeminiGrounded
from .crp_agent import CollectorCRP

from .engine import CollectionEngine

from .adapters import (
    CryptoExchangeAdapter,
    HttpAdapter,
    MockAdapter,
    PolymarketAdapter,
    SourceAdapter,
    get_adapter,
    register_adapter,
)

__all__ = [
    # Agents
    "CollectorHTTP",
    "CollectorLLM",
    "CollectorGraphRAG",
    "CollectorMock",
    "PANCollectorAgent",
    "CollectorGeminiGrounded",
    "CollectorCRP",
    # PAN config / runtime
    "PANCollectorConfig",
    "SearchAlgo",
    "SearchConfig",
    # Functions
    "collect_evidence",
    "get_collector",
    # Engine
    "CollectionEngine",
    # Adapters
    "SourceAdapter",
    "HttpAdapter",
    "PolymarketAdapter",
    "CryptoExchangeAdapter",
    "MockAdapter",
    "get_adapter",
    "register_adapter",
]
