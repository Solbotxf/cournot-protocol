# Cournot Protocol

**Deterministic, verifiable resolution for prediction markets using Proof of Reasoning (PoR).**

Cournot is a protocol for resolving prediction market questions with cryptographic verifiability. It produces a deterministic audit trail that can be independently verified, enabling trustless market resolution.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Directory Structure](#directory-structure)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [Module Documentation](#module-documentation)
- [Development Guide](#development-guide)
- [Testing](#testing)
- [CLI Reference](#cli-reference)
- [Contributing](#contributing)

## Overview

### What Problem Does Cournot Solve?

Prediction markets need trusted resolution. Traditional approaches rely on centralized oracles or manual arbitration. Cournot provides:

1. **Deterministic Resolution** - Same inputs always produce same outputs
2. **Verifiable Reasoning** - Every step is logged and can be audited
3. **Cryptographic Proofs** - Hash commitments enable on-chain verification
4. **Challenge Mechanism** - Invalid resolutions can be contested

### How It Works

```
User Question: "Will BTC be above $100k by end of 2025?"
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. Prompt Engineer                                              │
│     Compiles question into structured PromptSpec + ToolPlan     │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. Collector*                                                   │
│     Fetches evidence from data sources with receipts            │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. Auditor                                                      │
│     Analyzes evidence and builds reasoning trace                │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. Judge                                                        │
│     Produces final verdict (YES/NO/INVALID) with confidence     │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  5. Sentinel                                                     │
│     Verifies all artifacts and produces PoR bundle              │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
PoRBundle (on-chain commitment)
```

## Architecture

### Design Principles

1. **Registry-Based Agents** - All agents are registered and selected dynamically
2. **Capability-Based Selection** - Agents declare capabilities (LLM, NETWORK, etc.)
3. **Fail-Closed Production** - Production mode fails immediately on errors
4. **Fallback Development** - Development mode uses deterministic fallbacks
5. **Schema-First** - All artifacts are Pydantic models with validation
6. **Deterministic Hashing** - Canonical JSON serialization for consistent hashes

### Execution Modes

| Mode | Behavior | Use Case |
|------|----------|----------|
| `PRODUCTION` | Fail-closed, requires capabilities | Live deployments |
| `DEVELOPMENT` | Uses fallbacks, continues on errors | Local development |
| `TEST` | Deterministic mocks | Unit/integration tests |

### Agent System

Each pipeline step has multiple agent implementations:

| Step | Primary Agent | Fallback Agent | Primary Capability |
|------|--------------|----------------|-------------------|
| Prompt Engineer | `PromptEngineerLLM` | `PromptEngineerFallback` | LLM |
| Collector | `CollectorHTTP` | `CollectorMock` | NETWORK |
| Auditor | `AuditorLLM` | `AuditorRuleBased` | LLM |
| Judge | `JudgeLLM` | `JudgeRuleBased` | LLM |
| Sentinel | `SentinelStrict` | `SentinelBasic` | DETERMINISTIC |

## Directory Structure

```
cournot/
├── core/                           # Core infrastructure (no agent logic)
│   ├── schemas/                    # Pydantic models for all artifacts
│   │   ├── prompts.py             # PromptSpec, DataRequirement, ResolutionRule
│   │   ├── transport.py           # ToolPlan, SourceTarget
│   │   ├── evidence.py            # EvidenceBundle, EvidenceItem, Provenance
│   │   ├── reasoning.py           # ReasoningTrace, ReasoningStep
│   │   ├── verdict.py             # DeterministicVerdict
│   │   ├── verification.py        # VerificationResult, CheckResult
│   │   ├── proof.py               # ProofBundle, SentinelReport
│   │   ├── canonical.py           # Deterministic JSON serialization
│   │   └── versioning.py          # Schema version management
│   ├── por/                        # Proof of Reasoning
│   │   ├── por_bundle.py          # PoRBundle model
│   │   └── proof_of_reasoning.py  # Hash computation functions
│   ├── llm/                        # LLM client abstraction
│   │   ├── client.py              # LLMClient
│   │   ├── providers.py           # OpenAI, Anthropic, Google, Mock
│   │   └── determinism.py         # DecodingPolicy for reproducibility
│   ├── http/                       # HTTP client abstraction
│   │   └── client.py              # HttpClient (httpx/requests/urllib)
│   ├── receipts/                   # Audit logging
│   │   ├── models.py              # Receipt models
│   │   └── recorder.py            # ReceiptRecorder
│   ├── config/                     # Runtime configuration
│   │   └── runtime.py             # RuntimeConfig
│   └── crypto/                     # Cryptographic utilities
│       └── hashing.py             # SHA256 helpers
│
├── agents/                         # Agent implementations
│   ├── base.py                    # BaseAgent, AgentResult, AgentCapability
│   ├── registry.py                # AgentRegistry, agent registration
│   ├── context.py                 # AgentContext (shared resources)
│   ├── prompt_engineer/           # Module 04
│   │   ├── agent.py              # PromptEngineerLLM, PromptEngineerFallback
│   │   ├── llm_compiler.py       # LLM-based compilation
│   │   ├── fallback_compiler.py  # Rule-based compilation
│   │   └── prompts.py            # System prompts
│   ├── collector/                 # Module 05
│   │   ├── agent.py              # CollectorHTTP, CollectorMock
│   │   ├── engine.py             # Collection execution
│   │   └── adapters.py           # Source-specific adapters
│   ├── auditor/                   # Module 06
│   │   ├── agent.py              # AuditorLLM, AuditorRuleBased
│   │   ├── llm_reasoner.py       # LLM reasoning
│   │   ├── reasoner.py           # Rule-based reasoning
│   │   └── prompts.py            # System prompts
│   ├── judge/                     # Module 07
│   │   ├── agent.py              # JudgeLLM, JudgeRuleBased
│   │   ├── verdict_builder.py    # Verdict construction
│   │   └── prompts.py            # System prompts
│   └── sentinel/                  # Module 08
│       ├── agent.py              # SentinelStrict, SentinelBasic
│       └── engine.py             # Verification engine
│
├── orchestrator/                   # Pipeline orchestration
│   ├── pipeline.py                # Pipeline, PipelineConfig, ExecutionMode
│   ├── sop_executor.py            # SOPExecutor, PipelineState
│   ├── market_resolver.py         # High-level resolution API
│   └── artifacts/                 # Module 09B - Pack I/O
│       ├── manifest.py           # PackManifest
│       ├── io.py                 # save_pack, load_pack
│       └── pack.py               # validate_pack
│
├── cournot_cli/                    # Command-line interface
│   ├── main.py                    # Argument parsing, dispatch
│   ├── config.py                  # CLI configuration
│   └── commands/                  # Subcommands
│       ├── run.py                # Full pipeline execution
│       ├── steps.py              # Individual step execution
│       ├── verify.py             # Pack verification
│       ├── replay.py             # Evidence replay
│       └── pack.py               # Pack creation
│
└── tests/                          # Test suite
    ├── test_base_infrastructure.py
    ├── test_prompt_engineer.py
    ├── test_collector.py
    ├── test_auditor.py
    ├── test_judge.py
    ├── test_sentinel.py
    └── test_orchestrator.py
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/cournot-protocol/cournot-protocol.git
cd cournot

# Install dependencies
pip install -r requirement.txt
```

### Run Your First Resolution

```bash
# Using the CLI (development mode with fallback agents)
python -m cournot_cli --log-level DEBUG run "Will BTC close above 100000 USD on 2026-12-31 UTC? Use Coinbase."

# Save output as artifact pack
python -m cournot_cli run "Will ETH reach 5000?" --out result.zip

# Verify a pack
python -m cournot_cli verify result.zip
```


### Code to prompt:
```bash
code2prompt --exclude="*/venv/*" --exclude-from-tree .
```

### Programmatic Usage

```python
from orchestrator import create_pipeline, ExecutionMode

# Development mode (uses fallback agents when LLM unavailable)
pipeline = create_pipeline(mode=ExecutionMode.DEVELOPMENT)
result = pipeline.run("Will BTC be above $100k by end of 2025?")

print(f"Market ID: {result.market_id}")
print(f"Outcome: {result.outcome}")  # YES, NO, or INVALID
print(f"Confidence: {result.verdict.confidence}")
print(f"PoR Root: {result.por_bundle.por_root}")
```

### Production Mode with LLM

```python
from core.llm import create_llm_client
from agents import AgentContext
from orchestrator import create_production_pipeline

# Create LLM client
llm = create_llm_client(
    provider="anthropic",
    api_key="sk-ant-...",
    model="claude-sonnet-4-20250514",
)

# Create context with LLM
ctx = AgentContext.create_with_llm(llm)

# Run production pipeline (fails if LLM unavailable)
pipeline = create_production_pipeline(ctx, require_llm=True)
result = pipeline.run("Will the Fed raise rates in Q1 2026?")
```

## Core Concepts

### Artifacts

All pipeline outputs are **Pydantic models** with strict validation:

| Artifact | Description | Key Fields |
|----------|-------------|------------|
| `PromptSpec` | Compiled market specification | `market`, `data_requirements`, `resolution_rules` |
| `ToolPlan` | Evidence collection plan | `sources`, `requirements` |
| `EvidenceBundle` | Collected evidence | `items[]`, `bundle_id` |
| `ReasoningTrace` | Audit trail of reasoning | `steps[]`, `preliminary_outcome` |
| `DeterministicVerdict` | Final resolution | `outcome`, `confidence`, hash roots |
| `PoRBundle` | Cryptographic commitment | `por_root`, embedded verdict |

### Hash Commitments

Every artifact contributes to the **Proof of Reasoning (PoR)** root hash:

```
por_root = SHA256(
    prompt_spec_hash ||
    evidence_root ||
    reasoning_root ||
    verdict_hash
)
```

This single hash can be committed on-chain for trustless verification.

### Agent Capabilities

Agents declare what resources they need:

```python
class AgentCapability(Enum):
    LLM = "llm"              # Requires LLM client
    NETWORK = "network"      # Requires HTTP client
    DETERMINISTIC = "deterministic"  # Produces same output for same input
    REPLAY = "replay"        # Can replay from receipts
```

The registry selects agents based on available capabilities:

```python
# If LLM available → PromptEngineerLLM
# If no LLM → PromptEngineerFallback (deterministic)
agent = registry.get_agent(AgentStep.PROMPT_ENGINEER, context)
```

## Development Guide

### Setting Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install development dependencies
pip install -r requirement.txt

# Run tests
python -m pytest -q
```

### Adding a New Agent

1. **Create agent file** in the appropriate module:

```python
# agents/my_module/my_agent.py
from agents.base import BaseAgent, AgentResult, AgentCapability, AgentStep

class MyAgent(BaseAgent):
    name = "MyAgent"
    version = "v1"
    step = AgentStep.MY_STEP
    capabilities = {AgentCapability.DETERMINISTIC}
    priority = 100
    is_fallback = False
    
    def run(self, ctx: AgentContext, *args) -> AgentResult:
        # Implementation
        return AgentResult.success(output)
```

2. **Register the agent** in the module's `__init__.py`:

```python
from agents.registry import get_registry
from .my_agent import MyAgent

def _register():
    registry = get_registry()
    registry.register(MyAgent)

_register()
```

3. **Add tests** in `tests/test_my_module.py`

### Schema Changes

All schemas use **semantic versioning**:

```python
from core.schemas.versioning import SCHEMA_VERSION

class MyModel(BaseModel):
    schema_version: str = SCHEMA_VERSION  # "1.0.0"
```

When making breaking changes:
1. Increment major version
2. Add migration logic if needed
3. Update `is_compatible_version()` checks

### Canonical Serialization

For deterministic hashing, always use canonical JSON:

```python
from core.schemas.canonical import dumps_canonical

# Sorted keys, no whitespace, consistent formatting
json_bytes = dumps_canonical(model)
hash_hex = hashlib.sha256(json_bytes).hexdigest()
```

## Testing

### Running Tests

```bash
# All tests
python -m pytest tests/ -v

# Specific module
python -m pytest tests/test_orchestrator.py -v

# With coverage
python -m pytest tests/ --cov=. --cov-report=html
```

### Test Structure

```python
class TestMyAgent:
    def test_run_success(self):
        """Test successful execution."""
        ctx = AgentContext.create_minimal()
        agent = MyAgent()
        result = agent.run(ctx, input_data)
        
        assert result.success
        assert result.output is not None
    
    def test_handles_error(self):
        """Test error handling."""
        ctx = AgentContext.create_minimal()
        agent = MyAgent()
        result = agent.run(ctx, invalid_input)
        
        assert not result.success
        assert "error" in result.error.lower()
```

### Mock Providers

For testing without external dependencies:

```python
from core.llm import create_llm_client

# Create mock LLM
mock_llm = create_llm_client("mock")
mock_llm.provider.set_response('{"outcome": "YES"}')

# Use in context
ctx = AgentContext.create_with_llm(mock_llm)
```

## CLI Reference

### Commands

| Command | Description |
|---------|-------------|
| `run` | Execute full pipeline |
| `step` | Run individual steps |
| `verify` | Verify artifact pack |
| `replay` | Replay evidence collection |
| `pack` | Create pack from JSON files |
| `agents` | List registered agents |
| `config` | Manage configuration |

### Examples

```bash
# Full pipeline
cournot run "Will BTC hit 100k?" --out result.zip --json

# Individual steps
cournot step prompt "Query" --out ./step1
cournot step collect --prompt ./step1 --out ./step2
cournot step audit --prompt ./step1 --evidence ./step2 --out ./step3
cournot step judge --prompt ./step1 --evidence ./step2 --trace ./step3 --out ./step4
cournot step sentinel --prompt ./step1 --evidence ./step2 --trace ./step3 --verdict ./step4

# Verification
cournot verify result.zip --debug
cournot replay result.zip --timeout 60

# Agent info
cournot agents --step prompt_engineer --json
```

### Configuration

```bash
# Environment variables
export COURNOT_EXECUTION_MODE=production
export COURNOT_LLM_PROVIDER=anthropic
export COURNOT_LLM_API_KEY=sk-ant-...

# Or config file
cournot config --init  # Creates cournot.json template
```

## Contributing

### Code Style

- Python 3.12+
- Type hints required
- Docstrings for public APIs
- No external dependencies in `core/` (except pydantic)

### Pull Request Process

1. Fork the repository
2. Create feature branch (`git checkout -b feature/my-feature`)
3. Write tests for new functionality
4. Ensure all tests pass (`python -m pytest tests/ -v`)
5. Submit pull request

### Architecture Decisions

Key design decisions are documented in code comments. When proposing changes:

1. Explain the problem
2. Describe alternatives considered
3. Justify the chosen approach

## License

[License details here]

## Resources

- [Protocol Specification](docs/ARCHITECTURE.md)
- [Modules Reference](docs/modules)
- [Changes Reference](docs/changes)