"""
Tests for the agent base infrastructure.
"""

from datetime import datetime, timezone


def test_receipt_models():
    """Test receipt model creation and hashing."""
    from core.receipts import Receipt, LLMReceipt, HTTPReceipt, ReceiptRef
    
    # Create a basic receipt
    receipt = Receipt(
        receipt_id="rc_test_123",
        kind="llm",
        request={"model": "gpt-4", "messages": [{"role": "user", "content": "hello"}]},
        response={"content": "Hello!"},
    )
    
    # Compute hashes
    receipt.compute_hashes()
    
    assert receipt.request_hash is not None
    assert receipt.request_hash.startswith("0x")
    assert receipt.response_hash is not None
    
    # Convert to ref
    ref = receipt.to_ref()
    assert isinstance(ref, ReceiptRef)
    assert ref.receipt_id == "rc_test_123"


def test_receipt_recorder():
    """Test receipt recorder functionality."""
    from core.receipts import ReceiptRecorder
    
    recorder = ReceiptRecorder()
    
    # Start LLM receipt
    receipt = recorder.start_llm_receipt(
        provider="openai",
        model="gpt-4",
        messages=[{"role": "user", "content": "test"}],
    )
    
    assert receipt.receipt_id is not None
    assert receipt.provider == "openai"
    assert receipt.timing.started_at is not None
    
    # Complete the receipt
    recorder.complete(
        receipt,
        response={"content": "response"},
        input_tokens=10,
        output_tokens=5,
    )
    
    assert receipt.timing.ended_at is not None
    assert receipt.input_tokens == 10
    assert receipt.output_tokens == 5
    
    # Check recorded receipts
    receipts = recorder.get_receipts()
    assert len(receipts) == 1


def test_decoding_policy():
    """Test LLM decoding policy."""
    from core.llm.determinism import DecodingPolicy, policy_to_provider_args
    
    policy = DecodingPolicy(temperature=0.0, seed=42)
    
    assert policy.temperature == 0.0
    assert policy.seed == 42
    
    # Test with_temperature
    new_policy = policy.with_temperature(0.5)
    assert new_policy.temperature == 0.5
    assert new_policy.seed == 42  # Preserved
    
    # Test provider args conversion
    args = policy_to_provider_args(policy, "openai")
    assert args["temperature"] == 0.0
    assert args["seed"] == 42


def test_mock_provider():
    """Test mock LLM provider."""
    from core.llm import MockProvider, LLMClient
    from core.llm.determinism import DecodingPolicy
    
    provider = MockProvider(responses=['{"answer": "test"}'])
    client = LLMClient(provider)
    
    response = client.chat([{"role": "user", "content": "test"}])
    
    assert response.content == '{"answer": "test"}'
    assert response.provider == "mock"
    assert response.input_tokens == 100  # Mock default
    
    # Check that call was recorded
    assert len(provider.calls) == 1


def test_runtime_config():
    """Test runtime configuration."""
    from core.config import RuntimeConfig
    
    config = RuntimeConfig()
    
    assert config.llm.provider == "openai"
    assert config.pipeline.strict_mode == True
    assert config.pipeline.deterministic_timestamps == True
    
    # Test from dict
    config = RuntimeConfig.from_dict({
        "llm": {"provider": "anthropic", "model": "claude-3"},
        "pipeline": {"strict_mode": False},
    })
    
    assert config.llm.provider == "anthropic"
    assert config.pipeline.strict_mode == False


def test_agent_base():
    """Test agent base class."""
    from agents.base import BaseAgent, AgentCapability, AgentResult
    
    class TestAgent(BaseAgent):
        _name = "TestAgent"
        _version = "v1"
        _capabilities = {AgentCapability.DETERMINISTIC}
        
        def run(self, ctx):
            return AgentResult(output="test")
    
    agent = TestAgent()
    
    assert agent.name == "TestAgent"
    assert agent.version == "v1"
    assert agent.is_deterministic
    assert not agent.uses_llm


def test_agent_context_minimal():
    """Test minimal agent context creation."""
    from agents.context import AgentContext, FrozenClock
    
    ctx = AgentContext.create_minimal()
    
    assert ctx.recorder is not None
    assert isinstance(ctx.clock, FrozenClock)
    
    # Check frozen time
    t1 = ctx.now()
    t2 = ctx.now()
    assert t1 == t2


def test_agent_context_mock():
    """Test mock agent context creation."""
    from agents.context import AgentContext
    
    ctx = AgentContext.create_mock(
        llm_responses=['{"result": "mocked"}']
    )
    
    assert ctx.llm is not None
    
    # Test LLM call
    response = ctx.llm.chat([{"role": "user", "content": "test"}])
    assert response.content == '{"result": "mocked"}'
    
    # Check receipt was recorded
    receipts = ctx.get_receipts()
    assert len(receipts) == 1


def test_agent_registry():
    """Test agent registry."""
    from agents.registry import AgentRegistry
    from agents.base import BaseAgent, AgentCapability, AgentStep, AgentResult
    from agents.context import AgentContext
    
    class DummyAgent(BaseAgent):
        _name = "DummyAgent"
        _version = "v1"
        _capabilities = {AgentCapability.DETERMINISTIC}
    
    registry = AgentRegistry()
    
    # Register an agent
    registry.register(
        step=AgentStep.PROMPT_ENGINEER,
        name="DummyAgent",
        factory=lambda ctx: DummyAgent(),
        capabilities={AgentCapability.DETERMINISTIC},
    )
    
    # Check registration
    assert registry.has_agent("DummyAgent")
    assert registry.has_step(AgentStep.PROMPT_ENGINEER)
    
    # Get agent
    ctx = AgentContext.create_minimal()
    agent = registry.get_agent(AgentStep.PROMPT_ENGINEER, ctx)
    
    assert agent.name == "DummyAgent"


def test_canonical_hashing():
    """Test canonical hashing functions."""
    from core.crypto import hash_canonical, to_hex, from_hex
    
    obj = {"b": 2, "a": 1}
    
    h = hash_canonical(obj)
    assert len(h) == 32  # SHA-256
    
    hex_str = to_hex(h)
    assert hex_str.startswith("0x")
    assert len(hex_str) == 66  # 0x + 64 hex chars
    
    # Round trip
    decoded = from_hex(hex_str)
    assert decoded == h


if __name__ == "__main__":
    # Run tests
    test_receipt_models()
    print("✓ test_receipt_models")
    
    test_receipt_recorder()
    print("✓ test_receipt_recorder")
    
    test_decoding_policy()
    print("✓ test_decoding_policy")
    
    test_mock_provider()
    print("✓ test_mock_provider")
    
    test_runtime_config()
    print("✓ test_runtime_config")
    
    test_agent_base()
    print("✓ test_agent_base")
    
    test_agent_context_minimal()
    print("✓ test_agent_context_minimal")
    
    test_agent_context_mock()
    print("✓ test_agent_context_mock")
    
    test_agent_registry()
    print("✓ test_agent_registry")
    
    test_canonical_hashing()
    print("✓ test_canonical_hashing")
    
    print("\nAll tests passed!")
