"""
Pytest configuration and shared fixtures for Cournot protocol tests.

This conftest.py:
1. Adds project root to sys.path for imports
2. Provides commonly-used fixtures via pytest's autodiscovery
3. Configures pytest markers and settings
"""

import sys
from pathlib import Path

import pytest

# =============================================================================
# Path Setup - Must happen before any local imports
# =============================================================================

# Get the project root (parent of tests/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_TESTS_ROOT = Path(__file__).resolve().parent

# Add both project root and tests root to sys.path
for _path in [str(_PROJECT_ROOT), str(_TESTS_ROOT)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

# =============================================================================
# Import fixtures using importlib (more robust for pytest loading)
# =============================================================================

import importlib

# Import fixture modules
_common = importlib.import_module("fixtures.common")
_por = importlib.import_module("fixtures.por_fixtures")
_judge = importlib.import_module("fixtures.judge_fixtures")

# Extract factory functions
make_market_spec = _common.make_market_spec
make_prompt_spec = _common.make_prompt_spec
make_evidence_bundle = _common.make_evidence_bundle
make_evidence_item = _common.make_evidence_item
make_reasoning_trace = _common.make_reasoning_trace
make_reasoning_step = _common.make_reasoning_step

make_verdict = _por.make_verdict
make_tool_plan = _por.make_tool_plan
make_valid_por_package = _por.make_valid_por_package
make_tampered_bundle = _por.make_tampered_bundle

make_trace_with_eval_vars = _judge.make_trace_with_eval_vars


# =============================================================================
# Pytest Fixtures (autodiscovered by pytest)
# =============================================================================

@pytest.fixture
def market_spec():
    """Provide a default MarketSpec for tests."""
    return make_market_spec()


@pytest.fixture
def prompt_spec():
    """Provide a default PromptSpec for tests."""
    return make_prompt_spec()


@pytest.fixture
def evidence_bundle():
    """Provide a default EvidenceBundle for tests."""
    return make_evidence_bundle()


@pytest.fixture
def reasoning_trace():
    """Provide a default ReasoningTrace for tests."""
    return make_reasoning_trace()


@pytest.fixture
def verdict():
    """Provide a default DeterministicVerdict for tests."""
    return make_verdict()


@pytest.fixture
def valid_por_package():
    """Provide a complete valid PoR package (bundle + all artifacts)."""
    return make_valid_por_package()


@pytest.fixture
def por_bundle(valid_por_package):
    """Provide just the PoRBundle from a valid package."""
    bundle, _, _, _, _, _ = valid_por_package
    return bundle


# =============================================================================
# Pytest Configuration
# =============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


# =============================================================================
# Test Helpers (available to all tests via fixtures)
# =============================================================================

@pytest.fixture
def assert_check_passed():
    """Helper to assert a specific check passed in VerificationResult."""
    def _assert(result, check_id: str):
        checks = [c for c in result.checks if c.check_id == check_id]
        assert len(checks) == 1, f"Expected check '{check_id}' not found in {[c.check_id for c in result.checks]}"
        assert checks[0].ok, f"Check '{check_id}' failed: {checks[0].message}"
    return _assert


@pytest.fixture
def assert_check_failed():
    """Helper to assert a specific check failed in VerificationResult."""
    def _assert(result, check_id: str):
        checks = [c for c in result.checks if c.check_id == check_id]
        assert len(checks) == 1, f"Expected check '{check_id}' not found in {[c.check_id for c in result.checks]}"
        assert not checks[0].ok, f"Check '{check_id}' unexpectedly passed"
    return _assert


@pytest.fixture
def assert_has_challenge():
    """Helper to assert a specific challenge kind exists."""
    def _assert(challenges: list, kind: str):
        matching = [c for c in challenges if c.kind == kind]
        assert len(matching) >= 1, f"Expected challenge kind '{kind}' not found in {[c.kind for c in challenges]}"
        return matching[0]
    return _assert