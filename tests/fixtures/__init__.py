"""
Test fixtures package for Cournot protocol tests.

This package provides factory functions for creating test objects.
Organized into layers:
- common.py: Base factories shared by all modules
- por_fixtures.py: PoR bundle specific helpers (M03+, M08)
- judge_fixtures.py: Judge specific helpers (M07)

Usage:
    from tests.fixtures import make_prompt_spec, make_valid_por_package
    
    def test_something():
        prompt = make_prompt_spec(market_id="my_market")
        bundle, prompt, evidence, trace, verdict, tool_plan = make_valid_por_package()
"""

from .common import (
    make_market_spec,
    make_prompt_spec,
    make_evidence_bundle,
    make_evidence_item,
    make_reasoning_trace,
    make_reasoning_step,
)

from .por_fixtures import (
    make_verdict,
    make_tool_plan,
    make_valid_por_package,
    make_tampered_bundle,
)

from .judge_fixtures import (
    make_trace_with_eval_vars,
)

__all__ = [
    # Common
    "make_market_spec",
    "make_prompt_spec",
    "make_evidence_bundle",
    "make_evidence_item",
    "make_reasoning_trace",
    "make_reasoning_step",
    # PoR
    "make_verdict",
    "make_tool_plan",
    "make_valid_por_package",
    "make_tampered_bundle",
    # Judge
    "make_trace_with_eval_vars",
]