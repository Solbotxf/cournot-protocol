"""
Module 06 - Auditor Verification Submodule

Provides trace policy and logic sanity verification for the Auditor agent.

Public API:
- TracePolicyVerifier, verify_trace_policy: Trace structure validation
- LogicSanityChecker, check_logic_sanity: Logic and constraint validation
"""

from .logic_sanity import (
    LogicSanityChecker,
    check_logic_sanity,
)
from .trace_policy import (
    TracePolicyVerifier,
    verify_trace_policy,
)

__all__ = [
    # Trace policy verification
    "TracePolicyVerifier",
    "verify_trace_policy",
    # Logic sanity checking
    "LogicSanityChecker",
    "check_logic_sanity",
]