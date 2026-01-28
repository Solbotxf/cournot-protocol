"""
Module 08 - Validator/Sentinel: Replay Executor

Rebuilds evidence deterministically from PromptSpec + ToolPlan and compares
with the original bundle. Used for replay mode verification.

Owner: Protocol Verification Engineer
Module ID: M08
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol, TYPE_CHECKING

from core.crypto.hashing import hash_canonical, to_hex
from core.por.proof_of_reasoning import compute_evidence_root
from core.schemas.evidence import EvidenceBundle, EvidenceItem
from core.schemas.prompts import PromptSpec
from core.schemas.transport import ToolPlan
from core.schemas.verification import CheckResult, VerificationResult

if TYPE_CHECKING:
    # Avoid circular import - CollectorAgent interface
    pass


class CollectorProtocol(Protocol):
    """Protocol for collector agents used in replay."""

    def collect(
        self,
        prompt_spec: PromptSpec,
        tool_plan: ToolPlan,
    ) -> EvidenceBundle:
        """Collect evidence according to spec and plan."""
        ...


@dataclass
class ReplayResult:
    """Result of evidence replay operation."""
    replayed_bundle: Optional[EvidenceBundle]
    verification: VerificationResult
    divergent_items: list[dict[str, Any]]


class ReplayExecutor:
    """
    Executor for replaying evidence collection.

    Rebuilds evidence deterministically from PromptSpec + ToolPlan
    and compares with the original bundle.
    """

    def __init__(
        self,
        collector: Optional[CollectorProtocol] = None,
        *,
        enabled: bool = True,
    ) -> None:
        """
        Initialize the replay executor.

        Args:
            collector: Optional collector agent for evidence collection.
                      If None, replay will be disabled.
            enabled: Whether replay is enabled.
        """
        self._collector = collector
        self._enabled = enabled and (collector is not None)

    @property
    def is_enabled(self) -> bool:
        """Check if replay is enabled and possible."""
        return self._enabled and self._collector is not None

    def replay_evidence(
        self,
        prompt_spec: PromptSpec,
        tool_plan: ToolPlan,
    ) -> tuple[Optional[EvidenceBundle], VerificationResult]:
        """
        Replay evidence collection from PromptSpec + ToolPlan.

        Args:
            prompt_spec: The prompt specification with data requirements.
            tool_plan: The tool execution plan.

        Returns:
            Tuple of (replayed EvidenceBundle or None, VerificationResult).
        """
        checks: list[CheckResult] = []

        if not self.is_enabled:
            checks.append(CheckResult(
                check_id="replay_enabled",
                ok=False,
                severity="warn",
                message="Replay is disabled or collector not available",
                details={"enabled": self._enabled, "has_collector": self._collector is not None},
            ))
            return None, VerificationResult(ok=True, checks=checks)

        try:
            # Attempt replay
            assert self._collector is not None
            replayed_bundle = self._collector.collect(prompt_spec, tool_plan)

            checks.append(CheckResult(
                check_id="replay_collection",
                ok=True,
                severity="info",
                message="Evidence replay collection successful",
                details={"item_count": len(replayed_bundle.items)},
            ))

            return replayed_bundle, VerificationResult(ok=True, checks=checks)

        except Exception as e:
            checks.append(CheckResult(
                check_id="replay_collection",
                ok=False,
                severity="error",
                message=f"Evidence replay collection failed: {e}",
                details={"error": str(e), "error_type": type(e).__name__},
            ))
            return None, VerificationResult(ok=False, checks=checks)

    def compare_evidence(
        self,
        original: EvidenceBundle,
        replayed: EvidenceBundle,
    ) -> ReplayResult:
        """
        Compare original evidence bundle with replayed bundle.

        Args:
            original: The original evidence bundle from the PoR package.
            replayed: The replayed evidence bundle.

        Returns:
            ReplayResult with verification status and divergent items.
        """
        checks: list[CheckResult] = []
        divergent_items: list[dict[str, Any]] = []

        # Compare root hashes
        original_root = compute_evidence_root(original)
        replayed_root = compute_evidence_root(replayed)

        roots_match = original_root == replayed_root
        checks.append(CheckResult(
            check_id="evidence_root_match",
            ok=roots_match,
            severity="info" if roots_match else "error",
            message="Evidence roots match" if roots_match else "Evidence roots differ",
            details={
                "original_root": original_root,
                "replayed_root": replayed_root,
            },
        ))

        # Compare item counts
        count_match = len(original.items) == len(replayed.items)
        checks.append(CheckResult(
            check_id="item_count_match",
            ok=count_match,
            severity="info" if count_match else "warn",
            message="Item counts match" if count_match else "Item counts differ",
            details={
                "original_count": len(original.items),
                "replayed_count": len(replayed.items),
            },
        ))

        # Compare individual items by evidence_id
        original_by_id = {item.evidence_id: item for item in original.items}
        replayed_by_id = {item.evidence_id: item for item in replayed.items}

        # Find items that exist in original but not in replayed
        missing_ids = set(original_by_id.keys()) - set(replayed_by_id.keys())
        if missing_ids:
            checks.append(CheckResult(
                check_id="missing_items",
                ok=False,
                severity="error",
                message=f"Replay missing {len(missing_ids)} evidence items",
                details={"missing_ids": list(missing_ids)},
            ))
            for eid in missing_ids:
                divergent_items.append({
                    "evidence_id": eid,
                    "reason": "missing_in_replay",
                })

        # Find items that exist in replayed but not in original
        extra_ids = set(replayed_by_id.keys()) - set(original_by_id.keys())
        if extra_ids:
            checks.append(CheckResult(
                check_id="extra_items",
                ok=False,
                severity="error",
                message=f"Replay has {len(extra_ids)} extra evidence items",
                details={"extra_ids": list(extra_ids)},
            ))
            for eid in extra_ids:
                divergent_items.append({
                    "evidence_id": eid,
                    "reason": "extra_in_replay",
                })

        # Compare common items by fingerprint
        common_ids = set(original_by_id.keys()) & set(replayed_by_id.keys())
        fingerprint_mismatches = 0

        for eid in common_ids:
            orig_item = original_by_id[eid]
            repl_item = replayed_by_id[eid]

            # Compare response fingerprints if available
            orig_fp = getattr(orig_item.retrieval, 'response_fingerprint', None)
            repl_fp = getattr(repl_item.retrieval, 'response_fingerprint', None)

            if orig_fp and repl_fp and orig_fp != repl_fp:
                fingerprint_mismatches += 1
                divergent_items.append({
                    "evidence_id": eid,
                    "reason": "fingerprint_mismatch",
                    "original_fingerprint": orig_fp,
                    "replayed_fingerprint": repl_fp,
                })

            # Compare canonical hashes of items (excluding timestamps)
            orig_hash = to_hex(hash_canonical(orig_item))
            repl_hash = to_hex(hash_canonical(repl_item))

            if orig_hash != repl_hash and eid not in [d["evidence_id"] for d in divergent_items]:
                divergent_items.append({
                    "evidence_id": eid,
                    "reason": "content_hash_mismatch",
                    "original_hash": orig_hash,
                    "replayed_hash": repl_hash,
                })

        if fingerprint_mismatches > 0:
            checks.append(CheckResult(
                check_id="fingerprint_comparison",
                ok=False,
                severity="warn",
                message=f"{fingerprint_mismatches} items have different fingerprints",
                details={"mismatch_count": fingerprint_mismatches},
            ))

        # Overall result
        all_ok = roots_match and not divergent_items
        verification = VerificationResult(
            ok=all_ok,
            checks=checks,
        )

        return ReplayResult(
            replayed_bundle=replayed,
            verification=verification,
            divergent_items=divergent_items,
        )

    def replay_and_compare(
        self,
        prompt_spec: PromptSpec,
        tool_plan: ToolPlan,
        original_bundle: EvidenceBundle,
    ) -> ReplayResult:
        """
        Full replay workflow: collect new evidence and compare with original.

        Args:
            prompt_spec: The prompt specification.
            tool_plan: The tool execution plan.
            original_bundle: The original evidence bundle to compare against.

        Returns:
            ReplayResult with comparison results.
        """
        checks: list[CheckResult] = []

        # First, replay collection
        replayed_bundle, replay_result = self.replay_evidence(prompt_spec, tool_plan)
        checks.extend(replay_result.checks)

        if replayed_bundle is None:
            # Replay failed or disabled
            return ReplayResult(
                replayed_bundle=None,
                verification=VerificationResult(
                    ok=False,
                    checks=checks,
                ),
                divergent_items=[],
            )

        # Compare bundles
        compare_result = self.compare_evidence(original_bundle, replayed_bundle)
        checks.extend(compare_result.verification.checks)

        return ReplayResult(
            replayed_bundle=replayed_bundle,
            verification=VerificationResult(
                ok=compare_result.verification.ok,
                checks=checks,
            ),
            divergent_items=compare_result.divergent_items,
        )


__all__ = [
    "CollectorProtocol",
    "ReplayResult",
    "ReplayExecutor",
]