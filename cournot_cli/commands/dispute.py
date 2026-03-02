"""Module 09C - CLI Dispute (Off-chain)

NOTE: CLI must be a thin wrapper over the API-first implementation.
See: `api/dispute_verify.py`.
"""

from __future__ import annotations

import json
import sys
from argparse import Namespace

from api.dispute_verify import verify_pack


# Exit codes (match verify.py)
EXIT_SUCCESS = 0
EXIT_RUNTIME_ERROR = 1
EXIT_VERIFICATION_FAILED = 2


def _print_human_summary(report: dict) -> None:
    status = "OK" if report.get("ok") else "FAILED"
    print(f"dispute.verify: {status}", file=sys.stderr)
    if report.get("market_id"):
        print(f"market_id: {report.get('market_id')}", file=sys.stderr)
    print(f"por_root: {report.get('por_root')}", file=sys.stderr)
    if report.get("expected_por_root"):
        print(f"expected_por_root: {report.get('expected_por_root')}", file=sys.stderr)
    print(f"pack_uri: {report.get('pack_uri')}", file=sys.stderr)
    ch = report.get("challenge_ref")
    if ch:
        print(
            f"challenge_ref: kind={ch.get('kind')} leaf_index={ch.get('leaf_index')}",
            file=sys.stderr,
        )
    errs = report.get("errors") or []
    if errs:
        print("errors:", file=sys.stderr)
        for e in errs[:20]:
            print(f"  - {e}", file=sys.stderr)


def dispute_verify_cmd(args: Namespace) -> int:
    report = verify_pack(
        args.pack,
        expected_por_root=args.expected_por_root,
        pack_uri=args.pack_uri,
        baseline_pack=getattr(args, "baseline_pack", None),
    )

    # JSON output to stdout (stable contract)
    print(json.dumps(report, indent=2, sort_keys=True))

    if not args.quiet:
        _print_human_summary(report)

    if report.get("ok"):
        return EXIT_SUCCESS
    return EXIT_VERIFICATION_FAILED
