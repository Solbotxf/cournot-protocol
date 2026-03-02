from __future__ import annotations

import json
import hashlib
import zipfile
from pathlib import Path

import pytest

from api.dispute_verify import verify_pack
from core.por.proof_of_reasoning import build_por_bundle
from core.schemas import (
    DataRequirement,
    DeterministicVerdict,
    DisputePolicy,
    EvidenceBundle,
    EvidenceItem,
    EvidenceRef,
    MarketSpec,
    PredictionSemantics,
    PromptSpec,
    Provenance,
    ResolutionRule,
    ResolutionRules,
    ResolutionWindow,
    SelectionPolicy,
    SourceTarget,
    ToolPlan,
)
from core.por.reasoning_trace import ReasoningTrace, ReasoningStep
from orchestrator.artifacts.io import save_pack_zip
from orchestrator.pipeline import PoRPackage


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _make_valid_pack_zip(tmp_path: Path) -> Path:
    """Create a minimal valid pack zip with a correct PoRBundle."""

    market_id = "mk_dispute_test"

    from datetime import datetime, timezone

    now = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    prompt_spec = PromptSpec(
        market=MarketSpec(
            market_id=market_id,
            question="Will BTC be above $100k?",
            event_definition="price(BTC_USD) > 100000",
            resolution_deadline=now,
            resolution_window=ResolutionWindow(start=now, end=now),
            resolution_rules=ResolutionRules(
                rules=[ResolutionRule(rule_id="R1", description="rule", priority=100)]
            ),
            dispute_policy=DisputePolicy(dispute_window_seconds=1),
        ),
        prediction_semantics=PredictionSemantics(
            target_entity="bitcoin",
            predicate="price above threshold",
            threshold="100000",
        ),
        data_requirements=[
            DataRequirement(
                requirement_id="req_001",
                description="Get BTC price",
                source_targets=[
                    SourceTarget(source_id="coingecko", uri="https://example.com")
                ],
                selection_policy=SelectionPolicy(
                    strategy="single_best", min_sources=1, max_sources=1, quorum=1
                ),
            )
        ],
    )

    tool_plan = ToolPlan(
        plan_id="plan_001",
        requirements=["req_001"],
        sources=["coingecko"],
        min_provenance_tier=0,
    )

    evidence_bundle = EvidenceBundle(
        bundle_id="bundle_001",
        market_id=market_id,
        plan_id="plan_001",
    )
    evidence_bundle.add_item(
        EvidenceItem(
            evidence_id="ev_001",
            requirement_id="req_001",
            provenance=Provenance(
                source_id="coingecko",
                source_uri="https://example.com",
                tier=0,
            ),
            success=True,
            extracted_fields={"price_usd": 95000},
        )
    )

    trace = ReasoningTrace(
        trace_id="trace_001",
        steps=[
            ReasoningStep(
                step_id="step_0001",
                type="check",
                action="Compare price to threshold",
                evidence_ids=["ev_001"],
                output={"comparison": "95000 < 100000"},
            )
        ],
        evidence_refs=["ev_001"],
    )

    verdict = DeterministicVerdict(
        market_id=market_id,
        outcome="NO",
        confidence=0.7,
        resolution_rule_id="R1",
    )

    por_bundle = build_por_bundle(prompt_spec, evidence_bundle, trace, verdict)

    package = PoRPackage(
        bundle=por_bundle,
        prompt_spec=prompt_spec,
        tool_plan=tool_plan,
        evidence=evidence_bundle,
        trace=trace,
        verdict=verdict,
    )

    out_zip = tmp_path / "valid_pack.zip"
    save_pack_zip(package, out_zip)
    return out_zip


def _tamper_zip_file_and_update_manifest(
    *,
    src_zip: Path,
    dst_zip: Path,
    inner_path: str,
    mutate_fn,
) -> None:
    """Modify one JSON file in a pack zip and update manifest entry sha256/bytes.

    This simulates an attacker who updates the manifest but does NOT update PoR commitments.
    """
    with zipfile.ZipFile(src_zip, "r") as zin:
        files = {name: zin.read(name) for name in zin.namelist()}

    obj = json.loads(files[inner_path].decode("utf-8"))
    mutate_fn(obj)
    new_bytes = json.dumps(obj, separators=(",", ":"), sort_keys=True).encode("utf-8")

    # Update file content
    files[inner_path] = new_bytes

    # Update manifest
    manifest = json.loads(files["manifest.json"].decode("utf-8"))
    # Find the manifest key for this inner_path
    key_for_path = None
    for k, v in manifest.get("files", {}).items():
        if v.get("path") == inner_path:
            key_for_path = k
            break
    assert key_for_path is not None

    manifest["files"][key_for_path]["sha256"] = _sha256_hex(new_bytes)
    manifest["files"][key_for_path]["bytes"] = len(new_bytes)
    files["manifest.json"] = (
        json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )

    with zipfile.ZipFile(dst_zip, "w", zipfile.ZIP_DEFLATED) as zout:
        for name, data in files.items():
            zout.writestr(name, data)


def test_dispute_api_verify_valid_pack_ok(tmp_path: Path):
    pack = _make_valid_pack_zip(tmp_path)
    r = verify_pack(str(pack), expected_por_root=None)
    assert r["ok"] is True
    assert r["por_root"] is not None and r["por_root"].startswith("0x")
    assert r["pack_uri"] == str(pack)
    assert r["sentinel_version"]
    assert r["timestamp"]


def test_dispute_api_verify_tamper_evidence_leaf_kind(tmp_path: Path):
    pack = _make_valid_pack_zip(tmp_path)
    tampered = tmp_path / "tamper_evidence.zip"

    def mutate(o: dict):
        o["items"][0]["extracted_fields"]["price_usd"] = 123

    _tamper_zip_file_and_update_manifest(
        src_zip=pack,
        dst_zip=tampered,
        inner_path="evidence_bundle.json",
        mutate_fn=mutate,
    )

    r = verify_pack(str(tampered))
    assert r["ok"] is False
    assert r.get("challenge_ref") is not None
    assert r["challenge_ref"]["kind"] == "evidence_leaf"


def test_dispute_api_verify_tamper_reasoning_leaf_kind(tmp_path: Path):
    pack = _make_valid_pack_zip(tmp_path)
    tampered = tmp_path / "tamper_reasoning.zip"

    def mutate(o: dict):
        o["steps"][0]["output"]["comparison"] = "tampered"

    _tamper_zip_file_and_update_manifest(
        src_zip=pack,
        dst_zip=tampered,
        inner_path="reasoning_trace.json",
        mutate_fn=mutate,
    )

    r = verify_pack(str(tampered))
    assert r["ok"] is False
    assert r.get("challenge_ref") is not None
    assert r["challenge_ref"]["kind"] == "reasoning_leaf"


def test_dispute_api_verify_tamper_verdict_kind(tmp_path: Path):
    pack = _make_valid_pack_zip(tmp_path)
    tampered = tmp_path / "tamper_verdict.zip"

    def mutate(o: dict):
        o["outcome"] = "YES"

    _tamper_zip_file_and_update_manifest(
        src_zip=pack,
        dst_zip=tampered,
        inner_path="verdict.json",
        mutate_fn=mutate,
    )

    r = verify_pack(str(tampered))
    assert r["ok"] is False
    assert r.get("challenge_ref") is not None
    assert r["challenge_ref"]["kind"] == "verdict_hash"
