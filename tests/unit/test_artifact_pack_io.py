"""
Module 09B - Artifact Packaging IO Tests

Tests for save/load operations:
1. Save as dir, load back, objects round-trip
2. Save as zip, load back, objects round-trip
3. Tamper a JSON file → hash validation fails
"""

import json
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from core.por.por_bundle import PoRBundle
from core.por.reasoning_trace import ReasoningTrace, ReasoningStep, TracePolicy
from core.por.proof_of_reasoning import compute_roots
from core.schemas.evidence import (
    EvidenceBundle, EvidenceItem, SourceDescriptor,
    RetrievalReceipt, ProvenanceProof,
)
from core.schemas.market import (
    MarketSpec, ResolutionWindow, ResolutionRules, ResolutionRule, DisputePolicy,
)
from core.schemas.prompts import (
    PromptSpec, PredictionSemantics, DataRequirement,
    SourceTarget, SelectionPolicy,
)
from core.schemas.transport import ToolPlan
from core.schemas.verdict import DeterministicVerdict
from core.schemas.verification import CheckResult

from orchestrator.pipeline import PoRPackage
from orchestrator.artifacts.io import (
    save_pack_dir, save_pack_zip, save_pack,
    load_pack_dir, load_pack_zip, load_pack,
    load_manifest, validate_pack_files,
    PackHashMismatchError, PackMissingFileError, PackIOError,
    MANIFEST_FILE, VERDICT_FILE,
)
from orchestrator.artifacts.manifest import PackManifest, FORMAT_VERSION


# =============================================================================
# Test Fixtures
# =============================================================================

def make_market_spec(market_id: str = "mkt_test_pack") -> MarketSpec:
    now = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    return MarketSpec(
        market_id=market_id,
        question="Will BTC close above $100,000?",
        event_definition="BTC-USD daily close > 100000",
        resolution_deadline=datetime(2027, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        resolution_window=ResolutionWindow(
            start=datetime(2026, 12, 31, 0, 0, 0, tzinfo=timezone.utc),
            end=datetime(2027, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        ),
        resolution_rules=ResolutionRules(rules=[
            ResolutionRule(rule_id="R_BINARY", description="Binary decision", priority=1),
        ]),
        dispute_policy=DisputePolicy(dispute_window_seconds=3600),
    )


def make_prompt_spec(market_id: str = "mkt_test_pack") -> PromptSpec:
    return PromptSpec(
        market=make_market_spec(market_id),
        prediction_semantics=PredictionSemantics(
            target_entity="BTC-USD",
            predicate="close_price > threshold",
            threshold="100000",
            timeframe="2026-12-31",
        ),
        data_requirements=[
            DataRequirement(
                requirement_id="req_001",
                description="BTC price from exchange",
                source_targets=[
                    SourceTarget(
                        source_id="exchange",
                        uri="https://api.exchange.example.com/btc-usd",
                        method="GET",
                        expected_content_type="json",
                    ),
                ],
                selection_policy=SelectionPolicy(
                    strategy="single_best",
                    min_sources=1,
                    max_sources=1,
                    quorum=1,
                ),
            ),
        ],
        created_at=None,
        extra={"strict_mode": True},
    )


def make_tool_plan(market_id: str = "mkt_test_pack") -> ToolPlan:
    return ToolPlan(
        plan_id=f"plan_{market_id}",
        requirements=["req_001"],
        sources=["exchange"],
        min_provenance_tier=0,
    )


def make_evidence_bundle(market_id: str = "mkt_test_pack") -> EvidenceBundle:
    return EvidenceBundle(
        bundle_id=f"bundle_{market_id}",
        collector_id="collector_v1",
        collection_time=None,
        items=[
            EvidenceItem(
                evidence_id="ev_001",
                requirement_id="req_001",
                source=SourceDescriptor(
                    source_id="exchange",
                    uri="https://api.exchange.example.com/btc-usd",
                    provider="Example Exchange",
                ),
                retrieval=RetrievalReceipt(
                    retrieved_at=None,
                    method="http_get",
                    request_fingerprint="0x" + "a1" * 32,
                    response_fingerprint="0x" + "b2" * 32,
                ),
                provenance=ProvenanceProof(tier=1, kind="signature"),
                content_type="json",
                content={"price": 105000.0, "timestamp": "2026-12-31T23:59:59Z"},
                normalized={"numeric_value": 105000.0},
                confidence=0.95,
            ),
        ],
    )


def make_reasoning_trace(market_id: str = "mkt_test_pack") -> ReasoningTrace:
    return ReasoningTrace(
        trace_id=f"trace_{market_id}",
        policy=TracePolicy(max_steps=100),
        steps=[
            ReasoningStep(
                step_id="step_0001",
                type="extract",
                action="Extract price from evidence",
                inputs={"evidence_id": "ev_001"},
                output={"price": 105000.0},
                evidence_ids=["ev_001"],
            ),
            ReasoningStep(
                step_id="step_0002",
                type="check",
                action="Check price against threshold",
                inputs={"price": 105000.0, "threshold": 100000},
                output={"above_threshold": True},
                evidence_ids=["ev_001"],
                prior_step_ids=["step_0001"],
            ),
        ],
        evidence_refs=["ev_001"],
    )


def make_verdict(market_id: str = "mkt_test_pack", outcome: str = "YES") -> DeterministicVerdict:
    return DeterministicVerdict(
        market_id=market_id,
        outcome=outcome,
        confidence=0.85,
        resolution_time=None,
        resolution_rule_id="R_BINARY",
    )


def make_por_package(market_id: str = "mkt_test_pack") -> PoRPackage:
    """Create a complete PoRPackage for testing."""
    prompt_spec = make_prompt_spec(market_id)
    tool_plan = make_tool_plan(market_id)
    evidence = make_evidence_bundle(market_id)
    trace = make_reasoning_trace(market_id)
    verdict = make_verdict(market_id)
    
    # Compute roots and build bundle
    roots = compute_roots(prompt_spec, evidence, trace, verdict)
    
    bundle = PoRBundle(
        market_id=market_id,
        prompt_spec_hash=roots.prompt_spec_hash,
        evidence_root=roots.evidence_root,
        reasoning_root=roots.reasoning_root,
        verdict_hash=roots.verdict_hash,
        por_root=roots.por_root,
        verdict=verdict,
        created_at=datetime.now(timezone.utc),
    )
    
    return PoRPackage(
        bundle=bundle,
        prompt_spec=prompt_spec,
        tool_plan=tool_plan,
        evidence=evidence,
        trace=trace,
        verdict=verdict,
    )


# =============================================================================
# Directory Save/Load Tests
# =============================================================================

class TestSaveLoadDir:
    """Test saving and loading packs as directories."""
    
    def test_save_and_load_dir_roundtrip(self):
        """Save pack to dir, load it back, verify all fields match."""
        package = make_por_package()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            pack_path = Path(tmpdir) / "test_pack"
            
            # Save
            result_path = save_pack_dir(package, pack_path)
            assert result_path.is_dir()
            assert (result_path / MANIFEST_FILE).exists()
            
            # Load
            loaded = load_pack_dir(result_path)
            
            # Verify fields
            assert loaded.bundle.market_id == package.bundle.market_id
            assert loaded.bundle.por_root == package.bundle.por_root
            assert loaded.prompt_spec.market.market_id == package.prompt_spec.market.market_id
            assert loaded.verdict.outcome == package.verdict.outcome
            assert loaded.verdict.confidence == package.verdict.confidence
            assert len(loaded.evidence.items) == len(package.evidence.items)
            assert len(loaded.trace.steps) == len(package.trace.steps)
    
    def test_save_dir_creates_manifest(self):
        """Verify manifest is created with correct structure."""
        package = make_por_package()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            pack_path = Path(tmpdir) / "test_pack"
            save_pack_dir(package, pack_path)
            
            manifest = load_manifest(pack_path)
            
            assert manifest.format_version == FORMAT_VERSION
            assert manifest.market_id == package.bundle.market_id
            assert manifest.por_root == package.bundle.por_root
            assert "prompt_spec" in manifest.files
            assert "evidence_bundle" in manifest.files
            assert "reasoning_trace" in manifest.files
            assert "verdict" in manifest.files
            assert "por_bundle" in manifest.files
    
    def test_save_dir_with_tool_plan(self):
        """Verify tool_plan is saved when present."""
        package = make_por_package()
        assert package.tool_plan is not None
        
        with tempfile.TemporaryDirectory() as tmpdir:
            pack_path = Path(tmpdir) / "test_pack"
            save_pack_dir(package, pack_path)
            
            manifest = load_manifest(pack_path)
            assert "tool_plan" in manifest.files
            
            loaded = load_pack_dir(pack_path)
            assert loaded.tool_plan is not None
            assert loaded.tool_plan.plan_id == package.tool_plan.plan_id
    
    def test_save_dir_with_checks(self):
        """Verify checks are saved when provided."""
        package = make_por_package()
        checks = {
            "pipeline": [CheckResult.passed("test_check", "Test passed")],
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            pack_path = Path(tmpdir) / "test_pack"
            save_pack_dir(package, pack_path, include_checks=checks)
            
            manifest = load_manifest(pack_path)
            assert "checks_pipeline" in manifest.files
            assert (pack_path / "checks" / "pipeline_checks.json").exists()


# =============================================================================
# Zip Save/Load Tests
# =============================================================================

class TestSaveLoadZip:
    """Test saving and loading packs as zip files."""
    
    def test_save_and_load_zip_roundtrip(self):
        """Save pack to zip, load it back, verify all fields match."""
        package = make_por_package()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / "test_pack.zip"
            
            # Save
            result_path = save_pack_zip(package, zip_path)
            assert result_path.exists()
            assert zipfile.is_zipfile(result_path)
            
            # Load
            loaded = load_pack_zip(result_path)
            
            # Verify fields
            assert loaded.bundle.market_id == package.bundle.market_id
            assert loaded.bundle.por_root == package.bundle.por_root
            assert loaded.verdict.outcome == package.verdict.outcome
            assert len(loaded.evidence.items) == len(package.evidence.items)
    
    def test_zip_contains_all_files(self):
        """Verify zip contains all expected files."""
        package = make_por_package()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / "test_pack.zip"
            save_pack_zip(package, zip_path)
            
            with zipfile.ZipFile(zip_path, "r") as zf:
                names = zf.namelist()
                assert MANIFEST_FILE in names
                assert "prompt_spec.json" in names
                assert "evidence_bundle.json" in names
                assert "reasoning_trace.json" in names
                assert "verdict.json" in names
                assert "por_bundle.json" in names
                assert "tool_plan.json" in names


# =============================================================================
# Auto-detect Format Tests
# =============================================================================

class TestAutoDetectFormat:
    """Test auto-detection of pack format."""
    
    def test_save_pack_auto_dir(self):
        """save_pack without .zip extension creates directory."""
        package = make_por_package()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            pack_path = Path(tmpdir) / "test_pack"
            result_path = save_pack(package, pack_path)
            assert result_path.is_dir()
    
    def test_save_pack_auto_zip(self):
        """save_pack with .zip extension creates zip file."""
        package = make_por_package()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / "test_pack.zip"
            result_path = save_pack(package, zip_path)
            assert result_path.is_file()
            assert zipfile.is_zipfile(result_path)
    
    def test_load_pack_auto_dir(self):
        """load_pack auto-detects directory format."""
        package = make_por_package()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            pack_path = Path(tmpdir) / "test_pack"
            save_pack_dir(package, pack_path)
            
            loaded = load_pack(pack_path)
            assert loaded.bundle.market_id == package.bundle.market_id
    
    def test_load_pack_auto_zip(self):
        """load_pack auto-detects zip format."""
        package = make_por_package()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / "test_pack.zip"
            save_pack_zip(package, zip_path)
            
            loaded = load_pack(zip_path)
            assert loaded.bundle.market_id == package.bundle.market_id


# =============================================================================
# Hash Validation Tests
# =============================================================================

class TestHashValidation:
    """Test hash validation during load."""
    
    def test_validate_pack_files_success(self):
        """validate_pack_files returns ok for valid pack."""
        package = make_por_package()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            pack_path = Path(tmpdir) / "test_pack"
            save_pack_dir(package, pack_path)
            
            result = validate_pack_files(pack_path)
            assert result.ok is True
            assert all(c.ok for c in result.checks)
    
    def test_tampered_file_detected_dir(self):
        """Tampered JSON file is detected in directory pack."""
        package = make_por_package()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            pack_path = Path(tmpdir) / "test_pack"
            save_pack_dir(package, pack_path)
            
            # Tamper with verdict file
            verdict_path = pack_path / VERDICT_FILE
            data = json.loads(verdict_path.read_text())
            data["outcome"] = "NO"  # Change outcome
            verdict_path.write_text(json.dumps(data))
            
            # Validation should fail
            result = validate_pack_files(pack_path)
            assert result.ok is False
            
            # Find the failed check
            failed_checks = [c for c in result.checks if not c.ok]
            assert any("verdict" in c.check_id for c in failed_checks)
    
    def test_tampered_file_detected_zip(self):
        """Tampered JSON file is detected in zip pack."""
        package = make_por_package()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / "test_pack.zip"
            save_pack_zip(package, zip_path)
            
            # Extract, tamper, repack
            extract_dir = Path(tmpdir) / "extracted"
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)
            
            verdict_path = extract_dir / VERDICT_FILE
            data = json.loads(verdict_path.read_text())
            data["outcome"] = "NO"
            verdict_path.write_text(json.dumps(data))
            
            # Repack (but don't update manifest)
            tampered_zip = Path(tmpdir) / "tampered.zip"
            with zipfile.ZipFile(tampered_zip, "w") as zf:
                for f in extract_dir.iterdir():
                    zf.write(f, f.name)
            
            # Validation should fail
            result = validate_pack_files(tampered_zip)
            assert result.ok is False
    
    def test_load_with_hash_verification_fails(self):
        """Loading tampered pack with verify_hashes=True raises error."""
        package = make_por_package()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            pack_path = Path(tmpdir) / "test_pack"
            save_pack_dir(package, pack_path)
            
            # Tamper
            verdict_path = pack_path / VERDICT_FILE
            data = json.loads(verdict_path.read_text())
            data["outcome"] = "NO"
            verdict_path.write_text(json.dumps(data))
            
            # Load should raise
            with pytest.raises(PackHashMismatchError) as exc_info:
                load_pack_dir(pack_path, verify_hashes=True)
            
            assert "verdict" in exc_info.value.file_key
    
    def test_load_without_hash_verification_succeeds(self):
        """Loading tampered pack with verify_hashes=False succeeds."""
        package = make_por_package()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            pack_path = Path(tmpdir) / "test_pack"
            save_pack_dir(package, pack_path)
            
            # Tamper
            verdict_path = pack_path / VERDICT_FILE
            data = json.loads(verdict_path.read_text())
            data["outcome"] = "NO"
            verdict_path.write_text(json.dumps(data))
            
            # Load without verification
            loaded = load_pack_dir(pack_path, verify_hashes=False)
            assert loaded.verdict.outcome == "NO"  # Tampered value


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Test error handling for various failure cases."""
    
    def test_load_missing_manifest(self):
        """Loading pack without manifest raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pack_path = Path(tmpdir) / "empty_pack"
            pack_path.mkdir()
            
            with pytest.raises(PackMissingFileError) as exc_info:
                load_pack_dir(pack_path)
            
            assert exc_info.value.file_key == "manifest"
    
    def test_load_missing_required_file(self):
        """Loading pack with missing required file raises error."""
        package = make_por_package()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            pack_path = Path(tmpdir) / "test_pack"
            save_pack_dir(package, pack_path)
            
            # Delete verdict file
            (pack_path / VERDICT_FILE).unlink()
            
            with pytest.raises(PackMissingFileError) as exc_info:
                load_pack_dir(pack_path)
            
            assert "verdict" in exc_info.value.file_key
    
    def test_load_invalid_path(self):
        """Loading from non-existent path raises error."""
        with pytest.raises(PackIOError):
            load_pack_dir("/nonexistent/path")
    
    def test_load_file_as_dir(self):
        """Loading file with load_pack_dir raises error."""
        package = make_por_package()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / "test_pack.zip"
            save_pack_zip(package, zip_path)
            
            with pytest.raises(PackIOError):
                load_pack_dir(zip_path)


# =============================================================================
# Round-trip Consistency Tests
# =============================================================================

class TestRoundtripConsistency:
    """Test that round-tripped packages maintain consistency."""
    
    def test_deterministic_serialization(self):
        """Same package serializes to same bytes."""
        package = make_por_package()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            pack1 = Path(tmpdir) / "pack1"
            pack2 = Path(tmpdir) / "pack2"
            
            save_pack_dir(package, pack1)
            save_pack_dir(package, pack2)
            
            # Compare file contents
            for filename in ["prompt_spec.json", "verdict.json", "por_bundle.json"]:
                content1 = (pack1 / filename).read_bytes()
                content2 = (pack2 / filename).read_bytes()
                assert content1 == content2, f"{filename} differs between saves"
    
    def test_multiple_roundtrips(self):
        """Package survives multiple save/load cycles."""
        package = make_por_package()
        original_por_root = package.bundle.por_root
        
        with tempfile.TemporaryDirectory() as tmpdir:
            current = package
            for i in range(3):
                pack_path = Path(tmpdir) / f"pack_{i}"
                save_pack_dir(current, pack_path)
                current = load_pack_dir(pack_path)
            
            assert current.bundle.por_root == original_por_root
            assert current.verdict.outcome == package.verdict.outcome


if __name__ == "__main__":
    pytest.main([__file__, "-v"])