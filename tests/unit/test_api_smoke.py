"""
Module 09D - API Smoke Tests

Tests for the FastAPI endpoints:
1. GET /health returns ok
2. POST /run returns JSON summary
3. POST /run return_format=pack_zip returns zip and headers
4. POST /verify with a known-good pack zip returns ok
5. Tampered pack returns ok=false and meaningful checks
"""

import io
import json
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

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

from orchestrator.pipeline import PoRPackage
from orchestrator.artifacts.io import save_pack_zip

from api.app import app


# Create test client
client = TestClient(app)


# =============================================================================
# Test Fixtures
# =============================================================================

def make_test_package(market_id: str = "mkt_api_test") -> PoRPackage:
    """Create a complete PoRPackage for testing."""
    market = MarketSpec(
        market_id=market_id,
        question="Test question?",
        event_definition="Test event",
        resolution_deadline=datetime(2027, 1, 1, tzinfo=timezone.utc),
        resolution_window=ResolutionWindow(
            start=datetime(2026, 12, 31, tzinfo=timezone.utc),
            end=datetime(2027, 1, 1, tzinfo=timezone.utc),
        ),
        resolution_rules=ResolutionRules(rules=[
            ResolutionRule(rule_id="R_TEST", description="Test rule", priority=1),
        ]),
        dispute_policy=DisputePolicy(dispute_window_seconds=3600),
    )
    
    prompt_spec = PromptSpec(
        market=market,
        prediction_semantics=PredictionSemantics(
            target_entity="test_entity",
            predicate="test_predicate",
        ),
        data_requirements=[
            DataRequirement(
                requirement_id="req_001",
                description="Test requirement",
                source_targets=[
                    SourceTarget(
                        source_id="source_001",
                        uri="https://example.com/data",
                        method="GET",
                        expected_content_type="json",
                    ),
                ],
                selection_policy=SelectionPolicy(
                    strategy="single_best", min_sources=1, max_sources=1, quorum=1,
                ),
            ),
        ],
        created_at=None,
        extra={"strict_mode": True},
    )
    
    tool_plan = ToolPlan(
        plan_id=f"plan_{market_id}",
        requirements=["req_001"],
        sources=["source_001"],
    )
    
    evidence = EvidenceBundle(
        bundle_id=f"bundle_{market_id}",
        collector_id="test_collector",
        items=[
            EvidenceItem(
                evidence_id="ev_001",
                requirement_id="req_001",
                source=SourceDescriptor(
                    source_id="source_001",
                    uri="https://example.com/data",
                ),
                retrieval=RetrievalReceipt(
                    retrieved_at=None,
                    method="http_get",
                    request_fingerprint="0x" + "a1" * 32,
                    response_fingerprint="0x" + "b2" * 32,
                ),
                provenance=ProvenanceProof(tier=1, kind="signature"),
                content_type="json",
                content={"test": "data"},
                confidence=0.9,
            ),
        ],
    )
    
    trace = ReasoningTrace(
        trace_id=f"trace_{market_id}",
        policy=TracePolicy(max_steps=100),
        steps=[
            ReasoningStep(
                step_id="step_0001",
                type="extract",
                action="Test action",
                inputs={"evidence_id": "ev_001"},
                output={"result": True},
                evidence_ids=["ev_001"],
            ),
        ],
        evidence_refs=["ev_001"],
    )
    
    verdict = DeterministicVerdict(
        market_id=market_id,
        outcome="YES",
        confidence=0.85,
        resolution_time=None,
        resolution_rule_id="R_TEST",
    )
    
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


def create_test_pack_zip() -> bytes:
    """Create a valid pack zip file for testing."""
    package = make_test_package()
    
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    
    try:
        save_pack_zip(package, tmp_path)
        return tmp_path.read_bytes()
    finally:
        tmp_path.unlink(missing_ok=True)


def create_tampered_pack_zip() -> bytes:
    """Create a tampered pack zip file for testing."""
    package = make_test_package()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        pack_path = Path(tmpdir) / "pack.zip"
        save_pack_zip(package, pack_path)
        
        # Extract, tamper, repack
        extract_dir = Path(tmpdir) / "extracted"
        extract_dir.mkdir()
        
        with zipfile.ZipFile(pack_path, "r") as zf:
            zf.extractall(extract_dir)
        
        # Tamper with verdict
        verdict_file = extract_dir / "verdict.json"
        data = json.loads(verdict_file.read_text())
        data["outcome"] = "NO"  # Change outcome
        verdict_file.write_text(json.dumps(data))
        
        # Repack (manifest still has old hashes)
        tampered_path = Path(tmpdir) / "tampered.zip"
        with zipfile.ZipFile(tampered_path, "w") as zf:
            for f in extract_dir.iterdir():
                if f.is_file():
                    zf.write(f, f.name)
        
        return tampered_path.read_bytes()


# =============================================================================
# Health Endpoint Tests
# =============================================================================

class TestHealthEndpoint:
    """Test GET /health endpoint."""
    
    def test_health_returns_ok(self):
        """Health check returns ok status."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is True
        assert data["service"] == "cournot-protocol-api"
        assert data["version"] == "v1"


# =============================================================================
# Run Endpoint Tests
# =============================================================================

class TestRunEndpoint:
    """Test POST /run endpoint."""
    
    def test_run_returns_json_summary(self):
        """Run endpoint returns JSON summary with artifacts."""
        response = client.post(
            "/run",
            json={
                "user_input": "Will BTC exceed $100k?",
                "return_format": "json",
                "enable_sentinel_verify": False,
            },
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["ok"] is True
        assert "summary" in data
        assert "market_id" in data["summary"]
        assert "outcome" in data["summary"]
        assert "por_root" in data["summary"]
        assert "artifacts" in data
    
    def test_run_returns_artifacts(self):
        """Run endpoint includes full artifacts."""
        response = client.post(
            "/run",
            json={
                "user_input": "Test query",
                "return_format": "json",
                "enable_sentinel_verify": False,
            },
        )
        
        assert response.status_code == 200
        data = response.json()
        
        artifacts = data["artifacts"]
        assert "prompt_spec" in artifacts
        assert "evidence_bundle" in artifacts
        assert "reasoning_trace" in artifacts
        assert "verdict" in artifacts
        assert "por_bundle" in artifacts
    
    def test_run_with_sentinel_verification(self):
        """Run endpoint includes verification when enabled."""
        response = client.post(
            "/run",
            json={
                "user_input": "Test query",
                "return_format": "json",
                "enable_sentinel_verify": True,
            },
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "verification" in data
        assert data["verification"]["sentinel_ok"] is True
    
    def test_run_returns_pack_zip(self):
        """Run endpoint returns zip file when requested."""
        response = client.post(
            "/run",
            json={
                "user_input": "Test query for zip",
                "return_format": "pack_zip",
                "enable_sentinel_verify": False,
            },
        )
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/zip"
        assert "X-Market-Id" in response.headers
        assert "X-Por-Root" in response.headers
        assert "X-Outcome" in response.headers
        
        # Verify it's a valid zip
        zip_bytes = response.content
        assert zipfile.is_zipfile(io.BytesIO(zip_bytes))
    
    def test_run_zip_contains_artifacts(self):
        """Zip file contains expected artifact files."""
        response = client.post(
            "/run",
            json={
                "user_input": "Test query",
                "return_format": "pack_zip",
                "enable_sentinel_verify": False,
            },
        )
        
        assert response.status_code == 200
        
        with zipfile.ZipFile(io.BytesIO(response.content), "r") as zf:
            names = zf.namelist()
            assert "manifest.json" in names
            assert "prompt_spec.json" in names
            assert "verdict.json" in names
            assert "por_bundle.json" in names
    
    def test_run_with_include_checks(self):
        """Run endpoint includes checks when requested."""
        response = client.post(
            "/run",
            json={
                "user_input": "Test query",
                "return_format": "json",
                "enable_sentinel_verify": True,
                "include_checks": True,
            },
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Checks should be present in verification
        assert "verification" in data
    
    def test_run_empty_input_fails(self):
        """Run endpoint rejects empty input."""
        response = client.post(
            "/run",
            json={
                "user_input": "",
            },
        )
        
        assert response.status_code == 422  # Validation error


# =============================================================================
# Verify Endpoint Tests
# =============================================================================

class TestVerifyEndpoint:
    """Test POST /verify endpoint."""
    
    def test_verify_valid_pack_returns_ok(self):
        """Verify returns ok for valid pack."""
        pack_bytes = create_test_pack_zip()
        
        response = client.post(
            "/verify",
            files={"file": ("pack.zip", pack_bytes, "application/zip")},
            data={"enable_sentinel": "true"},
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["ok"] is True
        assert data["hashes_ok"] is True
        assert data["semantic_ok"] is True
        assert data["sentinel_ok"] is True
        assert "por_root" in data
        assert "market_id" in data
    
    def test_verify_tampered_pack_returns_failure(self):
        """Verify returns ok=false for tampered pack."""
        pack_bytes = create_tampered_pack_zip()
        
        response = client.post(
            "/verify",
            files={"file": ("pack.zip", pack_bytes, "application/zip")},
            data={"enable_sentinel": "false"},
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["ok"] is False
        assert data["hashes_ok"] is False
    
    def test_verify_with_checks(self):
        """Verify includes detailed checks when requested."""
        pack_bytes = create_test_pack_zip()
        
        response = client.post(
            "/verify",
            files={"file": ("pack.zip", pack_bytes, "application/zip")},
            data={"include_checks": "true", "enable_sentinel": "true"},
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "checks" in data
        assert len(data["checks"]) > 0
    
    def test_verify_no_sentinel(self):
        """Verify works without sentinel verification."""
        pack_bytes = create_test_pack_zip()
        
        response = client.post(
            "/verify",
            files={"file": ("pack.zip", pack_bytes, "application/zip")},
            data={"enable_sentinel": "false"},
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["sentinel_ok"] is None
    
    def test_verify_missing_file_fails(self):
        """Verify fails when no file is provided."""
        response = client.post("/verify")
        
        assert response.status_code == 422  # FastAPI validation error


# =============================================================================
# Replay Endpoint Tests
# =============================================================================

class TestReplayEndpoint:
    """Test POST /replay endpoint."""
    
    def test_replay_valid_pack(self):
        """Replay returns success for valid pack."""
        pack_bytes = create_test_pack_zip()
        
        response = client.post(
            "/replay",
            files={"file": ("pack.zip", pack_bytes, "application/zip")},
            data={"timeout_s": "30"},
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["ok"] is True
        assert data["replay_ok"] is True
        assert data["sentinel_ok"] is True
        assert "por_root" in data
        assert "market_id" in data
    
    def test_replay_with_checks(self):
        """Replay includes checks when requested."""
        pack_bytes = create_test_pack_zip()
        
        response = client.post(
            "/replay",
            files={"file": ("pack.zip", pack_bytes, "application/zip")},
            data={"include_checks": "true"},
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "checks" in data
    
    def test_replay_returns_divergence_info(self):
        """Replay returns divergence info when available."""
        pack_bytes = create_test_pack_zip()
        
        response = client.post(
            "/replay",
            files={"file": ("pack.zip", pack_bytes, "application/zip")},
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # For a valid pack, there should be no divergence
        assert data["replay_ok"] is True


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Test error handling across endpoints."""
    
    def test_invalid_json_returns_422(self):
        """Invalid JSON returns 422."""
        response = client.post(
            "/run",
            content="not json",
            headers={"Content-Type": "application/json"},
        )
        
        assert response.status_code == 422
    
    def test_missing_required_field_returns_422(self):
        """Missing required field returns 422."""
        response = client.post(
            "/run",
            json={},  # Missing user_input
        )
        
        assert response.status_code == 422


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for full workflow."""
    
    def test_run_then_verify_workflow(self):
        """Run pipeline then verify the resulting pack."""
        # Run pipeline and get pack
        run_response = client.post(
            "/run",
            json={
                "user_input": "Integration test query",
                "return_format": "pack_zip",
                "enable_sentinel_verify": False,
            },
        )
        
        assert run_response.status_code == 200
        pack_bytes = run_response.content
        
        # Verify the pack
        verify_response = client.post(
            "/verify",
            files={"file": ("pack.zip", pack_bytes, "application/zip")},
            data={"enable_sentinel": "true"},
        )
        
        assert verify_response.status_code == 200
        data = verify_response.json()
        
        assert data["ok"] is True
        assert data["hashes_ok"] is True
        assert data["semantic_ok"] is True
    
    def test_run_then_replay_workflow(self):
        """Run pipeline then replay the resulting pack."""
        # Run pipeline and get pack
        run_response = client.post(
            "/run",
            json={
                "user_input": "Replay test query",
                "return_format": "pack_zip",
                "enable_sentinel_verify": False,
            },
        )
        
        assert run_response.status_code == 200
        pack_bytes = run_response.content
        
        # Replay the pack
        replay_response = client.post(
            "/replay",
            files={"file": ("pack.zip", pack_bytes, "application/zip")},
        )
        
        assert replay_response.status_code == 200
        data = replay_response.json()
        
        assert data["ok"] is True
        assert data["replay_ok"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])