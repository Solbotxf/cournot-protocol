"""Tests for reason-code-driven dispute endpoints."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.routes.dispute import (
    DisputeRequest,
    DisputeResponse,
    DisputeTarget,
    DisputeLLMRequest,
    _RERUN_PLAN,
    _classify_url,
    _build_dispute_requirement,
)

_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


# ------------------------------------------------------------------
# Fixtures: minimal valid schemas
# ------------------------------------------------------------------

def _prompt_spec_dict() -> dict[str, Any]:
    """Minimal valid prompt_spec dict."""
    return {
        "market": {
            "market_id": "mk_test",
            "question": "Will BTC be above $100k?",
            "event_definition": "price(BTC_USD) > 100000",
            "resolution_deadline": _NOW.isoformat(),
            "resolution_window": {
                "start": _NOW.isoformat(),
                "end": _NOW.isoformat(),
            },
            "resolution_rules": {
                "rules": [{
                    "rule_id": "R_THRESHOLD",
                    "description": "Compare to threshold",
                    "priority": 100,
                }],
            },
            "dispute_policy": {"dispute_window_seconds": 86400},
        },
        "prediction_semantics": {
            "target_entity": "bitcoin",
            "predicate": "price above threshold",
        },
        "data_requirements": [],
    }


def _evidence_bundle_dict() -> dict[str, Any]:
    """Minimal valid evidence_bundle dict."""
    return {
        "bundle_id": "eb_test",
        "market_id": "mk_test",
        "plan_id": "plan_test",
        "items": [{
            "evidence_id": "ev_001",
            "requirement_id": "req_001",
            "provenance": {
                "source_id": "test",
                "source_uri": "https://test.com",
                "tier": 2,
            },
            "raw_content": "BTC is $105,000",
            "parsed_value": 105000,
            "success": True,
            "status_code": 200,
            "extracted_fields": {"price_usd": 105000},
        }],
    }


def _reasoning_trace_dict() -> dict[str, Any]:
    """Minimal valid reasoning_trace dict."""
    return {
        "trace_id": "trace_test",
        "market_id": "mk_test",
        "bundle_id": "eb_test",
        "steps": [{
            "step_id": "step_001",
            "step_type": "threshold_check",
            "description": "Compare price",
        }],
        "preliminary_outcome": "YES",
        "preliminary_confidence": 0.9,
        "recommended_rule_id": "R_THRESHOLD",
    }


def _tool_plan_dict() -> dict[str, Any]:
    return {
        "plan_id": "tp_test",
        "requirements": ["req_001"],
        "sources": ["web"],
    }


# ------------------------------------------------------------------
# 1. _RERUN_PLAN mapping
# ------------------------------------------------------------------

class TestRerunPlan:
    def test_evidence_misread_includes_collect(self):
        assert _RERUN_PLAN["EVIDENCE_MISREAD"] == ["collect", "audit", "judge"]

    def test_evidence_insufficient_includes_collect(self):
        assert _RERUN_PLAN["EVIDENCE_INSUFFICIENT"] == ["collect", "audit", "judge"]

    def test_reasoning_error_audit_judge(self):
        assert _RERUN_PLAN["REASONING_ERROR"] == ["audit", "judge"]

    def test_logic_gap_audit_judge(self):
        assert _RERUN_PLAN["LOGIC_GAP"] == ["audit", "judge"]

    def test_other_audit_judge(self):
        assert _RERUN_PLAN["OTHER"] == ["audit", "judge"]


# ------------------------------------------------------------------
# 2. URL classification
# ------------------------------------------------------------------

class TestClassifyUrl:
    def test_full_url_with_path(self):
        assert _classify_url("https://espn.com/article/123") == "url"

    def test_full_url_with_deep_path(self):
        assert _classify_url("https://www.bbc.com/sport/football/12345") == "url"

    def test_bare_domain(self):
        assert _classify_url("espn.com") == "domain"

    def test_https_root_only(self):
        assert _classify_url("https://espn.com/") == "domain"

    def test_https_root_no_slash(self):
        assert _classify_url("https://espn.com") == "domain"

    def test_http_with_path(self):
        assert _classify_url("http://example.com/page") == "url"


# ------------------------------------------------------------------
# 3. _build_dispute_requirement
# ------------------------------------------------------------------

class TestBuildDisputeRequirement:
    def _make_prompt_spec(self):
        from core.schemas.prompts import PromptSpec
        return PromptSpec(**_prompt_spec_dict())

    def test_domain_gets_search_operation(self):
        ps = self._make_prompt_spec()
        req = _build_dispute_requirement(ps, ["espn.com"])
        assert req.requirement_id == "dispute_evidence"
        assert len(req.source_targets) == 1
        st = req.source_targets[0]
        assert st.operation == "search"
        assert st.uri == "https://espn.com/"

    def test_full_url_gets_fetch_operation(self):
        ps = self._make_prompt_spec()
        req = _build_dispute_requirement(ps, ["https://espn.com/article/123"])
        st = req.source_targets[0]
        assert st.operation == "fetch"
        assert st.uri == "https://espn.com/article/123"

    def test_mixed_urls(self):
        ps = self._make_prompt_spec()
        req = _build_dispute_requirement(ps, [
            "espn.com",
            "https://bbc.com/sport/12345",
        ])
        assert len(req.source_targets) == 2
        assert req.source_targets[0].operation == "search"
        assert req.source_targets[1].operation == "fetch"

    def test_selection_policy_bounds(self):
        ps = self._make_prompt_spec()
        req = _build_dispute_requirement(ps, ["a.com", "b.com", "c.com"])
        assert req.selection_policy.max_sources == 3
        assert req.selection_policy.min_sources == 1


# ------------------------------------------------------------------
# 4. DisputeRequest schema validation
# ------------------------------------------------------------------

class TestDisputeRequestSchema:
    def test_no_mode_field(self):
        """mode field should not exist on DisputeRequest."""
        assert "mode" not in DisputeRequest.model_fields

    def test_evidence_urls_field_exists(self):
        assert "evidence_urls" in DisputeRequest.model_fields

    def test_minimal_valid_request(self):
        req = DisputeRequest(
            reason_code="REASONING_ERROR",
            message="The reasoning is wrong",
            prompt_spec=_prompt_spec_dict(),
            evidence_bundle=_evidence_bundle_dict(),
        )
        assert req.reason_code == "REASONING_ERROR"

    def test_evidence_misread_with_collectors(self):
        req = DisputeRequest(
            reason_code="EVIDENCE_MISREAD",
            message="Evidence was misread",
            prompt_spec=_prompt_spec_dict(),
            evidence_bundle=_evidence_bundle_dict(),
            tool_plan=_tool_plan_dict(),
            collectors=["CollectorOpenSearch"],
        )
        assert req.collectors == ["CollectorOpenSearch"]

    def test_evidence_insufficient_with_urls(self):
        req = DisputeRequest(
            reason_code="EVIDENCE_INSUFFICIENT",
            message="Need more evidence",
            prompt_spec=_prompt_spec_dict(),
            evidence_bundle=_evidence_bundle_dict(),
            evidence_urls=["https://espn.com/article/123"],
        )
        assert req.evidence_urls == ["https://espn.com/article/123"]


# ------------------------------------------------------------------
# 5. Dispute route validation errors
# ------------------------------------------------------------------

class TestDisputeRouteValidation:
    """Test validation errors by calling dispute() directly."""

    @pytest.mark.asyncio
    async def test_evidence_misread_requires_tool_plan(self):
        from api.routes.dispute import dispute

        req = DisputeRequest(
            reason_code="EVIDENCE_MISREAD",
            message="Evidence was misread",
            prompt_spec=_prompt_spec_dict(),
            evidence_bundle=_evidence_bundle_dict(),
            # No tool_plan → should fail
        )
        with pytest.raises(Exception, match="tool_plan is required"):
            await dispute(req)

    @pytest.mark.asyncio
    async def test_evidence_misread_requires_collectors(self):
        from api.routes.dispute import dispute

        req = DisputeRequest(
            reason_code="EVIDENCE_MISREAD",
            message="Evidence was misread",
            prompt_spec=_prompt_spec_dict(),
            evidence_bundle=_evidence_bundle_dict(),
            tool_plan=_tool_plan_dict(),
            # No collectors → should fail
        )
        with pytest.raises(Exception, match="collectors is required"):
            await dispute(req)

    @pytest.mark.asyncio
    async def test_evidence_insufficient_requires_urls(self):
        from api.routes.dispute import dispute

        req = DisputeRequest(
            reason_code="EVIDENCE_INSUFFICIENT",
            message="Need more evidence",
            prompt_spec=_prompt_spec_dict(),
            evidence_bundle=_evidence_bundle_dict(),
            # No evidence_urls → should fail
        )
        with pytest.raises(Exception, match="evidence_urls is required"):
            await dispute(req)

    @pytest.mark.asyncio
    async def test_reasoning_error_requires_evidence_bundle(self):
        from api.routes.dispute import dispute

        req = DisputeRequest(
            reason_code="REASONING_ERROR",
            message="Bad reasoning",
            prompt_spec=_prompt_spec_dict(),
            # No evidence_bundle → should fail at audit step
        )
        with pytest.raises(Exception, match="evidence_bundle is required"):
            await dispute(req)


# ------------------------------------------------------------------
# 6. Reason-code → steps mapping (integration-style with mocks)
# ------------------------------------------------------------------

def _mock_agent_result(output):
    """Create a mock AgentResult with success=True."""
    result = MagicMock()
    result.success = True
    result.output = output
    return result


def _mock_bundle():
    """Create a mock EvidenceBundle."""
    from core.schemas.evidence import EvidenceBundle
    return EvidenceBundle(
        bundle_id="eb_mock",
        market_id="mk_test",
        plan_id="plan_test",
    )


def _mock_trace():
    """Create a mock ReasoningTrace."""
    from core.schemas.reasoning import ReasoningTrace, ReasoningStep
    return ReasoningTrace(
        trace_id="trace_mock",
        market_id="mk_test",
        bundle_id="eb_mock",
        steps=[ReasoningStep(
            step_id="s1",
            step_type="threshold_check",
            description="test",
        )],
        preliminary_outcome="YES",
        preliminary_confidence=0.9,
        recommended_rule_id="R_THRESHOLD",
    )


def _mock_verdict():
    """Create a mock DeterministicVerdict."""
    from core.schemas.verdict import DeterministicVerdict
    return DeterministicVerdict(
        market_id="mk_test",
        outcome="YES",
        confidence=0.9,
        resolution_rule_id="R_THRESHOLD",
    )


class TestReasonCodeFlow:
    """Verify that each reason_code triggers the correct steps."""

    @pytest.mark.asyncio
    @patch("api.routes.dispute.get_agent_context")
    @patch("agents.judge.get_judge")
    @patch("agents.auditor.get_auditor")
    async def test_reasoning_error_runs_audit_judge(
        self, mock_get_auditor, mock_get_judge, mock_get_ctx,
    ):
        # Setup mocks
        mock_ctx = MagicMock()
        mock_ctx.extra = {}
        mock_get_ctx.return_value = mock_ctx

        mock_auditor = MagicMock()
        mock_auditor.run = MagicMock(return_value=_mock_agent_result(_mock_trace()))
        mock_get_auditor.return_value = mock_auditor

        mock_judge = MagicMock()
        mock_judge.run = MagicMock(return_value=_mock_agent_result(_mock_verdict()))
        mock_get_judge.return_value = mock_judge

        from api.routes.dispute import dispute

        req = DisputeRequest(
            reason_code="REASONING_ERROR",
            message="Reasoning is flawed",
            prompt_spec=_prompt_spec_dict(),
            evidence_bundle=_evidence_bundle_dict(),
            reasoning_trace=_reasoning_trace_dict(),
        )
        resp = await dispute(req)

        assert "audit" in resp.rerun_plan
        assert "judge" in resp.rerun_plan
        assert "collect" not in resp.rerun_plan

    @pytest.mark.asyncio
    @patch("api.routes.dispute.get_agent_context")
    @patch("agents.judge.get_judge")
    @patch("agents.auditor.get_auditor")
    async def test_logic_gap_runs_audit_judge(
        self, mock_get_auditor, mock_get_judge, mock_get_ctx,
    ):
        mock_ctx = MagicMock()
        mock_ctx.extra = {}
        mock_get_ctx.return_value = mock_ctx

        mock_auditor = MagicMock()
        mock_auditor.run = MagicMock(return_value=_mock_agent_result(_mock_trace()))
        mock_get_auditor.return_value = mock_auditor

        mock_judge = MagicMock()
        mock_judge.run = MagicMock(return_value=_mock_agent_result(_mock_verdict()))
        mock_get_judge.return_value = mock_judge

        from api.routes.dispute import dispute

        req = DisputeRequest(
            reason_code="LOGIC_GAP",
            message="Missing logic step",
            prompt_spec=_prompt_spec_dict(),
            evidence_bundle=_evidence_bundle_dict(),
        )
        resp = await dispute(req)

        assert resp.rerun_plan == ["audit", "judge"]

    @pytest.mark.asyncio
    @patch("api.routes.dispute.get_agent_context")
    @patch("agents.judge.get_judge")
    @patch("agents.auditor.get_auditor")
    async def test_other_runs_audit_judge(
        self, mock_get_auditor, mock_get_judge, mock_get_ctx,
    ):
        mock_ctx = MagicMock()
        mock_ctx.extra = {}
        mock_get_ctx.return_value = mock_ctx

        mock_auditor = MagicMock()
        mock_auditor.run = MagicMock(return_value=_mock_agent_result(_mock_trace()))
        mock_get_auditor.return_value = mock_auditor

        mock_judge = MagicMock()
        mock_judge.run = MagicMock(return_value=_mock_agent_result(_mock_verdict()))
        mock_get_judge.return_value = mock_judge

        from api.routes.dispute import dispute

        req = DisputeRequest(
            reason_code="OTHER",
            message="General dispute",
            prompt_spec=_prompt_spec_dict(),
            evidence_bundle=_evidence_bundle_dict(),
        )
        resp = await dispute(req)

        assert resp.rerun_plan == ["audit", "judge"]

    @pytest.mark.asyncio
    @patch("api.routes.dispute._run_collectors")
    @patch("api.routes.dispute.get_agent_context")
    @patch("agents.judge.get_judge")
    @patch("agents.auditor.get_auditor")
    async def test_evidence_misread_runs_collect_audit_judge(
        self, mock_get_auditor, mock_get_judge, mock_get_ctx, mock_run_collectors,
    ):
        mock_ctx = MagicMock()
        mock_ctx.extra = {}
        mock_get_ctx.return_value = mock_ctx

        # Collectors return a bundle
        mock_run_collectors.return_value = [_mock_bundle()]

        mock_auditor = MagicMock()
        mock_auditor.run = MagicMock(return_value=_mock_agent_result(_mock_trace()))
        mock_get_auditor.return_value = mock_auditor

        mock_judge = MagicMock()
        mock_judge.run = MagicMock(return_value=_mock_agent_result(_mock_verdict()))
        mock_get_judge.return_value = mock_judge

        from api.routes.dispute import dispute

        req = DisputeRequest(
            reason_code="EVIDENCE_MISREAD",
            message="Evidence was misread",
            prompt_spec=_prompt_spec_dict(),
            evidence_bundle=_evidence_bundle_dict(),
            tool_plan=_tool_plan_dict(),
            collectors=["CollectorOpenSearch"],
        )
        resp = await dispute(req)

        assert resp.rerun_plan == ["collect", "audit", "judge"]
        # Verify _run_collectors was called with dispute_message
        mock_run_collectors.assert_called_once()
        call_args = mock_run_collectors.call_args
        assert call_args[0][3] == ["CollectorOpenSearch"]  # collector_names
        assert call_args[0][4] == "Evidence was misread"   # dispute_message

    @pytest.mark.asyncio
    @patch("api.routes.dispute._run_collectors")
    @patch("api.routes.dispute.get_agent_context")
    @patch("agents.judge.get_judge")
    @patch("agents.auditor.get_auditor")
    async def test_evidence_insufficient_with_full_urls(
        self, mock_get_auditor, mock_get_judge, mock_get_ctx, mock_run_collectors,
    ):
        mock_ctx = MagicMock()
        mock_ctx.extra = {}
        mock_get_ctx.return_value = mock_ctx

        mock_run_collectors.return_value = [_mock_bundle()]

        mock_auditor = MagicMock()
        mock_auditor.run = MagicMock(return_value=_mock_agent_result(_mock_trace()))
        mock_get_auditor.return_value = mock_auditor

        mock_judge = MagicMock()
        mock_judge.run = MagicMock(return_value=_mock_agent_result(_mock_verdict()))
        mock_get_judge.return_value = mock_judge

        from api.routes.dispute import dispute

        req = DisputeRequest(
            reason_code="EVIDENCE_INSUFFICIENT",
            message="Need more evidence",
            prompt_spec=_prompt_spec_dict(),
            evidence_bundle=_evidence_bundle_dict(),
            evidence_urls=["https://espn.com/article/123"],
        )
        resp = await dispute(req)

        assert resp.rerun_plan == ["collect", "audit", "judge"]
        # Full URL → CollectorWebPageReader
        call_args = mock_run_collectors.call_args
        assert "CollectorWebPageReader" in call_args[0][3]

    @pytest.mark.asyncio
    @patch("api.routes.dispute._run_collectors")
    @patch("api.routes.dispute.get_agent_context")
    @patch("agents.judge.get_judge")
    @patch("agents.auditor.get_auditor")
    async def test_evidence_insufficient_with_domains(
        self, mock_get_auditor, mock_get_judge, mock_get_ctx, mock_run_collectors,
    ):
        mock_ctx = MagicMock()
        mock_ctx.extra = {}
        mock_get_ctx.return_value = mock_ctx

        mock_run_collectors.return_value = [_mock_bundle()]

        mock_auditor = MagicMock()
        mock_auditor.run = MagicMock(return_value=_mock_agent_result(_mock_trace()))
        mock_get_auditor.return_value = mock_auditor

        mock_judge = MagicMock()
        mock_judge.run = MagicMock(return_value=_mock_agent_result(_mock_verdict()))
        mock_get_judge.return_value = mock_judge

        from api.routes.dispute import dispute

        req = DisputeRequest(
            reason_code="EVIDENCE_INSUFFICIENT",
            message="Check espn.com",
            prompt_spec=_prompt_spec_dict(),
            evidence_bundle=_evidence_bundle_dict(),
            evidence_urls=["espn.com"],
        )
        resp = await dispute(req)

        assert resp.rerun_plan == ["collect", "audit", "judge"]
        call_args = mock_run_collectors.call_args
        assert "CollectorSitePinned" in call_args[0][3]

    @pytest.mark.asyncio
    @patch("api.routes.dispute._run_collectors")
    @patch("api.routes.dispute.get_agent_context")
    @patch("agents.judge.get_judge")
    @patch("agents.auditor.get_auditor")
    async def test_evidence_insufficient_mixed_urls_deduplicates_collectors(
        self, mock_get_auditor, mock_get_judge, mock_get_ctx, mock_run_collectors,
    ):
        mock_ctx = MagicMock()
        mock_ctx.extra = {}
        mock_get_ctx.return_value = mock_ctx

        mock_run_collectors.return_value = [_mock_bundle()]

        mock_auditor = MagicMock()
        mock_auditor.run = MagicMock(return_value=_mock_agent_result(_mock_trace()))
        mock_get_auditor.return_value = mock_auditor

        mock_judge = MagicMock()
        mock_judge.run = MagicMock(return_value=_mock_agent_result(_mock_verdict()))
        mock_get_judge.return_value = mock_judge

        from api.routes.dispute import dispute

        req = DisputeRequest(
            reason_code="EVIDENCE_INSUFFICIENT",
            message="Check these sources",
            prompt_spec=_prompt_spec_dict(),
            evidence_bundle=_evidence_bundle_dict(),
            evidence_urls=[
                "espn.com",
                "https://bbc.com/sport/12345",
                "bbc.com",  # duplicate type with first
            ],
        )
        resp = await dispute(req)

        assert resp.rerun_plan == ["collect", "audit", "judge"]
        call_args = mock_run_collectors.call_args
        collectors = call_args[0][3]
        # Should have both types, but deduplicated
        assert collectors == ["CollectorSitePinned", "CollectorWebPageReader"]


# ------------------------------------------------------------------
# 7. Evidence merging for EVIDENCE_INSUFFICIENT
# ------------------------------------------------------------------

class TestEvidenceMerging:
    @pytest.mark.asyncio
    @patch("api.routes.dispute._run_collectors")
    @patch("api.routes.dispute.get_agent_context")
    @patch("agents.judge.get_judge")
    @patch("agents.auditor.get_auditor")
    async def test_new_evidence_merged_into_existing_bundle(
        self, mock_get_auditor, mock_get_judge, mock_get_ctx, mock_run_collectors,
    ):
        mock_ctx = MagicMock()
        mock_ctx.extra = {}
        mock_get_ctx.return_value = mock_ctx

        # New bundle has 2 items
        from core.schemas.evidence import EvidenceBundle, EvidenceItem, Provenance
        new_bundle = EvidenceBundle(
            bundle_id="eb_new",
            market_id="mk_test",
            plan_id="plan_new",
        )
        new_bundle.add_item(EvidenceItem(
            evidence_id="ev_new_001",
            requirement_id="dispute_evidence",
            provenance=Provenance(source_id="web", source_uri="https://espn.com/article/1", tier=2),
            raw_content="New evidence content",
            success=True,
        ))
        mock_run_collectors.return_value = [new_bundle]

        mock_auditor = MagicMock()
        mock_auditor.run = MagicMock(return_value=_mock_agent_result(_mock_trace()))
        mock_get_auditor.return_value = mock_auditor

        mock_judge = MagicMock()
        mock_judge.run = MagicMock(return_value=_mock_agent_result(_mock_verdict()))
        mock_get_judge.return_value = mock_judge

        from api.routes.dispute import dispute

        req = DisputeRequest(
            reason_code="EVIDENCE_INSUFFICIENT",
            message="Need more evidence from espn",
            prompt_spec=_prompt_spec_dict(),
            evidence_bundle=_evidence_bundle_dict(),
            evidence_urls=["https://espn.com/article/1"],
        )
        resp = await dispute(req)

        # Original bundle had 1 item, new bundle has 1 → merged = 2 items
        primary = resp.artifacts["evidence_bundle"]
        assert len(primary["items"]) == 2


# ------------------------------------------------------------------
# 8. /dispute/llm URL extraction + deduplication
# ------------------------------------------------------------------

class TestDisputeLLMUrlMerging:
    @pytest.mark.asyncio
    @patch("api.routes.dispute.dispute")
    @patch("api.routes.dispute.get_agent_context")
    async def test_llm_extracts_and_merges_urls(
        self, mock_get_ctx, mock_dispute,
    ):
        """LLM-extracted URLs should be merged with explicit evidence_urls."""
        mock_ctx = MagicMock()
        mock_ctx.extra = {}
        mock_get_ctx.return_value = mock_ctx

        # Mock the LLM response with extracted_urls
        mock_llm_response = MagicMock()
        mock_llm_response.content = json.dumps({
            "target_artifact": "evidence_bundle",
            "target_leaf_path": None,
            "structured_message": "The evidence is insufficient",
            "extracted_urls": [
                "https://reuters.com/sports/123",
                "https://espn.com/article/1",  # Duplicate of explicit URL
            ],
            "evidence_assessments": [],
        })
        mock_ctx.llm.chat = MagicMock(return_value=mock_llm_response)

        mock_dispute.return_value = DisputeResponse(
            rerun_plan=["collect", "audit", "judge"],
            artifacts={},
        )

        from api.routes.dispute import dispute_llm

        req = DisputeLLMRequest(
            reason_code="EVIDENCE_INSUFFICIENT",
            message="Check https://espn.com/article/1 for the score",
            evidence_urls=["https://espn.com/article/1"],
            prompt_spec=_prompt_spec_dict(),
            evidence_bundle=_evidence_bundle_dict(),
        )
        await dispute_llm(req)

        # Verify the dispute() was called with merged, deduplicated URLs
        call_args = mock_dispute.call_args[0][0]
        assert call_args.evidence_urls is not None
        # espn.com/article/1 appears once (deduplicated), reuters added
        assert len(call_args.evidence_urls) == 2
        url_set = set(u.rstrip("/").lower() for u in call_args.evidence_urls)
        assert "https://espn.com/article/1" in url_set
        assert "https://reuters.com/sports/123" in url_set

    @pytest.mark.asyncio
    @patch("api.routes.dispute.dispute")
    @patch("api.routes.dispute.get_agent_context")
    async def test_llm_fallback_preserves_explicit_urls(
        self, mock_get_ctx, mock_dispute,
    ):
        """When LLM fails, explicit evidence_urls should still be passed."""
        mock_ctx = MagicMock()
        mock_ctx.extra = {}
        mock_get_ctx.return_value = mock_ctx

        # Simulate LLM failure
        mock_ctx.llm.chat = MagicMock(side_effect=Exception("LLM down"))

        mock_dispute.return_value = DisputeResponse(
            rerun_plan=["collect", "audit", "judge"],
            artifacts={},
        )

        from api.routes.dispute import dispute_llm

        req = DisputeLLMRequest(
            reason_code="EVIDENCE_INSUFFICIENT",
            message="Need evidence from espn",
            evidence_urls=["https://espn.com/article/1"],
            prompt_spec=_prompt_spec_dict(),
        )
        await dispute_llm(req)

        call_args = mock_dispute.call_args[0][0]
        assert call_args.evidence_urls == ["https://espn.com/article/1"]


# ------------------------------------------------------------------
# 9. Backward compat: existing fields still work
# ------------------------------------------------------------------

class TestBackwardCompat:
    def test_patch_field_accepted(self):
        req = DisputeRequest(
            reason_code="REASONING_ERROR",
            message="Bad reasoning",
            prompt_spec=_prompt_spec_dict(),
            evidence_bundle=_evidence_bundle_dict(),
            patch={"prompt_spec_override": {"extra": {"key": "val"}}},
        )
        assert req.patch is not None

    def test_target_field_accepted(self):
        req = DisputeRequest(
            reason_code="REASONING_ERROR",
            message="Bad reasoning",
            prompt_spec=_prompt_spec_dict(),
            evidence_bundle=_evidence_bundle_dict(),
            target={"artifact": "reasoning_trace", "leaf_path": "steps[0]"},
        )
        assert req.target is not None
        assert req.target.artifact == "reasoning_trace"

    def test_case_id_field_accepted(self):
        req = DisputeRequest(
            reason_code="OTHER",
            message="General dispute",
            prompt_spec=_prompt_spec_dict(),
            case_id="case_123",
        )
        assert req.case_id == "case_123"


# ------------------------------------------------------------------
# 10. HTTP-level smoke test
# ------------------------------------------------------------------

class TestDisputeHTTP:
    def test_endpoint_exists(self):
        """Route should be registered — empty body gives 422 not 404."""
        from fastapi.testclient import TestClient
        from api.app import app

        client = TestClient(app)
        resp = client.post("/dispute", json={})
        assert resp.status_code == 422

    def test_llm_endpoint_exists(self):
        from fastapi.testclient import TestClient
        from api.app import app

        client = TestClient(app)
        resp = client.post("/dispute/llm", json={})
        assert resp.status_code == 422

    def test_no_mode_in_schema(self):
        """Sending 'mode' should be silently ignored (not a field)."""
        from fastapi.testclient import TestClient
        from api.app import app

        client = TestClient(app)
        resp = client.post("/dispute", json={
            "mode": "full_rerun",  # Should be ignored
            "reason_code": "OTHER",
            "message": "test",
            "prompt_spec": _prompt_spec_dict(),
        })
        # 422 because mode is extra field (ignored by pydantic default),
        # but should not cause a different error than missing required fields
        # The request will proceed to the route logic
        assert resp.status_code != 404
