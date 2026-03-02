"""Tests for the dispute API feature."""

import json

from agents.context import AgentContext
from agents.auditor.llm_reasoner import LLMReasoner
from datetime import datetime, timezone

from core.schemas import (
    DisputePolicy,
    EvidenceBundle,
    EvidenceItem,
    MarketSpec,
    PredictionSemantics,
    PromptSpec,
    Provenance,
    ReasoningStep,
    ReasoningTrace,
    ResolutionRule,
    ResolutionRules,
    ResolutionWindow,
)

_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _make_test_prompt_spec():
    return PromptSpec(
        market=MarketSpec(
            market_id="mk_test",
            question="Will BTC be above $100k?",
            event_definition="price(BTC_USD) > 100000",
            resolution_deadline=_NOW,
            resolution_window=ResolutionWindow(start=_NOW, end=_NOW),
            resolution_rules=ResolutionRules(rules=[
                ResolutionRule(rule_id="R_THRESHOLD", description="Compare to threshold", priority=100),
            ]),
            dispute_policy=DisputePolicy(dispute_window_seconds=86400),
        ),
        prediction_semantics=PredictionSemantics(
            target_entity="bitcoin", predicate="price above threshold",
        ),
        data_requirements=[],
    )


def _make_test_evidence_bundle():
    bundle = EvidenceBundle(
        bundle_id="eb_test",
        market_id="mk_test",
        plan_id="plan_test",
    )
    bundle.add_item(EvidenceItem(
        evidence_id="ev_001",
        requirement_id="req_001",
        provenance=Provenance(source_id="test", source_uri="https://test.com", tier=2),
        raw_content="BTC is $105,000",
        parsed_value=105000,
        success=True,
        status_code=200,
        extracted_fields={"price_usd": 105000},
    ))
    return bundle


def _good_trace_json(**overrides):
    """Build a valid reasoning trace JSON string."""
    data = {
        "trace_id": "trace_test",
        "evidence_summary": "BTC price is $105,000",
        "reasoning_summary": "Price is above threshold",
        "steps": [{
            "step_id": "step_001", "step_type": "threshold_check",
            "description": "Compare price", "evidence_refs": [],
            "output_summary": "105000 > 100000", "conclusion": "YES",
        }],
        "conflicts": [],
        "preliminary_outcome": "YES",
        "preliminary_confidence": 0.9,
        "recommended_rule_id": "R_THRESHOLD",
    }
    data.update(overrides)
    return json.dumps(data)


def _good_verdict_json(**overrides):
    """Build a valid verdict JSON string."""
    data = {
        "outcome": "YES",
        "confidence": 0.9,
        "resolution_rule_id": "R_THRESHOLD",
        "reasoning_valid": True,
        "reasoning_issues": [],
        "final_justification": "Price is above threshold",
    }
    data.update(overrides)
    return json.dumps(data)


class TestAuditorDisputeInjection:
    def test_dispute_context_appended_to_messages(self):
        """When dispute_context is in ctx.extra, the LLM should receive a dispute instruction message."""
        ctx = AgentContext.create_mock(llm_responses=[_good_trace_json()])
        ctx.extra["dispute_context"] = {
            "reason_code": "REASONING_STEP_INCORRECT",
            "message": "The price comparison is wrong",
            "leaf_path": None,
            "target_step": "audit",
            "patch": None,
        }
        reasoner = LLMReasoner()
        reasoner.reason(ctx, _make_test_prompt_spec(), [_make_test_evidence_bundle()])
        calls = ctx.llm.provider._calls
        assert len(calls) >= 1
        messages = calls[0]["messages"]
        assert len(messages) == 3  # system, user_prompt, dispute_context
        assert "DISPUTE CONTEXT" in messages[2]["content"]

    def test_no_dispute_context_keeps_normal_messages(self):
        """Without dispute_context, normal message count is 2."""
        ctx = AgentContext.create_mock(llm_responses=[_good_trace_json()])
        reasoner = LLMReasoner()
        reasoner.reason(ctx, _make_test_prompt_spec(), [_make_test_evidence_bundle()])
        calls = ctx.llm.provider._calls
        messages = calls[0]["messages"]
        assert len(messages) == 2  # system, user_prompt only


def _make_test_reasoning_trace():
    return ReasoningTrace(
        trace_id="trace_test",
        market_id="mk_test",
        bundle_id="eb_test",
        steps=[ReasoningStep(
            step_id="step_001", step_type="threshold_check",
            description="Compare price",
        )],
        preliminary_outcome="YES",
        preliminary_confidence=0.9,
        recommended_rule_id="R_THRESHOLD",
    )


class TestJudgeDisputeInjection:
    def test_dispute_context_appended_to_judge_messages(self):
        """When dispute_context is in ctx.extra, judge LLM should receive dispute instruction."""
        ctx = AgentContext.create_mock(llm_responses=[_good_verdict_json()])
        ctx.extra["dispute_context"] = {
            "reason_code": "VERDICT_WRONG_MAPPING",
            "message": "The verdict should be NO",
            "leaf_path": None,
            "target_step": "judge",
            "patch": None,
        }
        from agents.judge.agent import JudgeLLM
        judge = JudgeLLM()
        judge.run(ctx, _make_test_prompt_spec(), [_make_test_evidence_bundle()], _make_test_reasoning_trace())
        calls = ctx.llm.provider._calls
        messages = calls[0]["messages"]
        assert len(messages) == 3
        assert "DISPUTE CONTEXT" in messages[2]["content"]

    def test_no_dispute_context_keeps_normal_judge_messages(self):
        """Without dispute_context, judge message count is 2."""
        ctx = AgentContext.create_mock(llm_responses=[_good_verdict_json()])
        from agents.judge.agent import JudgeLLM
        judge = JudgeLLM()
        judge.run(ctx, _make_test_prompt_spec(), [_make_test_evidence_bundle()], _make_test_reasoning_trace())
        calls = ctx.llm.provider._calls
        messages = calls[0]["messages"]
        assert len(messages) == 2


# --- Validation, Rerun Plan, Diff Summary tests ---

class TestDisputeValidation:
    def test_reasoning_only_cannot_target_collect(self):
        from api.routes.dispute import validate_dispute_request, DisputeRequest, DisputeTarget, DisputeContext
        req = DisputeRequest(
            mode="reasoning_only",
            target=DisputeTarget(step="collect"),
            reason_code="EVIDENCE_MISSING",
            message="Evidence is missing",
            context=DisputeContext(
                prompt_spec={},
                evidence_bundle=[{}],
            ),
        )
        errors = validate_dispute_request(req)
        assert len(errors) > 0
        assert "reasoning_only" in errors[0]

    def test_reasoning_only_cannot_target_prompt_spec(self):
        from api.routes.dispute import validate_dispute_request, DisputeRequest, DisputeTarget, DisputeContext
        req = DisputeRequest(
            mode="reasoning_only",
            target=DisputeTarget(step="prompt_spec"),
            reason_code="SPEC_AMBIGUOUS",
            message="Spec is ambiguous",
            context=DisputeContext(
                prompt_spec={},
                evidence_bundle=[{}],
            ),
        )
        errors = validate_dispute_request(req)
        assert len(errors) > 0

    def test_patch_whitelist_rejects_disallowed_keys(self):
        from api.routes.dispute import validate_dispute_request, DisputeRequest, DisputeTarget, DisputeContext
        req = DisputeRequest(
            mode="reasoning_only",
            target=DisputeTarget(step="audit"),
            reason_code="REASONING_STEP_INCORRECT",
            message="Wrong step",
            patch={"source_targets": ["new_source"]},
            context=DisputeContext(
                prompt_spec={},
                evidence_bundle=[{}],
            ),
        )
        errors = validate_dispute_request(req)
        assert len(errors) > 0
        assert "source_targets" in errors[0]

    def test_patch_whitelist_allows_valid_keys(self):
        from api.routes.dispute import validate_dispute_request, DisputeRequest, DisputeTarget, DisputeContext
        req = DisputeRequest(
            mode="reasoning_only",
            target=DisputeTarget(step="judge"),
            reason_code="VERDICT_WRONG_MAPPING",
            message="Wrong mapping",
            patch={"confidence_override": 0.3},
            context=DisputeContext(
                prompt_spec={},
                evidence_bundle=[{}],
                reasoning_trace={},
            ),
        )
        errors = validate_dispute_request(req)
        assert len(errors) == 0

    def test_judge_requires_reasoning_trace(self):
        from api.routes.dispute import validate_dispute_request, DisputeRequest, DisputeTarget, DisputeContext
        req = DisputeRequest(
            mode="reasoning_only",
            target=DisputeTarget(step="judge"),
            reason_code="VERDICT_WRONG_MAPPING",
            message="Wrong mapping",
            context=DisputeContext(
                prompt_spec={},
                evidence_bundle=[{}],
                reasoning_trace=None,
            ),
        )
        errors = validate_dispute_request(req)
        assert len(errors) > 0
        assert "reasoning_trace" in errors[0]


class TestRerunPlan:
    def test_judge_only(self):
        from api.routes.dispute import compute_rerun_plan
        assert compute_rerun_plan("judge") == ["judge"]

    def test_audit_then_judge(self):
        from api.routes.dispute import compute_rerun_plan
        assert compute_rerun_plan("audit") == ["audit", "judge"]

    def test_collect_chain(self):
        from api.routes.dispute import compute_rerun_plan
        assert compute_rerun_plan("collect") == ["collect", "audit", "judge"]

    def test_prompt_spec_full_chain(self):
        from api.routes.dispute import compute_rerun_plan
        assert compute_rerun_plan("prompt_spec") == ["collect", "audit", "judge"]


class TestDiffSummary:
    def test_verdict_changed(self):
        from api.routes.dispute import compute_diff_summary
        from core.schemas.verdict import DeterministicVerdict
        old = DeterministicVerdict(market_id="mk_1", outcome="YES", confidence=0.9, resolution_rule_id="R_1")
        new = DeterministicVerdict(market_id="mk_1", outcome="NO", confidence=0.8, resolution_rule_id="R_1")
        diff = compute_diff_summary(["audit", "judge"], old, new)
        assert diff.verdict_changed is True
        assert diff.outcome_before == "YES"
        assert diff.outcome_after == "NO"

    def test_verdict_same(self):
        from api.routes.dispute import compute_diff_summary
        from core.schemas.verdict import DeterministicVerdict
        old = DeterministicVerdict(market_id="mk_1", outcome="YES", confidence=0.9, resolution_rule_id="R_1")
        new = DeterministicVerdict(market_id="mk_1", outcome="YES", confidence=0.9, resolution_rule_id="R_1")
        diff = compute_diff_summary(["judge"], old, new)
        assert diff.verdict_changed is False

    def test_confidence_changed(self):
        from api.routes.dispute import compute_diff_summary
        from core.schemas.verdict import DeterministicVerdict
        old = DeterministicVerdict(market_id="mk_1", outcome="YES", confidence=0.9, resolution_rule_id="R_1")
        new = DeterministicVerdict(market_id="mk_1", outcome="YES", confidence=0.7, resolution_rule_id="R_1")
        diff = compute_diff_summary(["judge"], old, new)
        assert diff.verdict_changed is False
        assert diff.confidence_before == 0.9
        assert diff.confidence_after == 0.7
